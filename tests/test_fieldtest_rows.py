"""Field tests -- the rows a case's key leads to, fetched and laid out as the querier's rows file.

A stand-in PanDA answers the way the server documents its answers
(pandaserver/asyncprocess/db_read_handlers.py, api/v1/job_api.py
get_description_incl_archive, get_datasets_by_name and get_status_history,
task_api.py get_detailed_info): R rows keyed by lower-case names, times as
``{"__datetime__": iso}``, a key's row, H rows and dataset rows in the
columns' own spelling.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path

import pytest

from bamboo.scripts.panda.fieldtest import after as A
from bamboo.scripts.panda.fieldtest import fetch as FE
from bamboo.scripts.panda.fieldtest import rows as RB

AS_OF = "2026-10-13T10:00:00"
ASKED = "2026-10-13 10:00:05"


def T(text: str) -> dict:
    return {"__datetime__": text}


def _dt(value):
    return FE.when(value) if isinstance(value, dict) else value


def _holds(row: dict, cond: list) -> bool:
    col, op, value = cond[0].lower(), cond[1], cond[2]
    got = _dt(row.get(col))
    if op == "=":
        return got == value
    if op == ">=":
        return got is not None and got >= datetime.datetime.fromisoformat(value)
    if op == "in":
        return got in value
    raise AssertionError(op)


class FakePanda:
    """Tables of lower-case rows, rows by key, and status logs, answered as the server would."""

    #: the moment R gives a batch
    read_as_of = AS_OF

    def __init__(
        self,
        tables: dict[str, list[dict]],
        job_log: list[dict],
        task_log: list[dict],
        truncate=(),
        jobs: dict | None = None,
        task: dict | None = None,
        datasets: list[dict] = (),
    ):
        self.tables, self.job_log, self.task_log, self.truncate = (
            tables,
            job_log,
            task_log,
            set(truncate),
        )
        self.jobs, self.task_row, self.dataset_rows = jobs or {}, task, list(datasets)
        self.calls: list[tuple[str, object]] = []

    def parts(self) -> FE.Parts:
        return FE.Parts(
            db_read=self.db_read,
            jobs=self.get_jobs,
            task=self.get_task,
            datasets=self.get_datasets,
            job_history=self.job_history,
            task_history=self.task_history,
        )

    async def get_jobs(self, job_ids):
        self.calls.append(("jobs", job_ids))
        return {
            "success": True,
            "jobs": {i: self.jobs.get(i) for i in job_ids},
            "asked_at": ASKED,
            "calls": [],
        }

    async def get_task(self, task_id):
        self.calls.append(("task", task_id))
        if self.task_row is None:
            return {
                "success": False,
                "message": "Task not found or error retrieving task info",
                "data": None,
                "asked_at": ASKED,
            }
        return {
            "success": True,
            "message": "",
            "data": self.task_row,
            "asked_at": ASKED,
        }

    async def get_datasets(self, names):
        self.calls.append(("datasets", names))
        return {
            "success": True,
            "rows": [r for r in self.dataset_rows if r["name"] in names],
            "as_of": T(AS_OF),
            "retained_from": T("2026-08-01T00:00:00"),
            "calls": [],
        }

    async def db_read(self, queries):
        self.calls.append(("db_read", queries))
        answers = []
        for q in queries:
            tables = []
            for t in q["tables"]:
                rows = [
                    r
                    for r in self.tables.get(t, [])
                    if all(_holds(r, c) for c in q["where"])
                ]
                if "group_by" in q:
                    counts: dict = {}
                    for r in rows:
                        counts[r["jobstatus"]] = counts.get(r["jobstatus"], 0) + 1
                    rows = [{"jobstatus": s, "count(*)": n} for s, n in counts.items()]
                elif "columns" in q:
                    rows = [
                        {c.lower(): r.get(c.lower()) for c in q["columns"]}
                        for r in rows
                    ]
                tables.append(
                    {
                        "table": t,
                        "rows": rows,
                        "truncated": t in self.truncate,
                        "probes": 1,
                        "seconds": 0.0,
                        "statement": "SELECT",
                    }
                )
            answers.append(
                {
                    "status": "done",
                    "truncated": any(x["truncated"] for x in tables),
                    "tables": tables,
                }
            )
        return {
            "success": True,
            "message": "",
            "data": {"as_of": T(self.read_as_of), "answers": answers},
            "async_meta": {"status": "done"},
        }

    async def job_history(self, job_ids, since=None, until=None):
        self.calls.append(("job_history", {"job_ids": job_ids, "since": since}))
        rows = [r for r in self.job_log if r["PandaID"] in job_ids]
        return {
            "success": True,
            "rows": rows,
            "as_of": T(AS_OF),
            "retained_from": T("2026-07-12T00:00:00"),
            "since": T(since.replace(" ", "T")) if since else None,
            "until": None,
            "calls": [],
        }

    async def task_history(self, task_id):
        self.calls.append(("task_history", task_id))
        return {
            "success": True,
            "message": "",
            "data": {"rows": self.task_log, "as_of": T(AS_OF), "retained_from": None},
        }


def _job_state(pid, status, at):
    return {
        "PandaID": pid,
        "modificationTime": T(at[:19]),
        "jobStatus": status,
        "prodSourceLabel": "user",
        "cloud": "WORLD",
        "computingSite": "SITE_A",
        "modificationHost": "aipanda093",
        "modiftime_extended": T(at),
    }


PID = 7353014619
FILES = [
    {
        "PandaID": PID,
        "lfn": "a.log.tgz",
        "type": "log",
        "dataset": "user.x.log",
        "dispatchDBlock": None,
        "destinationDBlock": "user.x_sub1",
    },
    {
        "PandaID": PID,
        "lfn": "in.root",
        "type": "input",
        "dataset": "data.in",
        "dispatchDBlock": "panda.dis1",
        "destinationDBlock": None,
    },
]
#: the job as job/get_description_incl_archive answers it: every column by the class's name, and its files
JOB = {
    "PandaID": PID,
    "jobStatus": "failed",
    "specialHandling": "ddm:rucio",
    "creationTime": T("2026-10-07T13:06:45"),
    "modificationTime": T("2026-10-07T13:08:03"),
    "ddmErrorCode": 200,
    "prodDBlock": "data.in",
    "destinationDBlock": "user.x_sub1",
    "Files": FILES,
}
DATASETS = [
    {"name": "user.x_sub1", "creationdate": T("2026-10-07T13:06:46"), "subType": "sub"},
    {"name": "panda.dis1", "creationdate": T("2026-10-07T13:06:47"), "subType": "dis"},
]
JOB_LOG = [
    _job_state(PID, "pending", "2026-10-07T13:06:45.443717"),
    _job_state(
        PID, "activated", "2026-10-07T13:06:46"
    ),  # no microseconds: still written with six digits
    _job_state(PID, "failed", "2026-10-07T13:08:03.877599"),
]
SPELLINGS = {
    "JobSpec": {
        "pandaid": "PandaID",
        "jobstatus": "jobStatus",
        "specialhandling": "specialHandling",
        "creationtime": "creationTime",
        "modificationtime": "modificationTime",
        "ddmerrorcode": "ddmErrorCode",
        "proddblock": "prodDBlock",
        "destinationdblock": "destinationDBlock",
    },
    "FileSpec": {
        "pandaid": "PandaID",
        "lfn": "lfn",
        "type": "type",
        "dataset": "dataset",
        "dispatchdblock": "dispatchDBlock",
        "destinationdblock": "destinationDBlock",
    },
    "JediTaskSpec": {
        "jeditaskid": "jediTaskID",
        "status": "status",
        "creationdate": "creationDate",
        "modificationtime": "modificationTime",
        "splitrule": "splitRule",
    },
    "DatasetSpec": {
        "name": "name",
        "creationdate": "creationdate",
        "subtype": "subType",
    },
}
JOB_Q = {
    "kind": "default",
    "subject": "JobSpec.jobStatus",
    "observed": "failed",
    "key": str(PID),
}


def _job_panda(job: dict | None = JOB) -> FakePanda:
    return FakePanda({}, JOB_LOG, [], jobs={PID: job}, datasets=DATASETS)


async def _fetch(panda: FakePanda, question: dict) -> dict:
    return await FE.fetch(question, panda.parts())


# ------------------------------------------------------------------ a job's key


async def test_a_job_is_asked_by_its_key_then_the_datasets_its_rows_name_then_its_history_from_its_creation():
    panda = _job_panda()
    rec = await _fetch(panda, JOB_Q)
    assert [c[0] for c in panda.calls] == ["jobs", "datasets", "job_history"]
    assert panda.calls[0][1] == [PID] and isinstance(panda.calls[0][1][0], int)
    # every name the job's row and its files hold, once each; an empty column names none
    assert panda.calls[1][1] == ["data.in", "panda.dis1", "user.x.log", "user.x_sub1"]
    assert panda.calls[2][1] == {"job_ids": [PID], "since": "2026-10-07 13:06:45"}
    assert rec["key_class"] == "JobSpec" and [s["part"] for s in rec["steps"]] == [
        "jobs",
        "datasets",
        "job_history",
    ]


async def test_a_job_not_found_or_not_read_is_left_out_with_why_and_its_history_asked_whole():
    for job, why in (
        (None, "in no live table, nor in the archive's last 90 days"),
        ({"error": "ORA-03113"}, "the database failed reading it: ORA-03113"),
    ):
        panda = _job_panda(job)
        out = RB.build(await _fetch(panda, JOB_Q), SPELLINGS)
        # no creation time to start the history at, and no file to name a dataset
        assert [c[0] for c in panda.calls] == ["jobs", "job_history"]
        assert panda.calls[1][1] == {"job_ids": [PID], "since": None}
        assert "rows" not in out and "files" not in out and "collections" not in out
        assert f"rows.job, files: PandaID={PID}: {why}" in out["about"]
        assert [s["status"] for s in out["statuslog"]["states"]] == [
            "pending",
            "activated",
            "failed",
        ]


async def test_a_jobs_rows_file_keeps_the_class_columns_formats_its_times_and_keeps_only_the_key_before_the_decision():
    out = RB.build(await _fetch(_job_panda(), JOB_Q), SPELLINGS)
    assert out["classes"]["job"] == "JobSpec" and out["spellings"]["job"] == [
        "job",
        "self.job",
    ]
    # every write may have set every column until the map says which: only the key is the decision's
    assert (
        list(out["rows"]["job"]) == ["PandaID"]
        and out["rows"]["job"]["PandaID"]["value"] == PID
    )
    source = "job/get_description_incl_archive, asked at 2026-10-13 10:00:05 on this machine's clock (UTC)"
    assert out["rows"]["job"]["PandaID"]["source"] == source
    assert (
        out["after"]["specialHandling"] == "ddm:rucio"
        and out["after"]["modificationTime"] == "2026-10-07 13:08:03"
    )
    assert "Files" not in out["after"]
    assert out["after"]["source"].startswith(source + "; ")
    assert "decision at 2026-10-07 13:08:03.877599" in out["after"]["source"]
    states = out["statuslog"]["states"]
    assert [s["status"] for s in states] == ["pending", "activated", "failed"]
    # whole from where it was asked, since the partitions go back further
    assert (out["statuslog"]["as_of"], out["statuslog"]["retained_from"]) == (
        "2026-10-13 10:00:00",
        "2026-10-07 13:06:45",
    )
    assert (
        states[1]["time"] == "2026-10-07 13:06:46"
        and states[1]["time_extended"] == "2026-10-07 13:06:46.000000"
    )
    assert [f["lfn"]["value"] for f in out["files"]] == ["a.log.tgz", "in.root"]
    assert out["files"][0]["destinationDBlock"] == {
        "value": "user.x_sub1",
        "source": source,
    }
    assert out["collections"]["job.Files"] == {"rows": "files", "entity": "file"}
    # every name asked, so that a name with no row reads as not in the table rather than not asked
    datasets = out["datasets"]
    # the same name can be a JediDatasetSpec in JEDI_Datasets too: the section says whose rows these are
    assert (datasets["class"], datasets["table"]) == (
        "DatasetSpec",
        "ATLAS_PANDA.Datasets",
    )
    assert datasets["asked"] == ["data.in", "panda.dis1", "user.x.log", "user.x_sub1"]
    assert datasets["rows"] == [
        {
            "name": "user.x_sub1",
            "creationdate": "2026-10-07 13:06:46",
            "subType": "sub",
        },
        {"name": "panda.dis1", "creationdate": "2026-10-07 13:06:47", "subType": "dis"},
    ]
    assert (datasets["as_of"], datasets["retained_from"]) == (
        "2026-10-13 10:00:00",
        "2026-08-01 00:00:00",
    )
    assert "the clock of the machine that inserted the row" in datasets["source"]
    assert "not columns" not in out["about"] and "datasets:" not in out["about"]


async def test_a_refused_dataset_answer_leaves_the_datasets_out_and_says_why():
    panda = _job_panda()

    async def refused(names):
        return {
            "success": False,
            "message": "",
            "data": {
                "rejected": "probe_budget",
                "keys": 4,
                "partitions": 5,
                "budget": 1,
            },
            "calls": [],
        }

    panda_parts = panda.parts()
    panda_parts.datasets = refused
    out = RB.build(await FE.fetch(JOB_Q, panda_parts), SPELLINGS)
    assert "datasets" not in out and "files" in out
    assert "datasets: {'rejected': 'probe_budget'" in out["about"]


async def test_a_jobs_files_hold_only_their_class_columns():
    job = JOB | {"Files": [FILES[0] | {"modificationTime": T("2026-10-07T13:08:03")}]}
    out = RB.build(await _fetch(_job_panda(job), JOB_Q), SPELLINGS)
    assert "modificationTime" not in out["files"][0]
    assert (
        "1 key(s) of the answers that are not columns of their class, left out: ['modificationTime']"
        in out["about"]
    )


async def test_a_failed_history_leaves_the_statuslog_out_rather_than_empty():
    panda = _job_panda()

    async def refused(job_ids, since=None, until=None):
        return {
            "success": False,
            "message": "over budget",
            "data": {"rejected": "probe_budget"},
            "calls": [],
        }

    panda_parts = panda.parts()
    panda_parts.job_history = refused
    out = RB.build(await FE.fetch(JOB_Q, panda_parts), SPELLINGS)
    assert "statuslog" not in out and "statuslog: over budget" in out["about"]


def test_a_historys_span_is_cut_to_seconds_without_claiming_a_second_it_may_not_hold():
    bounds = RB.bounds_of(
        {
            "as_of": T("2026-10-13T10:00:00.900000"),
            "retained_from": T("2026-07-12T00:00:00.100000"),
        }
    )
    assert bounds == {
        "as_of": "2026-10-13 10:00:00",
        "retained_from": "2026-07-12 00:00:01",
    }


def test_a_historys_span_is_only_what_was_asked():
    bounds = RB.bounds_of(
        {
            "as_of": T("2026-10-13T10:00:00"),
            "retained_from": T("2026-07-12T00:00:00"),
            "since": T("2026-09-28T00:00:00"),
            "until": T("2026-10-01T00:00:00"),
        }
    )
    # until is exclusive: the last second held is the one before it
    assert bounds == {
        "as_of": "2026-09-30 23:59:59",
        "retained_from": "2026-09-28 00:00:00",
    }
    # a since does not make a history whole where the server cannot tell
    unknown = RB.bounds_of(
        {"as_of": T(AS_OF), "retained_from": None, "since": T("2026-09-28T00:00:00")}
    )
    assert unknown["retained_from"] is None


# ------------------------------------------------------------------ a task's key

TID = 52757505
TASK_Q = {
    "kind": "stuck",
    "subject": "JediTaskSpec.status",
    "observed": "finishing",
    "key": str(TID),
}


#: the task as task/get_detailed_info answers it: its columns, splitRule decoded, and keys that are not columns
TASK = {
    "jediTaskID": TID,
    "status": "finished",
    "creationDate": T("2026-09-27T06:50:00"),
    "modificationTime": T("2026-09-29T09:08:56"),
    "splitRule": "useScout=will_update_requirements,scoutSuccessRate=5",
    "split_rule_raw": "US=2,SS=5",
    "nFiles": 2,
    "raw_task_params": "{}",
}


def _task_panda(task: dict | None = TASK, truncate=()) -> FakePanda:
    job = {
        "jeditaskid": TID,
        "jobsubstatus": None,
        "prodsourcelabel": "user",
        "currentpriority": 1000,
        "computingsite": "S",
        "attemptnr": 1,
        "statechangetime": T("2026-09-29T00:08:30"),
        "endtime": T("2026-09-29T00:08:19"),
        "modificationtime": T("2026-09-29T00:08:30"),
    }
    j1 = job | {
        "pandaid": 11,
        "jobstatus": "finished",
        "creationtime": T("2026-09-27T07:08:32"),
    }
    j2 = job | {
        "pandaid": 12,
        "jobstatus": "failed",
        "creationtime": T("2026-09-27T07:08:40"),
    }
    tables = {
        "jobsArchived4": [j1, j2],
        "jobsArchived": [j1 | {"modificationtime": T("2026-10-01T00:00:00")}],
    }
    job_log = [
        _job_state(11, "running", "2026-09-28T01:00:00.5"),
        _job_state(11, "finished", "2026-09-29T00:08:30.1"),
        _job_state(12, "running", "2026-09-28T02:00:00.5"),
        _job_state(12, "failed", "2026-09-28T05:00:00.25"),
    ]
    task_log = [
        {
            "jediTaskID": TID,
            "modificationTime": T("2026-09-27T06:55:15.975402"),
            "status": "running",
            "modificationHost": "h",
            "attemptNr": 0,
            "reason": None,
        },
        {
            "jediTaskID": TID,
            "modificationTime": T("2026-09-29T09:08:56.000001"),
            "status": "finished",
            "modificationHost": "h",
            "attemptNr": 0,
            "reason": "soft finish by prodsys",
        },
    ]
    return FakePanda(tables, job_log, task_log, truncate=truncate, task=task)


async def test_a_task_is_asked_by_its_key_then_its_jobs_in_every_table_in_one_batch_and_their_histories_from_the_first_creation():
    panda = _task_panda()
    await _fetch(panda, TASK_Q)
    kinds = [c[0] for c in panda.calls]
    assert kinds == ["task", "db_read", "task_history", "job_history"]
    assert panda.calls[0][1] == TID
    batch = panda.calls[1][1]
    live = ["jobsDefined4", "jobsActive4", "jobsArchived4"]
    assert [q["tables"] for q in batch] == [
        live,
        live,
        ["jobsArchived"],
        ["jobsArchived"],
    ]
    assert batch[1]["group_by"] == ["jobStatus"] and batch[3]["aggregates"] == [
        ["count", "*"]
    ]
    # the archive from the task's creation less a day
    assert batch[2]["where"][1] == ["modificationTime", ">=", "2026-09-26T06:50:00"]
    assert panda.calls[3][1] == {"job_ids": [11, 12], "since": "2026-09-27 07:08:32"}


async def test_a_task_not_answered_bounds_the_archive_from_100_days_before_it_was_asked():
    panda = _task_panda(task=None)
    out = RB.build(await _fetch(panda, TASK_Q), SPELLINGS)
    assert panda.calls[1][1][2]["where"][1] == [
        "modificationTime",
        ">=",
        "2026-07-05T10:00:05",
    ]
    assert "rows" not in out and "after" not in out
    assert (
        f"rows.task: jediTaskID={TID}: Task not found or error retrieving task info"
        in out["about"]
    )
    # the task's jobs do not depend on its row
    assert len(out["job_rows"]["rows"]) == 3


async def test_a_cut_answer_is_left_out_and_said():
    out = RB.build(
        await _fetch(_task_panda(truncate={"jobsArchived"}), TASK_Q), SPELLINGS
    )
    assert "job_rows" not in out and "job_statuslog" not in out
    assert "job_rows, job_statuslog: ['truncated']" in out["about"]
    assert "counts: ['truncated']" in out["about"]


async def test_a_tasks_rows_file_holds_its_jobs_their_histories_and_counts():
    panda = _task_panda()
    panda.read_as_of = "2026-10-13T10:00:00.900000"
    out = RB.build(await _fetch(panda, TASK_Q), SPELLINGS)
    # the moment R read the counts and the rows, cut down to its second
    assert out["counts"]["as_of"] == out["job_rows"]["as_of"] == "2026-10-13 10:00:00"
    assert out["classes"]["task"] == "JediTaskSpec" and out["spellings"] == {
        "about": out["spellings"]["about"],
        "task": [],
    }
    rows = out["job_rows"]["rows"]
    # a PandaID in two tables is two rows
    assert [(r["tbl"], r["pandaid"]) for r in rows] == [
        ("jobsArchived4", 11),
        ("jobsArchived4", 12),
        ("jobsArchived", 11),
    ]
    assert (
        rows[0]["creationtime"] == "2026-09-27 07:08:32"
        and rows[0]["t_final"] == "2026-09-29 00:08:30"
    )
    assert rows[1]["t_final"] == "2026-09-28 05:00:00"
    assert (
        out["job_statuslog"]["by_pandaid"]["12"][1]["time_extended"]
        == "2026-09-28 05:00:00.250000"
    )
    assert sorted(
        (c["table"], c["jobStatus"], c["n"]) for c in out["counts"]["rows"]
    ) == [
        ("jobsArchived", "finished", 1),
        ("jobsArchived4", "failed", 1),
        ("jobsArchived4", "finished", 1),
    ]
    assert (
        out["job_statuslog"]["as_of"],
        out["job_statuslog"]["retained_from"],
    ) == ("2026-10-13 10:00:00", "2026-09-27 07:08:32")
    assert [s["status"] for s in out["statuslog"]["states"]] == ["running", "finished"]
    # the server could not tell how far back the task's history is whole
    assert (out["statuslog"]["as_of"], out["statuslog"]["retained_from"]) == (
        "2026-10-13 10:00:00",
        None,
    )
    assert (
        out["statuslog"]["states"][0]["time_extended"] == "2026-09-27 06:55:15.975402"
    )
    # no 'finishing' row: the decision time cannot be taken, so all but the key is after
    assert (
        list(out["rows"]["task"]) == ["jediTaskID"]
        and "no decision time" in out["after"]["source"]
    )
    assert out["after"]["source"].startswith(
        "task/get_detailed_info, asked at 2026-10-13 10:00:05 on this machine's clock (UTC); "
    )
    # the column as stored, not the server's decoding; what is not a column is named, not kept
    assert out["after"]["splitRule"] == "US=2,SS=5"
    assert "split_rule_raw" not in out["after"] and "nFiles" not in out["after"]
    assert (
        "2 key(s) of the answers that are not columns of their class, left out: ['nFiles', 'raw_task_params']"
        in out["about"]
    )


# ------------------------------------------------------------------ after


def _states(*pairs):
    return [{"status": s, "time": t[:19], "time_extended": t} for s, t in pairs]


ROW = {"PandaID": 1, "jobStatus": "failed", "ddmErrorCode": 200, "computingSite": "S"}
LOG = _states(
    ("activated", "2026-10-07 13:06:46.000000"),
    ("sent", "2026-10-07 13:07:00.000000"),
    ("failed", "2026-10-07 13:08:03.877599"),
)


def test_after_takes_the_decision_by_the_kind_of_question():
    assert A.decision_time("default", "failed", LOG)[0] == datetime.datetime(
        2026, 10, 7, 13, 8, 3, 877599
    )
    assert A.decision_time("late", "activated", LOG)[0] == datetime.datetime(
        2026, 10, 7, 13, 7
    )
    assert A.decision_time("stuck", "activated", LOG)[0] == datetime.datetime(
        2026, 10, 7, 13, 6, 46
    )
    assert A.decision_time("late", "failed", LOG) == (
        None,
        "no history row after 'failed'",
    )


def test_after_keeps_only_what_no_later_write_may_have_set():
    kept, later, why = A.split(ROW, "PandaID", "late", "activated", LOG, None)
    assert kept == {"PandaID": 1} and set(later) == {
        "jobStatus",
        "ddmErrorCode",
        "computingSite",
    }
    assert "2 write(s)" in why and "2 of them with no stated columns" in why

    def stated(write):
        return {"sent": {"jobStatus"}, "failed": {"jobStatus", "ddmErrorCode"}}[
            write.status
        ]

    kept, later, _ = A.split(ROW, "PandaID", "late", "activated", LOG, None, stated)
    assert kept == {"PandaID": 1, "computingSite": "S"} and set(later) == {
        "jobStatus",
        "ddmErrorCode",
    }


def test_after_counts_a_write_no_history_recorded():
    modified = datetime.datetime(2026, 10, 8)
    writes = A.writes_from(
        datetime.datetime(2026, 10, 7, 13, 8, 3, 877599), LOG, modified
    )
    assert writes[-1] == A.Write(modified, None) and len(writes) == 2


# ------------------------------------------------------------------ spelling


def test_spellings_come_from_each_class_column_list_in_the_pin(tmp_path):
    pin = tmp_path / "pin"
    (pin / "pkg").mkdir(parents=True)
    (pin / "pkg" / "JobSpec.py").write_text(
        "class JobSpec(object):\n    _attributes = ('PandaID', 'jobStatus')\n"
    )
    (pin / "pkg" / "JediTaskSpec.py").write_text(
        "class JediTaskSpec(object):\n    attributes = ('jediTaskID',)\n"
    )
    dump = tmp_path / "dump.json"
    dump.write_text(
        json.dumps(
            {
                "declaration_yields": {
                    "JobSpec (pkg/JobSpec.py)": 2,
                    "JediTaskSpec (pkg/JediTaskSpec.py)": 1,
                }
            }
        )
    )
    got = RB.column_spellings(pin, dump, ("JobSpec", "JediTaskSpec"))
    assert got == {
        "JobSpec": {"pandaid": "PandaID", "jobstatus": "jobStatus"},
        "JediTaskSpec": {"jeditaskid": "jediTaskID"},
    }


def test_a_key_of_a_class_with_no_plan_yet_is_refused():
    with pytest.raises(FE.FieldTestError, match="no plan yet for a key of FileSpec"):
        FE.asyncio.run(
            FE.fetch(
                {"subject": "FileSpec.status", "key": "1"},
                FakePanda({}, [], []).parts(),
            )
        )


def test_the_fetch_record_holds_every_request_and_response(tmp_path):
    rec = FE.asyncio.run(FE.fetch(JOB_Q, _job_panda().parts()))
    path = tmp_path / "answers.json"
    path.write_text(json.dumps(rec))
    again = json.loads(Path(path).read_text())
    # JSON spells the job's ID as a string; the rows file is the same
    assert list(again["steps"][0]["response"]["jobs"]) == [str(PID)]
    assert RB.build(again, SPELLINGS) == RB.build(rec, SPELLINGS)
    assert again["steps"][0]["request"] == {"job_ids": [PID]}
    assert [r["name"] for r in again["steps"][1]["response"]["rows"]] == [
        "user.x_sub1",
        "panda.dis1",
    ]
    assert again["steps"][2]["response"]["rows"][0]["jobStatus"] == "pending"
