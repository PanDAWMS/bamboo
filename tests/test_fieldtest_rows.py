"""Field tests -- the rows a case's key leads to, fetched and laid out as the querier's rows file.

A stand-in PanDA answers the way the server documents its answers
(pandaserver/asyncprocess/db_read_handlers.py, api/v1/job_api.py
get_description_incl_archive and get_status_history, task_api.py
get_detailed_info): R rows keyed by lower-case names, times as
``{"__datetime__": iso}``, a key's row and H rows in the columns' own
spelling.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path

import pytest

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
    ):
        self.tables, self.job_log, self.task_log, self.truncate = (
            tables,
            job_log,
            task_log,
            set(truncate),
        )
        self.jobs, self.task_row = jobs or {}, task
        self.calls: list[tuple[str, object]] = []

    def parts(self) -> FE.Parts:
        return FE.Parts(
            db_read=self.db_read,
            jobs=self.get_jobs,
            task=self.get_task,
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
}


def _table(
    name, spelled, rows_of, identity, inserted, family, creation, partitioned=()
):
    """A map dump's ``tables`` entry, cut to the fields the fetch reads."""
    return {
        "name": name,
        "spellings": [*spelled, name],
        "rows_of": rows_of,
        "identity_attrs": list(identity),
        "inserted": inserted,
        "family": list(family),
        "creation": list(creation),
        "partitioned_by_attrs": list(partitioned),
    }


JOBS = {
    "jobsactive4": "jobsActive4",
    "jobsarchived": "jobsArchived",
    "jobsarchived4": "jobsArchived4",
    "jobsdefined4": "jobsDefined4",
}
#: the dump's tables of a job and a task (x13vii3k), and two the fetch has no single key or spelling for
DUMP = {
    "tables": [
        *(
            _table(
                n,
                [s],
                "JobSpec",
                [] if n == "jobsarchived" else ["JobSpec.PandaID"],
                n != "jobsarchived",
                [f for f in JOBS if f != n],
                ["JobSpec.creationTime"],
                ["JobSpec.modificationTime"] if "archived" in n else [],
            )
            for n, s in JOBS.items()
        ),
        _table(
            "jedi_tasks",
            ["JEDI_Tasks"],
            "JediTaskSpec",
            ["JediTaskSpec.jediTaskID"],
            True,
            [],
            ["JediTaskSpec.creationDate"],
            ["JediTaskSpec.jediTaskID"],
        ),
        _table(
            "jedi_datasets",
            ["JEDI_Datasets"],
            "JediDatasetSpec",
            ["JediDatasetSpec.datasetID", "JediDatasetSpec.jediTaskID"],
            True,
            [],
            ["JediDatasetSpec.creationTime"],
        ),
        _table(
            "filestable_arch",
            ["filesTable_ARCH", "filesTable_Arch"],
            "FileSpec",
            ["FileSpec.row_ID"],
            False,
            [],
            [],
        ),
        _table("users", ["users"], None, [], True, [], []),
    ]
}
TABLES = FE.Tables.of(DUMP)
JOB_Q = {
    "kind": "default",
    "subject": "JobSpec.jobStatus",
    "observed": "failed",
    "key": str(PID),
}


def _job_panda(job: dict | None = JOB) -> FakePanda:
    return FakePanda({}, JOB_LOG, [], jobs={PID: job})


async def _fetch(panda: FakePanda, question: dict) -> dict:
    return await FE.fetch(question, TABLES, panda.parts())


# ------------------------------------------------------------------ a job's key


async def test_a_job_is_asked_by_its_key_then_its_history_from_its_creation():
    panda = _job_panda()
    rec = await _fetch(panda, JOB_Q)
    # the datasets its rows name are not asked: the querier reads no Datasets row
    assert [c[0] for c in panda.calls] == ["jobs", "job_history"]
    assert panda.calls[0][1] == [PID] and isinstance(panda.calls[0][1][0], int)
    assert panda.calls[1][1] == {"job_ids": [PID], "since": "2026-10-07 13:06:45"}
    assert rec["key_class"] == "JobSpec" and [s["part"] for s in rec["steps"]] == [
        "jobs",
        "job_history",
    ]


async def test_a_job_not_found_or_not_read_is_left_out_with_why_and_its_history_asked_whole():
    for job, why in (
        (None, "in no live table, nor in the archive's last 90 days"),
        ({"error": "ORA-03113"}, "the database failed reading it: ORA-03113"),
    ):
        panda = _job_panda(job)
        out = RB.build(await _fetch(panda, JOB_Q), SPELLINGS, TABLES)
        # no creation time to start the history at
        assert [c[0] for c in panda.calls] == ["jobs", "job_history"]
        assert panda.calls[1][1] == {"job_ids": [PID], "since": None}
        assert "rows" not in out and "files" not in out and "collections" not in out
        assert f"rows.job, files: PandaID={PID}: {why}" in out["about"]
        assert [s["status"] for s in out["statuslog"]["states"]] == [
            "pending",
            "activated",
            "failed",
        ]


async def test_a_jobs_rows_file_keeps_the_class_columns_formats_its_times_and_says_when_each_was_read():
    out = RB.build(await _fetch(_job_panda(), JOB_Q), SPELLINGS, TABLES)
    assert out["classes"]["job"] == "JobSpec" and out["spellings"]["job"] == [
        "job",
        "self.job",
    ]
    source = "job/get_description_incl_archive, asked at 2026-10-13 10:00:05 on this machine's clock (UTC)"
    # read by key, the values hold when they were asked, on this machine's clock: not at the history's as_of
    read = {"source": source, "as_of": "2026-10-13 10:00:05", "clock": "client"}
    job = out["rows"]["job"]
    assert job["PandaID"] == {"value": PID, **read}
    # every column is laid out with when it was read: which a later write may have changed is the querier's to say
    assert job["specialHandling"] == {"value": "ddm:rucio", **read}
    assert job["modificationTime"] == {"value": "2026-10-07 13:08:03", **read}
    assert "Files" not in job and "after" not in out
    states = out["statuslog"]["states"]
    assert [s["status"] for s in states] == ["pending", "activated", "failed"]
    # whole from where it was asked, since the partitions go back further
    assert (out["statuslog"]["as_of"], out["statuslog"]["retained_from"]) == (
        "2026-10-13 10:00:00",
        "2026-10-07 13:06:45",
    )
    assert out["statuslog"]["clock"] == "database"
    assert (
        states[1]["time"] == "2026-10-07 13:06:46"
        and states[1]["time_extended"] == "2026-10-07 13:06:46.000000"
    )
    # a job's history row holds the columns it copies from the job, spelt as the attributes
    assert (
        states[0]["computingSite"],
        states[0]["prodSourceLabel"],
        states[0]["cloud"],
    ) == ("SITE_A", "user", "WORLD")
    assert "site" not in states[0]
    assert [f["lfn"]["value"] for f in out["files"]] == ["a.log.tgz", "in.root"]
    assert out["files"][0]["destinationDBlock"] == {"value": "user.x_sub1", **read}
    assert out["collections"]["job.Files"] == {"rows": "files", "entity": "file"}
    assert "datasets" not in out
    assert "not columns" not in out["about"] and "datasets:" not in out["about"]


async def test_a_record_that_asked_the_datasets_lays_out_none():
    # a record fetched while the datasets were asked (every recorded fetch up to 2026-10-28) gives the same file
    rec = await _fetch(_job_panda(), JOB_Q)
    names = {"names": ["data.in", "user.x_sub1"]}
    rows = [{"name": "user.x_sub1", "creationdate": T("2026-10-07T13:06:46")}]
    response = {"success": True, "rows": rows, "as_of": T(AS_OF), "calls": []}
    old = rec | {
        "steps": [
            rec["steps"][0],
            {"part": "datasets", "request": names, "response": response},
            *rec["steps"][1:],
        ]
    }
    assert RB.build(old, SPELLINGS, TABLES) == RB.build(rec, SPELLINGS, TABLES)


async def test_a_jobs_files_hold_only_their_class_columns():
    job = JOB | {"Files": [FILES[0] | {"modificationTime": T("2026-10-07T13:08:03")}]}
    out = RB.build(await _fetch(_job_panda(job), JOB_Q), SPELLINGS, TABLES)
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
    out = RB.build(await FE.fetch(JOB_Q, TABLES, panda_parts), SPELLINGS, TABLES)
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
        "clock": "database",
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
        "clock": "database",
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
    # the dump's tables, in the order of their names
    live = ["jobsActive4", "jobsArchived4", "jobsDefined4"]
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
    out = RB.build(await _fetch(panda, TASK_Q), SPELLINGS, TABLES)
    assert panda.calls[1][1][2]["where"][1] == [
        "modificationTime",
        ">=",
        "2026-07-05T10:00:05",
    ]
    assert "rows" not in out
    assert (
        f"rows.task: jediTaskID={TID}: Task not found or error retrieving task info"
        in out["about"]
    )
    # the task's jobs do not depend on its row
    assert len(out["job_rows"]["rows"]) == 3


async def test_a_cut_answer_is_left_out_and_said():
    out = RB.build(
        await _fetch(_task_panda(truncate={"jobsArchived"}), TASK_Q), SPELLINGS, TABLES
    )
    assert "job_rows" not in out and "job_statuslog" not in out
    assert "job_rows, job_statuslog: ['truncated']" in out["about"]
    assert "counts: ['truncated']" in out["about"]


async def test_a_tasks_rows_file_holds_its_jobs_their_histories_and_counts():
    panda = _task_panda()
    panda.read_as_of = "2026-10-13T10:00:00.900000"
    out = RB.build(await _fetch(panda, TASK_Q), SPELLINGS, TABLES)
    # the moment R read the counts and the rows, cut down to its second
    assert out["counts"]["as_of"] == out["job_rows"]["as_of"] == "2026-10-13 10:00:00"
    assert out["counts"]["clock"] == out["job_rows"]["clock"] == "database"
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
    # a task's jobs' histories hold their columns too
    assert out["job_statuslog"]["by_pandaid"]["12"][1]["computingSite"] == "SITE_A"
    # the task's row was asked apart from R's batch: its moment and clock are its own, on every column
    task = out["rows"]["task"]
    read = {
        "source": "task/get_detailed_info, asked at 2026-10-13 10:00:05 on this machine's clock (UTC)",
        "as_of": "2026-10-13 10:00:05",
        "clock": "client",
    }
    assert task["jediTaskID"] == {"value": TID, **read}
    # the column as stored, not the server's decoding; what is not a column is named, not kept
    assert task["splitRule"] == {"value": "US=2,SS=5", **read}
    assert "split_rule_raw" not in task and "nFiles" not in task
    assert "after" not in out
    assert (
        "2 key(s) of the answers that are not columns of their class, left out: ['nFiles', 'raw_task_params']"
        in out["about"]
    )


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


def test_a_class_with_no_column_list_of_string_constants_has_no_spelling(tmp_path):
    # pin m's DataCarousel.py declares its columns as AttributeWithType calls (the map's note, 2026-10-29)
    pin = tmp_path / "pin"
    (pin / "pkg").mkdir(parents=True)
    (pin / "pkg" / "DataCarousel.py").write_text(
        "class DataCarouselRequestSpec(object):\n"
        "    attributes_with_types = (AttributeWithType('request_id', int),)\n"
    )
    dump = tmp_path / "dump.json"
    dump.write_text(
        json.dumps(
            {"declaration_yields": {"DataCarouselRequestSpec (pkg/DataCarousel.py)": 1}}
        )
    )
    got = RB.column_spellings(pin, dump, ("DataCarouselRequestSpec", "Undeclared"))
    assert got == {}


def test_a_classs_key_tables_creation_and_archive_come_from_the_dumps_tables():
    # what the fetch wrote by hand up to 2026-10-29, the order of the live tables aside
    assert TABLES.key == {
        "JobSpec": "PandaID",
        "JediTaskSpec": "jediTaskID",
        "FileSpec": "row_ID",
    }
    assert TABLES.created == {
        "JobSpec": "creationTime",
        "JediTaskSpec": "creationDate",
        "JediDatasetSpec": "creationTime",
    }
    assert TABLES.live["JobSpec"] == ("jobsActive4", "jobsArchived4", "jobsDefined4")
    assert TABLES.archive["JobSpec"] == ("jobsArchived",)
    assert TABLES.since == {"JobSpec": "modificationTime"}
    # a task's table is its own family, with no archive to bound
    assert (
        TABLES.live["JediTaskSpec"] == ("JEDI_Tasks",)
        and not TABLES.archive["JediTaskSpec"]
    )
    # a row named by two columns has no key; a table the code spells two ways gives its class no tables
    assert "JediDatasetSpec" not in TABLES.key and "FileSpec" not in TABLES.live
    assert FE.Tables.of({}) == FE.Tables({}, {}, {}, {}, {})


def test_a_key_of_a_class_with_no_plan_yet_is_refused():
    with pytest.raises(FE.FieldTestError, match="no plan yet for a key of FileSpec"):
        FE.asyncio.run(
            FE.fetch(
                {"subject": "FileSpec.status", "key": "1"},
                TABLES,
                FakePanda({}, [], []).parts(),
            )
        )


def test_the_fetch_record_holds_every_request_and_response(tmp_path):
    rec = FE.asyncio.run(FE.fetch(JOB_Q, TABLES, _job_panda().parts()))
    path = tmp_path / "answers.json"
    path.write_text(json.dumps(rec))
    again = json.loads(Path(path).read_text())
    # JSON spells the job's ID as a string; the rows file is the same
    assert list(again["steps"][0]["response"]["jobs"]) == [str(PID)]
    assert RB.build(again, SPELLINGS, TABLES) == RB.build(rec, SPELLINGS, TABLES)
    assert again["steps"][0]["request"] == {"job_ids": [PID]}
    assert again["steps"][1]["response"]["rows"][0]["jobStatus"] == "pending"


# ------------------------------------------------------------------ the same rows by link


async def test_a_jobs_rows_file_says_its_files_again_by_the_link_that_named_them():
    out = RB.build(await _fetch(_job_panda(), JOB_Q), SPELLINGS, TABLES)
    by_via = {(e["via"]["from"], e["via"]["to"]): e for e in out["linked"]}
    # the files are a job's one link: its rows name no dataset that is asked
    assert list(by_via) == [("JobSpec.PandaID", "FileSpec.PandaID")]
    files = by_via[("JobSpec.PandaID", "FileSpec.PandaID")]
    # held in the job's own answer: every file is the link's, read when the job was
    assert files["class"] == "FileSpec" and files["asked"] == [PID]
    assert [f["lfn"] for f in files["rows"]] == ["a.log.tgz", "in.root"]
    assert (files["as_of"], files["clock"]) == ("2026-10-13 10:00:05", "client")
    assert "table" not in files
    log = out["statuslogs"]["JobSpec"]
    assert log["key"] == "JobSpec.PandaID"
    assert log["by_key"] == {str(PID): out["statuslog"]["states"]}
    assert (log["as_of"], log["retained_from"]) == (
        out["statuslog"]["as_of"],
        out["statuslog"]["retained_from"],
    )


async def test_a_job_not_answered_links_nothing_and_its_history_stands_by_its_key():
    out = RB.build(await _fetch(_job_panda(job=None), JOB_Q), SPELLINGS, TABLES)
    assert out["linked"] == []
    assert list(out["statuslogs"]["JobSpec"]["by_key"]) == [str(PID)]


async def test_a_tasks_rows_file_says_its_jobs_again_by_link_and_table_and_each_history_by_its_rows_key():
    out = RB.build(await _fetch(_task_panda(), TASK_Q), SPELLINGS, TABLES)
    via = {"from": "JediTaskSpec.jediTaskID", "to": "JobSpec.jediTaskID"}
    jobs = [e for e in out["linked"] if e["via"] == via]
    # read by the task's key, so every row R gave is the link's; a PandaID in two tables is in both
    assert [(e["table"], [r["PandaID"] for r in e["rows"]]) for e in jobs] == [
        ("jobsArchived4", [11, 12]),
        ("jobsArchived", [11]),
    ]
    assert all(e["asked"] == [TID] and e["clock"] == "database" for e in jobs)
    # spelled as the class spells its columns, not R's lower case
    assert jobs[0]["rows"][0]["jobStatus"] == "finished"
    logs = out["statuslogs"]
    assert logs["JediTaskSpec"]["key"] == "JediTaskSpec.jediTaskID"
    assert [s["status"] for s in logs["JediTaskSpec"]["by_key"][str(TID)]] == [
        "running",
        "finished",
    ]
    assert {
        k: [s["status"] for s in v] for k, v in logs["JobSpec"]["by_key"].items()
    } == {
        "11": ["running", "finished"],
        "12": ["running", "failed"],
    }
