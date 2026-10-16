"""Ask PanDA for the rows a case's key leads to, and keep every answer as it came.

    python -m bamboo.scripts.panda.fieldtest.fetch <case.json> --out <answers.json>

Run it where PanDA's credentials load: the ``.env`` holding the OIDC token is
read only from /Users/tmaeno/bamboo or below.

**What is asked.**  The key's row, then the rows that hold the key: for a job,
its file rows, the datasets they name and its status history; for a task, its
status history, its jobs, their number by table and status, and the jobs'
status histories.  These are the rows the querier's rows file holds.  Nothing
here knows a case: the key's class (the question's subject) picks the plan.
Which column of a row holds which class's key is PanDA's schema, written here
(:data:`KEY`, :data:`LIVE`, :data:`DATASET_NAMES`); the map does not export
the joins.

**Two ways in.**  A row read by its key -- a job with its files, a task, a
dataset -- comes from a plain endpoint, answered at once.  The task's jobs and
their counts are read through ``db_read``, which runs in the async daemon's
cycle and may wait behind its greps; one batch holds all four of its queries.

**The archive.**  ``job/get_description_incl_archive`` looks in the archive
for the last 90 days only.  The task's jobs are asked of ATLAS_PANDAARCH with
a lower bound, since its tables are partitioned by ``modificationTime``: the
task's creation less :data:`CLOCK_SLACK` (the creating server's clock and the
database's differ), else :data:`NO_CREATION` before the task was asked, which
covers the 93 days the status history keeps.

**The record.**  Every step's request and response, in order, with the role
each query plays (:mod:`rows` lays a record out as a rows file).  A refused or
failed answer is kept as it came; nothing is merged or dropped here.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable

from bamboo.scripts.panda.fieldtest.run import FieldTestError, load_case
from bamboo.utils import panda_client as PC

#: the key column of a class's rows, as the R queries spell it
KEY = {"JobSpec": "PandaID", "JediTaskSpec": "jediTaskID"}
#: the live tables a job's row is in
LIVE = {"JobSpec": ("jobsDefined4", "jobsActive4", "jobsArchived4")}
#: the archive table of a job's row, partitioned by modificationTime
ARCHIVE = {"JobSpec": "jobsArchived"}
#: the column saying when a row of the class was made, as the class spells it
CREATED = {"JobSpec": "creationTime", "JediTaskSpec": "creationDate"}
#: the columns of a job's rows and file rows that hold a dataset's name
DATASET_NAMES = {
    "JobSpec": ("prodDBlock", "dispatchDBlock", "destinationDBlock"),
    "FileSpec": ("dataset", "prodDBlock", "dispatchDBlock", "destinationDBlock"),
}
#: where a job's answer holds its file rows
FILES = "Files"
#: the columns of a task's jobs: the querier reads pandaid and creationtime, people the rest
JOB_ROW_COLUMNS = [
    "PandaID",
    "jobStatus",
    "jobSubStatus",
    "creationTime",
    "modificationTime",
    "stateChangeTime",
    "endTime",
    "prodSourceLabel",
    "currentPriority",
    "computingSite",
    "attemptNr",
]
CLOCK_SLACK = datetime.timedelta(days=1)
NO_CREATION = datetime.timedelta(days=100)


@dataclass
class Parts:
    """The API parts a fetch calls; a test hands in stand-ins."""

    db_read: Callable[[list], Awaitable[dict]] = PC.db_read
    jobs: Callable[[list], Awaitable[dict]] = PC.get_jobs_incl_archive
    task: Callable[[int], Awaitable[dict]] = PC.get_task_detailed_info
    datasets: Callable[[list], Awaitable[dict]] = PC.get_datasets_by_name
    job_history: Callable[..., Awaitable[dict]] = PC.get_job_status_history
    task_history: Callable[[int], Awaitable[dict]] = PC.get_task_status_history


@dataclass
class Record:
    question: dict
    steps: list[dict] = field(default_factory=list)

    async def read(
        self, parts: Parts, asks: list[tuple[str, dict]]
    ) -> list[dict | None]:
        """Run one batch of (role, query); the answers by position (None when the batch gave none)."""
        queries = [q for _, q in asks]
        response = await parts.db_read(queries)
        self.steps.append(
            {
                "part": "db_read",
                "roles": [r for r, _ in asks],
                "request": {"queries": queries},
                "response": response,
            }
        )
        data = response.get("data") if response.get("success") else None
        if not isinstance(data, dict) or "answers" not in data:
            return [None] * len(asks)
        return data["answers"]

    async def call(self, part: str, call: Awaitable[dict], request: dict) -> dict:
        response = await call
        self.steps.append({"part": part, "request": request, "response": response})
        return response


def when(value: Any) -> datetime.datetime | None:
    """A time the API wrote (``{"__datetime__": iso}``), as a naive UTC datetime."""
    if isinstance(value, dict) and "__datetime__" in value:
        return datetime.datetime.fromisoformat(value["__datetime__"]).replace(
            tzinfo=None
        )
    return None


def rows_of(answer: dict | None) -> list[tuple[str, dict]] | None:
    """(table, row) of a done answer, every table's; None for an answer that is not done."""
    if not answer or answer.get("status") != "done":
        return None
    return [(t["table"], r) for t in answer["tables"] for r in t["rows"]]


def job_of(response: dict | None, key: Any) -> tuple[dict | None, str]:
    """The job's answer (its columns and :data:`FILES`), or None and why there is none."""
    if not response or not response.get("success"):
        return None, f"{(response or {}).get('message') or 'not asked'}"
    # the record's JSON spells the IDs as strings
    found = {str(k): v for k, v in response["jobs"].items()}.get(str(key))
    if found is None:
        return None, "in no live table, nor in the archive's last 90 days"
    if "error" in found:
        return None, f"the database failed reading it: {found['error']}"
    return found, ""


def task_of(response: dict | None) -> tuple[dict | None, str]:
    """The task's answer, or None and why there is none.

    The server says the same for a task it has no row of and one it failed to
    read.
    """
    if not response or not response.get("success"):
        return None, f"{(response or {}).get('message') or 'not asked'}"
    return response["data"], ""


def dataset_names(job: dict) -> list[str]:
    """Every dataset name the job's row and file rows hold, once each."""
    held = [job.get(c) for c in DATASET_NAMES["JobSpec"]] + [
        f.get(c) for f in job.get(FILES) or [] for c in DATASET_NAMES["FileSpec"]
    ]
    return sorted({n for n in held if n})


def asked_at(response: dict) -> datetime.datetime:
    return datetime.datetime.strptime(response["asked_at"], "%Y-%m-%d %H:%M:%S")


def query(
    cls: str, where: list, *, tables: tuple[str, ...] | None = None, **rest: Any
) -> dict:
    return {"tables": list(tables or LIVE[cls]), "where": where, **rest}


def since(t: datetime.datetime) -> list:
    return ["modificationTime", ">=", t.isoformat()]


def lower_bound(
    created: datetime.datetime | None, asked: datetime.datetime
) -> datetime.datetime:
    return created - CLOCK_SLACK if created is not None else asked - NO_CREATION


async def fetch_job(key: int, rec: Record, parts: Parts) -> None:
    response = await rec.call("jobs", parts.jobs([key]), {"job_ids": [key]})
    job, _ = job_of(response, key)
    if job is not None and (names := dataset_names(job)):
        await rec.call("datasets", parts.datasets(names), {"names": names})
    created = when(job.get(CREATED["JobSpec"])) if job else None
    window = {"since": f"{created:%Y-%m-%d %H:%M:%S}"} if created else {}
    await rec.call(
        "job_history", parts.job_history([key], **window), {"job_ids": [key], **window}
    )


async def fetch_task(key: int, rec: Record, parts: Parts) -> None:
    response = await rec.call("task", parts.task(key), {"task_id": key})
    task, _ = task_of(response)
    created = when(task.get(CREATED["JediTaskSpec"])) if task else None
    by_key = [["jediTaskID", "=", key]]
    bounded = by_key + [since(lower_bound(created, asked_at(response)))]
    arch = (ARCHIVE["JobSpec"],)
    counted = {"group_by": ["jobStatus"], "aggregates": [["count", "*"]]}
    live_rows, _, arch_rows, _ = await rec.read(
        parts,
        [
            ("job_rows", query("JobSpec", by_key, columns=JOB_ROW_COLUMNS)),
            ("job_counts", query("JobSpec", by_key, **counted)),
            (
                "job_rows",
                query("JobSpec", bounded, tables=arch, columns=JOB_ROW_COLUMNS),
            ),
            ("job_counts", query("JobSpec", bounded, tables=arch, **counted)),
        ],
    )
    await rec.call("task_history", parts.task_history(key), {"task_id": key})
    found = (rows_of(live_rows) or []) + (rows_of(arch_rows) or [])
    ids = sorted({int(r["pandaid"]) for _, r in found})
    if ids:
        made = min(
            (t for _, r in found if (t := when(r.get(CREATED["JobSpec"].lower())))),
            default=None,
        )
        window = {"since": f"{made:%Y-%m-%d %H:%M:%S}"} if made else {}
        await rec.call(
            "job_history", parts.job_history(ids, **window), {"job_ids": ids, **window}
        )


def key_class(question: dict) -> str:
    return question["subject"].partition(".")[0]


async def fetch(question: dict, parts: Parts | None = None) -> dict:
    """The record of everything asked for *question*'s key."""
    parts = parts or Parts()
    cls = key_class(question)
    rec = Record(question)
    started = datetime.datetime.now(datetime.timezone.utc)
    if cls == "JobSpec":
        await fetch_job(int(question["key"]), rec, parts)
    elif cls == "JediTaskSpec":
        await fetch_task(int(question["key"]), rec, parts)
    else:
        raise FieldTestError(f"no plan yet for a key of {cls}")
    return {
        "about": "Every request this fetch made and every response as it came, in order (bamboo.scripts.panda.fieldtest.fetch).",
        "question": question,
        "key_class": cls,
        "server": os.environ.get("PANDA_API_URL_SSL", ""),
        "started": started.isoformat(),
        "finished": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "steps": rec.steps,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m bamboo.scripts.panda.fieldtest.fetch",
        description=__doc__.split("\n\n")[0],
    )
    ap.add_argument("case")
    ap.add_argument("--out", required=True, help="where to write the record")
    a = ap.parse_args(argv)
    case = load_case(Path(a.case))
    record = asyncio.run(fetch(case["question"]))
    Path(a.out).write_text(json.dumps(record, indent=1) + "\n")
    print(
        f"{a.out}: {len(record['steps'])} step(s) for {record['key_class']} {case['question']['key']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
