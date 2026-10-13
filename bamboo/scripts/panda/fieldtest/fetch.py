"""Ask PanDA for the rows a case's key leads to, and keep every answer as it came.

    python -m bamboo.scripts.panda.fieldtest.fetch <case.json> --out <answers.json>

Run it where PanDA's credentials load: the ``.env`` holding the OIDC token is
read only from /Users/tmaeno/bamboo or below.

**What is asked.**  The key's row, then the rows that hold the key: for a job,
its file rows and its status history; for a task, its status history, its
jobs, their number by table and status, and the jobs' status histories.  These
are the rows the querier's rows file holds.  Nothing here knows a case: the
key's class (the question's subject) picks the plan.  Which column of a row
holds which class's key is PanDA's schema, written here (:data:`KEY`,
:data:`LIVE`); the map does not export the joins.

**The archive.**  ATLAS_PANDAARCH's tables are partitioned by
``modificationTime``, so they are asked with a lower bound: the key's
creation time less :data:`CLOCK_SLACK` (the creating server's clock and the
database's differ), else :data:`NO_CREATION` before the first answer's
``as_of``, which covers the 93 days the status history keeps.  The archive is
asked only for what the live tables did not hold.

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
#: the live tables a class's rows are in
LIVE = {
    "JobSpec": ("jobsDefined4", "jobsActive4", "jobsArchived4"),
    "FileSpec": ("filesTable4",),
    "JediTaskSpec": ("JEDI_Tasks",),
}
#: the archive table of a class, partitioned by modificationTime
ARCHIVE = {"JobSpec": "jobsArchived", "FileSpec": "filesTable_ARCH"}
#: the column saying when a row of the class was made, lower case as R answers
CREATED = {"JobSpec": "creationtime", "JediTaskSpec": "creationdate"}
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
    job_history: Callable[..., Awaitable[dict]] = PC.get_job_status_history
    task_history: Callable[[int], Awaitable[dict]] = PC.get_task_status_history


@dataclass
class Record:
    question: dict
    steps: list[dict] = field(default_factory=list)

    async def read(
        self, parts: Parts, asks: list[tuple[str, dict]]
    ) -> tuple[list[dict | None], datetime.datetime | None]:
        """Run one batch of (role, query); the answers by position (None when the batch gave none) and its as_of."""
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
            return [None] * len(asks), None
        return data["answers"], when(data["as_of"])

    async def history(self, part: str, call: Awaitable[dict], request: dict) -> dict:
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


def query(
    cls: str, where: list, *, tables: tuple[str, ...] | None = None, **rest: Any
) -> dict:
    return {"tables": list(tables or LIVE[cls]), "where": where, **rest}


def since(t: datetime.datetime) -> list:
    return ["modificationTime", ">=", t.isoformat()]


def lower_bound(
    created: datetime.datetime | None, as_of: datetime.datetime | None
) -> datetime.datetime:
    if created is not None:
        return created - CLOCK_SLACK
    if as_of is None:
        raise FieldTestError("no creation time and no as_of to bound the archive with")
    return as_of - NO_CREATION


async def fetch_job(key: int, rec: Record, parts: Parts) -> None:
    by_key = [["PandaID", "=", key]]
    (job, files), as_of = await rec.read(
        parts, [("row", query("JobSpec", by_key)), ("files", query("FileSpec", by_key))]
    )
    job_rows, file_rows = rows_of(job), rows_of(files)
    created = when(job_rows[0][1].get(CREATED["JobSpec"])) if job_rows else None
    archive = []
    if job_rows == []:
        archive.append(("row", ARCHIVE["JobSpec"], "JobSpec"))
    if file_rows == []:
        archive.append(("files", ARCHIVE["FileSpec"], "FileSpec"))
    if archive:
        bound = since(lower_bound(created, as_of))
        answers, _ = await rec.read(
            parts,
            [
                (role, query(cls, by_key + [bound], tables=(table,)))
                for role, table, cls in archive
            ],
        )
        for (role, _, _), answer in zip(archive, answers, strict=True):
            if role == "row" and (found := rows_of(answer)):
                created = when(found[0][1].get(CREATED["JobSpec"]))
    window = {"since": f"{created:%Y-%m-%d %H:%M:%S}"} if created else {}
    await rec.history(
        "job_history", parts.job_history([key], **window), {"job_ids": [key], **window}
    )


async def fetch_task(key: int, rec: Record, parts: Parts) -> None:
    by_key = [["jediTaskID", "=", key]]
    jobs = query("JobSpec", by_key, columns=JOB_ROW_COLUMNS)
    counts = query(
        "JobSpec", by_key, group_by=["jobStatus"], aggregates=[["count", "*"]]
    )
    (task, job_rows, _), as_of = await rec.read(
        parts,
        [
            ("row", query("JediTaskSpec", by_key)),
            ("job_rows", jobs),
            ("job_counts", counts),
        ],
    )
    task_rows = rows_of(task)
    created = when(task_rows[0][1].get(CREATED["JediTaskSpec"])) if task_rows else None
    arch = (ARCHIVE["JobSpec"],)
    bounded = by_key + [since(lower_bound(created, as_of))]
    (arch_rows, _), _ = await rec.read(
        parts,
        [
            (
                "job_rows",
                query("JobSpec", bounded, tables=arch, columns=JOB_ROW_COLUMNS),
            ),
            (
                "job_counts",
                query(
                    "JobSpec",
                    bounded,
                    tables=arch,
                    group_by=["jobStatus"],
                    aggregates=[["count", "*"]],
                ),
            ),
        ],
    )
    await rec.history("task_history", parts.task_history(key), {"task_id": key})
    found = (rows_of(job_rows) or []) + (rows_of(arch_rows) or [])
    ids = sorted({int(r["pandaid"]) for _, r in found})
    if ids:
        made = min(
            (t for _, r in found if (t := when(r.get(CREATED["JobSpec"])))),
            default=None,
        )
        window = {"since": f"{made:%Y-%m-%d %H:%M:%S}"} if made else {}
        await rec.history(
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
