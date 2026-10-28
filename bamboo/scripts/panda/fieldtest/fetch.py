"""Ask PanDA for the rows a case's key leads to, and keep every answer as it came.

    python -m bamboo.scripts.panda.fieldtest.fetch <case.json> --out <answers.json>

Run it where PanDA's credentials load: the ``.env`` holding the OIDC token is
read only from /Users/tmaeno/bamboo or below.

**What is asked.**  One plan for every class (:func:`fetch_key`): the key's
row, the rows its links name one step on, then the status histories -- the
key's, and those of the rows the links found.  For a job that is its file rows
(in its own answer), the datasets they and it name, and its history; for a
task, its jobs and their number by table and status, its history, and the
jobs' histories.  These are the rows the querier's rows file holds.  Nothing
here knows a case: the key's class (the question's subject) picks the calls.
Which column of a row names which rows of another class is PanDA's schema,
written here (:data:`LINKS`, with :data:`KEY`, :data:`LIVE`, :data:`ARCHIVE`
and :data:`CREATED`) until the map exports it.

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
#: where a job's answer holds its file rows
FILES = "Files"


@dataclass(frozen=True)
class Link:
    """A column of one class whose values name rows of another, ``Class.attr`` to ``Class.attr``.

    *held*: the key of the source row's answer that holds the rows the link
    names, so no call reads them (a job's files); empty where they are asked.
    """

    src: str
    dst: str
    held: str = ""

    @property
    def src_class(self) -> str:
        return self.src.partition(".")[0]

    @property
    def src_attr(self) -> str:
        return self.src.partition(".")[2]

    @property
    def dst_class(self) -> str:
        return self.dst.partition(".")[0]

    def via(self) -> dict[str, str]:
        """The link as the rows file says it."""
        return {"from": self.src, "to": self.dst}


#: which column of a class's rows names which rows of another
LINKS = (
    Link("JobSpec.PandaID", "FileSpec.PandaID", held=FILES),
    *(
        Link(f"{cls}.{c}", "DatasetSpec.name")
        for cls, cols in (
            ("JobSpec", ("prodDBlock", "dispatchDBlock", "destinationDBlock")),
            (
                "FileSpec",
                ("dataset", "prodDBlock", "dispatchDBlock", "destinationDBlock"),
            ),
        )
        for c in cols
    ),
    Link("JediTaskSpec.jediTaskID", "JobSpec.jediTaskID"),
)
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


def links_from(cls: str) -> list[Link]:
    return [link for link in LINKS if link.src_class == cls]


def value_of(row: dict, attr: str) -> Any:
    """A row's column, as the class spells it (an endpoint's row) or in lower case (R's)."""
    return row.get(attr, row.get(attr.lower()))


def with_and_without_scope(name: str) -> list[str]:
    """A dataset's name as written, and without its scope (``scope:name``) where it has one."""
    return [name, name.split(":")[-1]] if ":" in name else [name]


#: the forms a value is asked by where a link leads, where a row may be keyed by more than one: a dataset's
#: name with and without its scope, as PanDA asks one by name (``datasetName IN (:name1,:name2)``,
#: resetFileStatusInJEDI), since the Datasets rows are keyed by either
ASKED_AS = {"DatasetSpec.name": with_and_without_scope}


def asked_as(dst: str, values: set) -> set:
    """Every form *values* are asked by where *dst* leads."""
    form = ASKED_AS.get(dst)
    return {f for v in values for f in form(v)} if form else set(values)


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


@dataclass
class Asked:
    """The key's row as its call answered it, which a link's call may bound itself by."""

    cls: str
    key: int
    row: dict | None
    response: dict


async def _job_row(key: int, rec: Record, parts: Parts) -> tuple[dict | None, dict]:
    response = await rec.call("jobs", parts.jobs([key]), {"job_ids": [key]})
    return job_of(response, key)[0], response


async def _task_row(key: int, rec: Record, parts: Parts) -> tuple[dict | None, dict]:
    response = await rec.call("task", parts.task(key), {"task_id": key})
    return task_of(response)[0], response


async def _datasets(
    values: list, asked: Asked, rec: Record, parts: Parts
) -> list[dict]:
    response = await rec.call("datasets", parts.datasets(values), {"names": values})
    return (response.get("rows") or []) if response.get("success") else []


async def _task_jobs(
    values: list, asked: Asked, rec: Record, parts: Parts
) -> list[dict]:
    """A task's jobs in every table and their number by table and status, in one batch."""
    (key,) = values
    created = when(asked.row.get(CREATED[asked.cls])) if asked.row else None
    by_key = [["jediTaskID", "=", key]]
    bounded = by_key + [since(lower_bound(created, asked_at(asked.response)))]
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
    return [r for _, r in (rows_of(live_rows) or []) + (rows_of(arch_rows) or [])]


async def _job_history(
    ids: list[int], made: datetime.datetime | None, rec: Record, parts: Parts
) -> None:
    window = {"since": f"{made:%Y-%m-%d %H:%M:%S}"} if made else {}
    await rec.call(
        "job_history", parts.job_history(ids, **window), {"job_ids": ids, **window}
    )


async def _task_history(
    ids: list[int], made: datetime.datetime | None, rec: Record, parts: Parts
) -> None:
    """A task's history is asked whole."""
    (key,) = ids
    await rec.call("task_history", parts.task_history(key), {"task_id": key})


#: the call reading a key's row, by its class: the row (None when there is none) and the response
KEY_ROW = {"JobSpec": _job_row, "JediTaskSpec": _task_row}
#: the call reading the rows a link names, by where it leads: the rows found
LINK_ROWS = {"DatasetSpec.name": _datasets, "JobSpec.jediTaskID": _task_jobs}
#: the call reading the status histories of a class's rows, from the first one's creation where it takes a bound
HISTORY = {"JobSpec": _job_history, "JediTaskSpec": _task_history}


def made(cls: str, rows: list[dict]) -> datetime.datetime | None:
    """When the first of *rows* was made."""
    return min(
        (t for r in rows if (t := when(value_of(r, CREATED[cls])))), default=None
    )


def held_rows(cls: str, row: dict | None) -> dict[str, list[dict]]:
    """The key's row and the rows its answer holds (a job's files), by class."""
    rows = {cls: [row] if row is not None else []}
    for link in links_from(cls):
        if link.held:
            rows[link.dst_class] = [
                r for k in rows[cls] for r in k.get(link.held) or []
            ]
    return rows


def values_by(link: Link, got: list[dict], key_class: str, key: int) -> set:
    """The values *link* holds in *got*, rows of its source class.

    A link from the key's own column holds the key, answered row or not.
    """
    if link.src_class == key_class and link.src_attr == KEY[key_class]:
        return {key}
    return {v for r in got if (v := value_of(r, link.src_attr))}


def link_values(rows: dict[str, list[dict]], dst: str, asked: Asked) -> list:
    """Every value the links leading to *dst* hold in *rows*, in every form it is asked by, once each."""
    held = set()
    for cls, got in rows.items():
        for link in links_from(cls):
            if link.dst == dst and not link.held:
                held |= asked_as(dst, values_by(link, got, asked.cls, asked.key))
    return sorted(held)


async def fetch_key(cls: str, key: int, rec: Record, parts: Parts) -> None:
    """The key's row, the rows its links name one step on, then the histories of the key and of the rows found.

    The rows one step on are named by the key's row and the rows its answer
    holds (a job's files); all the values of the links leading to one place
    are asked in one call.
    """
    row, response = await KEY_ROW[cls](key, rec, parts)
    asked = Asked(cls, key, row, response)
    rows = held_rows(cls, row)
    found: dict[str, list[dict]] = {}
    for dst in dict.fromkeys(
        link.dst for c in rows for link in links_from(c) if not link.held
    ):
        if values := link_values(rows, dst, asked):
            got = await LINK_ROWS[dst](values, asked, rec, parts)
            found.setdefault(dst.partition(".")[0], []).extend(got)
    await HISTORY[cls]([key], made(cls, rows[cls]), rec, parts)
    for c, got in found.items():
        if c in HISTORY and (ids := sorted({int(value_of(r, KEY[c])) for r in got})):
            await HISTORY[c](ids, made(c, got), rec, parts)


def key_class(question: dict) -> str:
    return question["subject"].partition(".")[0]


async def fetch(question: dict, parts: Parts | None = None) -> dict:
    """The record of everything asked for *question*'s key."""
    parts = parts or Parts()
    cls = key_class(question)
    rec = Record(question)
    started = datetime.datetime.now(datetime.timezone.utc)
    if cls not in KEY_ROW:
        raise FieldTestError(f"no plan yet for a key of {cls}")
    await fetch_key(cls, int(question["key"]), rec, parts)
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
