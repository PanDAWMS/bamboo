"""Ask PanDA for the rows a case's key leads to, and keep every answer as it came.

    python -m bamboo.scripts.panda.fieldtest.fetch <case.json> --dump <map dump> --out <answers.json>

Run it where PanDA's credentials load: the ``.env`` holding the OIDC token is
read only from /Users/tmaeno/bamboo or below.

**What is asked.**  One plan for every class (:func:`fetch_key`): the key's
row, the rows its links name one step on, then the status histories -- the
key's, and those of the rows the links found.  For a job that is its file rows
(in its own answer) and its history; for a task, its jobs and their number by
table and status, its history, and the jobs' histories.  These are the rows the querier's rows file holds.  Nothing
here knows a case: the key's class (the question's subject) picks the calls.
A class's key, its tables, the column saying when its rows were made and how
its archive is partitioned are read from the map's dump (:class:`Tables`, the
case's pair's ``MAP_DUMP``); which column of a row names which rows of another
class is written here (:data:`LINKS`) until the map exports it.

**Two ways in.**  A row read by its key -- a job with its files, a task --
comes from a plain endpoint, answered at once.  The task's jobs and
their counts are read through ``db_read``, which runs in the async daemon's
cycle and may wait behind its greps; one batch holds all four of its queries.

**The archive.**  ``job/get_description_incl_archive`` looks in the archive
for the last 90 days only.  The task's jobs are asked of ATLAS_PANDAARCH with
a lower bound, since its tables are partitioned by time (``modificationTime``): the
task's creation less :data:`CLOCK_SLACK` (the creating server's clock and the
database's differ), else :data:`NO_CREATION` before the task was asked, which
covers the 93 days the status history keeps.

**The record.**  Every step's request and response, in order, with the role
each query plays (:mod:`rows` lays a record out as a rows file), and the dump
the tables were read from (``tables_from``, its path and sha256).  A refused or
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

from bamboo.scripts.panda.fieldtest.run import FieldTestError, load_case, sha256
from bamboo.utils import panda_client as PC

#: where a job's answer holds its file rows
FILES = "Files"


@dataclass(frozen=True)
class Tables:
    """What the map's dump says of each class's tables (its ``tables``), as the fetch asks by them.

    By class: *key*, the attribute naming a row, where the class's tables name
    each row by one; *created*, the attribute saying when a row was made;
    *live*, the tables of the class's family that rows are inserted into;
    *archive*, the family's tables none is inserted into; *since*, the
    attribute the archive is partitioned by; *identity*, every attribute the
    class's tables name a row by (a dataset of a task: ``datasetID`` and
    ``jediTaskID``).  A table is spelled as PanDA's
    code spells it, as ``db_read`` takes it, in the order of its name.  A class
    the dump gives none of these, or no one family spelled one way, has no
    entry.  The rules are the harness's reading of the dump, which the map
    confirmed (to-map-28).
    """

    key: dict[str, str]
    created: dict[str, str]
    live: dict[str, tuple[str, ...]]
    archive: dict[str, tuple[str, ...]]
    since: dict[str, str]
    identity: dict[str, tuple[str, ...]] = field(default_factory=dict)

    @classmethod
    def read(cls, dump: Path) -> Tables:
        return cls.of(json.loads(dump.read_text()))

    @classmethod
    def of(cls, dump: dict) -> Tables:
        by_class: dict[str, list[dict]] = {}
        for t in dump.get("tables") or []:
            if t["rows_of"]:
                by_class.setdefault(t["rows_of"], []).append(t)
        out = cls({}, {}, {}, {}, {})
        for c, tables in by_class.items():
            if attr := _one(a for t in tables for a in t["identity_attrs"]):
                out.key[c] = attr
            if named := sorted({a for t in tables for a in t["identity_attrs"]}):
                out.identity[c] = tuple(a.partition(".")[2] for a in named)
            if attr := _one(a for t in tables for a in t["creation"]):
                out.created[c] = attr
            spelled = [
                (t, _spelled(t)) for t in sorted(tables, key=lambda t: t["name"])
            ]
            families = {frozenset([t["name"], *t["family"]]) for t in tables}
            if len(families) != 1 or any(s is None for _, s in spelled):
                continue
            out.live[c] = tuple(s for t, s in spelled if t["inserted"])
            out.archive[c] = tuple(s for t, s in spelled if not t["inserted"])
            archived = [t for t in tables if not t["inserted"]]
            if attr := _one(a for t in archived for a in t["partitioned_by_attrs"]):
                out.since[c] = attr
        return out


def _one(values) -> str | None:
    """The attribute (``Class.attr``'s *attr*) of the one value *values* hold; None for none or several."""
    found = set(values)
    return found.pop().partition(".")[2] if len(found) == 1 else None


def _spelled(table: dict) -> str | None:
    """The table as PanDA's code spells it besides the dump's name; None where the code spells it several ways."""
    code = sorted(set(table["spellings"]) - {table["name"]})
    return table["name"] if not code else code[0] if len(code) == 1 else None


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


#: which column of a class's rows names which rows of another.  The datasets a job's rows name
#: (prodDBlock, dispatchDBlock, destinationDBlock, dataset -> DatasetSpec.name) are not asked:
#: the querier reads no Datasets row (its H6 times a maker by the job's statuslog, querier-vi)
LINKS = (
    Link("JobSpec.PandaID", "FileSpec.PandaID", held=FILES),
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


def asked_at(response: dict) -> datetime.datetime:
    return datetime.datetime.strptime(response["asked_at"], "%Y-%m-%d %H:%M:%S")


def query(tables: tuple[str, ...], where: list, **rest: Any) -> dict:
    return {"tables": list(tables), "where": where, **rest}


def since(attr: str, t: datetime.datetime) -> list:
    return [attr, ">=", t.isoformat()]


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


#: the part reading a key's row, and its request, by the row's class
KEY_REQUEST: dict[str, tuple[str, Callable[[Any], dict]]] = {
    "JobSpec": ("jobs", lambda key: {"job_ids": [key]}),
    "JediTaskSpec": ("task", lambda key: {"task_id": key}),
}


async def _job_row(key: int, rec: Record, parts: Parts) -> tuple[dict | None, dict]:
    part, request = KEY_REQUEST["JobSpec"]
    response = await rec.call(part, parts.jobs([key]), request(key))
    return job_of(response, key)[0], response


async def _task_row(key: int, rec: Record, parts: Parts) -> tuple[dict | None, dict]:
    part, request = KEY_REQUEST["JediTaskSpec"]
    response = await rec.call(part, parts.task(key), request(key))
    return task_of(response)[0], response


async def _task_jobs(
    values: list, asked: Asked, rec: Record, parts: Parts, tables: Tables
) -> list[dict]:
    """A task's jobs in every table and their number by table and status, in one batch."""
    (key,) = values
    created = when(asked.row.get(tables.created[asked.cls])) if asked.row else None
    by_key = [["jediTaskID", "=", key]]
    bound = lower_bound(created, asked_at(asked.response))
    bounded = by_key + [since(tables.since["JobSpec"], bound)]
    live, arch = tables.live["JobSpec"], tables.archive["JobSpec"]
    counted = {"group_by": ["jobStatus"], "aggregates": [["count", "*"]]}
    live_rows, _, arch_rows, _ = await rec.read(
        parts,
        [
            ("job_rows", query(live, by_key, columns=JOB_ROW_COLUMNS)),
            ("job_counts", query(live, by_key, **counted)),
            ("job_rows", query(arch, bounded, columns=JOB_ROW_COLUMNS)),
            ("job_counts", query(arch, bounded, **counted)),
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
LINK_ROWS = {"JobSpec.jediTaskID": _task_jobs}
#: the call reading the status histories of a class's rows, from the first one's creation where it takes a bound
HISTORY = {"JobSpec": _job_history, "JediTaskSpec": _task_history}


def made(attr: str, rows: list[dict]) -> datetime.datetime | None:
    """When the first of *rows* was made, by their *attr*."""
    return min((t for r in rows if (t := when(value_of(r, attr)))), default=None)


def held_rows(cls: str, row: dict | None) -> dict[str, list[dict]]:
    """The key's row and the rows its answer holds (a job's files), by class."""
    rows = {cls: [row] if row is not None else []}
    for link in links_from(cls):
        if link.held:
            rows[link.dst_class] = [
                r for k in rows[cls] for r in k.get(link.held) or []
            ]
    return rows


def values_by(
    link: Link, got: list[dict], key_class: str, key: int, tables: Tables
) -> set:
    """The values *link* holds in *got*, rows of its source class.

    A link from the key's own column holds the key, answered row or not.
    """
    if link.src_class == key_class and link.src_attr == tables.key[key_class]:
        return {key}
    return {v for r in got if (v := value_of(r, link.src_attr))}


def link_values(
    rows: dict[str, list[dict]], dst: str, asked: Asked, tables: Tables
) -> list:
    """Every value the links leading to *dst* hold in *rows*, once each."""
    held = set()
    for cls, got in rows.items():
        for link in links_from(cls):
            if link.dst == dst and not link.held:
                held |= values_by(link, got, asked.cls, asked.key, tables)
    return sorted(held)


async def fetch_key(
    cls: str, key: int, rec: Record, parts: Parts, tables: Tables
) -> None:
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
        if values := link_values(rows, dst, asked, tables):
            got = await LINK_ROWS[dst](values, asked, rec, parts, tables)
            found.setdefault(dst.partition(".")[0], []).extend(got)
    await HISTORY[cls]([key], made(tables.created[cls], rows[cls]), rec, parts)
    for c, got in found.items():
        if c in HISTORY and (
            ids := sorted({int(value_of(r, tables.key[c])) for r in got})
        ):
            await HISTORY[c](ids, made(tables.created[c], got), rec, parts)


def key_class(question: dict) -> str:
    return question["subject"].partition(".")[0]


async def fetch(question: dict, tables: Tables, parts: Parts | None = None) -> dict:
    """The record of everything asked for *question*'s key, by the dump's *tables*."""
    parts = parts or Parts()
    cls = key_class(question)
    rec = Record(question)
    started = datetime.datetime.now(datetime.timezone.utc)
    if cls not in KEY_ROW:
        raise FieldTestError(f"no plan yet for a key of {cls}")
    await fetch_key(cls, int(question["key"]), rec, parts, tables)
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
    ap.add_argument(
        "--dump",
        required=True,
        type=Path,
        help="the map's dump whose tables name each class's key and tables (the case's pair's MAP_DUMP)",
    )
    ap.add_argument("--out", required=True, help="where to write the record")
    a = ap.parse_args(argv)
    case = load_case(Path(a.case))
    record = asyncio.run(fetch(case["question"], Tables.read(a.dump)))
    record["tables_from"] = {"dump": str(a.dump), "sha256": sha256(a.dump)}
    Path(a.out).write_text(json.dumps(record, indent=1) + "\n")
    print(
        f"{a.out}: {len(record['steps'])} step(s) for {record['key_class']} {case['question']['key']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
