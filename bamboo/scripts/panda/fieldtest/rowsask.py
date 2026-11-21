"""Plan the answers to the querier's rows asks: for each ask, the calls that read the rows it names, or why none can.

    python -m bamboo.scripts.panda.fieldtest.rowsask <rows-asks.jsonl> --dump <map dump> [--out <plan.json>]
        [--records <dir>]

The querier writes a rows ask where a step needs rows its rows file lacks (``--rows-asks``, one JSON line
each: ``{"rows": {...}, "step", "why"}``; stage 1 of the rows ask).  This says, by rule and not by case, how
each is answered; with ``--records`` it asks PanDA (run it where PanDA's credentials load, as the fetch) and
writes the round's answers to ``rows-r<n>.json`` there, ``n`` one past the directory's last
(:func:`answer`; :func:`~bamboo.scripts.panda.fieldtest.rows.answered` lays them out):

- ``collection`` -- a row's attribute holding rows (``{"row": {"entity", "key"}, "attr", "entity"}``), or
  the rows of a class a statement joins to the row (S1's, ``attr`` null, with ``"where": {column: the row's
  value}`` and ``"via": [{"from": "<row's class>.<column>", "to": "<class>.<column>"}]``, querier-nr).  Rows
  the row's own answer holds (:data:`~bamboo.scripts.panda.fieldtest.fetch.LINKS` ``held``: a job's
  ``Files``, asked by the attribute or by the link's pair) are read by the call reading the row by its key.
  Else the ask names the element class: a joined class is read by ``db_read`` on its tables where the
  joins' ``to`` columns hold the row's values; an attribute's, where its tables name a row by the row's key
  among others (a task's datasets: ``JEDI_Datasets`` by ``datasetID`` and ``jediTaskID``), where that
  column holds the key.  An ask may name the attributes it reads (``"columns"``); the read asks them with
  the columns it selects by and, where histories follow, the class's key and creation column, and every
  column without them.  The rows of a class whose status histories a call reads
  (:data:`~bamboo.scripts.panda.fieldtest.fetch.HISTORY`: a job's, a task's) are followed by their
  histories, as a statement's are (a job's task, a task's parent; design H).
- ``statement`` -- the rows a call's statements read, per pass (``{"method", "site", "tables", "binds",
  "at"}``).  The read sites at ``site`` name the columns the statements select by (the dump's
  ``read_sites`` ``keys``: ``JobSpec.jediTaskID``); the call's argument of each one's name gives its values
  (a value the run did not show, ``"unknown"``, is left out).  ``db_read`` reads the ask's tables -- the
  ones the call's constants leave -- by them: the rows, and their number by status; the rows' histories
  follow once the rows are in.  An archive table is bounded by the read site's own clock term on the
  partition column (``modificationTime>(CURRENT_DATE-30)``) before the first pass, else by
  :data:`~bamboo.scripts.panda.fieldtest.fetch.NO_CREATION`.
- ``sql`` -- an answer L3 lacks (``{"name", "sql", "binds", "question"}``), a window over the history no call
  reads: listed for a hand, whose answer goes in the records directory as ``hand-r<n>.json``
  (:func:`~bamboo.scripts.panda.fieldtest.rows.hand_answers`).
- ``row`` -- another row of a class a fetch reads by its key (``{"row": {"entity", "key"}}``: a job a mark
  names by its ``PandaID``, a task by its ``jediTaskID``), read as a fetch reads its own key's
  (:data:`~bamboo.scripts.panda.fieldtest.fetch.KEY_REQUEST`), then its status history from its creation
  (:data:`~bamboo.scripts.panda.fieldtest.fetch.HISTORY`).

An ask no rule answers says why (``why``); stage 2's record keeps it, so the asks no API answers are counted
from the runs.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import json
import os
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

from bamboo.scripts.panda.fieldtest.fetch import (
    CLOCK_SLACK,
    DEFAULT_VIA,
    HISTORY,
    JOB_ROW_COLUMNS,
    KEY_REQUEST,
    NO_CREATION,
    TASK_DATASET_ROWS,
    TASK_DATASET_TABLE,
    Parts,
    Record,
    Tables,
    job_of,
    links_from,
    made,
    query,
    rows_of,
    since,
    task_of,
    value_of,
)
from bamboo.scripts.panda.fieldtest.run import sha256

#: a value of an argument the querier did not see in the run
UNKNOWN = "unknown"
#: of a class whose rows a statement selects: the columns read, the column they are counted by, and the roles of
#: the two answers as a fetch record names them (``rows`` lays them out as ``job_rows`` and ``counts``)
COLUMNS = {"JobSpec": (JOB_ROW_COLUMNS, "jobStatus", "job_rows", "job_counts")}
#: how the querier stamps a pass's time (a log record's time)
PASS_TIME = "%Y-%m-%d %H:%M:%S,%f"
#: how a plan's call of a part other than db_read is made, from its request
CALLS = {
    "jobs": lambda parts, request: parts.jobs(request["job_ids"]),
    "task": lambda parts, request: parts.task(request["task_id"]),
    TASK_DATASET_ROWS: lambda parts, request: parts.dataset_rows(request["task_id"]),
}
#: the row in the answer of a part reading a row by its key (fetch.KEY_REQUEST), from its request; None where none
KEY_ANSWER = {
    "jobs": lambda response, request: job_of(response, request["job_ids"][0])[0],
    "task": lambda response, request: task_of(response)[0],
}
#: the ways a plan reads rows: db_read (the async daemon), or the endpoints that replace it where one reads the
#: rows asked (task/get_dataset_rows, a task's datasets by jediTaskID; upstream 31f1a14a); fetch.DEFAULT_VIA unless named
VIAS = ("db_read", "endpoint")
#: a round's answers in a records directory
ROUND = re.compile(r"^rows-r(\d+)\.json$")


@dataclass
class Call:
    """One call a plan makes: the part (as a fetch record names it), the role its answer plays, the request."""

    part: str
    role: str
    request: dict


@dataclass
class Plan:
    """How one ask is answered: its calls, the class whose rows' histories follow them, or why it cannot be."""

    ask: dict
    calls: list[Call] = field(default_factory=list)
    histories: str | None = None
    #: the role of the calls whose rows the histories follow
    rows_role: str = ""
    why: str = ""


class Sites:
    """The dump's read sites by ``<file>:<line>``, as the querier names a statement's site."""

    def __init__(self, dump: dict) -> None:
        self.at = {
            f"{s['anchor']['file']}:{s['anchor']['line_start']}": s
            for s in dump.get("read_sites") or []
            if s.get("anchor")
        }


def plan(line: dict, tables: Tables, sites: Sites, via: str = DEFAULT_VIA) -> Plan:
    """The plan for one line of a rows-asks file, its rows read *via* (:data:`VIAS`)."""
    ask = line["rows"]
    kind = ask.get("kind")
    if kind == "collection":
        return _collection(ask, tables, via)
    if kind == "statement":
        return _statement(ask, tables, sites)
    if kind == "sql":
        return _sql(ask)
    if kind == "row":
        return _row(ask, tables)
    return Plan(ask, why=f"no rule for an ask of kind {kind!r}")


def _row(ask: dict, tables: Tables) -> Plan:
    """Another row of a class read by its key: the call a fetch reads its key's row by, then its history."""
    cls, key = ask["row"]["entity"], ask["row"]["key"]
    if cls not in KEY_REQUEST or list(key) != [tables.key.get(cls)]:
        return Plan(ask, why=f"no call reads a {cls} by {sorted(key)}")
    part, request = KEY_REQUEST[cls]
    (value,) = key.values()
    return Plan(
        ask, [Call(part, "row", request(int(value)))], histories=cls, rows_role="row"
    )


def _collection(ask: dict, tables: Tables, way: str = DEFAULT_VIA) -> Plan:
    cls, attr, element = ask["row"]["entity"], ask["attr"], ask.get("entity")
    key = ask["row"]["key"]
    via = ask.get("via") or []
    for link in links_from(cls):
        if link.held and (link.held == attr or link.via() in via):
            if cls not in KEY_REQUEST or len(key) != 1:
                return Plan(
                    ask,
                    why=f"{cls}.{attr} is held by the row's answer, and no call reads a {cls} by {sorted(key)}",
                )
            part, request = KEY_REQUEST[cls]
            (value,) = key.values()
            return Plan(ask, [Call(part, "collection", request(value))])
    if element is None:
        return Plan(
            ask,
            why=f"the ask names no element class, and no row's answer holds {cls}.{attr}",
        )
    if element not in tables.live:
        return Plan(ask, why=f"the dump gives no tables of {element}")
    if "where" in ask:
        return _joined(ask, cls, element, via, tables, way)
    if not set(key) <= set(tables.identity.get(element, ())):
        return Plan(
            ask,
            why=f"{element}'s tables name a row by {list(tables.identity.get(element, ()))}, not by {sorted(key)}",
        )
    return _read(ask, element, key, tables, way)


def _joined(
    ask: dict,
    cls: str,
    element: str,
    via: list[dict],
    tables: Tables,
    way: str = DEFAULT_VIA,
) -> Plan:
    """The rows of *element* a statement's joins name from the row: where the joins' ``to`` columns hold the row's
    values (the ask's ``where``), each join named in ``via`` from the row's class to *element*."""
    where = ask["where"]
    ends = {(p["from"].partition(".")[0], p["to"].partition(".")[0]) for p in via}
    if ends != {(cls, element)} or {p["to"].partition(".")[2] for p in via} != set(
        where
    ):
        return Plan(
            ask,
            why=f"the ask's via {via} does not join {cls} to {element} by its where's columns {sorted(where)}",
        )
    return _read(ask, element, where, tables, way)


def _read(
    ask: dict, element: str, by: dict, tables: Tables, way: str = DEFAULT_VIA
) -> Plan:
    """The rows of *element* where the columns of *by* hold its values, in every table of the class: the ask's
    ``columns`` with the ones it selects by, or every column; then their histories where a call reads a class's.

    Read *way* ``endpoint``, a class whose one table is the one task/get_dataset_rows reads, asked by
    ``jediTaskID`` alone, is read by that endpoint (every column, whatever the ask names)."""
    read = tables.live[element] + tables.archive.get(element, ())
    if (
        way == "endpoint"
        and read == (TASK_DATASET_TABLE,)
        and set(by) == {"jediTaskID"}
        and element not in HISTORY
    ):
        return Plan(
            ask,
            [Call(TASK_DATASET_ROWS, "collection", {"task_id": int(by["jediTaskID"])})],
        )
    terms = [[c, "=", v] for c, v in sorted(by.items())]
    histories = element if element in HISTORY else None
    named = ask.get("columns")
    if named:
        follow = [tables.key[element], tables.created[element]] if histories else []
        named = list(dict.fromkeys([*named, *sorted(by), *follow]))
    request = query(read, terms, **({"columns": named} if named else {}))
    return Plan(
        ask,
        [Call("db_read", "collection", request)],
        histories=histories,
        rows_role="collection" if histories else "",
    )


def _statement(ask: dict, tables: Tables, sites: Sites) -> Plan:
    got = [sites.at.get(s) for s in ask.get("site") or []]
    if not got or None in got:
        missing = [
            s for s, r in zip(ask.get("site") or [], got, strict=True) if r is None
        ]
        return Plan(ask, why=f"no read site in the dump at {missing or 'no site'}")
    found: list[dict] = [r for r in got if r is not None]
    keys = sorted({k for r in found for k in r["keys"]})
    classes = {k.partition(".")[0] for k in keys}
    if len(classes) != 1:
        return Plan(
            ask,
            why=f"the read sites select by {keys or 'no column'}, not by columns of one class",
        )
    (cls,) = classes
    if cls not in COLUMNS:
        return Plan(ask, why=f"no columns are read of {cls}'s rows yet")
    own = tables.live.get(cls, ()) + tables.archive.get(cls, ())
    if not ask["tables"] or not set(ask["tables"]) <= set(own):
        return Plan(
            ask, why=f"the tables {ask['tables']} are not all {cls}'s {list(own)}"
        )
    binds = {name.lower(): values for name, values in ask.get("binds", {}).items()}
    where = []
    for k in keys:
        attr = k.partition(".")[2]
        values = sorted(
            {v for v in binds.get(attr.lower(), []) if v != UNKNOWN}, key=str
        )
        if not values:
            return Plan(
                ask, why=f"no argument of {ask.get('method')} gives a value of {k}"
            )
        where.append([attr, "in", values])
    read = list(ask["tables"])
    columns, counted, rows_role, counts_role = COLUMNS[cls]
    live = [t for t in read if t in tables.live.get(cls, ())]
    arch = [t for t in read if t not in live]
    calls = []
    for group, terms in (
        (live, where),
        (arch, where + _archive_bound(cls, found, ask, tables)),
    ):
        if group:
            calls.append(
                Call("db_read", rows_role, query(tuple(group), terms, columns=columns))
            )
            calls.append(
                Call(
                    "db_read",
                    counts_role,
                    query(
                        tuple(group),
                        terms,
                        group_by=[counted],
                        aggregates=[["count", "*"]],
                    ),
                )
            )
    return Plan(ask, calls, histories=cls, rows_role=rows_role)


def _archive_bound(cls: str, found: list[dict], ask: dict, tables: Tables) -> list:
    """The term bounding an archive read: the read site's clock term on the partition column, before the first pass."""
    attr = tables.since.get(cls)
    if attr is None:
        return []
    first = min(datetime.datetime.strptime(t, PASS_TIME) for t in ask["at"])
    back = max(
        (
            -v["offset_seconds"]
            for r in found
            for c in _compares(t["tree"] for t in r.get("condition_trees") or [])
            if c["subject"] == f"{cls}.{attr}" and c["op"] in (">", ">=")
            for v in c["values"]
            if v.get("kind") == "clock" and (v.get("offset_seconds") or 0) < 0
        ),
        default=None,
    )
    reach = (
        datetime.timedelta(seconds=back) + CLOCK_SLACK
        if back is not None
        else NO_CREATION
    )
    return [since(attr, (first - reach).replace(microsecond=0))]


def _compares(trees) -> list[dict]:
    out, stack = [], list(trees)
    while stack:
        node = stack.pop()
        if node.get("node") == "compare":
            out.append(node)
        stack.extend(node.get("items") or [])
    return out


def _sql(ask: dict) -> Plan:
    return Plan(
        ask,
        why="a window over the history no call reads: listed for a hand (hand-r<n>.json in the records directory)",
    )


async def answer(line: dict, p: Plan, parts: Parts, tables: Tables) -> dict:
    """Make *p*'s calls for the ask *line*: its db_read queries in one batch, the other parts, then the histories
    of the rows found; every request and response in order, as a fetch record's steps."""
    if p.why:
        return {"ask": line, "why": p.why, "steps": []}
    rec = Record(line)
    batch = [(c.role, c.request) for c in p.calls if c.part == "db_read"]
    got = await rec.read(parts, batch) if batch else []
    called = [
        (c, await rec.call(c.part, CALLS[c.part](parts, c.request), c.request))
        for c in p.calls
        if c.part != "db_read"
    ]
    if p.histories:
        role, key = p.rows_role, tables.key[p.histories]
        found = [
            r
            for (r_role, _), a in zip(batch, got, strict=True)
            if r_role == role
            for _, r in rows_of(a) or []
        ] + [
            r
            for c, response in called
            if c.role == role
            and c.part in KEY_ANSWER
            and (r := KEY_ANSWER[c.part](response, c.request)) is not None
        ]
        if ids := sorted({int(value_of(r, key)) for r in found}):
            created = made(tables.created[p.histories], found)
            await HISTORY[p.histories](ids, created, rec, parts)
    return {"ask": line, "why": "", "steps": rec.steps}


def next_round(directory: Path) -> int:
    """One past the last round of rows answers in *directory*."""
    done = [
        int(m.group(1))
        for f in directory.glob("rows-r*.json")
        if (m := ROUND.match(f.name))
    ]
    return max(done, default=0) + 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m bamboo.scripts.panda.fieldtest.rowsask",
        description=__doc__.split("\n\n")[0],
    )
    ap.add_argument("asks", type=Path, help="the rows-asks file the querier wrote")
    ap.add_argument(
        "--dump", required=True, type=Path, help="the map's dump of the case's pair"
    )
    ap.add_argument("--out", type=Path, help="where to write the plans as JSON")
    ap.add_argument(
        "--records",
        type=Path,
        help="ask PanDA, and write the answers to rows-r<n>.json in this directory",
    )
    ap.add_argument(
        "--via",
        choices=VIAS,
        default=DEFAULT_VIA,
        help="how rows are read: endpoint (the endpoints that replace db_read where one reads the rows asked; the "
        "default) or db_read (the async daemon, for a server without them)",
    )
    a = ap.parse_args(argv)
    dump = json.loads(a.dump.read_text())
    tables, sites = Tables.of(dump), Sites(dump)
    lines = [json.loads(x) for x in a.asks.read_text().splitlines() if x.strip()]
    plans = [plan(line, tables, sites, a.via) for line in lines]
    for p in plans:
        what = f"{len(p.calls)} call(s)" + (
            f", then {p.histories}'s histories" if p.histories else ""
        )
        print(f"{p.ask['kind']}: {what if not p.why else 'none: ' + p.why}")
    if a.out:
        a.out.write_text(
            json.dumps([asdict(p) for p in plans], indent=1, default=str) + "\n"
        )
    if a.records is None or not plans:
        return 0
    started = datetime.datetime.now(datetime.timezone.utc)
    answers = asyncio.run(_answer_all(lines, plans, Parts(), tables))
    a.records.mkdir(parents=True, exist_ok=True)
    path = a.records / f"rows-r{next_round(a.records)}.json"
    doc = {
        "about": "The answers to the querier's rows asks, every request and response as it came (bamboo.scripts.panda.fieldtest.rowsask).",
        "asks": {"path": str(a.asks), "sha256": sha256(a.asks)},
        "tables_from": {"dump": str(a.dump), "sha256": sha256(a.dump)},
        "server": os.environ.get("PANDA_API_URL_SSL", ""),
        "started": started.isoformat(),
        "finished": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "answers": answers,
    }
    path.write_text(json.dumps(doc, indent=1) + "\n")
    print(
        f"{path}: {sum(not x['why'] for x in answers)} of {len(answers)} ask(s) asked"
    )
    return 0


async def _answer_all(
    lines: list[dict], plans: list[Plan], parts: Parts, tables: Tables
) -> list[dict]:
    return [
        await answer(line, p, parts, tables)
        for line, p in zip(lines, plans, strict=True)
    ]


if __name__ == "__main__":
    sys.exit(main())
