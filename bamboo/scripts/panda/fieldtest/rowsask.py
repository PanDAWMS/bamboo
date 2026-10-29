"""Plan the answers to the querier's rows asks: for each ask, the calls that read the rows it names, or why none can.

    python -m bamboo.scripts.panda.fieldtest.rowsask <rows-asks.jsonl> --dump <map dump> [--out <plan.json>]
        [--records <dir>]

The querier writes a rows ask where a step needs rows its rows file lacks (``--rows-asks``, one JSON line
each: ``{"rows": {...}, "step", "why"}``; stage 1 of the rows ask).  This says, by rule and not by case, how
each is answered; with ``--records`` it asks PanDA (run it where PanDA's credentials load, as the fetch) and
writes the round's answers to ``rows-r<n>.json`` there, ``n`` one past the directory's last
(:func:`answer`; :func:`~bamboo.scripts.panda.fieldtest.rows.answered` lays them out):

- ``collection`` -- a row's attribute holding rows (``{"row": {"entity", "key"}, "attr", "entity"}``).  Rows
  the row's own answer holds (:data:`~bamboo.scripts.panda.fieldtest.fetch.LINKS` ``held``: a job's
  ``Files``) are read by the call reading the row by its key.  Else, where the ask names the element class
  and its tables name a row by the row's key among others (a task's datasets: ``JEDI_Datasets`` by
  ``datasetID`` and ``jediTaskID``), by ``db_read`` on those tables where that column holds the key.
- ``statement`` -- the rows a call's statements read, per pass (``{"method", "site", "tables", "binds",
  "at"}``).  The read sites at ``site`` name the columns the statements select by (the dump's
  ``read_sites`` ``keys``: ``JobSpec.jediTaskID``); the call's argument of each one's name gives its values
  (a value the run did not show, ``"unknown"``, is left out).  ``db_read`` reads the ask's tables -- the
  ones the call's constants leave -- by them: the rows, and their number by status; the rows' histories
  follow once the rows are in.  An archive table is bounded by the read site's own clock term on the
  partition column (``modificationTime>(CURRENT_DATE-30)``) before the first pass, else by
  :data:`~bamboo.scripts.panda.fieldtest.fetch.NO_CREATION`.
- ``sql`` -- an answer L3 lacks (``{"name", "sql", "binds", "db_read"?}``).  Read by ``db_read`` where the ask
  carries a ``db_read`` form whose ``where`` is ``[column, op, value]`` terms: the groups and, for each, its
  number and its least ``least``.  Else the SQL is listed for a hand.

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
    HISTORY,
    JOB_ROW_COLUMNS,
    KEY_REQUEST,
    NO_CREATION,
    Parts,
    Record,
    Tables,
    links_from,
    made,
    query,
    rows_of,
    since,
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
}
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
    why: str = ""


class Sites:
    """The dump's read sites by ``<file>:<line>``, as the querier names a statement's site."""

    def __init__(self, dump: dict) -> None:
        self.at = {
            f"{s['anchor']['file']}:{s['anchor']['line_start']}": s
            for s in dump.get("read_sites") or []
            if s.get("anchor")
        }


def plan(line: dict, tables: Tables, sites: Sites) -> Plan:
    """The plan for one line of a rows-asks file."""
    ask = line["rows"]
    kind = ask.get("kind")
    if kind == "collection":
        return _collection(ask, tables)
    if kind == "statement":
        return _statement(ask, tables, sites)
    if kind == "sql":
        return _sql(ask)
    return Plan(ask, why=f"no rule for an ask of kind {kind!r}")


def _collection(ask: dict, tables: Tables) -> Plan:
    cls, attr, element = ask["row"]["entity"], ask["attr"], ask.get("entity")
    key = ask["row"]["key"]
    for link in links_from(cls):
        if link.held == attr:
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
    if not set(key) <= set(tables.identity.get(element, ())):
        return Plan(
            ask,
            why=f"{element}'s tables name a row by {list(tables.identity.get(element, ()))}, not by {sorted(key)}",
        )
    where = [[c, "=", v] for c, v in sorted(key.items())]
    read = tables.live[element] + tables.archive.get(element, ())
    return Plan(ask, [Call("db_read", "collection", query(read, where))])


def _statement(ask: dict, tables: Tables, sites: Sites) -> Plan:
    found = [sites.at.get(s) for s in ask.get("site") or []]
    if not found or None in found:
        missing = [
            s for s, r in zip(ask.get("site") or [], found, strict=True) if r is None
        ]
        return Plan(ask, why=f"no read site in the dump at {missing or 'no site'}")
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
    return Plan(ask, calls, histories=cls)


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
    form = ask.get("db_read")
    if form is None:
        return Plan(ask, why="no db_read form (a window question): listed for a hand")
    if not isinstance(form.get("where"), list):
        return Plan(
            ask,
            why="the db_read form's where is SQL text, not [column, op, value] terms: listed for a hand",
        )
    aggregates = [["count", "*"]] + (
        [["min", form["least"]]] if form.get("least") else []
    )
    table = form["table"].rpartition(".")[2]
    request = query(
        (table,), form["where"], group_by=list(form["group_by"]), aggregates=aggregates
    )
    return Plan(ask, [Call("db_read", "sql", request)])


async def answer(line: dict, p: Plan, parts: Parts, tables: Tables) -> dict:
    """Make *p*'s calls for the ask *line*: its db_read queries in one batch, the other parts, then the histories
    of the rows found; every request and response in order, as a fetch record's steps."""
    if p.why:
        return {"ask": line, "why": p.why, "steps": []}
    rec = Record(line)
    batch = [(c.role, c.request) for c in p.calls if c.part == "db_read"]
    got = await rec.read(parts, batch) if batch else []
    for c in p.calls:
        if c.part != "db_read":
            await rec.call(c.part, CALLS[c.part](parts, c.request), c.request)
    if p.histories:
        role, key = COLUMNS[p.histories][2], tables.key[p.histories]
        found = [
            r
            for (r_role, _), a in zip(batch, got, strict=True)
            if r_role == role
            for _, r in rows_of(a) or []
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
    a = ap.parse_args(argv)
    dump = json.loads(a.dump.read_text())
    tables, sites = Tables.of(dump), Sites(dump)
    lines = [json.loads(x) for x in a.asks.read_text().splitlines() if x.strip()]
    plans = [plan(line, tables, sites) for line in lines]
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
