"""Lay a fetch record out as the querier's rows file.

No API is called: the same record, pin and dump give the same file.  The
sections are the ones the querier reads (``rerun.py``): ``classes``,
``spellings``, ``rows``, ``statuslog``, and for a job ``files`` and
``collections``, for a task ``job_rows``, ``job_statuslog`` and ``counts``.

**The same rows by link.**  ``linked`` holds the rows each link of the fetch
(:data:`~bamboo.scripts.panda.fieldtest.fetch.LINKS`) named from the key's
rows, one entry per link and table: its ``class``, ``via`` (``{"from":
"JediTaskSpec.jediTaskID", "to": "JobSpec.jediTaskID"}``), the values
``asked`` by it, how the rows were read, and the ``rows`` -- those whose
``to`` column holds an asked value, or every row of the answer where the rows
do not carry that column (a task's jobs, read by the task's key).
``statuslogs`` holds each class's status histories by the key of the row each
belongs to.  Both say again what the sections above say, so a reader can move
to them; the sections above stay as they were until it has.

**Columns.**  The querier reads a row by the spec class's attribute
(``PandaID``, ``specialHandling``).  The class's column list in PanDA's source
-- ``_attributes``, or ``attributes`` in JEDI's specs -- gives the columns and
their spelling (:func:`column_spellings`), read from the pin with ``ast``.  A
key's row and a job's files are kept to those columns: what else an endpoint
adds (a task's progress, its parameters) is left out and named in ``about``,
and a task's ``splitRule`` is the column as stored (``split_rule_raw``), not
the server's decoding of it.  The task's job rows keep R's lower-case names,
as the querier reads them.

**When.**  The endpoints that read a row by its key say no time of the
database's, so each value's source says when it was asked, on this machine's
clock, and its ``as_of`` is that moment cut down to its second: the row was
read after it.  R's and H's ``as_of`` are the database's.  Every ``as_of``
names its clock beside it, ``clock`` :data:`CLIENT` or :data:`DATABASE`, since
the two differ and a value read by key is not read at a history's ``as_of``.
A key's row is laid out whole, every column with the moment it
was read: a column may have been written between the moment a step of the
querier asks about and that one, and only the querier knows the step's moment
(``held``).

**Times.**  The querier compares a history's times with log stamps as text
(``rerun.py`` S5, ``time_extended < when``), so ``time`` is
``%Y-%m-%d %H:%M:%S`` and ``time_extended`` always carries six digits of
microseconds.  All are UTC, as the database gives them.  A history also says
the span it is whole for, ``as_of`` and ``retained_from`` (:func:`bounds_of`),
on the clock of its own times, so the querier can compare them as text too.
A task's ``counts`` and ``job_rows`` say the moment R read them as ``as_of``,
cut down to its second: R runs a batch's queries in one read-only transaction
and gives its moment, on the database's clock in UTC (``db_read_handlers``).

**What is left out.**  A section whose answer is not complete -- rejected,
failed, past its deadline, truncated -- is left out, and ``about`` says why: a
cut set must not read as the whole.  The querier then stops where it needs the
section.  The entity names (``job``, ``file``, ``task``) and the spellings of a
variable holding a row (:data:`SPELLINGS`) are the querier's conventions, not
facts of a case.
"""

from __future__ import annotations

import ast
import datetime
import json
import re
from pathlib import Path
from typing import Any

from bamboo.scripts.panda.fieldtest.fetch import (
    FILES,
    Tables,
    held_rows,
    job_of,
    links_from,
    rows_of,
    task_of,
    values_by,
    when,
)

ENTITY = {"JobSpec": "job", "FileSpec": "file", "JediTaskSpec": "task"}
#: how PanDA's code spells a variable holding the row (the querier's stand-in evaluator binds by these)
SPELLINGS = {"job": ["job", "self.job"], "file": ["file"], "task": []}
#: the querier's (rerun.py FINAL_STATES): a job leaves its active tables at these
FINAL_STATES = ("finished", "failed", "cancelled", "closed")
#: the clock an ``as_of`` is on: the machine that called the API, or the database's
CLIENT = "client"
DATABASE = "database"
#: the columns of a job's history row besides its status, times and host, spelt as the job's
#: attributes they copy: recordStatusChange copies them from the job (base_module.py:589-591)
JOB_HISTORY_COLUMNS = ("prodSourceLabel", "cloud", "computingSite")
#: the endpoint a key's row comes from
ENDPOINT = {
    "JobSpec": "job/get_description_incl_archive",
    "JediTaskSpec": "task/get_detailed_info",
}
#: where task/get_detailed_info holds the splitRule column as stored
SPLIT_RULE_RAW = "split_rule_raw"
DECLARED = re.compile(r"^(?P<cls>\w+) \((?P<path>[^)]+)\)$")


def column_spellings(
    pin: Path, dump: Path, classes: tuple[str, ...] = tuple(ENTITY)
) -> dict[str, dict[str, str]]:
    """class -> {lower-case column: the class's spelling}, from the column list in each class's source.

    The dump's ``declaration_yields`` says which file declares a class.
    """
    declared = {}
    for name in json.loads(dump.read_text())["declaration_yields"]:
        if m := DECLARED.match(name):
            declared[m["cls"]] = m["path"]
    out = {}
    for cls in classes:
        tree = ast.parse((pin / declared[cls]).read_text())
        body = next(
            n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == cls
        ).body
        names = next(
            ast.literal_eval(n.value)
            for n in body
            if isinstance(n, ast.Assign)
            and any(
                getattr(t, "id", "") in ("_attributes", "attributes") for t in n.targets
            )
        )
        out[cls] = {c.lower(): c for c in names}
    return out


def text(value: Any) -> Any:
    """A value as the rows file holds it: a time as ``%Y-%m-%d %H:%M:%S`` (with microseconds when it has them)."""
    t = when(value)
    if t is None:
        return value
    return f"{t:%Y-%m-%d %H:%M:%S.%f}" if t.microsecond else f"{t:%Y-%m-%d %H:%M:%S}"


def down_to_second(value: str | None) -> str | None:
    """A time as :func:`text` writes it, cut down to its second: a moment claims no second it may not hold."""
    return value[:19] if value else None


def stamp(t: datetime.datetime | None) -> str:
    return f"{t:%Y-%m-%d %H:%M:%S.%f}" if t else ""


class Laid:
    """R's answers by role and why any is incomplete, and every other part's response."""

    def __init__(self, record: dict) -> None:
        self.rows: dict[str, list[tuple[str, dict]]] = {}
        self.incomplete: dict[str, list[str]] = {}
        self.as_of: dict[str, str] = {}
        self.responses: dict[str, dict] = {}
        self.requests: dict[str, dict] = {}
        for step in record["steps"]:
            if step["part"] != "db_read":
                self.responses[step["part"]] = step["response"]
                self.requests[step["part"]] = step["request"]
                continue
            response = step["response"]
            data = response.get("data") if response.get("success") else None
            answers = (
                data["answers"]
                if isinstance(data, dict) and "answers" in data
                else [None] * len(step["roles"])
            )
            for role, answer in zip(step["roles"], answers, strict=True):
                self.rows.setdefault(role, [])
                if data is not None:
                    self.as_of[role] = text(data["as_of"])
                got = rows_of(answer)
                if got is None:
                    why = (
                        (answer or {}).get("rule")
                        or (answer or {}).get("code")
                        or (answer or {}).get("status")
                        or response.get("message")
                        or "no answer"
                    )
                    self.incomplete.setdefault(role, []).append(str(why))
                elif answer.get("truncated"):
                    self.incomplete.setdefault(role, []).append("truncated")
                    self.rows[role] += got
                else:
                    self.rows[role] += got

    def complete(self, role: str) -> bool:
        return role in self.rows and role not in self.incomplete


def columns(answer: dict, spelling: dict[str, str], other: set[str]) -> dict:
    """The answer's values of the class's columns, spelled as the class spells them; any other key goes to *other*."""
    out = {}
    for key, value in answer.items():
        if (col := spelling.get(key.lower())) is None:
            other.add(key)
        else:
            out[col] = text(value)
    return out


def key_row(laid: Laid, cls: str, key: str) -> tuple[dict | None, dict, str]:
    """The answer for the key's row, as it came, and when its values were read (``source``, ``as_of``, ``clock``); None and why when there is none."""
    response = laid.responses.get("jobs" if cls == "JobSpec" else "task")
    found, why = job_of(response, key) if cls == "JobSpec" else task_of(response)
    if found is None:
        return None, {}, why
    if cls == "JediTaskSpec":
        found = {k: v for k, v in found.items() if k != SPLIT_RULE_RAW} | {
            "splitRule": found.get(SPLIT_RULE_RAW)
        }
    source = f"{ENDPOINT[cls]}, asked at {response['asked_at']} on this machine's clock (UTC)"
    read = {
        "source": source,
        "as_of": down_to_second(response["asked_at"]),
        "clock": CLIENT,
    }
    return found, read, ""


def states_of(response: dict, kind: str) -> tuple[list[dict] | None, str]:
    """The history's rows as the querier reads them, and the source; None when the call did not succeed."""
    if not response or not response.get("success"):
        data = (response or {}).get("data") or {}
        return None, f"{response.get('message') if response else 'not asked'} {data}"
    out = []
    for r in response["rows"] if "rows" in response else response["data"]["rows"]:
        if kind == "job":
            t, tx = (
                when(r["modificationTime"]),
                when(r["modiftime_extended"]) or when(r["modificationTime"]),
            )
            out.append(
                {
                    "pandaid": r["PandaID"],
                    "status": r["jobStatus"],
                    "time": f"{t:%Y-%m-%d %H:%M:%S}",
                    "time_extended": stamp(tx),
                    "host": r["modificationHost"],
                    **{c: r[c] for c in JOB_HISTORY_COLUMNS},
                }
            )
        else:
            t = when(r["modificationTime"])
            out.append(
                {
                    "status": r["status"],
                    "time": f"{t:%Y-%m-%d %H:%M:%S}",
                    "time_extended": stamp(t),
                    "attemptnr": r["attemptNr"],
                    "reason": r["reason"],
                    "host": r["modificationHost"],
                }
            )
    data = response if "as_of" in response else response["data"]
    source = (
        f"H as of {text(data['as_of'])}, kept from {text(data.get('retained_from'))}"
    )
    if data.get("since"):
        source += f", since {text(data['since'])}"
    return out, source


def bounds_of(response: dict) -> dict[str, str | None]:
    """The span this answer is whole for: every change from ``retained_from`` to ``as_of`` is in it.

    Not the server's two values as they are: the answer holds only what was
    asked, so ``retained_from`` is raised to the asked ``since`` and ``as_of``
    lowered to just before the asked ``until``.  All are on the clock of the
    history's own times, the database's, in UTC: the server sets every Oracle
    session to UTC (``WrappedCursor.initialize``), a job's history row takes
    ``CURRENT_DATE`` and a task's ``CURRENT_TIMESTAMP``, ``as_of`` is
    ``SYS_EXTRACT_UTC(CURRENT_TIMESTAMP)`` taken before the read, and
    ``retained_from`` is a bound of the partitions on that same column.  Each is
    cut to seconds the safe way, ``as_of`` down and ``retained_from`` up, so
    neither claims a second the answer may not hold.  ``retained_from`` is None
    when the server cannot tell.
    """
    data = response if "as_of" in response else response["data"]
    upper = when(data["as_of"])
    until = when(data.get("until"))
    if until is not None:
        upper = min(upper, until - datetime.timedelta(microseconds=1))
    lower = when(data.get("retained_from"))
    since = when(data.get("since"))
    if lower is not None and since is not None:
        lower = max(lower, since)
    if lower is not None and lower.microsecond:
        lower = lower.replace(microsecond=0) + datetime.timedelta(seconds=1)
    return {
        "as_of": f"{upper:%Y-%m-%d %H:%M:%S}",
        "clock": DATABASE,
        "retained_from": f"{lower:%Y-%m-%d %H:%M:%S}" if lower else None,
    }


def build(record: dict, spellings: dict[str, dict[str, str]], tables: Tables) -> dict:
    q = record["question"]
    cls = record["key_class"]
    entity = ENTITY[cls]
    laid = Laid(record)
    left_out: list[str] = []
    other: set[str] = set()
    out: dict[str, Any] = {
        "about": (
            f"Built by bamboo.scripts.panda.fieldtest.rows from a fetch record ({record['server']}, {record['started']}) "
            f"for {cls} {q['key']}.  Every value is as it was at the time its source names."
        ),
        "classes": {"about": "The class each row is.", entity: cls},
        "spellings": {
            "about": "How PanDA's code spells a variable holding the row (the querier's convention).",
            **{
                e: SPELLINGS[e]
                for e in ([entity, "file"] if cls == "JobSpec" else [entity])
            },
        },
    }

    # the history of the key
    history_part = "job_history" if cls == "JobSpec" else "task_history"
    states, history_source = states_of(
        laid.responses.get(history_part), "job" if cls == "JobSpec" else "task"
    )
    if states is None:
        left_out.append(f"statuslog: {history_source}")
    else:
        out["statuslog"] = {
            "source": history_source,
            **bounds_of(laid.responses[history_part]),
            "states": [{k: v for k, v in s.items() if k != "pandaid"} for s in states],
        }

    raw, read, why = key_row(laid, cls, q["key"])
    if raw is None:
        left_out.append(
            f"rows.{entity}{', files' if cls == 'JobSpec' else ''}: {tables.key[cls]}={q['key']}: {why}"
        )
    else:
        row = columns(
            {k: v for k, v in raw.items() if k != FILES}, spellings[cls], other
        )
        out["rows"] = {entity: {c: {"value": v, **read} for c, v in row.items()}}

    if cls == "JobSpec":
        if raw is not None:
            _job_files(out, raw, read, spellings, other)
    else:
        _task_sections(out, laid, left_out)
    out["linked"] = linked(laid, cls, int(q["key"]), raw, read, spellings, tables)
    out["statuslogs"] = statuslogs(laid, int(q["key"]), tables)

    if other:
        out["about"] += (
            f"  {len(other)} key(s) of the answers that are not columns of their class, left out: {sorted(other)}."
        )
    if left_out:
        out["about"] += "  Left out: " + "; ".join(left_out) + "."
    return out


def _job_files(
    out: dict,
    raw: dict,
    read: dict,
    spellings: dict[str, dict[str, str]],
    other: set[str],
) -> None:
    out["files"] = [
        {
            c: {"value": v, **read}
            for c, v in columns(f, spellings["FileSpec"], other).items()
        }
        for f in raw.get(FILES) or []
    ]
    out["collections"] = {
        "about": "JobSpec.Files holds the job's FileSpec rows (PanDA's schema).",
        "job.Files": {"rows": "files", "entity": "file"},
    }


#: how an aggregate's column comes back from R, and what L3 reads it as (its SQL's ``COUNT(*) AS n``,
#: ``MIN(pandaid) AS min_pandaid``)
AGGREGATE = re.compile(r"^(?P<function>count|min|max)\((?P<column>[^)]*)\)$")


def _aggregate_name(column: str) -> str:
    m = AGGREGATE.match(column)
    if m is None:
        return column
    return "n" if m["function"] == "count" else f"{m['function']}_{m['column']}"


def answered(
    docs: list[dict], key_class: str, key: str, spellings: dict[str, dict[str, str]]
) -> tuple[dict, list[str]]:
    """The sections the answers to the querier's rows asks give, round by round (a later round's over an
    earlier's), and what is left out and why.

    A statement's rows, their histories and counts are the sections a task's fetch gives (``job_rows``,
    ``job_statuslog``, ``counts``); the key's job's ``Files``, its ``files`` and ``collections``; an sql
    ask's groups, ``answers[<name>]``, with R's aggregate columns named as L3's SQL names them.  The rows of
    another row's collection have no section yet, and are left out.
    """
    out: dict[str, Any] = {}
    left_out: list[str] = []
    for doc in docs:
        for a in doc["answers"]:
            ask = a["ask"]["rows"]
            name = f"{ask['kind']} {ask.get('name') or ask.get('method') or ask.get('attr')}"
            if a["why"]:
                left_out.append(f"{name}: {a['why']}")
                continue
            laid = Laid({"steps": a["steps"]})
            if ask["kind"] == "statement":
                _task_sections(out, laid, left_out)
            elif ask["kind"] == "collection":
                row = ask["row"]
                mine = row["entity"] == key_class == "JobSpec" and [
                    str(v) for v in row["key"].values()
                ] == [str(key)]
                raw, read, why = (
                    key_row(laid, key_class, key) if mine else (None, {}, "")
                )
                if raw is not None:
                    _job_files(out, raw, read, spellings, set())
                else:
                    left_out.append(
                        f"{name}: {why or 'no section of the rows file holds the rows of ' + row['entity'] + '.' + ask['attr'] + ' yet'}"
                    )
            elif ask["kind"] == "sql":
                if not laid.complete("sql"):
                    left_out.append(
                        f"{name}: {laid.incomplete.get('sql', 'not asked')}"
                    )
                    continue
                out.setdefault("answers", {})[ask["name"]] = {
                    "source": f"R as of {laid.as_of.get('sql')}, by db_read's form of the ask",
                    "as_of": down_to_second(laid.as_of.get("sql")),
                    "clock": DATABASE,
                    "rows": [
                        {_aggregate_name(c): text(v) for c, v in r.items()}
                        for _, r in laid.rows["sql"]
                    ],
                }
    return out, left_out


def _task_sections(out: dict, laid: Laid, left_out: list[str]) -> None:
    # the counts are read apart from the rows, so neither waits on the other
    if laid.complete("job_counts"):
        out["counts"] = {
            "source": f"R as of {laid.as_of.get('job_counts')}, every job of the task by table and jobStatus",
            "as_of": down_to_second(laid.as_of.get("job_counts")),
            "clock": DATABASE,
            "rows": [
                {"table": t, "jobStatus": r["jobstatus"], "n": r["count(*)"]}
                for t, r in laid.rows["job_counts"]
            ],
        }
    else:
        left_out.append(f"counts: {laid.incomplete.get('job_counts', 'not asked')}")
    if not laid.complete("job_rows"):
        left_out.append(
            f"job_rows, job_statuslog: {laid.incomplete.get('job_rows', 'not asked')}"
        )
        return
    states, source = (
        states_of(laid.responses.get("job_history"), "job")
        if laid.rows["job_rows"]
        else ([], "no jobs")
    )
    if states is None:
        left_out.append(f"job_rows, job_statuslog: {source}")
        return
    by_pandaid: dict[str, list[dict]] = {}
    for s in states:
        by_pandaid.setdefault(str(s["pandaid"]), []).append(
            {k: v for k, v in s.items() if k != "pandaid"}
        )
    rows = []
    for table, r in laid.rows["job_rows"]:
        history = by_pandaid.get(str(r["pandaid"]), [])
        final = next((h["time"] for h in history if h["status"] in FINAL_STATES), None)
        rows.append(
            {"tbl": table, **{c: text(v) for c, v in r.items()}, "t_final": final}
        )
    out["job_rows"] = {
        "source": f"R as of {laid.as_of.get('job_rows')}, the jobs of the task in {sorted({t for t, _ in laid.rows['job_rows']})}; "
        "a PandaID in two tables is two rows",
        "as_of": down_to_second(laid.as_of.get("job_rows")),
        "clock": DATABASE,
        "rows": rows,
    }
    out["job_statuslog"] = {
        "source": source,
        **(bounds_of(laid.responses["job_history"]) if laid.rows["job_rows"] else {}),
        "by_pandaid": by_pandaid,
    }


Answer = list[tuple[str | None, dict, set | None, list[dict]]]


def _task_jobs_read(laid: Laid, spellings: dict[str, dict[str, str]]) -> Answer | None:
    if not laid.complete("job_rows"):
        return None
    read = {
        "source": f"R as of {laid.as_of.get('job_rows')}",
        "as_of": down_to_second(laid.as_of.get("job_rows")),
        "clock": DATABASE,
    }
    by_table: dict[str, list[dict]] = {}
    for table, r in laid.rows["job_rows"]:
        by_table.setdefault(table, []).append(columns(r, spellings["JobSpec"], set()))
    return [(t, read, None, rows) for t, rows in by_table.items()]


#: where the rows a link names are in a fetch record, by where it leads: for each table, the table, how
#: they were read, the values the call was asked with (None where it was asked by the key alone) and the
#: rows as their class spells them; None where that answer is not complete
LINK_ANSWERS = {
    "JobSpec.jediTaskID": _task_jobs_read,
}
#: the part answering a class's status histories, and how :func:`states_of` reads it
HISTORIES = {
    "JobSpec": ("job_history", "job"),
    "JediTaskSpec": ("task_history", "task"),
}


def linked(
    laid: Laid,
    key_class: str,
    key: int,
    raw: dict | None,
    read: dict,
    spellings: dict[str, dict[str, str]],
    tables: Tables,
) -> list[dict]:
    """The rows each link from the key's rows named: one entry per link and table, with the values it asked.

    The rows a link holds in its source row's answer (a job's files) are all
    that link's; of an answer asked for several links, a link's are those
    whose ``to`` column holds one of its asked values.
    """
    held = held_rows(key_class, raw)
    out = []
    for cls, got in held.items():
        for link in links_from(cls):
            values = sorted(values_by(link, got, key_class, key, tables))
            if not values or (link.held and not got):
                continue
            if link.held:
                spelling = spellings[link.dst_class]
                answers = [
                    (
                        None,
                        read,
                        None,
                        [columns(r, spelling, set()) for r in held[link.dst_class]],
                    )
                ]
            else:
                answers = LINK_ANSWERS[link.dst](laid, spellings)
                if answers is None:
                    continue
            forms, to = set(values), link.dst.partition(".")[2]
            for table, how, sent, rows in answers:
                if not (asked := sorted(forms if sent is None else forms & sent)):
                    continue
                named = {str(v) for v in asked}
                out.append(
                    {
                        "class": link.dst_class,
                        "via": link.via(),
                        "asked": asked,
                        **({"table": table} if table else {}),
                        **how,
                        "rows": [
                            r
                            for r in rows
                            if link.held or to not in r or str(r[to]) in named
                        ],
                    }
                )
    return out


def statuslogs(laid: Laid, key: int, tables: Tables) -> dict:
    """Each class's status histories that were answered, by the key of the row each belongs to; every key asked is there."""
    out = {}
    for cls, (part, kind) in HISTORIES.items():
        states, source = states_of(laid.responses.get(part), kind)
        if states is None:
            continue
        request = laid.requests[part]
        by_key: dict[str, list[dict]] = {
            str(k): [] for k in request.get("job_ids") or [request.get("task_id")]
        }
        for s in states:
            by_key.setdefault(str(s.get("pandaid", key)), []).append(
                {k: v for k, v in s.items() if k != "pandaid"}
            )
        out[cls] = {
            "key": f"{cls}.{tables.key[cls]}",
            "source": source,
            **bounds_of(laid.responses[part]),
            "by_key": by_key,
        }
    return out
