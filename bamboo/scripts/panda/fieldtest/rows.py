"""Lay a fetch record out as the querier's rows file.

No API is called: the same record, pin and dump give the same file.  The
sections are the ones the querier reads (``rerun.py``): ``classes``,
``spellings``, ``rows`` and ``after``, ``statuslog``, and for a job ``files`` and
``collections``, for a task ``job_rows``, ``job_statuslog`` and ``counts``.

**Spelling.**  R answers with lower-case names; the querier reads a row by the
spec class's attribute (``PandaID``, ``specialHandling``).  The class's column
list in PanDA's source -- ``_attributes``, or ``attributes`` in JEDI's specs --
gives the spelling (:func:`column_spellings`), read from the pin with ``ast``.
A column the list does not have is kept in lower case and counted in
``about``.

**Times.**  The querier compares a history's times with log stamps as text
(``rerun.py`` S5, ``time_extended < when``), so ``time`` is
``%Y-%m-%d %H:%M:%S`` and ``time_extended`` always carries six digits of
microseconds.  All are UTC, as the database gives them.  A history also says
the span it is whole for, ``as_of`` and ``retained_from`` (:func:`bounds_of`),
on the clock of its own times, so the querier can compare them as text too.

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

from bamboo.scripts.panda.fieldtest import after as A
from bamboo.scripts.panda.fieldtest.fetch import (
    ARCHIVE,
    KEY,
    LIVE,
    rows_of,
    when,
)

ENTITY = {"JobSpec": "job", "FileSpec": "file", "JediTaskSpec": "task"}
#: how PanDA's code spells a variable holding the row (the querier's stand-in evaluator binds by these)
SPELLINGS = {"job": ["job", "self.job"], "file": ["file"], "task": []}
#: the querier's (rerun.py FINAL_STATES): a job leaves its active tables at these
FINAL_STATES = ("finished", "failed", "cancelled", "closed")
#: the tables a class's row is taken from first: the live ones before the archive
ORDER = {cls: (*LIVE[cls], *([ARCHIVE[cls]] if cls in ARCHIVE else [])) for cls in LIVE}
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


def stamp(t: datetime.datetime | None) -> str:
    return f"{t:%Y-%m-%d %H:%M:%S.%f}" if t else ""


class Laid:
    """The record's answers by role, and why any is incomplete."""

    def __init__(self, record: dict) -> None:
        self.rows: dict[str, list[tuple[str, dict]]] = {}
        self.incomplete: dict[str, list[str]] = {}
        self.as_of: dict[str, str] = {}
        self.histories: dict[str, dict] = {}
        for step in record["steps"]:
            if step["part"] != "db_read":
                self.histories[step["part"]] = step["response"]
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


def spelled(row: dict, spelling: dict[str, str], unknown: set[str]) -> dict:
    out = {}
    for col, value in row.items():
        if col not in spelling:
            unknown.add(col)
        out[spelling.get(col, col)] = text(value)
    return out


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
                    "site": r["computingSite"],
                    "host": r["modificationHost"],
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
        "retained_from": f"{lower:%Y-%m-%d %H:%M:%S}" if lower else None,
    }


def build(record: dict, spellings: dict[str, dict[str, str]]) -> dict:
    q = record["question"]
    cls = record["key_class"]
    entity = ENTITY[cls]
    laid = Laid(record)
    left_out: list[str] = []
    unknown: set[str] = set()
    out: dict[str, Any] = {
        "about": (
            f"Built by bamboo.scripts.panda.fieldtest.rows from a fetch record ({record['server']}, {record['started']}) "
            f"for {cls} {q['key']}.  Every value is the row as it was at the answer's as_of."
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

    # the history of the key, first: the split of the row needs it
    history_part = "job_history" if cls == "JobSpec" else "task_history"
    states, history_source = states_of(
        laid.histories.get(history_part), "job" if cls == "JobSpec" else "task"
    )
    if states is None:
        left_out.append(f"statuslog: {history_source}")
    else:
        out["statuslog"] = {
            "source": history_source,
            **bounds_of(laid.histories[history_part]),
            "states": [{k: v for k, v in s.items() if k != "pandaid"} for s in states],
        }

    # the key's row, from the live-most table that has it
    found = sorted(
        laid.rows.get("row", []),
        key=lambda tr: ORDER[cls].index(tr[0]) if tr[0] in ORDER[cls] else 99,
    )
    if not laid.complete("row") and not found:
        left_out.append(f"rows.{entity}: {laid.incomplete.get('row')}")
    elif not found:
        left_out.append(
            f"rows.{entity}: no row of {KEY[cls]}={q['key']} in {list(ORDER[cls])}"
        )
    else:
        table, raw = found[0]
        row = spelled(raw, spellings[cls], unknown)
        source = f"R as of {laid.as_of.get('row')}, {table}"
        key = KEY[cls]
        kept, later, why = A.split(
            row,
            key,
            q["kind"],
            q["observed"],
            states or [],
            when(raw.get("modificationtime")),
        )
        out["rows"] = {
            entity: {c: {"value": v, "source": source} for c, v in kept.items()}
        }
        if later:
            out["after"] = {"source": f"{source}; {why}", **later}

    if cls == "JobSpec":
        if laid.complete("files"):
            src = f"R as of {laid.as_of.get('files')}"
            out["files"] = [
                {
                    c: {"value": v, "source": f"{src}, {t}"}
                    for c, v in spelled(r, spellings["FileSpec"], unknown).items()
                }
                for t, r in laid.rows["files"]
            ]
            out["collections"] = {
                "about": "JobSpec.Files holds the job's FileSpec rows (PanDA's schema).",
                "job.Files": {"rows": "files", "entity": "file"},
            }
        else:
            left_out.append(f"files: {laid.incomplete.get('files', 'not asked')}")
    else:
        _task_sections(out, laid, left_out)

    if unknown:
        out["about"] += (
            f"  {len(unknown)} column(s) not in the class's column list, kept in lower case: {sorted(unknown)}."
        )
    if left_out:
        out["about"] += "  Left out: " + "; ".join(left_out) + "."
    return out


def _task_sections(out: dict, laid: Laid, left_out: list[str]) -> None:
    if not laid.complete("job_rows"):
        left_out.append(
            f"job_rows, job_statuslog: {laid.incomplete.get('job_rows', 'not asked')}"
        )
        return
    states, source = (
        states_of(laid.histories.get("job_history"), "job")
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
        "rows": rows,
    }
    out["job_statuslog"] = {
        "source": source,
        **(bounds_of(laid.histories["job_history"]) if laid.rows["job_rows"] else {}),
        "by_pandaid": by_pandaid,
    }
    if laid.complete("job_counts"):
        out["counts"] = {
            "source": f"R as of {laid.as_of.get('job_counts')}, every job of the task by table and jobStatus",
            "rows": [
                {"table": t, "jobStatus": r["jobstatus"], "n": r["count(*)"]}
                for t, r in laid.rows["job_counts"]
            ],
        }
    else:
        left_out.append(f"counts: {laid.incomplete.get('job_counts', 'not asked')}")
