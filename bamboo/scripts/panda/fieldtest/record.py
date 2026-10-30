"""Answer a run's asks of PanDA's logs from production, one record per ask.

    python -m bamboo.scripts.panda.fieldtest.record --asks <asks.jsonl> --records <dir>
    python -m bamboo.scripts.panda.fieldtest.record --cases <file> --shared <dir>

Run it where PanDA's credentials load: the ``.env`` holding the OIDC token is
read only from /Users/tmaeno/bamboo or below.

**A round.**  The querier stops on the asks it holds no record of and writes
them to ``--asks``.  This answers every one and writes ``records-r<n>.jsonl``
into the case's directory, *n* one past the last round there; the querier
reads the directory with ``--records``.  A round is written once, when every
file it read is settled; a round that cannot settle (a day file still being
compressed after ``--max-wait``) writes nothing and is run again.  Every API
answer is kept under ``answers-r<n>-<time>/`` before it is read.  The formats
are bamboo-e1's and bamboo-8f's (analyzer-handover/stage2-record-format.md,
live-harness/notes/to-querier-15.md and -16.md); the reader is
``bamboo.querier.recorded``.

**Several cases.**  ``--cases`` names, a line each, a case's asks file and its
directory, and one round answers them all: one list of the logs, and one grep
per machine and file with the strings of every case's asks reading it (31
cases asked panda-DBProxy.log on every server machine on 2026-10-29).  Each
case gets its own next round, ``records-r<n>.jsonl`` as one case's round
writes it; the API answers are kept once under ``--shared``, and each case's
``answers-r<n>-<time>/shared.json`` says where.  A later round reads the
rotated files those answers hold again, as it reads its own.

**Machines.**  An ask names its log, and the machine when the run wanted is
one machine's; its service is never its to say ("unsettled").  Both services
list the logs asked by name, and a machine whose list names a log holds it.  A
list is known (``listed``) only when it is done, not truncated, and every name
in it is one asked: an old processor ignores ``log_names`` and lists all.  An
ask with a machine asks that machine alone: ``not listed`` when no list has it,
``no such log`` when its list is known and lacks the log.  An ask without one
asks every machine that lists the log, and says which machines' lists are not
known (``list cut``, ``list failed``).

**What a file covers.**  A day's rotated file holds the lines from the previous
rotation on, and a little past its own: logrotate keeps the mtime it saw
before renaming the file, and the processes write to the renamed file until
postrotate stops them (up to 146 s past the file's mtime among 169 files on
2026-10-14).  So a rotated file covers from the machine's previous rotation to
its own plus :data:`MARGIN`, and neighbours overlap.  A machine's rotation on
a date is the latest mtime among the files it rotated that day in the round's
lists: a quiet log's mtime is its last write, hours before.  The lists hold
only the logs asked, so a round asking only of quiet logs takes their own
mtimes, and a line one writes while its process is being stopped may fall
past the margin (none of the 169 files measured).  The oldest file's
start is not known.  The current file is read by its tail and covers from
where the read began (its first stamp; its rotation when the read began at
its first byte and that line predates the list) to when the grep was asked,
on this machine's clock, which ran 13-203 ms behind the database's.  One read
from its first byte with no stamped line holds no record and covers from its
mtime: server machines list an empty ``panda-JediDBProxy.log`` (2026-10-14).

**Greps.**  One request per machine and file, to that machine alone, with the
strings of every ask reading it.  An answer cut at 1 MB is asked again in
halves, down to a string alone, whose files are then said ``truncated``.  A
grep that fails fails its own strings alone: the other greps are still asked
(a two-string half timed out after its file's 19 strings were cut,
2026-10-30).  A rotated file still being compressed (its plain name listed, or a grep answered
:data:`BEING_COMPRESSED`) is waited for; past the day after its date it is
``unreadable``.  A rotated ``.gz`` does not change, so an earlier round's
answer for the same machine and file is read again when it held every string.

**A record.**  Each ask's record copies the ask and lists, for every machine
it asks, every file whose span meets the window, read or not, with the records
whose lines hold one of the ask's strings: not cut to the window.

**A deferred ask.**  The querier never asks again what a record answers, so a
file a record says ``failed`` would stay unsearched though the failure passed
(a grep cut by a server's timeout, 2026-10-14).  An ask any of whose greps or
lists failed -- refused, failed on a machine, or got no response at all (a
connection dropped, 2026-10-26); a grep is the ask's when it held one of the
ask's strings -- gets no record: it is written to the round's
``answers-r<n>-<time>/deferred.jsonl`` with why, and the querier asks it again.
The next round reads the rotated files this one answered from this round's
answers.  ``unreadable``, ``truncated``, ``list cut``, ``not listed`` and ``no
such log`` say what a file or a list is, and are recorded.  A round whose every
ask is deferred writes no records file, and exits 4.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable

from bamboo.scripts.panda.fieldtest.run import FieldTestError
from bamboo.utils import panda_client as PC

#: how far past a rotation its file is still written to: a slack over the 146 s measured, not a bound
MARGIN = datetime.timedelta(minutes=10)
#: the current file is read to its first byte up to this size; past it, by the window's share
WHOLE_UP_TO = 256 * 2**20
#: the API's largest tail_bytes
MAX_TAIL = 3 * 2**30
#: how much more tail than the window's share of the file since its rotation
TAIL_FACTOR = 1.5
#: how often a day file being compressed is looked at again
COMPRESS_POLL = datetime.timedelta(minutes=10)
#: a grep of a rotated file still being compressed fails with this at the end of its error (panda-server bf5c9fe2)
BEING_COMPRESSED = "Being compressed"
#: the keys of an ask, as the querier writes them
ASK_KEYS = {"name", "service", "machine", "strings", "window", "whole_records"}
FMT = "%Y-%m-%d %H:%M:%S"
ROUND = re.compile(r"^records-r(\d+)\.jsonl$")
#: in a case's round directory: where a round of several cases kept the API answers they shared
SHARED = "shared.json"
DATED = re.compile(r"^(?P<log>panda-.+)-(?P<date>\d{8})(?P<gz>\.gz)?$")


def utc_now() -> datetime.datetime:
    """This machine's clock in UTC, naive, cut to the second."""
    return datetime.datetime.now(datetime.timezone.utc).replace(
        tzinfo=None, microsecond=0
    )


def stamp(text: str) -> datetime.datetime:
    return datetime.datetime.strptime(text[:19], FMT)


def text(t: datetime.datetime | None) -> str | None:
    return None if t is None else t.strftime(FMT)


@dataclass
class Parts:
    """The API parts and clock a round calls; a test hands in stand-ins."""

    list_logs: Callable[..., Awaitable[dict]] = PC.list_logs
    grep_log: Callable[..., Awaitable[dict]] = PC.grep_log
    now: Callable[[], datetime.datetime] = utc_now
    sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep


async def responded(
    call: Callable[..., Awaitable[dict]], *args: Any, **kwargs: Any
) -> dict:
    """*call*'s response, or a refusal saying why when it had none to return.

    panda_client raises ``RuntimeError`` only when there is no response: the
    connection dropped, or what came back was no response object (2026-10-26:
    RemoteDisconnected on a grep's submit, 210 greps into a round, and the
    round wrote nothing).  Read as a refusal, the call's ask is deferred as a
    refused one is."""
    try:
        return await call(*args, **kwargs)
    except RuntimeError as exc:
        return {"success": False, "message": f"no response: {exc}"}


class NotSettled(Exception):
    """A file of the round could not be read within the wait: the round writes nothing."""


@dataclass
class Machine:
    """One machine as the lists say: its service, whether its list is known, and its files."""

    service: str
    status: str  # listed, list cut, list failed
    error: str | None
    truncated: bool
    entries: list[dict]  # {"name", "size", "mtime"}

    def holds(self, log: str) -> bool:
        return any(
            e["name"] == log or e["name"].startswith(log + "-") for e in self.entries
        )


@dataclass
class File:
    """A file of one log on one machine, and what it covers (start None: not known; end None: until read)."""

    machine: str
    log: str
    name: str
    kind: str  # rotated, compressing, current
    date: datetime.date | None
    mtime: datetime.datetime
    size: int
    start: datetime.datetime | None
    end: datetime.datetime | None
    margin: bool

    def meets(self, lo: datetime.datetime, hi: datetime.datetime) -> bool:
        return (self.start is None or self.start <= hi) and (
            self.end is None or lo <= self.end
        )


@dataclass
class Job:
    """The greps of one file on one machine, for every ask reading it, and what they returned."""

    file: File
    strings: set[str]
    since: datetime.datetime  # the earliest window start of the asks reading it
    status: str = ""
    error: str | None = None
    records: dict[int, dict] = field(default_factory=dict)  # by offset: head records
    fragments: dict[int, dict] = field(default_factory=dict)  # by offset: head false
    truncated: set[str] = field(default_factory=set)  # strings cut even alone
    failed: dict[str, str] = field(
        default_factory=dict
    )  # strings whose grep failed: why
    tails: list[tuple[str | None, int | None, datetime.datetime]] = field(
        default_factory=list
    )  # first_stamp, offset, asked
    tail_bytes: int | None = None
    requests: list[dict] = field(default_factory=list)


class Answers:
    """The round's directory of API answers, each kept as it came before it is read."""

    def __init__(self, root: Path, name: str) -> None:
        self.root, self.name = root, name
        self.count = 0

    def keep(self, kind: str, what: str, body: dict) -> str:
        self.count += 1
        rel = f"{self.name}/{self.count:04d}-{kind}-{re.sub(r'[^A-Za-z0-9._-]', '_', what)}.json"
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(body, indent=1, default=str) + "\n")
        return rel


# ---------------------------------------------------------------------------
# asks and rounds
# ---------------------------------------------------------------------------


def key_of(ask: dict) -> str:
    return json.dumps(ask, sort_keys=True, separators=(",", ":"))


def read_asks(path: Path) -> list[dict]:
    """The asks file's lines, each ``{"ask", "step", "why"}``, the ask checked to be in its canonical form."""
    out = []
    for n, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        entry = json.loads(line)
        ask = entry["ask"]
        where = f"{path}:{n}"
        if set(ask) != ASK_KEYS:
            raise FieldTestError(
                f"{where}: an ask has the keys {sorted(ASK_KEYS)}, not {sorted(ask)}"
            )
        if ask["strings"] != sorted(set(ask["strings"])) or not ask["strings"]:
            raise FieldTestError(f"{where}: strings are not sorted, unique and some")
        window = ask["window"]
        if (
            len(window) != 2
            or any(len(w) != 19 for w in window)
            or stamp(window[0]) > stamp(window[1])
        ):
            raise FieldTestError(
                f"{where}: the window is not [from, to] in seconds, from <= to"
            )
        if ask["whole_records"] is not True:
            raise FieldTestError(f"{where}: whole_records must be true")
        out.append(entry)
    return out


def recorded_keys(directory: Path) -> tuple[int, set[str]]:
    """The last round in *directory* and the keys of every ask recorded there."""
    last, keys = 0, set()
    for path in sorted(directory.glob("records-r*.jsonl")):
        m = ROUND.match(path.name)
        if not m:
            continue
        last = max(last, int(m[1]))
        keys |= {
            key_of(json.loads(x)["ask"])
            for x in path.read_text().splitlines()
            if x.strip()
        }
    return last, keys


# ---------------------------------------------------------------------------
# lists
# ---------------------------------------------------------------------------


def stray_names(entries: list[dict], names: list[str]) -> list[str]:
    """The listed names that are no asked log's (an old processor lists every file)."""
    return [
        e["name"]
        for e in entries
        if not any(e["name"] == n or e["name"].startswith(n + "-") for n in names)
    ]


async def list_machines(
    parts: Parts, names: list[str], answers: Answers
) -> tuple[dict[str, Machine], dict[str, str], dict[str, str], datetime.datetime]:
    """Every machine of both services as their lists of *names* say; the services whose list failed, and why; the
    kept answers by service; and when the lists were asked."""
    asked_at = parts.now()

    async def one(service: str, chunk: list[str]) -> tuple[str, list[str], dict]:
        return (
            service,
            chunk,
            await responded(parts.list_logs, service=service, log_names=chunk),
        )

    chunks = [
        names[i : i + PC.LIST_MAX_NAMES]
        for i in range(0, len(names), PC.LIST_MAX_NAMES)
    ]
    got = await asyncio.gather(*(one(s, c) for s in PC.LOG_SERVICES for c in chunks))
    machines: dict[str, Machine] = {}
    failed: dict[str, str] = {}
    kept: dict[str, str] = {}
    for service, chunk, response in got:
        kept[service] = answers.keep(
            "list",
            service,
            {"asked_at": text(asked_at), "log_names": chunk, "response": response},
        )
        if not response.get("success"):
            failed[service] = response.get("message") or "the list was refused"
            continue
        for name, a in PC.per_machine(response, True).items():
            entries = (a.get("result") or []) if a.get("status") == "done" else []
            stray = stray_names(entries, chunk)
            if a.get("status") != "done":
                status, error = "list failed", a.get("error") or a.get("status")
            elif a.get("truncated"):
                status, error = "list cut", "truncated at 1 MB"
            elif stray:
                status, error = "list cut", f"log_names not honoured: {stray[0]} listed"
            else:
                status, error = "listed", None
            m = machines.setdefault(name, Machine(service, "listed", None, False, []))
            m.entries.extend(entries)
            m.truncated = m.truncated or bool(a.get("truncated"))
            if status != "listed" and m.status != "list failed":
                m.status, m.error = status, error
    return machines, failed, kept, asked_at


# ---------------------------------------------------------------------------
# files
# ---------------------------------------------------------------------------


def rotations(entries: list[dict]) -> dict[datetime.date, datetime.datetime]:
    """When the machine rotated on each date: the latest mtime of the files it rotated that day."""
    out: dict[datetime.date, datetime.datetime] = {}
    for e in entries:
        m = DATED.match(e["name"])
        if m:
            day = datetime.datetime.strptime(m["date"], "%Y%m%d").date()
            out[day] = max(out.get(day, datetime.datetime.min), stamp(e["mtime"]))
    return out


def chain(machine: str, log: str, m: Machine) -> tuple[list[File], list[str]]:
    """The files of *log* on the machine, oldest first, with what each covers; and what was set aside, and why."""
    rot = rotations(m.entries)
    days: dict[datetime.date, dict[str, dict]] = {}
    current = None
    notes = []
    for e in m.entries:
        if e["name"] == log:
            current = e
            continue
        d = DATED.match(e["name"])
        if not d or d["log"] != log:
            if e["name"].startswith(log + "-"):
                notes.append(
                    f"{machine}: {e['name']} is no rotated file of {log}: not read"
                )
            continue
        day = datetime.datetime.strptime(d["date"], "%Y%m%d").date()
        days.setdefault(day, {})["gz" if d["gz"] else "plain"] = e
    dated = []
    for day, kinds in days.items():
        plain, gz = kinds.get("plain"), kinds.get("gz")
        e = plain or gz
        name = f"{log}-{day:%Y%m%d}.gz"
        kind = "compressing" if plain else "rotated"
        dated.append(
            File(
                machine,
                log,
                name,
                kind,
                day,
                stamp(e["mtime"]),
                int(e["size"]),
                None,
                None,
                True,
            )
        )
    dated.sort(key=lambda f: f.mtime)
    if [f.date for f in dated] != sorted(f.date for f in dated):
        notes.append(
            f"{machine}: the rotated files of {log} are not in their dates' order by mtime; read by mtime"
        )
    before = None
    for f in dated:
        f.start = before
        f.end = max(rot.get(f.date, f.mtime), f.mtime) + MARGIN
        before = max(rot.get(f.date, f.mtime), f.mtime)
    files = list(dated)
    if current is not None:
        files.append(
            File(
                machine,
                log,
                log,
                "current",
                None,
                stamp(current["mtime"]),
                int(current["size"]),
                before,
                None,
                False,
            )
        )
    return files, notes


def tail_for(f: File, since: datetime.datetime, now: datetime.datetime) -> int:
    """How much of the current file's end to read for a window from *since*."""
    if f.size <= WHOLE_UP_TO or f.start is None:
        return min(MAX_TAIL, max(1, f.size * 2 + 64 * 2**20))
    span = (now - f.start).total_seconds()
    want = (now - max(since - MARGIN, f.start)).total_seconds()
    share = min(1.0, max(want, 1.0) / max(span, 1.0))
    return min(MAX_TAIL, int(f.size * share * TAIL_FACTOR) + 1)


# ---------------------------------------------------------------------------
# greps
# ---------------------------------------------------------------------------


@dataclass
class Earlier:
    """Answers kept by earlier rounds for rotated files, by (machine, file): each leaf's strings and records."""

    leaves: dict[tuple[str, str], list[dict]] = field(default_factory=dict)

    @classmethod
    def read(cls, *directories: Path) -> Earlier:
        """The leaves kept in each case directory's rounds, and in the shared answers a round of several cases kept."""
        out = cls()
        kept: list[Path] = []
        for directory in directories:
            for round_dir in sorted(directory.glob("answers-r*")):
                if (round_dir / "leaves.jsonl").exists():
                    kept.append(round_dir)
                if (pointer := round_dir / SHARED).exists():
                    kept.append(Path(json.loads(pointer.read_text())["answers"]))
        for path in dict.fromkeys(d / "leaves.jsonl" for d in kept):
            if not path.exists():
                continue
            for line in path.read_text().splitlines():
                if line.strip():
                    leaf = json.loads(line)
                    out.leaves.setdefault((leaf["machine"], leaf["file"]), []).append(
                        leaf
                    )
        return out

    def cover(self, machine: str, file: str, strings: set[str]) -> list[dict] | None:
        """Leaves that together held every one of *strings* uncut, or None."""
        found = [
            leaf
            for leaf in self.leaves.get((machine, file), [])
            if not leaf["truncated"]
        ]
        held = {s for leaf in found for s in leaf["strings"]}
        return found if strings <= held else None


class Grepper:
    """Runs the round's jobs: a grep per machine and file, halved when cut, waited for when being compressed."""

    def __init__(
        self,
        parts: Parts,
        answers: Answers,
        earlier: Earlier,
        deadline: datetime.datetime,
        listed_at: datetime.datetime,
    ) -> None:
        self.parts, self.answers, self.earlier = parts, answers, earlier
        self.deadline, self.listed_at = deadline, listed_at
        self.leaves: list[dict] = []

    async def run(self, job: Job) -> None:
        f = job.file
        if f.kind == "compressing" and not await self.compressed(job):
            return
        if f.kind != "current":
            reused = self.earlier.cover(f.machine, f.name, job.strings)
            if reused is not None:
                for leaf in reused:
                    self.take(job, leaf["records"], leaf["fragments"])
                    job.requests.append(
                        {"machine": f.machine, "file": f.name, "reused": leaf["answer"]}
                    )
                job.status = "answered"
                return
        while True:
            job.status = ""
            job.failed.clear()
            outcome = await self.grep(job, sorted(job.strings))
            if outcome != "compressing":
                break
            if not await self.compressed(job):
                return
        if (
            job.failed and job.status != "answered"
        ):  # every grep failed: the file is failed
            job.status, job.error = "failed", next(iter(job.failed.values()))
        if f.kind == "current" and job.status == "answered":
            self.settle_tail(job)

    async def compressed(self, job: Job) -> bool:
        """Wait for the file's day to be compressed; False when it is now unreadable (the job is settled)."""
        f = job.file
        give_up = datetime.datetime.combine(
            f.date, datetime.time()
        ) + datetime.timedelta(days=2)
        while True:
            now = self.parts.now()
            if now >= give_up:
                job.status, job.error = (
                    "unreadable",
                    f"being compressed past {give_up - datetime.timedelta(seconds=1):%Y-%m-%d %H:%M:%S}",
                )
                return False
            if now >= self.deadline:
                raise NotSettled(
                    f"{f.machine} {f.name}: still being compressed at {text(now)}"
                )
            if f.kind == "compressing":
                await self.parts.sleep(COMPRESS_POLL.total_seconds())
                response = await responded(
                    self.parts.list_logs, machine=f.machine, log_names=[f.log]
                )
                kept = self.answers.keep(
                    "relist", f"{f.machine}-{f.log}", {"response": response}
                )
                got = PC.per_machine(response, True).get(f.machine) or {}
                names = (
                    {e["name"] for e in (got.get("result") or [])}
                    if got.get("status") == "done"
                    else set()
                )
                plain = f.name[: -len(".gz")]
                job.requests.append(
                    {
                        "machine": f.machine,
                        "file": f.log,
                        "list": kept,
                        "plain_listed": plain in names,
                    }
                )
                if (
                    got.get("status") == "done"
                    and plain not in names
                    and f.name in names
                ):
                    f.kind = "rotated"
                    return True
            else:
                f.kind = "compressing"

    async def grep(self, job: Job, strings: list[str]) -> str:
        """Grep *strings* in the job's file; halve them while the answer is cut.  'compressing' when refused for it;
        'failed' when a grep failed, once every other grep was asked (:attr:`Job.failed` holds its strings)."""
        f = job.file
        tail = None
        if f.kind == "current":
            tail = job.tail_bytes or tail_for(f, job.since, self.parts.now())
            job.tail_bytes = tail
        outcomes = []
        for i in range(0, len(strings), PC.GREP_MAX_STRINGS):
            chunk = strings[i : i + PC.GREP_MAX_STRINGS]
            outcome = await self.leaf(job, chunk, tail)
            if outcome == "compressing":
                return outcome
            outcomes.append(outcome)
        return "failed" if "failed" in outcomes else "answered"

    async def leaf(self, job: Job, strings: list[str], tail: int | None) -> str:
        f = job.file
        asked = self.parts.now()
        response = await responded(
            self.parts.grep_log, f.name, strings, machine=f.machine, tail_bytes=tail
        )
        answered = self.parts.now()
        kept = self.answers.keep(
            "grep",
            f"{f.machine}-{f.name}",
            {
                "asked_at": text(asked),
                "strings": strings,
                "tail_bytes": tail,
                "response": response,
            },
        )
        request = {
            "machine": f.machine,
            "file": f.name,
            "strings": len(strings),
            "tail_bytes": tail,
            "submitted_at": text(asked),
            "answered_at": text(answered),
            "answer": kept,
        }
        job.requests.append(request)
        if not response.get("success"):
            request["status"] = "refused"
            why = response.get("message") or "the grep was refused"
            job.failed.update(dict.fromkeys(strings, why))
            return "failed"
        got = PC.per_machine(response, True).get(f.machine) or {"status": "no result"}
        request["status"], request["error"] = got.get("status"), got.get("error")
        if got.get("status") == "failed" and (got.get("error") or "").rstrip().endswith(
            BEING_COMPRESSED
        ):
            return "compressing"
        if got.get("status") != "done":
            job.failed.update(
                dict.fromkeys(strings, got.get("error") or got.get("status"))
            )
            return "failed"
        if got.get("truncated") and len(strings) > 1:
            half = len(strings) // 2
            outcomes = []
            for part in (strings[:half], strings[half:]):
                outcome = await self.leaf(job, part, tail)
                if outcome == "compressing":
                    return outcome
                outcomes.append(outcome)
            return "failed" if "failed" in outcomes else "answered"
        result = got.get("result") or {}
        records = result.get("records") or []
        heads = [r for r in records if r.get("head")]
        fragments = [r for r in records if not r.get("head")]
        self.take(job, heads, fragments)
        if got.get("truncated"):
            job.truncated.update(strings)
        if f.kind == "current":
            window = result.get("window") or {}
            job.tails.append((window.get("first_stamp"), window.get("offset"), asked))
        else:
            self.leaves.append(
                {
                    "machine": f.machine,
                    "file": f.name,
                    "strings": strings,
                    "truncated": bool(got.get("truncated")),
                    "answer": kept,
                    "records": heads,
                    "fragments": fragments,
                }
            )
        job.status = "answered"
        return "answered"

    def take(self, job: Job, heads: list[dict], fragments: list[dict]) -> None:
        for r in heads:
            job.records.setdefault(r["offset"], r)
        for r in fragments:
            job.fragments.setdefault(r["offset"], r)

    def settle_tail(self, job: Job) -> None:
        """Note a tail that began after the window it was read for (read again once with twice the bytes).  A file
        read from its first byte with no stamped line has no more to read."""
        covers, covers_from = tail_covers(job, self.listed_at)
        if covers is None or covers_from == "unstamped":
            return
        need = max(job.since, job.file.start) if job.file.start else job.since
        if stamp(covers[0]) > need:
            job.error = f"the tail of {job.tail_bytes} bytes began at {covers[0]}, after {text(need)}"


def tail_covers(job: Job, listed_at: datetime.datetime) -> tuple[list[str] | None, str]:
    """What the current file's reads covered, and from what: every leaf's, narrowed to what all of them read.

    A read that began at the file's first byte with a line older than the list is the file listed, whole: it covers
    from the rotation before it.  Reads that all began at the first byte and found no stamped line read a file that
    holds no record: it covers from its mtime, after which nothing was written to it (an empty file a machine's
    processes never log to, such as JediDBProxy's on a server machine).  Otherwise it covers from its first stamp;
    one with no stamp covers nothing."""
    if job.tails and all(offset == 0 and not first for first, offset, _ in job.tails):
        return [
            text(job.file.mtime),
            text(min(asked for _, _, asked in job.tails)),
        ], "unstamped"
    starts = []
    whole_all = True
    for first, offset, _ in job.tails:
        if not first:
            return None, "first_stamp"
        t = stamp(first)
        whole = offset == 0 and t < listed_at
        whole_all = whole_all and whole
        starts.append(
            (job.file.start if job.file.start is not None else t) if whole else t
        )
    if not starts:
        return None, "first_stamp"
    return [
        text(max(starts)),
        text(min(asked for _, _, asked in job.tails)),
    ], "rotation" if whole_all else "first_stamp"


async def run_jobs(grepper: Grepper, jobs: list[Job]) -> None:
    async def one(job: Job) -> None:
        await grepper.run(job)
        f = job.file
        if (
            f.kind == "current"
            and job.status == "answered"
            and job.error
            and job.tail_bytes
            and job.tail_bytes < MAX_TAIL
        ):
            job.tail_bytes = min(MAX_TAIL, job.tail_bytes * 2)
            job.records.clear()
            job.fragments.clear()
            job.truncated.clear()
            job.tails.clear()
            job.error = None
            await grepper.run(job)

    await asyncio.gather(*(one(j) for j in jobs))


# ---------------------------------------------------------------------------
# the round
# ---------------------------------------------------------------------------


def holds_any(record: dict, strings: set[str]) -> bool:
    return any(s in line for line in record.get("lines", []) for s in strings)


def file_entry(
    job: Job | None, f: File, strings: set[str], listed_at: datetime.datetime
) -> dict:
    """One file of a record: what it covers and, read, the records of the ask's strings.

    The current file not read covers from its rotation to when it was listed: not searched, so said."""
    covers_from = "mtime"
    if f.kind != "current":
        covers = [text(f.start), text(f.end)]
    elif job and job.status == "answered":
        covers, covers_from = tail_covers(job, listed_at)
    else:
        covers, covers_from = [text(f.start), text(listed_at)], "list"
    # a grep that failed is the ask's when it held one of the ask's strings; the file's other greps answer the rest
    failed = sorted({job.failed[s] for s in strings & job.failed.keys()}) if job else []
    entry = {
        "file": f.name,
        "covers": covers,
        "covers_from": covers_from,
        "tail_bytes": job.tail_bytes if job else None,
        "status": "failed"
        if failed
        else job.status
        if job and job.status
        else "failed",
        "error": "; ".join(failed) if failed else job.error if job else "not read",
        "truncated": bool(job and strings & job.truncated),
        "records": [r for _, r in sorted(job.records.items()) if holds_any(r, strings)]
        if job
        else [],
        "fragments": sum(1 for r in job.fragments.values() if holds_any(r, strings))
        if job
        else 0,
    }
    if f.margin:
        entry["margin_s"] = int(MARGIN.total_seconds())
    if (
        entry["status"] == "answered"
        and f.kind == "current"
        and entry["covers"] is None
    ):
        entry["error"] = entry["error"] or "the tail held no stamp"
    return entry


async def record_round(
    asks: list[dict], directory: Path, parts: Parts, max_wait: datetime.timedelta
) -> tuple[Path | None, list[dict], list[dict]]:
    """Answer *asks* and write them as the next round in *directory*: the file written (None where every ask was
    deferred), its records, and the asks deferred with why."""
    (out,) = await record_rounds([(asks, directory)], parts, max_wait)
    return out


async def record_rounds(
    cases: list[tuple[list[dict], Path]],
    parts: Parts,
    max_wait: datetime.timedelta,
    shared: Path | None = None,
) -> list[tuple[Path | None, list[dict], list[dict]]]:
    """Answer the asks of several cases in one round, each written as the next round in its own directory.

    One list of the logs and one grep per machine and file serve every case's
    asks, and each record keeps its own ask's strings, as the asks of one case
    share them.  The API answers are kept once, under *shared*, and each case's
    round directory says where (:data:`SHARED`); with one case and no *shared*
    they sit in its own round directory.  By case, as :func:`record_round`
    returns them."""
    if shared is None and len(cases) != 1:
        raise FieldTestError(
            "a round of several cases keeps its answers in a shared directory"
        )
    rounds = []
    for case_asks, directory in cases:
        last, keys = recorded_keys(directory)
        again = [a["ask"] for a in case_asks if key_of(a["ask"]) in keys]
        if again:
            raise FieldTestError(
                f"{len(again)} ask(s) already recorded in {directory}, e.g. {key_of(again[0])}"
            )
        rounds.append(last + 1)
    started = parts.now()
    if shared is None:
        answers = Answers(cases[0][1], f"answers-r{rounds[0]}-{started:%Y%m%dT%H%M%S}")
    else:
        answers = Answers(shared, f"answers-{started:%Y%m%dT%H%M%S}")
    asks = [a for case_asks, _ in cases for a in case_asks]
    round_of = [
        n for (case_asks, _), n in zip(cases, rounds, strict=True) for _ in case_asks
    ]
    names = sorted({a["ask"]["name"] for a in asks})
    machines, failed, kept, listed_at = await list_machines(parts, names, answers)

    # each ask's machines and the files of each that meet its window
    plans: list[dict[str, dict]] = []
    jobs: dict[tuple[str, str], Job] = {}
    ignored: list[str] = []
    chains: dict[tuple[str, str], list[File]] = {}
    for entry in asks:
        ask = entry["ask"]
        lo, hi = (stamp(w) for w in ask["window"])
        plan: dict[str, dict] = {}
        if ask["machine"] is not None:
            m = machines.get(ask["machine"])
            if m is None:
                why = "; ".join(
                    f"the {s} list failed: {e}" for s, e in sorted(failed.items())
                )
                plan[ask["machine"]] = {
                    "status": "list failed" if failed else "not listed",
                    "error": why or None,
                    "service": None,
                }
            elif m.status != "listed":
                plan[ask["machine"]] = {
                    "status": m.status,
                    "error": m.error,
                    "service": m.service,
                }
            elif not m.holds(ask["name"]):
                plan[ask["machine"]] = {
                    "status": "no such log",
                    "error": None,
                    "service": m.service,
                }
            else:
                plan[ask["machine"]] = {
                    "status": "listed",
                    "error": None,
                    "service": m.service,
                }
        else:
            for name, m in sorted(machines.items()):
                if m.status != "listed":
                    plan[name] = {
                        "status": m.status,
                        "error": m.error,
                        "service": m.service,
                    }
                elif m.holds(ask["name"]):
                    plan[name] = {
                        "status": "listed",
                        "error": None,
                        "service": m.service,
                    }
            for service, why in sorted(failed.items()):
                plan[f"every {service} machine"] = {
                    "status": "list failed",
                    "error": why,
                    "service": service,
                }
        for name, p in plan.items():
            if p["status"] != "listed":
                continue
            at = (name, ask["name"])
            if at not in chains:
                chains[at], notes = chain(name, ask["name"], machines[name])
                ignored.extend(notes)
            p["files"] = [f for f in chains[at] if f.meets(lo, hi)]
            for f in p["files"]:
                job = jobs.setdefault((name, f.name), Job(f, set(), lo))
                job.strings.update(ask["strings"])
                job.since = min(job.since, lo)
        plans.append(plan)

    earlier = Earlier.read(*(directory for _, directory in cases))
    grepper = Grepper(parts, answers, earlier, started + max_wait, listed_at)
    await run_jobs(grepper, list(jobs.values()))

    records = []
    for entry, plan, n in zip(asks, plans, round_of, strict=True):
        ask = entry["ask"]
        strings = set(ask["strings"])
        out_machines = {}
        requests = []
        for name, p in plan.items():
            files = []
            for f in p.get("files", []):
                job = jobs.get((name, f.name))
                files.append(file_entry(job, f, strings, listed_at))
                requests.extend(job.requests if job else [])
            out_machines[name] = {
                "service": p["service"],
                "status": p["status"],
                "error": p["error"],
                "files": files,
            }
        records.append(
            {
                "ask": ask,
                "round": n,
                "machines": out_machines,
                "list_logs": {
                    "asked_at": text(listed_at),
                    "answers": kept,
                    "machines": {
                        m: {
                            "service": machines[m].service,
                            "status": machines[m].status,
                            "truncated": machines[m].truncated,
                            "error": machines[m].error,
                        }
                        for m in out_machines
                        if m in machines
                    },
                    "failed": failed,
                },
                "ignored": sorted(set(ignored)),
                "requests": requests,
            }
        )
    kept_dir = answers.root / answers.name
    kept_dir.mkdir(parents=True, exist_ok=True)
    (kept_dir / "leaves.jsonl").write_text(
        "".join(json.dumps(leaf) + "\n" for leaf in grepper.leaves)
    )
    out = []
    start = 0
    for (case_asks, directory), n in zip(cases, rounds, strict=True):
        mine = records[start : start + len(case_asks)]
        start += len(case_asks)
        # an ask a list or a grep failed for is asked again rather than recorded with a hole no round would fill
        deferred = [
            {"ask": r["ask"], "why": why}
            for r in mine
            if (why := failures(r["machines"]))
        ]
        mine = [r for r in mine if not failures(r["machines"])]
        round_dir = (
            kept_dir
            if shared is None
            else directory / f"answers-r{n}-{started:%Y%m%dT%H%M%S}"
        )
        round_dir.mkdir(parents=True, exist_ok=True)
        if shared is not None:
            (round_dir / SHARED).write_text(
                json.dumps({"answers": str(kept_dir)}) + "\n"
            )
        if deferred:
            (round_dir / "deferred.jsonl").write_text(
                "".join(json.dumps(d) + "\n" for d in deferred)
            )
        if not mine:
            out.append((None, [], deferred))
            continue
        path = directory / f"records-r{n}.jsonl"
        tmp = path.with_suffix(".jsonl.tmp")
        tmp.write_text("".join(json.dumps(r) + "\n" for r in mine))
        tmp.rename(path)
        out.append((path, mine, deferred))
    return out


def failures(machines: dict) -> list[str]:
    """What a later round may still read: each machine of a record whose list failed, each file whose grep failed."""
    out = []
    for name, m in machines.items():
        if m["status"] == "list failed":
            out.append(f"{name}: list failed: {m['error']}")
        out.extend(
            f"{name} {f['file']}: failed: {f['error']}"
            for f in m["files"]
            if f["status"] == "failed"
        )
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m bamboo.scripts.panda.fieldtest.record",
        description=__doc__.split("\n\n")[0],
    )
    ap.add_argument("--asks", help="the asks file the querier wrote")
    ap.add_argument("--records", help="the case's directory of records-r<n>.jsonl")
    ap.add_argument(
        "--cases",
        help="several cases in one round: a file of lines '<asks file> <records directory>'",
    )
    ap.add_argument(
        "--shared", help="with --cases: where the round keeps the API answers once"
    )
    ap.add_argument(
        "--max-wait",
        type=float,
        default=7200.0,
        help="seconds to wait for a day file being compressed",
    )
    a = ap.parse_args(argv)
    if bool(a.cases) == bool(a.asks or a.records) or bool(a.cases) != bool(a.shared):
        ap.error("give --asks and --records, or --cases and --shared")
    if a.cases:
        pairs = [
            line.split()
            for line in Path(a.cases).read_text().splitlines()
            if line.strip()
        ]
        shared = Path(a.shared)
        shared.mkdir(parents=True, exist_ok=True)
    else:
        pairs, shared = [[a.asks, a.records]], None
    cases = []
    for asks_file, records in pairs:
        asks = read_asks(Path(asks_file))
        if not asks:
            print(f"{asks_file}: no asks")
            continue
        directory = Path(records)
        directory.mkdir(parents=True, exist_ok=True)
        cases.append((asks, directory))
    if not cases:
        return 0
    try:
        out = asyncio.run(
            record_rounds(
                cases, Parts(), datetime.timedelta(seconds=a.max_wait), shared
            )
        )
    except NotSettled as e:
        print(f"not settled, nothing written: {e}", file=sys.stderr)
        return 3
    written = 0
    for (_, directory), (path, records, deferred) in zip(cases, out, strict=True):
        for d in deferred:
            print(f"deferred {key_of(d['ask'])}: {'; '.join(d['why'])}")
        if path is None:
            print(
                f"{directory}: nothing written: every one of {len(deferred)} ask(s) deferred",
                file=sys.stderr,
            )
            continue
        written += 1
        print(f"{path}: {len(records)} record(s), {len(deferred)} ask(s) deferred")
    return 0 if written else 4


if __name__ == "__main__":
    sys.exit(main())
