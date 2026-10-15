"""Log records, runs and traceback frames -- the evidence side of step 2 of the loop.

Three rules of the real run A, and one of them is two rules folded together:

* a *record* is a line that starts with a time stamp plus every line after it
  that does not (rule 2: the traceback under an ERROR line is part of it);
* a *run* is every record carrying the same instance mark as the records that
  carry the entity key (rules 1 and 7 at once).  PanDA's LogWrapper puts one
  token between the level and the message: the adder puts the PandaID there,
  the setupper a stamp of its own run.  Either way the token is what one run
  of one worker shares, and the key is what finds it;
* a *frame* is a ``File "...", line N, in f`` line of a traceback, placed in
  the corpus and checked against the source line printed under it -- a frame
  whose source line differs is from another version of the code and says
  nothing about this tree.

Absence is only ever read against a run and a level (rule 8): a line that is
missing from a run that printed nothing at that level proves nothing.  Nor
from a run of an extract that does not hold the key in its mark: a file made
by grepping the key holds only what the grep reached of such a run, and says
so in its first line (``merge_wassh.py --about``).
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol

#: the panda-server tree every reading here is taken from: ``PANDA_PIN``, a pinned copy (a ``git archive`` of the
#: commit the logs were written by), so every run reads a tree no one edits.  Required: a default to a live tree
#: would be read silently by a run that forgot the variable -- and once the stored map is of another commit than
#: a case's pin, a forgotten pin is a wrong answer.
if not os.environ.get("PANDA_PIN"):
    raise SystemExit("set PANDA_PIN to the pinned panda-server copy (a git archive of the commit) "
                     "the case's logs were written by")
CORPUS = Path(os.environ["PANDA_PIN"])

STAMP = re.compile(r"^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d")
#: the instance mark is the first thing after the level: a bare token (the adder's
#: PandaID, the setupper's stamp), a bracketed one, which JEDI's MsgWrapper writes
#: with blanks inside (``< jediTaskID=12345 >``) -- one mark, not ``<`` -- or a
#: method name and a bracketed tag, which a DB proxy's ``create_tagged_logger`` writes
#: (``getJobs < SITE_NAME 2026-01-01T00:00:00.000000 >``, ``updateJobStatus <
#: PandaID=12345678 >``): the name alone would make one run of every call of the
#: method, for every job
PREFIX = re.compile(
    r"^(?P<stamp>\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d{3}) (?P<logger>\S+): "
    r"(?P<level>DEBUG|INFO|WARNING|ERROR|CRITICAL)\s+"
    r"(?P<token>[A-Za-z_]\w* <[^<>\n]*>|<[^<>\n]*>|\S+) ?(?P<message>.*)$"
)
TRACEBACK = "Traceback (most recent call last):"
FRAME = re.compile(r'^\s*File "(?P<path>[^"]+)", line (?P<line>\d+), in (?P<func>\S+)')
EXCEPTION = re.compile(r"^(?P<type>[A-Za-z_][\w.]*): (?P<message>.*)$")
PACKAGES = ("pandaserver/", "pandajedi/")
#: the first line of a log made by grepping a key (``merge_wassh.py --about``)
EXTRACT = re.compile(r"^# extract by (?P<key>\S+):")

LEVEL_RANK = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40, "CRITICAL": 50}

#: rule 8 in its earlier, stricter form: a row's absence counts only if the file holds a line at its level from some run
FILE_LEVEL = os.environ.get("RULE8_FILE_LEVEL") == "1"
#: a filter added in code, the class, or a fileConfig whose file this cannot read; and a "filters" key, counted only in
#: a file that calls dictConfig (an API's query "filters" is not a logging one)
_FILTER = re.compile(r"\baddFilter\s*\(|\blogging\.Filter\b|\bfileConfig\s*\(")
_DICT_FILTERS = re.compile(r"[\"']filters[\"']\s*:")
_LEVEL_FILTERS: list[str] | None = None


def level_filters() -> list[str]:
    """``file:line`` of every logging filter in the corpus and in panda-common (the logger PanDA's code uses).

    Handlers have thresholds; only a filter can keep one level of a logger out of a file its lower levels reach,
    and ``Run.would_print`` reads a run's lower lines as the level reaching the file while there is none.
    None in the PanDA versions this was written against.  Scanned once per process.
    """
    global _LEVEL_FILTERS
    if _LEVEL_FILTERS is None:
        import importlib.util

        roots = [CORPUS]
        spec = importlib.util.find_spec("pandacommon")
        if spec and spec.submodule_search_locations:
            roots += [Path(p) for p in spec.submodule_search_locations]
        else:
            # not found is not "no filter": keep the old rule
            _LEVEL_FILTERS = ["pandacommon: not found, so its loggers were not read"]
            return _LEVEL_FILTERS
        found = []
        for root in roots:
            for path in sorted(root.rglob("*.py")):
                source = path.read_text(errors="replace")
                configs = "dictConfig" in source
                for n, text in enumerate(source.splitlines(), 1):
                    if _FILTER.search(text) or (configs and _DICT_FILTERS.search(text)):
                        found.append(f"{path}:{n}")
        _LEVEL_FILTERS = found
    return _LEVEL_FILTERS

#: the logging method a skeleton row is printed with, as the prefix spells its level
LEVEL_OF_METHOD = {
    "debug": "DEBUG",
    "info": "INFO",
    "warning": "WARNING",
    "warn": "WARNING",
    "error": "ERROR",
    "exception": "ERROR",
    "critical": "CRITICAL",
}


@dataclass
class Frame:
    path: str
    line: int
    func: str
    source: str
    rel: str = ""  # the corpus path when the frame is in it
    same_source: bool | None = None  # None when the frame is outside the corpus

    @property
    def in_corpus(self) -> bool:
        return bool(self.rel)


@dataclass
class Chain:
    """One ``Traceback (most recent call last):`` block and the exception under it."""

    frames: list[Frame]
    exc_type: str
    exc_message: str


@dataclass
class Record:
    file: str
    lineno: int  # of the first line, 1-based
    stamp: str
    logger: str
    level: str
    token: str
    message: str
    continuation: list[str] = field(default_factory=list)
    #: where a recorded answer found it (``recorded.RecordedLogs``): the machine and the byte offset in its file;
    #: "" and -1 in a file handed in, which has line numbers
    machine: str = ""
    offset: int = -1
    #: the search cut the record after 200 lines: its traceback and holes are not whole
    cut: bool = False

    @property
    def text(self) -> str:
        """The message and its continuation lines, as one string."""
        return "\n".join([self.message, *self.continuation])

    @property
    def span(self) -> str:
        if self.machine:
            return f"{self.machine.split('.')[0]}:{Path(self.file).name}@{self.offset}"
        last = self.lineno + len(self.continuation)
        return f"{Path(self.file).name}:{self.lineno}" + (f"-{last}" if last != self.lineno else "")

    def chains(self) -> list[Chain]:
        """The tracebacks in this record.

        The header is not always a line of its own: ``f"execute() : {e}" +
        traceback.format_exc()`` glues it to the end of the message, and the
        continuation starts at the first ``File`` line.
        """
        lines = self.continuation
        if self.message.rstrip().endswith(TRACEBACK) and not (lines and lines[0].startswith(TRACEBACK)):
            lines = [TRACEBACK, *lines]
        return _chains(lines)


def _place(path: str, line: int, source: str) -> tuple[str, bool | None]:
    for package in PACKAGES:
        at = path.rfind(package)
        if at < 0:
            continue
        rel = path[at:]
        target = CORPUS / rel
        if not target.exists():
            return "", None
        lines = target.read_text(errors="replace").splitlines()
        here = lines[line - 1].strip() if 0 < line <= len(lines) else ""
        return rel, here == source.strip()
    return "", None


def _chains(lines: list[str]) -> list[Chain]:
    chains: list[Chain] = []
    i = 0
    while i < len(lines):
        if not lines[i].startswith(TRACEBACK):
            i += 1
            continue
        i += 1
        frames: list[Frame] = []
        while i < len(lines):
            m = FRAME.match(lines[i])
            if not m:
                break
            source = lines[i + 1].strip() if i + 1 < len(lines) else ""
            i += 2
            # the caret line python 3.11 prints under a source line
            while i < len(lines) and set(lines[i].strip()) <= {"^", "~"} and lines[i].strip():
                i += 1
            path, line = m["path"], int(m["line"])
            rel, same = _place(path, line, source)
            frames.append(Frame(path, line, m["func"], source, rel, same))
        exc_type, message = "", []
        if i < len(lines):
            m = EXCEPTION.match(lines[i])
            if m:
                exc_type = m["type"]
                message.append(m["message"])
                i += 1
                while i < len(lines) and lines[i].strip() and not lines[i].startswith("The above exception"):
                    message.append(lines[i])
                    i += 1
        chains.append(Chain(frames, exc_type, "\n".join(message)))
    return chains


@dataclass
class Log:
    path: str
    records: list[Record]
    extract_key: str = ""  # the key the file was grepped by, when it says it is an extract
    #: what of the window asked was not searched, and why (a recorded answer; a file handed in says nothing)
    holes: list[str] = field(default_factory=list)
    #: a recorded search's test of a run or a pass, ``(machine, first stamp, last stamp) -> bool``: false where it may
    #: go on past what was searched, and rule 8 reads no row missing from it (None in a file handed in)
    whole: Callable[[str, str, str], bool] | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        """The log file the map names, which this file is a copy or a rotation of.

        The map says ``panda-adder.log``; production rotates it into
        ``panda-adder.log.1`` and a copy handed over may carry the key
        (``panda-adder.log.<PandaID>``).  Whatever follows ``.log`` is dropped.
        """
        base = Path(self.path).name
        at = base.find(".log")
        return base[: at + len(".log")] if at >= 0 else base

    @property
    def covers(self) -> tuple[str, str]:
        return (self.records[0].stamp, self.records[-1].stamp) if self.records else ("", "")

    def covers_time(self, stamp: str) -> bool:
        first, last = self.covers
        return bool(first) and first[: len(stamp)] <= stamp <= last[: len(stamp)] + "~"


def read_log(path: str) -> Log:
    records: list[Record] = []
    extract_key = ""
    for lineno, line in enumerate(Path(path).read_text(errors="replace").splitlines(), 1):
        if lineno == 1 and (m := EXTRACT.match(line)):
            extract_key = m["key"]
        elif STAMP.match(line):
            m = PREFIX.match(line)
            if m:
                records.append(
                    Record(path, lineno, m["stamp"], m["logger"], m["level"], m["token"], m["message"])
                )
            else:
                records.append(Record(path, lineno, line[:23], "", "", "", line[24:]))
        elif records:
            records[-1].continuation.append(line)
    return Log(path, records, extract_key)


#: (from, to) in the statuslog's form, ``YYYY-MM-DD HH:MM:SS``
Window = tuple[str, str]


class Unrecorded(Exception):
    """A door has no answer for these asks yet: ``asks`` is ``[(ask, why)]``, *why* empty when the asker's reason is
    the ask's.  The asker gathers them from every log of its step before it stops, so they are fetched together."""

    def __init__(self, asks: list) -> None:
        super().__init__(f"{len(asks)} ask(s) not recorded")
        self.asks = asks


class Logs(Protocol):
    """The logs a run reads, asked one at a time: (log name, key, window, machine) -> Log.

    A run asks only through here, so what answers -- files handed over, or a search of the servers' logs -- is the
    door's business.  The answer holds every run of the key that overlaps *window*, each whole: a run is not cut
    at the window's ends, and a door may return more runs than those.  *window* is when the asker needs the key's
    runs, or None for whenever the log holds them; *machine* is the one machine the run must be on, or None for any.
    """

    #: whether every ask needs a window: a door that searches the servers' logs reads only the files a window reaches
    needs_window: bool

    def names(self, wanted: list[str]) -> list[str]:
        """Of the logs *wanted*, the ones this door can be asked for, in its order."""
        ...

    def log(self, name: str, key: str, window: Window | None = None, *, machine: str | None = None) -> Log: ...


class FileLogs:
    """The door over files: one file per log, read once.

    A file holds what it holds, so the key, the window and the machine select nothing here: every ask of a log
    returns the same :class:`Log` (``pattern_finds`` caches by it).  A log wanted that was not handed in is not
    asked at all.
    """

    needs_window = False

    def __init__(self, paths: list[str]) -> None:
        self._logs: dict[str, Log] = {}
        for path in paths:
            log = read_log(path)
            if log.name in self._logs:
                raise SystemExit(f"{path} and {self._logs[log.name].path} are both {log.name}: give one file per log")
            self._logs[log.name] = log

    def names(self, wanted: list[str]) -> list[str]:
        return [name for name in self._logs if name in wanted]

    def log(self, name: str, key: str, window: Window | None = None, *, machine: str | None = None) -> Log:
        return self._logs[name]


def keyed(key: str) -> re.Pattern[str]:
    """The entity key as a whole token (rule 1)."""
    return re.compile(rf"(?<![0-9A-Za-z_]){re.escape(key)}(?![0-9A-Za-z_])")


@dataclass
class Run:
    log: Log
    token: str
    records: list[Record]
    found_by: list[Record]  # the records the key was on

    @property
    def first(self) -> str:
        return self.records[0].stamp

    @property
    def last(self) -> str:
        return self.records[-1].stamp

    @property
    def machine(self) -> str:
        return self.records[0].machine

    def has_level(self, level: str) -> bool:
        return any(r.level == level for r in self.records)

    def would_print(self, level: str) -> bool:
        """Whether a row at *level* missing from this run says it was not printed (rule 8).

        A logger passes every level at or above its threshold, so a run that
        printed anything at *level* or below had *level* switched on -- a run
        of DEBUG lines would have shown an ERROR one.  Asking for a line at the
        same level in the same run, as this did before, made every ERROR row
        unusable in a run that raised none (a failure that was
        logged at DEBUG).

        The same run's lines also say the level is written to this file: a
        handler passes every level at or above its own threshold too, so the
        handler that wrote the run's DEBUG line writes its ERROR line to the
        same file.  Only a filter could send one level elsewhere; where the
        corpus or panda-common has one (``level_filters``), or with
        ``RULE8_FILE_LEVEL=1``, the file must still hold a line at *level* from
        some run.  That condition made a verdict hang on other jobs: an
        extract of one job's lines held no ERROR line, a one-minute
        window held five of other jobs.

        Never in a run or pass a recorded search may have cut at an end of what it searched (``Log.whole``; a pass
        of a mark that spans days is judged by its own records, not its mark's), and never in a run of an
        extract whose mark does not hold the key: the
        grep kept such a run only from its first line holding the key on, so a
        row that prints before it is missing whether it printed or not (a
        getJobs call's ``Start`` read as absent, and the call that sent the
        job was excluded).  A run whose mark holds the key is whole in it.
        """
        if self.log.extract_key and not keyed(self.log.extract_key).search(self.token):
            return False
        if self.log.whole is not None and not self.log.whole(self.machine, self.first, self.last):
            return False
        rank = LEVEL_RANK.get(level)
        if rank is None:
            return False
        if not any(LEVEL_RANK.get(r.level, 99) <= rank for r in self.records):
            return False
        if FILE_LEVEL or level_filters():
            return any(r.level == level for r in self.log.records)
        return True

    def index(self, record: Record) -> int:
        return self.records.index(record)

    def part(self, start: int, end: int) -> "Run":
        """The records ``[start:end]`` as a run of their own: one pass, where a mark holds several."""
        kept = self.records[start:end]
        return Run(self.log, self.token, kept, [r for r in self.found_by if r in kept])

    def describe(self) -> str:
        where = f"{self.log.name}@{self.machine.split('.')[0]}" if self.machine else self.log.name
        return (
            f"{where} token={self.token} {len(self.records)} record(s) "
            f"{self.first} .. {self.last} (key on {len(self.found_by)})"
        )


def runs_for(log: Log, key: str) -> list[Run]:
    """Every run of *log* the key appears in, keyed by the machine and the instance token: one mark on two machines
    is two runs (a file handed in is one machine's)."""
    pattern = keyed(key)
    hits = [r for r in log.records if pattern.search(r.message) or (r.token and pattern.search(r.token))]
    runs: list[Run] = []
    for mark in dict.fromkeys((r.machine, r.token) for r in hits if r.token):
        runs.append(
            Run(log, mark[1], [r for r in log.records if (r.machine, r.token) == mark], [r for r in hits if (r.machine, r.token) == mark])
        )
    return runs
