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

import ast
import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Callable, Mapping, Protocol

from bamboo.querier import mapdump
from bamboo.querier import render as T

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
#: method, for every job.  A tag may hold brackets of its own, one level
#: (``checkTaskStatusJEDI <  < jediTaskID=12345 > >``: seven tags in the corpus), or
#: open one it does not close: ``create_tagged_logger`` writes ``f"{method} < {tag} >"``
#: and three tags begin with ``<`` alone (``insertNewJob < <JediTaskID=12345 idPool=2 >``),
#: so its `` >`` ends the mark -- not the method alone, nor a later ``>`` of the message.
#: A prefix that never closes (LogWrapper's ``put_file_recovery_request <
#: jediTaskID=12345``, one in the corpus) has nothing to end it and reads as the name;
#: it and every other bare mark with a blank are cut again by the map's marks (:func:`cut_mark`)
PREFIX = re.compile(
    r"^(?P<stamp>\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d{3}) (?P<logger>\S+): "
    r"(?P<level>DEBUG|INFO|WARNING|ERROR|CRITICAL)\s+"
    r"(?P<token>[A-Za-z_]\w* < <[^<>\n]* >|[A-Za-z_]\w* <(?:[^<>\n]|<[^<>\n]*>)*>"
    r"|<[^<>\n]*>|\S+) ?(?P<message>.*)$"
)
#: the builders whose mark a line carries as the logger was handed it, blanks and all (``callfacts.LOGGER_MAKERS``):
#: LogWrapper's prefix and MsgWrapper's token.  make_logger and create_tagged_logger write ``method <token>`` and
#: ``method < tag >``, which :data:`PREFIX` reads whole.
_BARE_BUILDERS = frozenset({"LogWrapper", "MsgWrapper"})
#: the dump :func:`_marks_of` read its marks from, and those marks by log
_MARKS: list = []
#: the dump :func:`_heads_of` read its rows from, and those rows by log
_HEADS: list = []


def log_name(path: str) -> str:
    """The log file the map names, which the file at *path* is a copy or a rotation of.

    The map says ``panda-adder.log``; production rotates it into ``panda-adder.log.1`` or
    ``panda-adder.log-20261030.gz``, and a copy handed over may carry the key (``panda-adder.log.<PandaID>``).
    Whatever follows ``.log`` is dropped.
    """
    base = Path(path).name
    at = base.find(".log")
    return base[: at + len(".log")] if at >= 0 else base


def _marks_of(log: str):
    """The map's marks with a blank of the runs printing into *log* (:func:`cut_mark`), as ``(by first word, the
    rest)``: the patterns of a mark whose first word is literal under that word, and ``(literal head, pattern)`` of one
    whose first word holds a hole.  None for a log no such run prints into, or with no map installed."""
    fragment = mapdump.fragment()
    if fragment is None:
        return None
    if not _MARKS or _MARKS[0] is not fragment:
        _MARKS[:] = [fragment, _bare_marks(fragment)]
    return _MARKS[1].get(log)


def runs_into(fragment) -> dict[str, list]:
    """The map's runs printing into each log: the runs a statement prints through, its own and those handed to it
    (``PrintStatement.runs``, ``handed_runs``), put under the files the map follows its logger to (basis ``logger``)."""
    nodes = {n.owner: n for n in getattr(fragment, "function_prints", None) or []}
    out: dict[str, list] = {}
    for node in nodes.values():
        runs = {r.line: r for r in node.runs}
        for st in node.statements:
            if st.log_files_basis != "logger":
                continue
            held = [runs.get(line) for line in st.runs]
            held += [next((r for r in other.runs if r.line == ref.line), None)
                     for ref in st.handed_runs if (other := nodes.get(ref.owner)) is not None]
            for log in st.log_files:
                out.setdefault(log, []).extend(run for run in held if run is not None)
    return out


def _bare_marks(fragment) -> dict[str, tuple[dict[str, list[re.Pattern[str]]], list[tuple[str, re.Pattern[str]]]]]:
    """:func:`_marks_of` for every log: the marks of the runs printing into it (:func:`runs_into`), only a builder's
    that writes the mark bare, with a blank inside it and a literal: ``{} {}`` (a file server's) would cut any two
    words."""
    found: dict[str, set[str]] = {}
    for log, runs in runs_into(fragment).items():
        for run in runs:
            mark = run.mark.strip() if run.builder in _BARE_BUILDERS else ""
            if " " in mark and re.sub(r"\{\}|\s", "", mark):
                found.setdefault(log, set()).add(mark)
    out = {}
    for log, marks in found.items():
        first: dict[str, list[re.Pattern[str]]] = {}
        rest: list[tuple[str, re.Pattern[str]]] = []
        for mark in sorted(marks):
            # a hole is one word, as a mark's value is: one holding a blank leaves the line as PREFIX cut it
            pattern = re.compile(r"\S*".join(re.escape(p) for p in mark.split("{}")) + r"(?=\s|$)")
            word = mark.split(" ", 1)[0]
            if "{}" in word:
                rest.append((word.split("{}", 1)[0], pattern))
            else:
                first.setdefault(word, []).append(pattern)
        out[log] = (first, rest)
    return out


def cut_mark(log: str, token: str, message: str) -> tuple[str, str]:
    """*token* and *message* as :data:`PREFIX` cut a line of the file *log*, cut again where a mark with a blank of a
    run printing into it starts the line and goes past *token* (KNOWN_ISSUES.md K59).

    PREFIX takes the first word for the mark where no bracket ends it, so ``LogWrapper(_logger, f"update_job
    PandaID={job_id} PID={os.getpid()}")`` (pilot_api.py:513) is the mark ``update_job``: every pilot's call, and the
    key in the message.  The map's mark ``update_job PandaID={} PID={}`` reads the line ``update_job PandaID=7389884699
    PID=2649684 Start`` as the mark ``update_job PandaID=7389884699 PID=2649684`` and the message ``Start``.  The
    longest mark that reads the line wins."""
    marks = _marks_of(log)
    if not marks or not token:
        return token, message
    first, rest = marks
    text = f"{token} {message}"
    end = len(token)
    for pattern in [*first.get(token, ()), *(p for head, p in rest if text.startswith(head))]:
        if (m := pattern.match(text)) and m.end() > end:
            end = m.end()
    if end == len(token):
        return token, message
    return text[:end], text[end + 1:] if text[end:end + 1] == " " else text[end:]


def head_word(pattern: str) -> str:
    """The literal first word of a map row's pattern, up to its first blank: ``failed`` of ``failed\\ to\\ get\\
    taskSpec\\ for\\ jediTaskID=``; "" where a hole or another construct comes before the blank, or none comes.  A
    line its logger writes with no mark starts with that word, which :data:`PREFIX` takes for the mark (K70)."""
    head, blank, _ = pattern.partition("\\ ")
    if not blank or not head or re.search(r"[][(){}?*+|^$.]", re.sub(r"\\.", "", head)):
        return ""
    return re.sub(r"\\(.)", r"\1", head)


def _heads_of(log: str) -> dict[str, list[re.Pattern[str]]]:
    """The map's rows printing into *log* with no prefix -- a logger that writes no mark -- by their pattern's first
    word (:func:`head_word`); empty for a log no such row prints into, or with no map installed."""
    fragment = mapdump.fragment()
    if fragment is None:
        return {}
    if not _HEADS or _HEADS[0] is not fragment:
        found: dict[str, dict[str, list[re.Pattern[str]]]] = {}
        for node in fragment.function_prints:
            for st in node.statements:
                if st.log_files_basis != "logger":
                    continue
                for row in st.rows:
                    if row.prefix or not row.pattern or not (word := head_word(row.pattern)):
                        continue
                    for name in st.log_files:
                        found.setdefault(name, {}).setdefault(word, []).append(re.compile(row.pattern, re.S))
        _HEADS[:] = [fragment, found]
    heads: dict[str, dict[str, list[re.Pattern[str]]]] = _HEADS[1]
    return heads.get(log, {})


def unmarked(log: str, token: str, text: str) -> bool:
    """Whether *token*, as :data:`PREFIX` cut a line of the file *log*, is no mark but the first word of a line its
    logger writes with none: a row of the map printing into *log* with no prefix reads the line as written from that
    word on.  ContentsFeederThread's ``self.logger`` is the module's logger, not wrapped yet where a task's spec was
    not got: ``failed to get taskSpec for jediTaskID=...`` (ContentsFeeder.py:157) is cut as the mark ``failed``, and
    every such line of a machine's log, of every task over days, would be one run of it (KNOWN_ISSUES.md K70)."""
    heads = _heads_of(log).get(token)
    if not heads:
        return False
    line = f"{token} {text}"
    return any(p.match(line) for p in heads)


TRACEBACK = "Traceback (most recent call last):"
FRAME = re.compile(r'^\s*File "(?P<path>[^"]+)", line (?P<line>\d+), in (?P<func>\S+)')
EXCEPTION = re.compile(r"^(?P<type>[A-Za-z_][\w.]*): (?P<message>.*)$")
PACKAGES = ("pandaserver/", "pandajedi/")
#: the first line of a log made by grepping a key (``merge_wassh.py --about``)
EXTRACT = re.compile(r"^# extract by (?P<key>\S+):")

LEVEL_RANK = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40, "CRITICAL": 50}

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
    #: the traceback's own line, where the corpus holds the frame's source at another line (``line``): the run's code
    #: was another version than the pin's (``_place``)
    moved_from: int | None = None
    #: where the frame is not placed so: the lines of its function holding the printed source, none or more than one
    printed_at: tuple[int, ...] = ()

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


@lru_cache(maxsize=None)
def _corpus_file(target: Path) -> tuple[list[str], dict[str, list[tuple[int, int]]]]:
    """*target*'s lines and, by name, the spans of its defs."""
    text = target.read_text(errors="replace")
    spans: dict[str, list[tuple[int, int]]] = {}
    try:
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                spans.setdefault(node.name, []).append((node.lineno, node.end_lineno or node.lineno))
    except SyntaxError:
        pass
    return text.splitlines(), spans


def _place(path: str, line: int, func: str, source: str) -> tuple[str, bool | None, int, int | None, tuple[int, ...]]:
    """``(rel, same_source, line, moved_from, printed_at)``: the frame's corpus path, whether the corpus holds the
    source line the traceback printed, the corpus's line of it, and the traceback's own line where that differs.  The
    pin is the code the case is read against, and a run's traceback prints the lines of the version it ran: where the
    corpus's line holds other text, the one line of *func* -- every def of its name in the file -- holding the printed
    text is the frame's, the traceback's line kept to say so; none, or more than one, leaves the frame not the corpus's
    source, with those lines (``printed_at``) (KNOWN_ISSUES.md K33)."""
    for package in PACKAGES:
        at = path.rfind(package)
        if at < 0:
            continue
        rel = path[at:]
        target = CORPUS / rel
        if not target.exists():
            return "", None, line, None, ()
        lines, spans = _corpus_file(target)
        printed = source.strip()
        here = lines[line - 1].strip() if 0 < line <= len(lines) else ""
        if here == printed or not printed:
            return rel, here == printed, line, None, ()
        found = sorted({n for lo, hi in spans.get(func, []) for n in range(lo, hi + 1) if lines[n - 1].strip() == printed})
        if len(found) == 1:
            return rel, True, found[0], line, ()
        return rel, False, line, None, tuple(found)
    return "", None, line, None, ()


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
            rel, same, at, moved, printed_at = _place(path, line, m["func"], source)
            frames.append(Frame(path, at, m["func"], source, rel, same, moved, printed_at))
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
    #: a recorded search's test of a run or a pass, ``(machine, mark, first stamp, last stamp) -> bool``: false where it
    #: may go on past what was searched for the strings that return the mark's records, and rule 8 reads no row missing
    #: from it (None in a file handed in)
    whole: Callable[[str, str, str, str], bool] | None = None
    #: why a run ``whole`` is false for may go on past what was searched, ``(machine, mark, first stamp, last stamp) ->
    #: str``, empty where it is whole: said on the run's line (:meth:`Run.describe`; None in a file handed in)
    edges: Callable[[str, str, str, str], str] | None = None
    #: ``(machine, mark) -> why`` for the marks whose records are the key's alone, the rest not asked: a run of one is
    #: read as cut at both ends, and rule 8 reads no row missing from it (a recorded search asked for S1's movers)
    partial: dict[tuple[str, str], str] = field(default_factory=dict)
    #: ``(machine, mark) -> method`` for the marks asked for the method's own rows and every line at WARNING and above
    #: (``recorded._read``): a row of another function below WARNING is not searched in a run of one
    narrowed: dict[tuple[str, str], str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    #: the machines a recorded listing says hold no file of the log (``no such log``): none of its runs is there (empty
    #: in a file handed in)
    without: frozenset[str] = frozenset()
    #: the machine an ask named that no recorded list has (``not listed``): the log was asked of every machine in its
    #: place (K13), so nothing is known of a file of it there
    unlisted: frozenset[str] = frozenset()
    #: the machines a recorded answer names JEDI's: their clocks were measured against the database's on some only, so
    #: a time on them is the database's within the windows' slack, not to the second (``recorded.JEDI_CLOCKS``; empty
    #: in a file handed in)
    jedi: frozenset[str] = frozenset()

    @property
    def name(self) -> str:
        """The log file the map names, which this file is a copy or a rotation of (:func:`log_name`)."""
        return log_name(self.path)

    @property
    def covers(self) -> tuple[str, str]:
        return (self.records[0].stamp, self.records[-1].stamp) if self.records else ("", "")

    def covers_time(self, stamp: str) -> bool:
        first, last = self.covers
        return bool(first) and first[: len(stamp)] <= stamp <= last[: len(stamp)] + "~"


def read_log(path: str) -> Log:
    records: list[Record] = []
    extract_key = ""
    name = log_name(path)
    for lineno, line in enumerate(Path(path).read_text(errors="replace").splitlines(), 1):
        if lineno == 1 and (m := EXTRACT.match(line)):
            extract_key = m["key"]
        elif STAMP.match(line):
            m = PREFIX.match(line)
            if m:
                token, message = cut_mark(name, m["token"], m["message"])
                records.append(Record(path, lineno, m["stamp"], m["logger"], m["level"], token, message))
            else:
                records.append(Record(path, lineno, line[:23], "", "", "", line[24:]))
        elif records:
            records[-1].continuation.append(line)
    return Log(path, records, extract_key)


#: (from, to) in the statuslog's form, ``YYYY-MM-DD HH:MM:SS``
Window = tuple[str, str]
#: the functions an asker reads runs of, by the name ``which_ran.of_mover`` reads in a method's mark, each with the fixed
#: text its own rows below WARNING start with (``rows.mark_heads``; empty where one may start with a hole): its mark
#: is asked for those rows and every line at WARNING and above
Owners = Mapping[str, tuple[str, ...]]


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

    def log(
        self, name: str, key: str, window: Window | None = None, *, machine: str | None = None, owners: Owners | None = None,
        strings: tuple[str, ...] = (), whole: bool = False,
    ) -> Log:
        """*owners*: the functions the asker reads runs of, as ``which_ran.of_mover`` names them, or None for any.
        *strings*: what to ask for, where not the key alone (F3's marks holding the key).  *whole*: the owners' marks
        are asked whole, none read as cut for the calls sharing it nor asked for the method's own rows alone -- for an
        asker that reads a run at a row's moment (H2, F1), whose must rows those would leave unread (KNOWN_ISSUES.md
        K13)."""
        ...


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

    def log(
        self, name: str, key: str, window: Window | None = None, *, machine: str | None = None, owners: Owners | None = None,
        strings: tuple[str, ...] = (), whole: bool = False,
    ) -> Log:
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
    #: a line its logger writes with no mark (:func:`unmarked`), a run of its own: nothing says which other lines are
    #: of its pass (KNOWN_ISSUES.md K70)
    alone: bool = False

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

    def holds(self, func: str | None, level: str) -> bool:
        """Whether the run's records hold every line *func* prints through its mark at *level*: not where the door
        asked the mark for another method's rows (``Log.narrowed``) and the level is below WARNING; *func* None for a
        reader that does not say whose row it reads."""
        method = self.log.narrowed.get((self.machine, self.token))
        return method is None or (func is not None and func == method) or LEVEL_RANK.get(level, 0) >= LEVEL_RANK["WARNING"]

    def would_print(self, level: str, func: str | None = None) -> bool:
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
        corpus or panda-common has one (``level_filters``), the file must still
        hold a line at *level* from some run.  Asked everywhere, as rule 8 did
        before, that condition made a verdict hang on other jobs: an
        extract of one job's lines held no ERROR line, a one-minute
        window held five of other jobs.

        Never in a run or pass a recorded search may have cut at an end of what it searched (``Log.whole``; a pass
        of a mark that spans days is judged by its own records, not its mark's), and never in a run of an
        extract whose mark does not hold the key: the
        grep kept such a run only from its first line holding the key on, so a
        row that prints before it is missing whether it printed or not (a
        getJobs call's ``Start`` read as absent, and the call that sent the
        job was excluded).  A run whose mark holds the key is whole in it.  Nor in a run of a mark whose other records
        were not asked (``Log.partial``), nor for a row of *func* a run asked for another's rows does not hold
        (:meth:`holds`).  Nor in a line read alone (:attr:`alone`): its pass's other lines are not under it.
        """
        if self.alone:
            return False
        if self.log.extract_key and not keyed(self.log.extract_key).search(self.token):
            return False
        if (self.machine, self.token) in self.log.partial:
            return False
        if not self.holds(func, level):
            return False
        if self.log.whole is not None and not self.log.whole(self.machine, self.token, self.first, self.last):
            return False
        rank = LEVEL_RANK.get(level)
        if rank is None:
            return False
        if not any(LEVEL_RANK.get(r.level, 99) <= rank for r in self.records):
            return False
        if level_filters():
            return any(r.level == level for r in self.log.records)
        return True

    def index(self, record: Record) -> int:
        return self.records.index(record)

    def part(self, start: int, end: int) -> "Run":
        """The records ``[start:end]`` as a run of their own: one pass, where a mark holds several."""
        kept = self.records[start:end]
        return Run(self.log, self.token, kept, [r for r in self.found_by if r in kept], self.alone)

    def said(self) -> T.Said:
        """The run's line as a record: where, its mark, its records' count and stamps; and, for a run a recorded search
        may have cut at an end of what it searched, which end and why (``Log.edges``): its records are not all the run's.
        Never for a line read alone (:attr:`alone`): it is all its run, and its first word, no mark, was neither
        searched for nor asked whole (the door, UMK), so what was searched for that word cuts nothing of it."""
        where = T.Said("log-at-machine", {"log": self.log.name, "machine": self.machine.split(".")[0]}) if self.machine else self.log.name
        cut = ""
        if not self.alone:
            cut = self.log.edges(self.machine, self.token, self.first, self.last) if self.log.edges is not None and self.machine else ""
            cut = self.log.partial.get((self.machine, self.token), cut)
        method = self.log.narrowed.get((self.machine, self.token))
        said_cut: T.Said | str = T.Said("read-as-cut", {"cut": cut}) if cut else ""
        if self.alone:
            said_cut = T.Said("read-alone", {"then": said_cut})
        return T.Said("run-line", {
            "where": where, "token": self.token, "n": len(self.records), "first": self.first, "last": self.last,
            "found": len(self.found_by), "cut": said_cut,
            "asked": T.Said("asked-for-its-rows", {"method": method}) if method else "",
        })

    def describe(self) -> str:
        """:meth:`said`, rendered."""
        return T.text(self.said())


def runs_for(log: Log, key: str) -> list[Run]:
    """Every run of *log* the key appears in, keyed by the machine and the instance token: one mark on two machines
    is two runs (a file handed in is one machine's).  A line its logger writes with no mark is a run of its own
    (:func:`unmarked`, ``Run.alone``): the word :data:`PREFIX` takes for its mark is its text's first, which joins the
    lines of every pass and object (KNOWN_ISSUES.md K70)."""
    pattern = keyed(key)
    heads = _heads_of(log.name)

    def alone(r: Record) -> bool:
        return r.token in heads and unmarked(log.name, r.token, r.text)

    # the pattern finds the key only where its text is, so a record without the text is passed over without it (most
    # of a day-long mark's records: MT-ready-stuck's S1, 62% of its time in this line)
    hits = [r for r in log.records if (key in r.message and pattern.search(r.message)) or (key in r.token and pattern.search(r.token))]
    # each mark's records in one pass over the log, not one pass a mark: a log of thousands of marks was read again for
    # each (MT-running-stuck's day-long marks: 879 of its 928 s under cProfile)
    by_mark: dict[tuple[str, str], list[Record]] = {}
    for r in log.records:
        if not alone(r):
            by_mark.setdefault((r.machine, r.token), []).append(r)
    found: dict[tuple[str, str, int], list[Record]] = {}
    for r in hits:
        found.setdefault((r.machine, r.token, id(r) if alone(r) else 0), []).append(r)
    return [Run(log, mark[1], held if mark[2] else by_mark[mark[:2]], held, bool(mark[2]))
            for mark, held in found.items() if mark[1]]
