"""The door over the recorded answers: PanDA's log search as the recording step fetched it, read and never asked.

The querier never calls the API itself (tmaeno, 2026-10-14).  The recording step (bamboo-e1's field test) fetches
what an asks file names, together by machine and file, and writes one record per ask.  A run given ``--records``
reads only those.  An ask with no record is gathered with the other asks of its step, written to ``--asks``, and the
run stops; the recording step fetches them and the run starts again.  An ask that depends on an answer comes a round
later: the key first, then the marks that do not hold the key, then the ends of the files.

Formats: analyzer-handover/stage2-record-format.md, as bamboo-e1 answered it (live-harness/notes/to-querier-15.md).  A
record is keyed by its ask, compared as JSON with sorted keys (:func:`key_of`); the recording step copies the ask
unchanged.  It writes one file per round (``records-r<n>.jsonl``); ``--records`` takes the files or their directory.
A later round may write an ask's record again with ``replaces``: n -- round n's records and what a continuation read
-- which is read instead of round n's; any other second record of an ask stops the run.

What is read from a record:

* every whole record (``head``) of every file read, with its machine, file and offset; its span is
  ``machine:file@offset``.  A fragment (``head`` false: a line inside a record whose head was not returned) is only
  counted;
* where it searched.  A machine's list of logs is ``listed``, or says why its files are not known (``not listed``,
  ``list cut``, ``list failed``: nothing on it was searched), or ``no such log`` (it holds no file of the name: there
  is nothing on it to search).  Each file has ``covers``: from the rotation before it to its own, plus ``margin_s``
  (a rotated file is written to until its process stops: up to 146 s after, measured 2026-10-14), so neighbours
  overlap; and its ``status``: ``answered``, or ``failed`` / ``unreadable`` (not read).  What was read is kept for
  each string of the record (:class:`_File`), and a file is one whichever records listed it, by its name or the
  rotation it starts at (:func:`_one_file`): today's ``panda-x.log``, read to the time it was asked, is tomorrow's
  ``panda-x.log-<date>.gz``.  A
  second *t* is searched on a machine for a string when a file whose start is known covers it and every file covering
  it was read through it for the string, by a record of a string it holds -- in an overlap, one file read and its
  neighbour failed is not searched.  ``truncated`` (1 MB, after the recording step halved the strings) reads a file
  for the strings its record names in ``read`` (each cut string's stretches of the covers; another string of the ask
  was read over all of them), else where every string of the ask was read, ``searched`` (closed stretches in the
  covers' form; one a cut ends stops the second before the cut string's last record's), else up to its last record
  returned -- whichever string's it is, so a string cut earlier is read past its cut; a second of the covers is
  counted read only inside one.  A file with no margin (today's: read to the time it
  was asked, or over the asks' windows to where its read stopped, ``covers_to``) is read up to a second before its
  end: a record stamped just before can be written after.  A file whose
  start is not known (the oldest kept) covers every second before its end but makes none searched.  ``covers`` null
  (a tail holding no stamp) counts for nothing.  What of the window is not searched for the key (or what the asker
  names in its place) is :attr:`Log.holes`: a file another mark's ask cut is not a hole of the key's search;
* the service of each machine: a log from JEDI's machines says that their clocks were not all measured
  (:data:`JEDI_CLOCKS`);
* ``cut`` (more than 200 lines after the match): the record's traceback and holes are not read.

A run is the records of one machine with one mark (``runs_for``).  The search for the key returns every record of a
run whose mark holds the key; a mark that does not (the setupper's stamp) is asked for on its own, where a record of
the key under it lies within :data:`EDGE` of the window -- one the calls share, by the map, only within 2 EDGE of those
records, one ask a machine (KNOWN_ISSUES.md K60).  A run whose part
near the window (its records split where they stop for :data:`EDGE` or more, the pieces that reach the window +- EDGE)
begins or ends within EDGE of an end of what was searched on its machine may go on past it.  Where its mark holds the
key, the window is widened past that end, once: the end is one the asks over the window itself searched, so a run that
a widened ask's record (or a wider ask's) shows reaching past them is not widened again.  A mark that does not hold the
key is shared -- a periodic method's, a pass every few minutes, one run as far as any record reaches and asked EDGE
further back each round (the harness's round 18, 2026-11-03), or the setupper's stamp over its jobs -- and its run is
not widened: the pass that decides sits within a second of its row (MT-toretry-late, rescueTime), so a candidate whose
window holds no line of the key under such a mark stays unexamined (2026-11-04).  A function whose pass prints the key
only farther before its write than EDGE would need it; the map is to say which (not read yet).  Not past an end inside
the window (a gap a new ask does not fill) nor past today's file's end (asking
again only reads it later).  A second past the end is enough because the recording step is the other half of the
rule: a file whose covers meet the window is read whole, and a string cut at 1 MB is asked again EDGE past each end of
the window (``bamboo.scripts.panda.fieldtest.record.CONTINUE_SLACK``, a copy of EDGE a test holds equal), so the run
is searched EDGE past the end it lay near.  A run, or a pass of one (``Run.part``), still that near an end is read as cut there
(:attr:`Log.whole`; rule 8 reads no row missing from it): each by its own first and last record, and by what was
searched for the strings that return its records -- the key's for a mark that holds it, its mark's own ask for one
that does not, nothing for one not asked -- so the passes inside
a mark that spans days are whole.  Once, because a mark such as a task's ``jediTaskID`` spans days, and widening at
every end would end only at the oldest file kept.  EDGE is the windows' slack, not a measured gap: a run whose records stop for longer
than it across a rotation is read as whole while it is not (KNOWN_ISSUES.md K7).

An ask with no record of its own is answered from the records of other asks of its log (``RecordedLogs._held``)
where they settle each of its strings over its window on each of its machines (:func:`_settled`) -- what a new ask
would read no further.  The recording step reads every file whose covers meet the window, a rotated one whole, so a
record's window is not the end of what it searched: its files' covers are (the harness's Ja, 2026-11-04).  A line
holding one of the ask's strings holds one of a record's, so the record returned it: each record is read for the
records whose lines hold the ask's strings (an ask that now names a mark with its tag, or drops a mark, needs no new
round).  Its machines are the one it names, or every machine a record of the log asked of every machine names: a record
of one machine says nothing of the others.  A second is settled for a string where it was searched for it -- by those
records alone, so a file a record does not list, its predecessor's margin at a file's start or its successor's at a
rotated file's end, leaves the second to a new ask, and a cut record that says nothing of where its strings were read
answers only an ask of its own strings -- or where the string itself was cut and the second lies inside its record's
window +- EDGE: the recording step reads a cut string on from where it stopped there (``--again``), or cannot, and an
ask of it would stop there too.  An ask of a string that holds it and more reads fewer lines a second and may pass,
so it is asked anew (round 5's getTasksToBeProcessed_JEDI marks, cut where one call wrote thousands of its callee's
lines in a second, asked for the method's own rows, ``_narrowed``).  A machine a record says holds no file of the log
is answered with nothing; one the every-machine records do not name is not in list_logs, and the run is looked for on
every machine, as where a record of the machine says so.  Such an answer is not the window's own: no run is widened from
its ends, as a new ask's would be (KNOWN_ISSUES.md K13).  The log's notes say which records answered.
"""
from __future__ import annotations

import datetime
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from bamboo.codemap.models import RUN_KEY_TIME
from bamboo.querier import mapdump
from bamboo.querier.records import (
    PREFIX,
    STAMP,
    Log,
    Owners,
    Record,
    Unrecorded,
    Window,
    cut_mark,
    keyed,
    log_name,
    runs_into,
)

EDGE = datetime.timedelta(minutes=10)
#: the windows are on the database's clock (statuslog, as_of); a log's stamps are its machine's.  The server's statuslog rows fell inside their runs' log times to the second (bamboo-e1,
#: 2026-10-13, 14 rows); JEDI's clocks were measured against the database on some of its machines only (2026-10-14)
JEDI_CLOCKS = "the windows take its log clock as the database's within their slack, measured on some of JEDI's machines only"
_SECOND = datetime.timedelta(seconds=1)


def key_of(ask: dict) -> str:
    return json.dumps(ask, sort_keys=True, separators=(",", ":"))


def _alike(ask: dict) -> tuple:
    """What an ask and a wider one share: all but the window, the machine and the strings."""
    return ask["name"], ask["service"], ask.get("whole_records")


def _t(text: str) -> datetime.datetime:
    return datetime.datetime.fromisoformat(text[:19])


def _text(t: datetime.datetime) -> str:
    return f"{t:%Y-%m-%d %H:%M:%S}"


@dataclass(frozen=True)
class Ask:
    """One ask of the logs, in the canonical form its record is keyed by."""

    name: str
    strings: tuple[str, ...]
    window: Window
    machine: str | None = None
    #: always "unsettled" here: the recording step asks the machines whose list_logs holds the name
    #: (KNOWN_ISSUES.md K8)
    service: str = "unsettled"

    def canonical(self) -> dict:
        return {
            "name": self.name,
            "service": self.service,
            "machine": self.machine,
            "strings": sorted(set(self.strings)),
            "window": [self.window[0][:19], self.window[1][:19]],
            "whole_records": True,
        }

    @property
    def key(self) -> str:
        return key_of(self.canonical())


@dataclass
class _File:
    """One file as one record read it: what it covers (start None: not known), through when it was read (None: not
    read), and for each string of the record it was read for, how far."""

    machine: str
    start: datetime.datetime | None
    end: datetime.datetime
    read_to: datetime.datetime | None
    name: str = ""
    #: why it was read only to ``read_to``, for a run's line (``_edges_in``); empty for a rotated file read whole
    stop: str = ""
    #: string -> (read up to, the closed stretches it was read in -- None: all of it up to there --, why it stops
    #: early): a cut string as the record says (``_read_for``), every other one as the file
    strings: dict[str, tuple[datetime.datetime | None, tuple | None, str]] = field(default_factory=dict)

    def covers(self, t: datetime.datetime) -> bool:
        return (self.start is None or self.start <= t) and t <= self.end

    def read(self, t: datetime.datetime, wanted: str) -> bool:
        """Whether second *t* of it was read for the string *wanted*: by a string of the record *wanted* holds, up to
        where that string was read, and inside its stretches where the record names them."""
        return any(s in wanted and _read_in(t, *how[:2]) for s, how in self.strings.items())

    def why(self, t: datetime.datetime, wanted: str) -> str:
        """Why *t* was not read for *wanted* here: the file's stop, or that of a string of the record *wanted* holds;
        empty where this record read it for no such string."""
        if self.read_to is None:
            return self.stop
        return next((stop or f"{self.name} was not read through it" for s, (to, stretches, stop) in self.strings.items()
                     if s in wanted and not _read_in(t, to, stretches)), "")


def _read_in(t: datetime.datetime, to: datetime.datetime | None, stretches: tuple | None) -> bool:
    return to is not None and t <= to and (stretches is None or any(a <= t <= b for a, b in stretches))


def _one_file(a: tuple[str, datetime.datetime | None], b: tuple[str, datetime.datetime | None]) -> bool:
    """Whether two entries ``(name, start)`` of one machine's log, whichever records listed them, are one file: by
    its name, or by the rotation it starts at -- today's ``panda-x.log``, read to when it was asked, is tomorrow's
    ``panda-x.log-<date>.gz`` -- which the recording step reads off a file's time, a few seconds apart between rounds
    (N-rec's panda-setupper.log-20261008.gz from 07:22:19 and from 07:22:16); a log rotates once a day."""
    return a[0] == b[0] or (a[1] is not None and b[1] is not None and abs(a[1] - b[1]) <= EDGE)


def _by_file(entries: list[_File]) -> list[list[_File]]:
    """*entries* (of one machine's log) grouped by the file they are (:func:`_one_file`)."""
    groups: list[list[_File]] = []
    for f in entries:
        group = next((g for g in groups if any(_one_file((f.name, f.start), (h.name, h.start)) for h in g)), None)
        if group is None:
            groups.append([f])
        else:
            group.append(f)
    return groups


class _Gathered:
    """The records and the files read for one log, from every record read for it, each log record once."""

    def __init__(self) -> None:
        self.records: dict[tuple[str, str, int], Record] = {}
        self.files: list[_File] = []
        self.silent: dict[str, str] = {}  # machine -> why its files are not known
        self.listed: set[str] = set()
        self.without: set[str] = set()  # machines that hold no file of the name
        self.services: dict[str, str] = {}  # machine -> its service, as the record says, for the machines read
        self.asked = False  # whether any record named a machine to ask
        self.fragments = 0
        self.rank: dict[tuple[str, str], datetime.datetime] = {}  # (machine, file) -> its end, to order the files
        #: (machine, last second read) of the files that go on past it: a rotated file read whole (the next file
        #: follows), or today's file whose read stopped before a record (``covers_to`` "end_stamp", the harness's
        #: a02e8fd: it is read over the asks' windows); past such an end there is more to widen into.  Today's file
        #: read to the time it was asked (``covers_to`` "asked", or no ``covers_to`` and no margin) has none: asking
        #: again only reads it later
        self.continued_ends: set[tuple[str, datetime.datetime]] = set()

    def add(self, record: dict) -> None:
        # a wider record read for a narrower ask (RecordedLogs._within): its records holding the ask's strings, each
        # line searched once for all of them; a cut file is still read as far as the whole record read it -- for the
        # strings the ask needs, where it says so
        only, needed = record.get("only"), record.get("needed")
        held = None if only is None else re.compile("|".join(map(re.escape, only))) if only else re.compile(r"(?!)")
        for machine, got in record.get("machines", {}).items():
            self.asked = True
            status = got.get("status")
            if status == "no such log":
                self.without.add(machine)
                continue
            if status != "listed":
                self.silent.setdefault(machine, f"{status}" + (f" ({got['error']})" if got.get("error") else ""))
                continue
            self.listed.add(machine)
            if got.get("service"):
                self.services[machine] = got["service"]
            for f in got.get("files", []):
                last = None
                for r in f.get("records", []):
                    kept = held is None or any(map(held.search, r["lines"]))
                    if not r.get("head"):
                        self.fragments += kept
                        continue
                    if kept:
                        self.records.setdefault((machine, f["file"], r["offset"]), _record(machine, f["file"], r))
                    last = r
                last = _record(machine, f["file"], last).stamp if last is not None else None
                if not f.get("covers"):
                    continue
                start, end = f["covers"]
                end = _t(end)
                self.rank[(machine, f["file"])] = end
                name = Path(f["file"]).name
                if f.get("status") != "answered":
                    read_to, stop = None, f"{name} was {f.get('status') or 'not read'}"
                elif (to := f.get("covers_to") or ("rotation" if f.get("margin_s") else "asked")) == "list":
                    read_to, stop = None, f"{name} was listed, not read"
                else:
                    read_to = end if f.get("margin_s") else end - _SECOND
                    stop = ("" if to == "rotation" else f"{name} was read to the time it was asked" if to == "asked"
                            else f"the read of {name} stopped at {_text(end)}, past the window asked")
                    if to in ("rotation", "end_stamp"):
                        self.continued_ends.add((machine, read_to))
                strings: dict[str, tuple[datetime.datetime | None, tuple | None, str]] = {}
                for s in (needed if needed is not None else record.get("ask", {}).get("strings", ())) if read_to is not None else ():
                    cut, searched = _read_for(f, [s])
                    if not cut:
                        strings[s] = (read_to, None, stop)
                    elif searched is None:
                        # a record that says nothing of where its strings were read: up to its last record, whichever
                        # string's it is -- another's may carry past a cut one's (an ask is not answered so, _settled)
                        strings[s] = (_t(last) if last else None, None, f"the answer of {name} was cut at 1 MB after its last record")
                    else:
                        strings[s] = (read_to, searched, f"the answer of {name} was cut at 1 MB, and its record says every "
                                                         "string was read only so far")
                self.files.append(_File(machine, _t(start) if start else None, end, read_to, name, stop, strings))

    def log(self, name: str) -> Log:
        order = sorted(self.records.values(), key=lambda r: (r.machine, self.rank.get((r.machine, r.file), datetime.datetime.max), r.offset))
        return Log(name, order)

    def searched(self, machine: str, t: datetime.datetime, wanted: tuple[str, ...]) -> bool:
        """Whether second *t* was searched on *machine* for every string of *wanted*: a file whose start is known covers
        it, and each file covering it was read through it for each string, by one record or another."""
        covering = [f for f in self.files if f.machine == machine and f.covers(t)]
        return any(f.start is not None for f in covering) and all(
            any(f.read(t, w) for f in entries) for entries in _by_file(covering) for w in wanted)

    def stretches(self, machine: str, wanted: tuple[str, ...], lo: datetime.datetime | None = None,
                  hi: datetime.datetime | None = None) -> list[tuple]:
        """The machine's seconds searched for *wanted* from *lo* to *hi* (all its files' span when not given), as
        closed stretches."""
        files = [f for f in self.files if f.machine == machine]
        if not files:
            return []
        lo = lo or min((f.start for f in files if f.start is not None), default=min(f.end for f in files))
        hi = hi or max(f.end for f in files)
        cuts = {lo, hi + _SECOND}
        for f in files:
            for t in (f.start, f.end + _SECOND, f.read_to + _SECOND if f.read_to else None,
                      *(x for to, stretches, _ in f.strings.values()
                        for x in (to + _SECOND if to else None, *(y for a, b in stretches or () for y in (a, b + _SECOND))))):
                if t is not None and lo < t <= hi:
                    cuts.add(t)
        points = sorted(cuts)
        out: list[list] = []
        for a, b in zip(points, points[1:], strict=False):
            if self.searched(machine, a, wanted):
                if out and out[-1][1] + _SECOND == a:
                    out[-1][1] = b - _SECOND
                else:
                    out.append([a, b - _SECOND])
        return [(a, b) for a, b in out]

    def holes(self, window: Window, wanted: tuple[str, ...]) -> list[str]:
        """What of *window* was not searched for *wanted*, per machine: the window and the stretches are closed, in
        seconds."""
        lo, hi = _t(window[0]), _t(window[1])
        out = [f"{m}: {why}" for m, why in sorted(self.silent.items())]
        for machine in sorted(self.listed - set(self.silent)):
            at = lo  # the first second not known to be searched
            for start, end in self.stretches(machine, wanted, lo, hi):
                if start > at:
                    out.append(f"{machine}: {_text(at)} .. {_text(start - _SECOND)}")
                at = end + _SECOND
            if at <= hi:
                out.append(f"{machine}: {_text(at)} .. {_text(hi)}")
        return out


def _settled(records: list[dict], machine: str, wanted: str, asked: list[str], lo: datetime.datetime,
             hi: datetime.datetime) -> bool:
    """Whether *records* (one log's) settle every second from *lo* to *hi* on *machine* for the string *wanted* of an
    ask of the strings *asked*: a file whose start is known covers it, and each file covering it was read there for
    *wanted* or left there to the recording step -- and no file the records do not know may cover it.

    Read for *wanted*: by a string of a record *wanted* holds, as :class:`_Gathered` reads a file, but a cut file that
    says nothing of where its strings were read only by a record of exactly the ask's strings: another string's records
    may carry past a cut one's last (``bamboo.scripts.panda.fieldtest.record``'s doc, 157 of pin m's 315 cut files).
    Left to the recording step: what a cut string stopped short of in its file, inside its record's window +- EDGE,
    where the string is *wanted* itself (or, said of no string, the record's strings are the ask's) -- the step reads
    it again from where it stopped (``--again``), or cannot: an ask of the string would stop there too.  A string that
    holds it and more reads fewer lines a second and may pass, so it is asked (MT-ready-stuck's getTasksToBeProcessed_JEDI
    marks, asked for the method's own rows, ``_narrowed``).  A file a record says is unreadable is unreadable again.
    A file the records do not know: a rotated file's last ``margin_s`` is its successor's first seconds too, and a
    file's first margin its predecessor's last -- unless a known file covers the second besides it (the records of one
    window know every file meeting it; those of others may not)."""
    files: list[dict] = []
    for r in records:
        strings = [s for s in r["ask"]["strings"] if s in wanted]
        exact = sorted(set(r["ask"]["strings"])) == sorted(set(asked))
        near = (_t(r["ask"]["window"][0]) - EDGE, _t(r["ask"]["window"][1]) + EDGE)
        for f in (r.get("machines", {}).get(machine) or {}).get("files", []):
            if not f.get("covers") or not f["covers"][1]:
                continue
            start, end = (_t(f["covers"][0]) if f["covers"][0] else None), _t(f["covers"][1])
            margin = datetime.timedelta(seconds=f.get("margin_s") or 0)
            name = Path(f["file"]).name
            seen = next((x for x in files if any(_one_file((name, start), one) for one in x["as"])), None)
            if seen is None:
                seen = {"as": [], "start": start, "end": end, "margin": margin, "read": [], "left": []}
                files.append(seen)
            seen["as"].append((name, start))
            seen["start"] = min((x for x in (seen["start"], start) if x is not None), default=None)
            seen["end"], seen["margin"] = max(seen["end"], end), max(seen["margin"], margin)
            if f.get("status") == "unreadable":
                seen["left"].append((start or datetime.datetime.min, end))
            if f.get("status") != "answered" or f.get("covers_to") == "list":
                continue
            read_to = end if f.get("margin_s") else end - _SECOND
            for s in strings:
                cut, stretches = _read_for(f, [s])
                if not cut:
                    read = [(start or datetime.datetime.min, read_to)]
                elif stretches is not None:
                    read = sorted((a, min(b, read_to)) for a, b in stretches if a <= min(b, read_to))
                elif exact:
                    stamps = [_t(x["lines"][0]) for x in f.get("records", []) if x.get("head") and STAMP.match(x["lines"][0])]
                    read = [(start or datetime.datetime.min, max(stamps))] if stamps else []
                else:
                    read = []
                seen["read"] += read
                # the string itself cut, as ``read`` says; or, where the record says no string, its strings the ask's
                if cut and ((f.get("read") is not None and s == wanted) or (exact and f.get("read") is None)):
                    ends = [b for _, b in read] or [(start or datetime.datetime.min) - _SECOND]
                    for b, nxt in zip(ends, [a for a, _ in read[1:]] + [read_to + _SECOND], strict=True):
                        a, z = max(b + _SECOND, near[0]), min(nxt - _SECOND, near[1])
                        if a <= z:
                            seen["left"].append((a, z))
    if not files:
        return False
    head = max(x["margin"] for x in files) or EDGE
    points = {lo, hi + _SECOND}
    for x in files:
        for t in (x["start"], x["end"] + _SECOND, x["start"] and x["start"] + head + _SECOND, x["margin"] and x["end"] - x["margin"],
                  *(y for a, b in x["read"] + x["left"] for y in (a, b + _SECOND))):
            if t and lo < t <= hi:
                points.add(t)
    for a in sorted(points)[:-1]:
        covering = [x for x in files if (x["start"] is None or x["start"] <= a) and a <= x["end"]]
        if not any(x["start"] is not None for x in covering):
            return False
        for x in covering:
            if len(covering) == 1 and ((x["margin"] and x["end"] - x["margin"] <= a)
                                       or (x["start"] is not None and a <= x["start"] + head)):
                return False
            if not any(p <= a <= q for p, q in x["read"] + x["left"]):
                return False
    return True


def _needed(strings: list[str], asked: list[str]) -> list[str]:
    """The strings of a record that the strings *asked* of a narrower ask hold: the ones whose lines it reads."""
    return [s for s in strings if any(s in w for w in asked)]


def _stretches(rows: list, top: datetime.datetime = datetime.datetime.max) -> tuple[tuple[datetime.datetime, datetime.datetime], ...]:
    """Closed stretches in a record's text form, a start not known (null) open, an end not known *top* (its file's
    end: the door adds a second to an end)."""
    return tuple((datetime.datetime.min if a is None else _t(a), top if b is None else _t(b)) for a, b in rows)


def _read_for(f: dict, needed: list[str] | None) -> tuple[bool, tuple | None]:
    """``(cut, searched)`` of a file entry for an ask that reads the lines of *needed* of its record's strings (None:
    every one).  Where the entry says what of its covers was read for each cut string (``read``, fieldtest ca43fb9), a
    string not among them was read over the covers: the file is cut for the ask only where a string it needs is, and
    searched where those strings' stretches meet.  An entry without ``read`` is cut for every ask of it, searched as its
    ``searched`` says (None: up to its last record)."""
    if not f.get("truncated"):
        return False, None
    top = _t(f["covers"][1]) if f.get("covers") and f["covers"][1] else datetime.datetime.max
    read = f.get("read")
    if read is None or needed is None:
        return True, _stretches(f["searched"], top) if f.get("searched") is not None else None
    keys = [s for s in needed if s in read]
    if not keys:
        return False, None
    meet = _stretches(read[keys[0]], top)
    for s in keys[1:]:
        meet = tuple(sorted((max(a, c), min(b, d)) for a, b in meet for c, d in _stretches(read[s], top) if max(a, c) <= min(b, d)))
    return True, meet


def _record(machine: str, file: str, r: dict) -> Record:
    head, *rest = r["lines"]
    m = PREFIX.match(head) if STAMP.match(head) else None
    if m:
        token, message = cut_mark(log_name(file), m["token"], m["message"])
        rec = Record(file, 0, m["stamp"], m["logger"], m["level"], token, message, list(rest))
    else:
        rec = Record(file, 0, head[:23], "", "", "", head[24:], list(rest))
    rec.machine, rec.offset, rec.cut = machine, r["offset"], bool(r.get("cut"))
    return rec


class RecordedLogs:
    """The door over the recording step's answers: ``--records`` files, one record per ask."""

    needs_window = True

    def __init__(self, paths: list[str]) -> None:
        self._records: dict[str, dict] = {}
        files = [f for p in paths for f in (sorted(Path(p).glob("records-r*.jsonl")) if Path(p).is_dir() else [Path(p)])]
        #: the files read, a directory's records-r*.jsonl in order (result.json's inputs)
        self.files: list[Path] = files
        by_ask: dict[str, list[tuple[Path, int, dict]]] = {}
        for path in files:
            for n, line in enumerate(path.read_text().splitlines(), 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                by_ask.setdefault(key_of(record["ask"]), []).append((path, n, record))
        for key, got in by_ask.items():
            # a record saying it replaces round n is read instead of the ask's record of that round, in either file
            # order (a later round writes an ask's record again: the earlier records and what a continuation read)
            left = [(p, n, r) for p, n, r in got
                    if not any(o is not r and "replaces" in o and o["replaces"] == r.get("round") for _, _, o in got)]
            if len(left) != 1:
                path, n, _ = (left or got)[-1]
                raise SystemExit(f"{path}:{n}: a second record for the ask {key}: which to read is not this door's choice")
            self._records[key] = left[0][2]
        self._alike: dict[tuple, list[dict]] = {}
        for record in self._records.values():
            self._alike.setdefault(_alike(record["ask"]), []).append(record)
        self._logs: dict[tuple, Log] = {}

    def names(self, wanted: list[str]) -> list[str]:
        return list(dict.fromkeys(wanted))

    def log(
        self, name: str, key: str, window: Window | None = None, *, machine: str | None = None, owners: Owners | None = None,
        strings: tuple[str, ...] = (), whole: bool = False,
    ) -> Log:
        if window is None:
            raise ValueError(f"{name}: the recorded logs are asked over a window")
        at = (name, key, window, machine, None if owners is None else tuple(sorted(owners.items())), tuple(sorted(strings)), whole)
        if at not in self._logs:
            self._logs[at] = self._read(name, key, window, machine, owners, strings, whole)
        return self._logs[at]

    def _get(self, asks: list[tuple[Ask, str]], notes: list[str]) -> list[list[dict]]:
        """The records of each of *asks*, or every one missing as :class:`Unrecorded`.  An ask with no record of its own
        is answered from other asks' (:meth:`_held`), and *notes* says which."""
        got = [(a, why, [record] if (record := self._records.get(a.key)) else self._held(a)) for a, why in asks]
        missing = [(a, why) for a, why, records in got if records is None]
        if missing:
            raise Unrecorded(missing)
        found = [records for _, _, records in got if records is not None]
        notes += [record["within"] for records in found for record in records if "within" in record]
        return found

    def _held(self, ask: Ask) -> list[dict] | None:
        """*ask*'s answer from the records of other asks of its log (the module's doc), or None where they do not settle
        each of its strings over its window on each of its machines (:func:`_settled`).  Its machines: the one it
        names, or every machine a record of the log asked of every machine names; one a record of one machine asked of
        says nothing of the others.  A machine the records name with no file of the log (``no such log``) holds none.
        One no record names, where such a record lists the log's machines, is not in list_logs (``not listed``): the
        asker of its run looks on every machine (:meth:`_read`), as when a record of the machine says so.  The records
        read: each holding a string of the ask with a file on one of its machines that meets its window, for the lines
        holding the ask's strings."""
        want = ask.canonical()
        lo, hi = _t(want["window"][0]), _t(want["window"][1])
        log = self._alike.get(_alike(want), [])
        every = sorted((r for r in log if r["ask"]["machine"] is None), key=lambda r: (r.get("round") or 0, key_of(r["ask"])))
        if want["machine"] is None and not every:
            return None
        machines = [want["machine"]] if want["machine"] is not None else sorted({m for r in every for m in r.get("machines", {})})
        if want["machine"] is not None and every and not any(want["machine"] in r.get("machines", {}) for r in log):
            return [self._answer(every[0], want, {want["machine"]: {"status": "not listed"}})]

        def holds_none(machine: str) -> bool:
            said = [r["machines"][machine].get("status") for r in log if machine in r.get("machines", {})]
            return bool(said) and all(x == "no such log" for x in said)

        if not all(_settled(log, m, w, want["strings"], lo, hi) for m in machines if not holds_none(m) for w in want["strings"]):
            return None
        none = {m: {"status": "no such log"} for m in machines if holds_none(m)}
        out = []
        order = sorted(log, key=lambda r: (r.get("round") or 0, key_of(r["ask"])))
        for r in order:
            mine = {m: got for m, got in r.get("machines", {}).items() if m in machines and m not in none and any(
                f.get("covers") and f["covers"][1] and (not f["covers"][0] or _t(f["covers"][0]) <= hi) and lo <= _t(f["covers"][1])
                for f in got.get("files", []))}
            if mine and any(s in w for s in r["ask"]["strings"] for w in want["strings"]):
                out.append(self._answer(r, want, mine))
        if none:
            if not out:
                # a record that says so, of the machine alone where no record of every machine was asked
                out.append(self._answer(next(r for r in order if none.keys() & r.get("machines", {}).keys()), want, {}))
            out[0]["machines"] = {**out[0]["machines"], **none}
        return out

    @staticmethod
    def _answer(record: dict, want: dict, machines: dict) -> dict:
        """*record* read for the ask *want*: on *machines*, the lines holding its strings, and a note of which record."""
        other = record["ask"]
        only = {} if other["strings"] == want["strings"] else {"only": want["strings"], "needed": _needed(other["strings"], want["strings"])}
        kind = "a wider" if other["window"][0] <= want["window"][0] and want["window"][1] <= other["window"][1] else "another"
        strings = "" if not only else f", {len(other['strings'])} string(s), the lines holding the ask's"
        return {**record, **only, "ask": want, "machines": machines, "within_window": other["window"],
                "within": f"answered from the record of {kind} ask ({other['window'][0]} .. {other['window'][1]}, "
                          f"{other['machine'] or 'every machine'}, round {record.get('round')}{strings})"}

    def _read(self, name: str, key: str, window: Window, machine: str | None, owners: Owners | None = None,
              strings: tuple[str, ...] = (), whole: bool = False) -> Log:
        notes: list[str] = []
        # the key, or what the asker names in its place: marks holding it (F3), whose lines are all a run's
        asked = tuple(strings) or (key,)
        (first,) = self._get([(Ask(name, asked, window, machine), "")], notes)
        unlisted: frozenset[str] = frozenset()
        if machine is not None and any(r.get("machines", {}).get(machine, {}).get("status") == "not listed" for r in first):
            (first,) = self._get([(Ask(name, asked, window), f"{machine} is not in list_logs: every machine asked")], notes)
            notes.append(f"{machine} is not in list_logs, so every machine was asked")
            unlisted = frozenset({machine})
        got = _Gathered()
        # the records of the asks over the window itself, apart: a widening is decided from the ends they searched
        # alone, so neither a widened ask's record nor a wider ask's widens again (once: the module's doc)
        own = _Gathered()

        def gather(record: dict) -> None:
            got.add(record)
            if tuple(record.get("within_window") or record["ask"]["window"]) == tuple(window):
                own.add(record)

        for record in first:
            gather(record)

        # the marks that do not hold the key: their records need not -- each of a key record within EDGE of the window;
        # a rotated file's record holds the key's lines of its whole day, and a run that began hours after the window or
        # ended before it is not what the window asks about (KNOWN_ISSUES.md K13); for an asker that reads some
        # functions' runs alone, a DB proxy method's mark is asked only where it is one of theirs and its run can be made
        # whole by it (_partial), and for the method's own rows and every line at WARNING and above where its rows below
        # WARNING all start with fixed text (_narrowed); for one that reads a run at a row's moment (*whole*), each of
        # theirs whole
        pattern = keyed(key)
        lo_w, hi_w = _t(window[0]) - EDGE, _t(window[1]) + EDGE
        marks: dict[str, set[str]] = {}
        for r in got.records.values():
            if r.token and pattern.search(r.message) and not pattern.search(r.token) and lo_w <= _t(r.stamp) <= hi_w:
                marks.setdefault(r.machine, set()).add(r.token)
        partial = _partial(got, marks, owners, whole)
        marks = {m: kept for m, t in marks.items() if (kept := {x for x in t if (m, x) not in partial})}
        narrowed = {} if whole else _narrowed(marks, owners)
        why = "the records of the key's runs whose mark does not hold the key"
        # a mark the calls share, by the map, only within 2 EDGE of the key's records under it on its machine, the
        # windows of one machine joined where they overlap (KNOWN_ISSUES.md K60)
        share = set() if whole else _calls_share(name, marks)
        own_marks = {m: kept for m, t in marks.items() if (kept := {x for x in t if (m, x) not in share})}
        asks = [(Ask(name, tuple(sorted(_strings(m, t, narrowed, owners))), window, m), why) for m, t in sorted(own_marks.items())]
        near: dict[str, list[tuple[datetime.datetime, datetime.datetime, str]]] = {}
        for m, token in sorted(share):
            stamps = sorted(_t(r.stamp) for r in got.records.values()
                            if (r.machine, r.token) == (m, token) and pattern.search(r.message))
            start, end = max(_t(window[0]), stamps[0] - 2 * EDGE), min(_t(window[1]), stamps[-1] + 2 * EDGE)
            if start <= end:
                near.setdefault(m, []).append((start, end, token))
        for m, spans in sorted(near.items()):
            joined: list[tuple[datetime.datetime, datetime.datetime, set[str]]] = []
            for start, end, token in sorted(spans):
                if joined and start <= joined[-1][1]:
                    joined[-1] = (joined[-1][0], max(joined[-1][1], end), joined[-1][2] | {token})
                else:
                    joined.append((start, end, {token}))
            asks += [(Ask(name, tuple(sorted(_strings(m, tokens, narrowed, owners))), (_text(start), _text(end)), m),
                      why + ", near them: a mark the calls share") for start, end, tokens in joined]
        if share:
            notes.append(f"{len(share)} mark(s) the calls share (the map) asked only within "
                         f"{int(2 * EDGE.total_seconds() // 60)} min of the key's records")
        for records in self._get(asks, notes):
            for record in records:
                gather(record)

        def strings_of(machine: str, token: str) -> tuple[str, ...]:
            # the strings that return a run's records: the key's (or what the asker names in its place) for a mark
            # that holds it, the ask of its own mark on its machine for one that does not
            return asked if not token or pattern.search(token) else tuple(sorted(_strings(machine, {token}, narrowed, owners)))

        # the ends of what was searched: widened past once, where a part of a run that reaches the window lies near one
        # and its mark holds the key; a shared mark's run is read as cut there (the module's doc)
        widen: dict[tuple[str, Window], set[str]] = {}
        shared = 0
        for (m, token), lo, hi in _runs(got, pattern, window, own, strings_of):
            if (m, token) in partial:
                continue
            # not past an end inside the window (a file read only in part, or not read, is not read better by asking
            # again), nor past today's file's end where it was read to the time asked: the run is read as cut there
            before = _text(lo[0] - _SECOND) if lo and lo[2] - lo[0] < EDGE and _text(lo[0] - _SECOND) < window[0] else None
            after = (_text(hi[1] + _SECOND) if hi and hi[1] - hi[2] < EDGE and _text(hi[1] + _SECOND) > window[1]
                     and (m, hi[1]) in own.continued_ends else None)
            if not pattern.search(token):
                shared += before is not None or after is not None
                continue
            if before is not None:
                widen.setdefault((m, (before, window[1])), set()).update(asked)
            if after is not None:
                widen.setdefault((m, (window[0], after)), set()).update(asked)
        if shared:
            notes.append(f"{shared} run(s) under a mark that does not hold the key lie within {int(EDGE.total_seconds() // 60)} "
                         "min of an end of what was searched: a shared mark, not widened, read as cut there")
        why = f"a run within {int(EDGE.total_seconds() // 60)} min of an end of what was searched"
        for records in self._get([(Ask(name, tuple(sorted(s)), w, m), why) for (m, w), s in sorted(widen.items())], notes):
            for record in records:
                got.add(record)
        log = got.log(name)
        log.holes = got.holes(window, asked)
        log.whole = _whole_in(got, strings_of)
        log.edges = _edges_in(got, strings_of)
        log.partial = partial
        log.narrowed = narrowed
        log.without = frozenset(got.without)
        log.unlisted = unlisted
        log.jedi = frozenset(m for m, service in got.services.items() if service == "jedi")
        if partial:
            kinds = sorted(set(partial.values()))
            notes.append(f"{len(partial)} mark(s) read from the key's records alone, as cut: {'; '.join(kinds)}")
        if narrowed:
            methods = sorted(set(narrowed.values()))
            notes.append(f"{len(narrowed)} mark(s) asked for their method's own rows and every line at WARNING and above: {methods}")
        if not got.asked:
            notes.append("no machine lists it: the log is on none")
        if got.without:
            notes.append(f"{sorted(got.without)} hold(s) no file of it")
        if jedi := sorted(m for m, service in got.services.items() if service == "jedi"):
            notes.append(f"JEDI's machine(s) {jedi}: {JEDI_CLOCKS}")
        log.notes = list(dict.fromkeys(notes)) + ([f"{got.fragments} fragment(s) of records whose head was not returned"] if got.fragments else [])
        return log


#: a DB proxy method's mark: the method's name, then its tag (``records.PREFIX``), as ``which_ran.of_mover`` reads it
_METHOD_MARK = re.compile(r"([A-Za-z_]\w*) <")


def _partial(got: _Gathered, marks: dict[str, set[str]], owners: Owners | None, whole: bool = False) -> dict[tuple[str, str], str]:
    """The marks among *marks* (``machine -> the marks not holding the key``) not asked for whole, each with why, for
    an asker that reads the runs of *owners* alone (None: every mark is asked).  A mark naming no method is a worker's,
    any function's of it, and is asked.  A DB proxy method's mark of no owner is not: no run of it is read.  Nor, but
    for an asker that reads a run at a row's moment (*whole*), is an owner's whose key records come in pieces EDGE or
    more apart: the calls share it (``getTasksToExecCommand_JEDI < vo=None label=None >``), so its run is every call in
    the window, which no ask makes whole -- more than 1 MB a file and cut (bamboo-e1's round of 2026-10-26).  Such a
    mark's key records are read as a run cut at both ends.  H2's run of that mark at a row's second is one call, 13
    records in MT-exhausted-late, whose must rows its choice was shown by (2026-11-02)."""
    if owners is None:
        return {}
    out = {}
    for machine, tokens in marks.items():
        for token in tokens:
            method = _METHOD_MARK.match(token)
            if method is None:
                continue
            if method[1] not in owners:
                out[(machine, token)] = "a method no step asked reads"
                continue
            if whole:
                continue
            stamps = sorted(_t(r.stamp) for r in got.records.values() if (r.machine, r.token) == (machine, token))
            if any(b - a >= EDGE for a, b in zip(stamps, stamps[1:], strict=False)):
                out[(machine, token)] = "a mark the method's calls share"
    return out


def _narrowed(marks: dict[str, set[str]], owners: Owners | None) -> dict[tuple[str, str], str]:
    """``(machine, mark) -> method`` for the marks among *marks* asked for the method's own rows: an owner's whose rows
    below WARNING all start with fixed text (``rows.mark_heads``).  The readers of its run read its own rows and the
    tracebacks its handlers print (``dump_error_message`` writes at ERROR), and a row of another function below
    WARNING in it as not searched (``records.Run.holds``).  A call of getTasksToBeProcessed_JEDI is thousands of lines
    its callee ``_make_dicts_tasks_datasets_with_unprocessed_inputs`` writes with its logger, one per task listed
    (381,600 of 382,661 in MT-ready-stuck, the harness's round 5): no reader of the method reads them, and a second of
    them is more than a page, which stops the recorder (to-querier-63)."""
    if owners is None:
        return {}
    out = {}
    for machine, tokens in marks.items():
        for token in tokens:
            method = _METHOD_MARK.match(token)
            if method is not None and owners.get(method[1]):
                out[(machine, token)] = method[1]
    return out


#: the dump :func:`_calls_share` read its marks from, and those marks by log
_SHARES: list = []
_TIME = r"\d{4}-\d\d-\d\d[/ T]\d\d:\d\d:\d\d(?:\.\d+)?"


def _calls_share(log: str, marks: dict[str, set[str]]) -> set[tuple[str, str]]:
    """The ``(machine, mark)`` among *marks* that every map mark of a run printing into *log* reading it says the calls
    share: none of the mark's holes holds a column that is a table's identity, and it is no time stamp
    (:func:`_share_index`).  A JEDI worker's ``process <host_pid-thread>``, ``getTasksToExecCommand_JEDI < vo=None
    label=None >``, AtlasAnalWatchDog's ``#ATM #KV doForPreStaging label=user``; not ``< PandaID=... >`` nor a
    stamp.  None with no map installed, or for a mark no map mark reads."""
    fragment = mapdump.fragment()
    if fragment is None:
        return set()
    if not _SHARES or _SHARES[0] is not fragment:
        _SHARES[:] = [fragment, _share_index(fragment)]
    index = _SHARES[1].get(log, [])
    out = set()
    for machine, tokens in marks.items():
        for token in tokens:
            hits = [shared for rx, shared in index if rx.fullmatch(token.strip())]
            if hits and all(hits):
                out.add((machine, token))
    return out


def _share_index(fragment) -> dict[str, list[tuple[re.Pattern[str], bool]]]:
    """For every log, each map mark of a run printing into it (``records.runs_into``) as a pattern of the token a line
    carries -- ``method < tag >`` for create_tagged_logger and make_logger, ``<stamp>`` for a MsgWrapper's stamp, the
    mark itself for the others, a hole any text -- and whether the calls share it."""
    ident = {a for t in getattr(fragment, "tables", None) or [] for a in t.identity_attrs or []}
    out: dict[str, list[tuple[re.Pattern[str], bool]]] = {}
    for log, runs in runs_into(fragment).items():
        for run in runs:
            time = any(RUN_KEY_TIME in k.columns for k in run.key)
            mark = run.mark.strip()
            if not mark and not time:
                continue
            body = _TIME if time and not mark else ".*?".join(re.escape(p) for p in mark.split("{}"))
            if run.builder in ("create_tagged_logger", "make_logger"):
                rx = re.escape(run.at.split("::")[-1].split(".")[-1]) + r" <\s*" + body + r"\s*>"
            elif run.builder == "MsgWrapper" and time and not mark:
                rx = "<" + body + ">"
            else:
                rx = body
            out.setdefault(log, []).append((re.compile(rx), not time and not any(ident & set(k.columns) for k in run.key)))
    return out


def _strings(machine: str, tokens: set[str], narrowed: dict[tuple[str, str], str], owners: Owners | None) -> set[str]:
    """What the records of *tokens* on *machine* are asked by: a mark, or -- one asked for its method's rows -- the mark
    and a blank before each head, and the level padded to 8 and a blank before the mark at each level from WARNING up
    (PanDA's ``%(levelname)-8s %(message)s``; LogWrapper and MsgWrapper write the mark, a blank, the message)."""
    out = set()
    for token in tokens:
        method = narrowed.get((machine, token))
        if method is None or owners is None:
            out.add(token)
            continue
        out |= {f"{token} {head}" for head in owners[method]}
        out |= {f"{level:<8} {token}" for level in ("WARNING", "ERROR", "CRITICAL")}
    return out


def _runs(got: _Gathered, pattern, window: Window, own: _Gathered,
          strings_of) -> list[tuple[tuple[str, str], tuple | None, tuple | None]]:
    """``((machine, mark), lo, hi)`` of every run the key is in that reaches *window*: *lo* the stretch *own* searched
    (the asks over the window itself) for the strings that return its records (*strings_of(machine, mark)*) the first
    record of its part near the window is in, as ``(start, end, first)``,
    and *hi* the one its last is in, as ``(start, end, last)``; None for a record outside every such stretch -- one a
    wider record holds, the run reaching past what the window's own asks searched.

    The part near the window: a mark's records split where they stop for EDGE or more, the pieces that reach the window
    +- EDGE.  A task's mark holds days of passes; its passes far from the window are not what the window asks about.
    """
    lo_w, hi_w = _t(window[0]) - EDGE, _t(window[1]) + EDGE
    hits = {(r.machine, r.token) for r in got.records.values() if r.token and (pattern.search(r.message) or pattern.search(r.token))}
    stamps_of: dict[tuple[str, str], list[datetime.datetime]] = {}
    for r in got.records.values():
        if (r.machine, r.token) in hits:
            stamps_of.setdefault((r.machine, r.token), []).append(_t(r.stamp))
    out = []
    for mark in sorted(hits):
        stamps = sorted(stamps_of[mark])
        pieces = [[stamps[0]]]
        for t in stamps[1:]:
            if t - pieces[-1][-1] >= EDGE:
                pieces.append([t])
            else:
                pieces[-1].append(t)
        near = [x for x in pieces if x[0] <= hi_w and x[-1] >= lo_w]
        if not near:
            continue
        first, last = near[0][0], near[-1][-1]
        stretches = own.stretches(mark[0], strings_of(*mark))
        lo, hi = _stretch_of(stretches, first), _stretch_of(stretches, last)
        out.append((mark, lo and (*lo, first), hi and (*hi, last)))
    return out


def _whole_in(got: _Gathered, strings_of):
    """``whole(machine, mark, first, last)``: the records from *first* to *last* lie in one stretch searched for the
    strings that return the mark's records (*strings_of(machine, mark)*), each more than EDGE from its ends."""
    stretches: dict[tuple, list] = {}

    def whole(machine: str, token: str, first: str, last: str) -> bool:
        at = (machine, strings_of(machine, token))
        if at not in stretches:
            stretches[at] = got.stretches(*at)
        lo = _stretch_of(stretches[at], _t(first))
        return lo is not None and lo == _stretch_of(stretches[at], _t(last)) and _t(first) - lo[0] >= EDGE and lo[1] - _t(last) >= EDGE

    return whole


def _edges_in(got: _Gathered, strings_of):
    """``edges(machine, mark, first, last)``: for records from *first* to *last* that ``_whole_in`` does not read as
    whole, the ends of what was searched for the mark's strings they lie near and why each is there, or a stretch not
    searched among them; ``""``
    where they are whole.  An end is where a file was read only so far (cut at 1 MB, read to the time it was asked or
    to where a read over the asks' windows stopped,
    not answered), where the oldest file kept starts with no known start, or where no file read covers what lies past
    it (the window asked)."""
    whole = _whole_in(got, strings_of)

    def why(machine: str, t: datetime.datetime, side: str, wanted: tuple[str, ...]) -> str:
        covering = [f for f in got.files if f.machine == machine and f.covers(t)]
        if not covering:
            return f"no file read covers what comes {side}"
        said = []
        for entries in _by_file(covering):
            for w in wanted:
                if not any(f.read(t, w) for f in entries):
                    # the records that read it for the string say why they stopped; one that did not, that it did not
                    said += [x for f in entries if (x := f.why(t, w))] or [f"{entries[0].name} was not read for it"]
        said += [f"{f.name} is the oldest kept, its start not known" for f in covering if f.start is None]
        return "; ".join(dict.fromkeys(said)) or "it was not searched"

    def edges(machine: str, token: str, first: str, last: str) -> str:
        if whole(machine, token, first, last):
            return ""
        a, b = _t(first), _t(last)
        wanted = strings_of(machine, token)
        stretches = got.stretches(machine, wanted)
        lo, hi = _stretch_of(stretches, a), _stretch_of(stretches, b)
        short = machine.split(".")[0]
        if lo is None or hi is None:
            return f"its records lie outside what was searched on {short}"
        said = []
        if lo != hi:
            said.append(f"what was searched on {short} stops at {_text(lo[1])} inside it: {why(machine, lo[1] + _SECOND, 'after', wanted)}")
        if a - lo[0] < EDGE:
            said.append(f"what was searched on {short} begins at {_text(lo[0])}: {why(machine, lo[0] - _SECOND, 'before', wanted)}")
        if hi[1] - b < EDGE:
            said.append(f"what was searched on {short} ends at {_text(hi[1])}: {why(machine, hi[1] + _SECOND, 'after', wanted)}")
        return "; ".join(said)

    return edges


def _stretch_of(stretches: list, at: datetime.datetime) -> tuple | None:
    """The searched stretch holding *at*; None when it lies where the search did not reach."""
    for start, end in stretches:
        if start <= at <= end:
            return start, end
    return None
