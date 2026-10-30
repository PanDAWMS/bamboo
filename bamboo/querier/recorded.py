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
  overlap; and its ``status``: ``answered``, or ``failed`` / ``unreadable`` (not read).  A second *t* is searched on a
  machine only when a file whose start is known covers it and every file covering it was read through it -- in an
  overlap, one file read and its neighbour failed is not searched.  ``truncated`` (1 MB, after the recording step halved
  the strings) reads a file only up to its last record returned -- whichever string's it is, so a string cut earlier
  is read past its cut -- unless the record says where every string of the ask was read: ``searched``, closed
  stretches in the covers' form (one a cut ends stops the second before the cut string's last record's), a second of
  the covers counted read only inside one.  A file with no margin (today's, read to the time it
  was asked) is read up to a second before its end: a record stamped just before can be written after.  A file whose
  start is not known (the oldest kept) covers every second before its end but makes none searched.  ``covers`` null
  (a tail holding no stamp) counts for nothing.  What of the window is not searched is :attr:`Log.holes`;
* the service of each machine: a log from JEDI's machines says that their clocks were not all measured
  (:data:`JEDI_CLOCKS`);
* ``cut`` (more than 200 lines after the match): the record's traceback and holes are not read.

A run is the records of one machine with one mark (``runs_for``).  The search for the key returns every record of a
run whose mark holds the key; a mark that does not (the setupper's stamp) is asked for on its own.  A run whose part
near the window (its records split where they stop for :data:`EDGE` or more, the pieces that reach the window +- EDGE)
begins or ends within EDGE of an end of what was searched on its machine may go on past it: the window is widened past
that end, once -- not past an end inside the window (a gap a new ask does not fill) nor past today's file's end (asking
again only reads it later).  A run, or a pass of one (``Run.part``), still that near an end is read as cut there
(:attr:`Log.whole`; rule 8 reads no row missing from it): each by its own first and last record, so the passes inside
a mark that spans days are whole.  Once, because a mark such as a task's ``jediTaskID`` spans days, and widening at
every end would end only at the oldest file kept.  EDGE is the windows' slack, not a measured gap: a run whose records stop for longer
than it across a rotation is read as whole while it is not (KNOWN_ISSUES.md K7).

An ask with no record of its own is answered from the record of a wider one (``RecordedLogs._within``): of the same
log and strings, its window holding the ask's, on the ask's machine or on every machine.  The recording step reads
every file whose covers meet the window, and every machine holding a file of the log when no machine is named, so the
wider record holds every file the ask reaches, each searched whole for the strings; the files past the ask's window
only add seconds that were searched.  It is read for the ask's machine alone, and a machine an every-machine record
does not name holds no file of the log there.  The log's notes say which record answered.
"""
from __future__ import annotations

import datetime
import json
import re
from dataclasses import dataclass
from pathlib import Path

from bamboo.querier.records import PREFIX, STAMP, Log, Record, Unrecorded, Window, keyed

EDGE = datetime.timedelta(minutes=10)
#: the windows are on the database's clock (statuslog, as_of); a log's stamps are its machine's.  The server's statuslog rows fell inside their runs' log times to the second (bamboo-e1,
#: 2026-10-13, 14 rows); JEDI's clocks were measured against the database on some of its machines only (2026-10-14)
JEDI_CLOCKS = "the windows take its log clock as the database's within their slack, measured on some of JEDI's machines only"
_SECOND = datetime.timedelta(seconds=1)


def key_of(ask: dict) -> str:
    return json.dumps(ask, sort_keys=True, separators=(",", ":"))


def _alike(ask: dict) -> tuple:
    """What an ask and a wider one share: all but the window and the machine."""
    return ask["name"], ask["service"], tuple(sorted(set(ask["strings"]))), ask.get("whole_records")


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
    """One file as one record read it: what it covers (start None: not known), and through when it was read (None: not
    read)."""

    machine: str
    start: datetime.datetime | None
    end: datetime.datetime
    read_to: datetime.datetime | None
    name: str = ""
    #: why it was read only to ``read_to``, for a run's line (``_edges_in``); empty for a rotated file read whole
    stop: str = ""
    #: the stretches of its covers the record says every string of the ask was read in (``searched``), closed; None:
    #: all of it up to ``read_to``
    searched: tuple[tuple[datetime.datetime, datetime.datetime], ...] | None = None

    def covers(self, t: datetime.datetime) -> bool:
        return (self.start is None or self.start <= t) and t <= self.end

    def read(self, t: datetime.datetime) -> bool:
        """Whether second *t* of it was read: up to ``read_to``, and inside a searched stretch where the record names
        them."""
        return (self.read_to is not None and t <= self.read_to
                and (self.searched is None or any(a <= t <= b for a, b in self.searched)))


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
        #: (machine, end) of the rotated files read whole: past such an end there is a file to widen into (today's
        #: file, read to the time it was asked, has none: asking again only reads it later)
        self.rotated_ends: set[tuple[str, datetime.datetime]] = set()

    def add(self, record: dict) -> None:
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
                    if not r.get("head"):
                        self.fragments += 1
                        continue
                    rec = _record(machine, f["file"], r)
                    self.records.setdefault((machine, f["file"], r["offset"]), rec)
                    last = rec.stamp
                if not f.get("covers"):
                    continue
                start, end = f["covers"]
                end = _t(end)
                self.rank[(machine, f["file"])] = end
                name = Path(f["file"]).name
                searched = tuple((_t(a), _t(b)) for a, b in f["searched"]) if f.get("searched") is not None else None
                if f.get("status") != "answered":
                    read_to, stop = None, f"{name} was {f.get('status') or 'not read'}"
                elif f.get("truncated") and searched is None:
                    read_to, stop = _t(last) if last else None, f"the answer of {name} was cut at 1 MB after its last record"
                else:
                    read_to = end if f.get("margin_s") else end - _SECOND
                    stop = "" if f.get("margin_s") else f"{name} was read to the time it was asked"
                    if f.get("truncated"):
                        # a string of the ask was cut: the record says where every one of them was read
                        stop = f"the answer of {name} was cut at 1 MB, and its record says every string was read only so far"
                    if f.get("margin_s"):
                        self.rotated_ends.add((machine, end))
                self.files.append(_File(machine, _t(start) if start else None, end, read_to, name, stop, searched))

    def log(self, name: str) -> Log:
        order = sorted(self.records.values(), key=lambda r: (r.machine, self.rank.get((r.machine, r.file), datetime.datetime.max), r.offset))
        return Log(name, order)

    def searched(self, machine: str, t: datetime.datetime) -> bool:
        covering = [f for f in self.files if f.machine == machine and f.covers(t)]
        return any(f.start is not None for f in covering) and all(f.read(t) for f in covering)

    def stretches(self, machine: str, lo: datetime.datetime | None = None, hi: datetime.datetime | None = None) -> list[tuple]:
        """The machine's searched seconds from *lo* to *hi* (all its files' span when not given), as closed stretches."""
        files = [f for f in self.files if f.machine == machine]
        if not files:
            return []
        lo = lo or min((f.start for f in files if f.start is not None), default=min(f.end for f in files))
        hi = hi or max(f.end for f in files)
        cuts = {lo, hi + _SECOND}
        for f in files:
            for t in (f.start, f.end + _SECOND, f.read_to + _SECOND if f.read_to else None,
                      *(x for a, b in f.searched or () for x in (a, b + _SECOND))):
                if t is not None and lo < t <= hi:
                    cuts.add(t)
        points = sorted(cuts)
        out: list[list] = []
        for a, b in zip(points, points[1:], strict=False):
            if self.searched(machine, a):
                if out and out[-1][1] + _SECOND == a:
                    out[-1][1] = b - _SECOND
                else:
                    out.append([a, b - _SECOND])
        return [(a, b) for a, b in out]

    def holes(self, window: Window) -> list[str]:
        """What of *window* was not searched, per machine: the window and the stretches are closed, in seconds."""
        lo, hi = _t(window[0]), _t(window[1])
        out = [f"{m}: {why}" for m, why in sorted(self.silent.items())]
        for machine in sorted(self.listed - set(self.silent)):
            at = lo  # the first second not known to be searched
            for start, end in self.stretches(machine, lo, hi):
                if start > at:
                    out.append(f"{machine}: {_text(at)} .. {_text(start - _SECOND)}")
                at = end + _SECOND
            if at <= hi:
                out.append(f"{machine}: {_text(at)} .. {_text(hi)}")
        return out


def _record(machine: str, file: str, r: dict) -> Record:
    head, *rest = r["lines"]
    m = PREFIX.match(head) if STAMP.match(head) else None
    if m:
        rec = Record(file, 0, m["stamp"], m["logger"], m["level"], m["token"], m["message"], list(rest))
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
        self, name: str, key: str, window: Window | None = None, *, machine: str | None = None, owners: frozenset[str] | None = None
    ) -> Log:
        if window is None:
            raise ValueError(f"{name}: the recorded logs are asked over a window")
        at = (name, key, window, machine, owners)
        if at not in self._logs:
            self._logs[at] = self._read(name, key, window, machine, owners)
        return self._logs[at]

    def _get(self, asks: list[tuple[Ask, str]], notes: list[str]) -> list[dict]:
        """The records of *asks*, or every one missing as :class:`Unrecorded`.  An ask with no record of its own is
        answered from a wider ask's (:meth:`_within`), and *notes* says which."""
        got = [(a, why, self._records.get(a.key) or self._within(a)) for a, why in asks]
        missing = [(a, why) for a, why, record in got if record is None]
        if missing:
            raise Unrecorded(missing)
        notes += [record["within"] for _, _, record in got if "within" in record]
        return [record for _, _, record in got]

    def _within(self, ask: Ask) -> dict | None:
        """*ask*'s answer from the record of a wider ask (the module's doc), or None where none holds it: its own
        machine's record before every machine's, then the narrowest window."""
        want = ask.canonical()
        lo, hi = want["window"]
        found = [
            r for r in self._alike.get(_alike(want), [])
            if r["ask"]["machine"] in (None, want["machine"]) and r["ask"]["window"][0] <= lo and hi <= r["ask"]["window"][1]
        ]
        if not found:
            return None
        record = min(found, key=lambda r: (r["ask"]["machine"] is None, _t(r["ask"]["window"][1]) - _t(r["ask"]["window"][0]), key_of(r["ask"])))
        wider = record["ask"]
        machines = record.get("machines", {})
        if want["machine"] is not None and wider["machine"] is None:
            machines = {want["machine"]: machines.get(want["machine"], {"status": "no such log"})}
        where = wider["machine"] or "every machine"
        return {**record, "ask": want, "machines": machines,
                "within": f"answered from the record of a wider ask ({wider['window'][0]} .. {wider['window'][1]}, {where}, round {record.get('round')})"}

    def _read(self, name: str, key: str, window: Window, machine: str | None, owners: frozenset[str] | None = None) -> Log:
        notes: list[str] = []
        (first,) = self._get([(Ask(name, (key,), window, machine), "")], notes)
        if machine is not None and first.get("machines", {}).get(machine, {}).get("status") == "not listed":
            (first,) = self._get([(Ask(name, (key,), window), f"{machine} is not in list_logs: every machine asked")], notes)
            notes.append(f"{machine} is not in list_logs, so every machine was asked")
        got = _Gathered()
        got.add(first)

        # the marks that do not hold the key: their records need not -- every such mark, near the window or not
        # (KNOWN_ISSUES.md K13); for an asker that reads some functions' runs alone, a DB proxy method's mark is asked
        # only where it is one of theirs and its run can be made whole by it (_partial)
        pattern = keyed(key)
        marks: dict[str, set[str]] = {}
        for r in got.records.values():
            if r.token and pattern.search(r.message) and not pattern.search(r.token):
                marks.setdefault(r.machine, set()).add(r.token)
        partial = _partial(got, marks, owners)
        marks = {m: kept for m, t in marks.items() if (kept := {x for x in t if (m, x) not in partial})}
        why = "the records of the key's runs whose mark does not hold the key"
        for record in self._get([(Ask(name, tuple(sorted(t)), window, m), why) for m, t in sorted(marks.items())], notes):
            got.add(record)

        # the ends of what was searched: widened past once, where a part of a run that reaches the window lies near one
        widen: dict[tuple[str, Window], set[str]] = {}
        for (m, token), lo, hi in _runs(got, pattern, window):
            if (m, token) in partial:
                continue
            strings = {key} if pattern.search(token) else {token}
            # not past an end inside the window (a file read only in part, or not read, is not read better by asking
            # again), nor past today's file's end: the run is read as cut there
            if lo and lo[2] - lo[0] < EDGE and _text(lo[0] - _SECOND) < window[0]:
                widen.setdefault((m, (_text(lo[0] - _SECOND), window[1])), set()).update(strings)
            if hi and hi[1] - hi[2] < EDGE and _text(hi[1] + _SECOND) > window[1] and (m, hi[1]) in got.rotated_ends:
                widen.setdefault((m, (window[0], _text(hi[1] + _SECOND))), set()).update(strings)
        why = f"a run within {int(EDGE.total_seconds() // 60)} min of an end of what was searched"
        for record in self._get([(Ask(name, tuple(sorted(s)), w, m), why) for (m, w), s in sorted(widen.items())], notes):
            got.add(record)
        log = got.log(name)
        log.holes = got.holes(window)
        log.whole = _whole_in(got)
        log.edges = _edges_in(got)
        log.partial = partial
        if partial:
            kinds = sorted(set(partial.values()))
            notes.append(f"{len(partial)} mark(s) read from the key's records alone, as cut: {'; '.join(kinds)}")
        if not got.asked:
            notes.append("no machine lists it: the log is on none")
        if got.without:
            notes.append(f"{sorted(got.without)} hold(s) no file of it")
        if jedi := sorted(m for m, service in got.services.items() if service == "jedi"):
            notes.append(f"JEDI's machine(s) {jedi}: {JEDI_CLOCKS}")
        log.notes = list(dict.fromkeys(notes)) + ([f"{got.fragments} fragment(s) of records whose head was not returned"] if got.fragments else [])
        return log


#: a DB proxy method's mark: the method's name, then its tag (``records.PREFIX``), as ``rerun._of_mover`` reads it
_METHOD_MARK = re.compile(r"([A-Za-z_]\w*) <")


def _partial(got: _Gathered, marks: dict[str, set[str]], owners: frozenset[str] | None) -> dict[tuple[str, str], str]:
    """The marks among *marks* (``machine -> the marks not holding the key``) not asked for whole, each with why, for
    an asker that reads the runs of *owners* alone (None: every mark is asked).  A mark naming no method is a worker's,
    any function's of it, and is asked.  A DB proxy method's mark of no owner is not: no run of it is read.  Nor is an
    owner's whose key records come in pieces EDGE or more apart: the calls share it (``getTasksToExecCommand_JEDI <
    vo=None label=None >``), so its run is every call in the window, which no ask makes whole -- more than 1 MB a file
    and cut (bamboo-e1's round of 2026-10-26).  Such a mark's key records are read as a run cut at both ends."""
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
            stamps = sorted(_t(r.stamp) for r in got.records.values() if (r.machine, r.token) == (machine, token))
            if any(b - a >= EDGE for a, b in zip(stamps, stamps[1:], strict=False)):
                out[(machine, token)] = "a mark the method's calls share"
    return out


def _runs(got: _Gathered, pattern, window: Window) -> list[tuple[tuple[str, str], tuple | None, tuple | None]]:
    """``((machine, mark), lo, hi)`` of every run the key is in that reaches *window*: *lo* the searched stretch the
    first record of its part near the window is in, as ``(start, end, first)``, and *hi* the one its last is in, as
    ``(start, end, last)``; None for a record outside every stretch.

    The part near the window: a mark's records split where they stop for EDGE or more, the pieces that reach the window
    +- EDGE.  A task's mark holds days of passes; its passes far from the window are not what the window asks about.
    """
    lo_w, hi_w = _t(window[0]) - EDGE, _t(window[1]) + EDGE
    hits = {(r.machine, r.token) for r in got.records.values() if r.token and (pattern.search(r.message) or pattern.search(r.token))}
    out = []
    for mark in sorted(hits):
        stamps = sorted(_t(r.stamp) for r in got.records.values() if (r.machine, r.token) == mark)
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
        stretches = got.stretches(mark[0])
        lo, hi = _stretch_of(stretches, first), _stretch_of(stretches, last)
        out.append((mark, lo and (*lo, first), hi and (*hi, last)))
    return out


def _whole_in(got: _Gathered):
    """``whole(machine, first, last)``: the records from *first* to *last* lie in one searched stretch, each more than
    EDGE from its ends."""
    stretches: dict[str, list] = {}

    def whole(machine: str, first: str, last: str) -> bool:
        if machine not in stretches:
            stretches[machine] = got.stretches(machine)
        lo = _stretch_of(stretches[machine], _t(first))
        return lo is not None and lo == _stretch_of(stretches[machine], _t(last)) and _t(first) - lo[0] >= EDGE and lo[1] - _t(last) >= EDGE

    return whole


def _edges_in(got: _Gathered):
    """``edges(machine, first, last)``: for records from *first* to *last* that ``_whole_in`` does not read as whole,
    the ends of what was searched they lie near and why each is there, or a stretch not searched among them; ``""``
    where they are whole.  An end is where a file was read only so far (cut at 1 MB, read to the time it was asked,
    not answered), where the oldest file kept starts with no known start, or where no file read covers what lies past
    it (the window asked)."""
    whole = _whole_in(got)

    def why(machine: str, t: datetime.datetime, side: str) -> str:
        covering = [f for f in got.files if f.machine == machine and f.covers(t)]
        if not covering:
            return f"no file read covers what comes {side}"
        said = [f.stop or f"{f.name} was not read through it" for f in covering if not f.read(t)]
        said += [f"{f.name} is the oldest kept, its start not known" for f in covering if f.start is None]
        return "; ".join(dict.fromkeys(said)) or "it was not searched"

    def edges(machine: str, first: str, last: str) -> str:
        if whole(machine, first, last):
            return ""
        a, b = _t(first), _t(last)
        stretches = got.stretches(machine)
        lo, hi = _stretch_of(stretches, a), _stretch_of(stretches, b)
        short = machine.split(".")[0]
        if lo is None or hi is None:
            return f"its records lie outside what was searched on {short}"
        said = []
        if lo != hi:
            said.append(f"what was searched on {short} stops at {_text(lo[1])} inside it: {why(machine, lo[1] + _SECOND, 'after')}")
        if a - lo[0] < EDGE:
            said.append(f"what was searched on {short} begins at {_text(lo[0])}: {why(machine, lo[0] - _SECOND, 'before')}")
        if hi[1] - b < EDGE:
            said.append(f"what was searched on {short} ends at {_text(hi[1])}: {why(machine, hi[1] + _SECOND, 'after')}")
        return "; ".join(said)

    return edges


def _stretch_of(stretches: list, at: datetime.datetime) -> tuple | None:
    """The searched stretch holding *at*; None when it lies where the search did not reach."""
    for start, end in stretches:
        if start <= at <= end:
            return start, end
    return None
