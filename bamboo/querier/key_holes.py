"""Which records holding the key's digits name the key (K67b B, out/gm/qk-design.md; tmaeno's OK
2026-11-17 22:42).

A record whose message holds the key's digits as a word, under a mark that does not, was read as a record of a run of
the key (``records.runs_for``, and the marks ``recorded`` asks for).  A child task's ``updateTask_JEDI < jediTaskID=<child>
>`` prints the parent's ID as ``:parent_tid`` (NR-T-53639338-modificationTime's H2 asked that mark), and a job's run
prints its files' jediTaskID (getJediFileAttributes 3003): neither is a run on the key's row.  The map says what each
hole holds.  The record is read by the rows of its log the map can read whole (whole first; a loosened reading, every
hole taking any characters, only where no row reads it whole, for a hole class the map guessed wrong: 1483's ``{sql}``
until the map's HTB).  For each hole holding the digits: in a WritePrint's printed bind map, the column of the bind
whose value they are; else the hole's ``columns``.  Where every such hole names columns and none is the class's
identity (the map's ``identity_attrs``), the record names another column: :func:`other_column`.  A record no row reads,
a hole naming no column, or a bind no WritePrint writes (a WHERE bind) stays as before, the key.  The spelling of a
name in the text decides nothing.
"""

from __future__ import annotations

from typing import Any

from bamboo.codemap import trace
from bamboo.codemap.panda import sqltext
from bamboo.querier import mapdump
from bamboo.querier import sources as SR
from bamboo.querier.records import Record, keyed

#: ``[fragment, {log: [(owner, line, map row, loosened literals)]}, {(file, printed line): [(WritePrint, its print)]},
#: {(log, token, message, key, cls): bool}]`` for the installed dump; the fragment is held, so a dump installed after is
#: told from it by identity
_TABLE: list = []

KEY, OTHER, OPEN = "key", "other", "open"


def _table() -> list | None:
    fragment = mapdump.fragment()
    if fragment is None:
        return None
    if not _TABLE or _TABLE[0] is not fragment:
        by_log: dict[str, list] = {}
        for node in fragment.function_prints:
            for st in node.statements:
                for name in st.log_files:
                    for row in st.rows:
                        if not trace._reads_whole_at_all(row):
                            continue
                        lits = (trace._literals(trace._loosened(row.prefix_capture))
                                + trace._literals(trace._loosened(row.capture)))
                        by_log.setdefault(name, []).append((node.owner, st.line, row, lits))
        printed: dict[tuple[str, int], list] = {}
        for wp in getattr(fragment, "write_prints", None) or []:
            for p in wp.printed_by:
                printed.setdefault((wp.owner.partition("::")[0], p.line), []).append((wp, p))
        _TABLE[:] = [fragment, by_log, printed, {}]
    return _TABLE


def other_column(record: Record, log: str, key: str, cls: str) -> bool:
    """Whether *record*, a line of *log* whose message holds *key*'s digits, names them only as columns other than
    *cls*'s identity (the module's doc).  False without a dump, for a record the search cut, and wherever it is not so
    read."""
    table = _table()
    if table is None or record.cut or not record.token:
        return False
    known: dict[tuple[str, str, str, str, str], bool] = table[3]
    at = (log, record.token, record.message, key, cls)
    if at not in known:
        known[at] = verdict(record, table[1].get(log, []), set(SR.identity_attrs(cls)), key)[0] == OTHER
    return known[at]


def _reading(record: Record, rows: list) -> list[tuple[str, int, Any, tuple]]:
    """``(owner, line, map row, hole values)`` of the rows reading *record* whole; else of those reading it with every
    hole taking any characters (``trace.reads_literally``)."""
    lines = [record.message, f"{record.token} {record.message}", f"{record.token}{record.message}"]
    whole, loose = [], []
    for owner, line, row, lits in rows:
        # a whole or loosened reading holds the loosened literals in order: rows without them are passed over
        if not any(trace._in_order(x, lits) for x in lines):
            continue
        got = trace.hole_values(row, record.message, record.token, whole=True)
        if got is not None:
            whole.append((owner, line, row, got))
            continue
        if not whole and trace.reads_literally(row, record.token, record.message):
            loosened = row.model_copy(update={"capture": trace._loosened(row.capture),
                                              "prefix_capture": trace._loosened(row.prefix_capture)})
            got = trace.hole_values(loosened, record.message, record.token, whole=True)
            if got is not None:
                loose.append((owner, line, row, got))
    return whole or loose


def verdict(record: Record, rows: list, identity: set[str], key: str) -> tuple[str, list[tuple[str, str, str]]]:
    """``(kind, said)``: :data:`KEY` where a hole holding the digits names the identity, :data:`OPEN` where one names
    nothing (or no row reads the record), :data:`OTHER` where every one names another column; *said* holds
    ``(kind, owner:line, what the hole is)`` for each."""
    pattern = keyed(key)
    table = _table()
    printed = table[2] if table else {}
    said: list[tuple[str, str, str]] = []
    for owner, line, row, got in _reading(record, rows):
        where = f"{owner.partition('::')[2]}:{line}"
        prints = printed.get((owner.partition("::")[0], line), [])
        for hole, value in zip(row.holes, got[0], strict=False):
            if value is None or not pattern.search(str(value)):
                continue
            if prints and getattr(hole, "entries", None) is not None:
                columns = {b.bind: b.column for wp, _ in prints for b in wp.binds}
                for _, p in prints:
                    cut = sqltext.cut(record.text, p.lead, p.gap)
                    if cut is None:
                        said.append((OPEN, where, "its bind map is not cut"))
                        continue
                    for bind, bound in cut[1].items():
                        if pattern.search(str(bound)):
                            column = columns.get(bind)
                            said.append((OPEN if column is None else KEY if column in identity else OTHER, where,
                                         f"{bind}->{column}"))
                continue
            columns_of = list(hole.columns or [])
            said.append((OPEN if not columns_of else KEY if identity & set(columns_of) else OTHER, where,
                         f"{hole.expression}->{','.join(columns_of) or '-'}"))
    if not said:
        return OPEN, [(OPEN, "-", "no row reads it")]
    kinds = {s[0] for s in said}
    return (KEY if KEY in kinds else OPEN if OPEN in kinds else OTHER), said
