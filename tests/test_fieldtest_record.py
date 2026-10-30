"""The recording step: lists, files and what they cover, greps, and one record per ask (fieldtest.record)."""

from __future__ import annotations

import datetime
import json
import re

import pytest

from bamboo.scripts.panda.fieldtest import record as R
from bamboo.scripts.panda.fieldtest.run import FieldTestError

NOW = datetime.datetime(2026, 10, 14, 11, 50, 0)
SERVER = ("aipanda093.cern.ch", "aipanda095.cern.ch")
JEDI = ("aipanda082.cern.ch",)


def answer(rows: dict[str, dict]) -> dict:
    """A per-machine response as get_result gives it."""
    return {
        "success": True,
        "data": {
            "overall_status": "complete",
            "expected_machines": list(rows),
            "results": [
                {
                    "machine_name": m,
                    "status": r.get("status", "done"),
                    "truncated": r.get("truncated", False),
                    "error_msg": r.get("error"),
                    "return_code": None,
                    "result": json.dumps(r["result"]) if "result" in r else None,
                }
                for m, r in rows.items()
            ],
        },
    }


def entry(name: str, mtime: str, size: int = 1000) -> dict:
    return {"name": name, "size": size, "mtime": mtime}


def rec(offset: int, *lines: str, head: bool = True) -> dict:
    return {"offset": offset, "head": head, "lines": list(lines), "cut": False}


class FakePanda:
    """list_logs and grep_log over a dict of files per machine; each grep answers from *greps* by (machine, file),
    and each pattern grep from *patterns*, by default with no record."""

    def __init__(
        self,
        lists: dict[str, dict[str, dict]],
        greps: dict[tuple[str, str], list],
        now=NOW,
        patterns: dict[tuple[str, str], list] | None = None,
    ) -> None:
        self.lists, self.greps = lists, greps
        self.patterns = patterns or {}
        self.clock = now
        self.calls: list[tuple] = []

    def parts(self) -> R.Parts:
        return R.Parts(
            list_logs=self.list_logs,
            grep_log=self.grep_log,
            now=lambda: self.clock,
            sleep=self.sleep,
        )

    async def sleep(self, seconds: float) -> None:
        self.clock += datetime.timedelta(seconds=seconds)

    async def list_logs(self, *, service=None, machine=None, log_names=None, **_):
        self.calls.append(("list", service or machine, tuple(log_names or ())))
        rows = (
            self.lists[service]
            if service
            else {machine: self.lists["by machine"][machine]}
        )
        out = {}
        for m, r in rows.items():
            r = dict(r)
            if "result" in r and not r.get("ignores_names"):
                r["result"] = [
                    e
                    for e in r["result"]
                    if any(
                        e["name"] == n or e["name"].startswith(n + "-")
                        for n in log_names
                    )
                ]
            out[m] = r
        return answer(out)

    async def grep_log(
        self,
        log_filename,
        fixed_strings=None,
        *,
        pattern=None,
        machine=None,
        tail_bytes=None,
        **_,
    ):
        if pattern is not None:
            self.calls.append(("pattern", machine, log_filename, pattern, tail_bytes))
            replies = self.patterns.get((machine, log_filename)) or [
                {"result": {"records": [], "window": None}}
            ]
            reply = replies.pop(0) if len(replies) > 1 else replies[0]
            return answer({machine: reply})
        self.calls.append(
            ("grep", machine, log_filename, tuple(fixed_strings), tail_bytes)
        )
        replies = self.greps[(machine, log_filename)]
        reply = replies.pop(0) if len(replies) > 1 else replies[0]
        if callable(reply):
            reply = reply(fixed_strings, tail_bytes)
        return answer({machine: reply})


def lists_with(
    server: dict[str, list[dict]], jedi: dict[str, list[dict]] | None = None
) -> dict:
    return {
        "server": {m: {"result": e} for m, e in server.items()},
        "jedi": {m: {"result": e} for m, e in (jedi or {m: [] for m in JEDI}).items()},
    }


def ask(
    name: str, strings: list[str], window: tuple[str, str], machine: str | None = None
) -> dict:
    return {
        "ask": {
            "name": name,
            "service": "unsettled",
            "machine": machine,
            "strings": sorted(strings),
            "window": list(window),
            "whole_records": True,
        },
        "step": "H2",
        "why": "test",
    }


async def run_round(tmp_path, panda: FakePanda, asks: list[dict], max_wait=3600):
    return await R.record_round(
        asks, tmp_path, panda.parts(), datetime.timedelta(seconds=max_wait)
    )


async def run(tmp_path, panda: FakePanda, asks: list[dict], max_wait=3600):
    """A round none of whose asks is deferred: the file written and its records."""
    path, records, deferred = await run_round(tmp_path, panda, asks, max_wait)
    assert deferred == []
    return path, records


ADDER = "panda-adder.log"
ADDER_FILES = [
    entry("panda-adder.log", "2026-10-14 11:49:00", 50_000),
    entry("panda-adder.log-20261013.gz", "2026-10-13 07:11:35"),
    entry("panda-adder.log-20261014.gz", "2026-10-14 06:59:58"),
    entry("panda-DBProxy.log-20261014.gz", "2026-10-14 07:00:01"),
]


def found(*records, first="2026-10-14 07:00:05,000", offset=0) -> dict:
    return {
        "result": {
            "records": list(records),
            "window": {"offset": offset, "first_stamp": first},
        }
    }


async def test_a_rotated_file_covers_from_the_rotation_before_it_to_its_own_plus_the_margin(
    tmp_path,
):
    panda = FakePanda(
        lists_with({SERVER[0]: ADDER_FILES, SERVER[1]: []}),
        {
            (SERVER[0], "panda-adder.log-20261014.gz"): [
                found(rec(10, "2026-10-14 06:30:00,1 x 7339"))
            ]
        },
    )
    # the round also asks of panda-DBProxy.log (of a machine that holds none), so the list holds its files too
    asks = [
        ask(ADDER, ["7339"], ("2026-10-14 06:30:00", "2026-10-14 06:30:00"), SERVER[0]),
        ask(
            "panda-DBProxy.log",
            ["7339"],
            ("2026-10-14 06:30:00", "2026-10-14 06:30:00"),
            SERVER[1],
        ),
    ]
    _, (r, other) = await run(tmp_path, panda, asks)
    assert other["machines"][SERVER[1]]["status"] == "no such log"
    (f,) = r["machines"][SERVER[0]]["files"]
    # the machine rotated at 07:00:01 that day (its DBProxy's mtime), not at the adder's own 06:59:58
    assert f["covers"] == ["2026-10-13 07:11:35", "2026-10-14 07:10:01"]
    assert f["margin_s"] == 600 and f["status"] == "answered"
    assert [x["offset"] for x in f["records"]] == [10]


async def test_a_window_across_a_rotation_reads_both_files_and_the_tail_covers_from_its_first_stamp(
    tmp_path,
):
    greps = {
        (SERVER[0], "panda-adder.log-20261014.gz"): [
            found(rec(10, "2026-10-14 06:59:59,1 a 7339"))
        ],
        (SERVER[0], ADDER): [
            found(
                rec(5, "2026-10-14 07:01:00,1 b 7339"),
                first="2026-10-14 07:00:30,000",
                offset=0,
            )
        ],
    }
    panda = FakePanda(lists_with({SERVER[0]: ADDER_FILES, SERVER[1]: []}), greps)
    _, (r,) = await run(
        tmp_path,
        panda,
        [
            ask(
                ADDER,
                ["7339"],
                ("2026-10-14 06:55:00", "2026-10-14 07:05:00"),
                SERVER[0],
            )
        ],
    )
    gz, cur = r["machines"][SERVER[0]]["files"]
    assert gz["file"] == "panda-adder.log-20261014.gz" and cur["file"] == ADDER
    # read from its first byte, with a line older than the list: the file listed, from the rotation before it (the
    # adder's own mtime: the round asked of no other log)
    assert (
        cur["covers"] == ["2026-10-14 06:59:58", "2026-10-14 11:50:00"]
        and cur["covers_from"] == "rotation"
    )
    assert "margin_s" not in cur


async def test_a_tail_that_began_inside_the_file_covers_from_its_first_stamp_and_none_without_one(
    tmp_path,
):
    greps = {(SERVER[0], ADDER): [found(first="2026-10-14 09:00:00,000", offset=4000)]}
    panda = FakePanda(lists_with({SERVER[0]: ADDER_FILES, SERVER[1]: []}), greps)
    _, (r,) = await run(
        tmp_path,
        panda,
        [
            ask(
                ADDER,
                ["7339"],
                ("2026-10-14 10:00:00", "2026-10-14 10:00:00"),
                SERVER[0],
            )
        ],
    )
    (cur,) = r["machines"][SERVER[0]]["files"]
    assert (
        cur["covers"] == ["2026-10-14 09:00:00", "2026-10-14 11:50:00"]
        and cur["covers_from"] == "first_stamp"
    )
    greps = {(SERVER[0], ADDER): [found(first=None, offset=4000)]}
    panda = FakePanda(lists_with({SERVER[0]: ADDER_FILES, SERVER[1]: []}), greps)
    _, (r,) = await run(
        tmp_path / "b",
        panda,
        [
            ask(
                ADDER,
                ["7339"],
                ("2026-10-14 10:00:00", "2026-10-14 10:00:00"),
                SERVER[0],
            )
        ],
    )
    assert r["machines"][SERVER[0]]["files"][0]["covers"] is None


async def test_an_empty_file_read_from_its_first_byte_covers_from_its_mtime(tmp_path):
    # aipanda095 lists a panda-JediDBProxy.log of 0 bytes last written 2025-12-19 (2026-10-14)
    log = "panda-JediDBProxy.log"
    greps = {(SERVER[1], log): [found(first=None, offset=0)]}
    panda = FakePanda(
        lists_with({SERVER[0]: [], SERVER[1]: [entry(log, "2025-12-19 15:30:35", 0)]}),
        greps,
    )
    _, (r,) = await run(
        tmp_path,
        panda,
        [ask(log, ["7358"], ("2026-10-14 10:00:00", "2026-10-14 10:00:00"), SERVER[1])],
    )
    (cur,) = r["machines"][SERVER[1]]["files"]
    assert cur["covers"] == ["2025-12-19 15:30:35", "2026-10-14 11:50:00"]
    assert cur["covers_from"] == "unstamped"
    assert cur["status"] == "answered" and cur["error"] is None and cur["records"] == []
    # nothing more to read: asked once, not again with twice the bytes
    assert len([c for c in panda.calls if c[0] == "grep"]) == 1


async def test_the_oldest_file_has_no_start(tmp_path):
    greps = {(SERVER[0], "panda-adder.log-20261013.gz"): [found()]}
    panda = FakePanda(lists_with({SERVER[0]: ADDER_FILES, SERVER[1]: []}), greps)
    _, (r,) = await run(
        tmp_path,
        panda,
        [
            ask(
                ADDER,
                ["7339"],
                ("2026-09-01 00:00:00", "2026-09-01 00:00:00"),
                SERVER[0],
            )
        ],
    )
    (f,) = r["machines"][SERVER[0]]["files"]
    assert f["covers"][0] is None


async def test_the_machine_an_ask_names_is_not_listed_or_holds_no_such_log(tmp_path):
    panda = FakePanda(lists_with({SERVER[0]: ADDER_FILES, SERVER[1]: []}), {})
    asks = [
        ask(
            ADDER,
            ["1"],
            ("2026-10-14 10:00:00", "2026-10-14 10:00:00"),
            "aipanda999.cern.ch",
        ),
        ask(ADDER, ["1"], ("2026-10-14 10:00:00", "2026-10-14 10:00:00"), SERVER[1]),
    ]
    _, (a, b) = await run(tmp_path, panda, asks)
    assert a["machines"] == {
        "aipanda999.cern.ch": {
            "service": None,
            "status": "not listed",
            "error": None,
            "files": [],
        }
    }
    assert (
        b["machines"][SERVER[1]]["status"] == "no such log"
        and b["machines"][SERVER[1]]["service"] == "server"
    )


async def test_a_list_cut_or_listing_names_not_asked_is_not_known(tmp_path):
    lists = lists_with(
        {
            SERVER[0]: ADDER_FILES,
            SERVER[1]: [entry("panda-Entry.log", "2026-10-14 11:00:00")],
        }
    )
    lists["server"][SERVER[1]]["ignores_names"] = True
    lists["jedi"] = {JEDI[0]: {"result": [], "truncated": True}}
    greps = {(SERVER[0], ADDER): [found(first="2026-10-14 07:00:30,000")]}
    panda = FakePanda(lists, greps)
    _, (r,) = await run(
        tmp_path,
        panda,
        [ask(ADDER, ["7339"], ("2026-10-14 10:00:00", "2026-10-14 10:00:00"))],
    )
    assert (
        r["machines"][SERVER[1]]["status"] == "list cut"
        and "log_names not honoured" in r["machines"][SERVER[1]]["error"]
    )
    assert r["machines"][JEDI[0]]["status"] == "list cut"
    assert r["machines"][SERVER[0]]["status"] == "listed"


async def test_asks_of_one_file_share_a_grep_and_each_record_keeps_its_own_strings(
    tmp_path,
):
    reply = found(
        rec(1, "2026-10-14 06:00:00,1 a 111"),
        rec(2, "2026-10-14 06:00:01,1 b 222"),
        rec(3, "x 111", head=False),
    )
    greps = {(SERVER[0], "panda-adder.log-20261014.gz"): [reply]}
    panda = FakePanda(lists_with({SERVER[0]: ADDER_FILES, SERVER[1]: []}), greps)
    w = ("2026-10-14 06:00:00", "2026-10-14 06:00:01")
    _, (a, b) = await run(
        tmp_path,
        panda,
        [ask(ADDER, ["111"], w, SERVER[0]), ask(ADDER, ["222"], w, SERVER[0])],
    )
    assert [c for c in panda.calls if c[0] == "grep"] == [
        ("grep", SERVER[0], "panda-adder.log-20261014.gz", ("111", "222"), None)
    ]
    fa, fb = a["machines"][SERVER[0]]["files"][0], b["machines"][SERVER[0]]["files"][0]
    assert [x["offset"] for x in fa["records"]] == [1] and fa["fragments"] == 1
    assert [x["offset"] for x in fb["records"]] == [2] and fb["fragments"] == 0


async def test_a_cut_answer_is_asked_again_in_halves_and_a_string_cut_alone_marks_its_files(
    tmp_path,
):
    def reply(strings, _tail):
        if len(strings) > 1 or strings == ["111"]:
            return {**found(rec(1, "a 111")), "truncated": True}
        return found(rec(2, "b 222"))

    greps = {(SERVER[0], "panda-adder.log-20261014.gz"): [reply]}
    panda = FakePanda(lists_with({SERVER[0]: ADDER_FILES, SERVER[1]: []}), greps)
    w = ("2026-10-14 06:00:00", "2026-10-14 06:00:01")
    _, (a, b) = await run(
        tmp_path,
        panda,
        [ask(ADDER, ["111"], w, SERVER[0]), ask(ADDER, ["222"], w, SERVER[0])],
    )
    assert [c[3] for c in panda.calls if c[0] == "grep"] == [
        ("111", "222"),
        ("111",),
        ("222",),
    ]
    assert a["machines"][SERVER[0]]["files"][0]["truncated"] is True
    assert b["machines"][SERVER[0]]["files"][0]["truncated"] is False


async def test_a_half_whose_grep_fails_defers_the_asks_of_its_strings_alone(tmp_path):
    day = "panda-adder.log-20261014.gz"

    def reply(strings, _tail):
        if len(strings) > 2:
            return {**found(rec(1, "2026-10-14 06:00:00,1 a 111")), "truncated": True}
        if strings == ["111", "222"]:
            return {"status": "failed", "error": "timeout"}
        return found(rec(3, "2026-10-14 06:00:00,3 c 333"), rec(4, "d 444"))

    panda = FakePanda(
        lists_with({SERVER[0]: ADDER_FILES, SERVER[1]: []}), {(SERVER[0], day): [reply]}
    )
    w = ("2026-10-14 06:00:00", "2026-10-14 06:00:01")
    asks = [ask(ADDER, [s], w, SERVER[0]) for s in ("111", "222", "333", "444")]
    _, records, deferred = await run_round(tmp_path, panda, asks)
    assert [c[3] for c in panda.calls if c[0] == "grep"] == [
        ("111", "222", "333", "444"),
        *[("111", "222")] * (1 + R.TIMEOUT_RETRIES),
        ("333", "444"),
    ]
    assert [d["ask"]["strings"] for d in deferred] == [["111"], ["222"]]
    assert {w for d in deferred for w in d["why"]} == {
        f"{SERVER[0]} {day}: failed: timeout"
    }
    assert [r["ask"]["strings"] for r in records] == [["333"], ["444"]]
    f333, f444 = (r["machines"][SERVER[0]]["files"][0] for r in records)
    assert f333["status"] == "answered" and [x["offset"] for x in f333["records"]] == [
        3
    ]
    assert f444["status"] == "answered" and [x["offset"] for x in f444["records"]] == [
        4
    ]


async def test_a_grep_stopped_at_the_timeout_is_asked_again_and_its_answer_taken(
    tmp_path,
):
    day = "panda-adder.log-20261014.gz"
    replies = [
        {"status": "failed", "error": "timeout"},
        found(rec(1, "2026-10-14 06:00:00,1 a 7339")),
    ]
    panda = FakePanda(
        lists_with({SERVER[0]: ADDER_FILES, SERVER[1]: []}), {(SERVER[0], day): replies}
    )
    w = ("2026-10-14 06:00:00", "2026-10-14 06:00:00")
    _, (r,) = await run(tmp_path, panda, [ask(ADDER, ["7339"], w, SERVER[0])])
    assert [c[3] for c in panda.calls if c[0] == "grep"] == [("7339",), ("7339",)]
    (f,) = r["machines"][SERVER[0]]["files"]
    assert f["status"] == "answered" and [x["offset"] for x in f["records"]] == [1]
    assert [q["error"] for q in r["requests"]] == ["timeout", None]


@pytest.mark.parametrize(
    "error, greps", [("timeout", 1 + R.TIMEOUT_RETRIES), ("killed", 1)]
)
async def test_a_grep_is_asked_again_only_after_a_timeout_and_only_so_often(
    tmp_path, error, greps
):
    day = "panda-adder.log-20261014.gz"
    panda = FakePanda(
        lists_with({SERVER[0]: ADDER_FILES, SERVER[1]: []}),
        {(SERVER[0], day): [{"status": "failed", "error": error}]},
    )
    w = ("2026-10-14 06:00:00", "2026-10-14 06:00:00")
    _, records, (d,) = await run_round(
        tmp_path, panda, [ask(ADDER, ["7339"], w, SERVER[0])]
    )
    assert len([c for c in panda.calls if c[0] == "grep"]) == greps
    assert records == [] and d["why"] == [f"{SERVER[0]} {day}: failed: {error}"]


async def test_a_day_being_compressed_is_waited_for_and_read_when_done(tmp_path):
    plain = [entry("panda-adder.log-20261014", "2026-10-14 07:00:03"), *ADDER_FILES[:2]]
    lists = lists_with({SERVER[0]: plain, SERVER[1]: []})
    lists["by machine"] = {SERVER[0]: {"result": ADDER_FILES}}
    greps = {(SERVER[0], "panda-adder.log-20261014.gz"): [found(rec(1, "a 7339"))]}
    panda = FakePanda(lists, greps)
    _, (r,) = await run(
        tmp_path,
        panda,
        [
            ask(
                ADDER,
                ["7339"],
                ("2026-10-14 06:00:00", "2026-10-14 06:00:00"),
                SERVER[0],
            )
        ],
    )
    (f,) = r["machines"][SERVER[0]]["files"]
    assert f["status"] == "answered" and [x["offset"] for x in f["records"]] == [1]
    # the plain file's mtime is its last write, when the processes stopped
    assert f["covers"][1] == "2026-10-14 07:10:03"
    assert panda.clock == NOW + R.COMPRESS_POLL


async def test_a_grep_refused_as_being_compressed_waits_and_a_day_past_its_next_is_unreadable(
    tmp_path,
):
    files = [
        entry("panda-adder.log", "2026-10-14 11:49:00"),
        entry("panda-adder.log-20261010.gz", "2026-10-10 07:00:00"),
        entry("panda-adder.log-20261011", "2026-10-11 07:00:00"),
    ]
    lists = lists_with({SERVER[0]: files, SERVER[1]: []})
    panda = FakePanda(lists, {})
    _, (r,) = await run(
        tmp_path,
        panda,
        [
            ask(
                ADDER,
                ["7339"],
                ("2026-10-11 06:00:00", "2026-10-11 06:00:00"),
                SERVER[0],
            )
        ],
    )
    (f,) = r["machines"][SERVER[0]]["files"]
    assert (
        f["status"] == "unreadable"
        and f["error"] == "being compressed past 2026-10-12 23:59:59"
    )
    assert not [c for c in panda.calls if c[0] == "grep"]


async def test_a_round_that_cannot_settle_writes_nothing(tmp_path):
    files = [
        entry("panda-adder.log", "2026-10-14 11:49:00"),
        entry("panda-adder.log-20261014", "2026-10-14 07:00:00"),
    ]
    lists = lists_with({SERVER[0]: files, SERVER[1]: []})
    lists["by machine"] = {SERVER[0]: {"result": files}}
    panda = FakePanda(lists, {})
    with pytest.raises(R.NotSettled):
        await run(
            tmp_path,
            panda,
            [
                ask(
                    ADDER,
                    ["7339"],
                    ("2026-10-14 06:00:00", "2026-10-14 06:00:00"),
                    SERVER[0],
                )
            ],
            max_wait=1200,
        )
    assert not list(tmp_path.glob("records-r*.jsonl"))


async def test_a_later_round_reads_a_rotated_files_earlier_answer_again(tmp_path):
    greps = {
        (SERVER[0], "panda-adder.log-20261014.gz"): [
            found(rec(1, "2026-10-14 06:00:00,1 a 7339"))
        ]
    }
    panda = FakePanda(lists_with({SERVER[0]: ADDER_FILES, SERVER[1]: []}), greps)
    first = ask(
        ADDER, ["7339"], ("2026-10-14 06:00:00", "2026-10-14 06:00:00"), SERVER[0]
    )
    p1, _ = await run(tmp_path, panda, [first])
    wider = ask(
        ADDER, ["7339"], ("2026-10-14 05:00:00", "2026-10-14 06:00:00"), SERVER[0]
    )
    p2, (r,) = await run(tmp_path, panda, [wider])
    assert (p1.name, p2.name) == ("records-r1.jsonl", "records-r2.jsonl")
    assert len([c for c in panda.calls if c[0] == "grep"]) == 1
    (f,) = r["machines"][SERVER[0]]["files"]
    assert [x["offset"] for x in f["records"]] == [1] and r["requests"][0][
        "reused"
    ].startswith("answers-r1-")


def by_string(strings, _tail):
    """A grep's answer: a record per string, at the offset of its first digit."""
    return found(
        *(rec(int(s[0]), f"2026-10-14 06:00:00,{s[0]} x {s}") for s in strings)
    )


async def test_a_later_round_asks_only_the_strings_no_earlier_answer_held(tmp_path):
    day = "panda-adder.log-20261014.gz"
    panda = FakePanda(
        lists_with({SERVER[0]: ADDER_FILES, SERVER[1]: []}),
        {(SERVER[0], day): [by_string]},
    )
    w1 = ("2026-10-14 06:00:00", "2026-10-14 06:00:00")
    await run(tmp_path, panda, [ask(ADDER, [s], w1, SERVER[0]) for s in ("111", "222")])
    w2 = ("2026-10-14 05:00:00", "2026-10-14 06:00:00")
    _, (a, b) = await run(
        tmp_path, panda, [ask(ADDER, [s], w2, SERVER[0]) for s in ("111", "333")]
    )
    assert [c[3] for c in panda.calls if c[0] == "grep"] == [("111", "222"), ("333",)]
    fa, fb = a["machines"][SERVER[0]]["files"][0], b["machines"][SERVER[0]]["files"][0]
    assert [x["offset"] for x in fa["records"]] == [1]
    assert [x["offset"] for x in fb["records"]] == [3]
    assert a["requests"][0]["reused"].startswith("answers-r1-")


async def test_a_held_strings_ask_is_answered_when_the_new_strings_grep_fails(
    tmp_path,
):
    day = "panda-adder.log-20261014.gz"

    def reply(strings, tail):
        if "333" in strings:
            return {"status": "failed", "error": "timeout"}
        return by_string(strings, tail)

    panda = FakePanda(
        lists_with({SERVER[0]: ADDER_FILES, SERVER[1]: []}), {(SERVER[0], day): [reply]}
    )
    w1 = ("2026-10-14 06:00:00", "2026-10-14 06:00:00")
    await run(tmp_path, panda, [ask(ADDER, ["111"], w1, SERVER[0])])
    w2 = ("2026-10-14 05:00:00", "2026-10-14 06:00:00")
    _, records, deferred = await run_round(
        tmp_path, panda, [ask(ADDER, [s], w2, SERVER[0]) for s in ("111", "333")]
    )
    assert [d["ask"]["strings"] for d in deferred] == [["333"]]
    (r,) = records
    (f,) = r["machines"][SERVER[0]]["files"]
    assert r["ask"]["strings"] == ["111"] and f["status"] == "answered"
    assert [x["offset"] for x in f["records"]] == [1]


DAY = "panda-adder.log-20261014.gz"


def t(s: str) -> datetime.datetime:
    return datetime.datetime.strptime(s, "%Y-%m-%d %H:%M:%S")


def cut_alone(cut_at: str, uncut: tuple[str, str] = ("222", "2026-10-14 06:30:00")):
    """A grep's answer where "111" alone is cut after its record at *cut_at*, and the *uncut* string is not."""

    def reply(strings, _tail):
        if len(strings) > 1:
            return {**found(rec(1, f"{cut_at},1 a 111")), "truncated": True}
        if strings == ["111"]:
            return {**found(rec(1, f"{cut_at},1 a 111")), "truncated": True}
        return found(rec(2, f"{uncut[1]},1 b {uncut[0]}"))

    return reply


async def test_a_cut_string_is_read_to_the_second_before_its_last_record_and_asked_again_by_pattern(
    tmp_path,
):
    # MR-T-53354737-queuedTime (2026-10-30): the file's other string ran past the cut one's last record, and the
    # querier read the window as searched; and the cut one's window lay past its cut
    panda = FakePanda(
        lists_with({SERVER[0]: ADDER_FILES, SERVER[1]: []}),
        {(SERVER[0], DAY): [cut_alone("2026-10-14 01:00:00")]},
        patterns={(SERVER[0], DAY): [found(rec(3, "2026-10-14 06:00:00,7 a 111"))]},
    )
    w = ("2026-10-14 06:00:00", "2026-10-14 06:00:01")
    _, (r,) = await run(tmp_path, panda, [ask(ADDER, ["111", "222"], w, SERVER[0])])
    assert [c[3] for c in panda.calls if c[0] == "grep"] == [
        ("111", "222"),
        ("111",),
        ("222",),
    ]
    wanted = {"111": [(t("2026-10-14 05:50:00"), t("2026-10-14 06:10:01"))]}
    assert [c[3] for c in panda.calls if c[0] == "pattern"] == [
        R.continuation_pattern(wanted)
    ]
    (f,) = r["machines"][SERVER[0]]["files"]
    assert f["truncated"] is True and [x["offset"] for x in f["records"]] == [1, 2, 3]
    assert f["searched"] == [
        [f["covers"][0], "2026-10-14 00:59:59"],
        ["2026-10-14 05:50:00", "2026-10-14 06:10:01"],
    ]


async def test_an_uncut_file_says_nothing_of_what_was_searched(tmp_path):
    panda = FakePanda(
        lists_with({SERVER[0]: ADDER_FILES, SERVER[1]: []}),
        {(SERVER[0], DAY): [cut_alone("2026-10-14 01:00:00")]},
    )
    w = ("2026-10-14 06:00:00", "2026-10-14 06:00:01")
    _, (a, b) = await run(
        tmp_path,
        panda,
        [ask(ADDER, ["111"], w, SERVER[0]), ask(ADDER, ["222"], w, SERVER[0])],
    )
    fa, fb = a["machines"][SERVER[0]]["files"][0], b["machines"][SERVER[0]]["files"][0]
    assert "searched" in fa and "searched" not in fb and fb["truncated"] is False


async def test_a_continuation_cut_again_goes_on_from_its_last_records_second_while_it_gets_further(
    tmp_path,
):
    panda = FakePanda(
        lists_with({SERVER[0]: ADDER_FILES, SERVER[1]: []}),
        {(SERVER[0], DAY): [cut_alone("2026-10-14 01:00:00")]},
        patterns={
            (SERVER[0], DAY): [
                {**found(rec(10, "2026-10-14 05:55:00,1 a 111")), "truncated": True},
                {**found(rec(11, "2026-10-14 05:55:00,9 a 111")), "truncated": True},
            ]
        },
    )
    w = ("2026-10-14 06:00:00", "2026-10-14 06:00:00")
    _, (r,) = await run(tmp_path, panda, [ask(ADDER, ["111"], w, SERVER[0])])
    assert [c[3] for c in panda.calls if c[0] == "pattern"] == [
        R.continuation_pattern(
            {"111": [(t("2026-10-14 05:50:00"), t("2026-10-14 06:10:00"))]}
        ),
        R.continuation_pattern(
            {"111": [(t("2026-10-14 05:55:00"), t("2026-10-14 06:10:00"))]}
        ),
    ]
    (f,) = r["machines"][SERVER[0]]["files"]
    assert [x["offset"] for x in f["records"]] == [1, 10, 11]
    assert f["status"] == "answered" and f["searched"] == [
        [f["covers"][0], "2026-10-14 00:59:59"],
        ["2026-10-14 05:50:00", "2026-10-14 05:54:59"],
    ]


async def test_a_page_that_reads_no_further_goes_on_without_the_string_most_of_its_records_hold(
    tmp_path,
):
    # pin m's round 5 (2026-10-30): a call printing a row per task filled 1 MB in one second of aipanda088's
    # JediDBProxy.log, and the rarer strings of other asks sharing the grep were not read past it
    def reply(strings, _tail):
        last = {"111": (1, "2026-10-14 01:00:00"), "333": (2, "2026-10-14 02:00:00")}
        return {
            **found(*(rec(last[s][0], f"{last[s][1]},1 {s}") for s in strings)),
            "truncated": True,
        }

    busy = "2026-10-14 05:50:00"
    panda = FakePanda(
        lists_with({SERVER[0]: ADDER_FILES, SERVER[1]: []}),
        {(SERVER[0], DAY): [reply]},
        patterns={
            (SERVER[0], DAY): [
                {
                    **found(
                        rec(10, f"{busy},1 111"),
                        rec(11, f"{busy},2 111"),
                        rec(12, f"{busy},3 333"),
                    ),
                    "truncated": True,
                },
                found(rec(13, "2026-10-14 06:00:00,5 333")),
            ]
        },
    )
    w = ("2026-10-14 06:00:00", "2026-10-14 06:00:00")
    a, b = ask(ADDER, ["111"], w, SERVER[0]), ask(ADDER, ["333"], w, SERVER[0])
    _, (ra, rb) = await run(tmp_path, panda, [a, b])
    around = [(t("2026-10-14 05:50:00"), t("2026-10-14 06:10:00"))]
    assert [c[3] for c in panda.calls if c[0] == "pattern"] == [
        R.continuation_pattern({"111": around, "333": around}),
        R.continuation_pattern({"333": around}),
    ]
    (fa,), (fb,) = (
        ra["machines"][SERVER[0]]["files"],
        rb["machines"][SERVER[0]]["files"],
    )
    assert fa["searched"] == [[fa["covers"][0], "2026-10-14 00:59:59"]]
    assert fb["searched"] == [
        [fb["covers"][0], "2026-10-14 01:59:59"],
        ["2026-10-14 05:50:00", "2026-10-14 06:10:00"],
    ]
    assert [x["offset"] for x in fb["records"]] == [2, 12, 13]
    assert R.cut_asks(tmp_path) == [{"ask": a["ask"]}]


@pytest.mark.parametrize(
    "entry, over",
    [
        (
            {
                "truncated": True,
                "covers": ["2026-10-14 00:00:00", "2026-10-14 07:00:00"],
            },
            False,
        ),
        ({"truncated": True, "covers": None, "searched": []}, True),
        (
            {
                "truncated": True,
                "covers": [None, "2026-10-14 07:00:00"],
                "searched": [
                    [None, "2026-10-14 01:00:00"],
                    ["2026-10-14 05:50:00", "2026-10-14 06:10:00"],
                ],
            },
            True,
        ),
        (
            {
                "truncated": True,
                "covers": ["2026-10-14 00:00:00", "2026-10-14 06:05:00"],
                "searched": [["2026-10-14 05:50:00", "2026-10-14 06:04:59"]],
            },
            False,
        ),
        (
            {
                "truncated": True,
                "covers": ["2026-10-14 06:05:00", "2026-10-14 07:00:00"],
                "searched": [["2026-10-14 06:05:00", "2026-10-14 06:10:00"]],
            },
            True,
        ),
    ],
)
def test_a_cut_file_is_searched_over_its_window_when_its_covers_there_are_all_searched(
    entry, over
):
    assert (
        R.searched_over(entry, ["2026-10-14 06:00:00", "2026-10-14 06:00:00"]) is over
    )


async def test_a_continuation_that_fails_defers_its_asks_after_its_timeouts(tmp_path):
    panda = FakePanda(
        lists_with({SERVER[0]: ADDER_FILES, SERVER[1]: []}),
        {(SERVER[0], DAY): [cut_alone("2026-10-14 01:00:00")]},
        patterns={(SERVER[0], DAY): [{"status": "failed", "error": "timeout"}]},
    )
    w = ("2026-10-14 06:00:00", "2026-10-14 06:00:00")
    _, records, deferred = await run_round(
        tmp_path,
        panda,
        [ask(ADDER, ["111"], w, SERVER[0]), ask(ADDER, ["222"], w, SERVER[0])],
    )
    assert len([c for c in panda.calls if c[0] == "pattern"]) == 1 + R.TIMEOUT_RETRIES
    assert [d["ask"]["strings"] for d in deferred] == [["111"]]
    assert deferred[0]["why"] == [f"{SERVER[0]} {DAY}: failed: timeout"]
    assert [r["ask"]["strings"] for r in records] == [["222"]]


async def test_a_cut_string_of_the_current_file_is_not_asked_again(tmp_path):
    current = "panda-adder.log"
    panda = FakePanda(
        lists_with({SERVER[0]: ADDER_FILES, SERVER[1]: []}),
        {
            (SERVER[0], current): [
                cut_alone("2026-10-14 10:00:00", ("222", "2026-10-14 11:30:00"))
            ]
        },
    )
    w = ("2026-10-14 11:00:00", "2026-10-14 11:00:01")
    _, (r,) = await run(tmp_path, panda, [ask(ADDER, ["111", "222"], w, SERVER[0])])
    assert [c for c in panda.calls if c[0] == "pattern"] == []
    (f,) = r["machines"][SERVER[0]]["files"]
    assert f["file"] == current and f["searched"] == [
        [f["covers"][0], "2026-10-14 09:59:59"]
    ]


async def test_a_later_round_reads_a_cut_string_from_the_earlier_answer_and_asks_only_the_rest(
    tmp_path,
):
    panda = FakePanda(
        lists_with({SERVER[0]: ADDER_FILES, SERVER[1]: []}),
        {(SERVER[0], DAY): [cut_alone("2026-10-14 01:00:00")]},
        patterns={(SERVER[0], DAY): [found(rec(3, "2026-10-14 06:00:00,7 a 111"))]},
    )
    w1 = ("2026-10-14 06:00:00", "2026-10-14 06:00:00")
    await run(tmp_path, panda, [ask(ADDER, ["111"], w1, SERVER[0])])
    before = len(panda.calls)
    w2 = ("2026-10-14 06:05:00", "2026-10-14 06:20:00")
    _, (r,) = await run(tmp_path, panda, [ask(ADDER, ["111"], w2, SERVER[0])])
    later = panda.calls[before:]
    assert [c for c in later if c[0] == "grep"] == []
    assert [c[3] for c in later if c[0] == "pattern"] == [
        R.continuation_pattern(
            {"111": [(t("2026-10-14 06:10:01"), t("2026-10-14 06:30:00"))]}
        )
    ]
    (f,) = r["machines"][SERVER[0]]["files"]
    assert [x["offset"] for x in f["records"]] == [1, 3]
    assert f["searched"] == [
        [f["covers"][0], "2026-10-14 00:59:59"],
        ["2026-10-14 05:50:00", "2026-10-14 06:30:00"],
    ]


async def test_again_records_a_cut_files_ask_once_more_saying_the_round_it_replaces(
    tmp_path,
):
    w = ("2026-10-14 06:00:00", "2026-10-14 06:00:00")
    a = ask(ADDER, ["111"], w, SERVER[0])
    old = {  # a record written before cut files said what was searched
        "ask": a["ask"],
        "round": 1,
        "machines": {SERVER[0]: {"files": [{"file": DAY, "truncated": True}]}},
    }
    (tmp_path / "records-r1.jsonl").write_text(json.dumps(old) + "\n")
    assert R.cut_asks(tmp_path) == [{"ask": a["ask"]}]
    panda = FakePanda(
        lists_with({SERVER[0]: ADDER_FILES, SERVER[1]: []}),
        {(SERVER[0], DAY): [cut_alone("2026-10-14 01:00:00")]},
    )
    with pytest.raises(FieldTestError, match="already recorded"):
        await run_round(tmp_path, panda, R.cut_asks(tmp_path))
    ((path, (r,), deferred),) = await R.record_rounds(
        [(R.cut_asks(tmp_path), tmp_path)],
        panda.parts(),
        datetime.timedelta(seconds=3600),
        None,
        True,
    )
    assert path.name == "records-r2.jsonl" and deferred == []
    assert r["round"] == 2 and r["replaces"] == 1
    assert "searched" in r["machines"][SERVER[0]]["files"][0]
    assert R.cut_asks(tmp_path) == []
    with pytest.raises(FieldTestError, match="not recorded"):
        await R.record_rounds(
            [([ask(ADDER, ["999"], w, SERVER[0])], tmp_path)],
            panda.parts(),
            datetime.timedelta(seconds=3600),
            None,
            True,
        )


@pytest.mark.parametrize(
    "lo, hi",
    [
        ("2026-10-14 05:50:00", "2026-10-14 06:10:01"),
        ("2026-10-13 23:59:58", "2026-10-14 00:00:02"),
        ("2026-10-14 05:55:07", "2026-10-14 05:55:07"),
        ("2026-10-14 05:59:59", "2026-10-14 07:00:00"),
        ("2026-10-14 05:53:00", "2026-10-14 06:21:30"),
        ("2026-10-14 05:53:07", "2026-10-14 05:55:31"),
    ],
)
def test_the_stamp_prefixes_of_a_stretch_hold_just_its_seconds(lo, hi):
    prefixes = R.stamp_prefixes(t(lo), t(hi))
    second = t(lo) - datetime.timedelta(hours=1)
    while second <= t(hi) + datetime.timedelta(hours=1):
        s = second.strftime("%Y-%m-%d %H:%M:%S")
        assert any(s.startswith(p) for p in prefixes) == (lo <= s <= hi), s
        second += datetime.timedelta(seconds=1)
    assert len(prefixes) <= 20


def test_the_stamp_prefixes_of_days_are_the_days():
    assert R.stamp_prefixes(t("2026-10-12 00:00:00"), t("2026-10-13 23:59:59")) == [
        "2026-10-12 ",
        "2026-10-13 ",
    ]


def test_a_continuation_pattern_matches_a_stamped_line_in_its_strings_stretch_holding_it():
    s = "getTasks_JEDI < vo=a.b label=c+d > (x)|$"
    pattern = R.continuation_pattern(
        {
            s: [(t("2026-10-14 05:50:00"), t("2026-10-14 06:10:00"))],
            "7339": [(t("2026-10-14 08:00:00"), t("2026-10-14 08:00:00"))],
        }
    )
    m = re.compile(pattern)
    assert m.search(f"2026-10-14 06:00:00,123 panda.log.X: DEBUG    {s} done")
    assert m.search("2026-10-14 08:00:00,9 x 7339")
    assert not m.search(f"2026-10-14 06:10:01,0 x {s}")
    assert not m.search("2026-10-14 06:00:00,0 x 7339")
    assert not m.search(f"  File x, line 3, in {s}")
    assert not m.search(
        "2026-10-14 06:00:00,0 x getTasks_JEDI < vo=aXb label=c+d > (x)|$"
    )


async def test_an_ask_already_recorded_is_refused(tmp_path):
    greps = {(SERVER[0], "panda-adder.log-20261014.gz"): [found()]}
    panda = FakePanda(lists_with({SERVER[0]: ADDER_FILES, SERVER[1]: []}), greps)
    a = ask(ADDER, ["7339"], ("2026-10-14 06:00:00", "2026-10-14 06:00:00"), SERVER[0])
    await run(tmp_path, panda, [a])
    with pytest.raises(FieldTestError, match="already recorded"):
        await run(tmp_path, panda, [a])


def test_an_ask_not_in_its_canonical_form_is_refused(tmp_path):
    bad = ask(ADDER, ["b", "a"], ("2026-10-14 06:00:00", "2026-10-14 06:00:00"))
    bad["ask"]["strings"] = ["b", "a"]
    path = tmp_path / "asks.jsonl"
    path.write_text(json.dumps(bad) + "\n")
    with pytest.raises(FieldTestError, match="sorted"):
        R.read_asks(path)


# ------------------------------------------------------------------ deferred asks


def _adder_on(machine: str) -> dict:
    return ask(ADDER, ["7339"], ("2026-10-14 06:00:00", "2026-10-14 06:00:00"), machine)


async def test_an_ask_a_grep_failed_for_is_deferred_and_answered_in_a_later_round(
    tmp_path,
):
    day = "panda-adder.log-20261014.gz"
    greps = {
        (SERVER[0], day): [
            {"status": "failed", "error": "timed out"},
            found(rec(1, "2026-10-14 06:00:00,1 a 7339")),
        ],
        (SERVER[1], day): [found(rec(2, "2026-10-14 06:00:00,2 b 7339"))],
    }
    lists = lists_with({SERVER[0]: ADDER_FILES, SERVER[1]: ADDER_FILES})
    panda = FakePanda(lists, greps)
    failing, fine = _adder_on(SERVER[0]), _adder_on(SERVER[1])

    p1, records, deferred = await run_round(tmp_path, panda, [failing, fine])
    assert p1.name == "records-r1.jsonl" and [r["ask"] for r in records] == [
        fine["ask"]
    ]
    (d,) = deferred
    assert d["ask"] == failing["ask"] and d["why"] == [
        f"{SERVER[0]} {day}: failed: timed out"
    ]
    (kept,) = tmp_path.glob("answers-r1-*/deferred.jsonl")
    assert [json.loads(x)["ask"] for x in kept.read_text().splitlines()] == [
        failing["ask"]
    ]

    p2, (r,) = await run(tmp_path, panda, [failing])
    assert p2.name == "records-r2.jsonl" and r["ask"] == failing["ask"]
    (f,) = r["machines"][SERVER[0]]["files"]
    assert f["status"] == "answered" and [x["offset"] for x in f["records"]] == [1]


async def test_a_round_whose_every_ask_is_deferred_writes_no_records(tmp_path):
    greps = {
        (SERVER[0], "panda-adder.log-20261014.gz"): [
            {"status": "failed", "error": "timed out"}
        ]
    }
    panda = FakePanda(lists_with({SERVER[0]: ADDER_FILES, SERVER[1]: []}), greps)
    path, records, (d,) = await run_round(tmp_path, panda, [_adder_on(SERVER[0])])
    assert path is None and records == [] and "failed: timed out" in d["why"][0]
    assert not list(tmp_path.glob("records-r*.jsonl"))
    assert list(tmp_path.glob("answers-r1-*/leaves.jsonl"))


async def test_an_ask_whose_machine_list_failed_is_deferred(tmp_path):
    lists = lists_with({SERVER[0]: ADDER_FILES, SERVER[1]: []})
    lists["server"][SERVER[0]] = {"status": "failed", "error": "no answer"}
    panda = FakePanda(lists, {})
    path, records, (d,) = await run_round(tmp_path, panda, [_adder_on(SERVER[0])])
    assert path is None and d["why"] == [f"{SERVER[0]}: list failed: no answer"]


DROPPED = (
    "PanDA connection error for 'async_process/submit_grep_request': ('Connection"
    " aborted.', RemoteDisconnected('Remote end closed connection without response'))"
)


def dropped(*_):
    raise RuntimeError(DROPPED)


async def test_an_ask_whose_grep_got_no_response_is_deferred_and_the_round_goes_on(
    tmp_path,
):
    # 2026-10-26: a connection dropped on a grep's submit raised out of the round, which wrote nothing
    day = "panda-adder.log-20261014.gz"
    greps = {
        (SERVER[0], day): [dropped, found(rec(1, "2026-10-14 06:00:00,1 a 7339"))],
        (SERVER[1], day): [found(rec(2, "2026-10-14 06:00:00,2 b 7339"))],
    }
    lists = lists_with({SERVER[0]: ADDER_FILES, SERVER[1]: ADDER_FILES})
    panda = FakePanda(lists, greps)
    failing, fine = _adder_on(SERVER[0]), _adder_on(SERVER[1])

    p1, records, (d,) = await run_round(tmp_path, panda, [failing, fine])
    assert p1.name == "records-r1.jsonl" and [r["ask"] for r in records] == [
        fine["ask"]
    ]
    assert d["ask"] == failing["ask"] and d["why"] == [
        f"{SERVER[0]} {day}: failed: no response: {DROPPED}"
    ]

    p2, (r,) = await run(tmp_path, panda, [failing])
    assert p2.name == "records-r2.jsonl" and r["ask"] == failing["ask"]


async def test_an_ask_whose_list_got_no_response_is_deferred(tmp_path):
    panda = FakePanda(lists_with({SERVER[0]: ADDER_FILES, SERVER[1]: []}), {})
    parts = panda.parts()

    async def list_dropped(**_):
        raise RuntimeError(DROPPED)

    parts.list_logs = list_dropped
    asking = ask(ADDER, ["7339"], ("2026-10-14 06:00:00", "2026-10-14 06:00:00"))
    path, records, (d,) = await R.record_round(
        [asking], tmp_path, parts, datetime.timedelta(seconds=3600)
    )
    assert path is None and records == [] and d["ask"] == asking["ask"]
    assert all(f"list failed: no response: {DROPPED}" in w for w in d["why"])
    assert {w.split(":")[0] for w in d["why"]} == {
        "every jedi machine",
        "every server machine",
    }


# ------------------------------------------------------------------ several cases in one round


async def run_cases(tmp_path, panda: FakePanda, cases: list, max_wait=3600):
    return await R.record_rounds(
        cases, panda.parts(), datetime.timedelta(seconds=max_wait), tmp_path / "shared"
    )


async def test_the_asks_of_several_cases_share_one_list_and_one_grep_and_each_case_keeps_its_own_round(
    tmp_path,
):
    reply = found(
        rec(1, "2026-10-14 06:00:00,1 a 111"),
        rec(2, "2026-10-14 06:00:01,1 b 222"),
    )
    greps = {
        (SERVER[0], "panda-adder.log-20261014.gz"): [reply],
        (SERVER[0], "panda-adder.log-20261013.gz"): [found()],
    }
    panda = FakePanda(lists_with({SERVER[0]: ADDER_FILES, SERVER[1]: []}), greps)
    w = ("2026-10-14 06:00:00", "2026-10-14 06:00:01")
    a, b = tmp_path / "a", tmp_path / "b"
    # b has a round of its own already: its next is its second
    await run(b, panda, [ask(ADDER, ["333"], ("2026-10-13 06:00:00",) * 2, SERVER[0])])
    calls_before = len(panda.calls)

    (pa, (ra,), _), (pb, (rb,), _) = await run_cases(
        tmp_path,
        panda,
        [
            ([ask(ADDER, ["111"], w, SERVER[0])], a),
            ([ask(ADDER, ["222"], w, SERVER[0])], b),
        ],
    )
    calls = panda.calls[calls_before:]
    assert [c for c in calls if c[0] == "grep"] == [
        ("grep", SERVER[0], "panda-adder.log-20261014.gz", ("111", "222"), None)
    ]
    assert (
        len([c for c in calls if c[0] == "list"]) == 2
    )  # once per service, for both cases
    assert (pa, pb) == (a / "records-r1.jsonl", b / "records-r2.jsonl")
    assert (ra["round"], rb["round"]) == (1, 2)
    fa, fb = (
        ra["machines"][SERVER[0]]["files"][0],
        rb["machines"][SERVER[0]]["files"][0],
    )
    assert [x["offset"] for x in fa["records"]] == [1]
    assert [x["offset"] for x in fb["records"]] == [2]
    # the answers are kept once, and each case's round says where
    (kept,) = (tmp_path / "shared").glob("answers-*/leaves.jsonl")
    for d, n in ((a, 1), (b, 2)):
        (pointer,) = d.glob(f"answers-r{n}-*/shared.json")
        assert json.loads(pointer.read_text()) == {"answers": str(kept.parent)}
        assert not list(d.glob(f"answers-r{n}-*/leaves.jsonl"))


async def test_a_later_round_of_one_case_reads_the_rotated_file_the_shared_round_answered(
    tmp_path,
):
    greps = {
        (SERVER[0], "panda-adder.log-20261014.gz"): [
            found(rec(1, "2026-10-14 06:00:00,1 a 7339"))
        ]
    }
    panda = FakePanda(lists_with({SERVER[0]: ADDER_FILES, SERVER[1]: []}), greps)
    a, b = tmp_path / "a", tmp_path / "b"
    first = _adder_on(SERVER[0])
    other = ask(ADDER, ["8000"], ("2026-10-14 06:00:00",) * 2, SERVER[0])
    await run_cases(tmp_path, panda, [([first], a), ([other], b)])
    wider = ask(
        ADDER, ["7339"], ("2026-10-14 05:00:00", "2026-10-14 06:00:00"), SERVER[0]
    )
    p2, (r,) = await run(a, panda, [wider])
    assert p2.name == "records-r2.jsonl"
    assert len([c for c in panda.calls if c[0] == "grep"]) == 1
    (f,) = r["machines"][SERVER[0]]["files"]
    assert [x["offset"] for x in f["records"]] == [1]
    assert r["requests"][0]["reused"].startswith("answers-")


async def test_a_deferred_ask_stays_in_its_own_cases_round(tmp_path):
    day = "panda-adder.log-20261014.gz"
    greps = {
        (SERVER[0], day): [found(rec(1, "2026-10-14 06:00:00,1 a 7339"))],
        (SERVER[1], day): [{"status": "failed", "error": "timeout"}],
    }
    panda = FakePanda(
        lists_with({SERVER[0]: ADDER_FILES, SERVER[1]: ADDER_FILES}), greps
    )
    a, b = tmp_path / "a", tmp_path / "b"
    (pa, ra, da), (pb, rb, (d,)) = await run_cases(
        tmp_path, panda, [([_adder_on(SERVER[0])], a), ([_adder_on(SERVER[1])], b)]
    )
    assert pa == a / "records-r1.jsonl" and len(ra) == 1 and da == []
    assert (
        pb is None and rb == [] and d["why"] == [f"{SERVER[1]} {day}: failed: timeout"]
    )
    assert not list(b.glob("records-r*.jsonl"))
    (kept,) = b.glob("answers-r1-*/deferred.jsonl")
    assert [json.loads(x)["ask"] for x in kept.read_text().splitlines()] == [d["ask"]]
    assert not list(a.glob("answers-r1-*/deferred.jsonl"))


async def test_several_cases_need_a_shared_directory(tmp_path):
    panda = FakePanda(lists_with({SERVER[0]: ADDER_FILES, SERVER[1]: []}), {})
    with pytest.raises(FieldTestError, match="shared directory"):
        await R.record_rounds(
            [([_adder_on(SERVER[0])], tmp_path / "a"), ([], tmp_path / "b")],
            panda.parts(),
            datetime.timedelta(seconds=60),
        )
