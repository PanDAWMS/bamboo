"""The recording step: lists, files and what they cover, greps, and one record per ask (fieldtest.record)."""

from __future__ import annotations

import datetime
import json

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
    """list_logs and grep_log over a dict of files per machine; each grep answers from *greps* by (machine, file)."""

    def __init__(
        self,
        lists: dict[str, dict[str, dict]],
        greps: dict[tuple[str, str], list],
        now=NOW,
    ) -> None:
        self.lists, self.greps = lists, greps
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
        self, log_filename, fixed_strings, *, machine=None, tail_bytes=None, **_
    ):
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


async def run(tmp_path, panda: FakePanda, asks: list[dict], max_wait=3600):
    return await R.record_round(
        asks, tmp_path, panda.parts(), datetime.timedelta(seconds=max_wait)
    )


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
