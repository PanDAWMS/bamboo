"""Tests for the status-history, db_read and log calls of bamboo.utils.panda_client.

A stand-in HttpClient answers by endpoint the way the server's handlers are
documented to (pandaserver/api/v1/job_api.py get_status_history,
async_process_api.py get_result): a refusal is ``success`` false with the reason
in ``data``, a poll carries ``async_meta`` at the top level, and a grep or log
list is answered per machine (panda-server 67fff963).
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from bamboo.utils import panda_client as PC


def _when(text: str) -> dict:
    return {"__datetime__": text}


def _patched(route):
    """Stub HttpClient so that get/post(url, data) return ``(0, route(endpoint, data))``.

    The calls are recorded as (endpoint, data); ``methods`` holds each call's method.
    """
    calls: list[tuple[str, dict]] = []
    methods: list[str] = []

    def answer(method):
        def call(url, data):
            endpoint = url.split("/api/v1/", 1)[1]
            calls.append((endpoint, data))
            methods.append(method)
            return 0, route(endpoint, data)

        return call

    instance = MagicMock()
    instance.get.side_effect = answer("get")
    instance.post.side_effect = answer("post")
    module = MagicMock()
    module.HttpClient = MagicMock(return_value=instance)
    module.api_url_ssl = "https://mock-panda.example.com/api/v1"
    _patched.methods = methods
    return patch.dict("sys.modules", {"pandaserver.api.v1.http_client": module}), calls


def _history_server(budget: int, partitions: int):
    """job/get_status_history refusing past budget // partitions, as the handler does."""

    def route(endpoint, data):
        ids = data["job_ids"]
        if len(ids) * partitions > budget:
            return {
                "success": False,
                "message": "over budget",
                "data": {
                    "rejected": "probe_budget",
                    "keys": len(ids),
                    "partitions": partitions,
                    "budget": budget,
                },
            }
        first = ids[0]
        return {
            "success": True,
            "message": "",
            "data": {
                "rows": [{"PandaID": i, "jobStatus": "defined"} for i in ids],
                "as_of": _when(f"2026-10-13T10:00:{first % 60:02d}"),
                "retained_from": _when(f"2026-07-{1 + first % 28:02d}T00:00:00"),
                "since": None,
                "until": None,
                "partitions": partitions,
                "budget": budget,
                "record_statuschange": True,
            },
        }

    return route


async def test_history_splits_by_the_budget_the_refusal_states():
    ctx, calls = _patched(_history_server(budget=1000, partitions=93))
    with ctx:
        got = await PC.get_job_status_history(list(range(100, 125)))
    # the first call names all 25 and is refused; the rest name 1000 // 93 = 10 each
    assert [len(d["job_ids"]) for _, d in calls] == [25, 10, 10, 5]
    assert got["success"] is True
    assert [r["PandaID"] for r in got["rows"]] == list(range(100, 125))
    assert (
        len(got["calls"]) == 4 and got["calls"][0]["data"]["rejected"] == "probe_budget"
    )


async def test_history_keeps_the_weakest_guarantee_of_its_calls():
    ctx, _ = _patched(_history_server(budget=1000, partitions=93))
    with ctx:
        got = await PC.get_job_status_history(list(range(100, 125)))
    # every change committed by the earliest as_of is in the rows of every call
    # (the answered calls start at 100, 110, 120: seconds 40, 50, 00)
    assert got["as_of"] == _when("2026-10-13T10:00:00")
    # and only the latest retained_from holds for all of them (days 17, 27, 09)
    assert got["retained_from"] == _when("2026-07-27T00:00:00")


async def test_history_posts_the_ids_and_names_at_most_what_one_in_list_holds():
    ctx, calls = _patched(_history_server(budget=10**6, partitions=1))
    with ctx:
        got = await PC.get_job_status_history(list(range(2500)))
    # a POST body, not the URL: a GET's request line holds about 420 IDs
    assert _patched.methods == ["post"] * 3
    assert [len(d["job_ids"]) for _, d in calls] == [1000, 1000, 500]
    assert got["success"] is True and len(got["rows"]) == 2500


async def test_history_asks_a_job_named_twice_once_and_passes_the_window():
    ctx, calls = _patched(_history_server(budget=1000, partitions=2))
    with ctx:
        await PC.get_job_status_history(["7", 7, 8], since="2026-10-01 00:00:00")
    assert calls == [
        ("job/get_status_history", {"job_ids": [7, 8], "since": "2026-10-01 00:00:00"})
    ]


async def test_history_returns_another_refusal_as_it_came():
    def route(endpoint, data):
        return {
            "success": False,
            "message": "ORA-00942",
            "data": {"error": "ORA-00942"},
        }

    ctx, calls = _patched(route)
    with ctx:
        got = await PC.get_job_status_history([1, 2])
    assert got["success"] is False and got["data"] == {"error": "ORA-00942"}
    assert len(calls) == 1 and got["calls"] == [route(None, None)]


async def test_history_gives_up_when_not_even_one_job_fits():
    ctx, calls = _patched(_history_server(budget=50, partitions=93))
    with ctx:
        got = await PC.get_job_status_history([1, 2, 3])
    assert len(calls) == 1
    assert got["success"] is False and got["data"]["rejected"] == "probe_budget"


async def test_db_read_polls_until_done_and_returns_the_answers():
    polls = iter(["pending", "running", "done"])

    def route(endpoint, data):
        if endpoint == "async_process/submit_db_read_request":
            return {
                "success": True,
                "message": "",
                "data": {"async_id": "a-1", "request_id": "a-1"},
            }
        status = next(polls)
        answers = {
            "as_of": _when("2026-10-13T10:00:00"),
            "answers": [{"status": "done", "truncated": False, "tables": []}],
        }
        return {
            "success": status == "done",
            "message": "",
            "data": answers if status == "done" else None,
            "async_meta": {"status": status, "attempts": 1},
        }

    ctx, calls = _patched(route)
    with ctx:
        got = await PC.db_read(
            [{"tables": ["jobsActive4"], "where": [["PandaID", "=", 1]]}],
            poll_seconds=0,
        )
    assert [e for e, _ in calls] == ["async_process/submit_db_read_request"] + [
        "async_process/get_result"
    ] * 3
    assert calls[0][1] == {
        "queries": [{"tables": ["jobsActive4"], "where": [["PandaID", "=", 1]]}]
    }
    assert calls[1][1] == {"async_id": "a-1"}
    assert got["async_id"] == "a-1" and got["async_meta"]["status"] == "done"
    assert got["data"]["answers"][0]["status"] == "done"


async def test_db_read_returns_a_refused_submit_and_a_failed_poll():
    def refused(endpoint, data):
        return {"success": False, "message": "DN not allowed", "data": None}

    ctx, calls = _patched(refused)
    with ctx:
        got = await PC.db_read([], poll_seconds=0)
    assert got["success"] is False and len(calls) == 1

    def not_found(endpoint, data):
        if endpoint.endswith("submit_db_read_request"):
            return {"success": True, "message": "", "data": {"async_id": "a-2"}}
        return {"success": False, "message": "not found", "data": None}

    ctx, calls = _patched(not_found)
    with ctx:
        got = await PC.db_read([], poll_seconds=0)
    assert got["success"] is False and got["async_id"] == "a-2" and len(calls) == 2


async def test_db_read_stops_waiting_and_names_the_request():
    def route(endpoint, data):
        if endpoint.endswith("submit_db_read_request"):
            return {"success": True, "message": "", "data": {"async_id": "a-3"}}
        return {
            "success": False,
            "message": "",
            "data": None,
            "async_meta": {"status": "running"},
        }

    ctx, _ = _patched(route)
    with ctx, pytest.raises(TimeoutError, match="a-3"):
        await PC.db_read([], poll_seconds=1, timeout_seconds=0)


async def test_task_history_returns_the_whole_response():
    def route(endpoint, data):
        assert (endpoint, data) == ("task/get_status_history", {"task_id": 40000001})
        return {"success": False, "message": "db", "data": {"error": "ORA-01013"}}

    ctx, _ = _patched(route)
    with ctx:
        got = await PC.get_task_status_history("40000001")
    assert got == {"success": False, "message": "db", "data": {"error": "ORA-01013"}}


def _per_machine_poll(expected, rows):
    complete = {
        r["machine_name"] for r in rows if r["status"] in ("done", "failed")
    } >= set(expected)
    return {
        "success": True,
        "message": "",
        "data": {
            "overall_status": "complete" if complete else "pending",
            "expected_machines": expected,
            "results": rows,
        },
    }


def _row(machine, status="done", result=None, **rest):
    return {
        "machine_name": machine,
        "status": status,
        "result": result,
        "truncated": 0,
        "error_msg": None,
        **rest,
    }


async def test_grep_sends_each_string_once_and_waits_for_every_machine():
    records = {
        "records": [
            {
                "offset": 120,
                "head": True,
                "lines": [
                    "2026-10-07 13:08:02,761 panda.log.adder: DEBUG    9000000001 new start"
                ],
                "cut": False,
            }
        ],
        "window": None,
    }
    polls = iter(
        [
            [_row("m1", "running")],
            [
                _row("m1", result=json.dumps(records)),
                _row(
                    "m2",
                    result=json.dumps({"records": [], "window": None}),
                    return_code=1,
                ),
            ],
        ]
    )

    def route(endpoint, data):
        if endpoint == "async_process/submit_grep_request":
            return {
                "success": True,
                "message": "",
                "data": {"async_id": "g-1", "request_id": "g-1"},
            }
        return _per_machine_poll(["m1", "m2"], next(polls))

    ctx, calls = _patched(route)
    with ctx:
        got = await PC.grep_log(
            "panda-adder.log-20261008.gz",
            ["9000000001", "9000000001"],
            service="server",
            poll_seconds=0,
        )
    assert calls[0] == (
        "async_process/submit_grep_request",
        {
            "log_filename": "panda-adder.log-20261008.gz",
            "fixed_strings": ["9000000001"],
            "whole_records": True,
            "service_name": "server",
        },
    )
    assert [e for e, _ in calls[1:]] == ["async_process/get_result"] * 2 and calls[1][
        1
    ] == {"async_id": "g-1"}
    assert got["async_id"] == "g-1" and got["data"]["overall_status"] == "complete"
    answers = PC.per_machine(got, json_result=True)
    assert (
        answers["m1"]["result"] == records and answers["m2"]["result"]["records"] == []
    )


async def test_grep_refuses_what_the_server_would_before_sending():
    ctx, calls = _patched(lambda e, d: pytest.fail("sent"))
    with ctx:
        for strings, target in (
            ([], {"service": "server"}),
            ([str(i) for i in range(1001)], {"service": "server"}),
            (["k"], {}),
            (["k"], {"service": "server", "machine": "m1"}),
        ):
            with pytest.raises(ValueError):
                await PC.grep_log("panda-adder.log", strings, **target)
    assert calls == []


def test_per_machine_tells_a_silent_machine_from_a_missing_file():
    response = _per_machine_poll(
        ["m1", "m2", "m3"],
        [
            _row(
                "m1",
                result=json.dumps(
                    {
                        "records": [],
                        "window": {"offset": 0, "first_stamp": "2026-10-13 09:00:00"},
                    }
                ),
            ),
            _row(
                "m2",
                "failed",
                error_msg="/var/log/panda/panda-adder.log-20261008.gz: No such file or directory",
            ),
        ],
    )
    got = PC.per_machine(response, json_result=True)
    assert got["m1"]["result"]["window"]["first_stamp"] == "2026-10-13 09:00:00"
    assert (
        got["m2"]["status"] == "failed"
        and "No such file" in got["m2"]["error"]
        and got["m2"]["result"] is None
    )
    assert got["m3"] == {"status": "no result"}


async def test_wait_returns_the_pending_answer_when_the_wait_ends_first():
    def route(endpoint, data):
        if endpoint.endswith("submit_log_list_request"):
            return {"success": True, "message": "", "data": {"async_id": "l-1"}}
        return _per_machine_poll(
            ["m1", "m2"],
            [
                _row(
                    "m1",
                    result=json.dumps(
                        [
                            {
                                "name": "panda-adder.log",
                                "size": 10,
                                "mtime": "2026-10-13 18:00:00",
                            }
                        ]
                    ),
                )
            ],
        )

    ctx, calls = _patched(route)
    with ctx:
        got = await PC.list_logs(machine="m1", poll_seconds=1, timeout_seconds=0)
    assert calls[0] == ("async_process/submit_log_list_request", {"machine_name": "m1"})
    assert got["async_id"] == "l-1" and got["data"]["overall_status"] == "pending"
    answers = PC.per_machine(got, json_result=True)
    assert answers["m1"]["result"][0]["name"] == "panda-adder.log" and answers[
        "m2"
    ] == {"status": "no result"}
