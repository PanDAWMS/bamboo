"""Field tests -- ``bamboo.scripts.panda.fieldtest``.

The real cases run the querier against production logs and rows, which stay
out of the repository.  These checks need neither: the runner is driven by a
stand-in querier in a stand-in run tree that writes a querier-shaped output,
and the census by a small dump written here.
"""

from __future__ import annotations

import hashlib
import json
import textwrap
from pathlib import Path

import pytest

from bamboo.scripts.panda.fieldtest import census as C
from bamboo.scripts.panda.fieldtest import run as F

#: result.json as the querier writes it (version 1), the parts the runner reads
STOPPED = {
    "version": 1,
    "outcome": {
        "kind": "stop",
        "step": "H2",
        "text": "STOP H2: 2 ask(s) of the logs not in the records; HAND used 0 time(s) []",
        "code": "asks",
        "hand_used": [],
        "branches": None,
    },
    "steps": ["H1", "H2"],
    "asks": {"path": "/b/C1/asks.jsonl", "count": 2},
}
ENDED = {
    "version": 1,
    "outcome": {
        "kind": "end",
        "step": "S5",
        "text": "END a legitimate wait: emptied at 09:04:17",
        "code": None,
        "hand_used": ["S2", "S2"],
        "branches": None,
    },
    "steps": ["S1", "S2", "S5"],
    "asks": None,
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ------------------------------------------------------------------ score


def test_score_reads_a_stop():
    s = F.score(STOPPED)
    assert s["steps"] == ["H1", "H2"] and s["terminal"] == "STOP" and s["step"] == "H2"
    assert s["code"] == "asks" and s["text"].startswith("STOP H2: 2 ask(s)")
    assert s["hand"] == 0 and s["asks"]["count"] == 2 and s["why"] is None


def test_score_reads_an_end_and_counts_each_hand_used():
    s = F.score(ENDED)
    assert s["terminal"] == "END" and s["step"] == "S5" and s["code"] is None
    assert s["text"].startswith("END a legitimate wait") and s["asks"] is None
    assert s["hand"] == 2 and s["hand_used"] == ["S2", "S2"]


def test_a_run_without_a_result_of_this_version_is_not_scored():
    assert F.score(None)["terminal"] == "NONE"
    assert "did not finish" in F.score(None)["why"]
    s = F.score(STOPPED | {"version": 2})
    assert s["terminal"] == "NONE" and s["why"] == "result.json version 2, this reads 1"
    assert s["steps"] == [] and s["hand"] is None


def test_lines_differ_leaves_out_comment_lines():
    assert F.lines_differ("# a\nx\ny\n", "# b\nx\ny\n") == 0
    assert F.lines_differ("x\ny\n", "x\nz\n") == 2


# ------------------------------------------------------------------ home, cases, pairs


def _home(tmp_path: Path) -> F.Home:
    root = tmp_path / "home"
    (root / "pairs").mkdir(parents=True)
    (root / "fieldtest.json").write_text("{}")
    return F.Home.load(root)


def test_home_needs_a_directory_with_its_config(tmp_path, monkeypatch):
    monkeypatch.delenv("FIELDTEST_HOME", raising=False)
    with pytest.raises(F.FieldTestError, match="no home"):
        F.Home.load(None)
    with pytest.raises(F.FieldTestError, match="fieldtest.json does not exist"):
        F.Home.load(tmp_path)
    home = _home(tmp_path)
    assert home.cwd == home.root and home.python  # defaults: the home, this interpreter


def _case(tmp_path: Path, **over) -> Path:
    rows, log = tmp_path / "rows.json", tmp_path / "panda-x.log"
    rows.write_text("{}")
    log.write_text("")
    case = {
        "name": "C1",
        "question": {
            "kind": "stuck",
            "subject": "JediTaskSpec.status",
            "observed": "finishing",
            "key": "5",
        },
        "inputs": {"rows": str(rows), "logs": [str(log)]},
        "modes": {"patterns": "outside", "decide": "positive"},
        "pin": "abc-f2",
    } | over
    path = tmp_path / "case.json"
    path.write_text(json.dumps(case))
    return path


def test_a_case_says_its_kind_and_its_pin(tmp_path):
    with pytest.raises(F.FieldTestError, match="kind"):
        F.load_case(_case(tmp_path, question={"kind": "why"}))
    path = _case(tmp_path)
    case = json.loads(path.read_text())
    del case["pin"]
    path.write_text(json.dumps(case))
    with pytest.raises(F.FieldTestError, match="lacks \\['pin'\\]"):
        F.load_case(path)


def _write(d: Path, name: str, obj) -> Path:
    p = d / name
    p.write_text(json.dumps(obj) if not isinstance(obj, str) else obj)
    return p


def _pair(
    tmp_path: Path,
    home: F.Home,
    *,
    map_sha="1234abc",
    querier="beefcafe0",
    tree_q="beefcaf",
    derived=None,
    sha=None,
    name="P",
) -> dict:
    pin = tmp_path / "pin-abc-f2"
    pin.mkdir(exist_ok=True)
    dump = tmp_path / f"map-abc-f2-{map_sha}.json"
    dump.write_text(
        json.dumps({"map_id": "panda", "derived_from": f"source-root:{derived or pin}"})
    )
    tree = tmp_path / (f"tree-{map_sha}-q{tree_q}" if tree_q else f"tree-{map_sha}")
    tree.mkdir(exist_ok=True)
    if querier:
        (tree / "QUERIER").write_text(querier + "\n")
    pair = {
        "PANDA_PIN": str(pin),
        "BAMBOO_TREE": str(tree),
        "MAP_DUMP": str(dump),
        "MAP_DUMP_SHA256": sha or _sha(dump),
    }
    _write(home.root / "pairs", f"{name}.json", pair)
    return pair


def test_a_pair_is_checked_like_run_baselines(tmp_path):
    home = _home(tmp_path)
    _pair(tmp_path, home)
    assert F.load_pair(home, "P")["QUERIER_SHA"] == "beefcafe0"


@pytest.mark.parametrize(
    ("over", "message"),
    [
        ({"sha": "0" * 64}, "has sha256"),
        ({"querier": ""}, "no QUERIER file"),
        ({"querier": "0123456"}, "named for querier"),
        ({"derived": "/elsewhere"}, "derived from"),
    ],
)
def test_a_pair_that_does_not_hold_together_stops(tmp_path, over, message):
    home = _home(tmp_path)
    _pair(tmp_path, home, **over)
    with pytest.raises(F.FieldTestError, match=message):
        F.load_pair(home, "P")


def test_a_dump_is_read_with_the_tree_of_its_bamboo_sha(tmp_path):
    home = _home(tmp_path)
    pair = _pair(tmp_path, home)
    other = tmp_path / "tree-9999999-qbeefcaf"
    other.mkdir()
    (other / "QUERIER").write_text("beefcafe0")
    _write(home.root / "pairs", "P.json", pair | {"BAMBOO_TREE": str(other)})
    with pytest.raises(F.FieldTestError, match="is read with tree-1234abc"):
        F.load_pair(home, "P")


def test_a_case_runs_on_the_pair_current_names_for_its_pin(tmp_path):
    home = _home(tmp_path)
    case = F.load_case(_case(tmp_path))
    _write(home.root / "pairs", "current.json", {"abc-f2": "P"})
    assert F.pair_name(home, case) == "P"
    assert F.pair_name(home, case, "Q") == "Q"
    _write(home.root / "pairs", "current.json", {})
    with pytest.raises(F.FieldTestError, match="no pair for pin abc-f2"):
        F.pair_name(home, case)


def test_the_command_carries_the_question(tmp_path):
    home = _home(tmp_path)
    case = F.load_case(_case(tmp_path))
    args = F.command(home, case, Path("/b/C1/out.txt"))
    assert args[1:4] == ["-m", "bamboo.querier.rerun", "--stuck"]
    assert args[args.index("--key") + 1] == "5"
    assert args[-6:] == [
        "--out",
        "/b/C1/out.txt",
        "--result",
        "/b/C1/result.json",
        "--asks",
        "/b/C1/asks.jsonl",
    ]
    case["question"]["kind"] = "default"
    assert "--stuck" not in F.command(home, case, Path("/out.txt"))


def test_a_run_says_whether_it_read_what_it_was_handed():
    pair = {
        "MAP_DUMP": "/d/map.json",
        "MAP_DUMP_SHA256": "ab" * 32,
        "BAMBOO_TREE": "/t/tree-1234abc-qbeefcaf",
        "QUERIER_SHA": "beefcafe0",
    }
    handed = {
        "rows": {"path": "/r/rows.json", "sha256": "01" * 32},
        "logs": [{"path": "/l/panda-x.log", "sha256": "02" * 32}],
        "records": None,
    }
    inputs = {
        "querier": {
            "tree": "/t/tree-1234abc-qbeefcaf",
            "label": "beefcafe0",
            "bamboo_py_sha256": "03" * 32,
        },
        "map_dump": {"path": "/d/map.json", "sha256": "ab" * 32},
        "rows": handed["rows"],
        "logs": handed["logs"],
        "records": None,
    }
    result = {"version": 1, "inputs": inputs}
    assert F.read_as_handed(result, pair, handed) == {
        "ok": True,
        "differ": [],
        "bamboo_py_sha256": "03" * 32,
    }
    other_dump = inputs | {"map_dump": {"path": "/d/map.json", "sha256": "cd" * 32}}
    other_label = inputs | {"querier": inputs["querier"] | {"label": "0123456"}}
    no_logs = inputs | {"logs": None}
    for bent, differ in (
        (other_dump, ["map_dump"]),
        (inputs | {"map_dump": None}, ["map_dump"]),
        (other_label, ["label"]),
        (no_logs, ["logs"]),
    ):
        assert (
            F.read_as_handed({"version": 1, "inputs": bent}, pair, handed)["differ"]
            == differ
        )
    assert F.read_as_handed(None, pair, handed) == {
        "ok": False,
        "differ": ["result.json"],
        "bamboo_py_sha256": None,
    }


# ------------------------------------------------------------------ a run, of a stand-in querier

#: a stand-in for bamboo.querier.rerun: writes a querier-shaped output and a result.json naming what it was handed
#: (none where FAKE_NO_RESULT is set), and says where it was imported from and which Neo4j it was given
FAKE_QUERIER = textwrap.dedent(
    """
    import hashlib, json, os, sys
    from pathlib import Path
    import bamboo
    def arg(name):
        return sys.argv[sys.argv.index(name) + 1]
    def sha(p):
        return hashlib.sha256(open(p, "rb").read()).hexdigest()
    logs = [sys.argv[i + 1] for i, a in enumerate(sys.argv) if a == "--log"]
    tree, dump = os.environ["BAMBOO_TREE"], os.environ["MAP_DUMP"]
    with open(arg("--out"), "w") as f:
        f.write("# start: tree " + tree + "\\n")
        f.write("== S1 what moves it on\\nEND a legitimate wait\\n== reached the end; HAND used 0 time(s) []\\n")
    # through the records door: stop on one ask until a round's records are there, then end
    records = arg("--records") if "--records" in sys.argv else None
    files = sorted(Path(records).glob("records-r*.jsonl")) if records else []
    asking = records is not None and not files
    if asking:
        Path(arg("--asks")).write_text(json.dumps({"ask": {"name": "panda-x.log"}}) + "\\n")
    if not os.environ.get("FAKE_NO_RESULT"):
        outcome = (
            {"kind": "stop", "step": "S1", "text": "STOP S1: 1 ask(s)", "code": "asks"}
            if asking
            else {"kind": "end", "step": "S1", "text": "END a legitimate wait", "code": None}
        )
        result = {
            "version": 1,
            "outcome": outcome | {"hand_used": [], "branches": None},
            "steps": ["S1"],
            "inputs": {
                "querier": {"tree": tree, "label": Path(tree, "QUERIER").read_text().strip(),
                            "bamboo_py_sha256": "0" * 64},
                "map_dump": {"path": dump, "sha256": sha(dump)},
                "rows": {"path": arg("--rows"), "sha256": sha(arg("--rows"))},
                "records": {"given": [records], "files": [{"path": str(f), "sha256": sha(f)} for f in files]}
                if records else None,
                "logs": [{"path": p, "sha256": sha(p)} for p in logs] or None,
            },
            "asks": {"path": arg("--asks"), "count": 1} if asking else None,
        }
        Path(arg("--result")).write_text(json.dumps(result))
    print(__name__, bamboo.__file__.startswith(tree), os.environ["NEO4J_URI"])
    """
)


def _querier_tree(pair: dict) -> None:
    pkg = Path(pair["BAMBOO_TREE"]) / "bamboo" / "querier"
    pkg.mkdir(parents=True)
    (pkg.parent / "__init__.py").write_text("")
    (pkg / "__init__.py").write_text("")
    (pkg / "rerun.py").write_text(FAKE_QUERIER)


def test_a_case_runs_the_querier_of_its_pair_and_is_scored(tmp_path):
    home = _home(tmp_path)
    _querier_tree(_pair(tmp_path, home))
    _write(home.root / "pairs", "current.json", {"abc-f2": "P"})
    baseline = _write(
        tmp_path,
        "base.txt",
        "# other header\n== S1 what moves it on\nEND a legitimate wait\n"
        "== reached the end; HAND used 0 time(s) []\n",
    )
    case_path = _case(tmp_path, baseline=str(baseline))

    rec = F.run_case(home, case_path, tmp_path / "batch")
    assert (
        rec["exit"] == 0
        and rec["score"]["terminal"] == "END"
        and rec["score"]["hand"] == 0
        and rec["baseline"]["lines_differ"] == 0
    )
    assert rec["read"]["ok"] and "NOT READ AS HANDED" not in F.line(rec)
    assert rec["result"]["path"] == str(
        tmp_path / "batch" / "C1" / "r1" / "result.json"
    )
    assert (
        rec["querier_sha"] == "beefcafe0"
        and rec["env"]["PYTHONPATH"] == rec["env"]["BAMBOO_TREE"]
    )
    assert len(rec["rounds"]) == 1 and rec["ended"] == "one round (no --record)"
    assert rec["rounds"][0]["argv"][1:3] == ["-m", "bamboo.querier.rerun"]
    assert rec["pin_settled"] and "rounds" not in F.line(rec)
    stdout = (tmp_path / "batch" / "C1" / "r1" / "stdout.log").read_text()
    assert stdout.split() == ["__main__", "True", F.NO_GRAPH["NEO4J_URI"]]
    assert (
        json.loads((tmp_path / "batch" / "C1" / "run.json").read_text())["name"] == "C1"
    )


def test_a_run_that_writes_no_result_is_not_scored_from_an_earlier_one(
    tmp_path, monkeypatch
):
    home = _home(tmp_path)
    _querier_tree(_pair(tmp_path, home))
    _write(home.root / "pairs", "current.json", {"abc-f2": "P"})
    case_path = _case(tmp_path)
    first = F.run_case(home, case_path, tmp_path / "batch")
    assert first["score"]["terminal"] == "END"

    monkeypatch.setenv("FAKE_NO_RESULT", "1")
    rec = F.run_case(home, case_path, tmp_path / "batch")
    assert rec["result"] is None and rec["score"]["terminal"] == "NONE"
    assert rec["read"] == {
        "ok": False,
        "differ": ["result.json"],
        "bamboo_py_sha256": None,
    }
    assert "did not finish" in F.line(
        rec
    ) and "NOT READ AS HANDED: result.json" in F.line(rec)


def test_a_case_of_another_pin_does_not_run(tmp_path):
    home = _home(tmp_path)
    _pair(tmp_path, home)
    with pytest.raises(F.FieldTestError, match="was written by pin-other"):
        F.run_case(home, _case(tmp_path, pin="other"), tmp_path / "batch", "P")


def test_a_case_whose_pin_is_unsettled_runs_only_on_a_pair_named_for_it(tmp_path):
    home = _home(tmp_path)
    _querier_tree(_pair(tmp_path, home))
    _write(home.root / "pairs", "current.json", {"abc-f2": "P"})
    case_path = _case(tmp_path, pin="unsettled: the version is not known")
    with pytest.raises(F.FieldTestError, match="name the pair it runs on with --pair"):
        F.run_case(home, case_path, tmp_path / "batch")
    rec = F.run_case(home, case_path, tmp_path / "batch", "P")
    assert rec["pair"] == "P" and not rec["pin_settled"] and rec["read"]["ok"]
    assert "pin unsettled" in F.line(rec)


def _records_case(tmp_path: Path) -> Path:
    records = tmp_path / "records"
    records.mkdir()
    return _case(
        tmp_path,
        inputs={
            "rows": str(_write(tmp_path, "rows.json", "{}")),
            "logs": [],
            "records": str(records),
        },
    )


#: a stand-in for the recording step: FAKE_RECORD says what it does -- "write" a round, write "nothing", or exit 3
#: as one that did not settle
FAKE_RECORD = textwrap.dedent(
    """
    import os, sys
    from pathlib import Path
    asks, records = sys.argv[1], Path(sys.argv[2])
    assert Path(asks).read_text()
    how = os.environ["FAKE_RECORD"]
    if how == "write":
        n = len(list(records.glob("records-r*.jsonl"))) + 1
        (records / f"records-r{n}.jsonl").write_text('{"ask": {"name": "panda-x.log"}}\\n')
    sys.exit(3 if how == "unsettled" else 0)
    """
)


@pytest.fixture
def recording(tmp_path, monkeypatch):
    """A home whose querier asks through a records directory, and a recording step stood in for."""
    home = _home(tmp_path)
    _querier_tree(_pair(tmp_path, home))
    _write(home.root / "pairs", "current.json", {"abc-f2": "P"})
    script = _write(tmp_path, "fake_record.py", FAKE_RECORD)
    monkeypatch.setattr(
        F,
        "record_command",
        lambda home, asks, records: [home.python, str(script), str(asks), str(records)],
    )
    return home


def test_a_recorded_run_answers_the_asks_and_runs_the_querier_again(
    tmp_path, recording, monkeypatch
):
    monkeypatch.setenv("FAKE_RECORD", "write")
    rec = F.run_case(
        recording, _records_case(tmp_path), tmp_path / "batch", record=True
    )
    first, second = rec["rounds"]
    assert first["score"]["code"] == "asks" and first["record"]["exit"] == 0
    assert first["record"]["written"] == [
        str(tmp_path / "records" / "records-r1.jsonl")
    ]
    assert rec["ended"] == "the querier asks nothing more" and "record" not in second
    assert (
        rec["score"]["terminal"] == "END" and rec["read"]["ok"] and first["read"]["ok"]
    )
    assert "rounds 2: the querier asks nothing more" in F.line(rec)


def test_a_recorded_run_stops_where_the_recording_step_does_not_settle(
    tmp_path, recording, monkeypatch
):
    monkeypatch.setenv("FAKE_RECORD", "unsettled")
    rec = F.run_case(
        recording, _records_case(tmp_path), tmp_path / "batch", record=True
    )
    assert len(rec["rounds"]) == 1 and rec["ended"] == "the recording step exited 3"
    assert (
        rec["score"]["code"] == "asks" and rec["rounds"][0]["record"]["written"] == []
    )


def test_a_recorded_run_stops_after_its_rounds(tmp_path, recording, monkeypatch):
    monkeypatch.setenv("FAKE_RECORD", "nothing")
    rec = F.run_case(
        recording, _records_case(tmp_path), tmp_path / "batch", record=True, rounds=2
    )
    assert len(rec["rounds"]) == 2 and rec["ended"] == "still asking after 2 round(s)"


def test_without_record_a_run_that_asks_is_one_round(tmp_path, recording):
    rec = F.run_case(recording, _records_case(tmp_path), tmp_path / "batch")
    assert len(rec["rounds"]) == 1 and rec["score"]["code"] == "asks"
    assert rec["read"]["ok"] and rec["rounds"][0]["argv"].count("--records") == 1


def test_a_case_hands_logs_or_records_and_records_to_record_into(tmp_path):
    home = _home(tmp_path)
    _pair(tmp_path, home)
    _write(home.root / "pairs", "current.json", {"abc-f2": "P"})
    rows = str(_write(tmp_path, "rows.json", "{}"))
    both = _case(
        tmp_path, inputs={"rows": rows, "logs": [rows], "records": str(tmp_path)}
    )
    with pytest.raises(F.FieldTestError, match="not both"):
        F.load_case(both)
    with pytest.raises(F.FieldTestError, match="--record needs inputs.records"):
        F.run_case(home, _case(tmp_path), tmp_path / "batch", record=True)


# ------------------------------------------------------------------ census

DUMP = {
    "derived_from": "source-root:/pin",
    "subjects": [
        {
            "spec_class": "JobSpec",
            "attribute": "jobStatus",
            "updated_by": {"failed": 1, "closed": 1},
            "selected_by": ["a"],
        },
        {
            "spec_class": "JediFileSpec",
            "attribute": "status",
            "updated_by": {"ready": 1},
            "selected_by": [],
        },
        {
            "spec_class": "JobSpec",
            "attribute": "cloud",
            "updated_by": {},
            "selected_by": [],
        },
    ],
    "entities": [
        {
            "name": "jobsactive4",
            "spec_class": "JobSpec",
            "tables": ["jobsActive4", "jobsactive4"],
        },
        {"name": "jobsdefined4", "spec_class": "JobSpec", "tables": ["jobsDefined4"]},
        {"name": "users", "spec_class": None, "tables": ["users"]},
    ],
    "read_sites": [
        {"tables": ["jobsActive4"], "keys": ["JobSpec.PandaID"]},
        {"tables": ["jobsactive4", "jobsDefined4"], "keys": ["JobSpec.jediTaskID"]},
        {"tables": ["CONFIG"], "keys": []},
        {"tables": ["users"], "keys": ["users.name"]},
    ],
    "reading_stops": [{"reason": "r1"}],
    "function_prints": [
        {
            "owner": "pandaserver/a.py::f",
            "statements": [{"log_files_basis": "logger", "log_files": ["panda-A.log"]}],
        },
        {
            "owner": "pandajedi/b.py::g",
            "statements": [{"log_files_basis": "module", "log_files": ["panda-A.log"]}],
        },
    ],
    "boundaries": [{"system": "pilot", "kind": "reports_state"}],
}


def test_census_says_which_written_subjects_have_a_history():
    h = C.histories(DUMP)
    assert h["written_subjects"] == 2 and h["with_history"] == 1
    assert {r["subject"]: r["history"] for r in h["rows"]} == {
        "JediFileSpec.status": None,
        "JobSpec.jobStatus": "jobs_StatusLog",
    }


def test_census_counts_an_entitys_spellings_as_one_table():
    assert C.table_reads(DUMP) == {
        "jobsActive4": 2,
        "jobsDefined4": 1,
        "CONFIG": 1,
        "users": 1,
    }


def test_a_whitelist_reads_a_site_only_when_it_holds_every_table():
    assert C.covered(DUMP, ["jobsActive4"]) == 1
    assert C.covered(DUMP, ["jobsactive4", "jobsDefined4"]) == 2
    wl = C.whitelist(DUMP, 2, ["users"])
    assert (
        wl["tables"][0] == "jobsActive4"
        and "users" not in wl["tables"]
        and wl["read_sites"] == 4
    )


def test_census_reports_logs_printed_from_both_packages(capsys):
    out = C.census(DUMP)
    assert out["L"]["files_printed_from_both_packages"] == ["panda-A.log"]
    assert out["R"]["tables_not_entities"] == [("CONFIG", 1)]
    assert out["R"]["classes"] == ["JobSpec"]
    C.report(out)
    assert "PanDA keeps a history of 1" in capsys.readouterr().out


def test_among_tables_read_equally_often_the_one_that_completes_more_reads_comes_first():
    # A is read 4 times; B, C and D twice each; E once.  With A taken, C completes the
    # site [A, C] and B or D complete nothing, so C comes before B although B sorts first.
    sites = [["A"], ["A"], ["A"], ["A", "C"], ["B", "D"], ["B", "D"], ["C", "E"]]
    dump = {"entities": [], "read_sites": [{"tables": t, "keys": []} for t in sites]}
    assert dict(C.table_reads(dump)) == {"A": 4, "B": 2, "C": 2, "D": 2, "E": 1}
    assert C.ranked(dump, []) == ["A", "C", "B", "D", "E"]
    assert C.whitelist(dump, 2, [])["covered"] == 4
    assert C.ranked(dump, ["C"])[:2] == ["A", "B"]


def test_a_case_with_answers_runs_on_the_rows_file_built_from_them(
    tmp_path, monkeypatch
):
    from bamboo.scripts.panda.fieldtest import rows as RB

    home = _home(tmp_path)
    _querier_tree(_pair(tmp_path, home))
    _write(home.root / "pairs", "current.json", {"abc-f2": "P"})
    record = {
        "question": {
            "kind": "stuck",
            "subject": "JediTaskSpec.status",
            "observed": "finishing",
            "key": "5",
        },
        "key_class": "JediTaskSpec",
        "server": "https://panda.example/api/v1",
        "started": "2026-10-13T10:00:00+00:00",
        "steps": [],
    }
    answers = _write(tmp_path, "answers.json", record)
    monkeypatch.setattr(RB, "column_spellings", lambda pin, dump: {"JediTaskSpec": {}})
    path = _case(tmp_path)
    case = json.loads(path.read_text())
    del case["inputs"]["rows"]
    case["inputs"]["answers"] = str(answers)
    path.write_text(json.dumps(case))

    rec = F.run_case(home, path, tmp_path / "batch")
    built = tmp_path / "batch" / "C1" / "rows.json"
    argv = rec["rounds"][0]["argv"]
    assert rec["exit"] == 0 and argv[argv.index("--rows") + 1] == str(built)
    assert set(rec["inputs_sha256"]) == {
        str(built),
        case["inputs"]["logs"][0],
        str(answers),
    }
    # nothing was fetched: every section the querier reads is left out, and said
    assert "Left out: statuslog" in json.loads(built.read_text())["about"]


def test_a_case_needs_rows_or_answers(tmp_path):
    path = _case(tmp_path)
    case = json.loads(path.read_text())
    del case["inputs"]["rows"]
    path.write_text(json.dumps(case))
    with pytest.raises(F.FieldTestError, match="inputs need rows, or answers"):
        F.load_case(path)


def test_cases_of_one_name_do_not_run_into_one_directory(tmp_path):
    one = _case(tmp_path)
    (tmp_path / "other").mkdir()
    two = _case(tmp_path / "other")
    assert F.distinct_names([one]) == ["C1"]
    with pytest.raises(F.FieldTestError, match="C1: .*case.json, .*other/case.json"):
        F.distinct_names([one, two])
    home = tmp_path / "home"
    (home / "pairs").mkdir(parents=True)
    (home / "fieldtest.json").write_text("{}")
    assert F.main(["--home", str(home), "run", str(one), str(two)]) == 2
    assert not (home / "runs").exists()
