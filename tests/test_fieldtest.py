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

STOPPED = """\
# rerun: patterns=outside, decide=positive, hand=none
== H1 candidates
== H2 which ran
STOP H2: a run of 7 at 2026-10-01 00:00:00; HAND used 0 time(s) []
"""
ENDED = """\
# start: bamboo /x
== S1 what moves it on
== S5 what the call returned
END a legitimate wait: emptied at 09:04:17
== reached the end; HAND used 0 time(s) []
"""


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ------------------------------------------------------------------ score


def test_score_reads_a_stop():
    s = F.score(STOPPED)
    assert s["hops"] == ["H1", "H2"] and s["last_hop"] == "H2"
    assert s["terminal"] == "STOP" and s["stop"] == {
        "hop": "H2",
        "need": "a run of 7 at 2026-10-01 00:00:00",
    }
    assert s["hand"] == 0 and s["end"] is None


def test_score_reads_an_end():
    s = F.score(ENDED)
    assert (
        s["terminal"] == "END"
        and s["end"].startswith("a legitimate wait")
        and s["hand"] == 0
    )
    assert s["hops"] == ["S1", "S5"]


def test_score_without_a_terminal_line():
    assert (
        F.score("== H1 candidates\nTraceback (most recent call last):\n")["terminal"]
        == "NONE"
    )


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
    args = F.command(home, case, Path("/out.txt"))
    assert args[1:4] == ["-m", "bamboo.querier.rerun", "--stuck"]
    assert args[args.index("--key") + 1] == "5" and args[-2:] == ["--out", "/out.txt"]
    case["question"]["kind"] = "default"
    assert "--stuck" not in F.command(home, case, Path("/out.txt"))


def test_a_run_says_whether_it_read_the_pairs_dump():
    pair = {"MAP_DUMP": "/d/map.json", "MAP_DUMP_SHA256": "ab" * 32}
    assert F.map_said(
        f"# x\n# map: dump /d/map.json (sha256 {'ab' * 8})\n== H1\n", pair
    ) == {
        "said": f"dump /d/map.json (sha256 {'ab' * 8})",
        "ok": True,
    }
    assert not F.map_said("# map: the one stored in Neo4j\n", pair)["ok"]
    assert not F.map_said(f"# map: dump /d/map.json (sha256 {'cd' * 8})\n", pair)["ok"]
    assert F.map_said("Traceback (most recent call last):\n", pair) == {
        "said": None,
        "ok": False,
    }


# ------------------------------------------------------------------ a run, of a stand-in querier

#: a stand-in for bamboo.querier.rerun: writes a querier-shaped output naming the dump it was handed, and says
#: where it was imported from and which Neo4j it was given
FAKE_QUERIER = textwrap.dedent(
    """
    import hashlib, os, sys
    import bamboo
    out = sys.argv[sys.argv.index("--out") + 1]
    dump = os.environ["MAP_DUMP"]
    sha = hashlib.sha256(open(dump, "rb").read()).hexdigest()[:16]
    with open(out, "w") as f:
        f.write("# start: tree " + os.environ["BAMBOO_TREE"] + "\\n")
        f.write(f"# map: dump {dump} (sha256 {sha})\\n")
        f.write("== S1 what moves it on\\nEND a legitimate wait\\n== reached the end; HAND used 0 time(s) []\\n")
    print(__name__, bamboo.__file__.startswith(os.environ["BAMBOO_TREE"]), os.environ["NEO4J_URI"])
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
        and rec["baseline"]["lines_differ"] == 0
    )
    assert rec["map"]["ok"] and "MAP NOT" not in F.line(rec)
    assert (
        rec["querier_sha"] == "beefcafe0"
        and rec["env"]["PYTHONPATH"] == rec["env"]["BAMBOO_TREE"]
    )
    assert rec["argv"][1:3] == ["-m", "bamboo.querier.rerun"]
    stdout = (tmp_path / "batch" / "C1" / "stdout.log").read_text()
    assert stdout.split() == ["__main__", "True", F.NO_GRAPH["NEO4J_URI"]]
    assert (
        json.loads((tmp_path / "batch" / "C1" / "run.json").read_text())["name"] == "C1"
    )


def test_a_case_of_another_pin_does_not_run(tmp_path):
    home = _home(tmp_path)
    _pair(tmp_path, home)
    with pytest.raises(F.FieldTestError, match="was written by pin-other"):
        F.run_case(home, _case(tmp_path, pin="other"), tmp_path / "batch", "P")


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
    assert rec["exit"] == 0 and rec["argv"][rec["argv"].index("--rows") + 1] == str(
        built
    )
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
