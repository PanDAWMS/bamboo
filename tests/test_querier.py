"""The querier's package -- ``bamboo.querier``.

Its modules read their environment when imported: ``PANDA_PIN`` names the
corpus (the PanDA source a map was built from) and goes on ``sys.path``, and
``BAMBOO_TREE`` must agree with where ``bamboo`` came from.  So every check
here runs in a child interpreter, with a temporary corpus, and leaves this
process as it was.  The case runs against production logs and rows are kept
out of the repository; these tests need neither.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MODULES = [
    "asked", "callers", "census", "census_p", "evaluate", "false_guard", "handlers", "held", "mapdump", "outbound", "parts",
    "printed", "raised", "recorded", "records", "reproduce", "rerun", "rows", "value_diff", "which_ran", "windows",
]


def _corpus(tmp_path: Path) -> Path:
    corpus = tmp_path / "corpus"
    for package in ("pandaserver", "pandajedi"):
        (corpus / package).mkdir(parents=True, exist_ok=True)
    return corpus


def _run(tmp_path: Path, code: str, **env: str | None) -> subprocess.CompletedProcess[str]:
    """Run *code* in a child interpreter that imports bamboo from this repository."""
    child = {k: v for k, v in os.environ.items() if k not in ("BAMBOO_TREE", "PANDA_PIN", "PYTHONPATH")}
    child |= {"PYTHONPATH": str(REPO), "PYTHONDONTWRITEBYTECODE": "1", "PANDA_PIN": str(_corpus(tmp_path))}
    for k, v in env.items():
        if v is None:
            child.pop(k, None)
        else:
            child[k] = v
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env=child,
        timeout=300,
    )


def _ok(done: subprocess.CompletedProcess[str]) -> str:
    assert done.returncode == 0, done.stderr[-3000:]
    return done.stdout


def test_every_module_imports_from_its_own_tree(tmp_path):
    out = _ok(
        _run(
            tmp_path,
            f"""
            import importlib
            for name in {MODULES!r}:
                importlib.import_module("bamboo.querier." + name)
            import bamboo.codemap.trace as trace
            import bamboo.querier as q
            print(q.TREE)
            print(trace.__file__)
            """,
        )
    )
    tree, trace_file = out.split()
    assert Path(tree) == REPO
    assert Path(trace_file).is_relative_to(REPO)


def test_a_tree_named_elsewhere_stops_the_import(tmp_path):
    done = _run(tmp_path, "import bamboo.querier", BAMBOO_TREE=str(tmp_path))
    assert done.returncode != 0
    assert "not BAMBOO_TREE=" in done.stderr


def test_the_same_tree_named_passes(tmp_path):
    _ok(_run(tmp_path, "import bamboo.querier", BAMBOO_TREE=str(REPO)))


def test_no_corpus_no_run(tmp_path):
    done = _run(tmp_path, "import bamboo.querier.records", PANDA_PIN=None)
    assert done.returncode != 0
    assert "set PANDA_PIN" in done.stderr


def test_the_header_names_the_dump_installed_not_the_one_asked_for(tmp_path):
    fragment = json.dumps({"map_id": "panda", "derived_from": f"source-root:{_corpus(tmp_path)}"})
    dump, other = tmp_path / "map.json", tmp_path / "other.json"
    dump.write_text(fragment)
    other.write_text(fragment)
    out = _ok(
        _run(
            tmp_path,
            f"""
            import json, os
            from bamboo.querier import mapdump

            asked = mapdump.described()
            mapdump.install_from_env()
            once = mapdump.described()
            mapdump.install(os.environ["MAP_DUMP"])
            try:
                mapdump.install({str(other)!r})
                refused = ""
            except RuntimeError as e:
                refused = str(e)
            print(json.dumps([asked, once, mapdump.described(), refused]))
            """,
            MAP_DUMP=str(dump),
        )
    )
    asked, once, twice, refused = json.loads(out)
    # MAP_DUMP alone swaps nothing: until a dump is installed, the graph client is Neo4j's
    assert asked == "the one stored in Neo4j"
    # a driver's install and the entry's are one install
    assert once == twice == f"dump {dump} (sha256 {hashlib.sha256(dump.read_bytes()).hexdigest()[:16]})"
    assert "installed already" in refused


def test_no_dump_no_map(tmp_path):
    # the map lives in dumps only: an entry without MAP_DUMP stops, and a reader with no dump installed stops before
    # it opens Neo4j's client, whose empty store would read as "nothing writes it"
    for unset in (None, ""):
        done = _run(tmp_path, "from bamboo.querier import mapdump; mapdump.install_from_env()", MAP_DUMP=unset)
        assert done.returncode != 0
        assert "set MAP_DUMP" in done.stderr
    out = _ok(
        _run(
            tmp_path,
            """
            from bamboo.database import graph_database_client
            from bamboo.querier import rows

            class Refused:
                def __init__(self, *a, **k):
                    raise AssertionError("a graph client was opened")

            graph_database_client.GraphDatabaseClient = Refused
            try:
                rows._function_prints("pandaserver/x.py::f")
            except RuntimeError as e:
                print(e)
            """,
            MAP_DUMP=None,
        )
    )
    assert "no map dump is installed" in out


def test_handed_to_binds_arguments_and_states_defaults(tmp_path):
    out = _ok(
        _run(
            tmp_path,
            """
            import ast, json
            from bamboo.querier.rerun import handed_to

            func = ast.parse("def m(self, a, b=1, c=2, *, d=3, e=None): pass").body[0]

            def bound(call):
                return {k: ast.unparse(v) for k, v in handed_to(func, ast.parse(call, mode="eval").body).items()}

            print(json.dumps([bound("x.m(5, c=7)"), bound("x.m(5, *rest)"), bound("x.m(**kw)")]))
            """,
        )
    )
    plain, starred, spread = json.loads(out)
    # what the call writes wins, the rest takes the default the method states (``e=None`` states one)
    assert plain == {"a": "5", "b": "1", "c": "7", "d": "3", "e": "None"}
    # what a * or ** spreads may fill any parameter: no default is stated for them
    assert starred == {"a": "5"}
    assert spread == {}


def test_evaluate_knows_what_the_env_holds_and_nothing_else(tmp_path):
    out = _ok(
        _run(
            tmp_path,
            """
            import json
            from bamboo.querier import evaluate as E

            env = E.Env()
            env.put("x", 3, "test")
            env.rows = {"self.job": "job", "job": "job"}
            env.put("self.job.status", "failed", "row")
            print(json.dumps([
                E.evaluate("x == 3", env),
                E.evaluate("job.status == 'failed'", env),
                E.evaluate("y == 1", env) is E.UNKNOWN,
                E.evaluate("x > 2", env) is E.UNKNOWN,
                env.used,
            ]))
            """,
        )
    )
    equal, by_other_spelling, unbound, ordered, used = json.loads(out)
    assert equal is True
    # ``self.job`` and ``job`` are one row once the question binds both spellings to it
    assert by_other_spelling is True
    assert unbound is True
    # only what can be decided exactly is: an ordering comparison stays unknown
    assert ordered is True
    assert used == {"x": "test", "job.status": "row"}


def test_solve_settles_the_unknown_atom_the_outcome_forces(tmp_path):
    out = _ok(
        _run(
            tmp_path,
            """
            import json
            from bamboo.querier import evaluate as E

            env = E.Env()
            env.put("x", 3, "test")
            s = E.solve("x == 3 and y == 1", True, env)
            t = E.solve("x == 3 or y == 1", True, env)
            print(json.dumps([s.known, s.settled, s.open, s.derived.get("y"), s.consistent, t.settled, t.open]))
            """,
        )
    )
    known, settled, open_, derived, consistent, or_settled, or_open = json.loads(out)
    assert known == {"x == 3": True}
    assert settled == {"y == 1": True}
    assert open_ == []
    # an equality the outcome forces true says what the unknown side held
    assert derived[0] == 1
    assert consistent is True
    # with ``x == 3`` already true, ``or`` forces nothing on y
    assert or_settled == {}
    assert or_open == ["y == 1"]


def test_runs_are_cut_by_the_instance_token_and_the_key_is_a_whole_token(tmp_path):
    log = tmp_path / "panda-adder.log"
    log.write_text(
        "2026-01-01 00:00:00,000 panda.log.Adder: DEBUG    <PandaID=12345> start\n"
        "2026-01-01 00:00:01,000 panda.log.Adder: ERROR    <PandaID=12345> failed with x\n"
        "Traceback (most recent call last):\n"
        "2026-01-01 00:00:02,000 panda.log.Adder: DEBUG    <PandaID=999> start for 123456\n"
        "2026-01-01 00:00:03,000 panda.log.Adder: INFO    <PandaID=12345> end\n"
    )
    out = _ok(
        _run(
            tmp_path,
            f"""
            import json
            from bamboo.querier.records import read_log, runs_for

            log = read_log({str(log)!r})
            runs = runs_for(log, "12345")
            print(json.dumps([
                [(r.token, [x.lineno for x in r.records], [x.lineno for x in r.found_by]) for r in runs],
                log.records[1].continuation,
                runs[0].would_print("ERROR"),
            ]))
            """,
        )
    )
    runs, continuation, would_print = json.loads(out)
    # 123456 is not the key 12345: the other job's line is not a hit
    assert runs == [["<PandaID=12345>", [1, 2, 5], [1, 2, 5]]]
    assert continuation == ["Traceback (most recent call last):"]
    # a run that printed DEBUG lines had ERROR switched on
    assert would_print is True


def test_the_file_door_answers_each_log_with_its_file(tmp_path):
    line = "2026-01-01 00:00:00,000 panda.log.Adder: DEBUG    <PandaID=12345> start\n"
    for name in ("panda-adder.log", "panda-setupper.log.12345", "panda-adder.log.1"):
        (tmp_path / name).write_text(line)
    out = _ok(
        _run(
            tmp_path,
            """
            import json
            from bamboo.querier.records import FileLogs

            logs = FileLogs(["panda-setupper.log.12345", "panda-adder.log"])
            one = logs.log("panda-adder.log", "12345")
            same = all(
                logs.log("panda-adder.log", key, window) is one
                for key in ("12345", "999")
                for window in (None, ("2026-01-01 00:00:00", "2026-01-01 00:00:00"), ("2027-01-01 00:00:00", "2027-01-01 00:00:00"))
            )
            try:
                FileLogs(["panda-adder.log", "panda-adder.log.1"])
                twice = ""
            except SystemExit as e:
                twice = str(e)
            print(json.dumps([logs.names(["panda-adder.log", "panda-setupper.log", "panda-closer.log"]), one.path, same, twice]))
            """,
        )
    )
    names, path, same, twice = json.loads(out)
    # the order given, the names the map uses, and only those wanted that were handed in
    assert names == ["panda-setupper.log", "panda-adder.log"]
    assert path == "panda-adder.log"
    # a file holds what it holds: neither the key nor the window selects from it
    assert same is True
    assert "are both panda-adder.log" in twice


def test_the_windows_come_from_the_rows_or_say_why_not(tmp_path):
    out = _ok(
        _run(
            tmp_path,
            """
            import json
            from bamboo.querier import windows as W

            def w(a):
                return [a.window, a.machine, a.why]

            state = {"status": "failed", "time": "2026-09-28 19:41:08", "host": "aipanda093.cern.ch"}
            ds = {"class": "DatasetSpec", "table": "ATLAS_PANDA.Datasets", "asked": ["a_sub1", "gone", "twice"],
                  "rows": [{"name": "a_sub1", "creationdate": "2026-09-28 08:31:12.345678"},
                           {"name": "twice", "creationdate": "2026-09-28 08:00:00"},
                           {"name": "twice", "creationdate": "2026-09-28 09:00:00"}]}
            log = {"as_of": "2026-10-13 12:00:00", "states": [
                {"status": "finishing", "time": "2026-10-13 01:00:00"},
                {"status": "running", "time": "2026-10-13 02:00:00"},
                {"status": "finishing", "time": "2026-10-13 10:00:00"}]}
            print(json.dumps([
                w(W.at_transition(state)),
                w(W.at_transition({"status": "failed", "time": "2026-09-28 19:41:08"})),
                w(W.around_creation(ds, "a_sub1", "2026-09-28 08:31:08")),
                w(W.around_creation(None, "a_sub1", "2026-09-28 08:31:08")),
                w(W.around_creation(ds, "never", "2026-09-28 08:31:08")),
                w(W.around_creation(ds, "gone", "2026-09-28 08:31:08")),
                w(W.around_creation(ds, "twice", "2026-09-28 08:31:08")),
                w(W.around_creation(ds, "a_sub1", "2026-09-28 09:00:00")),
                w(W.stuck(log, "finishing")),
                w(W.stuck({**log, "as_of": "2026-10-13 20:00:00"}, "finishing")),
                w(W.stuck({"states": log["states"]}, "finishing")),
                w(W.stuck(log, "ready")),
            ]))
            """,
        )
    )
    (h2, h2_any, h6, no_section, not_asked, gone, twice, too_early, stuck, stuck_long, no_as_of, no_row) = json.loads(out)
    # H2: the row's own time, on the machine that wrote it
    assert h2[:2] == [["2026-09-28 19:41:08", "2026-09-28 19:41:08"], "aipanda093.cern.ch"]
    assert h2_any[1] is None and "names no host" in h2_any[2]
    # H6: creationdate +- 10 min, cut outwards to seconds
    assert h6[0] == ["2026-09-28 08:21:12", "2026-09-28 08:41:13"]
    assert "inserting machine's clock" in h6[2]
    assert no_section[0] is None and "no datasets section" in no_section[2]
    assert not_asked[0] is None and "did not ask" in not_asked[2]
    assert gone[0] is None and "may have been purged" in gone[2]
    assert twice[0] is None and "2 ATLAS_PANDA.Datasets rows" in twice[2]
    assert too_early[0] is None and "before the job was created" in too_early[2]
    # --stuck: from the last row of the value less 10 min, but no earlier than as_of less 6 h
    assert stuck[0] == ["2026-10-13 09:50:00", "2026-10-13 12:00:00"]
    assert stuck_long[0] == ["2026-10-13 14:00:00", "2026-10-13 20:00:00"]
    assert no_as_of[0] is None and "as_of" in no_as_of[2]
    assert no_row[0] is None and "no row for 'ready'" in no_row[2]


def test_h2_bounds_a_value_the_history_holds_by_the_rows_holding_it(tmp_path):
    out = _ok(
        _run(
            tmp_path,
            """
            import json
            from bamboo.querier import parts as PT
            from bamboo.querier import windows as W

            def w(a):
                return [a.window, a.machine, a.why]

            def rows(*sites):
                times = ["2026-10-16 02:32:18", "2026-10-16 02:32:23", "2026-10-16 02:32:46"]
                return [{"status": s, "time": t, "computingSite": v}
                        for s, t, v in zip(["pending", "defined", "assigned"], times, sites)]

            made = "2026-10-16 02:32:04"
            print(json.dumps([
                w(W.value_held(rows("X", "AU", "AU"), "computingSite", "AU", made)),
                w(W.value_held(rows("AU", "X", "AU"), "computingSite", "AU", made)),
                w(W.value_held(rows("AU", "AU", "AU"), "computingSite", "AU", made)),
                w(W.value_held(rows("AU", "AU", "AU"), "computingSite", "AU", "2026-10-16 02:32:30.5")),
                w(W.value_held(rows("AU", "AU", "AU"), "computingSite", "AU", None)),
                w(W.value_held(rows("AU", "AU", "X"), "computingSite", "AU", made)),
                w(W.value_held([{"status": "pending", "time": "2026-10-16 02:32:18", "site": "AU"}], "computingSite", "AU", made)),
                [[h.subject, h.created] if (h := PT.history_holding(s)) else None
                 for s in ("JobSpec.computingSite", "JobSpec.jobStatus", "JobSpec.attemptNr", "JediTaskSpec.reason")],
            ]))
            """,
        )
    )
    later, again, first, early_clock, no_creation, after_last, no_column, holding = json.loads(out)
    # after the last row holding another value (a second on: the row's second), to the first of the last stretch
    assert later[:2] == [["2026-10-16 02:32:19", "2026-10-16 02:32:23"], None]
    assert "after the statuslog's 'pending'" in later[2] and "on every machine" in later[2]
    assert again[0] == ["2026-10-16 02:32:24", "2026-10-16 02:32:46"]
    # a stretch from the first row: from the row's creation, never after the first row
    assert first[0] == ["2026-10-16 02:32:04", "2026-10-16 02:32:18"] and "from the row's creation" in first[2]
    assert early_clock[0] == ["2026-10-16 02:32:18", "2026-10-16 02:32:18"]
    assert no_creation[0] is None and "creation time is not known" in no_creation[2]
    # no row closes the window; no row carries the column under its own name
    assert after_last[0] is None and "written after it" in after_last[2]
    assert no_column[0] is None and "carry no computingSite" in no_column[2]
    # the job's history holds computingSite beside the status, and knows when the row was made; a status is not held
    assert holding == [["JobSpec.jobStatus", "creationTime"], None, None, ["JediTaskSpec.status", ""]]


def test_h2_reads_the_rows_creation_from_its_row_or_after_whatever_the_time(tmp_path):
    out = _ok(
        _run(
            tmp_path,
            _M2_SETUP + """
    from bamboo.querier import parts as PT
    job, task = PT.history_holding("JobSpec.computingSite"), PT.history_holding("JediTaskSpec.reason")
    said = [r._created(job)]
    del r.data["rows"]["job"]["creationTime"]
    said.append(r._created(job))
    r.data["after"]["creationTime"] = "2026-01-01 00:00:05"
    said.append(r._created(job))
    r.data["after"]["creationTime"] = {"value": "2026-01-01 00:00:06", "as_of": "2026-01-03 00:00:00"}
    said.append(r._created(job))
    said.append(r._created(task))
    print(json.dumps(said))
    """,
        )
    )
    assert json.loads(out.splitlines()[-1]) == ["2026-01-01 00:00:00", None, "2026-01-01 00:00:05", "2026-01-01 00:00:06", None]


def test_s1_asks_the_rest_of_the_window_only_for_a_mover_with_no_run_in_its_last_part(tmp_path):
    out = _ok(
        _run(
            tmp_path,
            _M2_SETUP + """
    from bamboo.querier import windows as W

    r.data["statuslog"] = {"as_of": "2026-10-13 20:00:00", "states": [{"status": "finishing", "time": "2026-10-13 01:00:00"}]}
    first = W.stuck(r.data["statuslog"], "finishing")
    asked = []

    def run(at, token=""):
        return N(first=at, last=at, token=token)

    # what each log holds of the key, per window asked: the last part, then the rest
    held = {
        "ran-late": {first.window: [run("2026-10-13 19:00:00")]},
        "ran-early": {first.window: [], ("2026-10-13 00:50:00", "2026-10-13 14:00:00"): [run("2026-10-13 03:00:00")]},
        "another's": {first.window: [run("2026-10-13 19:00:00", "otherMethod < PandaID=1 >")]},
    }

    def ask(key, names, asking, owners=None):
        asked.append((names[0], asking.window))
        if names[0] == "holes":
            return {}, [], {"holes": ["2026-10-13 15:00:00 .. 2026-10-13 16:00:00"]}
        if names[0] == "unanswered":
            return {}, [("an ask", "why")], {}
        return {names[0]: held.get(names[0], {}).get(asking.window, [])}, [], {}

    r._ask = ask
    s = rerun.Stuck("JobSpec.jobStatus", "finishing", "1")
    got = {}
    r.logs = N(needs_window=True)
    for name in ["ran-late", "ran-early", "silent", "holes", "unanswered", "another's"]:
        found, missed, holes, searched = r._ask_mover(s, "a.py::mover", [name], first)
        got[name] = [searched.window, r._runs_of(found, "a.py::mover", searched)]
    # a writer that takes a row by its key: the last part alone, a run before it not looked for
    found, missed, holes, searched = r._ask_mover(s, "a.py::mover", ["ran-early"], first, rest=False)
    got["by key"] = [searched.window, r._runs_of(found, "a.py::mover", searched)]
    # a window that starts at the row has no rest
    r.data["statuslog"]["as_of"] = "2026-10-13 05:00:00"
    short = W.stuck(r.data["statuslog"], "finishing")
    got["short"] = [r._ask_mover(s, "a.py::mover", ["silent"], short)[3].window, W.rest(r.data["statuslog"], "finishing")]
    # the door over files reads each file whole: one ask
    r.logs = N(needs_window=False)
    got["files"] = [r._ask_mover(s, "a.py::mover", ["silent"], first)[3].window]
    print(json.dumps([got, asked, r.lines]))
    """,
        )
    )
    got, asked, lines = json.loads(out.splitlines()[-1])
    last, rest = ["2026-10-13 14:00:00", "2026-10-13 20:00:00"], ["2026-10-13 00:50:00", "2026-10-13 14:00:00"]
    # a run in the last six hours: the rest is not asked
    assert got["ran-late"] == [last, ["ran-late"]]
    # none there: the rest of the window, from the row less 10 min, and the run there is found
    assert got["ran-early"] == [rest, ["ran-early"]]
    assert got["silent"] == [rest, []]
    # another method's run is not the mover's: the rest is asked
    assert got["another's"] == [rest, []]
    # not searched whole, or not answered: no rest yet
    assert got["holes"][0] == got["unanswered"][0] == last
    assert got["short"] == [["2026-10-13 00:50:00", "2026-10-13 05:00:00"], None]
    assert got["files"] == [last]
    assert got["by key"] == [last, []]
    assert [w for n, w in asked if n == "ran-early"] == [last, rest, last]
    assert [w for n, w in asked if n == "silent"] == [last, rest, ["2026-10-13 00:50:00", "2026-10-13 05:00:00"], last]
    assert "  mover: no run of its own from 2026-10-13 14:00:00: asking the rest of the window" in lines


def test_s1_says_which_premises_of_absence_hold_for_a_mover_with_no_run_found(tmp_path):
    out = _ok(
        _run(
            tmp_path,
            _M2_SETUP + """
    from bamboo.querier import windows as W

    r.data["statuslog"] = {"as_of": "2026-10-13 20:00:00", "states": [{"status": "finishing", "time": "2026-10-13 01:00:00"}]}
    first, rest = W.stuck(r.data["statuslog"], "finishing"), W.rest(r.data["statuslog"], "finishing")
    s = rerun.Stuck("JobSpec.jobStatus", "finishing", "1")
    held = lambda p: [p["held"], p["why"]]
    got = {}
    r.logs = N(needs_window=True)
    for name, searched, missed, holes, to_hand in [
        ("whole", rest, [], {}, ["a.log"]),
        ("last part only", first, [], {}, ["a.log"]),
        ("holes", rest, [], {"a.log": ["m1: 2026-10-13 02:00:00 .. 2026-10-13 03:00:00"]}, ["a.log"]),
        ("unanswered", rest, [("an ask", "why")], {}, ["a.log"]),
        ("holes and unanswered", first, [("an ask", "why")], {"a.log": ["m1: 2026-10-13 15:00:00 .. 2026-10-13 16:00:00"]}, ["a.log", "b.log"]),
        ("no log", first, [], {}, []),
    ]:
        got[name] = held(r._searched_whole(s, to_hand, first, searched, missed, holes))
    # a window that starts at the row is whole in its last part
    r.data["statuslog"]["as_of"] = "2026-10-13 05:00:00"
    short = W.stuck(r.data["statuslog"], "finishing")
    got["short"] = held(r._searched_whole(s, ["a.log"], short, short, [], {}))
    r.logs = N(needs_window=False)
    got["files"] = held(r._searched_whole(s, ["a.log"], first, first, [], {}))

    # the map's prints, by owner
    p = lambda basis, *files: N(log_files_basis=basis, log_files=list(files))
    prints = {"logger": [p("logger", "a.log"), p("logger", "a.log", "b.log")], "module": [p("logger", "a.log"), p("module", "c.log")],
              "none": [], "nowhere": [p("logger")]}
    rerun.R._prints_of = lambda owner, lines: N(statements=prints[owner.split("::")[1]]) if owner.split("::")[1] in prints else None
    logs = {name: held(rerun.every_log_asked(f"x.py::{name}", [], asked)) for name, asked in [
        ("logger", ["a.log", "b.log"]), ("module", ["a.log", "c.log"]), ("none", ["a.log"]), ("nowhere", ["a.log"]),
        ("missing", ["a.log"])]}
    logs["one not asked"] = held(rerun.every_log_asked("x.py::logger", [], ["a.log"]))
    logs["prints nothing"] = [rerun.prints_nothing(f"x.py::{name}", []) for name in ("logger", "none", "missing")]
    rerun.R.ROWS_FROM = "walk"
    logs["walk"] = held(rerun.every_log_asked("x.py::logger", [], ["a.log", "b.log"]))
    rerun.R.ROWS_FROM = "graph"
    r.logs = N(needs_window=True)
    premises = r._premises(s, "x.py::logger", [], ["a.log", "b.log"], short, short, [], {})
    keys = [sorted(r._key_columns("JobSpec.jobStatus", "1")), sorted(r._key_columns("JobSpec.jobStatus", "2"))]
    print(json.dumps([got, logs, premises, keys]))
    """,
        )
    )
    got, logs, premises, keys = json.loads(out.splitlines()[-1])
    # the row's key is the column holding the key asked, as held.split reads it
    assert keys == [["JobSpec.PandaID"], []]
    # the whole window: the rest asked, or a last part that starts at the row, every ask answered, nothing unsearched
    assert got["whole"] == [True, "2026-10-13 00:50:00 .. 2026-10-13 20:00:00 searched in ['a.log'], none of it unsearched"]
    assert got["short"] == [True, "2026-10-13 00:50:00 .. 2026-10-13 05:00:00 searched in ['a.log'], none of it unsearched"]
    assert got["last part only"] == [False, "searched from 2026-10-13 14:00:00 only"]
    assert got["holes"] == got["holes and unanswered"] == [False, "a.log not searched all the window"]
    assert got["no log"] == [False, "no log was asked"]
    # not known yet: an ask not answered, or a file handed in, which does not say what of the window it holds
    assert got["unanswered"] == [None, "1 ask(s) not answered yet"]
    assert got["files"][0] is None
    # every file the map places the prints in, by their logger, asked
    assert logs["logger"] == [True, "every file the map places its prints in was asked: ['a.log', 'b.log']"]
    assert logs["one not asked"] == [False, "not asked: ['b.log']"]
    # no prints of its own, none at all or no statement: no run in a log is its own (K25)
    assert logs["prints nothing"] == [False, True, True]
    assert logs["module"][0] is False and "['module']" in logs["module"][1]
    assert logs["none"] == logs["missing"] == [False, "the map has no prints of its own for it"]
    assert logs["nowhere"] == [False, "the map places its prints in no file"]
    assert logs["walk"][0] is None
    # the third, prints_naming: a mover with no arm of its own prints, if at all, through what it calls
    assert list(premises) == ["window", "logs", "prints"]
    assert premises["window"]["held"] is True and premises["logs"]["held"] is True
    assert premises["prints"] == {"held": None, "why": "no arm of its own: what it calls prints, if anything"}


_RECORDED = """
    import json
    from pathlib import Path
    from bamboo.querier.recorded import Ask, RecordedLogs
    from bamboo.querier.records import Unrecorded, runs_for

    M1, M2 = "aipanda093.cern.ch", "aipanda094.cern.ch"
    # rotated at 06:40 each day; a file's covers run 10 min past its rotation (margin_s), so neighbours overlap
    TODAY = {"file": "panda-x.log-20260929.gz", "covers": ["2026-09-28 06:40:00", "2026-09-29 06:50:00"], "margin_s": 600,
             "covers_from": "mtime", "tail_bytes": None, "truncated": False, "status": "answered", "error": None}
    BEFORE = {**TODAY, "file": "panda-x.log-20260928.gz", "covers": ["2026-09-27 06:38:00", "2026-09-28 06:50:00"]}

    def line(t, msg, tok):
        return {"head": True, "cut": False, "lines": [f"2026-09-28 {t},000 panda.log.X: DEBUG    {tok} {msg}"]}

    def answered(*files):
        return {"service": "server", "status": "listed", "error": None, "files": list(files)}

    def record(ask, machines):
        return {"ask": ask.canonical(), "round": 1, "machines": machines}

    def ask_of(records, name, key, window, machine=None):
        path = Path(f"r{len(list(Path('.').glob('r*.jsonl')))}.jsonl")
        path.write_text("".join(json.dumps(r) + "\\n" for r in records))
        try:
            return RecordedLogs([str(path)]).log(name, key, window, machine=machine), []
        except Unrecorded as e:
            return None, [[a.canonical(), why] for a, why in e.asks]
"""


def test_the_record_door_asks_the_marks_that_do_not_hold_the_key_a_round_later(tmp_path):
    out = _ok(
        _run(
            tmp_path,
            textwrap.dedent(_RECORDED)
            + textwrap.dedent("""
            TOK, W = "2026-09-28/08:31:10.314981", ("2026-09-28 08:21:12", "2026-09-28 08:41:13")
            key_ask, tok_ask = Ask("panda-x.log", ("ds_sub1",), W), Ask("panda-x.log", (TOK,), W, M1)
            hit = {**line("08:31:12", "registered ds_sub1", TOK), "offset": 200}
            fragment = {"offset": 900, "head": False, "cut": False, "lines": ["  ds_sub1 in a traceback"]}
            key_rec = record(key_ask, {M1: answered({**TODAY, "records": [hit, fragment]}),
                                       M2: answered({**TODAY, "status": "failed", "error": "timeout", "records": []})})
            tok_rec = record(tok_ask, {M1: answered({**TODAY, "records": [
                {**line("08:31:10", "start", TOK), "offset": 100}, hit, {**line("08:31:14", "end", TOK), "offset": 300}]})})
            _, round0 = ask_of([], "panda-x.log", "ds_sub1", W)
            _, round1 = ask_of([key_rec], "panda-x.log", "ds_sub1", W)
            log, round2 = ask_of([key_rec, tok_rec], "panda-x.log", "ds_sub1", W)
            run, = runs_for(log, "ds_sub1")
            try:
                RecordedLogs([str(Path("r2.jsonl"))] * 2)
                twice = ""
            except SystemExit as e:
                twice = str(e)
            print(json.dumps([round0, round1, round2, [r.span for r in run.records], run.describe(), log.holes, log.notes,
                              run.would_print("ERROR"), twice]))
            """),
        )
    )
    round0, round1, round2, spans, described, holes, notes, would_print, twice = json.loads(out)
    # the key first, on every machine (no machine named)
    assert round0 == [[{"machine": None, "name": "panda-x.log", "service": "unsettled", "strings": ["ds_sub1"],
                        "whole_records": True, "window": ["2026-09-28 08:21:12", "2026-09-28 08:41:13"]}, ""]]
    # then the mark that does not hold it, on the machine it was found on
    assert round1[0][0]["strings"] == ["2026-09-28/08:31:10.314981"] and round1[0][0]["machine"] == "aipanda093.cern.ch"
    assert "mark does not hold the key" in round1[0][1]
    assert round2 == []
    # the run is whole: every record with the mark, each once, named by machine, file and offset
    assert spans == ["aipanda093:panda-x.log-20260929.gz@100", "aipanda093:panda-x.log-20260929.gz@200",
                     "aipanda093:panda-x.log-20260929.gz@300"]
    assert described.startswith("panda-x.log@aipanda093 token=2026-09-28/08:31:10.314981 3 record(s)")
    # the file that timed out was not searched; the fragment is counted, not placed
    assert holes == ["aipanda094.cern.ch: 2026-09-28 08:21:12 .. 2026-09-28 08:41:13"]
    assert notes == ["1 fragment(s) of records whose head was not returned"]
    assert would_print is True
    assert "a second record for the ask" in twice


def test_the_record_door_asks_for_s1_the_marks_its_movers_own_and_reads_a_shared_one_as_cut(tmp_path):
    out = _ok(
        _run(
            tmp_path,
            textwrap.dedent(_RECORDED)
            + textwrap.dedent("""
            K, W = "52977211", ("2026-09-28 08:21:12", "2026-09-28 08:41:13")
            SHARED, OWN, OTHER, WORKER = "getX < vo=None label=None >", "upd < 2026-09-28/08:31:10.1 >", "push < 2026-09-28/08:33:00.0 >", "STAMP1"
            NESTED = "chk <  < jediTaskID=52977211 > >"
            hits = [{**line("08:22:00", f"task {K}", SHARED), "offset": 10}, {**line("08:31:12", f"task {K}", OWN), "offset": 20},
                    {**line("08:33:01", f"job of {K}", OTHER), "offset": 30}, {**line("08:35:00", f"for {K}", WORKER), "offset": 40},
                    {**line("08:36:00", "in finishing with True", NESTED), "offset": 50}, {**line("08:40:00", f"task {K}", SHARED), "offset": 60}]
            key_rec = record(Ask("panda-x.log", (K,), W), {M1: answered({**TODAY, "records": hits})})
            Path("k.jsonl").write_text(json.dumps(key_rec) + "\\n")
            asked = {}
            for label, owners in (("s1", frozenset({"getX", "upd"})), ("any", None)):
                try:
                    RecordedLogs(["k.jsonl"]).log("panda-x.log", K, W, owners=owners)
                except Unrecorded as e:
                    asked[label] = [a.canonical()["strings"] for a, _ in e.asks]
            tok_ask = Ask("panda-x.log", tuple(sorted([OWN, WORKER])), W, M1)
            tok_rec = record(tok_ask, {M1: answered({**TODAY, "records": [hits[1], {**line("08:31:13", "done", OWN), "offset": 21},
                                                                           hits[3], {**line("08:35:01", "end", WORKER), "offset": 41}]})})
            Path("t.jsonl").write_text(json.dumps(tok_rec) + "\\n")
            log = RecordedLogs(["k.jsonl", "t.jsonl"]).log("panda-x.log", K, W, owners=frozenset({"getX", "upd"}))
            runs = {r.token: [len(r.records), r.would_print("DEBUG"), r.describe()] for r in runs_for(log, K)}
            print(json.dumps([asked, runs, log.notes]))
            """),
        )
    )
    asked, runs, notes = json.loads(out)
    # S1's movers own getX and upd: upd's mark, in one piece, and the worker's are asked; getX's, its key records 18 min
    # apart, is shared by calls, push's is no mover's; chk's tag holds brackets and the key, so it needs no round
    assert asked["s1"] == [["STAMP1", "upd < 2026-09-28/08:31:10.1 >"]]
    # without owners, every mark that does not hold the key, as before
    assert asked["any"] == [["STAMP1", "getX < vo=None label=None >", "push < 2026-09-28/08:33:00.0 >", "upd < 2026-09-28/08:31:10.1 >"]]
    assert runs["upd < 2026-09-28/08:31:10.1 >"][:2] == [2, True] and runs["STAMP1"][:2] == [2, True]
    assert runs["chk <  < jediTaskID=52977211 > >"][:2] == [1, True]
    # a mark not asked whole: its key records alone, read as cut -- no row missing from it is read as not printed
    assert runs["getX < vo=None label=None >"][:2] == [2, False] and "a mark the method's calls share" in runs["getX < vo=None label=None >"][2]
    assert runs["push < 2026-09-28/08:33:00.0 >"][:2] == [1, False] and "a method no step asked reads" in runs["push < 2026-09-28/08:33:00.0 >"][2]
    assert notes == ["2 mark(s) read from the key's records alone, as cut: a mark the method's calls share; a method no step asked reads"]


def test_the_record_door_widens_once_past_an_end_and_reads_a_run_still_there_as_cut(tmp_path):
    out = _ok(
        _run(
            tmp_path,
            textwrap.dedent(_RECORDED)
            + textwrap.dedent("""
            TOK, W = "PandaID=12345", ("2026-09-28 06:45:00", "2026-09-28 06:45:00")
            first = record(Ask("panda-x.log", ("12345",), W, M1), {M1: answered({**TODAY, "records": [
                {**line("06:42:00", "start 12345", TOK), "offset": 10}, {**line("06:46:00", "end", TOK), "offset": 20}]})})
            _, edge = ask_of([first], "panda-x.log", "12345", W, M1)
            wider = Ask(**{**{k: v for k, v in edge[0][0].items() if k in ("name", "machine")},
                           "strings": tuple(edge[0][0]["strings"]), "window": tuple(edge[0][0]["window"])})
            older = {**BEFORE, "records": [{"offset": 5, "head": True, "cut": False,
                                            "lines": ["2026-09-28 06:39:30,000 panda.log.X: DEBUG    PandaID=12345 got 12345"]}]}
            whole, _ = ask_of([first, record(wider, {M1: answered(older, {**TODAY, "records": []})})], "panda-x.log", "12345", W, M1)
            kept, _ = ask_of([first, record(wider, {M1: answered({**TODAY, "records": []})})], "panda-x.log", "12345", W, M1)
            moved = record(Ask("panda-x.log", ("12345",), W, M2), {M2: {"service": None, "status": "not listed", "files": []}})
            _, unlisted = ask_of([moved], "panda-x.log", "12345", W, M2)
            print(json.dumps([edge, [r.stamp for r in runs_for(whole, "12345")[0].records],
                              runs_for(whole, "12345")[0].would_print("ERROR"), runs_for(kept, "12345")[0].would_print("ERROR"),
                              unlisted]))
            """),
        )
    )
    edge, stamps, whole_prints, kept_prints, unlisted = json.loads(out)
    # 06:42 is within 10 min of the file's start: the window is widened past it, on the same machine
    assert edge[0][0]["window"] == ["2026-09-28 06:39:59", "2026-09-28 06:45:00"]
    assert edge[0][0]["machine"] == "aipanda093.cern.ch" and "within 10 min" in edge[0][1]
    # the older file holds the run's start; the joined stretch leaves it far from any end
    assert stamps == ["2026-09-28 06:39:30,000", "2026-09-28 06:42:00,000", "2026-09-28 06:46:00,000"]
    assert whole_prints is True
    # no older file to widen into: the run may go on before what was searched, so rule 8 reads nothing missing
    assert kept_prints is False
    # a machine list_logs does not hold: asked again on every machine
    assert unlisted[0][0]["machine"] is None and "not in list_logs" in unlisted[0][1]


def test_a_run_the_record_door_may_have_cut_says_which_end_of_the_search_it_is_near_and_why(tmp_path):
    out = _ok(
        _run(
            tmp_path,
            textwrap.dedent(_RECORDED)
            + textwrap.dedent("""
            TOK, LATE, EARLY = "PandaID=12345", ("2026-09-28 08:21:12", "2026-09-28 08:41:13"), ("2026-09-28 06:30:00", "2026-09-28 06:45:00")
            run = [{**line("08:31:10", "start 12345", TOK), "offset": 10}, {**line("08:31:14", "end", TOK), "offset": 20}]
            early = [{**line("06:42:00", "start 12345", TOK), "offset": 10}, {**line("06:46:00", "end", TOK), "offset": 20}]
            oldest = {**BEFORE, "covers": [None, "2026-09-28 06:50:00"], "records": []}

            def described(window, *files):
                log, asks = ask_of([record(Ask("panda-x.log", ("12345",), window, M1), {M1: answered(*files)})],
                                   "panda-x.log", "12345", window, M1)
                return asks or runs_for(log, "12345")[0].describe()

            # widened once past the file's start, and no older file there
            AT, WIDER = ("2026-09-28 06:45:00", "2026-09-28 06:45:00"), ("2026-09-28 06:39:59", "2026-09-28 06:45:00")
            log, _ = ask_of([record(Ask("panda-x.log", ("12345",), AT, M1), {M1: answered({**TODAY, "records": early})}),
                             record(Ask("panda-x.log", ("12345",), WIDER, M1), {M1: answered({**TODAY, "records": []})})],
                            "panda-x.log", "12345", AT, M1)
            print(json.dumps([
                described(LATE, {**TODAY, "truncated": True, "records": run}),
                described(LATE, {**TODAY, "margin_s": 0, "covers": ["2026-09-28 06:40:00", "2026-09-28 08:35:00"], "records": run}),
                described(EARLY, oldest, {**TODAY, "records": early}),
                runs_for(log, "12345")[0].describe(),
                described(LATE, {**TODAY, "records": run}),
            ]))
            """),
        )
    )
    cut, asked_then, oldest_first, nothing_before, whole = json.loads(out)
    assert cut.endswith("(key on 2; read as cut: what was searched on aipanda093 ends at 2026-09-28 08:31:14: the answer of "
                        "panda-x.log-20260929.gz was cut at 1 MB after its last record)")
    assert asked_then.endswith("ends at 2026-09-28 08:34:59: panda-x.log-20260929.gz was read to the time it was asked)")
    assert oldest_first.endswith("begins at 2026-09-28 06:40:00: panda-x.log-20260928.gz is the oldest kept, its start not "
                                 "known)")
    assert nothing_before.endswith("begins at 2026-09-28 06:40:00: no file read covers what comes before)")
    # a run far from every end says nothing more
    assert whole.endswith("(key on 2)")


def test_the_record_door_says_what_of_the_window_it_did_not_search(tmp_path):
    out = _ok(
        _run(
            tmp_path,
            textwrap.dedent(_RECORDED)
            + textwrap.dedent("""
            def holes(window, machines):
                ask = Ask("panda-x.log", ("12345",), window)
                log, _ = ask_of([record(ask, machines)], "panda-x.log", "12345", window)
                return log.holes, log.notes

            OVERLAP = ("2026-09-28 06:45:00", "2026-09-28 06:45:00")  # after the rotation, inside the older file's margin
            print(json.dumps([
                # the older file failed and only the newer was read: not searched
                holes(OVERLAP, {M1: answered({**BEFORE, "status": "failed", "error": "timeout", "records": []}, {**TODAY, "records": []})}),
                # both read: searched
                holes(OVERLAP, {M1: answered({**BEFORE, "records": []}, {**TODAY, "records": []})}),
                # the margin alone does not make the older file's second searched before the newer was asked
                holes(("2026-09-28 06:40:00", "2026-09-28 06:40:00"), {M1: answered({**BEFORE, "records": []})}),
                # truncated: read up to its last record
                holes(("2026-09-28 07:00:00", "2026-09-28 07:30:00"), {M1: answered({**TODAY, "truncated": True, "records": [
                    {**line("07:10:00", "12345", "PandaID=12345"), "offset": 1}]})}),
                # today's file, asked at 12:00: no margin, so read to a second before
                holes(("2026-09-28 11:00:00", "2026-09-28 12:00:00"), {M1: answered({**TODAY, "covers": ["2026-09-28 06:40:00", "2026-09-28 12:00:00"],
                                                                                    "margin_s": 0, "records": []})}),
                # the oldest file kept: its start is not known
                holes(OVERLAP, {M1: answered({**BEFORE, "covers": [None, "2026-09-28 06:50:00"], "records": []})}),
                # listed with no file reaching the window; a list not read; a machine without the log; no machine at all
                holes(OVERLAP, {M1: answered(), M2: {"service": "jedi", "status": "list failed", "error": "timeout", "files": []},
                                "aipanda095.cern.ch": {"service": "server", "status": "no such log", "files": []}}),
                holes(OVERLAP, {}),
            ]))
            """),
        )
    )
    failed, both, rotation, truncated, today, oldest, machines, nowhere = json.loads(out)
    assert failed == [["aipanda093.cern.ch: 2026-09-28 06:45:00 .. 2026-09-28 06:45:00"], []]
    assert both == [[], []]
    # 06:40:00 is in the older file's covers and margin, read through: searched (the newer may hold it too, and was
    # not asked here: only files the recording step chose are known)
    assert rotation == [[], []]
    # (and the run at its last record is not asked for again: the same window would read no further)
    assert truncated[0] == ["aipanda093.cern.ch: 2026-09-28 07:10:01 .. 2026-09-28 07:30:00"]
    assert today[0] == ["aipanda093.cern.ch: 2026-09-28 12:00:00 .. 2026-09-28 12:00:00"]
    assert oldest[0] == ["aipanda093.cern.ch: 2026-09-28 06:45:00 .. 2026-09-28 06:45:00"]
    assert machines[0] == ["aipanda094.cern.ch: list failed (timeout)", "aipanda093.cern.ch: 2026-09-28 06:45:00 .. 2026-09-28 06:45:00"]
    assert machines[1][-1] == "['aipanda095.cern.ch'] hold(s) no file of it"
    assert nowhere == [[], ["no machine lists it: the log is on none"]]


def test_a_fork_runs_each_branch_from_the_same_state_and_a_stop_of_any_is_the_runs(tmp_path):
    out = _ok(
        _run(
            tmp_path,
            """
            import json
            from dataclasses import dataclass
            from bamboo.querier import rerun

            @dataclass
            class Go:
                name: str
                stop: bool

            r = object.__new__(rerun.Rerun)
            r.lines, r.hop, r.hand, r.hand_used = [], "", {}, []
            r.opened, r.ends, r.branches = [], [], []
            r.touched, r.row_facts, r.question = [], {"kept": ("1", "before the fork")}, None
            seen = []

            def go(q):
                r.start("S2", q.name)
                seen.append((q.name, dict(r.row_facts)))
                r.row_facts[q.name] = ("x", "learned in " + q.name)
                if q.stop:
                    r.need("something only the second lacks", None)
                return None

            r.steps = {Go: go}
            r.hop = "S1"
            try:
                r.fork(rerun.Fork([Go("first", False), Go("second", True)], ["mover a", "mover b"]))
                stopped = ""
            except rerun.Stop as e:
                stopped = str(e)
            print(json.dumps([seen, stopped, [x for x in r.lines if x.startswith("====")], r.branches, r.hop]))
            """,
        )
    )
    # say() prints each line as it goes: the answer is the last
    seen, stopped, marks, branches, hop = json.loads(out.splitlines()[-1])
    # result.json's outcome.branches: each branch's end; the fork's stop is the fork's step
    assert branches == [
        {"label": "mover a", "kind": "end", "step": "S2", "text": None},
        {"label": "mover b", "kind": "stop", "step": "S2", "text": "STOP S2: something only the second lacks", "code": None},
    ]
    assert hop == "S1"
    # the second branch starts from the state the fork was reached in, not from what the first learned
    assert [names for names, _ in seen] == ["first", "second"]
    assert all(sorted(facts) == ["kept"] for _, facts in seen)
    assert marks == ["==== 1 of 2: mover a", "==== 1: reached the end", "==== 2 of 2: mover b",
                     "==== 2: STOP S2: something only the second lacks"]
    assert stopped == "1 of 2 branch(es) stopped: 2 (S2: something only the second lacks)"


def test_a_db_proxy_run_is_a_movers_only_when_its_mark_names_the_movers_method(tmp_path):
    out = _ok(
        _run(
            tmp_path,
            """
            import json
            from types import SimpleNamespace as N
            from bamboo.querier.rerun import _of_mover

            owner = "pandaserver/taskbuffer/db_proxy_mods/job_complex_module.py::getJobs"
            print(json.dumps([
                _of_mover(N(token="getJobs < BOINC_MCORE 2026-10-04T18:38:06.929035 >"), owner),
                _of_mover(N(token="updateJobStatus < PandaID=7338945928 >"), owner),
                _of_mover(N(token="< jediTaskID=52757505 >"), "pandajedi/jediorder/TaskCommando.py::runImpl"),
                _of_mover(N(token="7339032371"), "pandaserver/dataservice/adder_gen.py::run"),
            ]))
            """,
        )
    )
    assert json.loads(out.splitlines()[-1]) == [True, False, True, True]


def test_a_run_of_another_log_is_inside_a_call_only_between_the_prints_around_that_pass(tmp_path):
    corpus = _corpus(tmp_path)
    (corpus / "pandaserver" / "a.py").write_text(
        textwrap.dedent(
            """
            def f(self, jobs):
                self.logger.debug("start")
                for job in jobs:
                    self.logger.debug("updating DB")
                    done = self.taskBuffer.updateJobs([job])
                    self.logger.debug("retU")
                self.logger.debug("end")
            """
        )
    )
    out = _ok(
        _run(
            tmp_path,
            """
            import json
            from types import SimpleNamespace as N
            from bamboo.querier import rows as R
            from bamboo.querier.which_ran import inside_call

            node = R.functions("pandaserver/a.py", "f")[0]
            fn = N(rel="pandaserver/a.py", node=node, row_at=lambda s: N(line=s.lineno) if R.is_print(s) else None)
            stamps = ["00", "01", "02", "03", "04", "05"]  # start, updating DB, retU, updating DB, retU, end
            run = N(records=[N(stamp=f"2026-01-01 00:00:{s},000") for s in stamps])
            caller = N(fn=fn, run=run, unambiguous={3: [0], 5: [1, 3], 7: [2, 4], 8: [5]})
            inner = lambda a, b: N(first=f"2026-01-01 00:00:{a}", last=f"2026-01-01 00:00:{b}")
            print(json.dumps([
                inside_call(caller, inner("01,200", "01,800"), "pandaserver/a.py:6"),
                inside_call(caller, inner("03,200", "03,900"), "pandaserver/a.py:6"),
                inside_call(caller, inner("00,500", "01,500"), "pandaserver/a.py:6"),
                inside_call(caller, inner("01,500", "02,500"), "pandaserver/a.py:6"),
                inside_call(caller, inner("01,200", "01,800"), "pandaserver/b.py:6"),
            ]))
            """,
        )
    )
    first, second, early, late, elsewhere = json.loads(out.splitlines()[-1])
    # each pass through the call is the last print before it and the first after it
    assert first == [1, 2]
    assert second == [3, 4]
    # a run that starts before the pass or ends after it was not inside the call
    assert early is None
    assert late is None
    # the call is the caller's own
    assert elsewhere is None


def test_a_run_is_inside_only_the_calls_whose_pass_holds_it_and_calls_no_print_parts_are_named_together(tmp_path):
    corpus = _corpus(tmp_path)
    (corpus / "pandaserver" / "a.py").write_text(
        textwrap.dedent(
            """
            def f(self, job):
                self.logger.debug("start")
                self.check(job)
                self.logger.debug("status after")
                done = self.taskBuffer.updateJobs([job])
                self.logger.debug("retU")
                if job.x:
                    self.reset(job)
                else:
                    self.close(job)
                self.logger.debug("end")
            """
        )
    )
    out = _ok(
        _run(
            tmp_path,
            """
            import json
            from types import SimpleNamespace as N
            from bamboo.querier import rows as R
            from bamboo.querier.which_ran import held_in_calls, inside_call

            node = R.functions("pandaserver/a.py", "f")[0]
            fn = N(rel="pandaserver/a.py", node=node, row_at=lambda s: N(line=s.lineno) if R.is_print(s) else None)
            run = N(records=[N(stamp=f"2026-01-01 00:00:{s},000") for s in ["00", "01", "02", "03"]])  # start, status after, retU, end
            caller = N(fn=fn, run=run, unambiguous={3: [0], 5: [1], 7: [2], 12: [3]})
            inner = lambda a, b: N(first=f"2026-01-01 00:00:{a}", last=f"2026-01-01 00:00:{b}")
            at = lambda *lines: (lambda fn, c: [f"pandaserver/a.py:{n}" for n in lines])
            in_update, in_reset = inner("01,200", "01,800"), inner("02,200", "02,800")
            print(json.dumps([
                inside_call(caller, in_update, "pandaserver/a.py:4"),
                held_in_calls(in_update, [(None, caller)], at(4, 6)),
                held_in_calls(in_reset, [(None, caller)], at(4, 9, 11)),
                held_in_calls(in_update, [(None, caller)], at(4)),
            ]))
            """,
        )
    )
    in_check, update, reset_or_close, check_only = json.loads(out.splitlines()[-1])
    # a pass through check ends at the first print after it: the run of updateJobs is not inside check
    assert in_check is None
    assert update == [1, 2, "pandaserver/a.py:6"]
    # reset and close sit between the same two prints: the records hold the run in either, and both are named
    assert reset_or_close == [2, 3, "pandaserver/a.py:9 or pandaserver/a.py:11"]
    assert check_only is None


def test_every_call_that_goes_on_into_the_candidate_is_tried_at_each_of_its_lines(tmp_path):
    out = _ok(
        _run(
            tmp_path,
            """
            import json
            from types import SimpleNamespace as N
            from bamboo.codemap.models import Anchor, ReachCall, ReachNode
            from bamboo.querier import mapdump, rerun

            def node(owner, start, end, **k):
                return ReachNode(name=owner, map_id="m", derived_from="d", owner=owner, function=owner.rsplit(".", 1)[-1].split("::")[-1],
                                 anchor=Anchor(package="p", file=owner.split("::")[0], line_start=start, line_end=end), **k)

            archive = ReachCall(line=5, callee="archiveJob()", to="p/d.py::archiveJob")
            rows = [node("p/a.py::f", 1, 20, calls=[
                        ReachCall(line=12, lines=[4, 12], callee="self.taskBuffer.updateJobs()", to="p/t.py::updateJobs"),
                        ReachCall(line=6, callee="self.check()", to="p/a.py::check"),
                        ReachCall(line=8, callee="other.run()", to="p/w.py::Base.run", runs_as=["p/w.py::Other"]),
                        ReachCall(line=14, callee="gen.run()", to="p/w.py::Base.run", runs_as=["p/w.py::Gen"]),
                        ReachCall(line=10, callee="log()", to="p/l.py::log")]),
                    node("p/t.py::updateJobs", 1, 9, calls=[archive]),
                    node("p/a.py::check", 30, 35, calls=[archive]),
                    node("p/w.py::Base.run", 1, 5, classes=["p/w.py::Gen", "p/w.py::Other"], calls=[
                        ReachCall(line=3, callee="self.runImpl()", to="p/w.py::Gen.runImpl", for_classes=["p/w.py::Gen"], runs_as=["p/w.py::Gen"])]),
                    node("p/w.py::Gen.runImpl", 10, 15, classes=["p/w.py::Gen"], calls=[archive]),
                    node("p/l.py::log", 1, 3),
                    node("p/d.py::archiveJob", 1, 9)]
            mapdump._FRAGMENT[:] = [N(reach_graph=rows, junctions=[])]
            r = object.__new__(rerun.Rerun)
            r._graph_read, r._calls_into = None, {}
            caller = N(rel="p/a.py", node=N(body=[N(lineno=2)]))
            print(json.dumps([
                r._called_at(caller, N(rel="p/d.py", lines=[3])),
                r._called_at(caller, N(rel="p/a.py", lines=[3])),
                r._called_at(caller, N(rel="p/l.py", lines=[2])),
                sorted(map(list, r._calls_into)),
            ]))
            """,
        )
    )
    archive, itself, log, asked = json.loads(out.splitlines()[-1])
    # both lines of updateJobs, check's, and the worker entered as the class whose runImpl writes -- not the other
    assert archive == ["p/a.py:4", "p/a.py:6", "p/a.py:12", "p/a.py:14"]
    assert itself == []
    assert log == ["p/a.py:10"]
    # each (caller, callee) asked once
    assert asked == [["p/a.py::f", "p/d.py::archiveJob"], ["p/a.py::f", "p/l.py::log"]]


def test_a_candidate_with_no_run_says_why_for_every_log_it_was_looked_for_in(tmp_path):
    out = _ok(
        _run(
            tmp_path,
            """
            import json
            from types import SimpleNamespace as N
            from bamboo.querier.which_ran import Candidate, judge

            later = N(first="2026-01-01 01:00:00,000", last="2026-01-01 01:00:01,000",
                      log=N(covers=("2026-01-01 01:00:00,000", "2026-01-01 01:00:01,000"), records=[N(machine="")], holes=[]))
            logs = {"a.log": [], "b.log": [later]}
            both = Candidate("pandaserver/a.py", "f", [3], ["a.log", "b.log", "c.log"])
            none = Candidate("pandaserver/a.py", "g", [5], ["c.log"])
            j = judge([both, none], logs, window=("2026-01-01 00:00:00", "2026-01-01 00:00:00"))
            print(json.dumps([v.reasons for v in j.verdicts]))
            """,
        )
    )
    both, none = json.loads(out.splitlines()[-1])
    # each log handed says its own reason, in the candidate's order; the one not handed is not asked
    assert both == [
        "a.log: no run of the key at ('2026-01-01 00:00:00', '2026-01-01 00:00:00'); "
        "b.log covers 2026-01-01 01:00:00,000..2026-01-01 01:00:01,000, not 2026-01-01 00:00:00..2026-01-01 00:00:00"
    ]
    assert none == ["no log to hand (c.log)"]


def test_rule_6_asks_whether_the_guard_needs_the_value_whatever_value_a_hole_gave_it(tmp_path):
    corpus = _corpus(tmp_path)
    (corpus / "pandaserver" / "a.py").write_text(
        textwrap.dedent(
            """
            class A:
                def finalize(self):
                    if self.job.jobStatus == "failed" or self.job_status == "failed":
                        self.handle()

                def handle(self):
                    self.job.jobStatus = "failed"
            """
        )
    )
    out = _ok(
        _run(
            tmp_path,
            """
            import json
            from types import SimpleNamespace as N
            from bamboo.querier import evaluate as E, rows as R
            from bamboo.querier.which_ran import rewriter

            node = R.functions("pandaserver/a.py", "handle")[0]
            fn = N(rel="pandaserver/a.py", name="handle", node=node)
            stmt = node.body[0]
            def env(**held):
                return E.Env({k: E.Value(v, "hole") for k, v in held.items()}, {"self.job": "job"})
            print(json.dumps([
                rewriter(fn, stmt, "jobStatus", "failed", env(**{"self.job_status": "finished"})),
                rewriter(fn, stmt, "jobStatus", "failed", env(**{"self.job_status": "finished", "<job>.jobStatus": "failed"})),
                rewriter(fn, stmt, "jobStatus", "failed", env()),
            ]))
            """,
        )
    )
    bare, held, open_ = json.loads(out.splitlines()[-1])
    # the other term is false, so the guard needs the job already failed
    assert bare.startswith("rule 6: reached only when self.job.jobStatus is already 'failed'")
    # a hole that printed the job failed does not answer for the guard: it is still asked, and not shown as used
    assert held == bare
    # with the other term unknown, the guard does not need it
    assert open_ == ""


def test_the_tables_at_one_moment_check_where_reading_2_puts_a_job(tmp_path):
    out = _ok(
        _run(
            tmp_path,
            """
            import json
            from bamboo.querier.rerun import at_one_moment

            tables = ["jobsDefined4", "jobsActive4"]
            holds = [{"table": "jobsActive4", "jobStatus": "running", "n": 2},
                     {"table": "jobsArchived4", "jobStatus": "finished", "n": 312},
                     {"table": "jobsArchived", "jobStatus": "failed", "n": 31}]
            fails = [{"table": "JOBSACTIVE4", "jobStatus": "running", "n": 1},
                     {"table": "jobsActive4", "jobStatus": "finished", "n": 1},
                     {"table": "jobsArchived4", "jobStatus": "running", "n": 3}]
            print(json.dumps([at_one_moment(holds, tables), at_one_moment(fails, tables), at_one_moment([], tables)]))
            """,
        )
    )
    holds, fails, empty = json.loads(out.splitlines()[-1])
    assert holds == [{"jobsActive4/running": 2}, {}, {}]
    # a final job inside the call's tables, and one not yet final outside them, each break the assumption
    assert fails == [{"JOBSACTIVE4/running": 1, "jobsActive4/finished": 1}, {"jobsActive4/finished": 1},
                     {"jobsArchived4/running": 3}]
    assert empty == [{}, {}, {}]


def test_a_mover_reaches_a_writer_by_the_calls_the_map_placed_into_it(tmp_path):
    out = _ok(
        _run(
            tmp_path,
            """
            import json
            from types import SimpleNamespace as N
            from bamboo.querier.rerun import through_calls

            def writer(owner, outcomes, called_from, pre=()):
                return N(owner=owner, called_from=called_from,
                         branches=[N(line=10, outcome=o, path_condition=[], row_precondition=list(pre)) for o in outcomes])

            writers = [
                writer("p/proxy.py::killJob", ["closed"], ["d/cmd.py:7 -> p/tb.py:55"]),
                writer("p/proxy.py::archiveJob", ["failed"], ["p/tb.py:48", "… 7 more"]),
                writer("p/proxy.py::touch", ["running"], ["p/tb.py:50"]),
                writer("p/proxy.py::other", ["failed"], ["q/elsewhere.py:52"]),
                writer("p/proxy.py::getJobs", ["sent"], ["p/tb.py:45"], ["jobStatus='activated'"]),
            ]
            print(json.dumps([
                through_calls("p/tb.py", [(40, 60)], writers, "JobSpec.jobStatus", "running"),
                through_calls("d/cmd.py", [(1, 9)], writers, "JobSpec.jobStatus", "running"),
                through_calls("p/tb.py", [(100, 120)], writers, "JobSpec.jobStatus", "running"),
            ]))
            """,
        )
    )
    in_tb, one_out, elsewhere = json.loads(out.splitlines()[-1])
    # placed in the function: killJob's inner hop and archiveJob's own line; touch writes only the value observed, and
    # getJobs writes only rows holding another status
    assert in_tb == [["archiveJob", "killJob"], []]
    # one hop out: cmd.py calls tb.py, which calls killJob; archiveJob's list is cut, so it may be among the unlisted
    assert one_out == [["killJob"], ["archiveJob"]]
    assert elsewhere == [[], ["archiveJob"]]


def test_a_mover_is_the_def_its_statement_is_in(tmp_path):
    corpus = _corpus(tmp_path)
    (corpus / "pandaserver" / "metrics.py").write_text(textwrap.dedent("""\
        def count(cur):
            return cur.execute("SELECT 1")


        class FetchData:
            def count(self, cur):
                cur.execute("SELECT jobStatus FROM jobsActive4")
                return cur.fetchall()
        """))
    out = _ok(
        _run(
            tmp_path,
            """
            import json
            from bamboo.querier.rerun import defs_holding

            print(json.dumps([[(f.lineno, f.end_lineno) for f in defs_holding("pandaserver/metrics.py", "count", lines)]
                              for lines in ([7], [], [3])]))
            """,
        )
    )
    # the statement at 7 is FetchData.count's; with no statement every def of the name; a line between defs, none
    assert json.loads(out.splitlines()[-1]) == [[[6, 8]], [[1, 2], [6, 8]], []]


def test_s1_keeps_a_mover_with_no_arm_of_its_own_and_says_why(tmp_path):
    out = _ok(
        _run(
            tmp_path,
            """
            import json
            from bamboo.querier.rerun import kept_without_arms

            print(json.dumps([
                kept_without_arms(True, [], [], False, False),
                kept_without_arms(False, ["killJob"], ["archiveJob"], True, True),
            ]))
            """,
        )
    )
    nothing, every = json.loads(out.splitlines()[-1])
    # nothing the map lists reaches a writer: kept all the same, the map's called_from holding two hops
    assert nothing == ["no call it makes reaches a writer of another value within the two hops the map lists, "
                       "and a write further down is not seen (KNOWN_ISSUES.md K14)"]
    assert every == ["its def is not found", "it calls into ['killJob']",
                     "it may call into ['archiveJob'], whose callers the map lists only in part",
                     "it hands its rows on", "the map could not read where all its rows go"]


def test_s1_keeps_a_mover_for_a_gap_only_where_it_may_hide_where_the_rows_go_and_says_every_gap(tmp_path):
    out = _ok(
        _run(
            tmp_path,
            """
            import json
            from types import SimpleNamespace
            from bamboo.codemap.models import ReaderGap
            from bamboo.querier.rerun import gap_said, hides_where, map_gaps

            part = ReaderGap(kind="partial", file="p/m.py", line=472)
            unnoded = ReaderGap(kind="unnoded", file="p/m.py", line=222, stops=[("unreached", "dry_sql_query")])
            follow = SimpleNamespace(reader_gaps={"p/m.py::f": [part], "p/m.py::g": [part, unnoded]},
                                     guarded_by=[SimpleNamespace(owner="p/m.py::h", gaps=[unnoded]),
                                                 SimpleNamespace(owner="p/m.py::k", gaps=[])])
            gaps = map_gaps(follow)
            print(json.dumps([sorted(gaps), [hides_where(gaps[o]) for o in sorted(gaps)], hides_where([]),
                              gap_said(gaps["p/m.py::g"])]))
            """,
        )
    )
    owners, hidden, none, said = json.loads(out.splitlines()[-1])
    # a selecting function's gaps and a guarding one's, none for a function the map read all of
    assert owners == ["p/m.py::f", "p/m.py::g", "p/m.py::h"]
    # a statement read in part hides which rows it takes, not where they go: it alone keeps no mover
    assert hidden == [False, True, True] and none is False
    # every gap is said, a text per kind in the map's order, with its lines
    assert said == ("@222 dry_sql_query -- a statement here that reads these rows stopped being read, and has no node; "
                    "@472 -- a statement here that reads these rows was read only in part")


def test_s1_drops_a_mover_with_no_arm_only_where_the_reach_finds_no_write_and_nothing_open(tmp_path):
    out = _ok(
        _run(
            tmp_path,
            """
            import json
            from bamboo.querier.rerun import reach_said

            def reach(writes, holes=0, own=()):
                return {"starts": ["p/m.py::f"], "writes": writes, "holes": holes, "first_holes": ["p/m.py::f:3 x.y -- why"][:holes],
                        "assumptions": {"runner_runs_only_sql": 2}, "closure": 4, "own_arms": list(own)}

            w = [{"function": "p/w.py::W.killJob", "arms": ["failed@3"], "chain": ["p/m.py:7"]}]
            print(json.dumps([reach_said(None, False), reach_said(reach([]), False), reach_said(reach(None, 1), False),
                              reach_said(reach(w), False), reach_said(reach([]), True), reach_said(reach([], own=["p/m.py::f"]), False)]))
            """,
        )
    )
    unasked, nothing, holed, writes, gapped, own = json.loads(out.splitlines()[-1])
    assert unasked == [False, None]
    # only the map's [] drops: no write reached and no call left open
    assert nothing == [True, "no arm of its own, and reaches no write of another value with nothing left open (closure 4; "
                             "assuming ['runner_runs_only_sql']): dropped"]
    # None is not []: a call left open may go on into a write
    assert holed == [False, "reach: reaches no write of another value, but 1 call(s) on the way are left open, e.g. "
                            "['p/m.py::f:3 x.y -- why']"]
    assert writes == [False, "reach: reaches 1 writer(s) of another value: ['killJob']"]
    # a gap in the map's reading may hide a write the reach has no junction for (R1)
    assert gapped[0] is False and "could not read where all its rows go" in gapped[1]
    # an arm the junctions place in its own def, though S1 counts none: the two disagree, so it is kept
    assert own[0] is False and "though S1 counts none" in own[1]


def test_the_reach_starts_at_the_def_holding_the_statement_and_is_said_once(tmp_path):
    out = _ok(
        _run(
            tmp_path,
            """
            import json
            from types import SimpleNamespace as N
            from bamboo.codemap import reach as RE
            from bamboo.codemap.models import Anchor, ReachCall, ReachHole, ReachNode
            from bamboo.querier import mapdump, rerun

            def node(owner, start, end, **k):
                return ReachNode(name=owner, map_id="m", derived_from="d", owner=owner, function=owner.rsplit(".", 1)[-1].split("::")[-1],
                                 anchor=Anchor(package="p", file=owner.split("::")[0], line_start=start, line_end=end), **k)

            rows = [node("p/m.py::count", 1, 2),
                    node("p/m.py::FetchData.count", 6, 8, calls=[ReachCall(line=7, callee="write()", to="p/w.py::write")]),
                    node("p/w.py::write", 1, 5),
                    node("p/h.py::h", 1, 4, holes=[ReachHole(line=3, callee="d[k].append", why="an element of a dict")])]
            asked = []
            RE.moving_off = lambda junctions, graph, subject, value: asked.append((subject, value)) or RE.Destination({"p/w.py::write": ["failed@3"]}, [])
            q = rerun.Stuck("JobSpec.jobStatus", "running", "1")

            def fresh(reach_graph):
                mapdump._FRAGMENT[:] = [N(reach_graph=reach_graph, junctions=[])]
                r = object.__new__(rerun.Rerun)
                r.lines, r.result, r._reach, r._graph_read = [], {}, None, None
                return r

            r = fresh(rows)
            got = {name: r._reach_of(q, owner, lines) for name, owner, lines in [
                ("own statement", "p/m.py::count", [7]), ("rows handed", "p/m.py::count", []), ("nothing", "p/m.py::count", [1]),
                ("hole", "p/h.py::h", [2]), ("own arm", "p/w.py::write", [2]), ("none holds", "p/x.py::x", [3])]}
            bare = fresh([])
            print(json.dumps([got, r.lines, r.result["reach"]["functions"], asked,
                              [bare._reach_of(q, "p/m.py::count", [7]), bare._reach_of(q, "p/m.py::count", [1])], bare.lines, bare.result]))
            """,
        )
    )
    got, lines, functions, asked, bare, bare_lines, bare_result = json.loads(out.splitlines()[-1])
    # the def the statement is in, not every def of the name; every def where the mover takes rows another selects
    assert got["own statement"]["starts"] == ["p/m.py::FetchData.count"]
    assert [w["function"] for w in got["own statement"]["writes"]] == ["p/w.py::write"]
    assert got["own statement"]["writes"][0]["chain"] == ["p/m.py:7"]
    assert got["rows handed"]["starts"] == ["p/m.py::count", "p/m.py::FetchData.count"]
    assert got["nothing"]["writes"] == [] and got["nothing"]["closure"] == 1
    assert got["hole"]["writes"] is None and got["hole"]["holes"] == 1
    assert got["own arm"]["own_arms"] == ["p/w.py::write"]
    assert got["none holds"]["writes"] is None and got["none holds"]["starts"] == []
    # the record is read once, and the ends once per question
    assert len(lines) == 1 and lines[0].startswith("  reach: the dump's reach_graph, 4 function(s); every answer rests on [")
    assert functions == 4 and asked == [["JobSpec.jobStatus", "running"]]
    # a dump with no reach: nothing asked, said once
    assert bare == [None, None] and bare_result == {"reach": {"present": False}}
    assert bare_lines == ["  reach: the dump has no reach_graph: a mover with no arm of its own is kept (KNOWN_ISSUES.md K14)"]


_MOVERS_MAP = """
    import json
    from types import SimpleNamespace as N
    from bamboo.querier.rerun import arms_of, movers_of

    def branch(line, outcome, pre=()):
        return N(line=line, outcome=outcome, path_condition=[], row_precondition=list(pre))

    def writer(owner, *branches):
        return N(owner=owner, branches=list(branches), log_files=[])

    def stmt(owner, line, selects, region, crosses_to=(), result="rows"):
        return N(owner=owner, line=line, selects=selects, region=region, crosses_to=list(crosses_to), result=result)

    writers = [
        writer("p/jobs.py::getJobs", branch(242, "sent", ["jobStatus='activated'"])),
        writer("p/jobs.py::killJob", branch(30, "cancelled"), branch(31, "running")),
        writer("p/cmd.py::execCommand", branch(459, "aborting"), branch(462, "ready", ["jobStatus IN (:a,:b)"])),
        writer("p/jobs.py::unThrottle", branch(84, "activated", ["jobStatus='throttled'"]), branch(None, "failed")),
        writer("p/jobs.py::touch", branch(12, "failed", ["status='finished'"])),
    ]
"""


def test_an_arm_moves_the_value_unless_its_write_requires_another(tmp_path):
    out = _ok(
        _run(
            tmp_path,
            _MOVERS_MAP + """
    print(json.dumps({w.owner.split("::")[-1]: {line: a.outcomes for line, a in arms_of(writers, w.owner,
                      "JobSpec.jobStatus", "running").items()} for w in writers}))
    """,
        )
    )
    arms = json.loads(out.splitlines()[-1])
    # getJobs and unThrottle write only rows holding another status; writing the value observed is no move (killJob's
    # 31); a bind no one settled keeps execCommand's 462; touch's precondition is of another column
    assert arms == {"getJobs": {}, "killJob": {"30": ["cancelled"]}, "execCommand": {"459": ["aborting"], "462": ["ready"]},
                    "unThrottle": {}, "touch": {"12": ["failed"]}}


def test_a_writer_no_statement_gives_rows_takes_a_row_by_its_key(tmp_path):
    out = _ok(
        _run(
            tmp_path,
            _MOVERS_MAP + """
    from bamboo.querier.rerun import by_key_of

    print(json.dumps(by_key_of(writers, {"p/jobs.py::killJob": "x@1 selects the value"}, "JobSpec.jobStatus", "running")))
    """,
        )
    )
    # a mover already is not again; one with no arm moving the value off is no writer of it
    assert json.loads(out.splitlines()[-1]) == ["p/cmd.py::execCommand", "p/jobs.py::touch"]


def test_judgement_3s_prints_premise_holds_only_where_a_line_before_every_arm_names_the_written_rows_key(tmp_path):
    out = _ok(
        _run(
            tmp_path,
            """
    import json
    from types import SimpleNamespace as N
    from bamboo.querier.rerun import prints_naming

    def name(line, column, row="written"):
        return N(line=line, expression="job.x", column=column, row=row, why="" if row else "rebound")

    def branch(line, prints, names=None, hole=""):
        return N(line=line, must_prints=prints, must_names=names, must_hole=hole)

    key = name(5, "JobSpec.PandaID")
    writers = [N(owner="p/w.py::w", branches=[
        branch(10, [5], [key]),
        branch(11, [5], [key, name(5, "JobSpec.jobStatus")]),
        branch(20, [5], [name(5, "JobSpec.jobStatus")]),
        branch(21, [5], [name(5, "JobSpec.PandaID", row="")]),
        branch(22, [5], None, "names: the arm does not write the subject as an attribute of an object"),
        branch(30, []),
        branch(40, None, None, "not read: a lambda"),
    ])]
    cols = {"JobSpec.PandaID"}
    held = lambda lines, columns=cols: (lambda p: [p["held"], p["why"]])(prints_naming(writers, "p/w.py::w", lines, columns))
    print(json.dumps({"named": held([10, 11]), "another column": held([10, 20]), "not the row": held([21]),
                      "sql": held([22]), "none before": held([10, 30, 20]), "unread": held([10, 40]),
                      "no key column": held([10], set()), "no arm": held([])}))
    """,
        )
    )
    got = json.loads(out.splitlines()[-1])
    # a line before every arm names the written row's key: a run that wrote it printed what the asks search by
    assert got["named"] == [True, "a line before every arm at [10, 11] names the written row's ['JobSpec.PandaID']"]
    # a name of another column, of a row not known to be the written one, or none read (an SQL arm): not known
    for case in ("another column", "not the row", "sql"):
        assert got[case][0] is None and "the map pairs neither a run's mark nor an SQL statement's key bind" in got[case][1]
    assert "[20]" in got["another column"][1] and "[22]" in got["sql"][1]
    # an arm no line comes before on every path: a run need print nothing, whatever the other arms say
    assert got["none before"] == [False, "no line runs before the arm(s) at [30] on every path"]
    assert got["unread"] == [None, "the map did not read the lines before ['40: not read: a lambda']"]
    assert got["no key column"][0] is None and got["no arm"][0] is None


def test_s3_reads_what_an_arm_wrote_only_after_it_in_its_block_before_the_name_is_bound_again(tmp_path):
    out = _ok(
        _run(
            tmp_path,
            """
    import ast, json
    from bamboo.querier import rerun as RR

    tree = ast.parse("\\n".join([
        "def update(status, old):",
        "    log(f'status={status}')",
        "    if old == 'failed':",
        "        status = old",
        "    if ok:",
        "        row['status'] = status",
        "        n = execute(row)",
        "        log(f'old={old} new={status}')",
        "        if n:",
        "            log(f'again {status}')",
        "        status = 'x'",
        "        log(f'later {status}')",
        "    log(f'end {status}')",
    ]))
    fn = tree.body[0]
    arm = next(n for n in ast.walk(fn) if isinstance(n, ast.stmt) and n.lineno == 6)
    print(json.dumps([RR.lines_after(fn, arm, "status"), RR.lines_after(fn, arm, "n")]))
    """,
        )
    )
    status, n = json.loads(out.splitlines()[-1])
    # after the arm in its block: 7, 8 and the if at 9-10; 11 binds status again, so 12 and the end at 13 are not read
    assert status == [[7, 7], [8, 8], [9, 10]]
    # a statement binding the name ends the spans at once: 7 binds n
    assert n == []


def test_s3_goes_on_from_a_guards_if_above_the_arm_or_its_else_and_reads_no_guard_read_elsewhere(tmp_path):
    out = _ok(
        _run(
            tmp_path,
            """
    import ast, json
    from types import SimpleNamespace as N
    from bamboo.codemap.panda import pathcond
    from bamboo.querier import evaluate as E
    from bamboo.querier import rerun as RR

    tree = ast.parse("\\n".join([
        "def f(a, b, c, d):",
        "    if a == 1:",
        "        x = 1",
        "    else:",
        "        if b == 1:",
        "            x = 2",
        "        elif c == 1:",
        "            x = 3",
        "    if d == 1:",
        "        return",
    ]))
    pathcond.attach_parents(tree)
    fn = tree.body[0]

    def at(line):
        return next(n for n in ast.walk(fn) if isinstance(n, ast.stmt) and n is not fn and n.lineno == line)

    found = {line: {g: getattr(RR.guard_if(at(line), g), "lineno", None)
                    for g in [*pathcond.path_condition(at(line)), "b == 1", "not (a == 1)"]} for line in (3, 6, 8)}
    elsewhere = [E.read_elsewhere(g) for g in ["s in ['ready']  [in setOnHold()]", "not (s == '')  [at 391]",
                                               "not (x)  [x := f(y)]", "a == 1"]]

    env = E.Env()
    for name, value in {"a": 1, "b": 1, "c": 0, "d": 1}.items():
        env.put(name, value, "the test")
    r = RR.Rerun.__new__(RR.Rerun)
    r.lines, r.opened, r.hand, r.hand_used, r.hop = [], [], {}, [], None
    arms = [RR.Arm(6, ["two"], ["not (a == 1)", "b == 1", "s in ['ready']  [in setOnHold()]", "not (d == 1)  [at 9]"]),
            RR.Arm(8, ["three"], ["not (a == 1)", "not (b == 1)", "c == 1"])]
    went = r.blocked(RR.Blocked(N(node=fn, name="f"), 10, [None], [env], arms))
    print(json.dumps(found))
    print(json.dumps(elsewhere))
    print(json.dumps([went.guard, went.line]))
    """,
        )
    )
    found, elsewhere, went = (json.loads(line) for line in out.splitlines()[-3:])
    # the test above the arm, the arm in its body; not (t) is the if testing t with the arm in its else, an elif's
    # too; neither where the arm is on the other side, nor for a test no if above the arm holds (KNOWN_ISSUES.md K24)
    assert found == {"3": {"a == 1": 2, "b == 1": None, "not (a == 1)": None},
                     "6": {"not (a == 1)": 2, "b == 1": 5},
                     "8": {"not (a == 1)": 2, "not (b == 1)": 5, "c == 1": 7, "b == 1": None}}
    assert elsewhere == [True, True, False, False]
    # arm 6's callee test and its later write's test (false if read) are read neither false nor named, so arm 6 is
    # nearest with one guard, from the if at 2
    assert went == ["not (a == 1)", 2]


def test_a_statement_hands_its_rows_to_its_owner_only_where_the_owner_writes_them(tmp_path):
    out = _ok(
        _run(
            tmp_path,
            _MOVERS_MAP + """
    follow = N(guarded_by=[N(owner="p/jobs.py::killJob")], reader_statements=[
        # an arm of execCommand's lies in the region: its own rows; recordStatus writes nothing, killJob does
        stmt("p/cmd.py::execCommand", 450, ["JobSpec.jobStatus=running"], [440, 470],
             ["p/log.py::recordStatus", "p/jobs.py::killJob"]),
        # no arm of execCommand's in this region (orphans picked up for runImpl): to what it calls there
        stmt("p/cmd.py::execCommand", 591, ["JobSpec.jobStatus=running"], [573, 603], ["d/cmd.py::runImpl"]),
    ])
    print(json.dumps(movers_of(follow, writers, "JobSpec.jobStatus", "running")))
    """,
        )
    )
    assert json.loads(out.splitlines()[-1]) == {
        "p/cmd.py::execCommand": "execCommand@450 selects the value",
        "p/jobs.py::killJob": "execCommand@450 hands the rows it selects to it",
        "d/cmd.py::runImpl": "execCommand@591 hands the rows it selects to it",
    }


def test_only_a_statement_that_selects_the_value_gives_movers(tmp_path):
    out = _ok(
        _run(
            tmp_path,
            _MOVERS_MAP + """
    follow = N(guarded_by=[N(owner="p/jobs.py::getJobs")], reader_statements=[
        # the map keeps an UPDATE-only owner's statements of other values: they give no mover
        stmt("p/jobs.py::unThrottle", 80, [], [78, 90]),
        stmt("p/cmd.py::execCommand", 368, ["PRODSYS_COMM.comm_owner=DEFT"], [362, 378], ["d/cmd.py::runImpl"]),
        stmt("d/metrics.py::scan", 109, ["JobSpec.jobStatus=running", "JobSpec.VO=atlas"], [100, 120]),
        # a count hands on a value computed from the rows, kept or handed to what the owner calls: no row, no mover
        stmt("d/metrics.py::count", 130, ["JobSpec.jobStatus=running"], [128, 140], result="aggregate"),
        stmt("d/stats.py::rate", 200, ["JobSpec.jobStatus=running"], [198, 210], ["d/gen.py::doGenerate"], "aggregate"),
        # one whose result the map did not read may return rows
        stmt("d/metrics.py::unread", 150, ["JobSpec.jobStatus=running"], [148, 160], result=""),
    ])
    print(json.dumps(movers_of(follow, writers, "JobSpec.jobStatus", "running")))
    """,
        )
    )
    # a statement returning rows with no arm of its owner's keeps them (a mover with no arm of its own); the guarded
    # writer comes last
    assert json.loads(out.splitlines()[-1]) == {
        "d/metrics.py::scan": "scan@109 selects the value",
        "d/metrics.py::unread": "unread@150 selects the value",
        "p/jobs.py::getJobs": "its write requires the value",
    }


def test_a_statement_whose_owner_writes_none_hands_its_rows_to_all_it_calls(tmp_path):
    out = _ok(
        _run(
            tmp_path,
            _MOVERS_MAP + """
    follow = N(guarded_by=[], reader_statements=[
        # main writes nothing: Watcher.run and killJobs take its rows, a writer or not
        stmt("d/archive.py::main", 309, ["JobSpec.jobStatus=running"], [299, 321], ["d/watcher.py::run", "p/tb.py::killJobs"]),
    ])
    print(json.dumps(movers_of(follow, writers, "JobSpec.jobStatus", "running")))
    """,
        )
    )
    assert json.loads(out.splitlines()[-1]) == {
        "d/watcher.py::run": "main@309 hands the rows it selects to it",
        "p/tb.py::killJobs": "main@309 hands the rows it selects to it",
    }


_M2_SETUP = """
    import datetime
    import json
    from types import SimpleNamespace as N
    from bamboo.querier import rerun

    r = object.__new__(rerun.Rerun)
    r.lines, r.hop, r.hand, r.hand_used = [], "", {}, []
    r.result = {}
    r.args = N(subject="JobSpec.jobStatus", key="1")
    r.data = {
        "classes": {"job": "JobSpec"},
        "rows": {"job": {"PandaID": {"value": 1}, "creationTime": {"value": "2026-01-01 00:00:00"},
                         "gshare": {"value": "Express"}, "VO": {"value": None}}},
        "after": {"source": "job/get_description", "modificationTime": "2026-01-03 00:00:00"},
        "statuslog": {"states": []},
    }
    t0, t1 = datetime.datetime(2026, 1, 1, 12), datetime.datetime(2026, 1, 5)
    q = rerun.Missed("JobSpec.jobStatus", "activated", "sent", t0, t1, [], "", None)

    def val(constant="", base="", offset=None):
        return N(kind="bind", text=":v", constant=constant, base=base, offset_seconds=offset, needs=[], said="")

    def term(node, subject, op, values, settled=True):
        return N(node=node, items=[], subject=subject, op=op, values=values, settled=settled, text=f"{subject} {op}")

    def group(node, *items):
        return N(node=node, items=list(items))

    def spans(found):
        return [[str(a), str(b)] for a, b in found]

    status = term("equal", "JobSpec.jobStatus", "IN", [val("defined"), val("activated")])
    # creationTime < now - 2 d: true from creationTime + 2 d on
    old = term("compare", "JobSpec.creationTime", "<", [val(base="naive_utcnow", offset=-2 * 86400)])
    share = term("equal", "JobSpec.gshare", "=", [val("User")])
    touched = term("compare", "JobSpec.modificationTime", "<", [val(base="naive_utcnow", offset=-3600)])

    def stmt(keys=(), restricts=("JobSpec.jobStatus=activated",), refuses=(), trees=(), routes=()):
        return N(keys=list(keys), restricts=list(restricts), refuses=list(refuses),
                 condition_trees=[N(entry="e", tree=t) for t in trees], call_readings=list(routes))

    def route(trees=(), not_run=False, as_statement=False):
        return N(not_run=not_run, as_statement=as_statement, restricts=[], refuses=[],
                 condition_trees=[N(entry="e", tree=t) for t in trees])
"""


def test_a_tree_is_surely_true_only_where_every_term_it_needs_is_known_true(tmp_path):
    out = _ok(
        _run(
            tmp_path,
            _M2_SETUP + """
    at = datetime.datetime(2026, 1, 3, 12)
    stuck = rerun.Missed("JobSpec.jobStatus", "activated", "", at, at, [], "", None, at_end=True)
    print(json.dumps({
        "status": spans(r._true_spans(status, q)),
        "old": spans(r._true_spans(old, q)),
        "all": spans(r._true_spans(group("all", status, old), q)),
        "any": spans(r._true_spans(group("any", share, old), q)),
        "share": spans(r._true_spans(share, q)),
        "unsettled": spans(r._true_spans(term("equal", "JobSpec.gshare", "=", [val("Express")], settled=False), q)),
        "null": spans(r._true_spans(term("equal", "JobSpec.VO", "IN", [val("atlas")]), q)),
        "after, late": spans(r._true_spans(touched, q)),
        "after, at as_of": spans(r._true_spans(touched, stuck)),
    }))
    """,
        )
    )
    found = json.loads(out.splitlines()[-1])
    whole, turned = [["2026-01-01 12:00:00", "2026-01-05 00:00:00"]], [["2026-01-03 00:00:00", "2026-01-05 00:00:00"]]
    assert found["status"] == whole and found["old"] == turned and found["all"] == turned and found["any"] == turned
    # false, unsettled, a NULL (the NULL rule is not here), and late's after (read after the window): never true
    assert found["share"] == found["unsettled"] == found["null"] == found["after, late"] == []
    # at as_of, after holds the value then: modificationTime + 1 h is before as_of
    assert found["after, at as_of"] == [["2026-01-03 12:00:00", "2026-01-03 12:00:00"]]


def test_a_statement_is_surely_true_only_without_a_run_time_key_and_on_a_running_route(tmp_path):
    out = _ok(
        _run(
            tmp_path,
            _M2_SETUP + """
    print(json.dumps({
        "plain": spans(r._statement_true(stmt(trees=[old]), q)),
        "keyed": spans(r._statement_true(stmt(keys=["JobSpec.jediTaskID"], trees=[old]), q)),
        "restricted away": spans(r._statement_true(stmt(restricts=["JobSpec.jobStatus=defined"]), q)),
        "refused": spans(r._statement_true(stmt(refuses=["JobSpec.gshare=Express"]), q)),
        "no route runs": spans(r._statement_true(stmt(routes=[route(not_run=True)]), q)),
        "one route": spans(r._statement_true(stmt(trees=[old], routes=[route([share]), route(as_statement=True)]), q)),
        "a false route": spans(r._statement_true(stmt(routes=[route([share])]), q)),
    }))
    """,
        )
    )
    found = json.loads(out.splitlines()[-1])
    turned = [["2026-01-03 00:00:00", "2026-01-05 00:00:00"]]
    assert found["plain"] == turned and found["one route"] == turned
    assert found["keyed"] == found["restricted away"] == found["refused"] == found["no route runs"] == found["a false route"] == []


def test_l2_says_each_movers_verdict(tmp_path):
    out = _ok(
        _run(
            tmp_path,
            _M2_SETUP + """
    def st(owner, line, trees=()):
        return N(owner=owner, line=line, keys=[], selects=["JobSpec.jobStatus=activated"], excludes=[],
                 restricts=["JobSpec.jobStatus=activated"], refuses=[],
                 condition_trees=[N(entry="e", tree=t) for t in trees], call_readings=[])

    def writer(owner, outcome, pre=()):
        return N(owner=owner, log_files=[], branches=[N(line=5, outcome=outcome, path_condition=[], row_precondition=list(pre))])

    a, b, c = st("d/a.py::main", 10, [share]), st("d/a.py::main", 20, [status]), st("d/a.py::main", 30, [touched])
    q.movers = {"d/a.py::gone": "main@10 hands the rows it selects to it", "d/a.py::open": "main@20 hands the rows it selects to it",
                "d/a.py::dark": "main@30 hands the rows it selects to it", "p/w.py::armed": "main@10 hands the rows it selects to it",
                "p/w.py::moved": "its write requires the value"}
    q.paths = {"d/a.py::gone": [a], "d/a.py::open": [b], "d/a.py::dark": [c], "p/w.py::armed": [a], "p/w.py::moved": []}
    q.by_key, q.refusing, q.mover = ["p/w.py::keyed"], ["p/w.py::refusing"], "p/w.py::moved"
    q.writers = [writer("p/w.py::armed", "failed"), writer("p/w.py::keyed", "closed"), writer("p/w.py::moved", "sent"),
                 writer("p/w.py::refusing", "sent", ["jobStatus='defined'"])]
    verdicts = r._movers_in_window(q, {("d/a.py::main", 10): "refused", ("d/a.py::main", 20): "not settled",
                                      ("d/a.py::main", 30): "not settled"})
    print(json.dumps([verdicts, r.lines[-1], r.result["movers"]]))
    """,
        )
    )
    verdicts, last, movers = json.loads(out.splitlines()[-1])
    # result.json's movers: the same verdicts, and a class for each reason one is left
    by_owner = {m["owner"]: m for m in movers}
    assert {o: m["verdict"] for o, m in by_owner.items()} == {o: v.replace(" ", "-") for o, v in map(tuple, verdicts)}
    left = [m for m in movers if m["verdict"] not in ("moved-it", "write-refuses", "refused")]
    assert last == f"not ruled out: {len(left)} mover(s)"
    assert {o: [r["code"] for r in m["reasons"]] for o, m in by_owner.items()} == {
        "d/a.py::gone": ["refused"],
        "d/a.py::open": ["K14"],
        "d/a.py::dark": ["K14", "after-only"],  # late: modificationTime is the value after the move
        "p/w.py::armed": ["K15"],
        "p/w.py::moved": [],
        "p/w.py::keyed": ["K15"],
        "p/w.py::refusing": ["write-refuses"],
    }
    assert by_owner["d/a.py::dark"]["reasons"][1]["text"].startswith("main@30: JobSpec.modificationTime not compared (written at")
    assert [(m["reached"], m["statements"]) for m in movers[:5]] == [
        ("select", ["main@10"]), ("select", ["main@20"]), ("select", ["main@30"]), ("select", ["main@10"]), ("guard", [])]
    assert dict(map(tuple, verdicts)) == {
        "d/a.py::gone": "refused",  # no arm of its own, and its one statement refuses the row
        "d/a.py::open": "could have",  # its statement holds for the row all the window
        "d/a.py::dark": "undecided",  # modificationTime is after the window in late
        "p/w.py::armed": "by key",  # a writer: refused by its statement, not by its key
        "p/w.py::moved": "moved it",
        "p/w.py::keyed": "by key",
        "p/w.py::refusing": "write refuses",
    }
    assert last == "not ruled out: 4 mover(s)"


def test_the_window_end_reads_after_only_at_as_of(tmp_path):
    out = _ok(
        _run(
            tmp_path,
            _M2_SETUP + """
    at = datetime.datetime(2026, 1, 3, 12)
    stuck = rerun.Missed("JobSpec.jobStatus", "activated", "", at, at, [], "", None, at_end=True)
    print(json.dumps([r._held("JobSpec.modificationTime", q), r._held("JobSpec.modificationTime", stuck),
                      r._held("JobSpec.source", stuck), r._unanswered("JobSpec.modificationTime", q),
                      r._unanswered("JobSpec.modificationTime", stuck)]))
    """,
        )
    )
    late, stuck, source, late_why, stuck_why = json.loads(out.splitlines()[-1])
    assert late == [False, None] and stuck == [True, "2026-01-03 00:00:00"] and source == [False, None]
    assert late_why == "written at the move or later; its row here holds only the value after" and stuck_why is None


def test_a_value_the_builder_read_at_as_of_is_read_as_after_is_and_not_as_the_runs(tmp_path):
    out = _ok(
        _run(
            tmp_path,
            """
    import json
    from bamboo.querier import held
    read = {"as_of": "2026-01-10 00:00:00", "clock": "client", "source": "job/get_description"}
    data = {
        "classes": {"job": "JobSpec", "task": "JediTaskSpec"},
        "rows": {"job": {"PandaID": {"value": 1, **read}, "gshare": {"value": "Express", **read},
                         "VO": {"value": None, "source": "tmaeno SQL"}},
                 "task": {"jediTaskID": {"value": 5, **read}}},
        "after": {"source": "job/get_description", "destinationSE": "X"},
        "statuslog": {"states": [{"status": "activated", "time": "2026-01-01 00:00:10"},
                                 {"status": "running", "time": "2026-01-01 00:00:20"},
                                 {"status": "failed", "time": "2026-01-11 00:00:00"}]},
    }
    rows, after = held.split(data, "JobSpec", "1")
    kept, unread = held.at_run(data, "JobSpec", "1", "2026-01-01 00:00:15,000")
    _, untimed = held.at_run(data, "JobSpec", "1", None)
    print(json.dumps([{e: sorted(r) for e, r in rows.items()}, after, kept == rows, unread, untimed["<job>.gshare"]]))
    """,
        )
    )
    rows, after, same, unread, untimed = json.loads(out.splitlines()[-1])
    # the hand's value and the row's key stay; the subject's other value goes to after, with its time; another row's
    # value is read by neither
    assert rows == {"job": ["PandaID", "VO"], "task": []}
    assert after == {"source": "job/get_description", "destinationSE": "X", "gshare": "Express",
                     "as_of": "2026-01-10 00:00:00", "clock": "client"}
    assert same
    tail = ("and writes that leave no such row may lie between too; which writes between can change it is not known yet "
            "(KNOWN_ISSUES.md K22)")
    between = f"read at 2026-01-10 00:00:00 (client clock); 1 statuslog row(s) after the run's 2026-01-01 00:00:15,000, {tail}"
    assert unread == {"<job>.gshare": between, "<task>.jediTaskID": between, "<job>.destinationSE": between}
    assert untimed.startswith("read at 2026-01-10 00:00:00 (client clock), not at the step's run;")


def test_a_step_says_why_a_row_value_it_needs_is_not_its_runs(tmp_path):
    out = _ok(
        _run(
            tmp_path,
            _M2_SETUP + """
    from bamboo.querier import evaluate as E
    r.data["spellings"] = {"job": ["job", "self.job"]}
    for v in r.data["rows"]["job"].values():
        v.setdefault("source", "tmaeno SQL")
    r.data["rows"]["job"]["gshare"] |= {"as_of": "2026-01-03 00:00:00", "clock": "client"}
    r.row_facts = {}
    env = r.base_env(at="2026-01-01 00:00:01,000")
    test = "self.job.gshare != 'Express' and job.VO is None"
    said = [E.evaluate(test, env) is E.UNKNOWN, env.unread_in(test), "gshare" in env.table["job"],
            r._held("JobSpec.gshare", q), r._unanswered("JobSpec.gshare", q)]
    env.put("job.gshare", "Express", "printed by the run")
    said += [E.evaluate(test, env), env.unread_in(test)]
    print(json.dumps(said))
    """,
        )
    )
    unknown, why, in_table, held, late_why, printed, after_print = json.loads(out.splitlines()[-1])
    assert unknown and not in_table
    assert list(why) == ["self.job.gshare"] and why["self.job.gshare"].startswith(
        "read at 2026-01-03 00:00:00 (client clock); 0 statuslog row(s) after the run's 2026-01-01 00:00:01,000")
    # late reads it as it reads after: not over the window
    assert held == [False, None] and late_why == "written at the move or later; its row here holds only the value after"
    # what the run printed is the run's, and says nothing is missing
    assert printed is False and after_print == {}


def test_f1_asks_for_every_row_after_the_run_before_one_stop_and_then_reads_each(tmp_path):
    out = _ok(
        _run(
            tmp_path,
            _M2_SETUP + """
    r.opened, r.unanswered, r._rows_read, r._readings = [], [], set(), {}
    r.args.asks = "asks.jsonl"
    r.hop = "H7"
    r.data["statuslog"] = {"as_of": "2026-01-04 00:00:00", "states": [
        {"status": "defined", "time": "2026-01-01 00:00:00", "host": "m1"},
        {"status": "activated", "time": "2026-01-01 00:00:02", "time_extended": "2026-01-01 00:00:02.5", "host": "m1"},
        {"status": "sent", "time": "2026-01-02 00:00:00", "host": "m2"},
        {"status": "failed", "time": "2026-01-05 00:00:00", "host": "m3"}]}
    asked = []
    r._writers = lambda subject, value, key: asked.append((subject, value)) or [
        N(file=f"{value}.py", owner=f"{value}.py::C.write", lines=[1], log_files=[f"{value}.log"])]
    r.files_of = lambda owner, today, lines: list(today)
    answered = False

    def ask(key, names, asking, owners=None):
        if answered:
            return {}, [], {}
        return {}, [(N(key=f"{names[0]}@{asking.machine}"), asking.why)], {}

    r._ask = ask
    try:
        r.rows_between("2026-01-01 00:00:01,000")
        stop = None
    except rerun.Stop as e:
        stop = str(e)
    first = [stop, r.hop, list(asked), [a.key for _, a, _ in r.unanswered]]
    r._rows_read, r.unanswered, r.hop, answered = set(), [], "H7", True
    r.rows_between("2026-01-01 00:00:01,000")
    again = r.rows_between("2026-01-01 00:00:01,000")
    print(json.dumps(first + [r.hop, r.lines[-1], [x for x in r.lines if x.startswith("   no run")], r.opened]))
    """,
        )
    )
    stop, hop, asked, keys, hop2, last, none_run, opened = json.loads(out.splitlines()[-1])
    # the rows after the run's start to as_of: not the one before it, not the one after as_of; one stop for both
    assert asked == [["JobSpec.jobStatus", "activated"], ["JobSpec.jobStatus", "sent"]]
    assert keys == ["activated.log@m1", "sent.log@m2"]
    assert stop.startswith("F1: 2 ask(s) of the logs not in the records") and hop == "F1"
    # answered: each row read, none stops, and the step that asked goes on under its own name
    assert len(none_run) == 2 and last.startswith("F1: a writer chosen for 0 of 2 row(s)") and hop2 == "H7"
    assert opened == ["F1", "F1"]


_LEFT = """
def f(self):
    self.logger.debug("starting the update of the job")
    try:
        self.logger.debug("asking the broker for a site")
        x = g()
        self.logger.debug("the broker answered the request")
    except Exception:
        pass
    y = h(x)
    self.logger.debug("the update of the job was handed")
    if x:
        z = x.size
    self.logger.debug("finished the update of the job")
"""


def test_a_candidates_must_rows_stop_where_a_traceback_of_its_run_shows_the_call_left(tmp_path):
    (_corpus(tmp_path) / "pandaserver" / "a.py").write_text(_LEFT)
    stamp = "2026-01-01 00:00:0{},000 panda.log.X: DEBUG    <PandaID=1> {}\n"
    frame = '  File "/opt/panda/lib/python3.11/site-packages/pandaserver/a.py", line {}, in f\n    {}\n'
    tb = "Traceback (most recent call last):\n" + frame + "ValueError: no\n"
    logs = {
        "raised": stamp.format(0, "starting the update of the job") + stamp.format(1, "failed") + tb.format(10, "y = h(x)"),
        "unplaced": stamp.format(0, "starting the update of the job") + stamp.format(1, "failed") + tb.format(10, "y = h(x, 1)"),
        "none": stamp.format(0, "starting the update of the job"),
        "caught": stamp.format(0, "starting the update of the job") + stamp.format(1, "failed") + tb.format(6, "x = g()")
        + stamp.format(2, "the update of the job was handed") + stamp.format(3, "finished the update of the job"),
    }
    for name, text in logs.items():
        (tmp_path / f"{name}.log").write_text(text)
    out = _ok(
        _run(
            tmp_path,
            """
            import json
            from bamboo.querier import rows as R
            from bamboo.querier.records import read_log, runs_for
            from bamboo.querier.which_ran import Candidate, judge

            R.ROWS_FROM = "walk"
            fn = R.function("pandaserver/a.py", "f", (10,))
            site = R.statement_at(fn.node, 10)
            lines = lambda found: [[s.lineno for s in part] for part in found]
            must = {name: lines(R.must_rows(fn.node, site, raised)) for name, raised in
                    [("none", set()), ("site", {10}), ("in the if", {13}), ("in the try", {6}), ("unplaced", None)]}
            got = {}
            for name in ("raised", "unplaced", "none", "caught"):
                run, = runs_for(read_log(f"{name}.log"), "1")
                c = Candidate("pandaserver/a.py", "f", [10], [run.log.name], run=run)
                v, = judge([c], {run.log.name: [run]}, question_run=run).verdicts
                at = R.raised_in(fn, run)[1]
                got[name] = [sorted(at) if at is not None else None, v.kind, v.reasons]
            print(json.dumps([must, got]))
            """,
        )
    )
    must, got = json.loads(out.splitlines()[-1])
    # no traceback: every print after the call until one that can leave is a must row
    assert must["none"] == [[3], [11, 14]]
    # the call left at the statement holding the frame's line, the site itself included (a call after the site
    # stops the must rows anyway: it can leave)
    assert must["site"] == [[3], []]
    assert must["in the if"] == [[3], [11]]
    # a frame inside a try before the site: the try printed up to it, and the call went on to the site
    assert must["in the try"] == [[3, 5], [11, 14]]
    # where it left is not known: no print after the site is required
    assert must["unplaced"] == [[3], []]
    # the judge reads each candidate's own run: the rows after the call that raised are not looked for
    assert got["raised"][:2] == [[10], "kept"] and "a traceback frame raised at [10]" in got["raised"][2]
    assert not any(r.startswith("rule 4") for r in got["raised"][2])
    assert got["unplaced"][:2] == [None, "kept"]
    assert "a traceback frame of f cannot be placed in the corpus: no row after 10 is required" in got["unplaced"][2]
    # with no traceback the same run excludes it; a try that raised before the site owes its prints up to the frame
    assert got["none"][1] == "excluded" and "rule 4: must row(s) [11, 14] not in the run" in got["none"][2]
    assert got["caught"][:2] == [[6], "excluded"] and "rule 4: must row(s) [5] not in the run" in got["caught"][2]


def test_a_row_is_missing_only_from_a_run_of_a_file_the_map_places_it_in(tmp_path):
    (_corpus(tmp_path) / "pandaserver" / "a.py").write_text(_LEFT)
    for name in ("panda-a.log", "panda-b.log"):
        (tmp_path / name).write_text("2026-01-01 00:00:00,000 panda.log.X: DEBUG    <PandaID=1> asking the broker for a site\n")
    out = _ok(
        _run(
            tmp_path,
            """
            import json
            from types import SimpleNamespace as N
            from bamboo.querier import rows as R
            from bamboo.querier.records import read_log

            rel = "pandaserver/a.py"
            mapped = lambda text: [N(pattern=text, refused="", keyed=False)]
            prints = N(anchor=N(blob_sha=R._blob_sha(rel)), statements=[
                N(line=3, rows=mapped("starting the update of the job"), log_files=["panda-a.log"], log_files_basis="logger"),
                N(line=5, rows=mapped("asking the broker for a site"), log_files=["panda-b.log"], log_files_basis="logger"),
                N(line=7, rows=mapped("the broker answered the request"), log_files=["panda-b.log"], log_files_basis="module"),
            ])
            R._function_prints = lambda owner: prints
            R._prints_of = lambda owner, lines=(): prints
            rows = {r.line: r for r in R.function(rel, "f").rows}
            a, b = read_log("panda-a.log"), read_log("panda-b.log")
            print(json.dumps([
                {line: r.log_files for line, r in rows.items()},
                R.split_files(f"{rel}::f"),
                [R.absence_trusted(rows[3], log, "map") for log in (a, b)],
                [R.absence_trusted(rows[5], log, "map") for log in (a, b)],
                [rows[7].lands_in(log.name) for log in (a, b)],
            ]))
            """,
        )
    )
    files, split, row3, row5, row7 = json.loads(out.splitlines()[-1])
    # a statement's files are the map's where it follows the logger; another basis does not place it
    assert files == {"3": ["panda-a.log"], "5": ["panda-b.log"], "7": None, "11": None, "14": None}
    assert split == [["panda-a.log"], ["panda-b.log"]]
    # missing from a run of a file it lands in is evidence; from another file's run it is not
    assert row3 == [[True, ""], [False, "another file"]]
    assert row5 == [[False, "another file"], [True, ""]]
    assert row7 == [True, True]


def test_stuck_reads_at_the_moment_the_rows_file_says_its_values_were_read(tmp_path):
    out = _ok(
        _run(
            tmp_path,
            _M2_SETUP + """
    said = []
    r.data["statuslog"] = {"as_of": "2026-01-03 00:00:09", "clock": "database", "states": []}
    said.append(r._reading_moment("JobSpec.jobStatus", "2026-01-03 00:00:09"))  # the file says nothing
    r.data["after"] |= {"as_of": "2026-01-03 00:00:00", "clock": "client"}
    for v in r.data["rows"]["job"].values():
        v |= {"as_of": "2026-01-03 00:00:00", "clock": "client"}
    said.append(r._reading_moment("JobSpec.jobStatus", "2026-01-03 00:00:09"))  # one time
    said.append(r._reading_moment("JobSpec.jobStatus", "2026-01-02 23:59:59"))  # after the statuslog's as_of
    r.data["after"]["as_of"] = "2026-01-03 00:00:01"
    said.append(r._reading_moment("JobSpec.jobStatus", "2026-01-03 00:00:09"))  # two times
    print(json.dumps([[str(at), words, header] for at, words, header in said] + [r.result["readings_at"],
                      r._held("JobSpec.as_of", rerun.Missed("JobSpec.jobStatus", "activated", "", t0, t0, [], "", None, at_end=True))]))
    """,
        )
    )
    *said, readings_at, held = json.loads(out.splitlines()[-1])
    assert said[0] == ["2026-01-03 00:00:09", "the statuslog was read", "the movers' statements at the statuslog's as_of "
                       "2026-01-03 00:00:09; the rows file does not say when its values were read (a reading, not a drop: "
                       "KNOWN_ISSUES.md K17):"]
    assert said[1] == ["2026-01-03 00:00:00", "the row's values were read", "the movers' statements at 2026-01-03 00:00:00 "
                       "(client clock), when the rows file's values were read (the statuslog's as_of is 2026-01-03 00:00:09; "
                       "a reading, not a drop: KNOWN_ISSUES.md K17):"]
    assert "the statuslog's as_of is 2026-01-02 23:59:59; after it, so the value may have moved by then;" in said[2][2]
    assert said[3][0] == "2026-01-03 00:00:09" and "not at one time" in said[3][2]
    assert readings_at == {"at": "2026-01-03T00:00:09Z", "clock": "database", "from": "statuslog"}
    # when the file was read is not a column of the row
    assert held == [False, None]


def test_a_column_holding_null_fails_every_comparison_but_is_null(tmp_path):
    out = _ok(
        _run(
            tmp_path,
            _M2_SETUP + """
    unsettled = N(kind="bind", text=":es", constant="", base="", offset_seconds=None, needs=["a caller"], said="")
    print(json.dumps({
        "in, unsettled": r._leaf(term("equal", "JobSpec.VO", "IN", [unsettled], settled=False), q),
        "not in": r._leaf(term("unequal", "JobSpec.VO", "NOT IN", [val("atlas")]), q),
        "compare": r._leaf(term("compare", "JobSpec.VO", ">", [val("1")]), q),
        "is null": r._leaf(term("null", "JobSpec.VO", "", []), q),
        "is not null": r._leaf(term("not_null", "JobSpec.VO", "", []), q),
        "other": r._leaf(N(node="other", items=[], subject="JobSpec.VO", op="", values=[], settled=False, text="NVL(VO,'x')='x'"), q),
        "a pair": r._pair_fails("JobSpec.VO", {"atlas"}, q),
        "true": r._true_spans(term("equal", "JobSpec.VO", "IN", [val("atlas")]), q),
    }))
    """,
        )
    )
    found = json.loads(out.splitlines()[-1])
    null = "the row holds NULL, which no comparison but IS NULL takes"
    assert found["in, unsettled"] == found["not in"] == found["compare"] == found["a pair"] == null
    # IS NULL takes it, IS NOT NULL says what the row holds, as before; an unread term is not read; never true
    assert found["is null"] is None and found["is not null"] == "the row holds None"
    assert found["other"] is None and found["true"] == []


def test_l2_stops_on_a_late_mover_it_cannot_read_yet_and_sends_one_lost_on_order_to_l3(tmp_path):
    out = _ok(
        _run(
            tmp_path,
            _M2_SETUP + """
    r.hop = "L2"

    def st(line, order=False):
        return N(owner="p/w.py::moved", line=line, keys=[], restricts=["JobSpec.jobStatus=activated"], refuses=[],
                 condition_trees=[N(entry="e", tree=status)], call_readings=[], order_by=["x"] if order else [],
                 limits=["<= :n"] if order else [])

    q.mover, said = "p/w.py::moved", []
    for statements, ordered in (([], []), ([st(10)], []), ([st(10, True)], [st(10, True)])):
        q.statements = statements
        try:
            r._why_late(q, ordered)
            said.append("on to L3")
        except rerun.Stop as e:
            said.append(str(e))
    print(json.dumps(said))
    """,
        )
    )
    by_key, unordered, ordered = json.loads(out.splitlines()[-1])
    assert by_key == ("L2: why w.py::moved moved it only at 2026-01-05 00:00:00: it takes the row by its key, and its runs "
                      "in the window are read pass by pass (S2-S4 over the window; build order 2, KNOWN_ISSUES.md K15)")
    assert unordered == ("L2: why w.py::moved took it only at 2026-01-05 00:00:00: moved@10 held for the row all the window, "
                         "and it returns no first rows of an order (not read)")
    assert ordered == "on to L3"


def test_s2_stops_on_a_mover_with_no_arm_of_its_own(tmp_path):
    out = _ok(
        _run(
            tmp_path,
            """
            import json
            from bamboo.querier import rerun

            r = object.__new__(rerun.Rerun)
            r.lines, r.hop, r.hand, r.hand_used = [], "", {}, []
            r.opened, r.ends, r.branches = [], [], []
            try:
                r.passes(rerun.Passes("p/tb.py::killJobs", ["panda-TaskBuffer.log"], [], "1", None))
                stopped = ""
            except rerun.Stop as e:
                stopped = str(e)
            print(json.dumps(stopped))
            """,
        )
    )
    # before S3, whose nearest arm is the fewest guards over the arms, and none would raise
    assert json.loads(out.splitlines()[-1]) == (
        "S2: an arm of killJobs's own that writes another value (S2 reads no arm of what it calls: KNOWN_ISSUES.md K14)"
    )


def test_the_record_door_notes_the_jedi_clocks(tmp_path):
    out = _ok(
        _run(
            tmp_path,
            textwrap.dedent(_RECORDED)
            + textwrap.dedent("""
            W = ("2026-09-28 08:00:00", "2026-09-28 08:00:00")
            ask = Ask("panda-x.log", ("12345",), W)
            jedi, _ = ask_of([record(ask, {M1: {**answered({**TODAY, "records": []}), "service": "jedi"}})], "panda-x.log", "12345", W)
            server, _ = ask_of([record(ask, {M1: answered({**TODAY, "records": []})})], "panda-x.log", "12345", W)
            print(json.dumps([jedi.notes, server.notes]))
            """),
        )
    )
    jedi, server = json.loads(out)
    assert jedi == ["JEDI's machine(s) ['aipanda093.cern.ch']: the windows take its log clock as the database's within "
                    "their slack, measured on some of JEDI's machines only"]
    assert server == []


def test_a_pass_inside_a_mark_that_spans_the_search_is_whole_and_one_at_its_end_is_not(tmp_path):
    out = _ok(
        _run(
            tmp_path,
            textwrap.dedent(_RECORDED)
            + textwrap.dedent("""
            W, TOK = ("2026-09-28 12:00:00", "2026-09-28 12:00:00"), "< jediTaskID=12345 >"
            day = {**TODAY, "records": [{**line(t, "pass", TOK), "offset": n} for n, t in enumerate(["06:41:00", "12:00:00", "12:05:00"])]}
            first = record(Ask("panda-x.log", ("12345",), W), {M1: answered(day)})
            log, edges = ask_of([first], "panda-x.log", "12345", W)
            run, = runs_for(log, "12345")
            Path("asks-r1.jsonl").write_text(json.dumps({"ask": first["ask"]}) + "\\n")
            Path("records-r1.jsonl").write_text(json.dumps(first) + "\\n")
            in_dir = len(runs_for(RecordedLogs(["."]).log("panda-x.log", "12345", W), "12345"))
            print(json.dumps([[a["window"] for a, _ in edges], in_dir, run.would_print("ERROR"),
                              run.part(0, 2).would_print("ERROR"), run.part(1, 3).would_print("ERROR")]))
            """),
        )
    )
    edges, in_dir, whole_run, first_pass, inner_pass = json.loads(out)
    # the mark's first record is a minute after the file's start, but hours before the window: its part near the
    # window (12:00-12:05) is far from every end, so nothing is widened
    assert edges == []
    # a directory is read as its records-r*.jsonl: the asks file beside them is not a record
    assert in_dir == 1
    # the run, and its pass at that end, may go on before what was searched
    assert whole_run is False and first_pass is False
    # the passes inside are judged by their own records
    assert inner_pass is True


def test_the_result_says_the_outcome_with_its_code_and_a_run_that_does_not_finish_leaves_no_file(tmp_path):
    out = _ok(
        _run(
            tmp_path,
            """
            import json
            from pathlib import Path
            from bamboo.querier import rerun

            r = object.__new__(rerun.Rerun)
            r.lines, r.hop, r.hand, r.hand_used = [], "", {}, []
            r.opened, r.ends, r.branches = [], [], []
            r.start("L2", "why the others passed it over")
            r.end("lost on order: here")
            try:
                r.need("why it moved it only then", None, "K15")
            except rerun.Stop as e:
                stop = [str(e), e.code]
            rerun._write(str(Path("result.json")), "{}")
            times = [rerun._iso("2026-10-14 11:25:12,437"), rerun._iso("2026-10-10 15:26:10.014958"), rerun._iso(None)]
            data = {
                "statuslog": {"as_of": "2026-10-14 13:11:59", "clock": "database", "states": []},
                "after": {"source": "s", "as_of": "2026-10-14 13:11:50", "clock": "client", "modificationTime": "x"},
                "rows": {"job": {"PandaID": {"value": 1, "source": "s", "as_of": "2026-10-14 13:11:50", "clock": "client"},
                                 "jobStatus": {"value": "running", "source": "s"}}},
                "files": [{"lfn": {"value": "a", "source": "s", "as_of": "2026-10-14 13:11:51", "clock": "client"}}],
                "classes": {"job": "JobSpec"},
            }
            print(json.dumps([r.opened, r.ends, stop, sorted(p.name for p in Path().iterdir()), times,
                              rerun._read_at(data), rerun._read_at({"rows": {"job": {"x": {"value": 1, "source": "s"}}}})]))
            """,
        )
    )
    opened, ends, stop, files, times, read_at, none = json.loads(out.splitlines()[-1])
    assert opened == ["L2"]
    assert ends == [["L2", "END lost on order: here"]]
    assert stop == ["L2: why it moved it only then", "K15"]
    # written whole, through a file renamed over it
    assert "result.json" in files and "result.json.tmp" not in files
    assert times == ["2026-10-14T11:25:12.437Z", "2026-10-10T15:26:10.014958Z", None]
    assert read_at == [
        {"section": "statuslog", "at": "2026-10-14T13:11:59Z", "clock": "database"},
        {"section": "after", "at": "2026-10-14T13:11:50Z", "clock": "client"},
        {"section": "rows.job", "at": "2026-10-14T13:11:50Z", "clock": "client"},
        {"section": "files", "at": "2026-10-14T13:11:51Z", "clock": "client"},
    ]
    assert none is None
    # a run that stops before its end leaves none of an earlier run's files behind
    for name in ("o.txt", "o.txt.asks.jsonl", "o.txt.result.json"):
        (tmp_path / name).write_text("an earlier run's")
    done = _run(
        tmp_path,
        """
        import sys
        from bamboo.querier import rerun
        sys.argv = ["rerun", "--subject", "JobSpec.jobStatus", "--observed", "running", "--key", "1", "--log", "x.log",
                    "--rows", "rows.json", "--out", "o.txt"]
        rerun.main()
        """,
        MAP_DUMP=None,
    )
    assert done.returncode != 0 and "MAP_DUMP" in done.stderr
    assert not [n for n in ("o.txt", "o.txt.asks.jsonl", "o.txt.result.json") if (tmp_path / n).exists()]


def test_the_tree_hash_moves_with_the_code_only(tmp_path):
    out = _ok(
        _run(
            tmp_path,
            """
            import json
            from pathlib import Path
            from bamboo.querier import rerun

            root = Path("tree")
            (root / "bamboo/codemap/__pycache__").mkdir(parents=True)
            (root / "bamboo/codemap/trace.py").write_text("x = 1\\n")
            (root / "bamboo/a.py").write_text("y = 2\\n")
            seen = [rerun._tree_sha256(root)]
            (root / "bamboo/codemap/__pycache__/trace.cpython-311.pyc").write_bytes(b"\\0")
            (root / "bamboo/codemap/notes.txt").write_text("not code")
            seen.append(rerun._tree_sha256(root))
            (root / "bamboo/codemap/trace.py").write_text("x = 2\\n")
            seen.append(rerun._tree_sha256(root))
            print(json.dumps(seen))
            """,
        )
    )
    first, unmoved, moved = json.loads(out.splitlines()[-1])
    assert len(first) == 64 and first == unmoved and moved != first


_FILLED = """def f(self):
    d = {}
    d.setdefault("a", {})
    d["b"] = 2
    d.setdefault("b", 3)
    e = {}
    e["a"] = []
    e["a"].append(1)
    g = {}
    g.setdefault("a", []).append(1)
    return d, e, g
"""


def test_a_container_is_read_by_the_form_of_each_write_and_not_rebuilt_where_a_place_in_it_is_written(tmp_path):
    (_corpus(tmp_path) / "pandaserver" / "a.py").write_text(_FILLED)
    out = _ok(
        _run(
            tmp_path,
            """
            import json
            from types import SimpleNamespace as N
            from bamboo.querier import asked as A
            from bamboo.querier import evaluate as E
            from bamboo.querier import false_guard as G
            from bamboo.querier import rows as R

            R.ROWS_FROM = "walk"
            owner = "pandaserver/a.py::f"
            # the steps the map's walk makes for these writes: a fill carries its last argument and its method
            made = [("d", 2, "{}", "="), ("d", 3, "{}", ".setdefault"), ("d", 4, "2", "="), ("d", 5, "3", ".setdefault"),
                    ("e", 6, "{}", "="), ("e", 7, "[]", "="), ("g", 9, "{}", "="), ("g", 10, "[]", ".setdefault")]
            steps = [N(name=n, owner=owner, kind="binding", line=line, value=v, operator=op, guards=[], unseen=[], reads=[],
                       terminal="", detail="", depth=1, cls="") for n, line, v, op in made]
            walk = G.Walk(A.given(steps), E.Env(), {}, {})
            got = {}
            for name in ("d", "e", "g"):
                value, notes = walk.build(name, owner)
                keys = walk.keys(name, owner)
                got[name] = [None if value is None else {k: repr(v) for k, v in value.items()}, notes[:1],
                             [o.value for o in keys.origins], keys.opaque]
            print(json.dumps(got))
            """,
        )
    )
    got = json.loads(out.splitlines()[-1])
    # d.setdefault("a", {}) puts "a" (it was read as an empty dict putting nothing); d.setdefault("b", 3) leaves 2
    assert list(got["d"][0]) == ["a", "b"] and "{} for the call" in got["d"][0]["a"] and got["d"][0]["b"] == "2"
    assert got["d"][2] == ["a", "b", "b"] and got["d"][3] == []
    # e["a"].append(1) changes what e["a"] holds, and no step of e names it: not rebuilt as {"a": []}
    assert got["e"][0] is None
    assert got["e"][1] == ["e is also written or passed on at [8] in f, which the walk does not name"]
    assert got["e"][2] == ["a"]
    # the setdefault's statement goes on to append to what "a" holds: the key is known, its value is not
    assert list(got["g"][0]) == ["a"] and "what @10 makes of g['a'] for the call" in got["g"][0]["a"]
    assert got["g"][2] == ["a"]


def test_a_functions_own_handover_binds_the_callees_parameter_and_not_its_own_name(tmp_path):
    out = _ok(
        _run(
            tmp_path,
            """
            import json
            from types import SimpleNamespace as N
            from bamboo.querier import asked as A
            from bamboo.querier import evaluate as E

            owner = "pandaserver/a.py::f"
            step = lambda kind, line, value: N(name="original_name", owner=owner, kind=kind, line=line, value=value,
                                                guards=[], unseen=[], reads=[], terminal="", detail="", depth=1, cls="")
            # f binds its original_name at 2, and at 3 hands file.destinationDBlock to a callee's original_name
            t = E.taken(A.given([step("binding", 2, "'x'"), step("handover", 3, "file.destinationDBlock")]), "original_name",
                        E.Env(), owner=owner)
            print(json.dumps([t.step.line if t.step else None, repr(t.value), [s.line for s in t.kept]]))
            """,
        )
    )
    assert json.loads(out.splitlines()[-1]) == [2, "'x'", [2]]


def test_a_reader_asks_the_questions_it_reads_and_refuses_steps_that_cannot_ask(tmp_path):
    out = _ok(
        _run(
            tmp_path,
            """
            import json
            from bamboo.codemap import trace
            from bamboo.codemap.models import TraceStep
            from bamboo.querier import asked as A
            from bamboo.querier import evaluate as E
            from bamboo.querier import false_guard as G

            f, g = "pandaserver/a.py::f", "pandaserver/a.py::g"
            q = lambda kind, owner, name, line: trace.Question(kind, "pandaserver/a.py", owner, name, line)
            x3, y4, m5 = q("bindings", f, "x", 3), q("bindings", f, "y", 4), q("returns", f, "self.job.m()", 5)
            x7, z9 = q("bindings", f, "x", 7), q("bindings", g, "z", 9)
            step = lambda name, line, kind="binding": TraceStep(kind=kind, name=name, owner=f, line=line, value="1", depth=2)
            answers = {
                x3: trace.Answer([step("x", 2)], "", [x7, z9]),
                # the same binding from another place, a hop deeper, and cut there
                x7: trace.Answer([step("x", 2).model_copy(update={"depth": 3})], "the depth budget ran out", []),
                m5: trace.Answer([step("m()", 6, "return")], "", []),
            }

            class Asker:
                def ask(self, question):
                    return answers[question]

            steps = A.Asked([step("a", 1)], Asker(), [x3, y4, m5], "", "f:1")
            A.made[:] = [steps]
            walk = G.Walk(steps, E.Env(), {}, {})
            got = {"defs": [s.line for s in walk.defs("x", f)], "after_defs": [(a.name, a.line) for a in steps.asked]}
            got["returns"] = [s.line for s in G.returns_of("self.job.m()", walk)]
            got["asked"] = [(a.name, a.line) for a in steps.asked]
            got["steps"] = [(s.name, s.line) for s in steps]
            got["summary"] = A.summary()
            refused = []
            for read in (lambda: G.Walk([step("x", 2)], E.Env(), {}, {}).defs("x", f), lambda: E.taken([step("x", 2)], "x", E.Env(), owner=f)):
                try:
                    read()
                except TypeError as e:
                    refused.append(str(e)[:24])
            got["refused"] = refused
            print(json.dumps(got))
            """,
        )
    )
    got = json.loads(out.splitlines()[-1])
    # x at 3, then x at 7, which x's answer handed back; not y, and not g's z
    assert got["defs"] == [2] and got["after_defs"] == [["x", 3], ["x", 7]]
    assert got["returns"] == [6] and got["asked"] == [["x", 3], ["x", 7], ["self.job.m()", 5]]
    # x at 7 hands back the step x at 3 did: kept once
    assert got["steps"] == [["a", 1], ["x", 2], ["m()", 6]]
    assert got["summary"] == ["walk from f:1: asked 3 question(s), 3 step(s); cut: x in f: the depth budget ran out"]
    assert got["refused"] == ["the steps were not asked"] * 2
