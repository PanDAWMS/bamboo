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
    "asked", "attr_time", "callers", "census_p", "changed", "evaluate", "false_guard", "handlers", "held", "heldcol", "mapdump", "outbound", "parts",
    "printed", "raised", "recorded", "records", "reproduce", "rerun", "rows", "same_object", "value_diff", "which_ran",
    "windows",
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
    # the map lives in dumps only: an entry without MAP_DUMP stops, and a reader with no dump installed stops, opening
    # no graph client -- there is no other map, and reading none would say "nothing writes it"
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


def test_the_dump_stays_as_it_was_read(tmp_path):
    # every reader asks the installed dump through mapdump.code_map, which opens no graph client and stores nothing (a
    # store gave each node an id and a valid_for): with the client refused after the install, readers in turn get the
    # map's answers and leave the dump as it was read
    out = _ok(
        _run(
            tmp_path,
            f"""
            import asyncio, json
            from bamboo.codemap.models import Anchor, Branch, JunctionNode, MapFragment, SubjectNode
            from bamboo.database import graph_database_client
            from bamboo.querier import mapdump, rows

            class Refused:
                def __init__(self, *a, **k):
                    raise AssertionError("a graph client was opened")

            owner, subject, built = "pandajedi/a.py::A.run", "JediTaskSpec.status", "panda-server-source 1.0.2"
            fragment = MapFragment(
                map_id="panda",
                derived_from="source-root:{_corpus(tmp_path)}",
                subjects=[SubjectNode(name=subject, map_id="panda", derived_from=built, spec_class="JediTaskSpec", attribute="status")],
                junctions=[JunctionNode(
                    name=JunctionNode.make_name("panda", subject, owner), map_id="panda", derived_from=built, subject=subject,
                    owner=owner, branches=[Branch(outcome="ready", line=10)],
                    anchor=Anchor(package="pandajedi", file="pandajedi/a.py", line_start=42),
                )],
            )
            with open("map.json", "w") as f:
                f.write(fragment.model_dump_json())
            mapdump.install("map.json")
            graph_database_client.GraphDatabaseClient = Refused
            read = mapdump.fragment().model_dump_json()

            async def reader():
                code_map = mapdump.code_map("the test")
                return [len(await code_map.subjects()), len(await code_map.writers_of(subject)), await code_map.versions()]

            first, second = asyncio.run(reader()), asyncio.run(reader())
            print(json.dumps([first, second, rows._function_prints(owner), mapdump.fragment().model_dump_json() == read]))
            """,
        )
    )
    first, second, prints, unchanged = json.loads(out)
    assert first == second == [1, 1, ["panda-server-source 1.0.2"]]
    assert prints is None  # asked, and the dump holds no prints of the owner
    assert unchanged


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
                E.evaluate("x > 2", env),
                E.evaluate("x > y", env) is E.UNKNOWN,
                env.used,
            ]))
            """,
        )
    )
    equal, by_other_spelling, unbound, ordered, ordered_unbound, used = json.loads(out)
    assert equal is True
    # ``self.job`` and ``job`` are one row once the question binds both spellings to it
    assert by_other_spelling is True
    assert unbound is True
    # an ordering is decided as an equality is, where both sides are held, and stays unknown where one is not
    assert ordered is True
    assert ordered_unbound is True
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


def test_the_evaluator_is_the_maps_and_the_querier_hands_it_the_rows_a_run_read(tmp_path):
    out = _ok(
        _run(
            tmp_path,
            """
            import ast
            import json
            from bamboo.codemap.panda import evaluate as shared
            from bamboo.querier import evaluate as E

            E.PURE["twice"] = lambda x: 2 * x  # PURE not empty: no panda-server import
            env = E.Env(rows={"self.job": "job", "job": "job"}, table={"job": {"maxAttempt": E.Value(3, "row")}})
            copy = E.Env(values=dict(env.values), rows=dict(env.rows), table=env.table)
            solved = E.solve("job.maxAttempt == -1", True, env)
            print(json.dumps([
                E.UNKNOWN is shared.UNKNOWN,
                E.evaluate("self.job.maxAttempt == -1", env),
                shared.decidable(ast.parse("job.maxAttempt == -1", mode="eval").body, set(), env),
                E.evaluate("twice(self.job.maxAttempt) == 6", copy),
                [solved.known, solved.consistent, solved.derived],
            ], default=repr))
            """,
        )
    )
    same, signed, decidable, called, (known, consistent, derived) = json.loads(out)
    assert same is True
    # a signed literal is a value: the row's 3 is not -1 (the querier's own copy left the test unknown)
    assert signed is False
    # a row's field is known before its value is taken, as lookup will read it
    assert decidable is True
    # a copy made with keywords still runs the helpers read as pure
    assert called is True
    # and an outcome the row rules out is inconsistent, with nothing derived (the querier's own copy settled the
    # test True and derived '-1' = 3)
    assert known == {"job.maxAttempt == -1": False}
    assert consistent is False
    assert derived == {}


def test_what_a_local_holds_at_s_star_and_what_holes_decide_are_the_maps_readings(tmp_path):
    out = _ok(
        _run(
            tmp_path,
            """
            import ast
            import json
            import textwrap
            from bamboo.codemap.panda import pathcond
            from bamboo.querier import evaluate as E, printed as P

            E.PURE["twice"] = lambda x: 2 * x  # PURE not empty: no panda-server import
            tree = ast.parse(textwrap.dedent('''
                def f(res, items, given):
                    status = None
                    if res:
                        status, locked = res
                    limit = 20
                    n = -1
                    for item in items:
                        if status == "x" and limit > 3 and n == -1 and given and later:
                            pass
                    later = 1
            '''))
            pathcond.attach_parents(tree)
            func = tree.body[0]
            s_star = func.body[4].body[0]
            names = ["status", "limit", "n", "given", "later", "self.later"]

            def test(text):
                return ast.parse(text, mode="eval").body

            print(json.dumps([
                P.entry_state(func, s_star, set(names)),
                {name: P.unbound_before(func, s_star, name) for name in names},
                [
                    P.covered(test("n == -1"), {"n"}, {}),
                    P.covered(test("status == 'x'"), set(), {"status": "x"}),
                    P.covered(test("twice(n) == -2"), {"n"}, {}),
                    P.covered(test("limit > 3"), {"limit"}, {}),
                    P.covered(test("twice(**kw) == 2"), {"kw"}, {}),
                ],
            ]))
            """,
        )
    )
    entry, unbound, covered = json.loads(out)
    # one literal every way binds last: 20 outside a loop that binds no limit, and a signed -1; the tuple under
    # ``if res:`` binds status again, so None is one of two (the querier's own reading passed over the tuple, and
    # dropped both literals)
    assert entry == {"limit": 20, "n": -1}
    # only the entry reaches later's read; a parameter holds what the call handed; a dotted name is not a local
    assert unbound == {"status": False, "limit": False, "n": False, "given": False, "later": True, "self.later": False}
    # a printed hole or a path's constant decides an equality, a signed literal, a pure helper's value and an
    # ordering; not a call whose ** spread is not read (the evaluator does not decide it)
    assert covered == [True, True, True, True, False]


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


def test_a_tag_that_opens_a_bracket_it_does_not_close_is_one_mark_up_to_the_loggers_close(tmp_path):
    # create_tagged_logger writes f"{method} < {tag} >", and insertNewJob's tag is f"<JediTaskID={...} idPool={...}"
    log = tmp_path / "panda-DBProxy.log"
    log.write_text(
        "2026-01-01 00:00:00,000 panda.log.DBProxy: DEBUG    insertNewJob < <JediTaskID=111 idPool=3 > start\n"
        "2026-01-01 00:00:01,000 panda.log.DBProxy: DEBUG    insertNewJob < <JediTaskID=222 idPool=1 > start\n"
        "2026-01-01 00:00:02,000 panda.log.DBProxy: DEBUG    insertNewJob < <JediTaskID=111 idPool=3 > 12345 -> defined\n"
        "2026-01-01 00:00:03,000 panda.log.DBProxy: DEBUG    checkTaskStatusJEDI <  < jediTaskID=111 > > done\n"
    )
    out = _ok(
        _run(
            tmp_path,
            f"""
            import json
            from bamboo.querier.records import read_log, runs_for

            log = read_log({str(log)!r})
            runs = runs_for(log, "111")
            print(json.dumps([[(r.token, [x.lineno for x in r.records]) for r in runs], log.records[2].message]))
            """,
        )
    )
    runs, message = json.loads(out)
    # each call is its own mark, holding the task's id: the other task's call is no line of the run
    assert runs == [
        ["insertNewJob < <JediTaskID=111 idPool=3 >", [1, 3]],
        ["checkTaskStatusJEDI <  < jediTaskID=111 > >", [4]],
    ]
    # the message's own '>' is not read into the mark
    assert message == "12345 -> defined"


def test_rule_8_wants_the_level_in_the_file_where_a_filter_may_send_it_elsewhere_or_panda_common_is_not_found(tmp_path):
    quiet, loud = tmp_path / "panda-x.log", tmp_path / "panda-y.log"
    quiet.write_text(
        "2026-01-01 00:00:00,000 panda.log.X: DEBUG    <PandaID=12345> start\n"
        "2026-01-01 00:00:01,000 panda.log.X: DEBUG    <PandaID=999> start\n"
    )
    loud.write_text(quiet.read_text() + "2026-01-01 00:00:02,000 panda.log.X: ERROR    <PandaID=999> failed\n")
    out = _ok(
        _run(
            tmp_path,
            f"""
            import importlib.util
            import json
            from bamboo.querier import records
            from bamboo.querier.records import read_log, runs_for

            def prints(path):
                return runs_for(read_log(path), "12345")[0].would_print("ERROR")

            found = importlib.util.find_spec
            importlib.util.find_spec = lambda name, *rest: None if name == "pandacommon" else found(name, *rest)
            missing = records.level_filters()
            not_found = [prints({str(quiet)!r}), prints({str(loud)!r})]
            records.level_filters = lambda: []
            none = prints({str(quiet)!r})
            records.level_filters = lambda: ["p/logger.py:7"]
            one = [prints({str(quiet)!r}), prints({str(loud)!r})]
            print(json.dumps([missing, not_found, none, one]))
            """,
        )
    )
    missing, not_found, none, one = json.loads(out)
    # panda-common not found is not "no filter": its loggers were not read
    assert missing == ["pandacommon: not found, so its loggers were not read"]
    # so a run's DEBUG lines do not say its ERROR reaches this file: some run's ERROR line in the file must
    assert not_found == [False, True]
    # no filter anywhere: the run's own lines say the level was on and the file takes it
    assert none is True
    # a filter found is read as panda-common not found
    assert one == [False, True]


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
                # the why is a record; rendered, as a stop says it
                return [a.window, a.machine, str(a.why)]

            state = {"status": "failed", "time": "2026-09-28 19:41:08", "host": "aipanda093.cern.ch"}
            job = [{"status": "pending", "time": "2026-09-28 08:31:07", "host": "aipanda087.cern.ch"},
                   {"status": "defined", "time": "2026-09-28 08:31:08", "host": "aipanda098.cern.ch"},
                   {"status": "activated", "time": "2026-09-28 08:31:12", "host": "aipanda098.cern.ch"},
                   {"status": "sent", "time": "2026-09-28 10:14:38", "host": "aipanda092.cern.ch"}]
            log = {"as_of": "2026-10-13 12:00:00", "states": [
                {"status": "finishing", "time": "2026-10-13 01:00:00"},
                {"status": "running", "time": "2026-10-13 02:00:00"},
                {"status": "finishing", "time": "2026-10-13 10:00:00"}]}
            print(json.dumps([
                w(W.at_transition(state)),
                w(W.at_transition({"status": "failed", "time": "2026-09-28 19:41:08"})),
                [[s["status"], w(a)] for s, a in W.rows_until(job, "2026-09-28 08:31:12,959")],
                [s["status"] for s, _ in W.rows_until(job, "2026-09-28 08:31:11,999")],
                w(W.stuck(log, "finishing")),
                w(W.stuck({**log, "as_of": "2026-10-13 20:00:00"}, "finishing")),
                w(W.stuck({"states": log["states"]}, "finishing")),
                w(W.stuck(log, "ready")),
            ]))
            """,
        )
    )
    (h2, h2_any, h6, h6_early, stuck, stuck_long, no_as_of, no_row) = json.loads(out)
    # H2: the row's own time, on the machine that wrote it
    assert h2[:2] == [["2026-09-28 19:41:08", "2026-09-28 19:41:08"], "aipanda093.cern.ch"]
    assert h2_any[1] is None and "names no host" in h2_any[2]
    # H6: the job's own rows up to the run's last record, in order, each at its second on its host, as H2's
    assert [status for status, _ in h6] == ["pending", "defined", "activated"]
    assert h6[2][1][:2] == [["2026-09-28 08:31:12", "2026-09-28 08:31:12"], "aipanda098.cern.ch"]
    assert h6[0][1][1] == "aipanda087.cern.ch"
    assert h6_early == ["pending", "defined"]
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
                # the why is a record; rendered, as a stop says it
                return [a.window, a.machine, str(a.why)]

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
    first, rest = W.stuck(r.data["statuslog"], "finishing"), W.rest(r.data["statuslog"], "finishing")
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
    got = {}
    r.logs = N(needs_window=True)
    for name in ["ran-late", "ran-early", "silent", "holes", "unanswered", "another's"]:
        found, missed, holes, searched = r._ask_mover("1", "a.py::mover", [name], first, rest)
        got[name] = [searched.window, r._runs_of(found, "a.py::mover", searched)]
    # a writer that takes a row by its key: the last part alone, a run before it not looked for
    found, missed, holes, searched = r._ask_mover("1", "a.py::mover", ["ran-early"], first, None)
    got["by key"] = [searched.window, r._runs_of(found, "a.py::mover", searched)]
    # a window that starts at the row has no rest
    r.data["statuslog"]["as_of"] = "2026-10-13 05:00:00"
    short = W.stuck(r.data["statuslog"], "finishing")
    got["short"] = [r._ask_mover("1", "a.py::mover", ["silent"], short, W.rest(r.data["statuslog"], "finishing"))[3].window,
                    W.rest(r.data["statuslog"], "finishing")]
    # the door over files reads each file whole: one ask
    r.logs = N(needs_window=False)
    got["files"] = [r._ask_mover("1", "a.py::mover", ["silent"], first, rest)[3].window]
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
        got[name] = held(r._searched_whole(to_hand, first, searched, missed, holes, rest))
    # a window that starts at the row is whole in its last part
    r.data["statuslog"]["as_of"] = "2026-10-13 05:00:00"
    short = W.stuck(r.data["statuslog"], "finishing")
    got["short"] = held(r._searched_whole(["a.log"], short, short, [], {}, W.rest(r.data["statuslog"], "finishing")))
    r.logs = N(needs_window=False)
    got["files"] = held(r._searched_whole(["a.log"], first, first, [], {}, rest))

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
    premises = r._premises("JobSpec.jobStatus", "1", "x.py::logger", [], ["a.log", "b.log"], short, short, [], {}, [], None)
    keys = [sorted(r._key_columns("JobSpec.jobStatus", "1")), sorted(r._key_columns("JobSpec.jobStatus", "2"))]
    print(json.dumps([got, logs, premises, keys], default=str))
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
    from bamboo.querier import records
    from bamboo.querier.recorded import Ask, RecordedLogs
    from bamboo.querier.records import Unrecorded, runs_for

    # rule 8 where no logging filter sends a level elsewhere, pinned rather than read from this machine: without
    # panda-common installed (CI) its loggers are "not found" and the stricter rule holds -- a test of its own
    records.level_filters = lambda: []

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
            # a later round writes the key's ask again, saying it replaces round 1: read instead of it, in either order
            again = {**key_rec, "round": 2, "replaces": 1,
                     "machines": {M1: answered({**TODAY, "records": [hit]}), M2: answered({**TODAY, "records": []})}}
            Path("old.jsonl").write_text(json.dumps(key_rec) + "\\n" + json.dumps(tok_rec) + "\\n")
            Path("new.jsonl").write_text(json.dumps(again) + "\\n")
            Path("two.jsonl").write_text(json.dumps(again) + "\\n" + json.dumps({**again, "round": 3}) + "\\n")
            replaced = [RecordedLogs(paths).log("panda-x.log", "ds_sub1", W).holes for paths in (["old.jsonl", "new.jsonl"],
                                                                                              ["new.jsonl", "old.jsonl"])]
            try:
                RecordedLogs(["old.jsonl", "two.jsonl"])
                two = ""
            except SystemExit as e:
                two = str(e)
            print(json.dumps([round0, round1, round2, [r.span for r in run.records], run.describe(), log.holes, log.notes,
                              run.would_print("ERROR"), twice, replaced, two]))
            """),
        )
    )
    round0, round1, round2, spans, described, holes, notes, would_print, twice, replaced, two = json.loads(out)
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
    # round 2's record of the key's ask, which replaces round 1's: aipanda094 answered there
    assert replaced == [[], []]
    # two records replacing the same round: which to read is not the door's choice
    assert "two.jsonl:2: a second record for the ask" in two


def test_the_record_door_asks_a_mark_that_does_not_hold_the_key_only_of_a_key_record_near_the_window(tmp_path):
    out = _ok(
        _run(
            tmp_path,
            textwrap.dedent(_RECORDED)
            + textwrap.dedent("""
            W = ("2026-09-28 08:21:12", "2026-09-28 08:41:13")
            key_ask = Ask("panda-x.log", ("ds_sub1",), W)
            # the key under five marks: in the window, 8 min past its end, 10 min before its start, 16 min before it,
            # and 3 h after it -- a rotated file's record holds the key's lines of its whole day
            hits = [{**line(t, "registered ds_sub1", f"{tok} < x >"), "offset": i} for i, (t, tok) in enumerate(
                [("08:31:12", "inside"), ("08:49:13", "past"), ("08:11:12", "edge"), ("08:05:00", "before"),
                 ("11:41:00", "after")])]
            _, round1 = ask_of([record(key_ask, {M1: answered({**TODAY, "records": hits})})], "panda-x.log", "ds_sub1", W)
            print(json.dumps(round1))
            """),
        )
    )
    round1 = json.loads(out)
    # the marks of the key records within EDGE (10 min) of the window, its ends held; a run that began hours after the
    # window, or ended before it, is not what it asks about (K13)
    assert [a["strings"] for a, _ in round1] == [["edge < x >", "inside < x >", "past < x >"]]
    assert all(a["window"] == ["2026-09-28 08:21:12", "2026-09-28 08:41:13"] for a, _ in round1)


def test_a_bare_mark_holding_a_blank_is_cut_by_the_maps_marks_of_its_log(tmp_path):
    # LogWrapper(_logger, f"update_job PandaID={job_id} PID={os.getpid()}") (pilot_api.py:513): PREFIX alone reads the
    # mark "update_job", every pilot's call, and the door asks it over the window (K59)
    local = tmp_path / "panda-x.log"
    local.write_text(
        "2026-09-28 08:31:12,000 panda.log.X: DEBUG     #ATM #KV doForPreStaging label=user action=throttle jediTaskID=5\n"
        "2026-09-28 08:31:13,000 panda.log.X: DEBUG    get_file a b\n"
    )
    (tmp_path / "panda-y.log").write_text("2026-09-28 08:31:12,000 panda.log.Y: DEBUG    update_job PandaID=12345 PID=777 Start\n")
    out = _ok(
        _run(
            tmp_path,
            textwrap.dedent(_RECORDED)
            + textwrap.dedent(f"""
            from types import SimpleNamespace as N
            from bamboo.querier import mapdump
            from bamboo.querier.records import read_log

            W = ("2026-09-28 08:21:12", "2026-09-28 08:41:13")
            call = [{{**line(t, f"PandaID=12345 PID=777 {{msg}}", "update_job"), "offset": i}}
                    for i, (t, msg) in enumerate([("08:31:12", "Start"), ("08:31:12", "executing"), ("08:31:13", "Done.")])]
            key_rec = record(Ask("panda-x.log", ("12345",), W), {{M1: answered({{**TODAY, "records": call}})}})
            _, bare = ask_of([key_rec], "panda-x.log", "12345", W)

            def run(line, builder, mark, logs=("panda-x.log",)):
                return N(owner=f"p/x.py::f{{line}}",
                         runs=[N(line=line, at=f"p/x.py::f{{line}}", builder=builder, mark=mark, key=[])],
                         statements=[N(log_files_basis="logger", runs=[line], handed_runs=[], log_files=list(logs), rows=[])])

            mapdump._FRAGMENT[:] = [N(function_prints=[
                run(513, "LogWrapper", "update_job PandaID={{}} PID={{}}"),
                # MsgWrapper's token as handed, a blank first
                run(135, "MsgWrapper", " #ATM #KV doForPreStaging label=user"),
                # a mark with no literal would cut any two words; a tagged logger's PREFIX reads whole
                run(507, "LogWrapper", "{{}} {{}}"),
                run(42, "create_tagged_logger", "get_file {{}}"),
            ])]
            log, cut = ask_of([key_rec], "panda-x.log", "12345", W)
            print(json.dumps([bare, cut, [(r.token, [x.message for x in r.records]) for r in runs_for(log, "12345")],
                              [(r.token, r.message) for r in read_log({str(local)!r}).records],
                              [r.token for r in read_log({str(tmp_path / "panda-y.log")!r}).records]]))
            """),
        )
    )
    bare, cut, runs, local_records, other_log = json.loads(out)
    # the first word: a mark not holding the key, asked over the window
    assert [(a["strings"], why) for a, why in bare] == [(["update_job"], "the records of the key's runs whose mark does not hold the key")]
    # the map's mark: the call is its own run, the key in its mark, and nothing more is asked
    assert cut == []
    assert runs == [["update_job PandaID=12345 PID=777", ["Start", "executing", "Done."]]]
    assert local_records == [["#ATM #KV doForPreStaging label=user", "action=throttle jediTaskID=5"], ["get_file", "a b"]]
    # a mark is read only in the files its run prints into
    assert other_log == ["update_job"]


def test_a_mark_the_calls_share_is_asked_only_near_the_keys_records_joined_per_machine(tmp_path):
    # the key's lines under marks not holding it: two the calls share on M1 (a method's tag with no identity in it, a
    # worker's), one holding a job's id, and a shared one on M2 near the window's end (K60)
    out = _ok(
        _run(
            tmp_path,
            textwrap.dedent(_RECORDED)
            + textwrap.dedent("""
            from types import SimpleNamespace as N
            from bamboo.querier import mapdump

            W = ("2026-09-28 08:00:00", "2026-09-28 12:00:00")
            S1, S2, P = "getTasks_JEDI < vo=None label=None >", "process <h_1-2>", "updateJob < PandaID=777 >"
            at = lambda i, t, tok: {**line(t, "task 12345", tok), "offset": i}
            key_rec = record(Ask("panda-x.log", ("12345",), W), {
                M1: answered({**TODAY, "records": [at(1, "09:00:00", S1), at(2, "09:05:00", S1), at(3, "09:20:00", S2),
                                                   at(4, "10:00:00", P)]}),
                M2: answered({**TODAY, "records": [at(5, "11:50:00", S1)]})})

            def asks():
                _, got = ask_of([key_rec], "panda-x.log", "12345", W)
                return [(a["machine"], a["strings"], a["window"], why.rsplit(", ", 1)[-1]) for a, why in got]

            bare = asks()

            def run(at, builder, mark, columns):
                return N(owner=at, runs=[N(line=1, at=at, builder=builder, mark=mark, key=[N(columns=c) for c in columns])],
                         statements=[N(log_files_basis="logger", runs=[1], handed_runs=[], log_files=["panda-x.log"], rows=[])])

            mapdump._FRAGMENT[:] = [N(tables=[N(identity_attrs=["JobSpec.PandaID"])], function_prints=[
                run("p/x.py::C.getTasks_JEDI", "create_tagged_logger", "vo={} label={}", [["JediTaskSpec.vo"], []]),
                run("p/x.py::C.worker", "MsgWrapper", "process <{}>", [[]]),
                run("p/x.py::C.updateJob", "create_tagged_logger", "PandaID={}", [["JobSpec.PandaID"]]),
            ])]
            print(json.dumps([bare, asks()]))
            """),
        )
    )
    bare, shared = json.loads(out)
    hold = "the records of the key's runs whose mark does not hold the key"
    # with no map every mark is asked over the window, one ask a machine
    assert bare == [["aipanda093.cern.ch", ["getTasks_JEDI < vo=None label=None >", "process <h_1-2>", "updateJob < PandaID=777 >"],
                     ["2026-09-28 08:00:00", "2026-09-28 12:00:00"], hold],
                    ["aipanda094.cern.ch", ["getTasks_JEDI < vo=None label=None >"],
                     ["2026-09-28 08:00:00", "2026-09-28 12:00:00"], hold]]
    # the job's mark over the window; the shared ones within 20 min of the key's records under them, M1's two windows
    # joined (08:40-09:25 and 09:00-09:40), M2's cut at the window's end
    assert shared == [["aipanda093.cern.ch", ["updateJob < PandaID=777 >"], ["2026-09-28 08:00:00", "2026-09-28 12:00:00"], hold],
                      ["aipanda093.cern.ch", ["getTasks_JEDI < vo=None label=None >", "process <h_1-2>"],
                       ["2026-09-28 08:40:00", "2026-09-28 09:40:00"], "near them: a mark the calls share"],
                      ["aipanda094.cern.ch", ["getTasks_JEDI < vo=None label=None >"],
                       ["2026-09-28 11:30:00", "2026-09-28 12:00:00"], "near them: a mark the calls share"]]


def test_the_record_door_answers_an_ask_from_the_record_of_a_wider_one(tmp_path):
    out = _ok(
        _run(
            tmp_path,
            textwrap.dedent(_RECORDED)
            + textwrap.dedent("""
            TOK, W = "2026-09-28/08:31:10.314981", ("2026-09-28 08:21:12", "2026-09-28 08:41:13")
            AT, M3 = ("2026-09-28 08:31:12", "2026-09-28 08:31:12"), "aipanda095.cern.ch"
            hit = {**line("08:31:12", "registered ds_sub1", TOK), "offset": 200}
            key_rec = record(Ask("panda-x.log", ("ds_sub1",), W), {M1: answered({**TODAY, "records": [hit]}),
                                                                  M2: answered({**TODAY, "records": []})})
            tok_rec = record(Ask("panda-x.log", (TOK,), W, M1), {M1: answered({**TODAY, "records": [
                {**line("08:31:10", "start", TOK), "offset": 100}, hit, {**line("08:31:14", "end", TOK), "offset": 300}]})})
            wider = [key_rec, tok_rec]
            log, at_m1 = ask_of(wider, "panda-x.log", "ds_sub1", AT, M1)
            run, = runs_for(log, "ds_sub1")
            nowhere, at_m3 = ask_of(wider, "panda-x.log", "ds_sub1", AT, M3)
            past_log, past = ask_of(wider, "panda-x.log", "ds_sub1", ("2026-09-28 08:31:12", "2026-09-28 08:41:14"), M1)
            _, beyond = ask_of(wider, "panda-x.log", "ds_sub1", ("2026-09-29 06:45:00", "2026-09-29 07:00:00"), M1)
            _, other = ask_of(wider, "panda-x.log", "ds_sub2", AT, M1)
            own = record(Ask("panda-x.log", ("ds_sub1",), AT, M1), {M1: answered({**TODAY, "records": []})})
            mine, _ = ask_of(wider + [own], "panda-x.log", "ds_sub1", AT, M1)
            print(json.dumps([at_m1, [r.span for r in run.records], log.holes, log.notes, at_m3, [r.span for r in nowhere.records],
                              nowhere.notes, past, past_log.holes, past_log.notes, beyond, other, mine.records, mine.notes]))
            """),
        )
    )
    (at_m1, spans, holes, notes, at_m3, nowhere, nowhere_notes, past, past_holes, past_notes, beyond, other, mine,
     mine_notes) = json.loads(out)
    # an ask inside a recorded window, on a machine the every-machine record answered: that machine's part of it, and
    # the mark's ask from the mark's record on the machine -- each said
    assert at_m1 == []
    assert spans == ["aipanda093:panda-x.log-20260929.gz@100", "aipanda093:panda-x.log-20260929.gz@200",
                     "aipanda093:panda-x.log-20260929.gz@300"]
    assert holes == []
    assert notes == [
        "answered from the record of a wider ask (2026-09-28 08:21:12 .. 2026-09-28 08:41:13, every machine, round 1)",
        "answered from the record of a wider ask (2026-09-28 08:21:12 .. 2026-09-28 08:41:13, aipanda093.cern.ch, round 1)",
    ]
    # a machine the every-machine record does not name is not in list_logs: the run is looked for on every machine, as
    # where a record of the machine says so -- H6's row host need not hold the log its run wrote
    assert at_m3 == [] and nowhere == spans
    assert "aipanda095.cern.ch is not in list_logs, so every machine was asked" in nowhere_notes
    # a window past the recorded one, inside the file the record read whole: answered, and said so -- the recording step
    # reads a rotated file whole whatever window asked (the harness's Ja, 2026-11-04); past every file it read, or other
    # strings: not answered
    assert past == [] and past_holes == []
    assert past_notes[0] == ("answered from the record of another ask (2026-09-28 08:21:12 .. 2026-09-28 08:41:13, every "
                             "machine, round 1)")
    assert beyond[0][0]["window"] == ["2026-09-29 06:45:00", "2026-09-29 07:00:00"]
    assert other[0][0]["strings"] == ["ds_sub2"]
    # an ask's own record comes first
    assert mine == [] and mine_notes == []


def test_the_record_door_answers_a_narrower_ask_from_a_wider_record_by_the_lines_holding_its_strings(tmp_path):
    out = _ok(
        _run(
            tmp_path,
            textwrap.dedent(_RECORDED)
            + textwrap.dedent("""
            W, AT = ("2026-09-28 08:21:12", "2026-09-28 08:41:13"), ("2026-09-28 08:31:10", "2026-09-28 08:31:15")
            A, B = "insertNewJob < <JediTaskID=111 idPool=3 >", "insertNewJob < <JediTaskID=222 idPool=1 >"
            TOK = "getJobs < SITE 2026-09-28T08:31:00.000000 >"
            hit = {**line("08:31:12", "12345 -> defined", A), "offset": 150}
            # an earlier round asked the bare method (its mark read so): every insertNewJob line, cut at 1 MB at 08:31:20,
            # its record saying every string was read to there; then the same from before ``searched``
            records = [{**line("08:31:11", "start", A), "offset": 100}, hit, {**line("08:31:13", "start", B), "offset": 200},
                       {**line("08:31:14", "got 3", TOK), "offset": 300}, {**line("08:31:20", "end", B), "offset": 400},
                       {"offset": 500, "head": False, "cut": False, "lines": [f"  {A} in a traceback"]}]
            key_rec = record(Ask("panda-x.log", ("12345",), W), {M1: answered({**TODAY, "records": [hit]})})
            got = []
            for more in ({"searched": [["2026-09-28 06:40:00", "2026-09-28 08:31:20"]]}, {}):
                wide = record(Ask("panda-x.log", ("insertNewJob", TOK), W, M1), {M1: answered(
                    {**TODAY, "truncated": True, **more, "records": records})})
                log, asks = ask_of([key_rec, wide], "panda-x.log", "12345", AT, M1)
                got.append(asks if log is None else [asks, [r.span for r in log.records], runs_for(log, "12345")[0].token,
                                                     log.holes, log.notes])
            print(json.dumps(got))
            """),
        )
    )
    (asks, spans, token, holes, notes), before = json.loads(out)
    # the mark with its tag is asked of no new round: the bare method's record holds every line holding it
    assert asks == []
    assert token == "insertNewJob < <JediTaskID=111 idPool=3 >"
    # only the lines holding the ask's strings: not the other task's call, nor the other mark asked with it
    assert spans == ["aipanda093:panda-x.log-20260929.gz@100", "aipanda093:panda-x.log-20260929.gz@150"]
    # the cut file is read as far as the whole record read it (08:31:20), not to the last line kept
    assert holes == []
    assert notes == [
        "answered from the record of a wider ask (2026-09-28 08:21:12 .. 2026-09-28 08:41:13, every machine, round 1)",
        "answered from the record of a wider ask (2026-09-28 08:21:12 .. 2026-09-28 08:41:13, aipanda093.cern.ch, round 1, "
        "2 string(s), the lines holding the ask's)",
        "1 fragment(s) of records whose head was not returned",
    ]
    # a cut record that does not say where its strings were read answers no narrower ask: another string's records may
    # carry past a cut one's last (the recording step's doc, 157 of pin m's 315 cut files)
    assert [a["strings"] for a, _ in before] == [["insertNewJob < <JediTaskID=111 idPool=3 >"]]


def test_a_wider_records_lines_are_kept_for_a_narrower_ask_by_its_strings_as_written(tmp_path):
    out = _ok(
        _run(
            tmp_path,
            textwrap.dedent(_RECORDED)
            + textwrap.dedent("""
            from bamboo.querier.recorded import _Gathered

            A = "getJobs < SITE 2026-09-28T08:31:00.000000 (x) >"
            wide = {"machines": {M1: answered({**TODAY, "records": [
                {**line("08:31:11", "start", A), "offset": 100},
                # the dot and the brackets as written: a pattern of them would take this line too
                {**line("08:31:12", "start", "getJobs < SITE 2026-09-28T08:31:00X000000 x >"), "offset": 200},
                {**line("08:31:20", "end", "other < mark >"), "offset": 300}]})}}
            kept = []
            # the ask's string, none (an ask of no string holds no line), and no narrower ask (every line)
            for only in ([A], [], None):
                got = _Gathered()
                got.add({**wide, **({} if only is None else {"only": only, "needed": None})})
                kept.append(sorted(offset for _, _, offset in got.records))
            print(json.dumps(kept))
            """),
        )
    )
    assert json.loads(out) == [[100], [], [100, 200, 300]]


def test_the_record_door_asks_the_strings_handed_in_place_of_the_key_and_reads_no_other_mark_naming_it(tmp_path):
    out = _ok(
        _run(
            tmp_path,
            textwrap.dedent(_RECORDED)
            + textwrap.dedent("""
            W, AT = ("2026-09-28 08:21:12", "2026-09-28 08:41:13"), ("2026-09-28 08:31:10", "2026-09-28 08:31:15")
            OWN, OTHER = "changeTaskSplitRulePanda < jediTaskID=53 >", "getTasksToBeProcessed_JEDI < vo=atlas >"
            MARK = ("changeTaskSplitRulePanda < jediTaskID=53 >",)
            got = {}
            # nothing recorded: the ask is for the strings handed, not the key
            Path("none.jsonl").write_text("")
            try:
                RecordedLogs(["none.jsonl"]).log("panda-x.log", "53", W, strings=MARK)
            except Unrecorded as e:
                got["asked"] = [a.canonical() for a, _ in e.asks]
            # a record of the key alone: its own run's lines, and another call listing the task under its mark
            bare = record(Ask("panda-x.log", ("53",), W), {M1: answered({**TODAY, "records": [
                {**line("08:31:11", "changing splitRule=x", OWN), "offset": 100},
                {**line("08:31:12", "jediTaskID=53 status=running", OTHER), "offset": 200}]})})
            Path("bare.jsonl").write_text(json.dumps(bare) + "\\n")
            log = RecordedLogs(["bare.jsonl"]).log("panda-x.log", "53", AT, strings=MARK)
            got["marked"] = [[r.token for r in runs_for(log, "53")], [r.span for r in log.records]]
            # the key alone over the same window: the other mark naming it is asked whole, a round later
            try:
                RecordedLogs(["bare.jsonl"]).log("panda-x.log", "53", AT)
            except Unrecorded as e:
                got["bare"] = [a.canonical()["strings"] for a, _ in e.asks]
            # a run of the mark near the file's start: the window is widened past it for the same strings
            EDGE_W = ("2026-09-28 06:45:00", "2026-09-28 06:45:00")
            edge = record(Ask("panda-x.log", MARK, EDGE_W, M1), {M1: answered({**TODAY, "records": [
                {**line("06:42:00", "start", OWN), "offset": 10}, {**line("06:46:00", "end", OWN), "offset": 20}]})})
            Path("edge.jsonl").write_text(json.dumps(edge) + "\\n")
            try:
                RecordedLogs(["edge.jsonl"]).log("panda-x.log", "53", EDGE_W, machine=M1, strings=MARK)
            except Unrecorded as e:
                got["widened"] = [[a.canonical()["strings"], a.canonical()["window"]] for a, _ in e.asks]
            print(json.dumps(got))
            """),
        )
    )
    got = json.loads(out)
    assert got["asked"] == [{"name": "panda-x.log", "service": "unsettled", "machine": None, "strings": [
        "changeTaskSplitRulePanda < jediTaskID=53 >"], "window": ["2026-09-28 08:21:12", "2026-09-28 08:41:13"], "whole_records": True}]
    # the mark read from the key's record by the lines holding it: one run, its own, and no other mark asked
    assert got["marked"] == [["changeTaskSplitRulePanda < jediTaskID=53 >"], ["aipanda093:panda-x.log-20260929.gz@100"]]
    assert got["bare"] == [["getTasksToBeProcessed_JEDI < vo=atlas >"]]
    assert got["widened"] == [[["changeTaskSplitRulePanda < jediTaskID=53 >"], ["2026-09-28 06:39:59", "2026-09-28 06:45:00"]]]


def test_the_record_door_asks_a_narrower_ask_anew_where_the_wider_record_was_cut_over_its_window(tmp_path):
    out = _ok(
        _run(
            tmp_path,
            textwrap.dedent(_RECORDED)
            + textwrap.dedent("""
            W, AT = ("2026-09-28 08:21:12", "2026-09-28 08:41:13"), ("2026-09-28 08:31:10", "2026-09-28 08:31:15")
            A = "insertNewJob < <JediTaskID=111 idPool=3 >"
            hit = {**line("08:31:12", "12345 -> defined", A), "offset": 150}
            key_rec = record(Ask("panda-x.log", ("12345",), W), {M1: answered({**TODAY, "records": [hit]})})
            got = []
            # the bare method's record cut at 1 MB at 08:31:12, inside the window; then the same with the record saying
            # every string was read to 08:35:00; then the tagged mark's own record, cut alike
            for ask, more in ((("insertNewJob",), {}), (("insertNewJob",), {"searched": [["2026-09-28 06:40:00", "2026-09-28 08:35:00"]]}),
                              ((A,), {})):
                cut = record(Ask("panda-x.log", ask, W), {M1: answered({**TODAY, "truncated": True, **more, "records": [
                    {**line("08:31:11", "start", A), "offset": 100}, hit]})})
                log, asks = ask_of([key_rec, cut], "panda-x.log", "12345", AT, M1)
                got.append([[a["strings"] for a, _ in asks], log.holes if log else None,
                            [r.describe() for r in runs_for(log, "12345") if r.token == A] if log else None])
            print(json.dumps(got))
            """),
        )
    )
    cut, searched, own = json.loads(out)
    # an ask of the tagged mark reads fewer lines a second than the bare method's cut one: asked anew
    assert cut == [[["insertNewJob < <JediTaskID=111 idPool=3 >"]], None, None]
    # a cut record that read every string through the window answers it, as does a record of the ask's own strings,
    # cut or not: what it left is the recording step's to read on (--again)
    assert searched[:2] == [[], []] and own[:2] == [[], []]
    # the key's search is whole: the mark's cut file is a hole of the mark's run alone, read as cut where it stops
    run, = own[2]
    assert run.endswith("ends at 2026-09-28 08:31:12: the answer of panda-x.log-20260929.gz was cut at 1 MB after its last record)")


def test_the_record_door_reads_a_cut_file_for_the_strings_an_ask_needs_where_it_says_what_was_read_for_each(tmp_path):
    out = _ok(
        _run(
            tmp_path,
            textwrap.dedent(_RECORDED)
            + textwrap.dedent("""
            W, AT = ("2026-09-28 08:21:12", "2026-09-28 08:41:13"), ("2026-09-28 08:31:10", "2026-09-28 08:31:15")
            A, B = "getX < 2026-09-28/08:31:10.1 >", "getX < 2026-09-28/08:31:13.0 >"
            hit = {**line("08:31:12", "12345 -> defined", A), "offset": 150}
            key_rec = record(Ask("panda-x.log", ("12345",), W), {M1: answered({**TODAY, "records": [hit]})})
            stop = [["2026-09-28 06:40:00", "2026-09-28 08:31:12"]]
            whole = [["2026-09-28 06:40:00", "2026-09-29 06:50:00"]]
            got = []
            # the two calls' marks asked together, cut where B's call filled the second after 08:31:12; then A's call
            # filling it; then a record of before the field
            for read in ({B: stop}, {A: stop, B: whole}, None):
                entry = {**TODAY, "truncated": True, "searched": stop, "records": [{**line("08:31:11", "start", A), "offset": 100}, hit]}
                if read is not None:
                    entry["read"] = read
                cut = record(Ask("panda-x.log", (A, B), W), {M1: answered(entry)})
                log, asks = ask_of([key_rec, cut], "panda-x.log", "12345", AT, M1)
                got.append([[a["strings"] for a, _ in asks], log.holes if log else None])
            print(json.dumps(got))
            """),
        )
    )
    other, own, before = json.loads(out)
    # the ask of A's mark alone needs A's lines, read whole: answered from the cut record, with no hole
    assert other == [[], []]
    # A's own string was cut: the recording step reads on from where it stopped (--again), inside the record's window +-
    # EDGE, and an ask of A would stop there too -- not asked, a hole of A's run; the key's search is whole
    assert own == [[], []]
    # a record that does not say which string was cut: A's lines were read only where every string was, and an ask of A
    # alone reads fewer lines a second -- asked anew
    assert before == [[["getX < 2026-09-28/08:31:10.1 >"]], None]


def test_the_record_door_answers_from_other_asks_records_only_what_a_new_ask_would_not_read_further(tmp_path):
    out = _ok(
        _run(
            tmp_path,
            textwrap.dedent(_RECORDED)
            + textwrap.dedent("""
            W, A = ("2026-09-28 08:21:12", "2026-09-28 08:41:13"), "getX < 2026-09-28/08:31:10.1 >"

            def asked(records, window, machine=None, strings=()):
                path = Path(f"r{len(list(Path('.').glob('r*.jsonl')))}.jsonl")
                path.write_text("".join(json.dumps(r) + "\\n" for r in records))
                try:
                    RecordedLogs([str(path)]).log("panda-x.log", "12345", window, machine=machine, strings=strings)
                    return []
                except Unrecorded as e:
                    return [[a.canonical()["machine"], a.canonical()["window"]] for a, _ in e.asks]

            got = {}
            # a record of one machine says nothing of the others; one of every machine names them
            one = record(Ask("panda-x.log", ("12345",), W, M1), {M1: answered({**TODAY, "records": []})})
            every = record(Ask("panda-x.log", ("12345",), W), {M1: answered({**TODAY, "records": []}),
                                                             M2: answered({**TODAY, "records": []})})
            AT = ("2026-09-28 09:00:00", "2026-09-28 09:05:00")
            got["one machine"] = [asked([one], AT), asked([one], AT, M1), asked([every], AT)]
            # a machine its own records say holds no file of the log holds none, with no record of every machine
            gone = record(Ask("panda-x.log", ("12345",), W, M2), {M2: {"service": "server", "status": "no such log", "files": []}})
            got["one machine"].append(asked([gone], AT, M2))
            # 06:42 lies in today's file's first 10 min, which the older file's margin covers too: a record that does
            # not list the older file does not answer it; one that read it does
            EARLY = ("2026-09-28 06:42:00", "2026-09-28 06:43:00")
            both = record(Ask("panda-x.log", ("12345",), ("2026-09-28 06:45:00", "2026-09-28 06:45:00"), M1),
                          {M1: answered({**BEFORE, "records": []}, {**TODAY, "records": []})})
            got["older file"] = [asked([one], EARLY, M1), asked([one, both], EARLY, M1)]
            # A's own string cut at 08:31:12: inside its record's window +- 10 min the recording step reads on; past it,
            # a new ask reads on from the stop
            cut = record(Ask("panda-x.log", (A,), W, M1), {M1: answered({**TODAY, "truncated": True, "read": {
                A: [["2026-09-28 06:40:00", "2026-09-28 08:31:12"]]}, "records": []})})
            got["cut"] = [asked([cut], ("2026-09-28 08:31:00", "2026-09-28 08:50:00"), M1, (A,)),
                          asked([cut], ("2026-09-28 10:00:00", "2026-09-28 10:00:30"), M1, (A,))]
            # today's file read to when it was asked, and tomorrow's .gz of it another ask read for another string: one
            # file, read for the key up to then
            current = record(Ask("panda-x.log", ("12345",), ("2026-09-28 08:50:00", "2026-09-28 08:55:00"), M1), {M1: answered(
                {**TODAY, "file": "panda-x.log", "covers": ["2026-09-28 06:40:00", "2026-09-28 09:00:00"], "margin_s": 0,
                 "covers_to": "asked", "records": []})})
            # (its start read off the file's time, a few seconds apart between rounds)
            other = record(Ask("panda-x.log", ("PandaID=99",), ("2026-09-29 01:00:00", "2026-09-29 01:00:00"), M1),
                           {M1: answered({**TODAY, "covers": ["2026-09-28 06:40:03", "2026-09-29 06:50:00"], "records": []})})
            got["one file"] = [asked([current, other], ("2026-09-28 08:00:00", "2026-09-28 08:30:00"), M1),
                               asked([current, other], ("2026-09-28 09:30:00", "2026-09-28 09:40:00"), M1)]
            # one rotated file two rounds listed from starts 3 s apart, one of them reading the key
            jitter = record(Ask("panda-x.log", ("12345",), ("2026-09-28 09:00:00", "2026-09-28 09:00:00"), M1),
                            {M1: answered({**TODAY, "records": []})})
            got["one file"].append(asked([jitter, other], ("2026-09-28 10:00:00", "2026-09-28 10:00:00"), M1))
            print(json.dumps(got))
            """),
        )
    )
    got = json.loads(out)
    W_AT = ["2026-09-28 09:00:00", "2026-09-28 09:05:00"]
    assert got["one machine"] == [[[None, W_AT]], [], [], []]
    assert got["older file"] == [[["aipanda093.cern.ch", ["2026-09-28 06:42:00", "2026-09-28 06:43:00"]]], []]
    assert got["cut"] == [[], [["aipanda093.cern.ch", ["2026-09-28 10:00:00", "2026-09-28 10:00:30"]]]]
    assert got["one file"] == [[], [["aipanda093.cern.ch", ["2026-09-28 09:30:00", "2026-09-28 09:40:00"]]], []]


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
            for label, owners, whole in (("s1", {"getX": (), "upd": ()}, False), ("any", None, False),
                                         ("h2", {"getX": ("start",), "upd": ()}, True)):
                try:
                    RecordedLogs(["k.jsonl"]).log("panda-x.log", K, W, owners=owners, whole=whole)
                except Unrecorded as e:
                    asked[label] = [a.canonical()["strings"] for a, _ in e.asks]
            tok_ask = Ask("panda-x.log", tuple(sorted([OWN, WORKER])), W, M1)
            tok_rec = record(tok_ask, {M1: answered({**TODAY, "records": [hits[1], {**line("08:31:13", "done", OWN), "offset": 21},
                                                                           hits[3], {**line("08:35:01", "end", WORKER), "offset": 41}]})})
            Path("t.jsonl").write_text(json.dumps(tok_rec) + "\\n")
            log = RecordedLogs(["k.jsonl", "t.jsonl"]).log("panda-x.log", K, W, owners={"getX": (), "upd": ()})
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
    # H2 at a row's moment: no mark of a method none of its candidates is, and each candidate's whole -- getX's though
    # its calls share it, and not for its own rows' heads
    assert asked["h2"] == [["STAMP1", "getX < vo=None label=None >", "upd < 2026-09-28/08:31:10.1 >"]]
    assert runs["upd < 2026-09-28/08:31:10.1 >"][:2] == [2, True] and runs["STAMP1"][:2] == [2, True]
    assert runs["chk <  < jediTaskID=52977211 > >"][:2] == [1, True]
    # a mark not asked whole: its key records alone, read as cut -- no row missing from it is read as not printed
    assert runs["getX < vo=None label=None >"][:2] == [2, False] and "a mark the method's calls share" in runs["getX < vo=None label=None >"][2]
    assert runs["push < 2026-09-28/08:33:00.0 >"][:2] == [1, False] and "a method no step asked reads" in runs["push < 2026-09-28/08:33:00.0 >"][2]
    assert notes == ["2 mark(s) read from the key's records alone, as cut: a mark the method's calls share; a method no step asked reads"]


def test_the_record_door_asks_an_s1_method_mark_for_its_own_rows_and_a_run_of_it_holds_no_other_functions_row(tmp_path):
    out = _ok(
        _run(
            tmp_path,
            textwrap.dedent(_RECORDED)
            + textwrap.dedent("""
            K, W, TOK = "52410322", ("2026-09-28 08:21:12", "2026-09-28 08:41:13"), "getX < vo=atlas pid=1 >"
            hit = {**line("08:31:12", f"getting jediTaskID={K}", TOK), "offset": 20}
            warn = {"head": True, "cut": False, "offset": 40,
                    "lines": [f"2026-09-28 08:31:15,000 panda.log.X: WARNING  {TOK} too many datasets"]}
            # the call's lines: its own rows, a callee's line for another task written with its logger, a warning
            whole = [{**line("08:31:10", "start label=managed", TOK), "offset": 10}, hit,
                     {**line("08:31:13", "jediTaskID=1 datasetID=2 tmpNumFiles=3", TOK), "offset": 30}, warn,
                     {**line("08:31:16", "returning 1 tasks", TOK), "offset": 50}]
            Path("k.jsonl").write_text(json.dumps(record(Ask("panda-x.log", (K,), W), {M1: answered({**TODAY, "records": [hit]})})) + "\\n")
            Path("t.jsonl").write_text(json.dumps(record(Ask("panda-x.log", (TOK,), W, M1), {M1: answered({**TODAY, "records": whole})})) + "\\n")
            got = {}
            for label, heads in (("narrowed", ("getting jediTaskID=", "returning ", "start label=")), ("whole", ())):
                try:
                    RecordedLogs(["k.jsonl"]).log("panda-x.log", K, W, owners={"getX": heads})
                except Unrecorded as e:
                    asked = [list(a.canonical()["strings"]) for a, _ in e.asks]
                log = RecordedLogs(["k.jsonl", "t.jsonl"]).log("panda-x.log", K, W, owners={"getX": heads})
                run, = runs_for(log, K)
                got[label] = [asked, [r.message for r in run.records],
                              [run.would_print(level, func) for level, func in (("DEBUG", "getX"), ("DEBUG", "f"), ("DEBUG", None), ("WARNING", "f"))],
                              run.describe(), log.notes]
            print(json.dumps(got))
            """),
        )
    )
    got = json.loads(out)
    asked, messages, would, described, notes = got["narrowed"]
    # the mark followed by each head of the method's own rows, and the mark after each level from WARNING up
    assert sorted(asked[0]) == sorted([
        "getX < vo=atlas pid=1 > getting jediTaskID=", "getX < vo=atlas pid=1 > returning ", "getX < vo=atlas pid=1 > start label=",
        "WARNING  getX < vo=atlas pid=1 >", "ERROR    getX < vo=atlas pid=1 >", "CRITICAL getX < vo=atlas pid=1 >"]) and len(asked) == 1
    # answered from the whole mark's record by the lines holding them: not the callee's line for another task
    assert messages == ["start label=managed", "getting jediTaskID=52410322", "too many datasets", "returning 1 tasks"]
    # a row of the method missing from the run is evidence; another function's below WARNING, or a row not said whose,
    # is not searched; at WARNING and above every line was asked
    assert would == [True, False, False, True]
    assert "asked for getX's rows and WARNING and above" in described
    assert notes[-1] == "1 mark(s) asked for their method's own rows and every line at WARNING and above: ['getX']"
    # a method whose rows do not all start with fixed text: its mark asked whole, as before
    asked, messages, would, described, notes = got["whole"]
    assert asked == [["getX < vo=atlas pid=1 >"]] and len(messages) == 5 and would == [True, True, True, True]
    assert "asked for" not in described


_HEADS = """
def h(self, tmp_log, x):
    tmp_log.debug("start")
    tmp_log.info(f"got {x} rows")
    tmp_log.error(f"{x} failed")
    if x:
        msg = "done"
    else:
        msg = "skipped"
    tmp_log.debug(msg)


def k(self, tmp_log, x):
    tmp_log.debug("start")
    tmp_log.debug(f"{x} rows")


def c(self, tmp_log, x):
    tmp_log.debug("start")
    tmp_log.debug("done" if x else "skipped")
"""


def test_a_methods_mark_heads_are_its_own_rows_below_warning_and_none_where_one_starts_with_a_hole(tmp_path):
    (_corpus(tmp_path) / "pandaserver" / "a.py").write_text(_HEADS)
    out = _ok(
        _run(
            tmp_path,
            """
            import json
            from bamboo.querier import rows as R

            R.ROWS_FROM = "walk"
            print(json.dumps([R.mark_heads("pandaserver/a.py", f) for f in ("h", "k", "c")]))
            """,
        )
    )
    h, k, c = json.loads(out.splitlines()[-1])
    # each shape's leading text, of every row below WARNING (a message bound in two branches: both); the error row is
    # asked by its level, whatever it starts with
    assert h == ["done", "got ", "skipped", "start"]
    # a row that may start with anything -- a leading hole, an expression read as one -- and the mark is asked whole
    assert k == [] and c == []


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


def test_the_record_door_widens_a_run_of_the_keys_own_mark_once_and_a_shared_marks_run_never(tmp_path):
    # a periodic method's mark, a pass every few minutes: its records join into one run as far as any record reaches,
    # and it was asked 10 min earlier each round (the harness's round 18: rescueUnLockedTasksWithPicked_JEDI's mark);
    # a mark that does not hold the key is shared, and the pass that decides sits within a second of its row
    out = _ok(
        _run(
            tmp_path,
            textwrap.dedent(_RECORDED)
            + textwrap.dedent("""
            SHARED, OWN = "getX < vo=atlas label=user >", "PandaID=12345"
            W, WIDER = ("2026-09-28 06:45:00", "2026-09-28 06:45:00"), ("2026-09-28 06:39:59", "2026-09-28 06:45:00")
            STAMPS = ("06:41:00", "06:42:00", "06:44:00", "06:46:00")
            shared = [{**line(t, "rescued 12345" if t == "06:42:00" else "rescued 1", SHARED), "offset": i} for i, t in enumerate(STAMPS)]
            first = record(Ask("panda-x.log", ("12345",), W, M1), {M1: answered({**TODAY, "records": shared[1:2]})})
            mark = record(Ask("panda-x.log", (SHARED,), W, M1), {M1: answered({**TODAY, "records": shared})})
            log, never = ask_of([first, mark], "panda-x.log", "12345", W, M1)
            cut = [runs_for(log, "12345")[0].describe(), log.notes]
            own = [{**line(t, "rescued", OWN), "offset": i} for i, t in enumerate(STAMPS)]
            older = [{**line(t, "rescued", OWN), "offset": 100 + i} for i, t in enumerate(("06:30:20", "06:35:00", "06:39:30"))]
            _, once = ask_of([record(Ask("panda-x.log", ("12345",), W, M1), {M1: answered({**TODAY, "records": own})})],
                             "panda-x.log", "12345", W, M1)
            # the widened ask's record alone: the older file cut at 1 MB, read EDGE before the widened window
            widened = record(Ask("panda-x.log", ("12345",), WIDER, M1), {M1: answered(
                {**BEFORE, "truncated": True, "searched": [["2026-09-28 06:29:59", "2026-09-28 06:50:00"]], "records": older},
                {**TODAY, "records": own})})
            log, again = ask_of([widened], "panda-x.log", "12345", W, M1)
            run, = runs_for(log, "12345")
            print(json.dumps([never, cut, once, again, [r.stamp for r in run.records][:2], run.describe()]))
            """),
        )
    )
    never, (cut, notes), once, again, stamps, described = json.loads(out)
    # the shared mark's run lies near the file's start: not widened, read as cut there, and said
    assert never == [] and "read as cut: what was searched on aipanda093 begins at 2026-09-28 06:40:00" in cut
    assert ("1 run(s) under a mark that does not hold the key lie within 10 min of an end of what was searched: a shared "
            "mark, not widened, read as cut there") in notes
    # the key's own mark, the same run: widened past the file's start
    assert [(a["strings"], a["window"]) for a, _ in once] == [(["12345"], ["2026-09-28 06:39:59", "2026-09-28 06:45:00"])]
    # the key's ask answered from the widened record alone: the run reaches past what the window's own asks searched,
    # so it was widened once, and is read as cut at the widened record's start rather than asked again
    assert again == [] and stamps == ["2026-09-28 06:30:20,000", "2026-09-28 06:35:00,000"]
    assert "read as cut: what was searched on aipanda093 begins at 2026-09-28 06:29:59" in described


def test_the_record_door_widens_a_second_where_the_recording_step_reads_a_cut_string_edge_past_the_window(tmp_path):
    # one rule in two halves, two modules (the recording step imports no querier module): the door asks a second past
    # an end of what was searched, and the recording step asks a cut string again EDGE past each end of the window
    out = _ok(
        _run(
            tmp_path,
            """
            from bamboo.querier import recorded
            from bamboo.scripts.panda.fieldtest import record
            print(record.CONTINUE_SLACK == recorded.EDGE, recorded.EDGE)
            """,
        )
    )
    assert out.split()[0] == "True"


def test_a_run_the_record_door_may_have_cut_says_which_end_of_the_search_it_is_near_and_why(tmp_path):
    out = _ok(
        _run(
            tmp_path,
            textwrap.dedent(_RECORDED)
            + textwrap.dedent("""
            TOK, LATE, EARLY = "PandaID=12345", ("2026-09-28 08:21:12", "2026-09-28 08:41:13"), ("2026-09-28 06:30:00", "2026-09-28 06:45:00")
            run = [{**line("08:31:10", "start 12345", TOK), "offset": 10}, {**line("08:31:14", "end", TOK), "offset": 20}]
            early = [{**line("06:42:00", "start 12345", TOK), "offset": 10}, {**line("06:46:00", "end", TOK), "offset": 20}]
            near = [{**line("08:40:10", "start 12345", TOK), "offset": 10}, {**line("08:50:50", "end", TOK), "offset": 20}]
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
                described(LATE, {**TODAY, "truncated": True, "searched": [["2026-09-28 06:40:00", "2026-09-28 08:31:20"]], "records": run}),
                described(LATE, {**TODAY, "margin_s": 0, "covers": ["2026-09-28 06:40:00", "2026-09-28 08:35:00"], "records": run}),
                described(EARLY, oldest, {**TODAY, "records": early}),
                runs_for(log, "12345")[0].describe(),
                described(LATE, {**TODAY, "records": run}),
                # today's file read over the asks' windows (covers_to): a run near where the read stopped is asked
                # past it; one near the time the file was asked is read as cut
                described(LATE, {**TODAY, "margin_s": 0, "covers_to": "end_stamp", "covers": ["2026-09-28 06:40:00", "2026-09-28 08:51:00"],
                                 "records": near}),
                described(LATE, {**TODAY, "margin_s": 0, "covers_to": "asked", "covers": ["2026-09-28 06:40:00", "2026-09-28 08:51:00"],
                                 "records": near}),
            ]))
            """),
        )
    )
    cut, cut_searched, asked_then, oldest_first, nothing_before, whole, stopped, asked_now = json.loads(out)
    assert [[{k: a[k] for k in ("name", "machine", "strings", "window")}, why] for a, why in stopped] == [[
        {"name": "panda-x.log", "machine": "aipanda093.cern.ch", "strings": ["12345"],
         "window": ["2026-09-28 08:21:12", "2026-09-28 08:51:00"]}, "a run within 10 min of an end of what was searched"]]
    assert asked_now.endswith("ends at 2026-09-28 08:50:59: panda-x.log-20260929.gz was read to the time it was asked)")
    assert cut.endswith("(key on 2; read as cut: what was searched on aipanda093 ends at 2026-09-28 08:31:14: the answer of "
                        "panda-x.log-20260929.gz was cut at 1 MB after its last record)")
    assert cut_searched.endswith("(key on 2; read as cut: what was searched on aipanda093 ends at 2026-09-28 08:31:20: the "
                                 "answer of panda-x.log-20260929.gz was cut at 1 MB, and its record says every string was read "
                                 "only so far)")
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
                # truncated, its record saying where every string was read: only there, whatever its last record
                holes(("2026-09-28 07:00:00", "2026-09-28 07:30:00"), {M1: answered({**TODAY, "truncated": True, "searched": [
                    ["2026-09-28 06:40:00", "2026-09-28 07:09:59"], ["2026-09-28 07:15:00", "2026-09-28 07:20:00"]], "records": [
                    {**line("07:25:00", "12345", "PandaID=12345"), "offset": 1}]})}),
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
    failed, both, rotation, truncated, searched, today, oldest, machines, nowhere = json.loads(out)
    assert failed == [["aipanda093.cern.ch: 2026-09-28 06:45:00 .. 2026-09-28 06:45:00"], []]
    assert both == [[], []]
    # 06:40:00 is in the older file's covers and margin, read through: searched (the newer may hold it too, and was
    # not asked here: only files the recording step chose are known)
    assert rotation == [[], []]
    # (and the run at its last record is not asked for again: the same window would read no further)
    assert truncated[0] == ["aipanda093.cern.ch: 2026-09-28 07:10:01 .. 2026-09-28 07:30:00"]
    # the last record (07:25) is another string's, read past the cut: the stretches the record names are what was read
    assert searched[0] == ["aipanda093.cern.ch: 2026-09-28 07:10:00 .. 2026-09-28 07:14:59",
                           "aipanda093.cern.ch: 2026-09-28 07:20:01 .. 2026-09-28 07:30:00"]
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
            kept = rerun._records_json({"branches": r.branches})["branches"]
            said = [rerun.T.text(rerun.T.from_json(b["record"])) if b["record"] else None for b in kept]
            print(json.dumps([seen, stopped, [x for x in r.lines if x.startswith("====")], kept, said, r.hop]))
            """,
        )
    )
    # say() prints each line as it goes: the answer is the last
    seen, stopped, marks, branches, said, hop = json.loads(out.splitlines()[-1])
    # result.json's outcome.branches: each branch's end as a record; the fork's stop is the fork's step
    assert branches == [
        {"label": "mover a", "kind": "end", "step": "S2", "record": None},
        {"label": "mover b", "kind": "stop", "step": "S2", "code": None, "record": {"kind": "stop-line", "facts": {"stop": {"said": {
            "kind": "at-step", "facts": {"step": "S2", "what": "something only the second lacks"}}}}}},
    ]
    # and reads back as the run said it
    assert said == [None, "STOP S2: something only the second lacks"]
    assert hop == "S1"
    # the second branch starts from the state the fork was reached in, not from what the first learned
    assert [names for names, _ in seen] == ["first", "second"]
    assert all(sorted(facts) == ["kept"] for _, facts in seen)
    assert marks == ["==== 1 of 2: mover a", "==== 1: reached the end", "==== 2 of 2: mover b",
                     "==== 2: STOP S2: something only the second lacks"]
    assert stopped == "1 of 2 branch(es) stopped: 2 (S2: something only the second lacks)"


def test_a_stop_is_kept_as_a_record_taken_when_said_and_result_json_reads_back_as_the_run_said(tmp_path):
    out = _ok(
        _run(
            tmp_path,
            """
            import json
            from bamboo.querier import render as T, rerun

            unread = ["'a' not read", "'b' read false"]
            conditions = T.Said("its-conditions", {"unread": unread})
            unread.append("'c' not read")
            stop = rerun.Stop(T.at("H3", T.Said("values-the-guard-reads", {"conditions": conditions})), "K1")
            odd = T.Said("in-parens", {"what": ("x", 1)})
            kept = rerun._records_json({"kind": "stop", "record": stop.said,
                                        "branches": [{"label": "l", "record": T.Said("stop-line", {"stop": stop.said})}]})
            print(json.dumps([str(stop), stop.args[0], T.text(odd), T.same_read_back(odd), kept,
                              T.outcome_text(kept | {"hand_used": ["H1"]}),
                              T.outcome_text({"kind": "end", "record": T.to_json(rerun._ends([("H1", T.Said("by-elimination")), ("H2", "x")]))}),
                              T.outcome_text({"kind": "end", "record": rerun._ends([])})]))
            """,
        )
    )
    said, arg, odd, back, kept, stop_line, end_lines, no_end = json.loads(out)
    # the list is taken when the sentence is said, as the f-string it replaces took it
    assert said == arg == "H3: the values its guard reads at its run (its conditions: 'a' not read, 'b' read false)"
    # a value JSON does not hold keeps its repr and str
    assert odd == " (('x', 1))" and back is True
    # result.json keeps the records, nested, no sentence
    assert kept["record"] == {"kind": "at-step", "facts": {"step": "H3", "what": {"said": {
        "kind": "values-the-guard-reads", "facts": {"conditions": {"said": {
            "kind": "its-conditions", "facts": {"unread": ["'a' not read", "'b' read false"]}}}}}}}}
    assert kept["branches"][0]["record"]["kind"] == "stop-line" and kept["branches"][0]["label"] == "l"
    # and they read back as the run's last lines said them
    assert stop_line == f"STOP {said}; HAND used 1 time(s) ['H1']"
    assert end_lines == "END by elimination: \nEND x" and no_end is None


def test_a_db_proxy_run_is_a_movers_only_when_its_mark_names_the_movers_method(tmp_path):
    out = _ok(
        _run(
            tmp_path,
            """
            import json
            from types import SimpleNamespace as N
            from bamboo.querier.which_ran import of_mover

            owner = "pandaserver/taskbuffer/db_proxy_mods/job_complex_module.py::getJobs"
            print(json.dumps([
                of_mover(N(token="getJobs < BOINC_MCORE 2026-10-04T18:38:06.929035 >"), owner),
                of_mover(N(token="updateJobStatus < PandaID=7338945928 >"), owner),
                of_mover(N(token="< jediTaskID=52757505 >"), "pandajedi/jediorder/TaskCommando.py::runImpl"),
                of_mover(N(token="7339032371"), "pandaserver/dataservice/adder_gen.py::run"),
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


def test_a_statement_runs_between_two_prints_on_a_path_that_prints_nothing_else(tmp_path):
    corpus = _corpus(tmp_path)
    (corpus / "pandaserver" / "a.py").write_text(
        textwrap.dedent(
            """
            def f(self, x):
                self.logger.debug("before")
                self.tb.update(x)
                self.lock.release()
                if x:
                    self.logger.debug("one")
                else:
                    self.logger.debug("two")
                for y in x:
                    self.logger.debug("loop")
                    self.tb.inner(y)
                try:
                    self.tb.risky(x)
                except Exception:
                    self.logger.error("failed")
                self.logger.debug("end")
            """
        )
    )
    out = _ok(
        _run(
            tmp_path,
            """
            import json
            from bamboo.codemap.panda import flow
            from bamboo.querier import rows as R

            node = R.functions("pandaserver/a.py", "f")[0]
            at = lambda line: R.statement_at(node, line)
            prints = [s for s in R.own_nodes(node) if R.is_print(s)]
            between = lambda a, b, s: flow.runs_between(at(a), at(b), at(s), prints)
            print(json.dumps({
                "the next statement calls": [between(3, 7, 4), between(3, 9, 5)],
                "not after the first": between(7, 11, 4),
                "a loop's next pass": between(11, 11, 12),
                "after the loop": [between(11, 17, 14), between(11, 16, 14)],
                "a print on every way": between(3, 17, 14),
                "a print is no call": between(3, 7, 3),
            }))
            """,
        )
    )
    got = json.loads(out.splitlines()[-1])
    # the update and the release run after "before" and before either arm's print
    assert got["the next statement calls"] == [True, True]
    assert got["not after the first"] is False
    # a loop's body runs again after its own print
    assert got["a loop's next pass"] is True
    # the risky call runs before "end", or raises to the handler's print
    assert got["after the loop"] == [True, True]
    # every way from "before" to "end" prints an arm's line and the loop's: never two in a row
    assert got["a print on every way"] is False
    assert got["a print is no call"] is False


def test_a_call_with_no_print_after_it_on_every_path_is_placed_between_the_records_around_the_run(tmp_path):
    corpus = _corpus(tmp_path)
    (corpus / "pandaserver" / "a.py").write_text(
        textwrap.dedent(
            """
            def f(self, job):
                self.logger.debug("start")
                if job.x:
                    self.logger.debug("set A")
                    self.tb.update(job)
                if self.check(job):
                    self.logger.debug("on hold")
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
            stmt = R.statement_at(node, 6)
            # start, set A, another function's record, end
            run = N(records=[N(stamp=f"2026-01-01 00:00:{s},000") for s in ["00", "01", "02", "03"]])
            placed = {3: [0], 5: [1], 9: [3]}
            hits = {k: [(x, 9, "") for x in v] for k, v in placed.items()}
            every = lambda row: True
            caller = N(fn=fn, run=run, unambiguous=placed, hits=hits, would_show=every)
            # "on hold" matched the third record too, though not alone: the function may have printed there
            read_between = N(fn=fn, run=run, unambiguous=placed, hits={**hits, 8: [(2, 9, "")]}, would_show=every)
            # "set A" not read in the run: "start" and "end" are the two records around the call
            around = {3: [0], 9: [3]}
            unread = {k: [(x, 9, "") for x in v] for k, v in around.items()}
            shown = N(fn=fn, run=run, unambiguous=around, hits=unread, would_show=every)
            unshown = N(fn=fn, run=run, unambiguous=around, hits=unread, would_show=lambda row: row.line != 5)
            inner = lambda a, b: N(first=f"2026-01-01 00:00:{a}", last=f"2026-01-01 00:00:{b}")
            print(json.dumps([
                [len(x) for x in R.must_rows(node, stmt)],
                inside_call(caller, inner("01,200", "01,800"), "pandaserver/a.py:6"),
                inside_call(caller, inner("00,200", "00,800"), "pandaserver/a.py:6"),
                inside_call(caller, inner("03,200", "03,800"), "pandaserver/a.py:6"),
                inside_call(read_between, inner("01,200", "01,800"), "pandaserver/a.py:6"),
                inside_call(shown, inner("01,200", "01,800"), "pandaserver/a.py:6"),
                inside_call(unshown, inner("01,200", "01,800"), "pandaserver/a.py:6"),
            ]))
            """,
        )
    )
    shape, inside, before_its_arm, after_the_end, printed_between, set_a_shown, set_a_unshown = json.loads(out.splitlines()[-1])
    # the call ends an arm and the next statement calls: no print every path runs after it
    assert shape == [2, 0]
    # "set A" and "end", two of the function's records in a row, hold the run, and the call is between their prints
    assert inside == [1, 3]
    # between "start" and "set A" the call does not run: "set A" is printed first
    assert before_its_arm is None
    assert after_the_end is None
    # a row of the function read between the two: they are not one pass's
    assert printed_between is None
    # every path from "start" to "end" through the call prints "set A": where the run would show it, its absence
    # between the two says no such path ran; where it would not, it may have printed unseen (KNOWN_ISSUES.md K67)
    assert set_a_shown is None
    assert set_a_unshown == [0, 3]


def test_two_records_in_a_row_show_a_line_ran_only_where_every_path_between_their_prints_runs_it(tmp_path):
    corpus = _corpus(tmp_path)
    (corpus / "pandajedi" / "r.py").write_text(
        textwrap.dedent(
            """
            def run(self, task):
                self.logger.info("extracting common")
                try:
                    self.impl.extract(task)
                except Exception:
                    self.logger.error("failed")
                    return
                if task.x:
                    self.tb.update(task)
                self.logger.info("checking length")

            def skipping(self, task):
                self.logger.info("a")
                if task.y:
                    self.logger.debug("skip")
                else:
                    self.impl.extract(task)
                self.logger.info("b")
            """
        )
    )
    out = _ok(
        _run(
            tmp_path,
            """
            import json
            from types import SimpleNamespace as N
            from bamboo.codemap.panda import flow
            from bamboo.querier import rows as R
            from bamboo.querier.which_ran import rows_around

            node = R.functions("pandajedi/r.py", "run")[0]
            at = lambda line: R.statement_at(node, line)
            prints = [s for s in R.own_nodes(node) if R.is_print(s)]
            through = lambda a, b, s: flow.passes_through(at(a), at(b), at(s), prints)
            row_at = lambda s: N(line=s.lineno) if R.is_print(s) else None
            fn = N(node=node, row_at=row_at)
            every = lambda row: True
            # record 1 is another function's (the callee's, say)
            placed = {3: [0], 11: [2]}
            hits = {k: [(x, 9, "") for x in v] for k, v in placed.items()}
            seen = lambda unambiguous, hits, shows=every: N(unambiguous=unambiguous, hits=hits, would_show=shows)
            # "a" and "b" in a row: "skip" closes the way round the call where the run would show it
            node2 = R.functions("pandajedi/r.py", "skipping")[0]
            fn2 = N(node=node2, row_at=row_at)
            ab = {14: [0], 19: [1]}
            ab_hits = {k: [(x, 9, "") for x in v] for k, v in ab.items()}
            print(json.dumps({
                "the call": through(3, 11, 5),
                "an arm": through(3, 11, 10),
                "raised to the handler": through(3, 7, 5),
                "around the call": rows_around(fn, seen(placed, hits), at(5)),
                "around the arm": rows_around(fn, seen(placed, hits), at(10)),
                "a row read between": rows_around(fn, seen(placed, {**hits, 7: [(1, 9, "")]}), at(5)),
                "only the first": rows_around(fn, seen({3: [0]}, {3: [(0, 9, "")]}), at(5)),
                "skip shown": rows_around(fn2, seen(ab, ab_hits), R.statement_at(node2, 18)),
                "skip unshown": rows_around(fn2, seen(ab, ab_hits, lambda row: row.line != 16), R.statement_at(node2, 18)),
            }))
            """,
        )
    )
    got = json.loads(out.splitlines()[-1])
    # "extracting common" then "checking length" with nothing of it between: the call ran (and returned)
    assert got["the call"] is True and got["around the call"] == [3, 11]
    # the update is one arm's: a pass from 3 to 11 need not run it
    assert got["an arm"] is False and got["around the arm"] is None
    # the handler's print after 3 is reached only by the call raising
    assert got["raised to the handler"] is True
    # no two rows in a row: nothing shown; "failed" matched the record between, though not alone: not a pass's two
    assert got["only the first"] is None
    assert got["a row read between"] is None
    # "skip" is the only other way from "a" to "b": shown, its absence says the call ran; unshown, it may have printed
    # unseen, and "a" then "b" no longer show the call ran (KNOWN_ISSUES.md K67)
    assert got["skip shown"] == [14, 19]
    assert got["skip unshown"] is None


def test_rule_3_places_a_printed_row_before_or_after_the_site_only_where_the_flow_lets_it_go_one_way(tmp_path):
    corpus = _corpus(tmp_path)
    (corpus / "pandaserver" / "o.py").write_text(
        textwrap.dedent(
            """
            def f(self, x):
                self.logger.debug("first")
                if x:
                    self.logger.debug("arm")
                    return
                for y in x:
                    self.logger.debug("loop")
                    self.tb.update(y)
                self.tb.write(x)
                self.logger.debug("last")
            """
        )
    )
    out = _ok(
        _run(
            tmp_path,
            """
            import json
            from bamboo.querier import rows as R
            from bamboo.querier.which_ran import _order

            node = R.functions("pandaserver/o.py", "f")[0]
            at = lambda line: R.statement_at(node, line)
            loop = at(7)
            print(json.dumps([
                _order(at(3), at(10), []), _order(at(11), at(10), []), _order(at(5), at(10), []),
                _order(at(8), at(9), []), _order(at(8), at(9), [loop]),
            ]))
            """,
        )
    )
    first, last, arm, looped, once = json.loads(out.splitlines()[-1])
    assert (first, last) == ("before", "after")
    # the arm returns: one call cannot print it and run the write, so it places nothing (its line order said before)
    assert arm is None
    # round a loop the print comes both before and after the update; a loop gone round once has no way back
    assert looped is None
    assert once == "before"


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
            ], default=str))
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
            from bamboo.querier import render as T
            from bamboo.querier.rerun import _gaps_json, hides_where, map_gaps

            part = ReaderGap(kind="partial", file="p/m.py", line=472)
            unnoded = ReaderGap(kind="unnoded", file="p/m.py", line=222, stops=[("unreached", "dry_sql_query")])
            follow = SimpleNamespace(reader_gaps={"p/m.py::f": [part], "p/m.py::g": [part, unnoded]},
                                     guarded_by=[SimpleNamespace(owner="p/m.py::h", gaps=[unnoded]),
                                                 SimpleNamespace(owner="p/m.py::k", gaps=[])])
            gaps = map_gaps(follow)
            print(json.dumps([sorted(gaps), [hides_where(gaps[o]) for o in sorted(gaps)], hides_where([]),
                              T.text(T.Said("map-could-not-read-all", {"gaps": _gaps_json(gaps["p/m.py::g"])}))], default=str))
            """,
        )
    )
    owners, hidden, none, said = json.loads(out.splitlines()[-1])
    # a selecting function's gaps and a guarding one's, none for a function the map read all of
    assert owners == ["p/m.py::f", "p/m.py::g", "p/m.py::h"]
    # a statement read in part hides which rows it takes, not where they go: it alone keeps no mover
    assert hidden == [False, True, True] and none is False
    # every gap is said from its record, a text per kind in the map's order, with its lines
    assert said == ("the map could not read all of it (@222 dry_sql_query -- a statement here that reads these rows "
                    "stopped being read, and has no node; @472 -- a statement here that reads these rows was read only in "
                    "part)")


def test_s1_drops_a_mover_with_no_arm_only_where_the_reach_finds_no_write_and_nothing_open(tmp_path):
    out = _ok(
        _run(
            tmp_path,
            """
            import json
            from bamboo.querier.rerun import reach_said

            def reach(writes, holes=0, own=(), closed=0):
                return {"starts": ["p/m.py::f"], "writes": writes, "holes": holes, "first_holes": ["p/m.py::f:3 x.y -- why"][:holes],
                        "assumptions": {"runner_runs_only_sql": 2, **({"closed_by_name": closed} if closed else {})},
                        "closed": closed, "first_closed": ["p/m.py::f:5 d[k].append -- an element of a dict"][:closed],
                        "closure": 4, "own_arms": list(own)}

            w = [{"function": "p/w.py::W.killJob", "arms": ["failed@3"], "chain": ["p/m.py:7"]}]
            print(json.dumps([reach_said(None, False), reach_said(reach([]), False), reach_said(reach(None, 1), False),
                              reach_said(reach(w), False), reach_said(reach([]), True), reach_said(reach([], own=["p/m.py::f"]), False),
                              reach_said(reach([], closed=1), False)], default=str))
            """,
        )
    )
    unasked, nothing, holed, writes, gapped, own, by_name = json.loads(out.splitlines()[-1])
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
    # a drop that rests on calls closed by their name names them (O1)
    assert by_name == [True, "no arm of its own, and reaches no write of another value with nothing left open (closure 4; "
                             "assuming ['closed_by_name', 'runner_runs_only_sql']; 1 call(s) closed by name, e.g. "
                             "['p/m.py::f:5 d[k].append -- an element of a dict']): dropped"]


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
                    node("p/h.py::h", 1, 4, holes=[ReachHole(line=3, callee="f", why=RE.HOLE_LOCAL_CALLED)]),
                    node("p/n.py::n", 1, 4, holes=[ReachHole(line=3, callee="d[k].append", why="an element of a dict")])]
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
                ("hole", "p/h.py::h", [2]), ("by name", "p/n.py::n", [2]), ("own arm", "p/w.py::write", [2]),
                ("none holds", "p/x.py::x", [3])]}
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
    # a called local: no name closes it, so it stays open
    assert got["hole"]["writes"] is None and got["hole"]["holes"] == 1 and got["hole"]["closed"] == 0
    # a call no function of its name can lead to a write: closed by its name, said (O1)
    assert got["by name"]["writes"] == [] and got["by name"]["holes"] == 0
    assert got["by name"]["closed"] == 1 and got["by name"]["first_closed"] == ["p/n.py::n:3 d[k].append -- an element of a dict"]
    assert got["by name"]["assumptions"] == {"closed_by_name": 1}
    assert got["own arm"]["own_arms"] == ["p/w.py::write"]
    assert got["none holds"]["writes"] is None and got["none holds"]["starts"] == []
    # the record is read once, and the ends once per question
    assert len(lines) == 1 and lines[0].startswith("  reach: the dump's reach_graph, 5 function(s); every answer rests on [")
    assert functions == 5 and asked == [["JobSpec.jobStatus", "running"]]
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

    def branch(line, prints, names=None, hole="", whole=False):
        return N(line=line, must_prints=prints, must_names=names, must_hole=hole, must_names_whole=whole)

    key = name(5, "JobSpec.PandaID")
    writers = [N(owner="p/w.py::w", branches=[
        branch(10, [5], [key]),
        branch(11, [5], [key, name(5, "JobSpec.jobStatus")]),
        branch(12, [5], [key], whole=True),
        branch(20, [5], [name(5, "JobSpec.jobStatus")]),
        branch(21, [5], [name(5, "JobSpec.PandaID", row="")]),
        branch(22, [5], None, "names: the arm does not write the subject as an attribute of an object"),
        branch(23, [5], [name(5, "JobSpec.jobStatus")], whole=True),
        branch(24, [5], [], whole=True),
        branch(30, []),
        branch(40, None, None, "not read: a lambda"),
    ])]
    cols = {"JobSpec.PandaID"}
    held = lambda lines, columns=cols: (lambda p: [p["held"], p["why"]])(prints_naming(writers, "p/w.py::w", lines, columns))
    print(json.dumps({"named": held([10, 11]), "another column": held([10, 20]), "not the row": held([21]),
                      "sql": held([22]), "none before": held([10, 30, 20]), "unread": held([10, 40]),
                      "no key column": held([10], set()), "no arm": held([]), "named whole": held([10, 12]),
                      "whole, another column": held([10, 23]), "whole, none": held([24]), "whole before unread": held([40, 24]),
                      "none before whole": held([23, 30]), "whole and not": held([20, 23])}, default=str))
    """,
        )
    )
    got = json.loads(out.splitlines()[-1])
    # a line before every arm names the written row's key: a run that wrote it printed what the asks search by
    assert got["named"] == [True, "a line before every arm at [10, 11] names the written row's ['JobSpec.PandaID']"]
    # a name of another column, of a row not known to be the written one, or none read (an SQL arm), where the map did
    # not read the names whole: not known
    for case in ("another column", "not the row", "sql"):
        assert got[case][0] is None and "the map did not read their names whole" in got[case][1]
    # read whole: the key named still holds; none naming it says no run need print it, ahead of an arm not known
    assert got["named whole"] == [True, "a line before every arm at [10, 12] names the written row's ['JobSpec.PandaID']"]
    whole = ("were read whole and none names the written row's ['JobSpec.PandaID']: a search by the key cannot show a "
             "run absent")
    assert got["whole, another column"] == [False, f"the lines before the arm(s) at [23] {whole}"]
    assert got["whole, none"] == got["whole before unread"] == [False, f"the lines before the arm(s) at [24] {whole}"]
    assert got["whole and not"] == [False, f"the lines before the arm(s) at [23] {whole}"]
    assert got["none before whole"] == [False, "no line runs before the arm(s) at [30] on every path"]
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


def test_an_arm_past_the_region_is_the_statements_where_its_rows_fill_the_key_and_an_unread_key_keeps_the_owner(tmp_path):
    out = _ok(
        _run(
            tmp_path,
            _MOVERS_MAP + """
    from bamboo.querier import mapdump
    from bamboo.querier.rerun import keyed_arms

    owner = "p/fin.py::prepare"
    writers.append(writer(owner, branch(3913, "finishing"), branch(3995, "failed"),
                          N(line=4059, outcome="done", path_condition=[], row_precondition=[], other_lines=[4082]),
                          branch(4300, "broken"), branch(4400, "running")))

    def rows(line):
        return N(kind="rows", name=f"panda:{owner}@{line}", site_owner=owner, site_line=line)

    def key(term, writes, *sources, unread=""):
        return N(owner=owner, term=term, writes=writes, unread=unread, sources=list(sources))

    def follow(line):
        return N(guarded_by=[], reader_statements=[
            stmt(owner, line, ["JobSpec.jobStatus=running"], [line, line + 10], [f"p/fin.py::helper{line}"])])

    before = movers_of(follow(3723), writers, "JobSpec.jobStatus", "running")
    mapdump._FRAGMENT[:] = [N(key_froms=[
        # 3723's rows (and 3768's) fill the key of the writes at 3913, and at 4082, a line of the branch at 4059
        key("jediTaskID=:jediTaskID", {"JobSpec.jobStatus": [3913, 4082]}, rows(3723), rows(3768)),
        # a parameter fills 3995's; another subject's write keyed by 3723's rows; 4400 writes the value observed
        key("jediTaskID=:jediTaskID", {"JobSpec.jobStatus": [3995]}, N(kind="parameter", name="taskID")),
        key("jediTaskID=:jediTaskID", {"JobSpec.lockedBy": [3995]}, rows(3723)),
        key("jediTaskID=:jediTaskID", {"JobSpec.jobStatus": [4400]}, rows(3723)),
        # 4300's key was not read, nor followed where it was
        key("PandaID=:PandaID", {"JobSpec.jobStatus": [4300]}, unread="no bind found"),
        key("jobsetID=:jobsetID", {"JobSpec.jobStatus": [4300]}, N(kind="unknown", name="jsid")),
    ])]
    arms = arms_of(writers, owner, "JobSpec.jobStatus", "running")
    print(json.dumps([keyed_arms(writers, follow(n).reader_statements[0], "JobSpec.jobStatus", arms) for n in (3723, 3800)]))
    print(json.dumps([before, movers_of(follow(3723), writers, "JobSpec.jobStatus", "running"),
                      movers_of(follow(3800), writers, "JobSpec.jobStatus", "running")]))
    """,
        )
    )
    unread = ["PandaID=:PandaID (no bind found)", "jobsetID=:jobsetID (not followed: jsid)"]
    # 3800's rows fill no key: only the unread one may be its
    assert json.loads(out.splitlines()[-2]) == [[[3913, 4059], {"4300": unread}, {}, {}], [[], {"4300": unread}, {}, {}]]
    before, keyed, kept = json.loads(out.splitlines()[-1])
    # no arm in the region: before KeyFrom only the helper took the rows; now the owner is a mover by its keyed arms
    # too, and one whose key was not read is kept, said; the helper takes them as before
    assert before == {"p/fin.py::helper3723": "prepare@3723 hands the rows it selects to it"}
    assert keyed == {"p/fin.py::prepare": "prepare@3723 selects the value, and its rows fill the key of the write(s) at [3913, 4059]",
                     "p/fin.py::helper3723": "prepare@3723 hands the rows it selects to it"}
    assert kept == {"p/fin.py::prepare": f"prepare@3800 selects the value, and the map could not read what fills "
                                         f"the key of the write(s) at [4300] (4300: {', '.join(unread)}): kept",
                    "p/fin.py::helper3800": "prepare@3800 hands the rows it selects to it"}


def test_rows_read_again_by_their_key_are_handed_on_to_writers_where_the_reread_runs_with_the_value(tmp_path):
    out = _ok(
        _run(
            tmp_path,
            _MOVERS_MAP + """
    from bamboo.querier import mapdump
    from bamboo.querier.rerun import paths_of

    owner = "p/fin.py::getTasks"
    for name, line, outcome in [("post", 165, "done"), ("other", 5, "failed"), ("consumers", 808, "cancelled"),
                                ("already", 3, "failed"), ("keyed", 1750, "failed")]:
        writers.append(writer(f"p/{name}.py::{name}", branch(line, outcome)))
    writers.append(writer(owner, branch(1746, "failed")))

    def rows(line, same_rows=True, never_at=(), via=(), reaches=()):
        return N(kind="rows", name=f"panda:{owner}@{line}", site_owner=owner, site_line=line, same_rows=same_rows,
                 never_at=list(never_at), via=list(via), reaches=list(reaches))

    def read(at, statement, *sources):
        return N(owner=at, term="jediTaskID=:jediTaskID", writes={}, reads=True, unread="", sources=list(sources),
                 statement=statement)

    def site(line, region, crosses_to):
        return N(owner=owner, anchor=N(line_start=line), region=region, crosses_to=crosses_to)

    follow = N(guarded_by=[], reader_statements=[
        stmt(owner, 1676, ["JobSpec.jobStatus=running", "JobSpec.jobStatus=scouted"], [1634, 1690],
             ["p/already.py::already"])])
    before = movers_of(follow, writers, "JobSpec.jobStatus", "running")
    mapdump._FRAGMENT[:] = [N(
        key_froms=[
            # 1731 reads 1676's rows again for each value but scouted, and hands them to post, record and already
            read(owner, 1731, rows(1676, never_at=["JobSpec.jobStatus=scouted"])),
            # 1740 reads them by another column: not followed
            read(owner, 1740, rows(1676, same_rows=False)),
            # killJob, through a call, reads them again: its rows reach consumers and events
            read("p/kill.py::killJob", 991, rows(1676, via=["p/fin.py:1700"],
                                                 reaches=["p/consumers.py::consumers", "p/events.py::events"])),
            # 1745 reads them again; its rows fill the key of the owner's write at 1746, outside 1731's region
            read(owner, 1745, rows(1676)),
            N(owner=owner, term="jediTaskID=:jediTaskID", writes={"JobSpec.jobStatus": [1746]}, unread="",
              sources=[N(kind="rows", name=f"panda:{owner}@1745", site_owner=owner, site_line=1745)]),
        ],
        read_sites=[site(1731, [1729, 1737], ["p/post.py::post", "p/record.py::record", "p/already.py::already"]),
                    site(1740, [1738, 1756], ["p/other.py::other"]), site(1745, [1745, 1745], [])])]
    found = [movers_of(follow, writers, "JobSpec.jobStatus", v) for v in ("running", "scouted")]
    paths = paths_of(follow, writers, "JobSpec.jobStatus", "running")
    print(json.dumps([before, found, {o: [st.line for st in p] for o, p in paths.items()}]))
    """,
        )
    )
    before, (running, scouted), paths = json.loads(out.splitlines()[-1])
    again = "getTasks@1676 selects the value, and {} reads its rows again"
    assert before == {"p/already.py::already": "getTasks@1676 hands the rows it selects to it"}
    # the owner first: 1745 takes 1676's rows again, so the key of its write at 1746 is 1676's rows' (key_chain), as
    # the statement is read; after every other way: to the writers 1731 calls (not record, which writes nothing;
    # already keeps its first why), and the writer killJob's rows reach through a call (not events)
    assert list(running.items()) == [
        ("p/fin.py::getTasks", "getTasks@1676 selects the value, and its rows fill the key of the write(s) at [1746]"),
        ("p/already.py::already", "getTasks@1676 hands the rows it selects to it"),
        ("p/post.py::post", again.format("getTasks@1731") + " and hands them to it"),
        ("p/consumers.py::consumers", again.format("killJob@991") + " and hands them to it"),
    ]
    # 1731 never runs for a scouted row: post is not reached then
    assert "p/post.py::post" not in scouted and "p/consumers.py::consumers" in scouted
    # the statement whose rows were taken is 1676, for L2 to read
    assert paths["p/post.py::post"] == [1676] and paths["p/fin.py::getTasks"] == [1676]


def test_a_key_filled_through_the_functions_reads_is_the_statements_and_an_arm_keyed_elsewhere_is_said(tmp_path):
    out = _ok(
        _run(
            tmp_path,
            _MOVERS_MAP + """
    from bamboo.querier import mapdump
    from bamboo.querier.rerun import elsewhere_of, keyed_arms

    S = "JediFileSpec.status"
    reset, status, hop, bare = "p/te.py::reset", "p/jc.py::updateStatus", "p/te.py::hop", "p/jc.py::bare"
    writers[:] = [writer(reset, branch(5035, "ready")), writer(status, branch(241, "failed")),
                  writer(hop, branch(70, "ready")), writer(bare, branch(90, "ready"))]

    def rows(owner, line):
        return N(kind="rows", name=f"panda:{owner}@{line}", site_owner=owner, site_line=line)

    def key(owner, term, *sources, writes=None, statement=0, reads=False):
        return N(owner=owner, term=term, writes=writes or {}, reads=reads, statement=statement, unread="", sources=list(sources))

    mapdump._FRAGMENT[:] = [N(key_froms=[
        # reset: 5001 selects running files, 5009 takes PandaIDs by their datasetID (another table), 5026 the job's
        # files by those PandaIDs, and 5038 sets them ready by 5026's keys
        key(reset, "datasetID=:datasetID", rows(reset, 5001), reads=True, statement=5009),
        key(reset, "PandaID=:PandaID", rows(reset, 5009), reads=True, statement=5026),
        key(reset, "fileID=:fileID", rows(reset, 5026), writes={S: [5035]}, statement=5038),
        # updateStatus: its one arm writes the row a parameter names
        key(status, "PandaID=:PandaID", N(kind="parameter", name="pandaID"), writes={S: [241]}, statement=241),
        # hop: 60's rows fill 70's key, and 60 takes its rows by a value the map did not follow
        key(hop, "PandaID=:PandaID", N(kind="unknown", name="ids"), reads=True, statement=60),
        key(hop, "fileID=:fileID", rows(hop, 60), writes={S: [70]}, statement=71),
        # bare: no KeyFrom names 90's row
    ])]
    follow = N(guarded_by=[], reader_statements=[
        stmt(reset, 5001, [f"{S}=running"], [4997, 5004]), stmt(status, 294, [f"{S}=running"], [288, 295]),
        stmt(hop, 50, [f"{S}=running"], [49, 52]), stmt(bare, 80, [f"{S}=running"], [79, 82])])
    arms = arms_of(writers, reset, S, "running")
    print(json.dumps([keyed_arms(writers, follow.reader_statements[0], S, arms), movers_of(follow, writers, S, "running"),
                      elsewhere_of(follow, writers, S, "running")]))
    """,
        )
    )
    keyed, movers, elsewhere = json.loads(out.splitlines()[-1])
    # three statements on, 5035's key is still 5001's rows'
    assert keyed == [[5035], {}, {}, {}]
    assert movers["p/te.py::reset"] == "reset@5001 selects the value, and its rows fill the key of the write(s) at [5035]"
    # each stays a mover; only updateStatus's arm is known to name another row -- hop's key may come from 50's rows
    # through the value not followed, and bare's row is named by nothing the map read
    assert set(movers) == {"p/te.py::reset", "p/jc.py::updateStatus", "p/te.py::hop", "p/jc.py::bare"}
    assert elsewhere == {"p/jc.py::updateStatus": [241]}


def test_an_arm_keyed_by_rows_taken_from_the_same_table_by_the_same_values_is_the_selecting_statements(tmp_path):
    out = _ok(
        _run(
            tmp_path,
            _MOVERS_MAP + """
    from bamboo.querier import mapdump
    from bamboo.querier.rerun import elsewhere_of, keyed_arms

    S, T = "JediFileSpec.status", "JEDI_Dataset_Contents"
    owners = {n: f"p/tc.py::{n}" for n in ("ins", "both", "par", "unk", "tab", "dis", "val")}
    writers[:] = [writer(o, branch(1113, "lost")) for o in owners.values()]

    def src(kind, name, attr="", site_owner="", site_line=None):
        return N(kind=kind, name=name, attr=attr, column="", site_owner=site_owner, site_line=site_line, paths=[])

    def rows(o, line):
        return src("rows", f"panda:{o}@{line}", site_owner=o, site_line=line)

    def param(name, attr=""):
        return src("parameter", name, attr)

    tid, did = param("datasetSpec", "JediDatasetSpec.jediTaskID"), param("datasetSpec", "JediDatasetSpec.datasetID")
    status = [src("constant", "'pending'"), src("constant", "'ready'")]

    def read(o, line, term, *sources, table=T):
        return N(owner=o, term=term, writes={}, reads=True, statement=line, unread="", table=table, sources=list(sources))

    def write(o, term, *sources):
        return N(owner=o, term=term, writes={S: [1113]}, reads=False, statement=1130, unread="", table=T, sources=list(sources))

    kfs = []
    for n, o in owners.items():
        # 923 selects the dataset's ready and pending files; 757 (none for val) the dataset's files, as each says
        kfs += [read(o, 923, "jediTaskID=:jediTaskID", param("parent_tid") if n == "par" else src("unknown", "ids") if n == "unk" else tid),
                read(o, 923, "datasetID=:datasetID", did), read(o, 923, "status=:status", *status)]
        wide = {"ins": [("jediTaskID=:jediTaskID", tid), ("datasetID=:datasetID", did)],
                "both": [("jediTaskID=:jediTaskID", tid), ("datasetID=:datasetID", did)],
                "par": [("jediTaskID=:jediTaskID", tid)], "unk": [("jediTaskID=:jediTaskID", src("unknown", "ids"))],
                "tab": [("jediTaskID=:jediTaskID", tid), ("datasetID=:datasetID", did)],
                "dis": [("datasetName=:datasetName", param("datasetSpec", "JediDatasetSpec.datasetName"))], "val": []}[n]
        kfs += [read(o, 757, term, s, table="JEDI_Datasets" if n == "tab" else T) for term, s in wide]
        if n == "val":
            # the write's terms filled from the parameters 923 compares: values alike, no statement's rows
            kfs += [write(o, "jediTaskID=:jediTaskID", tid), write(o, "datasetID=:datasetID", did),
                    write(o, "fileID=:fileID", param("fileSpec", "JediFileSpec.fileID"))]
        elif n == "both":
            # 923's own rows fill fileID beside 757's; datasetID by 757's alone
            kfs += [write(o, "fileID=:fileID", rows(o, 757), rows(o, 923)), write(o, "datasetID=:datasetID", rows(o, 757))]
        else:
            kfs += [write(o, "fileID=:fileID", rows(o, 757)), write(o, "jediTaskID=:jediTaskID", tid)]
    mapdump._FRAGMENT[:] = [N(key_froms=kfs)]
    follow = N(guarded_by=[], reader_statements=[stmt(o, 923, [f"{S}=ready"], [916, 1084]) for o in owners.values()])
    print(json.dumps([[keyed_arms(writers, rs, S, arms_of(writers, rs.owner, S, "ready")) for rs in follow.reader_statements[:2]],
                      movers_of(follow, writers, S, "ready"), elsewhere_of(follow, writers, S, "ready")]))
    """,
        )
    )
    (ins, both), movers, elsewhere = json.loads(out.splitlines()[-1])
    # 757 takes the dataset's files by the values 923 takes its ready ones by: a file 923 selects can be the one 1113
    # writes by 757's rows, and the line says through which
    assert ins == [[1113], {}, {}, {"1113": ["fileID=:fileID by ins@757's rows"]}]
    assert movers["p/tc.py::ins"] == ("ins@923 selects the value, and its rows fill the key of the write(s) at [1113] (1113: "
                                      "fileID=:fileID by ins@757's rows, which take the same table by the same values)")
    # 923's own rows fill one key of the arm: it is 923's, said so, whatever the other key's rows
    assert both == [[1113], {}, {}, {}]
    assert movers["p/tc.py::both"] == "both@923 selects the value, and its rows fill the key of the write(s) at [1113]"
    # a term filled otherwise (the parent task's), another table, no term in common, and terms filled by the same
    # values but no rows (updateJob's if and elif): the rows are not 923's; a value not followed keeps the mover
    assert elsewhere == {f"p/tc.py::{n}": [1113] for n in ("par", "tab", "dis", "val")}
    assert movers["p/tc.py::unk"] == "unk@923 selects the value"


def test_a_crossing_hands_on_only_the_values_its_routes_back_up_select(tmp_path):
    out = _ok(
        _run(
            tmp_path,
            _MOVERS_MAP + """
    S = "JediTaskSpec.status"
    writers[:] = [writer("p/tc.py::getTasks", branch(1900, "ready")), writer("p/tb.py::setCloud", branch(204, "assigning")),
                  writer("p/jg.py::run", branch(90, "ready")), writer("p/kill.py::killJobs", branch(5, "aborted"))]
    # 1898 has an arm of its owner's in its region; 95's owner has none
    region = stmt("p/tc.py::getTasks", 1898, [f"{S}=running", f"{S}=scouting"], [1890, 1910], ["p/tb.py::setCloud", "p/jg.py::run"])
    past = stmt("p/tc.py::other", 95, [f"{S}=running", f"{S}=failed"], [93, 96], ["p/kill.py::killJobs", "p/jg.py::run"])
    follow = N(guarded_by=[], reader_statements=[region, past])
    before = [movers_of(follow, writers, S, v) for v in ("running", "failed")]
    # setCloud is reached only back up a route that selects no status; killJobs only back up one that selects running
    region.crossing_selects = {"p/tb.py::setCloud": []}
    past.crossing_selects = {"p/kill.py::killJobs": [f"{S}=running"]}
    after = [movers_of(follow, writers, S, v) for v in ("running", "failed")]
    print(json.dumps([[sorted(m) for m in before], [sorted(m) for m in after]]))
    """,
        )
    )
    before, after = json.loads(out.splitlines()[-1])
    # a dump with no crossing_selects pairs every crossing with every value the statement selects
    assert before == [["p/jg.py::run", "p/kill.py::killJobs", "p/tb.py::setCloud", "p/tc.py::getTasks"],
                      ["p/jg.py::run", "p/kill.py::killJobs"]]
    # in the region and past it, a crossing takes only what its routes select; one the map names none of takes all
    assert after == [["p/jg.py::run", "p/kill.py::killJobs", "p/tc.py::getTasks"], ["p/jg.py::run"]]


def test_a_mover_whose_rows_reach_none_of_its_arms_is_one_by_key_where_the_reach_without_them_writes_nothing(tmp_path):
    out = _ok(
        _run(
            tmp_path,
            """
            import json
            from types import SimpleNamespace as N
            from bamboo.codemap import reach as RE
            from bamboo.codemap.models import Anchor, ReachHole, ReachNode
            from bamboo.querier import mapdump, rerun

            def node(owner, start, end, **k):
                return ReachNode(name=owner, map_id="m", derived_from="d", owner=owner, function=owner.split("::")[-1],
                                 anchor=Anchor(package="p", file=owner.split("::")[0], line_start=start, line_end=end), **k)

            status, insert = "p/jc.py::updateStatus", "p/jc.py::insert"
            rows = [node(status, 200, 300), node(insert, 1, 50, holes=[ReachHole(line=9, callee="f", why=RE.HOLE_LOCAL_CALLED)])]
            RE.moving_off = lambda junctions, graph, subject, value: RE.Destination({status: ["failed@241"], insert: ["ready@30"]}, [])
            rerun.elsewhere_of = lambda follow, writers, subject, observed: {status: [241], insert: [30]}
            rerun.defs_holding = lambda rel, func, lines: [object()]
            mapdump._FRAGMENT[:] = [N(reach_graph=rows, junctions=[])]
            r = object.__new__(rerun.Rerun)
            r.lines, r.result, r._reach, r._graph_read = [], {}, None, None
            q = rerun.Stuck("JobSpec.jobStatus", "waiting", "1")
            follow = N(reader_statements=[N(owner=status, line=294, selects=["JobSpec.jobStatus=waiting"], result="rows"),
                                          N(owner=insert, line=20, selects=["JobSpec.jobStatus=waiting"], result="rows")])
            said = r._rows_elsewhere(q, follow, [], {status: "", insert: ""}, {})
            print(json.dumps([{o: [to_key, text, reach["own_arms"], reach["writes"]] for o, (to_key, text, reach) in said.items()},
                              r._reach_of(q, status, [294])["own_arms"]], default=str))
            """,
        )
    )
    said, own = json.loads(out.splitlines()[-1])
    # with its arm 241 left out, updateStatus reaches no write with nothing open: a writer by its key, said
    to_key, text, own_arms, writes = said["p/jc.py::updateStatus"]
    assert to_key is True and own_arms == [] and writes == []
    assert text.startswith("the rows updateStatus@294 select(s) reach none of its arms [241]: the map names each one's row by a "
                           "key they do not fill, and the reach from it, those arms left out, writes no other value")
    assert text.endswith(": a writer that takes a row by its key (KNOWN_ISSUES.md K16)")
    # a call left open may go on into a write: insert stays a mover, said why
    to_key, text, own_arms, writes = said["p/jc.py::insert"]
    assert to_key is False and writes is None
    assert "; kept: reach: reaches no write of another value, but 1 call(s) on the way are left open" in text
    # asked as before, the own arm keeps it
    assert own == ["p/jc.py::updateStatus"]


def test_an_arm_no_write_in_its_function_persists_keeps_the_owner_said_with_where_its_fields_go(tmp_path):
    out = _ok(
        _run(
            tmp_path,
            _MOVERS_MAP + """
    from bamboo.querier import mapdump
    from bamboo.querier.rerun import keyed_arms

    owner = "d/dm.py::run"
    writers.append(writer(owner, branch(713, "failed"), branch(745, "merging"), branch(958, "failed"),
                          branch(2786, "output"), branch(4300, "broken"), branch(5000, "lost")))

    def kf(writes, term="", unread="", obj="", handed_to=(), *sources):
        return N(owner=owner, term=term, writes={"FileSpec.status": writes}, unread=unread, object=obj,
                 handed_to=list(handed_to), sources=list(sources))

    def follow(line):
        return N(guarded_by=[], reader_statements=[stmt(owner, line, ["FileSpec.status=running"], [line, line + 10])])

    mapdump._FRAGMENT[:] = [N(key_froms=[
        # an element of a container handed on, which the map does not read as the container's: no use found
        kf([713, 745], unread="no use found", obj="file"),
        kf([958], unread="no use found", obj="tmpFile"),
        # one object's fields handed to two calls and returned: one record per use
        kf([2786], unread="handed to a call", obj="spec", handed_to=["p/tb.py::updateJobs", "p/spec.py::addFile"]),
        kf([2786], unread="returned", obj="spec"),
        # a key the map did not read beside them
        kf([4300], "PandaID=:PandaID", "no bind found"),
        # 5000 is persisted by a write keyed by 359's rows, and handed on as well: keyed
        kf([5000], "PandaID=:PandaID", "", "", (), N(kind="rows", name=f"panda:{owner}@359", site_owner=owner, site_line=359)),
        kf([5000], unread="kept in what it was handed", obj="job"),
    ])]
    arms = arms_of(writers, owner, "FileSpec.status", "running")
    print(json.dumps(keyed_arms(writers, follow(359).reader_statements[0], "FileSpec.status", arms)))
    held = mapdump._FRAGMENT[0]
    held.key_froms = [k for k in held.key_froms if k.sources == []]
    print(json.dumps(movers_of(follow(359), writers, "FileSpec.status", "running")))
    held.key_froms = [k for k in held.key_froms if k.term == ""]
    print(json.dumps(movers_of(follow(359), writers, "FileSpec.status", "running")))
    """,
        )
    )
    to = ["spec (handed to a call: updateJobs, addFile)", "spec (returned)"]
    unpersisted = {"713": ["file (no use found)"], "745": ["file (no use found)"], "958": ["tmpFile (no use found)"],
                   "2786": to}
    # 5000 is keyed by 359's rows, whatever else its object reaches; the others are no key the map could not read
    assert json.loads(out.splitlines()[-3]) == [[5000], {"4300": ["PandaID=:PandaID (no bind found)"]}, unpersisted, {}]
    said = "713: file (no use found); 745: file (no use found); 958: tmpFile (no use found); 2786: " + ", ".join(to)
    # with no keyed arm both are said, joined; 5000's object is kept by the function, so it is one of them
    both, alone = (json.loads(line)["d/dm.py::run"] for line in out.splitlines()[-2:])
    assert both == (f"run@359 selects the value, and the map could not read what fills the key of the write(s) at "
                    f"[4300] (4300: PandaID=:PandaID (no bind found)), and no write in the function persists the "
                    f"value(s) set at [713, 745, 958, 2786, 5000] ({said}; 5000: job (kept in what it was handed)): kept")
    # without the unread key, the unpersisted alone
    assert alone == (f"run@359 selects the value, and no write in the function persists the value(s) set at "
                     f"[713, 745, 958, 2786, 5000] ({said}; 5000: job (kept in what it was handed)): kept")


def test_a_write_that_makes_the_row_is_no_arm_where_every_key_from_writing_it_there_says_a_new_row(tmp_path):
    out = _ok(
        _run(
            tmp_path,
            _MOVERS_MAP + """
    from bamboo.querier import mapdump
    from bamboo.querier.rerun import birth_why, births_of, by_key_of, refused_arms

    S = "JobSpec.jobStatus"
    files, archive, mix, unread, make = (f"p/{n}.py::{n}" for n in ("files", "archive", "mix", "unread", "make"))
    writers += [
        writer(files, branch(455, "ready"), branch(887, "staging"),
               N(line=892, outcome="pending", path_condition=[], row_precondition=[], other_lines=[893]), branch(1116, "ready")),
        writer(archive, branch(50, "cancelled")),
        writer(mix, branch(70, "done")),
        writer(unread, branch(90, "done")),
        writer(make, branch(10, "defined")),
    ]

    def key(owner, statement, lines, new_row=None, unread="", by="", at=None):
        kf = N(owner=owner, statement=statement, term="k=:k", writes={S: lines}, unread=unread, sources=[])
        if new_row is not None:
            kf.new_row, kf.new_row_by, kf.new_row_line = new_row, by, at
        return kf

    def follow():
        return N(guarded_by=[], reader_statements=[stmt(make, 8, [f"{S}=running"], [5, 12], ["p/make.py::helper"])])

    before = [movers_of(follow(), writers, S, "running"), make in by_key_of(writers, {}, S, "running")]
    mapdump._FRAGMENT[:] = [N(key_froms=[
        # one INSERT (930) of files: a KeyFrom per key column, each with the statement's mark
        *(key(files, 930, [455, 887, 892], new_row=True, by="sequence", at=623) for _ in range(3)),
        key(files, 1116, [1116]),                         # an UPDATE
        key(archive, 55, [50], new_row=False),            # an existing row moved into another table of its class
        # line 70 written by an INSERT and by an UPDATE run at the same call (one statement)
        key(mix, 71, [70], new_row=True, by="literal", at=71), key(mix, 71, [70]),
        key(unread, 91, [90], unread="no bind found"),    # an INSERT the map did not read
        key(make, 12, [10], new_row=True),
    ])]
    arms = {o.split("::")[-1]: sorted(arms_of(writers, o, S, "running")) for o in (files, archive, mix, unread, make)}
    print(json.dumps([arms, births_of(writers, files, S, "running"), births_of(writers, make, S, "running"),
                      refused_arms(writers, make, S, "running"), birth_why(files, S, [455, 887])], default=str))
    print(json.dumps(before + [movers_of(follow(), writers, S, "running"), make in by_key_of(writers, {}, S, "running")], default=str))
    """,
        )
    )
    arms, births, made, refused, why = json.loads(out.splitlines()[-2])
    # 455 and 887 are the INSERT's alone; 892's other line no KeyFrom names, so it is not known to be a birth; the
    # UPDATE, the archive, a line an UPDATE writes too and an unread INSERT stay arms
    assert arms == {"files": [892, 1116], "archive": [50], "mix": [70], "unread": [90], "make": []}
    assert births == [455, 887] and made == [10] and refused == [] and why == "sequence at 623"
    before, by_key_before, after, by_key_after = json.loads(out.splitlines()[-1])
    # a statement whose region holds only a birth gives its owner no arm there: the rows go to what it calls; and a
    # writer of births alone is no writer that takes a row by its key
    assert before == {"p/make.py::make": "make@8 selects the value"} and by_key_before
    assert after == {"p/make.py::helper": "make@8 hands the rows it selects to it"} and not by_key_after


def test_a_statement_no_placed_caller_runs_persists_nothing(tmp_path):
    out = _ok(
        _run(
            tmp_path,
            _MOVERS_MAP + """
    from bamboo.querier import mapdump
    from bamboo.querier.rerun import key_froms

    S = "JobSpec.jobStatus"
    names = ("every", "open", "entry", "some", "bare")
    owners = {n: f"p/{n}.py::{n}" for n in names}
    writers = [writer(owners[n], branch(4016, "activated")) for n in names]
    route = ["p/c.py:814", "p/t.py:815"]

    def records(owner, **fields):
        # a new row's INSERT and the old row's UPDATE, both writing what 4016 set
        birth = N(owner=owner, statement=4177, term="PandaID=:PandaID", writes={S: [4016]}, unread="", sources=[],
                  new_row=True, new_row_by="sequence", new_row_line=4177)
        update = N(owner=owner, statement=4146, term="PandaID=:PandaID", writes={S: [4016]}, unread="", sources=[], **fields)
        return [birth, update]

    mapdump._FRAGMENT[:] = [N(key_froms=[
        *records(owners["every"], routes=3, out_on=[route] * 3, routes_open=False),   # every route hands True
        *records(owners["open"], routes=3, out_on=[route] * 3, routes_open=True),     # a call placed nowhere
        *records(owners["entry"], routes=0, out_on=[], routes_open=True),             # no call placed: reached from outside
        *records(owners["some"], routes=3, out_on=[route] * 2, routes_open=False),    # one route runs it
        *records(owners["bare"]),                                                     # a dump without the fields
    ])]
    first, again = key_froms(), key_froms()
    print(json.dumps([{n: sorted(arms_of(writers, owners[n], S, "failed")) for n in names},
                      [(k.owner.split("::")[-1], k.statement) for k in first], first is again]))
    """,
        )
    )
    arms, read, same = json.loads(out.splitlines()[-1])
    # the UPDATE that no placed caller runs is not read: what 4016 sets reaches only the new row, a birth, so no arm
    assert arms == {"every": [], "open": [4016], "entry": [4016], "some": [4016], "bare": [4016]}
    assert ["every", 4146] not in read and ["every", 4177] in read and len(read) == 9
    # one list per dump, so that the births' index (held by the list's id) is built once
    assert same


def test_the_reach_reads_the_junctions_without_their_births(tmp_path):
    out = _ok(
        _run(
            tmp_path,
            """
            import json
            from types import SimpleNamespace as N
            from bamboo.codemap import reach as RE
            from bamboo.querier import mapdump, rerun

            S = "FileSpec.type"
            def br(line, outcome):
                return N(line=line, outcome=outcome, path_condition=[], row_precondition=[], other_lines=[])
            junctions = [N(owner="p/j.py::update", subject=S, branches=[br(434, "zipinput"), br(500, "output")]),
                         N(owner="p/j.py::insert", subject=S, branches=[br(20, "log")]),
                         N(owner="p/j.py::insert", subject="FileSpec.status", branches=[br(20, "ready")])]
            def kf(owner, statement, line, new_row=None):
                return N(owner=owner, statement=statement, term="row_ID=:row_ID", writes={S: [line], "FileSpec.status": [line]},
                         unread="", sources=[], new_row=new_row)
            mapdump._FRAGMENT[:] = [N(reach_graph=[], junctions=junctions, key_froms=[
                kf("p/j.py::update", 438, 434, True), kf("p/j.py::update", 510, 500), kf("p/j.py::insert", 25, 20, True)])]
            seen = []
            RE.moving_off = lambda js, graph, subject, value: seen.append(
                [(j.owner.split("::")[-1], j.subject, [b.line for b in j.branches]) for j in js]) or RE.Destination({}, [])
            r = object.__new__(rerun.Rerun)
            r.lines, r.result, r._graph_read = [], {}, None
            r._reach = (RE, N(function_at=lambda rel, n: None, named=lambda owner: []), {})
            r._reach_of(rerun.Stuck(S, "input", "1"), "p/j.py::update", [434])
            print(json.dumps([seen, [len(j.branches) for j in junctions]]))
            """,
        )
    )
    seen, untouched = json.loads(out.splitlines()[-1])
    # the reach is handed the subject's junctions less their births; another subject's, and the dump's own, are as read
    assert seen == [[["update", "FileSpec.type", [500]], ["insert", "FileSpec.type", []], ["insert", "FileSpec.status", [20]]]]
    assert untouched == [2, 1, 1]


def test_a_write_that_makes_the_row_is_no_candidate_for_a_status_the_row_had_a_row_before(tmp_path):
    out = _ok(
        _run(
            tmp_path,
            _M2_SETUP + """
    from bamboo.codemap.models import Reading
    from bamboo.querier import held as HD, mapdump

    S = "JobSpec.jobStatus"
    es, act, mix, half = (f"p/{n}.py::{n}" for n in ("es", "act", "mix", "half"))
    def branch(line, other=()):
        return N(line=line, outcome="activated", other_lines=list(other))
    writers = [N(owner=es, branches=[branch(5556), branch(5558)]), N(owner=act, branches=[branch(71)]),
               N(owner=mix, branches=[branch(10), branch(20)]), N(owner=half, branches=[branch(30, [31])])]
    fetched = []

    async def writers_of(subject):
        fetched.append(subject)
        return writers
    r._writers_of = writers_of

    def key(owner, lines, new_row=None):
        return N(owner=owner, statement=lines[0], term="k=:k", writes={S: lines}, unread="", sources=[],
                 new_row=new_row, new_row_by="sequence" if new_row else "", new_row_line=5706 if new_row else None)

    def reading(owner, lines):
        return Reading(owner=owner, file=owner.split("::")[0], lines=lines, log_files=["x.log"])

    readings = [reading(es, [5556, 5558]), reading(act, [71]), reading(mix, [10, 20]), reading(half, [30])]
    rows = [{"status": "pending", "time": "2026-01-01 00:00:00", "time_extended": "2026-01-01 00:00:00.2"},
            {"status": "defined", "time": "2026-01-01 00:00:00", "time_extended": "2026-01-01 00:00:00.7"},
            {"status": "activated", "time": "2026-01-01 00:00:04", "time_extended": "2026-01-01 00:00:04.1"}]
    r.data["statuslog"] = {"states": rows}
    def unborn(state):
        kept, said = r._unborn(readings, S, state)
        return [[x.owner.split("::")[-1], x.lines] for x in kept], said

    no_marks, fetched_before = unborn(rows[2]), list(fetched)
    mapdump._FRAGMENT[:] = [N(key_froms=[
        key(es, [5556, 5558], new_row=True), key(act, [71]), key(mix, [10], new_row=True), key(mix, [20]),
        key(half, [30], new_row=True)])]   # half's 31, another line of the branch, no KeyFrom names
    later, first = unborn(rows[2]), unborn(rows[0])
    handed = unborn({"status": "activated", "time": "2026-01-01 00:00:00"})   # a tie at the second: not before
    print(json.dumps([no_marks, fetched_before, later, first, handed, unborn(None),
                      HD.row_before(rows, rows[1]), HD.row_before(rows[:1], rows[0])]))
    """,
        )
    )
    no_marks, fetched, later, first, handed, no_state, defined_after_pending, alone = json.loads(out.splitlines()[-1])
    every = [["es", [5556, 5558]], ["act", [71]], ["mix", [10, 20]], ["half", [30]]]
    # with no birth marked the map's writers are not read, and every reading stays
    assert no_marks == [every, ""] and fetched == []
    # 'activated' after 'pending' and 'defined': the row was there, so the new jobs' INSERTs are no candidates; a
    # function keeps its other lines; a branch whose other line is not a birth stays whole; an UPDATE stays
    kept, said = later
    assert kept == [["act", [71]], ["mix", [20]], ["half", [30]]]
    assert said == ("the key's row was there before 'activated' at 2026-01-01 00:00:04, so its writes that make a row are "
                    "no candidates: es [5556, 5558] (births: sequence at 5706); mix [10] (births: sequence at 5706) "
                    "(KNOWN_ISSUES.md K29)")
    # the row's first status, a tie at the second the log keeps, and no state: the birth may be the row's
    assert first == [every, ""] and handed == [every, ""] and no_state == [every, ""]
    assert defined_after_pending is True and alone is False


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
    # judgement 3: these writers' runs are not known to print the key, so their logs are not asked
    rerun.prints_nothing = lambda owner, lines: False
    rerun.prints_naming = lambda writers, owner, lines, columns: {"held": None, "why": "not known"}
    rerun.every_log_asked = lambda owner, lines, asked: {"held": True, "why": "every file"}
    r.files_of = lambda owner, today, lines: today
    r.logs = N(names=lambda files: list(files), needs_window=True)
    verdicts = r._movers_in_window(q, {("d/a.py::main", 10): "refused", ("d/a.py::main", 20): "not settled",
                                      ("d/a.py::main", 30): "not settled"})
    print(json.dumps([verdicts, r.lines[-1], r.result["movers"]], default=str))
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
    assert by_owner["d/a.py::dark"]["reasons"][1]["record"].startswith("main@30: JobSpec.modificationTime not compared (written at")
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
    assert by_owner["p/w.py::armed"]["premises"]["window"] == {"held": None, "why": "not asked"}
    assert by_owner["p/w.py::armed"]["reasons"][0]["record"] == (
        "its logs are not asked: a run not found would not rule it out (prints: not known) (KNOWN_ISSUES.md K15)")


def test_l2_asks_a_movers_logs_only_where_judgement_3_could_rule_it_out_and_goes_on_past_unanswered_asks(tmp_path):
    out = _ok(
        _run(
            tmp_path,
            _M2_SETUP + """
    def writer(owner):
        return N(owner=owner, log_files=[f"{owner.split('::')[1]}.log"],
                 branches=[N(line=5, outcome="sent", path_condition=[], row_precondition=[])])

    q.by_key = [f"p/w.py::{n}" for n in ["unknown", "silent", "printless", "ran", "unanswered", "holes"]]
    # a writer reached by a guard, not by its key: the part before the last is asked too
    q.movers = {"p/w.py::selected": "its write requires the value"}
    q.writers, q.mover = [writer(o) for o in [*q.movers, *q.by_key]], "p/w.py::moved"
    # a stay of three hours: the window's last part is the whole of it
    q.since, q.moved = datetime.datetime(2026, 1, 1, 12), datetime.datetime(2026, 1, 1, 15)
    rerun.prints_nothing = lambda owner, lines: owner.endswith("printless")
    rerun.prints_naming = lambda writers, owner, lines, columns: (
        {"held": None, "why": "not known"} if owner.endswith("unknown") else {"held": True, "why": "named"})
    rerun.every_log_asked = lambda owner, lines, asked: (
        {"held": False, "why": "the map has no prints of its own for it"} if owner.endswith("printless")
        else {"held": True, "why": "every file"})
    rerun.owners_of = lambda ids: {}
    r.files_of, r.warn_split = (lambda owner, today, lines: today), (lambda owner, lines: None)
    r.logs, r.unanswered, asked = N(names=lambda files: list(files), needs_window=True), [], []

    def ask(key, logs, asking, owners=None):
        asked.append([logs[0], list(asking.window)])
        if logs[0] == "unanswered.log":
            return {}, [(N(key="k1"), "why")], {}
        if logs[0] == "holes.log":
            return {}, [], {"holes.log": ["m1: 2026-01-01 13:00:00 .. 2026-01-01 14:00:00"]}
        return {logs[0]: [N(first="2026-01-01 13:00:00", last="2026-01-01 13:00:01", token="")] if logs[0] == "ran.log" else []}, [], {}

    r._ask = ask
    verdicts = dict(map(tuple, r._movers_in_window(q, {})))
    movers = {m["owner"].split("::")[1]: m for m in r.result["movers"]}
    lines, r.lines, kept = r.lines, [], len(r.unanswered)
    # a stay of 51 hours: a writer reached by its key is asked over the last six alone
    q.since = datetime.datetime(2025, 12, 30, 12)
    long = dict(map(tuple, r._movers_in_window(q, {})))
    silent = next(m for m in r.result["movers"] if m["owner"].endswith("silent"))
    print(json.dumps([verdicts, {n: [m["premises"], [x["code"] for x in m["reasons"]]] for n, m in movers.items()},
                      asked, kept, lines, long, silent["premises"]["window"], r.lines], default=str))
    """,
        )
    )
    verdicts, movers, asked, unanswered, lines, long, window, long_lines = json.loads(out.splitlines()[-1])
    assert verdicts == {"p/w.py::selected": "did not run", "p/w.py::unknown": "by key", "p/w.py::silent": "did not run",
                        "p/w.py::printless": "prints nothing",
                        "p/w.py::ran": "ran", "p/w.py::unanswered": "asks unanswered", "p/w.py::holes": "not searched",
                        "p/w.py::moved": "moved it"}
    # a premise the map decides not held: not asked, and said so
    assert movers["unknown"] == [{"window": {"held": None, "why": "not asked"}, "logs": {"held": True, "why": "every file"},
                                  "prints": {"held": None, "why": "not known"}}, ["K15"]]
    # asked over the stay, from its row less 10 min to the move, one ask per log; the others not asked
    stay = ["2026-01-01 11:50:00", "2026-01-01 15:00:00"]
    assert asked[:5] == [["selected.log", stay], ["silent.log", stay], ["ran.log", stay], ["unanswered.log", stay],
                         ["holes.log", stay]]
    assert movers["silent"][0]["window"] == {"held": True, "why": "2026-01-01 11:50:00 .. 2026-01-01 15:00:00 searched in "
                                                                   "['silent.log'], none of it unsearched"}
    assert movers["silent"][1] == ["judgement-3"] and movers["printless"] == [None, ["K25"]]
    assert movers["ran"] == [None, ["K15"]] and movers["unanswered"][1] == ["asks"]
    assert movers["unanswered"][0]["window"] == {"held": None, "why": "1 ask(s) not answered yet"}
    assert movers["holes"][0]["window"]["held"] is False
    # the ask no record answers is kept for the asks file, and the walk goes on
    assert unanswered == 1
    assert any(x.startswith("  w.py::unanswered: asks unanswered -- a writer [5], which takes a row by its key too; 1 ask(s) "
                            "of its logs not in the records ('activated' since 2026-01-01 12:00:00; from its row less 10 min "
                            "to the move 2026-01-01 15:00:00): whether it ran is not known yet") for x in lines)
    assert any(x.startswith("  w.py::ran: ran -- a writer [5], which takes a row by its key too; ran: a run of the key in "
                            "['ran.log'] between 2026-01-01 11:50:00 and 2026-01-01 15:00:00, and L1 names another mover")
               for x in lines)
    # did not run is ruled out; prints nothing, ran, unanswered and not searched are not
    assert lines[-1] == "not ruled out: 5 mover(s)"
    # the long stay: the last six hours alone, so a run not found there rules nothing out
    last = ["2026-01-01 09:00:00", "2026-01-01 15:00:00"]
    assert [a for a in asked[5:] if a[0] == "silent.log"] == [["silent.log", last]]
    assert long["p/w.py::silent"] == "no run found" and long["p/w.py::ran"] == "ran"
    # the writer reached by a guard: the rest asked where the last six hours hold no run, so it did not run
    assert [a for a in asked[5:] if a[0] == "selected.log"] == [["selected.log", last],
                                                                 ["selected.log", ["2025-12-30 11:50:00", "2026-01-01 09:00:00"]]]
    assert long["p/w.py::selected"] == "did not run"
    assert window == {"held": False, "why": "searched from 2026-01-01 09:00:00 only"}
    assert "  silent: no run of its own found in ['silent.log'] from 2026-01-01 09:00:00, the window's last part, which alone " \
           "is asked of a writer that takes a row by its key (KNOWN_ISSUES.md K16): not followed, though not ruled out " \
           "(KNOWN_ISSUES.md K19)" in long_lines


def test_a_class_a_statement_compares_and_the_rows_file_lacks_is_asked_by_the_statements_join_to_the_row(tmp_path):
    out = _ok(
        _run(
            tmp_path,
            _M2_SETUP + """
    from bamboo.querier import mapdump
    # the joins are the dump's read site's, by the map's name of it; the follow-up's statement carries none
    sites = []
    mapdump._FRAGMENT[:] = [N(tables=[N(spec_class="JobSpec", identity_attrs=["JobSpec.PandaID"])], read_sites=sites)]
    r.rows_asks, r.hop = {}, "M2"
    r.data["rows"]["job"]["jediTaskID"] = {"value": 5}
    r.data["classes"]["task"], r.data["rows"]["task"] = "JediTaskSpec", {"jediTaskID": {"value": 5}}
    ftype = term("equal", "FileSpec.type", "IN", [val("input")])

    def pair(a, b, a_source="f", b_source="j"):
        return N(a_attr=a, b_attr=b, a_source=a_source, b_source=b_source)

    def files(line, joins, routes=(), trees=(ftype,)):
        if joins is not None:
            sites.append(N(name=f"panda:d/a.py::main@{line}", joins=joins))
        return N(owner="d/a.py::main", line=line, keys=[], selects=[], excludes=[], refuses=[], restricts=[],
                 conditions=[], condition_trees=[N(entry="e", tree=t) for t in trees], call_readings=list(routes),
                 callers_complete=True, callers_stop=[])

    by_job = [pair("FileSpec.PandaID", "JobSpec.PandaID"), pair("FileSpec.jediTaskID", "JobSpec.jediTaskID")]
    said = {}
    for name, st in {
        # one key of two columns, the row's values; a second alias of the table, a second key
        "joined": files(10, by_job + [pair("JobSpec.PandaID", "FileSpec.PandaID", "j", "g")]),
        # the same rows asked again: once
        "again": files(11, by_job),
        # every route refuses the row: nothing the rows could change (rows no other statement asks)
        "refused": files(12, by_job[1:], routes=[N(**vars(route([share])), constants={}, chain=["d/b.py:3"])]),
        # through another table only (the task's, whose row the file holds: not the row asked), a class-less table, a
        # column the row does not hold
        "through": files(13, [pair("JediFileSpec.jediTaskID", "JediTaskSpec.jediTaskID")], trees=[term("equal", "JediFileSpec.status", "=", [val("ready")])]),
        "class-less": files(14, [pair("", "JobSpec.PandaID", "t", "j")], trees=[term("equal", "T_TASK.total_done_jobs", "=", [val("0")])]),
        "unheld": files(15, [pair("FileSpec.dispatchDBlock", "JobSpec.dispatchDBlock")]),
        # no read site of its name in the dump: said, not asked
        "no site": files(18, None),
    }.items():
        before = len(r.rows_asks)
        kind, lines = r._passed_over(st, q)
        said[name] = [kind, len(r.rows_asks) - before, [x for x in lines if x.startswith(("not compared", "the dump has"))]]
    asked = list(r.rows_asks.values())
    # rows the file's linked holds by a pair the statement joins on, for the row's value: said, not asked; another
    # pair, or another value, is asked
    r.data["linked"] = [{"class": "FileSpec", "via": {"from": "JobSpec.PandaID", "to": "FileSpec.PandaID"}, "asked": [1], "rows": []},
                        {"class": "FileSpec", "via": {"from": "JobSpec.jediTaskID", "to": "FileSpec.jediTaskID"}, "asked": [6], "rows": []}]
    r.rows_asks = {}
    held = [r._passed_over(files(19, by_job[:1]), q)[1][-1], len(r.rows_asks), r._passed_over(files(20, by_job[1:]), q)[1][-1], len(r.rows_asks)]
    r.data.pop("linked")
    # L2's movers: a statement it did not judge asks too, one it judged refused does not
    r.rows_asks = {}
    q.movers, q.by_key, q.refusing, q.mover, q.writers = {"d/a.py::open": "main@16 hands the rows it selects to it"}, [], [], "p/w.py::moved", []
    q.paths = {"d/a.py::open": [files(16, by_job[:1]), files(17, [pair("FileSpec.fileID", "JobSpec.PandaID")])]}
    r._movers_in_window(q, {("d/a.py::main", 17): "refused"})
    print(json.dumps([said, asked, [a["rows"]["where"] for a in r.rows_asks.values()], held]))
    """,
        )
    )
    said, asked, by_l2, held = json.loads(out.splitlines()[-1])
    assert by_l2 == [{"PandaID": 1}]
    assert held == ["the rows file's linked holds FileSpec's rows by JobSpec.PandaID -> FileSpec.PandaID, which this step does not "
                    "compare (KNOWN_ISSUES.md K36)", 0, "not compared: FileSpec.type (no row of FileSpec)", 1]
    assert said["joined"] == ["not settled", 2, ["not compared: FileSpec.type (no row of FileSpec)"]]
    assert said["again"][:2] == ["not settled", 0]
    assert said["refused"][:2] == ["refused", 0] and said["refused"][2] == said["joined"][2]
    assert [said[n][1] for n in ("through", "class-less", "unheld")] == [0, 0, 0]
    assert said["through"][2] == ["not compared: JediFileSpec.status (no row of JediFileSpec)"]
    assert said["no site"] == ["not settled", 0, [said["joined"][2][0], "the dump has no read site panda:d/a.py::main@18: its joins "
                                                  "are not read, and the rows of FileSpec.type not asked"]]
    why = "main@10 compares FileSpec's type with the row's, and neither the rows file's rows nor its linked hold FileSpec's"
    row = {"entity": "JobSpec", "key": {"PandaID": 1}}
    by_id = {"from": "JobSpec.PandaID", "to": "FileSpec.PandaID"}
    assert asked == [
        {"rows": {"kind": "collection", "row": row, "attr": None, "entity": "FileSpec", "where": {"PandaID": 1, "jediTaskID": 5},
                  "via": [by_id, {"from": "JobSpec.jediTaskID", "to": "FileSpec.jediTaskID"}]}, "step": "M2", "why": why},
        {"rows": {"kind": "collection", "row": row, "attr": None, "entity": "FileSpec", "where": {"PandaID": 1}, "via": [by_id]},
         "step": "M2", "why": why},
    ]


def test_the_window_end_reads_after_only_at_as_of(tmp_path):
    out = _ok(
        _run(
            tmp_path,
            _M2_SETUP + """
    at = datetime.datetime(2026, 1, 3, 12)
    stuck = rerun.Missed("JobSpec.jobStatus", "activated", "", at, at, [], "", None, at_end=True)
    print(json.dumps([r._held("JobSpec.modificationTime", q), r._held("JobSpec.modificationTime", stuck),
                      r._held("JobSpec.source", stuck), r._unanswered("JobSpec.modificationTime", q),
                      r._unanswered("JobSpec.modificationTime", stuck)], default=str))
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
    print(json.dumps(said, default=str))
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
    rerun.owners_of = lambda ids: {}
    answered = False

    def ask(key, names, asking, owners=None, whole=False):
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


def test_a_frame_of_another_version_is_read_at_the_one_line_of_its_function_holding_its_source(tmp_path):
    (_corpus(tmp_path) / "pandaserver" / "b.py").write_text(
        "def f(x):\n    a = 1\n    y = h(x)\n    if x:\n        return None\n    return None\n\n\ndef g():\n    y = h(x)\n"
    )
    out = _ok(
        _run(
            tmp_path,
            """
            import json
            from bamboo.querier.records import _chains

            def frame(line, source):
                tb = ["Traceback (most recent call last):",
                      f'  File "/opt/panda/lib/python3.11/site-packages/pandaserver/b.py", line {line}, in f',
                      f"    {source}", "ValueError: no"]
                f, = _chains(tb)[0].frames
                return [f.line, f.same_source, f.moved_from, f.printed_at]

            print(json.dumps({"moved": frame(7, "y = h(x)"), "same": frame(3, "y = h(x)"),
                              "twice": frame(9, "return None"), "absent": frame(2, "z = 3")}))
            """,
        )
    )
    got = json.loads(out.splitlines()[-1])
    # the run's code had the line at 7: f holds it once, at 3 (g's own is another function's)
    assert got["moved"] == [3, True, 7, []]
    assert got["same"] == [3, True, None, []]
    # printed twice in f, or nowhere: not placed, as before, with the lines of f that hold it
    assert got["twice"] == [9, False, None, [5, 6]] and got["absent"] == [2, False, None, []]


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


def test_the_run_holding_most_rows_reads_each_run_once_and_each_rows_patterns_once(tmp_path):
    (_corpus(tmp_path) / "pandaserver" / "a.py").write_text(textwrap.dedent("""
        def f(self, job):
            self.logger.debug("starting the update of the job")
            self.logger.debug("the update of the job was handed")
            self.logger.debug("finished the update of the job")
        """))
    stamp = "2026-01-01 00:00:0{},000 panda.log.X: DEBUG    <PandaID=1> {}\n"
    (tmp_path / "a.log").write_text(stamp.format(0, "starting the update of the job"))
    (tmp_path / "b.log").write_text(
        stamp.format(0, "starting the update of the job") + stamp.format(1, "the update of the job was handed")
        + stamp.format(2, "finished the update of the job")
    )
    out = _ok(
        _run(
            tmp_path,
            """
            import json
            from bamboo.querier import rows as R
            from bamboo.querier.records import read_log, runs_for
            from bamboo.querier.which_ran import Candidate, run_holding

            R.ROWS_FROM = "walk"
            fn = R.function("pandaserver/a.py", "f", (3,))
            found = [run for name in ("a", "b") for run in runs_for(read_log(f"{name}.log"), "1")]
            calls = {"seen": 0, "regex": 0}
            seen, regex = R.seen, R._regex
            def counted(name, f):
                def call(*a, **k):
                    calls[name] += 1
                    return f(*a, **k)
                return call
            R.seen, R._regex = counted("seen", seen), counted("regex", regex)
            held = run_holding(Candidate("pandaserver/a.py", "f", [3], ["a.log"]), found, "outside")
            row = fn.rows[0]
            again = row.patterns("outside") is row.patterns("outside")
            modes = [len(row.patterns(mode)) for mode in ("map", "outside")]
            print(json.dumps([held.log.name, calls, len(fn.rows), again, modes]))
            """,
        )
    )
    # the run where most rows are proven; each run read once, not once per row, and each row's patterns made once,
    # not once per record it is asked of -- per mode: the walk's pattern alone, and with the shape built here
    assert json.loads(out.splitlines()[-1]) == ["b.log", {"seen": 2, "regex": 3}, 3, True, [1, 2]]


def test_runs_of_a_key_try_its_pattern_only_on_a_message_or_mark_holding_the_keys_text(tmp_path):
    log = tmp_path / "panda-x.log"
    stamp = "2026-01-01 00:00:0{},000 panda.log.X: DEBUG    <PandaID={}> {}\n"
    log.write_text(
        stamp.format(0, 7, "starting") + stamp.format(1, 8, "other") + stamp.format(2, 8, "about 7 too")
        + stamp.format(3, 8, "no 17 here")
    )
    out = _ok(
        _run(
            tmp_path,
            f"""
            import json
            from bamboo.querier import records as Rec
            from bamboo.querier.records import read_log, runs_for

            searched, keyed = [], Rec.keyed
            class Counted:
                def __init__(self, pattern):
                    self.pattern = pattern
                def search(self, text):
                    searched.append(text)
                    return self.pattern.search(text)
            Rec.keyed = lambda key: Counted(keyed(key))
            runs = runs_for(read_log({str(log)!r}), "7")
            print(json.dumps([[(r.token, [x.lineno for x in r.records], [x.lineno for x in r.found_by]) for r in runs], searched]))
            """,
        )
    )
    runs, searched = json.loads(out.splitlines()[-1])
    # the same runs as the pattern alone finds (17 is not the key 7), the pattern tried only where "7" is written
    assert runs == [["<PandaID=7>", [1], [1]], ["<PandaID=8>", [2, 3, 4], [3]]]
    assert searched == ["<PandaID=7>", "about 7 too", "no 17 here"]


def test_the_statement_at_a_line_is_the_innermost_the_first_of_equal_depth_and_the_function_is_walked_once(tmp_path):
    out = _ok(
        _run(
            tmp_path,
            """
            import ast, json
            from bamboo.codemap.panda import pathcond
            from bamboo.querier import rows as R

            fn = ast.parse("def f(x):\\n    a = 1; b = 2\\n    if x: y = 3\\n    return a\\n").body[0]
            pathcond.attach_parents(fn)
            walks, own_nodes = [0], R.own_nodes
            def counted(root):
                walks[0] += 1
                return own_nodes(root)
            R.own_nodes = counted
            got = [R.statement_at(fn, line) for line in (1, 2, 3, 4, 5, 3, 2)]
            print(json.dumps([[None if s is None else ast.unparse(s) for s in got], walks[0]]))
            """,
        )
    )
    # not the def itself; of two on a line, the one own_nodes meets first (its stack takes the later); the if's body,
    # not the if; nothing where none starts -- and one walk of the function for every line asked
    assert json.loads(out.splitlines()[-1]) == [[None, "b = 2", "y = 3", "return a", None, "y = 3", "b = 2"], 1]


# activateJob, cut down: f prints its files' UPDATE alike in both arms (92 and 124); g's other arm prints another text
_ALIKE = """
def f(self, job, ok):
    self.logger.debug("activating the job now")
    if ok:
        job.jobStatus = "activated"
        self.logger.debug(f"UPDATE ATLAS_PANDA.filesTable4 SET {job.changes} WHERE row_ID=:row_ID")
    else:
        job.modificationTime = 1
        self.logger.debug(f"UPDATE ATLAS_PANDA.filesTable4 SET {job.changes} WHERE row_ID=:row_ID")
    self.logger.debug("finished activating the job")


def g(self, job, ok):
    self.logger.debug("activating the job now")
    if ok:
        job.jobStatus = "activated"
        self.logger.debug(f"UPDATE ATLAS_PANDA.filesTable4 SET {job.changes} WHERE row_ID=:row_ID")
    else:
        job.modificationTime = 1
        self.logger.debug(f"UPDATE ATLAS_PANDA.jobsDefined4 SET {job.changes} WHERE PandaID=:PandaID")
    self.logger.debug("finished activating the job")
"""


def test_a_row_proves_its_arm_only_by_a_record_no_other_row_of_its_function_prints_alike(tmp_path):
    (_corpus(tmp_path) / "pandaserver" / "a.py").write_text(_ALIKE)
    stamp = "2026-01-01 00:00:0{},000 panda.log.X: DEBUG    <PandaID=1> {}\n"
    (tmp_path / "x.log").write_text(
        stamp.format(0, "activating the job now")
        + stamp.format(1, "UPDATE ATLAS_PANDA.filesTable4 SET modificationTime=:modificationTime WHERE row_ID=:row_ID")
        + stamp.format(2, "finished activating the job")
    )
    out = _ok(
        _run(
            tmp_path,
            """
            import json
            from bamboo.querier import rows as R
            from bamboo.querier.records import read_log, runs_for
            from bamboo.querier.which_ran import Candidate, judge

            R.ROWS_FROM = "walk"
            run, = runs_for(read_log("x.log"), "1")
            got = {}
            for func, line in (("f", 5), ("g", 16)):
                c = Candidate("pandaserver/a.py", func, [line], [run.log.name], run=run)
                v, = judge([c], {run.log.name: [run]}, question_run=run).verdicts
                got[func] = [v.kind, v.reasons]
            print(json.dumps(got))
            """,
        )
    )
    got = json.loads(out.splitlines()[-1])
    # f's other arm prints the same text: the record is either arm's, and proves neither
    assert got["f"][0] != "positive", got["f"]
    assert not any("print only on a path through it" in r for r in got["f"][1])
    # g's other arm prints another text: the record is the arm's own
    assert got["g"] == ["positive", ["row(s) [17] print only on a path through it"]]


_SIBLINGS = """
class Anal:
    def doRefine(self, tmpLog):
        tmpLog.debug(f"start taskType={self.taskType}")
        self.reqID = 1
        tmpLog.debug("done")

    def post(self, tmpLog, status):
        self.endTime = 1
        tmpLog.debug(f"post done with status={status}")


class Prod:
    def doRefine(self, tmpLog):
        tmpLog.debug(f"start taskType={self.taskType}")
        tmpLog.debug("done")
"""


def test_a_record_another_function_prints_whole_proves_neither_and_one_it_prints_a_part_of_still_does(tmp_path):
    (_corpus(tmp_path) / "pandaserver" / "a.py").write_text(_SIBLINGS)
    stamp = "2026-01-01 00:00:0{},000 panda.log.X: DEBUG    < jediTaskID=1 > {}\n"
    (tmp_path / "x.log").write_text(
        "".join(stamp.format(i, m) for i, m in enumerate(("start taskType=prod", "done", "post done with status=finished")))
    )
    out = _ok(
        _run(
            tmp_path,
            """
            import json
            from types import SimpleNamespace as N
            from bamboo.codemap.models import PrintRow
            from bamboo.querier import mapdump, rows as R
            from bamboo.querier.records import read_log, runs_for
            from bamboo.querier.which_ran import rows_around

            R.ROWS_FROM = "walk"
            run, = runs_for(read_log("x.log"), "1")
            mark, cap = "<\\\\ jediTaskID=.*?\\\\ >\\\\ ", "<\\\\ jediTaskID=(.*?)\\\\ >"
            row = lambda pattern, capture, prefix=mark, prefix_capture=cap, hole="": PrintRow(
                text=pattern, pattern=pattern, capture=capture, prefix=prefix, prefix_capture=prefix_capture, prefix_hole=hole)
            prints = lambda owner, *rs: N(
                owner=owner, statements=[N(rows=[r], log_files=[run.log.name], log_files_basis="logger") for r in rs])
            start = row("start\\\\ taskType=", "start\\\\ taskType=(.*)")
            anal = prints("pandaserver/a.py::Anal.doRefine", start, row("done", "done"))
            post = prints("pandaserver/a.py::Anal.post", row("post\\\\ done\\\\ with\\\\ status=", "post\\\\ done\\\\ with\\\\ status=(.*)"))
            # AtlasProd's doRefine: the same two lines, under the same mark, in the same log
            prod = prints("pandaserver/a.py::Prod.doRefine", start, row("done", "done"))
            # PostProcessor's: "done" alone, which "post done with ..." holds but is not
            exactly_done = prints("pandaserver/b.py::B.start", row("done", "done"))
            # a logger whose mark the map cannot read: the row cannot say it reads a line whole
            unread_mark = prints("pandaserver/b.py::B.dump", row("post\\\\ done", "post\\\\ done(.*)", "", "", "made with the time"))

            def placed(func, arm, *others):
                fragment = N(function_prints=[anal, post, *others])
                mapdump.fragment = lambda: fragment
                fn = R.function("pandaserver/a.py", func, [arm])
                sn = R.seen(fn, run, "outside")
                return [sorted(sn.unambiguous), rows_around(fn, sn, R.statement_at(fn.node, arm)) is not None]

            print(json.dumps({
                "its own": placed("doRefine", 5),
                "a sibling prints both whole": placed("doRefine", 5, prod),
                "another prints a part of it": placed("post", 9, exactly_done),
                "another cannot say": placed("post", 9, unread_mark),
            }))
            """,
        )
    )
    got = json.loads(out.splitlines()[-1])
    assert got == {
        # the two records around the store are its function's alone: they show it ran
        "its own": [[4, 6], True],
        # AtlasAnal's doRefine:260 in MR-T-53344406-reqID, whose run printed "refining with AtlasProdTaskRefiner"
        "a sibling prints both whole": [[], False],
        "another prints a part of it": [[10], False],
        "another cannot say": [[10], False],
    }


_FALLS_OUT = '''
def adder(self, a):
    if a:
        try:
            log.debug("will call")
            call()
            log.debug("is back")
        except Exception:
            log.error("excepted")
    self.status = "failed"


def top(self):
    log.debug("at the top")
    self.status = "failed"


def loop(self, z):
    for y in z:
        log.debug("in a loop")
    self.status = "failed"


def final(self):
    try:
        log.debug("in a try")
    finally:
        log.debug("in a finally")
    self.status = "failed"


def a_call_after(self, a):
    if a:
        log.debug("before a call")
        call()
    self.status = "failed"


def an_else(self):
    try:
        log.debug("before an else")
    except Exception:
        pass
    else:
        call()
    self.status = "failed"


def handled(self):
    try:
        call()
    except Exception:
        log.debug("handled")
    finally:
        cleanup()
    self.status = "failed"


def within(self):
    with lock:
        log.debug("in a with")
    self.status = "failed"
'''


def test_a_print_nested_in_an_earlier_statement_implies_the_site_when_nothing_on_its_way_out_may_leave(tmp_path):
    (tmp_path / "f.py").write_text(_FALLS_OUT)
    out = _ok(
        _run(
            tmp_path,
            """
            import ast, json
            from bamboo.codemap.panda import pathcond
            from bamboo.querier import rows as R

            tree = ast.parse(open("f.py").read())
            pathcond.attach_parents(tree)
            print(json.dumps({fn.name: sorted(s.value.args[0].value for s in R.implying_rows(fn, fn.body[-1])) for fn in tree.body}))
            """,
        )
    )
    assert json.loads(out.splitlines()[-1]) == {
        # handle_failed_job 374: "... is back" and the handler's print fall out of the if; the call after "will call"
        # may leave
        "adder": ["excepted", "is back"],
        "top": ["at the top"],
        "loop": [],
        "final": ["in a try"],
        "a_call_after": [],
        "an_else": [],
        "handled": [],
        "within": [],
    }


_NARROWED = """
def f(self, job, ok):
    self.logger.debug("start f")
    if ok:
        job.jobStatus = "activated"
        self.logger.debug("activated it in f")
    self.logger.debug("end f")


def g(self, job, ok):
    self.logger.debug("start g")
    if ok:
        job.jobStatus = "activated"
        self.logger.debug("activated it in g")
    self.logger.debug("end g")
"""


def test_rule_8_reads_a_row_of_another_function_than_the_one_a_run_was_asked_for_as_not_searched(tmp_path):
    (_corpus(tmp_path) / "pandaserver" / "a.py").write_text(_NARROWED)
    stamp = "2026-01-01 00:00:0{},000 panda.log.X: DEBUG    <PandaID=1> {}\n"
    (tmp_path / "x.log").write_text("".join(stamp.format(i, m) for i, m in enumerate(("start f", "end f", "start g", "end g"))))
    out = _ok(
        _run(
            tmp_path,
            """
            import json
            from bamboo.querier import records
            from bamboo.querier import rows as R
            from bamboo.querier.records import read_log, runs_for
            from bamboo.querier.which_ran import Candidate, judge

            records.level_filters = lambda: []
            R.ROWS_FROM = "walk"
            got = {}
            for label, narrowed in (("whole", None), ("f's rows", "f")):
                run, = runs_for(read_log("x.log"), "1")
                if narrowed:
                    run.log.narrowed = {(run.machine, run.token): narrowed}
                for func, line in (("f", 5), ("g", 13)):
                    c = Candidate("pandaserver/a.py", func, [line], [run.log.name], run=run)
                    v, = judge([c], {run.log.name: [run]}, question_run=run).verdicts
                    got[f"{label} {func}"] = v.kind
            print(json.dumps(got))
            """,
        )
    )
    got = json.loads(out.splitlines()[-1])
    # a run asked whole: each arm's row missing is evidence
    assert got["whole f"] == got["whole g"] == "excluded"
    # a run asked for f's rows alone: f's missing row still is; g's below WARNING was not searched
    assert got["f's rows f"] == "excluded" and got["f's rows g"] != "excluded"


def test_h2_reads_a_window_in_one_run_only_where_it_is_the_one_run_of_one_log_at_it(tmp_path):
    out = _ok(
        _run(
            tmp_path,
            """
            import json
            from bamboo.querier.rerun import _question_run

            two_day, own, other = "prepareTasksToBeFinished_JEDI", "updateTask_JEDI", "panda-other"
            print(json.dumps([
                _question_run({"panda-JediDBProxy.log": [own]}),
                _question_run({"panda-JediDBProxy.log": [two_day, own]}),
                _question_run({"panda-JediDBProxy.log": [own], "panda-DBProxy.log": [other]}),
                _question_run({}),
            ]))
            """,
        )
    )
    # the row's second is no exception: a mark spanning days covers it beside the pass that wrote
    assert json.loads(out.splitlines()[-1]) == ["updateTask_JEDI", None, None, None]


def test_s3_reads_an_early_exit_above_an_arm_as_its_guard_and_says_an_arm_in_an_except_handler(tmp_path):
    out = _ok(
        _run(
            tmp_path,
            """
            import ast, json, textwrap
            from types import SimpleNamespace as N
            from bamboo.codemap.panda import pathcond
            from bamboo.codemap.panda.pathcond import attach_parents
            from bamboo.querier import evaluate as E
            from bamboo.querier import rerun as RR
            from bamboo.querier.rerun import Arm, arm_guards, early_exit_if, guard_if

            # check_output_file_metadata's shape: an early return, then the arms; and one in a handler
            func = ast.parse(textwrap.dedent('''
                def check(self, files):
                    if self.job_status in ["failed", "cancelled", "closed"]:
                        return 0
                    for f in files:
                        if f.type == "output":
                            self.job.jobStatus = "failed"
                    try:
                        x = 1
                    except Exception:
                        self.job.jobStatus = "failed"
            ''')).body[0]
            attach_parents(func)  # as the corpus is read
            stmt, in_handler = (next(n for n in ast.walk(func) if isinstance(n, ast.Assign) and n.lineno == line) for line in (7, 11))
            # each arm's guards as the map's branches hold them: its ifs' tests, then the early exits above it
            after, handled = Arm(7, ["failed"], pathcond.arm_condition(stmt)), Arm(11, ["failed"], pathcond.arm_condition(in_handler))
            (g1, h1), (g2, h2) = arm_guards(func, after), arm_guards(func, handled)
            guard = g1[1]
            read = [g1, [u.kind for u in h1], g2, [(u.kind, u.line) for u in h2],
                    guard_if(stmt, guard) is None, early_exit_if(func, stmt, guard).lineno]

            # S3 itself (it says its lines on stdout): a pass that left by the return at 4, and a function whose only arm is in a handler
            lone = ast.parse(textwrap.dedent('''
                def lone(self):
                    try:
                        x = 1
                    except Exception:
                        self.job.jobStatus = "failed"
                    return 0
            ''')).body[0]
            attach_parents(lone)
            r = RR.Rerun.__new__(RR.Rerun)
            r.lines, r.opened, r.hand, r.hand_used, r.hop = [], [], {}, [], None
            went = r.blocked(RR.Blocked(N(node=func, name="check"), 4, [None], [E.Env()], [after, handled]))
            try:
                r.blocked(RR.Blocked(N(node=lone, name="lone"), 7, [None], [E.Env()], [Arm(6, ["failed"], [])]))
                stop = None
            except RR.Stop as e:
                stop = e.args[0]
            print(json.dumps(read))
            print(json.dumps([went.guard, went.line, stop]))
            """,
        )
    )
    g1, h1, g2, h2, not_enclosing, early_line = json.loads(out.splitlines()[-2])
    went_guard, went_line, stop = json.loads(out.splitlines()[-1])
    # the branch's guards as they are: the path condition's, then the early exit's (what follows the return holds its
    # test false), each once (KNOWN_ISSUES.md K61)
    assert g1 == ["f.type == 'output'", "not (self.job_status in ['failed', 'cancelled', 'closed'])"] and h1 == []
    # an arm in a handler: the early exit above the try, and the handler itself, said apart (no guard says it)
    assert g2 == ["not (self.job_status in ['failed', 'cancelled', 'closed'])"] and h2 == [["except", 10]]
    # S3 goes on from the early exit's if, which no if around the arm tests, found by the line the map gives it
    assert not_enclosing and early_line == 3
    # the exit taken settles the early exit's test true, which keeps both arms off: S3 names it and goes on from 3
    assert went_guard == "not (self.job_status in ['failed', 'cancelled', 'closed'])" and went_line == 3
    # an arm in a handler with nothing shown false stops, saying where it is, not that its guard was made false
    assert stop == "S3: a guard the passes made false against every arm (none shown against arm 6 (in an except handler))"


_RETRY = textwrap.dedent("""
    def f(self, job):
        tmp_log.debug("start")
        for iTry in range(3):
            try:
                res = self.cur.fetchall()
                if len(res) == 0:
                    job.jobStatus = "activated"
                    tmp_log.debug("activated the job")
                else:
                    tmp_log.debug("kept the job defined")
                    self.cur.execute(sql, varMap)
                tmp_log.debug("done")
                return True
            except Exception:
                if iTry + 1 < 3:
                    tmp_log.debug(f"retry: {iTry}")
                    continue
                return False
        for (x,) in self.cur:
            break
""")


def test_a_loop_whose_way_back_prints_a_row_the_run_lacks_went_round_once_for_s2_and_s3(tmp_path):
    (_corpus(tmp_path) / "pandaserver" / "a.py").write_text(_RETRY)
    stamp = "2026-01-01 00:00:0{},000 panda.log.X: DEBUG    <PandaID=1> {}\n"
    logs = {
        "once": ["start", "kept the job defined", "done"],
        "retried": ["start", "kept the job defined", "retry: 0", "activated the job", "done"],
        "twice": ["start", "kept the job defined", "done"] * 2,
    }
    for name, said in logs.items():
        (tmp_path / f"{name}.log").write_text("".join(stamp.format(i, t) for i, t in enumerate(said)))
    out = _ok(
        _run(
            tmp_path,
            """
            import ast, json, textwrap
            from bamboo.querier import evaluate as E
            from bamboo.querier import rerun as RR
            from bamboo.querier import rows as R
            from bamboo.querier.records import read_log, runs_for
            from bamboo.querier.which_ran import Candidate, judge, once_loops_of

            R.ROWS_FROM = "walk"
            E.PURE.setdefault("<no pure function is read here>", None)  # len(res) is not one: no panda-server import
            fn = R.function("pandaserver/a.py", "f", (8, 12))
            shape = {"all": sorted(R.once_loops(fn.node, R.is_print).values()),
                     "none": sorted(R.once_loops(fn.node, lambda s: False).values()),
                     "start only": sorted(R.once_loops(fn.node, lambda s: s.lineno == 3).values()),
                     "completes": [R.completes(n.body) for n in R.own_nodes(fn.node) if isinstance(n, R.LOOPS)]}
            block = lambda text: ast.parse(textwrap.dedent(text)).body
            shape["blocks"] = [R.completes(block(t)) for t in (
                "if a:\\n    return 1\\nelse:\\n    x = 1\\n", "if a:\\n    return 1\\nelse:\\n    raise E\\n",
                "try:\\n    return 1\\nexcept E:\\n    x = 1\\n", "try:\\n    return 1\\nexcept E:\\n    return 2\\n",
                "try:\\n    x = 1\\nfinally:\\n    return 2\\n")]
            got = {}
            for name in ("once", "retried", "twice"):
                run, = runs_for(read_log(f"{name}.log"), "1")
                c = Candidate("pandaserver/a.py", "f", [8, 12], [run.log.name], run=run)
                verdicts = {v.line: [v.kind, v.reasons] for v in judge([c], {run.log.name: [run]}, question_run=run).verdicts}
                sn = R.claim([R.seen(fn, run, "outside")])[0]
                r = RR.Rerun.__new__(RR.Rerun)
                r.lines, r.opened, r.hand, r.hand_used, r.hop, r.mode = [], [], {}, [], None, "outside"
                try:
                    went = r.blocked(RR.Blocked(fn, 14, [run], [E.Env()], [RR.Arm(8, ["activated"], ["len(res) == 0"])]))
                    s3 = [went.guard, went.line]
                except RR.Stop as e:
                    s3 = e.args[0]
                got[name] = [sorted(once_loops_of(fn, sn, run, "outside").values()), verdicts, s3,
                             [x for x in r.lines if x.startswith("settled")]]
            print(json.dumps([shape, got]))
            """,
        )
    )
    shape, got = json.loads(out.splitlines()[-1])
    # the retry loop goes round only by its continue, after "retry"; the last loop's body always breaks
    assert shape["all"] == ["the loop at 20 cannot go round: no way back",
                            "the loop at 4 went round once: its way back prints row(s) [17], not in the run"]
    assert shape["none"] == ["the loop at 20 cannot go round: no way back"]
    assert shape["completes"] == [False, False]
    # the way back's row is the retry's own: "start", before the loop, being absent says nothing of it
    assert shape["start only"] == ["the loop at 20 cannot go round: no way back"]
    # an else or a handler that runs to its end lets the block end; a finally that returns does not
    assert shape["blocks"] == [True, False, True, False, False]
    once, verdicts, s3, settled = got["once"]
    assert once == shape["all"]
    # no "retry": the else's row leaves the arm out, the loop gone round once (the map's flow.together, once)
    assert verdicts["8"][0] == "excluded"
    assert ("row 11 printed, and one call cannot print it and run 8 (the loop at 4 went round once: its way back prints "
            "row(s) [17], not in the run)") in verdicts["8"][1]
    # S3 reads the else's path condition off its row, though the exit (the return at 14) is reached either way
    assert settled == ["settled by the exit taken: {}", "settled by the rows the pass printed on the way [11]: {'len(res) == 0': False}"]
    assert s3 == ["len(res) == 0", 7]
    # retried: the loop may have gone round, so the arm's own row decides, and S3 reads no row in the loop
    once, verdicts, s3, settled = got["retried"]
    assert once == ["the loop at 20 cannot go round: no way back"]
    assert verdicts["8"][0] == "positive"
    assert settled == ["settled by the exit taken: {}"]
    assert s3 == "S3: a guard the passes made false against every arm (none shown against arm 8)"
    # two calls in one run: no loop is read as gone round once, and S3 reads no row
    once, verdicts, s3, settled = got["twice"]
    assert once == [] and settled == ["settled by the exit taken: {}"]


_PASSES = textwrap.dedent("""
    def rescue(self, ids, who):
        tmp_log = MsgWrapper(logger, f"rescue < {who} >")
        tmp_log.debug("start")
        for task_id in ids:
            tmp_log.debug(f"[jediTaskID={task_id}] to check")
            self.cur.execute(sql, {":jediTaskID": task_id})
            tmp_log.debug(f"[jediTaskID={task_id}] done")
        tmp_log.debug("finished")


    def reactivate(self, ids, who):
        tmp_log = MsgWrapper(logger, f"reactivate < {who} >")
        tmp_log.debug("start reactivating")
        for task_id in ids:
            tmp_log.debug("to release")
            self.release(task_id)
            tmp_log.debug("released")
        tmp_log.debug("reactivated")


    def release(self, task_id):
        tmp_log = MsgWrapper(logger, f"release < jediTaskID={task_id} >")
        tmp_log.debug("start releasing")
        self.cur.execute(sql, {":jediTaskID": task_id})
        tmp_log.debug("released the task")


    def finish(self, ids, who):
        tmp_log = MsgWrapper(logger, f"finish < {who} >")
        tmp_log.debug("start")
        for task_id in ids:
            tmp_log.debug(f"start {task_id}")
            self.cur.execute(sql, {":jediTaskID": task_id})
        tmp_log.debug("done")


    def loose(self, ids):
        tmp_log.debug("start")
        for task_id in ids:
            self.cur.execute(sql, {":jediTaskID": task_id})
        tmp_log.debug("done")
""")


def test_a_candidate_is_read_on_its_pass_that_holds_the_key_at_the_window_and_the_one_left_is_chosen_unread(tmp_path):
    (_corpus(tmp_path) / "pandaserver" / "a.py").write_text(_PASSES)
    line = "2026-01-01 {},000 panda.log.X: DEBUG    {} {}\n"

    def passes(at: str, mark: str, key: int) -> list[str]:
        return [line.format(at, mark, t) for t in ("start", f"[jediTaskID={key}] to check", f"[jediTaskID={key}] done", "finished")]

    vo = "rescue < vo=atlas >"
    logs = {
        # a mark that names no key, on two machines: the key in A's pass at 19:00, in B's at 13:00
        "a": passes("13:00:00", vo, 5) + passes("19:00:00", vo, 777),
        "b": passes("13:00:00", vo, 777),
        # another method's mark, found alone
        "c": [line.format("13:00:00", "other < vo=atlas >", "[jediTaskID=777] checked")],
        # the key only in the callee's mark, inside the call; and once more, at 18:00, in the caller's own
        "d": [line.format("13:00:00", m, t).replace(",000", f",{n}00") for n, (m, t) in enumerate([
            ("reactivate < vo=atlas >", "start reactivating"), ("reactivate < vo=atlas >", "to release"),
            ("release < jediTaskID=777 >", "start releasing"), ("release < jediTaskID=777 >", "released the task"),
            ("reactivate < vo=atlas >", "released"), ("reactivate < vo=atlas >", "reactivated")])]
        + [line.format("18:00:00", "reactivate < vo=atlas >", t) for t in ("start reactivating", "jediTaskID=777 left alone")],
        # a mark that holds the key, two passes with a gap around the window
        "e": passes("10:00:00", "rescue < jediTaskID=777 >", 777) + passes("16:00:00", "rescue < jediTaskID=777 >", 777),
        # the log starts inside the pass that holds the key
        "f": passes("13:00:00", vo, 777)[1:] + passes("14:00:00", vo, 5),
        # one pass whose loop prints "start <id>": the entry row's "start" finds it, but does not read it whole
        "g": [line.format("13:00:00", "finish < vo=atlas >", t) for t in ("start", "start 5", "start 777", "done")],
        # two passes through a logger the function does not bind: the map cannot read its entry row whole
        "h": [line.format("13:00:00", "loose < jediTaskID=777 >", t) for t in ("start", "done", "start", "done")],
        # the callee's run inside the caller's pass, before the call
        "i": [line.format("13:00:00", m, t).replace(",000", f",{n}00") for n, (m, t) in enumerate([
            ("reactivate < vo=atlas >", "start reactivating"), ("release < jediTaskID=777 >", "start releasing"),
            ("release < jediTaskID=777 >", "released the task"), ("reactivate < vo=atlas >", "to release"),
            ("reactivate < vo=atlas >", "released"), ("reactivate < vo=atlas >", "reactivated")])]
        + [line.format("18:00:00", "reactivate < vo=atlas >", t) for t in ("start reactivating", "jediTaskID=777 left alone")],
    }
    for name, said in logs.items():
        (tmp_path / name).mkdir()
        (tmp_path / name / "w.log").write_text("".join(said))
    out = _ok(
        _run(
            tmp_path,
            """
            import json
            from dataclasses import replace
            from bamboo.querier import rows as R
            from bamboo.querier import windows as W
            from bamboo.querier.records import Run, read_log, runs_for
            from bamboo.querier.which_ran import Candidate, judge, passes_at

            R.ROWS_FROM = "walk"
            at = ("2026-01-01 13:00:00", "2026-01-01 13:00:00")
            runs = {name: runs_for(read_log(f"{name}/w.log"), "777") for name in "abcdefghi"}
            span = lambda r: [r.first, r.last, len(r.records)]
            rescue = Candidate("pandaserver/a.py", "rescue", [7], ["w.log"])
            got = {}
            for name in ("ab", "c", "e", "f"):
                found = [r for n in name for r in runs[n]]
                kept, why = passes_at(rescue, found, at, {"w.log": found}, "outside")
                j = judge([rescue], {"w.log": found}, window=at)
                write = judge([rescue], {"w.log": found}, window=at, holds_write=True)
                strict = judge([rescue], {"w.log": found}, window=at, decide="strict", holds_write=True)
                v, = j.verdicts
                got[name] = [[span(p) for p in kept], why, v.kind, j.how,
                             span(j.runs[rescue.owner]) if j.runs.get(rescue.owner) else None, write.how, strict.how]
            # the callee's run: its function by the map's rows, the call by the map's reach
            R.line_writers = lambda record, log: frozenset({"pandaserver/a.py::release"})
            R.print_lines = lambda owner: [24]
            called = lambda fn, callee: ["pandaserver/a.py:17"] if callee.func == "release" else []
            reactivate = Candidate("pandaserver/a.py", "reactivate", [17], ["w.log"])
            mine = [r for r in runs["d"] if r.token.startswith("reactivate")]
            # the callee's run on another machine is not inside this pass's call
            elsewhere = [Run(r.log, r.token, [replace(x, machine="aipanda002") for x in r.records],
                             [replace(x, machine="aipanda002") for x in r.found_by]) if r.token.startswith("release") else r
                         for r in runs["d"]]
            got["d"] = [[span(p) for p in passes_at(reactivate, mine, at, {"w.log": runs["d"]}, "outside", called)[0]],
                        passes_at(reactivate, mine, at, {"w.log": runs["d"]}, "outside")[1],
                        [span(p) for p in passes_at(reactivate, runs["d"], at, {"w.log": runs["d"]}, "outside", called)[0]],
                        [span(p) for p in passes_at(reactivate, mine, at, {"w.log": elsewhere}, "outside", called)[0]],
                        list(passes_at(reactivate, [r for r in runs["i"] if r.token.startswith("reactivate")], at,
                                       {"w.log": runs["i"]}, "outside", called))]
            got["d"][-1][0] = [span(p) for p in got["d"][-1][0]]
            finish = Candidate("pandaserver/a.py", "finish", [34], ["w.log"])
            fin = R.function(finish.rel, finish.func, finish.lines)
            got["g"] = [R.entries(fin, runs["g"][0], "outside"),
                        [span(p) for p in passes_at(finish, runs["g"], at, {"w.log": runs["g"]}, "outside")[0]],
                        R.passes(fin, runs["g"][0], "outside")[1], R.passes(fin, runs["g"][0].part(1, 4), "outside")[1]]
            # one statement, the map's rows of it one per logger it may print through
            import copy
            import re
            from types import SimpleNamespace

            def marked(name):
                return SimpleNamespace(capture="start", prefix=re.escape(f"{name} < ") + ".*?" + re.escape(" > "), prefix_hole="",
                                       prefix_capture=re.escape(f"{name} < ") + "(.*?)" + re.escape(" >"))

            uncaptured = SimpleNamespace(capture="", prefix="", prefix_hole="", prefix_capture="")

            def whole(*map_rows):
                entry = copy.copy(R.entry_row(fin))
                entry.map_rows = list(map_rows)
                return entry.reads_whole(runs["g"][0].records[0])

            got["whole"] = [whole(marked("finish"), marked("other")), whole(marked("other"), marked("finish")),
                            whole(marked("other")), whole(uncaptured), whole(marked("other"), uncaptured)]
            loose = R.function("pandaserver/a.py", "loose", [41])
            got["h"] = [R.entries(loose, runs["h"][0], "outside"), R.passes(loose, runs["h"][0], "outside")[1]]
            # the window that holds the write is the time value's alone
            got["windows"] = [W.value_now("rescueTime", "2026-01-01 13:00:00").holds_write,
                              W.Asking(at, None, "a statuslog row").holds_write]
            print(json.dumps(got))
            """,
        )
    )
    got = json.loads(out.splitlines()[-1])
    kept, why, kind, how, run, write, strict = got["ab"]
    # A's run spans the window but its pass there holds no key; B's pass is the one, read alone
    assert kept == [["2026-01-01 13:00:00,000", "2026-01-01 13:00:00,000", 4]]
    assert why == ["w.log 'rescue < vo=atlas >': its pass at the window holds no record of the key, nor a run of it inside a call"]
    assert run == kept[0] and kind == "positive"
    # another method's run, found alone, is not the candidate's: unexamined.  The one left is chosen by elimination
    # only in a window that holds the write (a time its writer sets to when it runs), and not when strict
    kept, why, kind, how, run, write, strict = got["c"]
    assert kept == [] and why == ["1 run(s) under another method's mark (other)"]
    assert kind == "unexamined" and how == "1 left, all unexamined" and run is None
    assert write == "elimination over a complete list" and strict == "strict: 1 unexamined"
    # a keyed mark with a gap: no pass of it is at the window
    kept, why, kind, how, run, write, strict = got["e"]
    assert kept == [] and why == ["w.log 'rescue < jediTaskID=777 >': none of its 2 pass(es) of rescue is at the window"]
    # the records before the first entry are a pass of their own: the log began inside it
    kept, why, kind, how, run, write, strict = got["f"]
    assert kept == [["2026-01-01 13:00:00,000", "2026-01-01 13:00:00,000", 3]]
    # the caller's pass holds the key only through the callee's run inside its call; with no reach, it holds none
    through, without, every, elsewhere, before = got["d"]
    assert through == [["2026-01-01 13:00:00,000", "2026-01-01 13:00:00,500", 4]]
    assert without == ["w.log 'reactivate < vo=atlas >': its pass at the window holds no record of the key, nor a run of it inside a call"]
    # the callee's own run is another method's mark for the caller; a callee's run on another machine is no part of it
    assert every == through and elsewhere == []
    # a callee's run inside the pass but before its call ran in no call of it
    assert before == [[], without]
    # a pass begins at a record its entry row reads whole: "start 777" is the loop's row, not a pass begun
    entries, kept, cut, inside = got["g"]
    assert entries == [0] and kept == [["2026-01-01 13:00:00,000", "2026-01-01 13:00:00,000", 4]]
    # the loop's records the map reads as not the entry row's line neither cut nor count as unread
    assert cut == "cut at entry row 31 of finish: 1 pass(es)"
    assert inside == "entry row 31 of finish is read whole at none of its 2 record(s) in the run: the run is not split"
    # any of the statement's map rows that reads the record whole does; None only where none can say
    assert got["whole"] == [True, True, False, None, False]
    # a record the map cannot read whole does not cut: the two passes stay one, and the word says why
    assert got["h"] == [[], "entry row 39 of loose is read whole at none of its 2 record(s) in the run (2 the map "
                            "cannot read whole): the run is not split"]
    assert got["windows"] == [True, False]


_KILLS = textwrap.dedent("""
    def kill(self, pandaID, code, user):
        tmp_log = MsgWrapper(logger, f"kill < PandaID={pandaID} >")
        tmp_log.debug(f"code={code} user={user} wg=[]")
        for table in tables:
            tmp_log.debug("using prod role")
        tmp_log.debug(f"done code={code}")


    def kill_twin(self, pandaID, code, user, who):
        tmp_log = MsgWrapper(logger, f"kill_twin < PandaID={pandaID} >")
        tmp_log.debug(f"code={code} user={user} wg=[]")
        tmp_log.debug(f"user={who} wg=[]")
        tmp_log.debug(f"done code={code}")
""")


def test_sc_an_entry_record_read_only_literally_cuts_where_nothing_else_can_have_printed_it(tmp_path):
    # SC (out/gm/qk-design.md): killJob's 840 closes `user` at its first blank, so "finish by ..." is read whole
    # nowhere and MJE-7396802500's two calls (05:27:04, 05:40:46) stayed one pass
    (_corpus(tmp_path) / "pandaserver" / "a.py").write_text(_KILLS)
    line = "2026-01-01 05:{},000 panda.log.X: DEBUG    {} < PandaID=777 > {}\n"

    def calls(mark: str, user: str) -> str:
        return "".join(line.format(at, mark, t) for at, t in [
            ("27:04", f"code=52 user={user} wg=[]"), ("27:04", "done code=52"),
            ("40:46", f"code=52 user={user} wg=[]"), ("40:46", "using prod role"), ("40:46", "done code=52")])

    for name, mark, user in [("blank", "kill", "finish by A B"), ("bare", "kill", "jedi"), ("twin", "kill_twin", "finish by A B")]:
        (tmp_path / name).mkdir()
        (tmp_path / name / "w.log").write_text(calls(mark, user))
    out = _ok(
        _run(
            tmp_path,
            """
            import json
            from bamboo.querier import rows as R
            from bamboo.querier.records import read_log, runs_for

            R.ROWS_FROM = "walk"
            run = {name: runs_for(read_log(f"{name}/w.log"), "777")[0] for name in ("blank", "bare", "twin")}
            kill, twin = R.function("pandaserver/a.py", "kill", [4]), R.function("pandaserver/a.py", "kill_twin", [12])
            import re

            # the writers' index as the dump gives it, built from these functions' own map rows (the walk's), with
            # "other" holding kill's rows too where another function prints through kill's logger
            def rows_of(fn, owner):
                return [(re.compile(m.pattern, re.S), owner, R._prefix_regex((m.prefix,)) if m.prefix else None, None, m)
                        for r in fn.rows for m in r.map_rows if m.pattern]

            dump = object()
            R.mapdump.fragment = lambda: dump
            got = {}
            for writers in ("alone", "shared"):
                shared = rows_of(kill, "pandaserver/a.py::other") if writers == "shared" else []
                R._WRITERS[:] = [dump, rows_of(kill, "pandaserver/a.py::kill") + rows_of(twin, "pandaserver/a.py::kill_twin")
                                 + shared, {}, {}, {}, {}, {}]
                first = run["blank"].records[0]
                got[writers] = [R.entries(kill, run["blank"], "outside"), R.passes(kill, run["blank"], "outside")[1],
                                sorted(R.line_writers(first, "w.log")), sorted(R.literal_writers(first, "w.log"))]
            got["bare"] = [R.entries(kill, run["bare"], "outside"), R.passes(kill, run["bare"], "outside")[1]]
            got["twin"] = [R.entries(twin, run["twin"], "outside"), R.passes(twin, run["twin"], "outside")[1]]
            print(json.dumps(got))
            """,
        )
    )
    got = json.loads(out.splitlines()[-1])
    # read as the entry row's line only with its holes taking any characters, no other row of kill finding it and no
    # other function reading it so: each call is a pass, and the word says how the cut was read.  The map's own pattern
    # (user's hole takes no blank) finds the record for no function, kill's row neither: line_writers is empty, and
    # only the literal reading names kill (the measure scb missed MJE-7396802500 on a guard by line_writers)
    assert got["alone"] == [[0, 2], "cut at entry row 4 of kill (2 read as its line only with its holes taking any "
                                    "characters, nothing else printing it): 2 pass(es)", [], ["pandaserver/a.py::kill"]]
    # another function reads it so too: it may be that function's line, and the run stays one
    assert got["shared"] == [[], "entry row 4 of kill is read whole at none of its 2 record(s) in the run (2 the map "
                                 "cannot read whole): the run is not split", [],
                             ["pandaserver/a.py::kill", "pandaserver/a.py::other"]]
    # a user with no blank is read whole, as before
    assert got["bare"] == [[0, 2], "cut at entry row 4 of kill: 2 pass(es)"]
    # another row of the function finds the record ("user={who} wg=[]"): it may be that row's line, no cut
    assert got["twin"] == [[], "entry row 12 of kill_twin is read whole at none of its 2 record(s) in the run (2 the "
                               "map cannot read whole): the run is not split"]


def test_a_run_no_entry_row_cuts_is_at_the_window_only_where_a_record_of_it_lies_near(tmp_path):
    # K57: a child task's run held only its two key lines, hours apart, and was read for the parent's write between
    (_corpus(tmp_path) / "pandaserver" / "a.py").write_text(_PASSES)
    line = "2026-01-01 {},000 panda.log.X: DEBUG    rescue < jediTaskID=777 > [jediTaskID=777] {}\n"
    logs = {
        # two passes' key lines, no entry row among them, hours either side of the window
        "far": [line.format("10:00:00", "to check"), line.format("16:00:00", "done")],
        # the same, one of them 9 min before the window
        "near": [line.format("12:51:00", "to check"), line.format("16:00:00", "done")],
    }
    for name, said in logs.items():
        (tmp_path / name).mkdir()
        (tmp_path / name / "w.log").write_text("".join(said))
    out = _ok(
        _run(
            tmp_path,
            """
            import json
            from bamboo.querier import rows as R
            from bamboo.querier.records import read_log, runs_for
            from bamboo.querier.which_ran import Candidate, at_window, judge, passes_at, spans

            R.ROWS_FROM = "walk"
            at = ("2026-01-01 13:00:00", "2026-01-01 13:00:00")
            rescue = Candidate("pandaserver/a.py", "rescue", [7], ["w.log"])
            got = {}
            for name in ("far", "near"):
                run, = runs_for(read_log(f"{name}/w.log"), "777")
                kept, why = passes_at(rescue, [run], at, {"w.log": [run]}, "outside")
                v, = judge([rescue], {"w.log": [run]}, window=at).verdicts
                got[name] = [spans(run, at), at_window(run, at), len(kept), why, v.kind, v.reasons]
            print(json.dumps(got))
            """,
        )
    )
    got = json.loads(out.splitlines()[-1])
    # its span holds the window, but no record of it lies within 10 min: no pass of it is at the window
    assert got["far"][:4] == [True, False, 0, ["w.log 'rescue < jediTaskID=777 >': none of its 1 pass(es) of rescue is at "
                                               "the window (1 span(s) it with no record within 10 min of it)"]]
    assert got["far"][4:] == ["unexamined", ["w.log: no pass of the key at ('2026-01-01 13:00:00', '2026-01-01 13:00:00'): "
                                             "w.log 'rescue < jediTaskID=777 >': none of its 1 pass(es) of rescue is at "
                                             "the window (1 span(s) it with no record within 10 min of it)"]]
    # a record within 10 min: the run is at the window, and judged on
    assert got["near"][:4] == [True, True, 1, []]
    assert got["near"][4] != "unexamined"


def test_a_late_question_asks_the_stay_it_names_and_stops_where_the_value_has_several_and_it_names_none(tmp_path):
    out = _ok(
        _run(
            tmp_path,
            """
            import json
            from bamboo.querier.rerun import _stay_of

            # MT-exhausted-late's shape: exhausted twice, its second stay begun in the second its toretry row holds
            row = lambda t, s, ext=True: {"time": t[:19], "status": s, **({"time_extended": t} if ext else {})}
            states = [row("2026-10-19 04:01:59.100000", "toretry"), row("2026-10-19 04:01:59.200000", "exhausted"),
                      row("2026-10-29 04:02:22.726170", "finishing"), row("2026-10-29 09:33:01.581347", "toretry"),
                      row("2026-10-29 09:33:01.716796", "exhausted"), row("2026-10-29 11:14:23.613054", "toretry")]
            # the requirement not met is a record; rendered, as the stop says it
            said = lambda got: [got[0], str(got[1])]
            print(json.dumps([
                _stay_of(states, "exhausted", "2026-10-29 09:33:01.716796")[0],
                _stay_of(states, "exhausted", "2026-10-29 09:33:01.581347")[0],
                said(_stay_of(states, "exhausted", None)),
                said(_stay_of(states, "finishing", None)),
                _stay_of([row("2026-10-29 09:33:01.716796", "exhausted", ext=False)], "exhausted", "2026-10-29 09:33:01.716796")[0],
                said(_stay_of(states, "running", None)),
            ]))
            """,
        )
    )
    named, other_row, unnamed, one, by_time, none = json.loads(out.splitlines()[-1])
    # the row whose time_extended it names; the toretry row of the same second begins no stay of exhausted
    assert named == 4 and other_row is None
    # several stays and none named: a stop that says them, the first by default no longer
    assert unnamed[0] is None and "2 stays" in unnamed[1] and "--since" in unnamed[1]
    assert "the first 2026-10-19 04:01:59.200000 (240.01 h)" in unnamed[1] and "the last 2026-10-29 09:33:01.716796 (1.69 h)" in unnamed[1]
    # one stay needs no name; a row with no time_extended is named by its time
    assert one == [2, ""] and by_time == 0
    assert none[0] is None and none[1] == "a statuslog row for 'running'"


# the adder's process_job_report, cut down: 416 binds the job's status from self.job_status, 441 sets it failed
_ADDER = """
class Adder:
    def __init__(self, job, job_status, report):
        self.logger = logger
        self.job = job
        self.job_status = job_status
        self.report = report
        self.plugin_class = Plugin
        self.add_result = None

    def process(self):
        self.logger.debug("start the report of the job")
        if self.job.jobStatus not in ["transferring"]:
            self.job.jobStatus = self.job_status
        self.add_result = None
        parsed = self.parse()
        if parsed < 2:
            self.run_plugin()
            if self.add_result is None:
                self.job.jobStatus = "failed"
        self.logger.debug(f"status after plugin call :job.jobStatus={self.job.jobStatus} jobStatus={self.job_status}")
        self.logger.debug("updating the job in the DB")
        return True

    def parse(self):
        if self.report is None:
            self.logger.debug("no report of the job to parse")
            self.job.jobStatus = "failed"
            return 2
        return 0

    def run_plugin(self):
        self.add_result = self.plugin_class(self.job).execute()
"""


def test_p_reads_the_status_a_target_was_bound_from_at_a_later_row_with_the_judges_other_verdicts(tmp_path):
    files = {
        "c.py": _ADDER,
        # a subclass runs another parse, which binds self.job_status
        "d.py": _ADDER + "\n\nclass Adder2(Adder):\n    def parse(self):\n        self.job_status = 'failed'\n        return 0\n",
        # a method lets self out, though no call here runs a store of job_status: a call is read by what it runs
        "e.py": _ADDER + "\n    def register(self):\n        REGISTRY.append(self)\n",
    }
    # the status bound from an object below self: handed to the plugin, which writes it, the object may hold another
    # status after (a plugin of the corpus: one the map cannot read stores nothing on the corpus's objects)
    info = _ADDER.replace("self.job_status", "self.info.status").replace("self.report = report", "self.report, self.info = report, info")
    files["h.py"] = info.replace("self.plugin_class(self.job)", "self.plugin_class(self.job, self.info)") + (
        "\n\nclass Plugin:\n    def __init__(self, job, info):\n        info.status = 'closed'\n\n    def execute(self):\n        return None\n"
    )
    files["i.py"] = info
    for name, text in files.items():
        (_corpus(tmp_path) / "pandaserver" / name).write_text(text)
    lines = _ADDER.splitlines()
    arm, parse_arm, call = (lines.index(f"{' ' * n}{s}") + 1 for n, s in
                            [(16, 'self.job.jobStatus = "failed"'), (12, 'self.job.jobStatus = "failed"'), (8, "parsed = self.parse()")])
    stamp = "2026-01-01 00:00:0{},000 panda.log.X: DEBUG    <PandaID=1> {}\n"
    after = "status after plugin call :job.jobStatus={} jobStatus={}"
    logs = {
        "core": [stamp.format(0, "start the report of the job"), stamp.format(1, after.format("failed", "finished")),
                 stamp.format(2, "updating the job in the DB")],
        # parse's arm printed its row: a writer of 'failed' still standing, reached from the call at 'parsed = ...'
        "live": [stamp.format(0, "start the report of the job"), stamp.format(1, "no report of the job to parse"),
                 stamp.format(2, after.format("failed", "finished")), stamp.format(3, "updating the job in the DB")],
        "other": [stamp.format(0, "start the report of the job"), stamp.format(1, after.format("finished", "closed")),
                  stamp.format(2, "updating the job in the DB")],
    }
    for name, text in logs.items():
        (tmp_path / f"{name}.log").write_text("".join(text))
    code = """
        import json
        from bamboo.querier import printed as P
        from bamboo.querier import rows as R
        from bamboo.querier.records import read_log, runs_for
        from bamboo.querier.which_ran import Candidate, judge

        R.ROWS_FROM = "walk"
        got = {}
        for file, log in [("c.py", "core"), ("c.py", "live"), ("d.py", "core"), ("e.py", "core"), ("h.py", "core"), ("i.py", "core")]:
            run, = runs_for(read_log(f"{log}.log"), "1")
            rel = f"pandaserver/{file}"
            cs = [Candidate(rel, "process", [ARM], [run.log.name], run=run), Candidate(rel, "parse", [PARSE_ARM], [run.log.name], run=run)]
            at = lambda fn, c, rel=rel: [f"{rel}:CALL"] if c.func == "parse" else []
            j = judge(cs, {run.log.name: [run]}, question_run=run, subject="jobStatus", observed="failed", called_at=at)
            got[f"{file} {log}"] = [[v.kind, v.reasons] for v in j.verdicts] + [j.how]
        run, = runs_for(read_log("other.log"), "1")
        fn = R.function("pandaserver/c.py", "process", (ARM,))
        sn = R.seen(fn, run, "outside")
        for value in ("failed", "finished"):
            proof = P.prove(fn, sn, R.statement_at(fn.node, ARM), P.Reaching(value, frozenset()))
            got[f"other {value}"] = [proof.kind, proof.dropped]
        print(json.dumps(got))
    """
    out = _ok(_run(tmp_path, code.replace("PARSE_ARM", str(parse_arm)).replace("ARM", str(arm)).replace("CALL", str(call))))
    got = json.loads(out.splitlines()[-1])
    (kind, why), (parse_kind, _), how = got["c.py core"]
    # the call that could have written 'failed' reaches only parse's arm, which the run excludes: the path that skips the
    # arm holds self.job_status, printed 'finished' at the row that printed the job 'failed' -- and the one through
    # 'transferring' cannot print 'failed' either (the if's level above the arm's chain)
    assert parse_kind == "excluded" and kind == "positive" and how == "positive evidence"
    assert "every path through 13 that fits" in why[-1] and "E4 deferred" in why[-1] and "E4 prior" in why[-1]
    assert why[-1].endswith("read with the other verdicts: a call reaching no writer of 'failed' left standing does not write it")
    # a writer of 'failed' standing behind the call: the job's status may be its
    assert got["c.py live"][0][0] == "kept"
    # parse resolved to two methods: the call may run Adder2.parse, which binds self.job_status
    assert got["d.py core"][0][0] == "kept"
    # self let out by a method not called here: the calls are read by what they run, and none of it writes job_status
    assert got["e.py core"][0][0] == "positive"
    # bound from self.info.status: read at the row where no call is handed self.info, not where the plugin is
    assert got["i.py core"][0][0] == "positive" and got["h.py core"][0][0] == "kept"
    # a row printing another value than the observed one: any call handed the job may have written it
    assert got["other failed"][0] == "none" and "E4 deferred" not in got["other failed"][1]
    assert got["other finished"][0] == "inconsistent" and got["other finished"][1]["E4 deferred"] > 0


# the adder with its finalize: a failed job's status is set "failed" once more, in a method two calls down
_FINALIZING = _ADDER.replace(
    '        self.logger.debug("updating the job in the DB")\n',
    '        self.finalize()\n        self.logger.debug("updating the job in the DB")\n',
) + """
    def finalize(self):
        if self.job.jobStatus == "failed" or self.job_status == "failed":
            self.handle_failed()

    def handle_failed(self):
        self.logger.debug("handling the failed job")
        self.job.jobStatus = "failed"
"""


def test_of_one_pass_s_writes_the_last_that_changed_the_value_is_the_answer_and_one_that_kept_it_a_stay(tmp_path):
    files = {
        "c.py": _FINALIZING,
        # the job handed to a call between the row and finalize: what the row printed is not read past it
        "g.py": _FINALIZING.replace("        self.finalize()\n", "        touch(self.job)\n        self.finalize()\n"),
    }
    at = {}
    for name, text in files.items():
        (_corpus(tmp_path) / "pandaserver" / name).write_text(text)
        lines = text.splitlines()
        at[name] = {k: lines.index(f"{' ' * n}{s}") + 1 for k, n, s in [
            ("from_report", 12, "self.job.jobStatus = self.job_status"), ("plugin", 16, 'self.job.jobStatus = "failed"'),
            ("parse", 12, 'self.job.jobStatus = "failed"'), ("handle", 8, 'self.job.jobStatus = "failed"'),
            ("parse_call", 8, "parsed = self.parse()"), ("finalize_call", 8, "self.finalize()")]}
        at[name]["row"] = next(n for n, line in enumerate(lines, 1) if "status after plugin call" in line)
    stamp = "2026-01-01 00:00:0{},000 panda.log.X: DEBUG    <PandaID=1> {}\n"
    after = "status after plugin call :job.jobStatus={} jobStatus={}"
    logs = {
        # the report said failed: parse's arm and handle_failed's both ran, and the row before finalize shows failed
        "reported": ["start the report of the job", "no report of the job to parse", after.format("failed", "failed"),
                     "handling the failed job", "updating the job in the DB"],
        # the report said finished: the plugin's arm changed it, and handle_failed set it again
        "plugin": ["start the report of the job", after.format("failed", "finished"), "handling the failed job",
                   "updating the job in the DB"],
    }
    for name, text in logs.items():
        (tmp_path / f"{name}.log").write_text("".join(stamp.format(n, t) for n, t in enumerate(text)))
    code = """
        import json
        from bamboo.querier import rows as R
        from bamboo.querier.records import read_log, runs_for
        from bamboo.querier.which_ran import Candidate, judge

        R.ROWS_FROM = "walk"
        got = {}
        for file, log, reach in [("c.py", "reported", True), ("c.py", "plugin", True), ("g.py", "reported", True),
                                 ("c.py", "reported", False)]:
            at = AT[file]
            run, = runs_for(read_log(f"{log}.log"), "1")
            rel = f"pandaserver/{file}"
            lines = {"process": [at["from_report"], at["plugin"]], "parse": [at["parse"]], "handle_failed": [at["handle"]]}
            cs = [Candidate(rel, func, ls, [run.log.name], run=run) for func, ls in lines.items()]
            # the map's reach: process calls on into parse and, through finalize, into handle_failed
            calls = {"parse": at["parse_call"], "handle_failed": at["finalize_call"] if reach else None}
            def called_at(fn, c, rel=rel, calls=calls):
                return [f"{rel}:{calls[c.func]}"] if fn.name == "process" and calls.get(c.func) else []
            j = judge(cs, {run.log.name: [run]}, question_run=run, subject="jobStatus", observed="failed", called_at=called_at)
            got[f"{file} {log}{'' if reach else ' no reach'}"] = {
                "kinds": {k: v.kind for v in j.verdicts for k in [{at["from_report"]: "from_report", at["plugin"]: "plugin",
                                                                   at["parse"]: "parse", at["handle"]: "handle"}[v.line]]},
                "why": {k: v.reasons[-1] for v in j.verdicts for k in [f"{v.candidate.func}:{v.line}"] if v.reasons},
                "chosen": f"{j.chosen.candidate.func}:{j.chosen.line}" if j.chosen else None, "how": j.how}
        print(json.dumps(got))
    """
    out = _ok(_run(tmp_path, code.replace("AT", repr(at))))
    got = json.loads(out.splitlines()[-1])
    c = at["c.py"]
    # the row before finalize shows failed, and nothing on the way into handle_failed binds it: handle_failed's
    # assignment is a stay.  The three before the row are left, the last first: which changed it is not read
    reported = got["c.py reported"]
    assert reported["kinds"] == {"from_report": "kept", "plugin": "kept", "parse": "positive", "handle": "excluded"}
    assert reported["why"][f"handle_failed:{c['handle']}"].startswith(
        f"rule R: a stay, self.job.jobStatus already held 'failed' before it (row process:{c['row']} at reported.log:3"
    ) and reported["why"][f"handle_failed:{c['handle']}"].endswith("the frames between process and handle_failed not read)")
    assert reported["chosen"] is None and reported["how"] == (
        f"3 may have changed it last in the pass: c.py::process:{c['plugin']}, c.py::parse:{c['parse']}, c.py::process:{c['from_report']}"
    )
    # the report said finished: the assignment from it wrote 'finished' (the row's jobStatus), and the plugin's arm,
    # which ran, is the last change -- handle_failed's, after it, a stay by the value it assigned and the row before
    # finalize alike, said by the row
    plugin = got["c.py plugin"]
    assert plugin["kinds"] == {"from_report": "excluded", "plugin": "positive", "parse": "excluded", "handle": "excluded"}
    assert plugin["why"][f"process:{c['from_report']}"] == (
        f"rule R: it assigns 'finished' (self.job_status, printed at row process:{c['row']} at plugin.log:2), no write of 'failed'"
    )
    assert plugin["why"][f"handle_failed:{c['handle']}"].startswith(
        f"rule R: a stay, self.job.jobStatus already held 'failed' before it (row process:{c['row']} at plugin.log:2"
    )
    assert plugin["chosen"] == f"process:{c['plugin']}" and plugin["how"] == "the last change in the pass"
    # the job handed to a call on the way: handle_failed's may be the change, and it is the last
    g = got["g.py reported"]
    assert g["kinds"]["handle"] == "positive" and g["chosen"] is None
    assert g["how"].startswith(f"4 may have changed it last in the pass: g.py::handle_failed:{at['g.py']['handle']}, ")
    # no way placed from process into handle_failed: no order, and R says nothing
    assert got["c.py reported no reach"]["how"] == "2 positive"
    assert got["c.py reported no reach"]["kinds"]["handle"] == "positive"


def test_a_value_carried_past_a_call_the_maps_reach_reads_as_keeping_it_or_past_its_own_constant_stays(tmp_path):
    files = {
        # the job handed to a call between the row and finalize: kept where the reach says it assigns no other status
        "g.py": _FINALIZING.replace("        self.finalize()\n", "        touch(self.job)\n        self.finalize()\n"),
        # a method of self binding self.job anew: what the row printed is another object's, whatever the reach says
        "h.py": _FINALIZING.replace("        self.finalize()\n", "        self.reload()\n        self.finalize()\n")
        + "\n    def reload(self):\n        self.job = self.job.copy()\n",
        # the status set failed once more on the way, and set to another value
        "k.py": _FINALIZING.replace("        self.finalize()\n", '        self.job.jobStatus = "failed"\n        self.finalize()\n'),
        "m.py": _FINALIZING.replace("        self.finalize()\n", '        self.job.jobStatus = "closed"\n        self.finalize()\n'),
    }
    at = {}
    for name, text in files.items():
        (_corpus(tmp_path) / "pandaserver" / name).write_text(text)
        lines = text.splitlines()
        at[name] = {k: lines.index(f"{' ' * n}{s}") + 1 for k, n, s in [
            ("from_report", 12, "self.job.jobStatus = self.job_status"), ("plugin", 16, 'self.job.jobStatus = "failed"'),
            ("parse", 12, 'self.job.jobStatus = "failed"'), ("handle", 8, 'self.job.jobStatus = "failed"'),
            ("parse_call", 8, "parsed = self.parse()"), ("finalize_call", 8, "self.finalize()")]}
        # handle_failed's is the last: k.py sets failed in process too
        at[name]["handle"] = len(lines) - lines[::-1].index('        self.job.jobStatus = "failed"')
        at[name]["way"] = at[name]["finalize_call"] - 1
    stamp = "2026-01-01 00:00:0{},000 panda.log.X: DEBUG    <PandaID=1> {}\n"
    text = ["start the report of the job", "no report of the job to parse",
            "status after plugin call :job.jobStatus=failed jobStatus=failed", "handling the failed job",
            "updating the job in the DB"]
    (tmp_path / "reported.log").write_text("".join(stamp.format(n, t) for n, t in enumerate(text)))
    code = """
        import json
        from bamboo.querier import rows as R
        from bamboo.querier.records import read_log, runs_for
        from bamboo.querier.which_ran import Candidate, judge

        R.ROWS_FROM = "walk"
        got = {}
        for file, kept in [("g.py", True), ("g.py", False), ("h.py", True), ("k.py", True), ("m.py", True)]:
            at = AT[file]
            run, = runs_for(read_log("reported.log"), "1")
            rel = f"pandaserver/{file}"
            lines = {"process": [at["from_report"], at["plugin"]], "parse": [at["parse"]], "handle_failed": [at["handle"]]}
            cs = [Candidate(rel, func, ls, [run.log.name], run=run) for func, ls in lines.items()]
            calls = {"parse": at["parse_call"], "handle_failed": at["finalize_call"]}
            def called_at(fn, c, rel=rel, calls=calls):
                return [f"{rel}:{calls[c.func]}"] if fn.name == "process" and calls.get(c.func) else []
            # the map's reach as rerun reads it: the call on the way (touch, reload) assigns no other status, 2 closed
            asked = []
            def keeps(fn, node, value, kept=kept, way=at["way"]):
                asked.append([fn.name, node.lineno, value])
                return 2 if kept and node.lineno == way else None
            j = judge(cs, {run.log.name: [run]}, question_run=run, subject="jobStatus", observed="failed",
                      called_at=called_at, keeps=keeps)
            got[f"{file}{'' if kept else ' open'}"] = {
                "handle": next([v.kind, v.reasons[-1]] for v in j.verdicts if v.line == at["handle"]),
                "how": j.how, "asked": sorted(set(map(tuple, asked)))}
        print(json.dumps(got))
    """
    out = _ok(_run(tmp_path, code.replace("AT", repr(at))))
    got = json.loads(out.splitlines()[-1])
    g, h, k = at["g.py"], at["h.py"], at["k.py"]
    # the reach reads the call handed the job as keeping 'failed': handle_failed's assignment is a stay again, said so
    kind, why = got["g.py"]["handle"]
    assert kind == "excluded" and why.startswith("rule R: a stay, self.job.jobStatus already held 'failed' before it (row process:")
    assert f"nothing on the way binding it; the calls at g.py:{g['way']} reach no assignment of jobStatus but 'failed' " \
           "(the map's reach, 2 call(s) closed by their name); the frames between process and handle_failed not read)" in why
    assert ["process", g["way"], "failed"] in got["g.py"]["asked"]
    # the reach cannot say: the call cuts the value, as before
    assert got["g.py open"]["handle"][0] == "positive"
    assert got["g.py open"]["how"].startswith(f"4 may have changed it last in the pass: g.py::handle_failed:{g['handle']}, ")
    # self.job bound anew by a method of self: the reach of a field is not asked, the value cut
    assert got["h.py"]["handle"][0] == "positive" and ["process", h["way"], "failed"] not in got["h.py"]["asked"]
    # failed set once more on the way: kept, said where; closed set on the way: cut
    kind, why = got["k.py"]["handle"]
    assert kind == "excluded" and why.endswith(
        f"nothing on the way binding it but to 'failed' again at k.py:{k['way']}; the frames between process and handle_failed not read)")
    assert got["m.py"]["handle"][0] == "positive"


# the adder's pass with a call H2 holds as an arm (finalize, by the map's reading) and a call the DB's run ran inside
_ARCHIVING_ADDER = """
class Adder:
    def __init__(self, job, proxy):
        self.logger = logger
        self.job = job
        self.proxy = proxy

    def process(self):
        self.logger.debug("start the report of the job")
        if self.job.broken:
            raise RuntimeError("a broken job")
        self.job.jobStatus = self.job.reported
        self.logger.debug(f"status before finalize :job.jobStatus={self.job.jobStatus}")
        self.finalize()
        self.proxy.archive(self.job)
        self.logger.debug("updating the job in the DB")
        return True

    def finalize(self):
        self.job.jobStatus = self.job.final
"""
_ARCHIVING_PROXY = """
class Proxy:
    def __init__(self):
        self.logger = logger

    def archive(self, job):
        self.logger.debug(f"archive start status={job.jobStatus}")
        if job.check:
            self.logger.debug("set jobStatus=failed due to inconsistent input")
            job.jobStatus = "failed"
        if job.fine:
            job.jobStatus = "finished"
        self.logger.debug("archive done")
        return True

    def unmerged(self, job):
        self.logger.debug("updating the unmerged jobs")
        um_job = self.fetch(job)
        um_job.jobStatus = job.jobStatus
        return True
"""


def test_a_call_and_a_write_inside_one_are_stays_read_before_the_order_the_inside_one_in_its_own_frame(tmp_path):
    corpus = _corpus(tmp_path) / "pandaserver"
    (corpus / "a.py").write_text(_ARCHIVING_ADDER)
    (corpus / "db.py").write_text(_ARCHIVING_PROXY)
    adder, proxy = _ARCHIVING_ADDER.splitlines(), _ARCHIVING_PROXY.splitlines()
    at: dict[str, int] = {}
    for k, text, n, s in [
        ("raise", adder, 12, 'raise RuntimeError("a broken job")'),
        ("assign", adder, 8, "self.job.jobStatus = self.job.reported"), ("finalize", adder, 8, "self.finalize()"),
        ("call", adder, 8, "self.proxy.archive(self.job)"), ("failed", proxy, 12, 'job.jobStatus = "failed"'),
        ("write", proxy, 12, 'job.jobStatus = "finished"'), ("unmerged", proxy, 8, "um_job.jobStatus = job.jobStatus"),
    ]:
        at[k] = text.index(f"{' ' * n}{s}") + 1
    for k, text, s in [("row", adder, "status before finalize"), ("start", proxy, "archive start"),
                       ("inconsistent", proxy, "due to inconsistent input")]:
        at[k] = next(n for n, line in enumerate(text, 1) if s in line)
    stamp = "2026-01-01 00:00:0{},000 panda.log.X: DEBUG    <PandaID=1> {}\n"
    before = "status before finalize :job.jobStatus={}"
    logs = {
        "finished": ["start the report of the job", before.format("finished"), "updating the job in the DB"],
        "failed": ["start the report of the job", before.format("failed"), "updating the job in the DB"],
        "clean": ["archive start status=finished", "archive done"],
        "ran": ["archive start status=finished", "set jobStatus=failed due to inconsistent input", "archive done"],
        "twice": ["archive start status=finished", "archive done", "archive start status=finished", "archive done"],
        "both": ["archive start status=finished", "archive done", "updating the unmerged jobs"],
    }
    for name, text in logs.items():
        (tmp_path / f"{name}.log").write_text("".join(stamp.format(n, t) for n, t in enumerate(text)))
    code = """
        import json
        from bamboo.querier import changed as CH
        from bamboo.querier import rows as R
        from bamboo.querier.records import read_log, runs_for
        from bamboo.querier.which_ran import Candidate, Verdict

        R.ROWS_FROM = "walk"
        got = {}
        for name, adder, db, writes in [("stays", "finished", "clean", ["write"]), ("ran", "finished", "ran", ["write"]),
                                        ("twice", "finished", "twice", ["write"]), ("changed", "failed", "clean", ["write"]),
                                        ("unmerged", "finished", "both", ["write", "unmerged"]),
                                        ("raise", "finished", "clean", ["write", "raise"])]:
            question, = runs_for(read_log(f"{adder}.log"), "1")
            inner, = runs_for(read_log(f"{db}.log"), "1")
            verdicts, seens, runs_of = [], {}, {}
            mine = [("assign", "positive"), ("finalize", "positive")] + ([("raise", "kept")] if "raise" in writes else [])
            arms = [("pandaserver/a.py", "process", mine, question),
                    ("pandaserver/db.py", "archive", [("write", "kept")], inner)]
            if "unmerged" in writes:
                arms.append(("pandaserver/db.py", "unmerged", [("unmerged", "kept")], inner))
            for rel, func, lines, run in arms:
                c = Candidate(rel, func, [AT[k] for k, _ in lines], [run.log.name], run=run)
                seens[c.owner] = R.seen(R.function(rel, func, tuple(c.lines)), run, "outside")
                runs_of[c.owner] = run
                verdicts += [Verdict(c, AT[k], kind) for k, kind in lines]
            call = AT["call"]
            ch = CH.last_change(verdicts, seens, runs_of, question, {id(inner): f"pandaserver/a.py:{call}"},
                                "jobStatus", "finished", None)
            got[name] = None if ch is None else {
                "dropped": [[CH._name(v), why] for v, why in ch.dropped], "left": [CH._name(v) for v in ch.left],
                "chosen": CH._name(ch.chosen) if ch.chosen else None, "how": ch.how}
        print(json.dumps(got))
    """
    out = _ok(_run(tmp_path, code.replace("AT", repr(at))))
    got = json.loads(out.splitlines()[-1])
    assign, finalize, write = f"a.py::process:{at['assign']}", f"a.py::process:{at['finalize']}", f"db.py::archive:{at['write']}"
    # the row before finalize shows 'finished': the call is a stay (R1), the way inside it unread; archive's start row
    # shows it too, and the one way to change it on the way there, 'failed', did not run -- its must row is missing
    # (R2): the assignment from the report is the last change
    stays = got["stays"]
    assert stays["dropped"] == [
        [finalize, f"rule R: a stay, self.job.jobStatus already held 'finished' before it (row process:{at['row']} at finished.log:2, "
                   "nothing on the way binding it; what the call runs before its write not read)"],
        [write, f"rule R: a stay, job.jobStatus already held 'finished' before it in its own frame (row archive:{at['start']} at "
                f"clean.log:1, nothing on the way binding it; not run: db.py:{at['failed']} (must row(s) [{at['inconsistent']}] "
                "not in the run))"],
    ]
    assert stays["chosen"] == assign and stays["how"] == "the last change in the pass"
    # the 'failed' arm printed its row: the start row's value is not carried, and the write is read at the call, where
    # its own target is bound before it -- it may be the change
    assert got["ran"]["dropped"] == [stays["dropped"][0]] and got["ran"]["chosen"] is None
    assert got["ran"]["how"] == f"2 may have changed it last in the pass: {write}, {assign}"
    # the start row printed twice: not read as the one call that ran the write
    assert got["twice"] == got["ran"]
    # another value before finalize: the call is no stay, and no write of another value either -- it changed it last
    assert [name for name, _ in got["changed"]["dropped"]] == [write]
    assert got["changed"]["chosen"] == finalize and got["changed"]["how"] == "the last change in the pass"
    # two writes inside the one call, no order between them: archive's is a stay read without the order, and the
    # order is read between the rest -- the unmerged job's write (another object, K58) is left
    unmerged = got["unmerged"]
    assert [name for name, _ in unmerged["dropped"]] == [finalize, write] and unmerged["chosen"] is None
    assert unmerged["how"] == f"2 may have changed it last in the pass: db.py::unmerged:{at['unmerged']}, {assign}"
    # a raise writes through a handler, not through the call that builds its exception: none of the pass's writes
    assert got["raise"] == stays


def test_a_call_keeps_a_field_where_the_maps_reach_from_it_alone_assigns_it_no_other_value_in_memory(tmp_path):
    out = _ok(
        _run(
            tmp_path,
            """
            import json
            from types import SimpleNamespace as N
            from bamboo.codemap.models import Anchor, Branch, JunctionNode, ReachCall, ReachHole, ReachNode
            from bamboo.querier import mapdump, rerun

            def node(owner, start, end, **k):
                return ReachNode(name=owner, map_id="m", derived_from="d", owner=owner, function=owner.rsplit(".", 1)[-1].split("::")[-1],
                                 anchor=Anchor(package="p", file=owner.split("::")[0], line_start=start, line_end=end), **k)

            def junction(owner, *outcomes, pre=()):
                return JunctionNode(name=JunctionNode.make_name("m", "JobSpec.jobStatus", owner), map_id="m", derived_from="d",
                                    subject="JobSpec.jobStatus", owner=owner,
                                    branches=[Branch(outcome=o, line=2 + n, row_precondition=list(pre)) for n, o in enumerate(outcomes)],
                                    anchor=Anchor(package="p", file=owner.split("::")[0], line_start=2, line_end=4))

            rows = [node("p/a.py::f", 1, 20, calls=[
                        ReachCall(line=4, callee="same()", to="p/s.py::same"),
                        ReachCall(line=6, callee="other()", to="p/o.py::other"),
                        ReachCall(line=8, callee="row_only()", to="p/r.py::row_only")],
                        holes=[ReachHole(line=10, callee="x.get", why="the receiver's class is not declared"),
                               ReachHole(line=11, callee="x.other", why="the receiver's class is not declared")]),
                    node("p/s.py::same", 1, 9), node("p/o.py::other", 1, 9), node("p/r.py::row_only", 1, 9)]
            junctions = [junction("p/s.py::same", "failed"), junction("p/o.py::other", "finished"),
                         junction("p/r.py::row_only", "finished", pre=["jobStatus=defined"])]
            mapdump._FRAGMENT[:] = [N(reach_graph=rows, junctions=junctions)]
            r = object.__new__(rerun.Rerun)
            r._graph_read, r._kept, r._in_memory = None, {}, {}
            keeps = r._keeps("JobSpec.jobStatus")
            fn = N(rel="p/a.py", node=N(body=[N(lineno=2)]))
            at = lambda line: N(lineno=line, end_lineno=line)
            print(json.dumps([keeps(fn, at(n), "failed") for n in (4, 6, 8, 10, 11, 12)] + [keeps(fn, at(4), "finished"), len(r._kept)]))
            """,
        )
    )
    same, other, row_only, closed, hole, none, off, asked = json.loads(out.splitlines()[-1])
    # a call whose reach assigns only the value carried keeps it; one assigning another value does not
    assert same == 0 and other is None
    # in memory the row is not read: an arm whose row must hold 'defined' still moves the object
    assert row_only is None
    # a hole whose name no function reaching an arm has is closed by it, and counted; one whose name such a function
    # has may be the way to it
    assert closed == 1 and hole is None
    # no call the reach places at the lines: it goes on to nothing
    assert none == 0
    # the value carried decides: 'same' moves a 'finished' off
    assert off is None
    # each (function, lines, subject, value) asked once
    assert asked == 7


def test_a_print_that_begins_with_a_hole_is_read_on_the_line_its_markless_logger_wrote(tmp_path):
    (_corpus(tmp_path) / "pandaserver" / "f.py").write_text(
        "def finish(job):\n"
        "    logger.debug(f\"{job.PandaID} Finisher : Kill\")\n"
        "    logger.debug(f\"{job.PandaID}  Priority:{job.currentPriority} End:{job.endTime}\")\n"
        "    logger.debug(\"Finisher : Kill all\")\n"
        "    logger.debug(\"update\")\n"
        "    job.jobStatus = 'failed'\n"
    )
    stamp = "2026-01-01 00:00:00,000 panda.log.datasetManager: DEBUG    {}\n"
    (tmp_path / "f.log").write_text("".join(stamp.format(t) for t in [
        "7392812614 Finisher : Kill", "7392812614  Priority:900 End:2026-10-31 12:45:38", "<PandaID=1> Finisher : Kill all",
        "updateDatasetLocality_JEDI < taskID=1 insert"]))
    code = """
        import json
        from bamboo.querier import rows as R
        from bamboo.querier.records import read_log

        R.ROWS_FROM = "walk"
        fn = R.function("pandaserver/f.py", "finish", (6,))
        log = read_log("f.log")
        got = {}
        for row in fn.rows:
            got[row.line] = [[row.matches(rec, "outside")[0] > 0, row.capture(rec), len(row.texts(rec))] for rec in log.records]
        print(json.dumps([[rec.token, rec.message] for rec in log.records]))
        print(json.dumps(got))
    """
    out = _ok(_run(tmp_path, code)).splitlines()
    parsed, got = json.loads(out[-2]), {int(k): v for k, v in json.loads(out[-1]).items()}
    # a logger with no mark: the parser takes the message's first word for the token; a mark that never closes, its name
    assert parsed[0] == ["7392812614", "Finisher : Kill"] and parsed[3][0] == "updateDatasetLocality_JEDI"
    # the print beginning with a hole finds its record on the line as written, and reads the hole there
    kill, priority, literal, update = got[2], got[3], got[4], got[5]
    assert kill[0][0] is True and kill[0][1] == {"job.PandaID": "7392812614"} and kill[0][2] == 2
    assert priority[1][0] is True and priority[1][1].get("job.currentPriority") == "900"
    assert kill[1][0] is False and priority[0][0] is False
    # a print with no hole first is read on the message alone, as before: "update" is not found in a mark's name
    assert literal[2][0] is True and literal[2][2] == 1
    assert update[3][0] is False and update[3][2] == 1


# datasetManager's FinisherThr.run, cut down: each job set failed in the loop, all written by one call after it
_BATCHED = """
def run(jobs):
    up_jobs = []
    for job in jobs:
        logger.debug(f"{job.PandaID} Finisher : Kill")
        job.jobStatus = "failed"
        up_jobs.append(job)
    logger.debug("updating ...")
    taskBuffer.updateJobs(up_jobs, False)


def each(jobs):
    for job in jobs:
        job.jobStatus = "failed"
        taskBuffer.updateJobs([job], False)
"""


def test_a_writer_that_writes_its_loop_s_objects_after_it_is_read_on_its_latest_run_before_the_row(tmp_path):
    (_corpus(tmp_path) / "pandaserver" / "w.py").write_text(_BATCHED)
    lines = _BATCHED.splitlines()
    arm, call, loop, call2 = (lines.index(s) + 1 for s in [
        '        job.jobStatus = "failed"', "    taskBuffer.updateJobs(up_jobs, False)", "    for job in jobs:",
        "        taskBuffer.updateJobs([job], False)"])
    # each's assignment is the second of the same text
    arm2 = len(lines) - lines[::-1].index('        job.jobStatus = "failed"')
    stamp = "{} panda.log.X: DEBUG    <PandaID=1> {}\n"
    (tmp_path / "w.log").write_text(stamp.format("2026-01-01 00:10:28,854", "Finisher : Kill"))
    (tmp_path / "old.log").write_text(stamp.format("2025-12-31 23:59:00,000", "Finisher : Kill"))
    code = """
        import json
        from types import SimpleNamespace as N
        from bamboo.codemap.models import KEY_HANDED
        from bamboo.querier import mapdump, rerun
        from bamboo.querier.records import read_log, runs_for
        from bamboo.querier.which_ran import Candidate

        def kf(func, statement, line, unread=KEY_HANDED):
            return N(owner=f"pandaserver/w.py::{func}", unread=unread, object="job", table="", statement=statement,
                     writes={"JobSpec.jobStatus": [line]}, handed_to=["pandaserver/t.py::TaskBuffer.updateJobs"])

        mapdump._FRAGMENT[:] = [N(key_froms=[kf("run", CALL, ARM), kf("each", CALL2, ARM2)], function_prints=[])]
        found = rerun.batched("pandaserver/w.py", "run", [ARM], "JobSpec.jobStatus")
        got = {"batched": [found[0], found[2]] if found else None,
               "inside": rerun.batched("pandaserver/w.py", "each", [ARM2], "JobSpec.jobStatus"),
               "other": rerun.batched("pandaserver/w.py", "run", [ARM], "JobSpec.lockedBy")}
        r = object.__new__(rerun.Rerun)
        r.data = {"statuslog": {"states": [{"status": "transferring", "time": "2026-01-01 00:00:00"},
                                           {"status": "failed", "time": "2026-01-01 00:10:42"}]}}
        state = r.data["statuslog"]["states"][1]
        for name, holes in [("w", {}), ("w holed", None), ("old", {}), ("w no row", {})]:
            log = read_log(f"{name.split()[0]}.log")
            c = Candidate("pandaserver/w.py", "run", [ARM], [log.name])
            r.lines = []
            handed = r._batched([c], {}, {log.name: runs_for(log, "1")}, {log.name: ["a part"]} if holes is None else holes,
                                None if name.endswith("no row") else state, "2026-01-01 00:10:42", "JobSpec.jobStatus")
            got[name] = [{k: [v.first, v.last] for k, v in handed.items()}, c.run is not None, r.lines]
        print(json.dumps(got))
    """
    out = _ok(_run(tmp_path, code.replace("CALL2", str(call2)).replace("ARM2", str(arm2)).replace("CALL", str(call)).replace("ARM", str(arm))))
    got = json.loads(out.splitlines()[-1])
    # the job assigned in the loop and handed on after it: a batch; handed on in the loop, or another subject: not
    assert got["batched"] == [arm, loop] and got["inside"] is None and got["other"] is None
    # the key's run between the row before and the row, its log searched whole: the candidate's run, said so
    handed, set_run, said = got["w"]
    assert handed == {"pandaserver/w.py::run": ["2026-01-01 00:10:28,854", "2026-01-01 00:10:28,854"]} and set_run
    assert said == [f"  run:{arm} writes in a batch (assigned in the loop at {loop}, handed on to TaskBuffer.updateJobs at {call}): "
                    "its run is the key's latest from the row before (2026-01-01 00:00:00) to 2026-01-01 00:10:42, "
                    "2026-01-01 00:10:28,854..2026-01-01 00:10:28,854 in w.log, the log searched with no hole; it rests on no "
                    "other pass of it printing the key between"]
    # a part of the log not searched, or the run before the row before: none held, said, nothing asked
    for name in ("w holed", "old"):
        assert got[name][:2] == [{}, False]
        assert got[name][2][0].endswith("no run of the key held from the row before (2026-01-01 00:00:00) to 2026-01-01 00:10:42; not asked")
    # no row read: nothing
    assert got["w no row"] == [{}, False, []]


def test_a_call_leaves_the_path_knowing_nothing_of_what_it_changes(tmp_path):
    checker = (
        "\nclass Checker:\n    def __init__(self):\n        self.logger = logger\n        self.flag = None\n        self.result = None\n\n"
        "    def check(self, cond):\n        self.logger.debug('checking the input of the job')\n        if cond:\n"
        "            self.flag = None\n            self.probe()\n            self.logger.debug('probed the input of the job once')\n"
        "            if self.flag is None:\n                self.result = 'none found'\n"
        "        self.logger.debug('done with the input of the job')\n\n    def probe(self):\n        self.flag = compute()\n"
    )
    (_corpus(tmp_path) / "pandaserver" / "f.py").write_text(checker)
    # self let out: the call is read as before, as one that may change any of self
    (_corpus(tmp_path) / "pandaserver" / "g.py").write_text(checker + "\n    def register(self):\n        REGISTRY.append(self)\n")
    stamp = "2026-01-01 00:00:0{},000 panda.log.X: DEBUG    <PandaID=1> {}\n"
    (tmp_path / "x.log").write_text(
        "".join(stamp.format(i, t) for i, t in enumerate(["checking the input of the job", "probed the input of the job once",
                                                            "done with the input of the job"]))
    )
    out = _ok(
        _run(
            tmp_path,
            """
            import json
            from bamboo.querier import printed as P
            from bamboo.querier import rows as R
            from bamboo.querier.records import read_log, runs_for

            R.ROWS_FROM = "walk"
            run, = runs_for(read_log("x.log"), "1")
            got = []
            for file in ("f.py", "g.py"):
                fn = R.function(f"pandaserver/{file}", "check", (15,))
                proof = P.prove(fn, R.seen(fn, run, "outside"), R.statement_at(fn.node, 15))
                got.append([proof.kind, proof.reason])
            print(json.dumps(got))
            """,
        )
    )
    # probe binds self.flag: the None bound before it says nothing of the test after, and a path skips the arm --
    # read from what probe binds, or, self let out, from the call handed self
    for kind, reason in json.loads(out.splitlines()[-1]):
        assert kind == "none" and "miss it" in reason


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


def test_h6_names_a_rows_host_only_for_a_door_that_asks_by_machine(tmp_path):
    out = _ok(
        _run(
            tmp_path,
            _M2_SETUP + """
    r.opened = []
    r.data["statuslog"]["states"] = [
        {"status": "pending", "time": "2026-10-07 13:06:45", "host": "aipanda087.cern.ch"},
        {"status": "defined", "time": "2026-10-07 13:06:45", "host": "aipanda093.cern.ch"},
        {"status": "failed", "time": "2026-10-07 13:08:03"},
    ]
    # no make call: every row is asked, and the stop is at the makers' need
    rerun.O.instances_in = lambda rel: ["Setupper"]
    rerun.O.calls_through = lambda receiver: []
    r._ask = lambda obj, logs, asking: ({}, [], {})
    q = rerun.Missing("user.x_sub01", N(rel="pandaserver/dataservice/setupper.py"), N(last="2026-10-07 13:08:04,059"))
    said = []
    for door in (True, False):
        r.lines, r.logs = [], N(needs_window=door)
        try:
            r.missing(q)
        except rerun.Stop:
            pass
        said.append([line for line in r.lines if line.startswith("-- ") or "statuslog row(s)" in line])
    print(json.dumps(said))
    """,
        )
    )
    by_machine, files = json.loads(out.splitlines()[-1])
    rows = "3 statuslog row(s) by the run's last record 2026-10-07 13:08:04,059: each asked at its second, in turn, until a run holds it"
    none = ": 0 run(s) of the name hold that second, 0 with a make call's print and the job's key"
    # the door that asks by machine asks each row on its host, every machine where the row names none
    assert by_machine == [
        rows,
        "-- 'pending' at 2026-10-07 13:06:45 on aipanda087.cern.ch" + none,
        "-- 'defined' at 2026-10-07 13:06:45 on aipanda093.cern.ch" + none,
        "-- 'failed' at 2026-10-07 13:08:03 on every machine" + none,
    ]
    # the door over files reads the files given, whatever machine wrote them
    assert files == [
        rows,
        "-- 'pending' at 2026-10-07 13:06:45 in the files given, their machine not known" + none,
        "-- 'defined' at 2026-10-07 13:06:45 in the files given, their machine not known" + none,
        "-- 'failed' at 2026-10-07 13:08:03 in the files given, their machine not known" + none,
    ]


_PROXY = '''\
class DBProxy:
    def updateTask_JEDI(self, spec, criteria, updateDEFT):
        sql = "UPDATE ATLAS_PANDA.JEDI_Tasks SET status=:status WHERE jediTaskID=:jediTaskID"
        self.cur.execute(sql, {})
        if updateDEFT:
            sqlD = "UPDATE ATLAS_DEFT.T_TASK SET status=:status WHERE taskid=:jediTaskID"
            self.cur.execute(sqlD, {})
'''


def test_s5_reads_the_facade_the_receiver_is_declared_to_be(tmp_path):
    corpus = _corpus(tmp_path)
    files = {
        "pandaserver/taskbuffer/OraDBProxy.py": _PROXY,
        "pandajedi/jedicore/JediDBProxy.py": _PROXY,
        "pandaserver/taskbuffer/TaskBuffer.py": '''\
class TaskBuffer:
    def updateTask_JEDI(self, spec, criteria, updateDEFT=True):
        with self.proxyPool.get() as proxy:
            return proxy.updateTask_JEDI(spec, criteria, updateDEFT)
''',
        "pandajedi/jedicore/JediTaskBuffer.py": '''\
from pandaserver.taskbuffer.TaskBuffer import TaskBuffer

class JediTaskBuffer(TaskBuffer):
    def updateTask_JEDI(self, spec, criteria, updateDEFT=False):
        with self.proxyPool.get() as proxy:
            return proxy.updateTask_JEDI(spec, criteria, updateDEFT)
''',
        "pandajedi/jedidog/Dog.py": '''\
from pandajedi.jedicore.JediTaskBuffer import JediTaskBuffer

class Dog:
    def __init__(self, taskBufferIF: JediTaskBuffer):
        self.taskBufferIF = taskBufferIF

    def run(self, spec):
        return self.taskBufferIF.updateTask_JEDI(spec, {})

class Stray:
    def __init__(self, taskBufferIF):
        self.taskBufferIF = taskBufferIF

    def run(self, spec):
        return self.taskBufferIF.updateTask_JEDI(spec, {})
''',
    }
    for rel, text in files.items():
        (corpus / rel).parent.mkdir(parents=True, exist_ok=True)
        (corpus / rel).write_text(text)
    out = _ok(
        _run(
            tmp_path,
            """
            import ast
            import json
            from bamboo.querier.rerun import proxy_reading, receiver_methods

            call = ast.parse("self.taskBufferIF.updateTask_JEDI(spec, {})", mode="eval").body
            got = {}
            for name, line in (("declared", 8), ("undeclared", 15)):
                classes, theirs = receiver_methods("updateTask_JEDI", call, "pandajedi/jedidog/Dog.py", line)
                reading = proxy_reading("updateTask_JEDI", call, theirs if classes else None)
                got[name] = [classes, sorted(theirs), reading[3], reading[5]]
            # a declared class with no method of the name: no facade's at that receiver
            got["not its method"] = proxy_reading("updateTask_JEDI", call, set())
            print(json.dumps(got))
            """,
        )
    )
    got = json.loads(out.splitlines()[-1])
    # JediTaskBuffer's updateTask_JEDI hands updateDEFT=False on: the DEFT table is not read, nor its statement run
    assert got["declared"] == [
        ["pandajedi/jedicore/JediTaskBuffer.py::JediTaskBuffer"], [["pandajedi/jedicore/JediTaskBuffer.py", 4]], ["JEDI_Tasks"],
        ["pandajedi/jedicore/JediDBProxy.py:4"],
    ]
    # no declaration: every facade of the name, TaskBuffer's True among them
    assert got["undeclared"] == [[], [], ["JEDI_Tasks", "T_TASK"],
                                 ["pandaserver/taskbuffer/OraDBProxy.py:4", "pandaserver/taskbuffer/OraDBProxy.py:7", "pandajedi/jedicore/JediDBProxy.py:4"]]
    assert got["not its method"] is None


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
    for statements, ordered in (([st(10)], []), ([st(10, True)], [st(10, True)])):
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
    unordered, ordered = json.loads(out.splitlines()[-1])
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
    # each END as said at its step (result.json's outcome.record renders "END " before it)
    assert ends == [["L2", "lost on order: here"]]
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


_LOOPED = """def f(job):
    for f in job.Files:
        x = "a"
    x = "b"
    d = {}
    for f in job.Files:
        d[f.lfn] = 1
    e = {}
    for s in job.Datasets:
        e[s.name] = 2
    return x, d, e
"""


def test_a_binding_in_a_loop_over_rows_not_had_is_unknown_and_a_loop_over_no_rows_wrote_nothing(tmp_path):
    (_corpus(tmp_path) / "pandaserver" / "a.py").write_text(_LOOPED)
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
            made = [("x", 3, "'a'", "=", "loop at 2: for f in job.Files"), ("x", 4, "'b'", "=", ""),
                    ("d", 5, "{}", "=", ""), ("d", 7, "1", "=", "loop at 6: for f in job.Files"),
                    ("e", 8, "{}", "=", ""), ("e", 10, "2", "=", "loop at 9: for s in job.Datasets")]
            steps = [N(name=n, owner=owner, kind="binding", line=line, value=v, operator=op, guards=[], unseen=[u] if u else [],
                       reads=[], terminal="", detail="", depth=1, cls="") for n, line, v, op, u in made]
            # the rows file has the job's datasets, none of them, and not its files
            told = []
            walk = G.Walk(A.given(steps), E.Env(), {"job.Datasets": {"rows": "datasets", "entity": "JediDatasetSpec"}},
                          {"datasets": []}, ask=lambda iterable, step, env: told.append([iterable, step.line]))
            show = lambda v: "UNKNOWN" if v is E.UNKNOWN else v
            got = {"x": [[o.step.line, show(o.value), show(o.reached), o.why] for o in walk.origins("x", owner)],
                   "chosen": show(walk._choose("x", owner, None, {})[0])}
            for name in ("d", "e"):
                value, notes = walk.build(name, owner)
                keys = walk.keys(name, owner)
                got[name] = [value, notes[:1], [o.value for o in keys.origins], keys.opaque, keys.complete()]
            got["told"] = sorted(map(list, {tuple(t) for t in told}))
            print(json.dumps(got))
            """,
        )
    )
    got = json.loads(out.splitlines()[-1])
    # the walk tells its asker of each loop over the files it does not have, not of the datasets it has
    assert got["told"] == [["job.Files", 3], ["job.Files", 7]]
    # x = "a" may have run, for files not known: kept, so "b" is not taken as all x held
    assert got["x"] == [[3, "UNKNOWN", "UNKNOWN", "@3 runs in a loop over rows this investigation does not have"],
                        [4, "b", True, ""]]
    assert got["chosen"] == "UNKNOWN"
    assert got["d"][0] is None and got["d"][1] == ["@7 runs over rows this investigation does not have"]
    assert got["d"][3] == ["@7 runs in a loop over rows this investigation does not have"] and got["d"][4] is False
    # a loop over no rows ran no write: e is built empty and its key set is whole
    assert got["e"][0] == {} and got["e"][2:] == [[], [], True]


def test_a_collection_the_rows_file_lacks_is_asked_by_its_rows_class_key_and_attribute(tmp_path):
    out = _ok(
        _run(
            tmp_path,
            """
            import json
            from types import SimpleNamespace as N
            from bamboo.querier import evaluate as E
            from bamboo.querier import mapdump, rerun

            # the dump's tables name a class's identity
            mapdump._FRAGMENT[:] = [N(tables=[N(spec_class="FileSpec", identity_attrs=["FileSpec.row_ID"]),
                                              N(spec_class="JobSpec", identity_attrs=["JobSpec.PandaID"])])]
            r = object.__new__(rerun.Rerun)
            r.rows_asks, r.hop = {}, "R2"
            # the walk spells the case's row by its name in the rows file, whose classes give its class
            r.data = {"classes": {"job": "JobSpec"}}
            env = E.Env()
            env.table["job"] = {"PandaID": E.Value(7, "the rows file"), "jobStatus": E.Value("failed", "the rows file")}
            step = N(line=7, owner="pandaserver/a.py::f")
            for iterable in ("<job>.Files", "<job>.Files", "tmp_list"):
                r.ask_collection(iterable, step, env)
            print(json.dumps(list(r.rows_asks.values())))
            """,
        )
    )
    asked = json.loads(out.splitlines()[-1])
    # once, by the job's key the dump names; a local name is no row to ask of
    assert asked == [{"rows": {"kind": "collection", "row": {"entity": "JobSpec", "key": {"PandaID": 7}}, "attr": "Files", "entity": None},
                      "step": "R2", "why": "@7 f loops over <job>.Files, which the rows file does not have"}]


def test_l3_counts_the_rows_held_ahead_at_the_move_by_the_orders_items_before_the_key_and_the_key_within(tmp_path):
    out = _ok(
        _run(
            tmp_path,
            _M2_SETUP + """
    from bamboo.querier import rows as R

    r.opened, r.ends = [], []
    r.args.patterns, r.args.key = "outside", "7"
    sql = ("SELECT * FROM (SELECT PandaID FROM ATLAS_PANDA.jobsActive4 tab WHERE {} ORDER BY {}) WHERE ROWNUM<=:njobs")
    plain = "jobStatus=:s AND computingSite=:site AND resource_type=:rt"
    # the call's print, cut into its text and binds as the map's printed_by says
    R.function = lambda rel, func, within=(): N(rows=[N(line=12, matches=lambda rec, mode: (1, ""))])
    rerun.sqltext.cut = lambda text, lead, gap: (text, {":s": "activated", ":site": "pic", ":rt": "SCORE"})
    rerun.Rerun._row_of = lambda self, cls: {"PandaID": {"value": 7}, "currentPriority": {"value": 100}}
    held = "q@10 held 'activated' at the move, by the order's items before the key"
    left = "q@10 left 'activated' in the window, by the value it went to and the key"

    def run(took, other, answers=None, where=plain):
        r.lines, r.ends, r.rows_asks = [], [], {}
        r.data["answers"] = answers or {}
        st = N(owner="pandaserver/a.py::q", line=10, limits=["<= :njobs"], order_by=[took, other],
               printed_by=[N(line=12, lead="", gap="")], order_holes=[])
        record = N(text=sql.format(where, took), span="a.log@1", stamp="2026-01-05 00:00:00,000")
        try:
            r.ordered(rerun.Ordered(st, N(records=[record]), q))
            return ["END", f"END {r.ends[-1][1]}"]
        except rerun.Stop as e:
            return ["STOP", str(e), [a["rows"]["question"] | {"name": a["rows"]["name"]} for a in r.rows_asks.values()],
                    next((x for x in r.lines if "the row sorts by" in x), None)]

    def at(*groups, first="2026-01-02 00:00:00"):
        return {"source": "the hand's SQL", "rows": [
            {"CURRENTPRIORITY": p, "N": n, "N_KEY_AHEAD": k, "N_SELF": me, "FIRST_ENTERED": first} for p, n, k, me in groups]}

    window = {"source": "the hand's SQL", "rows": [
        {"next_value": "sent", "currentpriority": 100, "n": 3, "n_key_ahead": 1, "min_h": 1, "median_h": 1.5, "max_h": 2}]}
    groups = [(200, 5, 3, 0), (100, 4, 2, 1), (50, 9, 9, 0)]
    alone = run("currentpriority desc, pandaid asc", "pandaid asc")
    # the items after the key decide nothing: modificationtime is no group
    grouped = run("pandaid asc", "currentpriority desc, pandaid asc, modificationtime asc", {held: at(*groups), left: window})
    keyless = run("pandaid asc", "currentpriority desc", {held: at(*groups), left: window})
    print(json.dumps({
        "alone": alone,
        "grouped": grouped,
        "keyless": keyless,
        "least": run("pandaid asc", "currentpriority desc, pandaid asc", {held: at(*groups, first="2025-12-28 00:30:00"), left: window}),
        "not self": run("pandaid asc", "currentpriority desc, pandaid asc", {held: at((100, 4, 2, 0), (50, 1, 1, 1)), left: window}),
        "lacking": run("pandaid asc", "currentpriority desc, pandaid asc",
                       {held: {"source": "s", "rows": [{"currentpriority": 100, "n": 4}]}, left: window}),
        "mixed": run("pandaid asc", "currentpriority desc, pandaid asc",
                     where="(jobStatus=:s OR computingSite=:site) AND resource_type=:rt"),
    }))
    """,
        )
    )
    got = json.loads(out.splitlines()[-1])
    # an order of the key alone is one group: the two questions asked, by their structure beside the SQL; the terms on
    # the subject go (the history's value stands for them), those on a column the history holds are read at the move,
    # the others on the row as it is now
    kind, why, asks, sorts = got["alone"]
    assert kind == "STOP" and why.startswith("L3: the answers to")
    assert sorts.startswith("  under 'pandaid asc' the row sorts by its key alone, pandaid asc -> ")
    at_ask, left_ask = asks
    assert at_ask["name"] == "q@10 held 'activated' at the move, by the order's items before the key"
    assert at_ask["part"] == "H held at" and at_ask["group_by"] == [] and at_ask["key_order"] == "asc"
    assert at_ask["at"] == "2026-01-05 00:00:00" and at_ask["lookback"] == "2025-12-28 00:00:00"
    assert at_ask["terms_at"] == ["computingSite='pic'"] and at_ask["terms_now"] == ["resource_type='SCORE'"]
    assert at_ask["columns"] == ["n", "n_key_ahead", "n_self", "first_entered"]
    assert left_ask["part"] == "H moved on" and left_ask["columns"] == ["next_value", "n", "n_key_ahead", "min_h", "median_h", "max_h"]
    assert left_ask["terms_at"] == ["computingSite='pic'"] and left_ask["terms_now"] == ["resource_type='SCORE'"]
    # grouped by the items before the key, the key ranking within its own group: 5 in the group ahead, 2 of its own
    kind, end = got["grouped"]
    assert kind == "END" and end == (
        "END lost on order: q@10 returns the first rows of 'currentpriority desc, pandaid asc, modificationtime asc' or of "
        "'pandaid asc'; under the "
        "first, at the move (2026-01-05 00:00:00) 7 row(s) at the call's WHERE held 'activated' ahead of it (5 in groups "
        "ahead, 2 of its own group's 4 with a key ahead); its group left 'activated': 3 to sent after 1-2 h (1 with a key "
        "ahead) (this row to 'sent'); read as now: resource_type='SCORE', the order's currentpriority; the calls that "
        "passed it over are not read (their orders and binds); the call that took it used 'pandaid asc'")
    # an order without the key: its ties are in an order the database picks, and none of its own group is counted ahead
    assert "(5 in groups ahead, its own group's 4, their order among them not known)" in got["keyless"][1]
    # an earliest entry within an hour of the lookback: "at least"
    assert "-- at least: the earliest entry counted, 2025-12-28 00:30:00, is within an hour of the lookback" in got["least"][1]
    # the row itself not in its own group: a term read as now fails for it
    assert got["not self"][:2] == ["STOP", "L3: the row itself, once, in its own group of those that held 'activated' at "
                                   "the move under the call's WHERE (the answer counts it 0 time(s) there, 1 in all; the "
                                   "terms read on the row as it is now: [\"resource_type='SCORE'\"])"]
    assert got["lacking"][1].startswith("L3: the answer to \"q@10 held 'activated' at the move, by the order's items before "
                                        "the key\" holding the column(s) ['n_key_ahead', 'n_self', 'first_entered']")
    # a term reading the subject with another column: neither the moment nor now reads it alone
    assert got["mixed"][1] == ("L3: a term of the call's WHERE on JobSpec.jobStatus alone, which the history's value stands "
                               "for ((jobStatus='activated' OR computingSite='pic') reads ['computingsite'] too)")


def test_l3s_questions_read_the_history_at_the_move_and_the_row_as_it_is_now(tmp_path):
    out = _ok(
        _run(
            tmp_path,
            """
            import datetime as dt
            import json
            from bamboo.querier import parts as PT

            h = PT.history("JobSpec.jobStatus")
            at = dt.datetime(2026, 1, 5)
            def held(cols, order):
                return PT.HeldAt(h, "activated", at, at - dt.timedelta(days=8), ["computingSite='pic'"], ["resource_type='SCORE'"],
                                 cols, "pandaid", 7, order)
            def moved(now):
                return PT.MovedOn(h, "activated", dt.datetime(2026, 1, 1), at, ["computingSite='pic'"], now, [], "pandaid", 7, "desc")
            print(json.dumps([PT.draw(held([], "asc")), PT.draw(held([("currentpriority", "desc")], None)), PT.draw(moved([])),
                              PT.draw(moved(["resource_type='SCORE'"]))]))
            """,
        )
    )
    alone, keyless, moved, moved_now = json.loads(out.splitlines()[-1])
    lines = alone.splitlines()
    entries, latest = next(x for x in lines if "modificationTime >= " in x), next(x for x in lines if "WHERE rn = 1" in x)
    # the terms the history holds, read on each key's last row before the move; the others on the row as it is now
    assert "jobStatus = 'activated'" in entries and "computingSite='pic'" in entries
    assert "modificationTime < TO_DATE('2026-01-05 00:00:00'" in entries and "TO_DATE('2025-12-28 00:00:00'" in entries
    assert latest == "  WHERE rn = 1 AND jobStatus = 'activated' AND computingSite='pic'),"
    assert "ORDER BY l.modificationTime DESC, l.modiftime_extended DESC) AS rn" in alone
    assert all("resource_type='SCORE'" in x for x in lines if x.startswith("    SELECT pandaid FROM ATLAS_PANDA"))
    # the key alone: one group, the rows ahead those with a smaller key, the row itself counted
    assert ("SELECT COUNT(*) AS n, COUNT(CASE WHEN h.pandaid < 7 THEN 1 END) AS n_key_ahead, COUNT(CASE WHEN h.pandaid = 7 "
            "THEN 1 END) AS n_self, MIN(h.entered) AS first_entered") in alone and "GROUP BY j." not in alone
    # an order without the key counts none ahead by it; grouped by the order's items
    assert "n_key_ahead" not in keyless and "GROUP BY j.currentpriority" in keyless
    # the window's moves: the key's rank by its order, and no join to the rows when nothing groups or filters them
    assert "COUNT(CASE WHEN s.pandaid > 7 THEN 1 END) AS n_key_ahead" in moved and "JOIN j" not in moved
    # the terms the history holds, read on the row each left; the others on the row as it is now
    left = moved_now.splitlines()
    assert "         LAG(l.computingsite) OVER (PARTITION BY l.pandaid ORDER BY l.modificationTime, l.modiftime_extended) AS computingsite," in left
    assert next(x for x in left if "WHERE before_value" in x).endswith("AND computingSite='pic'),")
    assert all("resource_type='SCORE'" in x for x in left if x.startswith("    SELECT pandaid FROM ATLAS_PANDA"))
    assert "FROM s JOIN j ON j.pandaid = s.pandaid" in moved_now and "resource_type" not in moved


def test_the_writers_the_map_hands_lose_the_arms_it_says_write_no_row(tmp_path):
    out = _ok(
        _run(
            tmp_path,
            """
            import asyncio, json
            from types import SimpleNamespace as N
            from bamboo.querier import mapdump, rerun

            off = N(why="no_write", functions=["pandaserver/a.py::C.g"], writes=[])
            junctions = [
                N(owner="a.py::C.g", subject="JediTaskSpec.splitRule",
                  branches=[N(line=10, outcome="x", off_row=None), N(line=11, outcome="y", off_row=off)]),
                # every arm off its row: the owner is no writer
                N(owner="a.py::C.h", subject="JediTaskSpec.splitRule", branches=[N(line=20, outcome="z", off_row=off)]),
                # a dump built before OS: no off_row
                N(owner="a.py::C.k", subject="JediTaskSpec.splitRule", branches=[N(line=30, outcome="w")]),
                N(owner="a.py::C.e", subject="JediTaskSpec.splitRule", branches=[]),
            ]

            async def writers_of(subject):
                return junctions

            mapdump.code_map = lambda who: N(writers_of=writers_of)
            r = object.__new__(rerun.Rerun)
            got = asyncio.run(r._writers_of("JediTaskSpec.splitRule"))
            # a dump's junction is a model: copied with the branches kept
            from bamboo.codemap.models import Anchor, Branch, JunctionNode

            subject, owner = "JediTaskSpec.splitRule", "pandajedi/a.py::C.g"
            model = JunctionNode(name=JunctionNode.make_name("panda", subject, owner), map_id="panda", derived_from="x",
                                 subject=subject, owner=owner, branches=[Branch(outcome="x", line=10), Branch(outcome="y", line=11)],
                                 anchor=Anchor(package="pandajedi", file="pandajedi/a.py", line_start=1))
            copied = rerun._with_branches(model, model.branches[:1])
            print(json.dumps([[[j.owner, [b.line for b in j.branches]] for j in got], [b.line for b in junctions[0].branches],
                              got[1] is junctions[2], [type(copied).__name__, [b.line for b in copied.branches],
                                                       [b.line for b in model.branches]]]))
            """,
        )
    )
    got, map_own, same, model = json.loads(out.splitlines()[-1])
    assert got == [["a.py::C.g", [10]], ["a.py::C.k", [30]], ["a.py::C.e", []]]
    # the map's junction is not changed, and one with nothing left out is handed on as it is
    assert map_own == [10, 11] and same is True
    assert model == ["JunctionNode", [10], [10, 11]]


def test_s1_names_the_writes_a_movers_rows_wait_on_that_the_map_states_and_keys_by_rows_at_the_value(tmp_path):
    out = _ok(
        _run(
            tmp_path,
            """
            import json
            from types import SimpleNamespace as N
            from bamboo.querier import mapdump, rerun

            asked = "JediTaskSpec.status=defined"
            restart, prepare = "s.py::restartTasksForContentsUpdate_JEDI", "c.py::prepareTasksToBeFinished_JEDI"
            site = lambda owner, line, selects, wakers=(): N(owner=owner, anchor=N(line_start=line), name=f"panda:{owner}@{line}",
                                                             selects=selects, wakers=list(wakers))
            wake = lambda owner, line, value, **kw: N(**{"pairs": [asked], "term": "tabD.status IN (:s)", "subject": "JediDatasetSpec.status",
                                                         "value": value, "owner": owner, "line": line, "stated": True,
                                                         "keyed_by": [], "keyed_at": [asked], **kw})
            keyed_by_restart = [f"panda:{restart}@1830"]
            keyed_by_prepare = [f"panda:{prepare}@3694", f"panda:{prepare}@3980"]
            sites = [
                site("c.py::getDatasetsToFeedContents_JEDI", 130, [asked], [
                    wake(restart, 1844, "toupdate", keyed_by=keyed_by_restart),
                    wake(restart, 1844, "toupdate", keyed_by=keyed_by_restart[::-1]),  # the same write twice: said once
                    wake(restart, 1889, "defined", keyed_by=keyed_by_restart, stated=False),  # tier 2: not said
                    wake(restart, 1900, "toupdate", keyed_by=keyed_by_restart, keyed_at=[]),  # keyed by rows of other values
                    wake(restart, 1910, "toupdate", keyed_by=keyed_by_restart, pairs=["JediTaskSpec.status=ready"]),
                ]),
                site("c.py::_get_tasks_datasets_with_unprocessed_inputs", 1871, [asked], [
                    wake(prepare, 4001, "done", keyed_by=keyed_by_prepare),
                    # keyed only by a statement selecting other values: its function's arms do not make it a candidate
                    wake(prepare, 4006, "done", keyed_by=[f"panda:{prepare}@3980"]),
                ]),
                site(restart, 1830, [asked]),
                site(prepare, 3694, [asked]),
                site(prepare, 3980, ["JediDatasetSpec.status=ready"]),
                site("x.py::other", 50, [asked], [wake(restart, 1844, "toupdate", keyed_by=keyed_by_restart)]),  # no mover's
            ]
            mapdump.fragment = lambda: N(read_sites=sites)
            statement = lambda owner, line: N(owner=owner, line=line)
            rerun.paths_of = lambda follow, writers, subject, observed: {
                "f.py::feed_contents_to_tasks": [statement("c.py::getDatasetsToFeedContents_JEDI", 130)],
                "g.py::runImpl": [statement("c.py::_get_tasks_datasets_with_unprocessed_inputs", 1871)],
                "k.py::byKey": [],
            }
            # prepareTasksToBeFinished_JEDI moves the task on from the value itself; restart does not
            rerun.arms_of = lambda writers, owner, subject, observed: {3720: "arm"} if owner == prepare else {}
            got = rerun.wakers_of(None, [], "JediTaskSpec.status", "defined")
            # a dropped mover is said as a waker of the movers kept only: not one kept, nor one only a dropped one waits on
            other = rerun.Wake("s.py::restartTasksForContentsUpdate_JEDI", "s.py::r@1830", "t", "x=y", "z.py::other", 5)
            kept_waker = rerun.Wake("f.py::feed_contents_to_tasks", "c.py::g@130", "t", "x=y", prepare, 6)
            entries = [{"owner": "f.py::feed_contents_to_tasks", "verdict": "ran"}, {"owner": restart, "verdict": "reaches-no-write"},
                       {"owner": prepare, "verdict": "no-log"}, {"owner": "z.py::other", "verdict": "reaches-no-write"}]
            wakes = {"f.py::feed_contents_to_tasks": [*got["f.py::feed_contents_to_tasks"], kept_waker], restart: [other]}
            print(json.dumps([{m: [[w.owner.split("::")[-1], w.line, w.writes, w.reader, list(w.candidate)] for w in ws]
                               for m, ws in got.items()},
                              [rerun.wake_said(w) for ws in got.values() for w in ws],
                              rerun.woken_said(got["f.py::feed_contents_to_tasks"]),
                              {o: [w.line for w in ws] for o, ws in rerun.woken_of(entries, wakes).items()}], default=str))
            """,
        )
    )
    got, said, woken, dropped = json.loads(out.splitlines()[-1])
    assert dropped == {"s.py::restartTasksForContentsUpdate_JEDI": [1844]}
    assert got == {
        "f.py::feed_contents_to_tasks": [
            ["restartTasksForContentsUpdate_JEDI", 1844, "JediDatasetSpec.status=toupdate", "getDatasetsToFeedContents_JEDI@130", []]
        ],
        "g.py::runImpl": [
            ["prepareTasksToBeFinished_JEDI", 4001, "JediDatasetSpec.status=done", "_get_tasks_datasets_with_unprocessed_inputs@1871",
             ["c.py::prepareTasksToBeFinished_JEDI"]],
            ["prepareTasksToBeFinished_JEDI", 4006, "JediDatasetSpec.status=done", "_get_tasks_datasets_with_unprocessed_inputs@1871", []],
        ],
    }
    assert said[0] == ("its rows wait also on restartTasksForContentsUpdate_JEDI@1844 writing JediDatasetSpec.status=toupdate, "
                       "which getDatasetsToFeedContents_JEDI@130 asks ('tabD.status IN (:s)'), for rows selected at the value")
    assert said[1].endswith("-- a candidate: prepareTasksToBeFinished_JEDI, whose rows key it, moves the row on itself")
    assert "candidate" not in said[2]
    assert woken == ("not a mover, but it writes JediDatasetSpec.status=toupdate at 1844 for rows selected at the value, which "
                     "feed_contents_to_tasks (getDatasetsToFeedContents_JEDI@130) asks: its waker (the map's wakers; "
                     "KNOWN_ISSUES.md K39)")


def test_an_attribute_no_history_records_takes_no_time_from_the_status_rows(tmp_path):
    out = _ok(
        _run(
            tmp_path,
            """
            import json
            from types import SimpleNamespace as N
            from bamboo.querier import rerun
            from bamboo.querier import rows as R

            R.ROWS_FROM = "walk"  # the reading's logs
            r = object.__new__(rerun.Rerun)
            r.lines, r.opened, r.hop, r.hand, r.hand_used = [], [], "", {}, []
            r.data = {"statuslog": {"states": [{"status": "running", "time": "2026-01-01 00:00:00"},
                                               {"status": "done", "time": "2026-01-02 00:00:00"}]}}
            logged = N(file="pandaserver/t.py", owner="pandaserver/t.py::TaskModule.updateTask_JEDI", lines=[1], log_files=["panda-DBProxy.log"])
            r._writers = lambda subject, value, key: [logged] if subject == "JediTaskSpec.status" else []

            def past(readings, subject, state):
                raise RuntimeError(f"past H2's time: {state['time']}")

            r._unborn = past
            got = {}
            for subject in ("JediTaskSpec.status", "JediTaskSpec.superStatus"):
                try:
                    r._which_wrote(subject, "done", "1")
                except rerun.Stop as e:
                    got[subject] = e.args[0]
                except RuntimeError as e:
                    got[subject] = str(e)
            print(json.dumps(got))
            """,
        )
    )
    got = json.loads(out.splitlines()[-1])
    # the task's status is the history's own: its row for 'done' gives the time
    assert got["JediTaskSpec.status"] == "past H2's time: 2026-01-02 00:00:00"
    # superStatus is spelled with the status's values, but no history records it: not the status's time -- its
    # writers' arms are asked (K37), and with none in the map it stops there
    assert got["JediTaskSpec.superStatus"] == "H2: a writer of JediTaskSpec.superStatus in the map, to say when it took 'done'"


def test_an_attributes_time_is_its_value_where_every_writer_writes_now_else_the_rows_its_writers_passes_write(tmp_path):
    out = _ok(
        _run(
            tmp_path,
            """
            import json
            from types import SimpleNamespace as N
            from bamboo.querier import mapdump, rerun

            def branch(line, outcome, *pairs, hole=""):
                return N(line=line, outcome=outcome, history=list(pairs), history_hole=hole)

            def pair(row, writes=(), why=""):
                return N(row=row, row_why=why, writes=list(writes), caller="pandajedi/r.py::Refiner.runImpl", call=226,
                         callee="self.taskBufferIF.updateTask_JEDI", order="after")

            upd = "pandaserver/t.py::TaskModule.updateTask_JEDI:1570"
            mapdump._FRAGMENT[:] = [N(
                junctions=[
                    N(subject="JediTaskSpec.rescueTime", owner="pandaserver/t.py::TaskModule.rescue", branches=[branch(737, "runtime(CURRENT_DATE)")]),
                    N(subject="JediTaskSpec.transUses", owner="pandajedi/r.py::Refiner.extract", branches=[branch(140, "runtime(p)", pair("can", [upd]))]),
                    N(subject="JediTaskSpec.walltime", owner="pandajedi/r.py::Refiner.extract", branches=[branch(150, "runtime(w)", pair("can", ["pandaserver/t.py::TaskModule.mark:9"]))]),
                    N(subject="JediTaskSpec.splitRule", owner="pandajedi/r.py::Refiner.extract", branches=[branch(160, "runtime(s)", pair("can", [upd]))]),
                    N(subject="JediTaskSpec.splitRule", owner="pandajedi/w.py::Dog.act", branches=[branch(20, "runtime(s)", pair("", why="not followed"))]),
                    # two writers: one whose pass can write any row; one whose line 1187's pass records 'defined' only
                    # and line 1188's 'ready' only
                    N(subject="JediTaskSpec.workDiskUnit", owner="pandajedi/r.py::Refiner.extract", branches=[branch(182, "runtime(u)", pair("can", [upd]))]),
                    N(subject="JediTaskSpec.workDiskUnit", owner="pandaserver/t.py::TaskModule.scout", branches=[
                        branch(1187, "runtime(u)", pair("can", ["pandaserver/t.py::TaskModule.mark:9"])),
                        branch(1188, "runtime(v)", pair("same", ["pandaserver/t.py::TaskModule.ready:5"]))]),
                    # on no row of the task: a birth (the INSERT that makes the row, 1188); one marking its runs by the
                    # task's key (408); one marking them by a job's (520)
                    N(subject="JediTaskSpec.taskName", owner="pandajedi/r.py::Refiner.extract", branches=[branch(126, "runtime(n)", pair("can", [upd]))]),
                    N(subject="JediTaskSpec.taskName", owner="pandaserver/s.py::getTasksToRefine_JEDI", branches=[branch(1188, "runtime(str(uuid.uuid4()))")]),
                    N(subject="JediTaskSpec.cpuTimeUnit", owner="pandajedi/r.py::Refiner.extract", branches=[branch(202, "runtime(c)", pair("can", [upd]))]),
                    N(subject="JediTaskSpec.cpuTimeUnit", owner="pandaserver/m.py::initialize_cpu_time_task", branches=[branch(408, "runtime(c)", pair("", why="not followed"))]),
                    N(subject="JediTaskSpec.cpuTime", owner="pandajedi/d.py::Dog.massage", branches=[branch(520, "runtime(t)", pair("", why="not followed"))]),
                ],
                reach_graph=[N(owner="pandaserver/t.py::TaskModule.updateTask_JEDI", history=[N(line=1570, values=["runtime(taskSpec.status)"])]),
                             N(owner="pandaserver/t.py::TaskModule.mark", history=[N(line=9, values=["defined"])]),
                             N(owner="pandaserver/t.py::TaskModule.ready", history=[N(line=5, values=["ready"])])],
                key_froms=[N(owner="pandaserver/s.py::getTasksToRefine_JEDI", writes={"JediTaskSpec.taskName": [1188]}, new_row=True)],
                tables=[N(spec_class="JediTaskSpec", rows_of="JediTaskSpec", identity_attrs=["JediTaskSpec.jediTaskID"],
                          creation=["JediTaskSpec.creationDate"]),
                        N(spec_class="JobSpec", rows_of="JobSpec", identity_attrs=["JobSpec.PandaID"], creation=["JobSpec.creationTime"])],
            )]
            # each function's run: its mark's holes' columns, the mark, and the prefix its rows print it in (as the map's)
            marks = {"initialize_cpu_time_task": ([["JobSpec.PandaID"], ["JediTaskSpec.jediTaskID", "JobSpec.jediTaskID"], []],
                                                  "PandaID={}; jediTaskID={}; siteID={}",
                                                  r"initialize_cpu_time_task\\ <\\ PandaID=.*?;\\ jediTaskID=.*?;\\ siteID=.*?\\ >\\ "),
                     "massage": ([["JobSpec.PandaID"]], "PandaID={}", r"massage\\ <\\ PandaID=.*?\\ >\\ "),
                     "changeTaskSplitRulePanda": ([["JediTaskSpec.jediTaskID"]], "jediTaskID={}",
                                                  r"changeTaskSplitRulePanda\\ <\\ jediTaskID=.*?\\ >\\ "),
                     # a prefix of other holes than the mark's (another logger's): no text to ask
                     "odd": ([["JediTaskSpec.jediTaskID"]], "jediTaskID={}", r"odd\\ <\\ a=.*?\\ b=.*?\\ >\\ "),
                     # one of the mark's holes but not its text
                     "odd2": ([["JediTaskSpec.jediTaskID"]], "jediTaskID={}", r"odd2\\ <\\ taskid=.*?\\ >\\ ")}

            def prints(owner, lines=()):
                if owner.split("::")[-1] not in marks:
                    return None
                holes, mark, prefix = marks[owner.split("::")[-1]]
                return N(runs=[N(key=[N(columns=c) for c in holes], mark=mark)], statements=[N(rows=[N(prefix=prefix), N(prefix="")])])

            rerun.R._prints_of = prints
            r = object.__new__(rerun.Rerun)
            r.lines, r.opened, r.hop, r.hand, r.hand_used = [], [], "H2", {}, []
            states = [{"status": s, "time": t, "host": "h"} for s, t in
                      (("defined", "2026-10-29 07:08:40"), ("ready", "2026-10-29 07:08:41"), ("done", "2026-10-29 08:36:33"))]
            read = {"value": "x", "as_of": "2026-10-29 14:02:50"}
            r.data = {"statuslog": {"states": states, "as_of": "2026-10-29 14:03:01"}, "classes": {"task": "JediTaskSpec"},
                      "rows": {"task": {"jediTaskID": {"value": 53, "as_of": "2026-10-29 14:02:50"},
                                        "creationDate": {**read, "value": "2026-10-29T07:08:39"}, "cpuTimeUnit": read}}}
            r.args = N(subject="JediTaskSpec.taskName", key="53")
            reading = lambda owner, *lines: N(owner=owner, lines=list(lines))
            asks = {
                "now": ("JediTaskSpec.rescueTime", "2026-10-29 00:49:09", [reading("pandaserver/t.py::rescue", 737)]),
                "not a time": ("JediTaskSpec.rescueTime", "soon", [reading("pandaserver/t.py::rescue", 737)]),
                "any row": ("JediTaskSpec.transUses", "Atlas", [reading("pandajedi/r.py::extract", 140)]),
                "its rows": ("JediTaskSpec.walltime", "3", [reading("pandajedi/r.py::extract", 150)]),
                "one not placed": ("JediTaskSpec.splitRule", "x", [reading("pandajedi/r.py::extract", 160), reading("pandajedi/w.py::act", 20)]),
                "no writer": ("JediTaskSpec.transUses", "Atlas", []),
                "a birth": ("JediTaskSpec.taskName", "n", [reading("pandajedi/r.py::extract", 126), reading("pandaserver/s.py::getTasksToRefine_JEDI", 1188)]),
                "marked by the key": ("JediTaskSpec.cpuTimeUnit", "x", [reading("pandajedi/r.py::extract", 202), reading("pandaserver/m.py::initialize_cpu_time_task", 408)]),
                "marked by a job's": ("JediTaskSpec.cpuTime", "5", [reading("pandajedi/d.py::massage", 520)]),
            }
            got = {}

            def tries_of(name, subject, value, readings):
                try:
                    tries = r._attribute_tries(subject, subject.split(".")[1], value, readings)
                    got[name] = [[s["status"] if s else None, a.window if a else None, a.machine if a else None]
                                 + ([s["f3"], s["time"], [[x.owner.split("::")[-1], x.lines] for x in own], list(a.strings) if a else None]
                                    if s and s.get("f3") else [])
                                 for s, a, own in tries]
                except rerun.Stop as e:
                    got[name] = e.args[0]

            for name, (subject, value, readings) in asks.items():
                tries_of(name, subject, value, readings)
            got["F3 said"] = [x for x in r.lines if x.startswith("F3") or "408 is on no" in x]
            got["key marks"] = [list(rerun.key_marks(f"p.py::{f}", 1, cls, "53")) for f, cls in (
                ("changeTaskSplitRulePanda", "JediTaskSpec"), ("initialize_cpu_time_task", "JobSpec"), ("massage", "JediTaskSpec"),
                ("odd", "JediTaskSpec"), ("odd2", "JediTaskSpec"), ("none", "JediTaskSpec"))]
            # the row's creation not known: neither the life nor the birth's row can be placed
            r.data["rows"]["task"].pop("creationDate")
            tries_of("no creation, F3", *asks["marked by the key"])
            tries_of("no creation, a birth", *asks["a birth"])
            # each row with the writers that can write it, a reading cut to those lines where it has more
            Reading = __import__("bamboo.codemap.models", fromlist=["Reading"]).Reading
            two = [reading("pandajedi/r.py::extract", 182), Reading(owner="pandaserver/t.py::scout", lines=[1187, 1188])]
            tries = r._attribute_tries("JediTaskSpec.workDiskUnit", "workDiskUnit", "MB", two)
            got["two writers"] = [[s["status"], [[x.owner.split("::")[-1], x.lines] for x in own]] for s, _, own in tries]
            got["said"] = [x for x in r.lines if "row(s)" in x][-3:]
            # an arm whose pass writes history rows at two calls is one of a row's writers once
            AT = rerun.AT
            twice = AT.Arm("pandajedi/r.py::extract", 182, "runtime(u)", (N(writes=[upd]), N(writes=[upd])), "")
            got["once"] = [len(arms) for _, arms in AT.row_writers([twice], states, {upd: ["runtime(taskSpec.status)"]})]
            print(json.dumps(got))
            """,
        )
    )
    got = json.loads(out.splitlines()[-1])
    # every writer writes the time it runs: the value is the time, asked on every machine
    assert got["now"] == [[None, ["2026-10-29 00:49:09", "2026-10-29 00:49:09"], None]]
    assert got["not a time"] == "H2: a time for soon: rescueTime='soon', which its writers set to the time they run, is not a time"
    # a pass that records the task's status at run time can write any row: every row, newest first, at its host
    assert [g[0] for g in got["any row"]] == ["done", "ready", "defined"] and got["any row"][0][1:] == [["2026-10-29 08:36:33"] * 2, "h"]
    # one that records 'defined' only: that row
    assert got["its rows"] == [["defined", ["2026-10-29 07:08:40"] * 2, "h"]]
    # a writer on no row of the task, neither a birth nor marking its runs by the task's key, leaves the time not known
    k37 = "on a history row of the object, a birth or marking its runs by the object's key, to say when it took"
    assert got["one not placed"] == f"H2: every writer of JediTaskSpec.splitRule {k37} 'x' (1 of 2 not: KNOWN_ISSUES.md K37)"
    assert got["marked by a job's"] == f"H2: every writer of JediTaskSpec.cpuTime {k37} '5' (1 of 1 not: KNOWN_ISSUES.md K37)"
    assert got["no writer"] == "H2: a writer of JediTaskSpec.transUses in the map, to say when it took 'Atlas'"
    # a birth wrote as the row was made: after the rows, at the row's creation, asking nothing
    assert got["a birth"][:3] == got["any row"]
    assert got["a birth"][3] == [None, None, None, "made", "2026-10-29 07:08:39", [["getTasksToRefine_JEDI", [1188]]], None]
    # one marking its runs by the task's key is asked over the task's life, from its creation to when the value was
    # read, on every machine (F3), for its mark as its lines print it with the key, not the key alone: before the rows
    assert got["marked by the key"][0] == [None, ["2026-10-29 07:08:39", "2026-10-29 14:02:50"], None, "span",
                                           "2026-10-29 14:02:50", [["initialize_cpu_time_task", [408]]], ["; jediTaskID=53; siteID="]]
    assert got["marked by the key"][1:] == got["any row"]
    assert got["F3 said"] == [
        "  initialize_cpu_time_task:408 is on no history row of the object: the object is not followed to its passes' "
        "history call (not followed); its runs are marked by the object's key",
        "F3: initialize_cpu_time_task:408 asked over the object's life, from the row's creation at 2026-10-29 07:08:39 to "
        "when the value was read at 2026-10-29 14:02:50, on every machine, for '; jediTaskID=53; siteID='"]
    # the text around the key's hole of the mark as the rows print it: a job's key at the first hole takes the
    # function's name; no text for a mark not by the class's key, a prefix of other holes, or no prints
    assert got["key marks"] == [["changeTaskSplitRulePanda < jediTaskID=53 >"], ["initialize_cpu_time_task < PandaID=53; jediTaskID="], [], [], [], []]
    assert got["no creation, F3"] == ("H2: the object's life, to ask initialize_cpu_time_task:408 over: the row's creation time "
                                      "is not known")
    assert got["no creation, a birth"] == ("H2: when the row was made, for the birth getTasksToRefine_JEDI:1188 (its creation "
                                           "column, attr_time.creation_of)")
    # a row is asked of the writers that can write it alone (to-querier-61, step 1), a reading cut to its lines that can
    assert got["two writers"] == [["done", [["extract", [182]]]], ["ready", [["extract", [182]], ["scout", [1188]]]],
                                  ["defined", [["extract", [182]], ["scout", [1187]]]]]
    assert got["said"] == ["  extract:182: 3 row(s)", "  scout:1187: 1 row(s)", "  scout:1188: 1 row(s)"]
    assert got["once"] == [1, 1, 1]


def test_h2_stops_on_no_log_for_its_writers_and_not_on_no_run_in_logs_it_never_asked(tmp_path):
    out = _ok(
        _run(
            tmp_path,
            """
            import json
            from types import SimpleNamespace as N
            from bamboo.querier import rerun
            from bamboo.querier import rows as R

            R.ROWS_FROM = "walk"  # the reading's logs, as for a function the map has no prints of its own for
            r = object.__new__(rerun.Rerun)
            r.lines, r.opened, r.hop, r.hand, r.hand_used = [], [], "H2", {"H2": "2026-10-29 00:48:00"}, []
            reading = lambda owner, *logs: N(file=owner.partition("::")[0], owner=owner, lines=[939], log_files=list(logs))
            silent = reading("pandaserver/b.py::setSuperStatus_JEDI")
            got = {"one logged": r._writer_logs("JediTaskSpec.superStatus", [silent, reading("pandajedi/r.py::extract", "panda-TaskRefiner.log")])}
            for name, readings in (("none", [silent]), ("no writer", [])):
                try:
                    r._writer_logs("JediTaskSpec.superStatus", readings)
                except rerun.Stop as e:
                    got[name] = e.args[0]
            print(json.dumps(got))
            """,
        )
    )
    got = json.loads(out.splitlines()[-1])
    assert got["one logged"] == ["panda-TaskRefiner.log"]
    # no writer has a log: no log is asked, and a time handed for H2 gives none
    assert got["none"] == ("H2: a log to look for a run of a writer of JediTaskSpec.superStatus in (setSuperStatus_JEDI: the map "
                           "names none; KNOWN_ISSUES.md K25)")
    assert got["no writer"] == "H2: a writer of JediTaskSpec.superStatus in the map"


def test_h2_asks_and_reads_each_row_of_an_attribute_on_the_logs_of_the_writers_that_can_write_it(tmp_path):
    out = _ok(
        _run(
            tmp_path,
            """
            import json
            from types import SimpleNamespace as N
            from bamboo.querier import rerun

            reading = lambda owner, log: N(file=owner.partition("::")[0], owner=owner, lines=[1], log_files=[log])
            a, b = reading("p/a.py::refine", "a.log"), reading("p/b.py::scout", "b.log")
            row = lambda status: {"status": status, "time": "t"}

            def h2(tries):
                r = object.__new__(rerun.Rerun)
                r.lines, r.opened, r.hop, r.hand, r.hand_used, r.result, r.data = [], [], "", {}, [], {}, {}
                r._writers = lambda subject, value, key: [a, b]
                r._attribute_tries = lambda subject, attr, value, readings: tries
                r._h2_candidates = lambda subject, readings: [N(owner=x.owner, log_files=x.log_files) for x in readings]
                seen = {"asked": [], "read": None}
                rerun.owners_of = lambda ids: [i.split("::")[-1] for i in ids]
                r._ask = lambda key, names, asking, owners, whole: (None, [f"{asking.why} {n} {','.join(owners)} {whole}" for n in names], None)
                r._stop_unanswered = lambda asks: seen["asked"].extend(asks)
                r._scan = lambda *args: "scanned"
                def wrote_at(subject, attr, value, key, readings, state, asking, last):
                    seen["read"] = [x.owner for x in readings]
                    return N(chosen=N(candidate=N(owner="x"), line=0))
                r._wrote_at = wrote_at
                r._which_wrote("JediTaskSpec.workDiskUnit", "MB", "1")
                return seen

            print(json.dumps({
                "rows": h2([(row("running"), N(why="running"), [a]), (row("defined"), N(why="defined"), [a, b])]),
                "one row": h2([(row("defined"), N(why="defined"), [b])]),
            }))
            """,
        )
    )
    got = json.loads(out.splitlines()[-1])
    # every row's asks in one round, each of the logs of the writers that can write it, for their marks alone, each whole
    assert got["rows"]["asked"] == ["running a.log refine True", "defined a.log refine,scout True", "defined b.log refine,scout True"]
    # a single window is read on its own writers
    assert got["one row"]["read"] == ["p/b.py::scout"]


def test_a_writer_that_prints_nothing_is_read_at_its_history_pairs_call_and_its_own_condition_is_left_open(tmp_path):
    corpus = _corpus(tmp_path)
    (corpus / "pandajedi" / "r.py").write_text(
        textwrap.dedent(
            """
            class Refiner:
                def extract(self, task, taskParamMap):
                    if task.x:
                        self.taskSpec.transPath = taskParamMap["transPath"]
            """
        )
    )
    out = _ok(
        _run(
            tmp_path,
            """
            import json
            from types import SimpleNamespace as N
            from bamboo.querier import evaluate as E
            from bamboo.querier import mapdump, rerun
            from bamboo.querier import rows as R

            R.ROWS_FROM = "graph"
            rerun.prints_nothing = lambda owner, lines: owner.endswith("::extract")
            pair = N(row="can", row_why="", writes=[], caller="pandajedi/o.py::RefinerThread.runImpl", point=211, call=226,
                     level=1, callee="self.taskBufferIF.updateTask_JEDI", order="after")
            mapdump._FRAGMENT[:] = [N(junctions=[N(subject="JediTaskSpec.transPath", owner="pandajedi/r.py::Refiner.extract",
                                                   branches=[N(line=5, outcome="runtime(taskParamMap['transPath'])", history=[pair], history_hole="")])],
                                      reach_graph=[])]
            r = object.__new__(rerun.Rerun)
            r.lines, r.opened, r.hop, r.hand, r.hand_used, r._stand_ins = [], [], "H2", {}, [], {}
            r.files_of = lambda owner, today, lines: list(today) or [owner.split("::")[-1] + ".log"]
            silent = N(file="pandajedi/r.py", owner="pandajedi/r.py::extract", lines=[5], log_files=["panda-TaskRefiner.log"])
            printing = N(file="pandaserver/t.py", owner="pandaserver/t.py::setPath", lines=[9], log_files=[])
            cands = r._h2_candidates("JediTaskSpec.transPath", [silent, printing])
            row = {"status": "ready", "time": "2026-10-29 10:06:19"}
            v = lambda kind: N(candidate=cands[0], line=211, kind=kind, reasons=["rows 205 and 563 printed in a row"])
            said = lambda ps: [[p.arm, p.wrote, p.premises] for p in ps]
            print(json.dumps({
                "candidates": [[c.owner, c.lines, c.log_files] for c in cands],
                "shown, not read": said(r._possibles(v("positive"), row, E.Env())),
                "shown, read false": said(r._possibles(v("positive"), row, E.Env({"task.x": E.Value(False, "row")}))),
                "shown, read true": said(r._possibles(v("positive"), row, E.Env({"task.x": E.Value(True, "row")}))),
                "kept": said(r._possibles(v("kept"), row, E.Env({"task.x": E.Value(True, "row")}))),
            }))
            """,
        )
    )
    got = json.loads(out.splitlines()[-1])
    # the writer stands in at its pair's caller, on the logs its reading names; the printing writer stays itself
    assert got["candidates"] == [["pandajedi/o.py::runImpl", [211], ["panda-TaskRefiner.log"]],
                                 ["pandaserver/t.py::setPath", [9], ["setPath.log"]]]
    # the call shown run: the arm wrote only where its own condition is read true
    assert got["shown, not read"] == [["r.py::extract:5", False, ["its condition 'task.x' not read"]]]
    assert got["shown, read false"] == []
    assert got["shown, read true"] == [["r.py::extract:5", True, []]]
    assert got["kept"] == [["r.py::extract:5", False, ["its run does not show the line ran (rows 205 and 563 printed in a row)"]]]


def test_h2_reads_the_rows_newest_first_keeping_those_that_may_have_and_names_the_arms_left(tmp_path):
    out = _ok(
        _run(
            tmp_path,
            """
            import json
            from types import SimpleNamespace as N
            from bamboo.querier import rerun

            P = rerun.Possible
            rows = [{"status": s, "time": t} for s, t in (("running", "11:38"), ("scouting", "10:06"), ("ready", "10:05"), ("defined", "04:10"))]
            tries = [(s, N(why=f"the statuslog's {s['status']!r}"), None) for s in rows]
            answer = N(question="the run")

            seen = []

            def scan(found, tries=tries):
                r = object.__new__(rerun.Rerun)
                r.lines, r.opened, r.hop, r.hand, r.hand_used, r.result, r.question = [], [], "H2", {}, [], {}, None
                r._named = lambda subject, key, readings, tries: {"a.log"}
                r._row_at = lambda subject, attr, value, key, readings, state, asking, named: (seen.append([state["status"], readings, sorted(named)]), found(state))[1]
                try:
                    got = r._scan("JediTaskSpec.cloud", "cloud", "WORLD", "1", ["every writer"], tries)
                    if isinstance(got, rerun.OneOf):
                        raise r._one_of("JediTaskSpec.cloud", "WORLD", got)
                    return ["answer", got is answer, r.question]
                except rerun.Stop as e:
                    return [e.code, e.args[0], r.result.get("possibilities")]

            def table(by):
                return lambda state: by.get(state["status"], [])

            print(json.dumps({
                "the newest shows a printing writer": scan(table({"running": [P("a.py::gen:44", rows[0], True, [], answer)]})),
                "one that may have, then one shown": scan(table({
                    "running": [P("a.py::gen:44", rows[0], False, ["its condition 'x' not read"])],
                    "ready": [P("r.py::extract:226", rows[2], True, [])],
                    "defined": [P("o.py::old:1", rows[3], True, [], answer)],
                })),
                "none": scan(table({})),
                # a try naming the writers that can write its row is read on them alone
                "own": (seen.clear(), scan(table({}), [(rows[0], N(why="r"), ["its writer"]), (rows[1], N(why="s"), None)]), seen)[2],
            }))
            """,
        )
    )
    got = json.loads(out.splitlines()[-1])
    # and every row is read with the logs any row's records show naming the key
    assert got["own"] == [["running", ["its writer"], ["a.log"]], ["scouting", ["every writer"], ["a.log"]]]
    assert got["the newest shows a printing writer"] == ["answer", True, "the run"]
    # the newer that may have stays; the scan ends at the row shown; the older is not read
    code, text, possibilities = got["one that may have, then one shown"]
    assert code == "one-of"
    assert text == ("H2: one arm that wrote JediTaskSpec.cloud='WORLD' last, of 2 left: a.py::gen:44 (may have at 'running' 11:38); "
                    "r.py::extract:226 (wrote at 'ready' 10:05) (KNOWN_ISSUES.md K38)")
    assert [p["arm"] for p in possibilities] == ["a.py::gen:44", "r.py::extract:226"]
    assert possibilities[0]["rows"] == [{"status": "running", "time": "11:38", "wrote": False, "premises": ["its condition 'x' not read"]}]
    assert got["none"][:2] == ["one-of", "H2: a pass that wrote JediTaskSpec.cloud='WORLD' at the 4 statuslog row(s) its writers' passes "
                                         "can write (none shows a writer's pass: KNOWN_ISSUES.md K38)"]


def test_h2_reads_f3s_life_run_by_run_among_the_rows_and_a_births_row_last_as_one_that_may_have(tmp_path):
    out = _ok(
        _run(
            tmp_path,
            """
            import json
            from types import SimpleNamespace as N
            from bamboo.querier import rerun

            P, W = rerun.Possible, rerun.W
            init = [N(owner="pandaserver/m.py::initialize_cpu_time_task", file="pandaserver/m.py", lines=[408])]
            birth = [N(owner="pandaserver/s.py::getTasksToRefine_JEDI", file="pandaserver/s.py", lines=[1188])]
            MARK = ("; jediTaskID=53; siteID=",)
            life = W.Asking(("2026-10-29 04:10:00", "2026-10-29 14:02:50"), None, "the object's life", MARK)
            rows = [{"status": s, "time": t} for s, t in (("running", "2026-10-29 11:38:00"), ("defined", "2026-10-29 04:10:00"))]
            tries = ([({"status": None, "time": "2026-10-29 14:02:50", "f3": "span"}, life, init)]
                     + [(s, N(why=f"the statuslog's {s['status']!r}"), None) for s in rows]
                     + [({"status": None, "time": "2026-10-29 04:10:00", "f3": "made"}, None, birth)])
            log = "panda-JediDBProxy.log"
            run = lambda first, last, token="initialize_cpu_time_task < PandaID=1; jediTaskID=53; siteID=X >": N(first=first, last=last, token=token)
            runs = {log: [run("2026-10-29 10:00:00,123", "2026-10-29 10:00:05,000"), run("2026-10-29 12:00:00,000", "2026-10-29 12:00:01,000"),
                          run("2026-10-29 15:00:00,000", "2026-10-29 15:00:01,000"),  # after the value was read
                          # another call naming the task in its lines, under its own mark
                          run("2026-10-29 11:00:00,000", "2026-10-29 11:00:02,000", "getTasksToBeProcessed_JEDI < vo=atlas >")]}
            holes = {log: ["2026-10-29 05:00:00..2026-10-29 06:00:00"]}
            answer = N(question="the run")

            def rerun_with(runs, holes, at_rows=None):
                r = object.__new__(rerun.Rerun)
                r.lines, r.opened, r.hop, r.hand, r.hand_used, r.result, r.question = [], [], "H2", {}, [], {}, None
                r._h2_candidates = lambda subject, readings: [N(owner=init[0].owner, log_files=[log])]
                rerun.owners_of = lambda ids: {}
                r._ask = lambda key, names, asking, owners, whole: (runs, [], holes)
                r._named = lambda subject, key, readings, tries: {log}
                r.read = []
                r._row_at = lambda subject, attr, value, key, readings, state, asking, named: (
                    r.read.append(rerun._f3(state) or state["status"]), (at_rows or {}).get(state["status"], []))[1]
                return r

            def rows_of(runs, holes):
                return [[rerun._f3(s) or s["status"], s["time"]] + (list(a.window) + [list(a.strings) == list(MARK)] if rerun._f3(s) == "run" else [])
                        for s, a, _ in rerun_with(runs, holes)._runs_as_rows("JediTaskSpec.cpuTimeUnit", "53", tries)]

            def scan(runs, holes, at_rows=None):
                r = rerun_with(runs, holes, at_rows)
                try:
                    got = r._scan("JediTaskSpec.taskName", "taskName", "n", "53", ["every writer"], tries)
                    if isinstance(got, rerun.OneOf):
                        raise r._one_of("JediTaskSpec.taskName", "n", got)
                    return ["answer", got is answer, r.read]
                except rerun.Stop as e:
                    return [e.code, e.args[0], r.result.get("possibilities"), r.read]

            wrote = {"running": [P("r.py::extract:226", rows[0], True, [], answer)]}

            def walked(runs, holes, at_rows=None):
                left = rerun_with(runs, holes, at_rows)._scan("JediTaskSpec.taskName", "taskName", "n", "53", ["every writer"], tries).left
                return [[p.arm] + ([] if p.branch is None else [p.branch.question, p.branch.chosen.candidate.rel, p.branch.chosen.line,
                                                                [x.kind for x in p.premises]]) for p in left]

            print(json.dumps({
                "runs": rows_of(runs, {}), "no run": rows_of({}, {}), "not searched whole": rows_of({log: runs[log][1:2]}, holes),
                "nothing shown": scan({}, {}), "shown, life open": scan({}, holes, wrote), "shown, no run in the life": scan({}, {}, wrote),
                "walked": {"nothing shown": walked({}, {}), "shown, life open": walked({}, holes, wrote)},
            }))
            """,
        )
    )
    got = json.loads(out.splitlines()[-1])
    # each run of the key in the life is a row at its first record, newest first among the rows, the birth's last of
    # its second (each asked for the life's marks); a run after the value was read is no run of the life, and another's
    # run naming the key in its lines alone is none of the writers'
    assert got["runs"] == [["run", "2026-10-29 12:00:00", "2026-10-29 12:00:00", "2026-10-29 12:00:01", True], ["running", "2026-10-29 11:38:00"],
                           ["run", "2026-10-29 10:00:00", "2026-10-29 10:00:00", "2026-10-29 10:00:05", True], ["defined", "2026-10-29 04:10:00"],
                           ["made", "2026-10-29 04:10:00"]]
    # no run: the life stays one try, read as a row is (no run there is no pass only of a log shown to name the key)
    assert got["no run"] == [["span", "2026-10-29 14:02:50"], ["running", "2026-10-29 11:38:00"], ["defined", "2026-10-29 04:10:00"],
                             ["made", "2026-10-29 04:10:00"]]
    # a log not searched whole over the life: the life stays open, newer than any row, beside the runs it holds
    assert got["not searched whole"] == [["open", "2026-10-29 14:02:50"], ["run", "2026-10-29 12:00:00", "2026-10-29 12:00:00", "2026-10-29 12:00:01", True],
                                         ["running", "2026-10-29 11:38:00"], ["defined", "2026-10-29 04:10:00"], ["made", "2026-10-29 04:10:00"]]
    # nothing shown to write: the birth is left, as one that may have, read on no run
    code, text, possibilities, read = got["nothing shown"]
    assert code == "one-of" and read == ["span", "running", "defined"]
    assert text == ("H2: one arm that wrote JediTaskSpec.taskName='n' last, of 1 left: s.py::getTasksToRefine_JEDI:1188 (may have at "
                    "the row's making 2026-10-29 04:10:00) (KNOWN_ISSUES.md K38)")
    # result.json keeps the premise as a record and its kind (version 3)
    assert possibilities == [{"arm": "s.py::getTasksToRefine_JEDI:1188", "rows": [
        {"status": None, "time": "2026-10-29 04:10:00", "f3": "made", "wrote": False,
         "premises": [{"kind": "a-birth-no-run", "facts": {"time": "2026-10-29 04:10:00"}, "premise_kind": "not read"}]}]}]
    # a writer shown at a row, F3's writer open over the life: both are named
    code, text, possibilities, read = got["shown, life open"]
    assert code == "one-of" and read == ["running"]
    assert text == ("H2: one arm that wrote JediTaskSpec.taskName='n' last, of 2 left: m.py::initialize_cpu_time_task:408 (may have at "
                    "the object's life 2026-10-29 14:02:50); r.py::extract:226 (wrote at 'running' 2026-10-29 11:38:00) (KNOWN_ISSUES.md K38)")
    # F3's writer with no run in the life searched whole: the row's writer is the answer, the birth not read
    assert got["shown, no run in the life"] == ["answer", True, ["span", "running"]]
    # each arm left is walked on, if a fork does, on no run, its premise of a kind: a birth's not read, a life not
    # searched whole not recorded (K38)
    assert got["walked"] == {"nothing shown": [["s.py::getTasksToRefine_JEDI:1188", None, "pandaserver/s.py", 1188, ["not read"]]],
                             "shown, life open": [["m.py::initialize_cpu_time_task:408", None, "pandaserver/m.py", 408, ["not recorded"]],
                                                  ["r.py::extract:226"]]}


def test_h2_reads_no_run_of_the_key_at_a_row_as_no_pass_only_of_a_log_some_record_shows_naming_the_key(tmp_path):
    out = _ok(
        _run(
            tmp_path,
            """
            import json
            from types import SimpleNamespace as N
            from bamboo.querier import evaluate as E
            from bamboo.querier import rerun
            from bamboo.querier.which_ran import Candidate

            refine = Candidate("pandajedi/o.py", "runImpl", [211], ["panda-TaskRefiner.log"])
            gen = Candidate("pandajedi/g.py", "doGenerate", [1251], ["panda-JobGenerator.log"])
            r = object.__new__(rerun.Rerun)
            r.lines, r.opened, r.hop, r.hand, r.hand_used = [], [], "H2", {}, []
            r._stand_ins = {("pandajedi/o.py::runImpl", 211): [rerun.StandIn("pandajedi/r.py::extract:5", 1, ("task.x",)),
                                                               rerun.StandIn("pandajedi/r.py::extract:7", 1, ("task.y",))]}
            own = Candidate("pandaserver/s.py", "storeJobs", [410], ["panda-TaskBuffer.log"])
            r._h2_candidates = lambda subject, readings: [own] if readings == ["its writer"] else [refine, gen]
            r.window = lambda asking: asking.window
            # the refiner's log holds a run of the key on the older row's machine only; the generator's on none; a
            # try naming its own writers is read on theirs
            held = {("panda-TaskRefiner.log", "m91"): ["a run"], ("panda-TaskBuffer.log", "m86"): ["a run"]}

            def log(name, key, window, machine=None, owners=None, strings=(), whole=False):
                if name == "panda-DataService.log":
                    raise rerun.Unrecorded([])
                return N(runs=held.get((name, machine), []))

            r.logs = N(names=lambda wanted: wanted + ["panda-DataService.log"], log=log)
            rerun.owners_of = lambda ids: {}
            rerun.runs_for = lambda log, key: log.runs
            tries = [({"status": "pending"}, rerun.W.Asking(("10:05", "10:05"), "m86", "the row"), None),
                     ({"status": "defined"}, rerun.W.Asking(("04:10", "04:10"), "m91", "the row"), None)]
            named = r._named("JobSpec.nucleus", "7", ["every writer"], tries)
            named_own = r._named("JobSpec.nucleus", "7", ["every writer"], [(tries[0][0], tries[0][1], ["its writer"])])

            env = E.Env({"task.y": E.Value(False, "row")})
            row = {"status": "pending", "time": "10:05"}
            said = lambda ps: [[p.arm, p.wrote, p.premises] for p in ps]
            left = r._unshown(["every writer"], [refine, gen], "7", row, env, ("10:05", "10:05"))
            unshown = said(left)
            # each walked on, if a fork does, on no run: a stand-in's arm read at its caller's call
            walked = [[p.branch.question, p.branch.chosen.candidate.rel, p.branch.chosen.line, p.branch.via and p.branch.via.line,
                       p.branch.window, [x.kind for x in p.premises]] for p in left]

            def row_at(named, holes={}):
                r.lines = []
                r._unborn = lambda readings, subject, state: (readings, "")
                r._ask = lambda key, names, asking, owners, whole: ({}, [], holes)
                r.base_env = lambda at: env
                got = r._row_at("JobSpec.nucleus", "nucleus", "MPPMU", "7", ["every writer"], row, N(window=("10:05", "10:05"), machine=None), named)
                return [said(got), r.lines]

            # a log not searched whole goes to the judge, which here reads no verdict
            r.mode, r.decide, r._called_at, r.warn_split = "outside", "positive", None, lambda owner, lines: None
            rerun.stable_holes = lambda seens, env: None
            rerun.judge = lambda cands, runs, **kw: N(verdicts=[], runs={}, report=lambda question: "judged")
            print(json.dumps({"named": sorted(named), "named own": sorted(named_own), "unshown": unshown, "walked": walked,
                              "generator unshown": row_at(named), "both shown": row_at({"panda-TaskRefiner.log", "panda-JobGenerator.log"}),
                              "generator not searched whole": row_at(named, {"panda-JobGenerator.log": ["10:00..10:05"]}),
                              "refiner not searched whole": row_at(set(), {"panda-TaskRefiner.log": ["10:00..10:05"]})}))
            """,
        )
    )
    got = json.loads(out.splitlines()[-1])
    # a log is shown to name its runs by the key by a run in any row's records, on any machine; one unanswered is not
    assert got["named"] == ["panda-TaskRefiner.log"]
    assert got["named own"] == ["panda-TaskBuffer.log"]
    # a writer not shown so may have written at a row with no run of the key: a printing one at its line, a stand-in at
    # its arms less one whose condition is read false
    def why(log: str) -> str:
        return f"no line of {log} in the records holds 7: its runs are not shown to name it"

    assert got["unshown"] == [
        ["r.py::extract:5", False, [why("panda-TaskRefiner.log"), "its condition 'task.x' not read"]],
        ["g.py::doGenerate:1251", False, [why("panda-JobGenerator.log")]],
    ]
    # runs not shown to name the key, a condition not read: both in the data, not read
    assert got["walked"] == [[None, "pandajedi/r.py", 5, 211, ["10:05", "10:05"], ["not read", "not read"]],
                             [None, "pandajedi/g.py", 1251, None, ["10:05", "10:05"], ["not read"]]]
    # no run at the row: no pass of the refiner, whose log names the key; the generator's may have
    assert got["generator unshown"] == [
        [["g.py::doGenerate:1251", False, [why("panda-JobGenerator.log")]]],
        ["  no run of 7 at 10:05: no writer's pass of a log that names it",
         f"  may have written: g.py::doGenerate:1251 ({why('panda-JobGenerator.log')})"],
    ]
    # with every log shown naming the key, no run at the row is no writer's pass, said as before
    assert got["both shown"] == [[], ["  no run of 7 at 10:05: no writer's pass"]]
    # a writer whose log was not searched whole is the judge's (K38), not one more that may have; past the judge the
    # writers not shown naming the key are added
    assert got["generator not searched whole"][0] == []
    assert got["refiner not searched whole"][0] == [["g.py::doGenerate:1251", False, [why("panda-JobGenerator.log")]]]


def test_h2_reads_an_arm_whose_logs_never_name_the_key_on_the_run_that_recorded_the_row(tmp_path):
    out = _ok(
        _run(
            tmp_path,
            """
            import json
            from types import SimpleNamespace as N
            from bamboo.querier import render as T
            from bamboo.querier import rerun

            r = object.__new__(rerun.Rerun)
            r.lines, r.hop, r.unanswered, r.mode = [], "H2", [], "outside"
            r.data = {"classes": {"about": "x", "job": "JobSpec"}, "rows": {"job": {"jediTaskID": {"value": 5, "source": "now"}}}}
            r.window = lambda asking: asking.window
            files = {"pandaserver/j.py::insertNewJob": ["panda-DBProxy.log", "panda-JediDBProxy.log"],
                     "pandajedi/g.py::runImpl": ["panda-JobGenerator.log"]}
            r.files_of = lambda owner, today, lines: files[owner]
            r._called_at = lambda fn, callee: ["pandajedi/g.py:845"]
            r._key_columns = lambda subject, key: {"JobSpec.PandaID"}
            first = [True]
            rerun.prints_key_before = lambda owner, lines, columns: first[0] and columns == {"JobSpec.PandaID"} and owner == "pandaserver/j.py::insertNewJob"
            rec = rerun.Recorder("pandajedi/g.py::G.runImpl", 845, (("pandaserver/j.py::J.insertNewJob", (3631,)),), (1280, 840))
            # another call of the same function recording the row by the same method: one reason, said once
            rec2 = rerun.Recorder("pandajedi/g.py::G.runImpl", 850, (("pandaserver/j.py::J.insertNewJob", (3631,)),), (840,))
            recs = {"pandajedi/g.py::doGenerate": [rec], "pandajedi/j.py::setSplitRule": [rec, rec2]}
            rerun.recorders = lambda subject, owner, line: recs.get(owner, [])
            rerun.owners_of = lambda ids: {}
            rerun.runs_for = lambda log, key: log.runs
            rerun.R.function = lambda rel, name, arms=(): N(rel=rel, name=name, owner=f"{rel}::{name}")
            # the method's mark holds the task (``mark_columns``); the pass's mark is asked for it (``_pass_marks``)
            rerun.mark_columns = lambda owner, lines, run: {"JobSpec.jediTaskID": run.token.split("=")[1].split()[0]}
            rerun._pass_marks = lambda owner, line, values: [(v, (f"<t={v} ",)) for c, v in values.items() if c == "JobSpec.jediTaskID"]
            asked = []
            rerun.recorded_in_pass = lambda fn, found, window, inner, sites, mode: (
                asked.append([fn.owner, [f.token for f in found], [i.token for i in inner], sites]),
                (found[0], sites[0]) if found else None)[1]

            def run(token, first, last=None, machine="m85"):
                return N(token=token, first=first, last=last or first, machine=machine,
                         said=lambda: T.Said("log-at-machine", {"log": token, "machine": machine}))

            def read(held, host="m85"):
                # held: log name -> None (no answer), "absent" (no such log), [runs], ("holes", [runs]), or
                # {"unlisted": [runs]} (the host not listed: asked of every machine)
                r.lines, r.unanswered, asked[:] = [], [], []

                def log(name, key, window, machine=None, owners=None, strings=(), whole=False):
                    got = held.get((name, strings))
                    if got is None:
                        raise rerun.Unrecorded([(N(key=(name, strings, window)), "")])
                    if got == "absent":
                        return N(without=frozenset({machine}), unlisted=frozenset(), holes=[], runs=[])
                    if isinstance(got, dict):
                        return N(without=frozenset(), unlisted=frozenset({machine}), holes=[], runs=got["unlisted"])
                    holes, runs = got if isinstance(got, tuple) else ([], got)
                    return N(without=frozenset(), unlisted=frozenset(), holes=holes, runs=runs)

                r.logs = N(names=lambda wanted: wanted, log=log)
                why = rerun.Premise(T.Said("no-line-holds-the-key", {"logs": ["panda-JobGenerator.log"], "key": "7"}), rerun.NOT_READ)

                def possible(arm, rel, func, line, via=None):
                    return rerun.Possible(arm, {"status": "defined"}, False, [why],
                                          branch=N(chosen=N(candidate=N(rel=rel, func=func), line=line), via=via))

                def caller(rel, func, line):
                    return N(candidate=N(rel=rel, func=func, owner=f"{rel}::{func}"), line=line)

                left = [possible("g.py::doGenerate:1284", "pandajedi/g.py", "doGenerate", 1284),
                        possible("s.py::other:9", "pandajedi/s.py", "other", 9),
                        # a writer that prints nothing, read at the recording function's call on the way to it, and
                        # at another caller's
                        possible("j.py::setSplitRule:925", "pandajedi/j.py", "setSplitRule", 925, caller("pandajedi/g.py", "runImpl", 840)),
                        possible("j.py::setSplitRule:925", "pandajedi/j.py", "setSplitRule", 925, caller("pandajedi/w.py", "doForTaskBoost", 723))]
                state = {"status": "defined", "time": "2026-10-29 01:26:11"}
                asking = rerun.W.Asking(("2026-10-29 01:26:11", "2026-10-29 01:26:11"), host, "the row")
                got = r._recorded("JobSpec.diskIO", "7", state, asking, left)
                asks = []
                for _, a, _ in r.unanswered:
                    asks += [list(a.key)] if list(a.key) not in asks else []
                return {"left": [[p.arm + (f"@{p.branch.via.line}" if p.branch.via else ""), [x.kind for x in p.premises],
                                  [str(x) for x in p.premises]] for p in got],
                        "asks": asks, "lines": r.lines, "pass": list(asked)}

            M = lambda name: (name, ())
            G = ("panda-JobGenerator.log", ("<t=5 ",))
            mine = run("insertNewJob < <JediTaskID=5 idPool=1 >", "2026-10-29 01:26:09")
            far = run("insertNewJob < <JediTaskID=5 idPool=1 >", "2026-10-29 00:30:00")
            other = run("resetDefinedJob < PandaID=7 >", "2026-10-29 01:26:11")
            gen = run("<t=5 d=1>", "2026-10-29 01:20:00", "2026-10-29 01:27:00")
            # the method's run of the key near the row on another machine
            elsewhere = run("insertNewJob < <JediTaskID=5 idPool=1 >", "2026-10-29 01:26:09", machine="m84")
            def not_first(held):
                first[0] = False
                try:
                    return read(held)
                finally:
                    first[0] = True

            print(json.dumps({
                "unrecorded": read({}),
                "none": read({M("panda-DBProxy.log"): [other], M("panda-JediDBProxy.log"): "absent"}),
                "key not first": not_first({M("panda-DBProxy.log"): [other], M("panda-JediDBProxy.log"): "absent"}),
                "far": read({M("panda-DBProxy.log"): [far], M("panda-JediDBProxy.log"): "absent"}),
                "both absent": read({M("panda-DBProxy.log"): "absent", M("panda-JediDBProxy.log"): "absent"}),
                "not whole": read({M("panda-DBProxy.log"): (["01:20..01:26"], []), M("panda-JediDBProxy.log"): "absent"}),
                "unlisted": read({M("panda-DBProxy.log"): [other], M("panda-JediDBProxy.log"): {"unlisted": [elsewhere]}, G: [gen]}),
                "pass": read({M("panda-DBProxy.log"): "absent", M("panda-JediDBProxy.log"): [mine], G: [gen]}),
                "pass unrecorded": read({M("panda-DBProxy.log"): "absent", M("panda-JediDBProxy.log"): [mine]}),
                "no pass": read({M("panda-DBProxy.log"): "absent", M("panda-JediDBProxy.log"): [mine], G: []}),
            }))
            """,
        )
    )
    got = json.loads(out.splitlines()[-1])
    key = "no line of panda-JobGenerator.log in the records holds 7: its runs are not shown to name it"
    as_was = [[arm, ["not read"], [key]] for arm in ("g.py::doGenerate:1284", "s.py::other:9", "j.py::setSplitRule:925@840",
                                                     "j.py::setSplitRule:925@723")]
    # no record of the method's logs: as it was, and the method's logs and the pass's, for the value the rows file holds
    # now, asked in one round with no stop -- the methods' over the row less and plus EDGE, the pass's at the row
    near = ["2026-10-29 01:16:11", "2026-10-29 01:36:11"]
    row = ["2026-10-29 01:26:11", "2026-10-29 01:26:11"]
    assert got["unrecorded"]["left"] == as_was
    assert got["unrecorded"]["asks"] == [["panda-DBProxy.log", [], near], ["panda-JediDBProxy.log", [], near],
                                         ["panda-JobGenerator.log", ["<t=5 "], row]]
    # the method's logs on the row's machine, one there and read whole, hold no run of its of the key near the row (a
    # run of another method's, or one far from it, is none): the arm's pass did not write the row; an arm with no
    # recording call stays; a writer read at a caller's call is read on that caller's recording calls only, and is said
    # read there, its two calls' one reason once; read at a caller with no recording call, it stays
    none = "insertNewJob ran no run of 7 near the row on m85 (panda-DBProxy.log, read whole there): no pass through its call that records the row wrote it"
    assert got["none"]["left"] == [as_was[1], as_was[3]]
    assert got["none"]["lines"] == ["  not its row: g.py::doGenerate:1284 -- " + none,
                                    "  not its row: j.py::setSplitRule:925, read at g.py::runImpl:840 -- " + none]
    assert got["far"]["left"] == [as_was[1], as_was[3]]
    # a method the map does not say prints the key before it records the row: no run of it is not its absence
    assert got["key not first"]["left"] == as_was
    # none of its logs on the machine, or one not read whole there: as it was
    assert got["both absent"]["left"] == as_was
    assert got["not whole"]["left"] == as_was
    # a log the host is not listed for, asked of every machine: not read whole on the host, and another machine's run
    # of the key is not the host's -- as it was, and no pass is asked for it
    assert got["unlisted"]["left"] == as_was
    assert got["unlisted"]["pass"] == []
    # its run of the key near the row inside the call of a pass of the recording call's function, that pass asked for
    # the mark the method's run holds: said in place of the key not being named, still not read
    assert got["pass"]["pass"] == [["pandajedi/g.py::runImpl", ["<t=5 d=1>"], ["insertNewJob < <JediTaskID=5 idPool=1 >"],
                                    ["pandajedi/g.py:845"]]] * 3
    shown = ["its pass at the row is <t=5 d=1>@m85: insertNewJob's run of 7 ran inside its call at pandajedi/g.py:845; "
             "whether the line ran in it is not read"]
    assert got["pass"]["left"][0] == ["g.py::doGenerate:1284", ["not read"], shown]
    assert got["pass"]["left"][2] == ["j.py::setSplitRule:925@840", ["not read"], shown]
    # another caller's pass is not the pass of a writer read at doForTaskBoost's call
    assert got["pass"]["left"][1::2] == [as_was[1], as_was[3]]
    # the pass's log not recorded: as it was, asked; no pass holding the run: as it was
    assert got["pass unrecorded"]["left"] == as_was
    assert got["pass unrecorded"]["asks"] == [["panda-JobGenerator.log", ["<t=5 "], row]]
    assert got["no pass"]["left"] == as_was


def test_a_recorder_prints_the_key_first_by_a_hole_or_by_a_mark_every_run_holds(tmp_path):
    out = _ok(
        _run(
            tmp_path,
            """
            import json
            from types import SimpleNamespace as N
            from bamboo.querier import rerun

            def hole(*cols):
                return N(columns=list(cols))

            # line 10 writes the history; must prints before it are the statements at the lines given
            def method(before, statements, runs):
                rerun.R._prints_of = lambda owner, lines: N(statements=statements, runs=runs)
                rerun.R.function = lambda rel, name, arms=(): N(node="node")
                rerun.R.statement_at = lambda node, line: N(lineno=line)
                rerun.R.must_rows = lambda node, stmt: ([N(lineno=x) for x in before], [])
                return rerun.prints_key_before("p/j.py::insertNewJob", [10], {"JobSpec.PandaID"})

            st = lambda line, *holes, prefix=(): N(line=line, rows=[N(holes=list(holes), prefix_holes=list(prefix))])
            task_mark = [N(key=[N(columns=["JobSpec.jediTaskID"])])]
            key_mark = [N(key=[N(columns=["JobSpec.PandaID"])])]
            print(json.dumps({
                "hole before": method([3], [st(3, hole("JobSpec.PandaID"))], task_mark),
                "in its prefix": method([3], [st(3, prefix=[hole("JobSpec.PandaID")])], []),
                "only after": method([], [st(12, hole("JobSpec.PandaID"))], task_mark),
                "another column": method([3], [st(3, hole("JobSpec.jediTaskID"))], task_mark),
                "every mark holds it": method([3], [st(3)], key_mark),
                "a mark without it": method([3], [st(3)], key_mark + task_mark),
                "a mark, no print before": method([], [st(12)], key_mark),
            }))
            """,
        )
    )
    got = json.loads(out.splitlines()[-1])
    assert got == {"hole before": True, "in its prefix": True, "only after": False, "another column": False,
                   "every mark holds it": True, "a mark without it": False, "a mark, no print before": False}


def test_recorders_are_the_calls_after_the_arm_with_the_calls_on_the_way_to_it(tmp_path):
    out = _ok(
        _run(
            tmp_path,
            """
            import json
            from types import SimpleNamespace as N
            from bamboo.querier import rerun

            def pair(caller, point, call, writes, order="after"):
                return N(caller=caller, point=point, call=call, writes=writes, order=order)

            placed = [pair("w.py::W.boost", 723, 724, ["j.py::J.updateTask:1570"]),
                      # the same recording call reached by another call on the way to the arm: one recorder, both calls
                      pair("w.py::W.boost", 730, 724, ["j.py::J.updateTask:1570"]),
                      pair("w.py::W.boost", 723, 724, ["j.py::J.updateTask:1570"]),
                      pair("p.py::P.reassign", 312, 313, ["j.py::J.updateTask:1570", "j.py::J.updateTask:1580"]),
                      # before the arm, or writing no history line: not a recorder
                      pair("q.py::Q.before", 5, 4, ["j.py::J.updateTask:1570"], order="before"),
                      pair("q.py::Q.none", 5, 6, [])]
            rerun.mapdump.fragment = lambda: None
            rerun.AT.arms = lambda fragment, subject, at: [N(placed=placed)]
            print(json.dumps([[r.caller, r.call, [[m, list(ls)] for m, ls in r.methods], list(r.points)]
                              for r in rerun.recorders("JediTaskSpec.splitRule", "j.py::setSplitRule", 925)]))
            """,
        )
    )
    assert json.loads(out.splitlines()[-1]) == [
        ["w.py::W.boost", 724, [["j.py::J.updateTask", [1570]]], [723, 730]],
        ["p.py::P.reassign", 313, [["j.py::J.updateTask", [1570, 1580]]], [312]],
    ]


def test_recorded_in_pass_reads_a_pass_at_the_row_holding_every_run_of_the_recorder_inside_its_call(tmp_path):
    out = _ok(
        _run(
            tmp_path,
            """
            import json
            from types import SimpleNamespace as N
            from bamboo.querier import which_ran as W

            fn = N(owner="pandajedi/g.py::runImpl")
            W.R.entries = lambda fn, run, mode: run.entries
            W.R.seen = lambda fn, part, mode: part
            W.at_window = lambda part, window: part.first <= window[0] <= part.last
            # inside the call where the pass's span holds the inner run, on its machine
            W.inside_call = lambda seen, inner, site: (0, 1) if site.endswith(":845") and seen.first <= inner.first and inner.last <= seen.last else None

            def run(token, first, last, machine="m85", entries=()):
                return N(token=token, first=first, last=last, machine=machine, entries=list(entries), records=[0, 1],
                         part=lambda a, b: run(token, first, last, machine))

            gen = run("<t=5 d=1>", "01:20", "01:27")
            ins = run("insertNewJob < <JediTaskID=5 >", "01:26", "01:26")
            other_machine = run("insertNewJob < <JediTaskID=5 >", "01:26", "01:26", machine="m86")
            late = run("insertNewJob < <JediTaskID=5 >", "01:28", "01:28")
            say = lambda got: None if got is None else [got[0].token, got[1]]
            print(json.dumps({
                "inside": say(W.recorded_in_pass(fn, [gen], ("01:26", "01:26"), [ins], ["pandajedi/g.py:845"], "outside")),
                "no inner": say(W.recorded_in_pass(fn, [gen], ("01:26", "01:26"), [], ["pandajedi/g.py:845"], "outside")),
                "another site": say(W.recorded_in_pass(fn, [gen], ("01:26", "01:26"), [ins], ["pandajedi/g.py:900"], "outside")),
                "another machine": say(W.recorded_in_pass(fn, [gen], ("01:26", "01:26"), [ins, other_machine], ["pandajedi/g.py:845"], "outside")),
                "one outside": say(W.recorded_in_pass(fn, [gen], ("01:26", "01:26"), [ins, late], ["pandajedi/g.py:845"], "outside")),
                "not at the row": say(W.recorded_in_pass(fn, [gen], ("01:30", "01:30"), [ins], ["pandajedi/g.py:845"], "outside")),
                "another method's mark": say(W.recorded_in_pass(fn, [run("getJobs < x >", "01:20", "01:27")], ("01:26", "01:26"), [ins],
                                                                ["pandajedi/g.py:845"], "outside")),
            }))
            """,
        )
    )
    got = json.loads(out.splitlines()[-1])
    assert got["inside"] == ["<t=5 d=1>", "pandajedi/g.py:845"]
    # every run of the recorder near the row must be inside one call of the pass, on its machine
    assert got["no inner"] is None
    assert got["another site"] is None
    assert got["another machine"] is None
    assert got["one outside"] is None
    assert got["not at the row"] is None
    assert got["another method's mark"] is None


def test_h2_goes_on_from_a_stand_ins_arm_shown_to_write_read_on_its_callers_run(tmp_path):
    out = _ok(
        _run(
            tmp_path,
            """
            import dataclasses
            import json
            from types import SimpleNamespace as N
            from bamboo.querier import evaluate as E
            from bamboo.querier import rerun
            from bamboo.querier.which_ran import Candidate, Verdict

            caller = Candidate("pandajedi/o.py", "runImpl", [211], ["panda-TaskRefiner.log"])
            r = object.__new__(rerun.Rerun)
            r.lines, r.opened, r.hop, r.hand, r.hand_used, r.result, r.question, r.mode = [], [], "H2", {}, [], {}, None, "outside"
            r._stand_ins = {("pandajedi/o.py::runImpl", 211): [rerun.StandIn("pandajedi/r.py::extract:5", 1, ()),
                                                               rerun.StandIn("pandajedi/r.py::extract:7", 1, ("task.y",))]}
            env = E.Env({})
            v = Verdict(caller, 211, "positive", ["P: shown"])
            answer = rerun.Wrote(["readings"], N(records=list(range(20))), ["cands"], env, v)
            row = {"status": "defined", "time": "04:10"}
            got = r._possibles(v, row, env, answer)
            stood = got[0].answer
            possibles = [[p.arm, p.wrote, p.premises, p.by, p.answer is not None] for p in got]
            chosen = [stood.chosen.candidate.rel, stood.chosen.candidate.func, stood.chosen.candidate.lines, stood.chosen.line,
                      stood.via is v, stood.question is answer.question, stood.readings]
            kept = [[p.arm, p.wrote, p.answer is None, [x.kind for x in p.premises], p.branch.via is not None]
                    for p in r._possibles(Verdict(caller, 211, "kept", ["must rows"]), row, env, answer)]
            unsearched = dataclasses.replace(answer, unsearched={"panda-TaskRefiner.log": ["m91: 04:09 .. 04:10"]})
            kept += [[p.arm, [x.kind for x in p.premises]] for p in r._possibles(Verdict(caller, 211, "kept", ["must rows"]), row, env, unsearched)]

            P = rerun.Possible
            newer = {"status": "running", "time": "11:38"}

            def answer_of(left):
                r.lines = []
                got = r._answer(left)
                return [None if got is None else got is stood, r.lines]

            shown = P("r.py::extract:5", row, True, [], stood, "o.py::runImpl:211")
            other = P("r.py::extract:5", row, False, ["its run does not show the line ran"], None, "o.py::finish:1680")

            # a single window's pick (a status's row, a value held): the same reading of a stand-in
            r._stand_ins[("pandajedi/o.py::runImpl", 300)] = [rerun.StandIn("pandajedi/r.py::extract:5", 1, ())]

            def single(kind, line):
                w = rerun.Wrote(["readings"], answer.question, ["cands"], env, Verdict(caller, line, kind, ["why"]))
                try:
                    got = r._stood("JediTaskSpec.transUses", "A", w, row)
                    if isinstance(got, rerun.OneOf):
                        # each arm left carries what a fork walks on: the writer's arm, read at the caller's call
                        return ["one-of", [[p.branch.chosen.line, p.branch.via is w.chosen] for p in got.left]]
                    return ["answer", got is w, got.chosen.line, None if got.via is None else got.via is w.chosen]
                except rerun.Stop as e:
                    return [e.code]

            singles = {"not a stand-in": single("positive", 212), "shown": single("positive", 300), "not shown": single("kept", 300)}

            # ... as H2 at a status's row reads it
            def cands(subject, readings):
                r._stand_ins[("pandajedi/o.py::runImpl", 300)] = [rerun.StandIn("pandajedi/r.py::extract:5", 1, ())]
                return [caller]

            r._writers, r._h2_candidates = (lambda subject, value, key: ["readings"]), cands
            r._writer_logs = lambda subject, readings, cands=None: ["panda-TaskRefiner.log"]
            r._wrote_at = lambda *a: rerun.Wrote(["readings"], answer.question, ["cands"], env, Verdict(caller, 300, "positive", []))
            w = r._which_wrote("JobSpec.jobStatus", "running", "7", {"status": "running", "time": "2026-10-29 04:10:00"})
            singles["at a status's row"] = [w.chosen.line, w.via.line]

            # where H3 reads the run up to: the chosen line's function, or the caller at its call for a stand-in
            rerun.R = N(function=lambda rel, func, lines: N(name=func, node=func), statement_at=lambda node, line: (node, line),
                        must_rows=lambda node, site: (None, [N(lineno=site[1] + 1)]),
                        seen=lambda fn, run, mode: N(hits={212: [(3, None, None)], 6: [(9, None, None)]}))
            at = r._guard_at(stood, N(name="extract", node="extract"))
            own = r._guard_at(rerun.Wrote([], answer.question, [], env, Verdict(Candidate("pandajedi/r.py", "extract", [5]), 5, "positive")),
                              N(name="extract", node="extract"))
            print(json.dumps({
                "possibles": possibles, "chosen": chosen, "kept": kept,
                "alone": answer_of([shown]),
                "another call, the same row": answer_of([other, shown]),
                "a newer row may have": answer_of([P("r.py::extract:5", newer, False, ["x"], None, "o.py::runImpl:211"), shown]),
                "two arms": answer_of([P("r.py::extract:7", row, False, ["x"]), shown]),
                "no run to read": answer_of([P("r.py::extract:5", row, True, [], None)]),
                "guard at the call": at.name, "guard at the arm": own.name, "single": singles,
            }))
            """,
        )
    )
    got = json.loads(out.splitlines()[-1])
    # an arm of a writer that prints nothing, its call shown and no condition open, wrote: it carries the answer, the
    # writer's arm chosen on the caller's run; one whose condition is not read may have, and carries none
    assert got["possibles"] == [
        ["r.py::extract:5", True, [], "o.py::runImpl:211", True],
        ["r.py::extract:7", False, ["its condition 'task.y' not read"], "o.py::runImpl:211", False],
    ]
    assert got["chosen"] == ["pandajedi/r.py", "extract", [5], 5, True, True, ["readings"]]
    # the call not shown: each arm may have written
    # ... each walked on, if a fork does, at the call: its run not showing the line ran is not in the data, or, its
    # logs not searched whole, not recorded; its condition not read
    assert got["kept"] == [["r.py::extract:5", False, True, ["not in the data"], True],
                           ["r.py::extract:7", False, True, ["not in the data", "not read"], True],
                           ["r.py::extract:5", ["not recorded"]], ["r.py::extract:7", ["not recorded", "not read"]]]
    # one arm shown to write, nothing newer that may have: the answer; another call of it in the same pass is said
    assert got["alone"] == [True, []]
    assert got["another call, the same row"] == [True, [
        "  r.py::extract:5 wrote, read at o.py::runImpl:211; in the same pass also o.py::finish:1680, the same arm: "
        "which call was last is not read"]]
    assert got["a newer row may have"] == [None, []]
    assert got["two arms"] == [None, []]
    assert got["no run to read"] == [None, []]
    # H3 reads the stand-in's run up to the record past its call (211 -> 212), not past the arm's own line (5 -> 6)
    assert got["guard at the call"] == "runImpl"
    assert got["guard at the arm"] == "extract"
    # a single window's pick: kept as it is unless it stands in; a stand-in's arm shown to write answers on the call's
    # run, one not shown stops naming it
    assert got["single"] == {"not a stand-in": ["answer", True, 212, None], "shown": ["answer", False, 5, True], "not shown": ["one-of", [[5, True]]],
                             "at a status's row": [5, 300]}


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


_CLOSURE = """
def main():
    LIMIT = get_config() or 3
    N = 3

    class Thr:
        def run(self, x):
            if x < LIMIT and x < N:
                return x

    def keeps(x):
        N = 5
        return x < N


def other(x):
    return x < LIMIT
"""


def test_a_name_a_nested_function_reads_free_is_bound_by_the_def_around_it(tmp_path):
    # datasetManager's FinisherThr.run reads main's TRANSFER_TIMEOUT_HI_PRIORITY (84): the map's trace answers it with
    # main's step, which only run's own steps were taken from before
    (_corpus(tmp_path) / "pandaserver" / "a.py").write_text(_CLOSURE)
    out = _ok(
        _run(
            tmp_path,
            """
            import json
            from types import SimpleNamespace as N
            from bamboo.querier import asked as A
            from bamboo.querier import evaluate as E

            f = "pandaserver/a.py::"
            step = lambda name, owner, line, value, terminal="", cls="": N(
                name=name, owner=f + owner, kind="binding", line=line, value=value, guards=[], unseen=[], reads=[],
                terminal=terminal, detail="", depth=2, cls=cls)
            steps = A.given([step("x", "run", 7, "1", cls=f + "Thr"),
                             step("LIMIT", "main", 3, "get_config() or 3", terminal="a configuration value, set outside the code"),
                             step("N", "main", 4, "3"), step("N", "keeps", 11, "5")])
            got = {}
            for name, owner in (("LIMIT", "run"), ("N", "run"), ("N", "keeps"), ("LIMIT", "other")):
                t = E.taken(steps, name, E.Env(), owner=f + owner)
                got[f"{name} {owner}"] = [t.step.line if t.step else None, repr(t.value), [s.line for s in t.kept]]
            print(json.dumps(got))
            """,
        )
    )
    got = json.loads(out.splitlines()[-1])
    # main's binding: a terminal is unknown, a literal its value
    assert got["LIMIT run"] == [3, "UNKNOWN", [3]] and got["N run"] == [4, "3", [4]]
    # a def that binds the name itself reads its own; a def that main does not hold reads none of main's
    assert got["N keeps"] == [11, "5", [11]] and got["LIMIT other"] == [None, "UNKNOWN", []]


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


def test_a_step_whose_dump_holds_models_is_kept_once_at_any_depth(tmp_path):
    out = _ok(
        _run(
            tmp_path,
            """
            import json
            from pydantic import BaseModel
            from bamboo.codemap import trace
            from bamboo.codemap.models import TraceStep
            from bamboo.querier import asked as A

            class Call(BaseModel):
                call: str
                made: str

            # a field holding models, dumped as a list of dicts, as the map's TraceStep.calls
            class Step(TraceStep):
                calls: list[Call] = []

            f = "pandaserver/a.py::f"
            x3 = trace.Question("bindings", "pandaserver/a.py", f, "x", 3)
            step = lambda depth: Step(kind="binding", name="x", owner=f, line=2, value="g()", depth=depth,
                                      calls=[Call(call="g()", made="corpus")])

            class Asker:
                def ask(self, question):
                    return trace.Answer([step(3)], "", [])

            steps = A.Asked([step(2)], Asker(), [x3], "", "f:1")
            steps.ask(lambda q: True)
            print(json.dumps({"steps": [(s.name, s.line, s.depth) for s in steps], "asked": [(a.name, a.line) for a in steps.asked]}))
            """,
        )
    )
    got = json.loads(out.splitlines()[-1])
    # x at 3 hands back the step the walk started with, a hop deeper: kept once
    assert got == {"steps": [["x", 2, 2]], "asked": [["x", 3]]}


def test_a_rows_prefix_is_read_without_its_leading_blanks_which_the_records_cut_takes_with_the_level(tmp_path):
    out = _ok(
        _run(
            tmp_path,
            """
            import json
            from types import SimpleNamespace as N
            from bamboo.querier import rows

            def by(prefixes, token, text):
                row = object.__new__(rows.Row)
                row.map_prefixes = prefixes
                return row.written_by(N(token=token, text=text))

            # MsgWrapper(self.logger, f" < jediTaskID={jediTaskID} >"): "INFO      < jediTaskID=53099750 > start for retry"
            mark = ["\\\\ <\\\\ jediTaskID=.*?\\\\ >\\\\ "]
            print(json.dumps({
                "the mark's line": by(mark, "< jediTaskID=53099750 >", "start for retry"),
                "another mark": by(mark, "<run>", "start"),
                "inside an outer mark": by(mark, "<run>", " < jediTaskID=5 > start"),
                "a proxy's tag": by(["archiveJob\\\\ <\\\\ .*?\\\\ >\\\\ "], "archiveJob < PandaID=1 >", "start"),
                "another method's tag": by(["archiveJob\\\\ <\\\\ .*?\\\\ >\\\\ "], "getJobs < x >", "start"),
            }))
            """,
        )
    )
    got = json.loads(out.splitlines()[-1])
    # the record's token holds no leading blank (records.PREFIX reads it with the level's padding): the mark's line is
    # the row's, whose prefix keeps the blank as the logger writes it
    assert got == {"the mark's line": True, "another mark": False, "inside an outer mark": True, "a proxy's tag": True,
                   "another method's tag": False}


def test_a_row_shorter_than_min_proof_proves_only_as_its_whole_message_no_other_function_can_print_there(tmp_path):
    (_corpus(tmp_path) / "pandaserver" / "m.py").write_text(
        textwrap.dedent(
            """
            class M:
                def activateJob(self):
                    tmp_log.debug("start")

                def configurator_read_panda_sites(self):
                    tmp_log.debug("done")
            """
        )
    )
    out = _ok(
        _run(
            tmp_path,
            """
            import json
            from types import SimpleNamespace as N
            from bamboo.querier import mapdump, printed, rows

            own = "activateJob\\\\ <\\\\ PandaID=.*?\\\\ >\\\\ "
            # configurator_read_panda_sites's comment names configurator_read_sites: both write its mark
            shared = "configurator_read_sites\\\\ <\\\\ .*?\\\\ >\\\\ "
            prints = lambda owner, prefix, pattern, logs=("j.log",), basis="logger": N(
                owner=owner, statements=[N(rows=[N(prefix=prefix, pattern=pattern)], log_files=list(logs), log_files_basis=basis)])
            base = [prints("pandaserver/m.py::M.activateJob", own, "start"),
                    prints("pandaserver/m.py::M.configurator_read_panda_sites", shared, "done"),
                    prints("pandaserver/n.py::N.configurator_read_sites", shared, "done"),
                    # a logger whose mark the map cannot spell (the adder's), in another log
                    prints("pandaserver/a.py::A.run", ".*?\\\\ ", "start", logs=("other.log",))]

            def proves(name, prefix, text, token, message, cut=False, also=()):
                fragment = N(function_prints=base + list(also))
                mapdump.fragment = lambda: fragment
                row = object.__new__(rows.Row)
                row.map_prefixes, row.map_patterns, row.compiled = [prefix] if prefix else [], [text], {}
                row.stmt = N(lineno=3)
                fn = N(rel="pandaserver/m.py", node=rows.functions("pandaserver/m.py", name)[0])
                run = N(records=[N(token=token, text=message, cut=cut)], log=N(name="j.log"))
                sn = rows.Seen(fn, run, "map", {3: [(0, len(text), "map")]}, {3: [0]})
                return [sn.proves(row, 0, len(text)), sn.proven(row), printed._proven_hits(sn, row)]

            mark = "activateJob < PandaID=1 >"
            start = lambda **k: proves("activateJob", own, "start", mark, "start", **k)
            # calculateRWwithPrio_JEDI makes its logger with two tags: the map gives its rows no prefix
            marks_on_paths = lambda **k: prints("pandaserver/c.py::C.calculateRWwithPrio_JEDI", "", "start", **k)
            got = {
                "its whole message, no other function's line there": start(),
                "inside another word": proves("activateJob", own, "start", mark, "restart"),
                "a record the search cut": start(cut=True),
                "another function's mark": proves("activateJob", own, "start", "getJobs < PandaID=1 >", "start"),
                "a mark two functions write": proves("configurator_read_panda_sites", shared, "done",
                                                     "configurator_read_sites < 2026-10-31/09:00:00 >", "done"),
                "a logger with marks on its paths, in the log": start(also=[marks_on_paths()]),
                "a logger with marks on its paths, in another log": start(also=[marks_on_paths(logs=("other.log",))]),
                "lines the map does not place": start(also=[marks_on_paths(logs=(), basis="module")]),
                "a row of holes alone": start(also=[prints("pandaserver/c.py::C.dump", "", "")]),
                "another text with no prefix, in the log": start(also=[prints("pandaserver/c.py::C.dump", "", "done")]),
                "a row with no prefix": proves("activateJob", "", "start", mark, "start"),
                "a row of MIN_PROOF": proves("activateJob", "", "a long row", mark, "in a long row too"),
            }
            print(json.dumps(got))
            """,
        )
    )
    got = json.loads(out.splitlines()[-1])
    yes, no = [True, True, 1], [False, False, 0]
    assert got == {"its whole message, no other function's line there": yes, "inside another word": no,
                   "a record the search cut": no, "another function's mark": no, "a mark two functions write": no,
                   "a logger with marks on its paths, in the log": no, "a logger with marks on its paths, in another log": yes,
                   "lines the map does not place": no, "a row of holes alone": yes,
                   "another text with no prefix, in the log": yes, "a row with no prefix": no,
                   "a row of MIN_PROOF": yes}


def test_s2_takes_the_one_positive_arm_as_a_passs_way_and_the_positive_exits_as_where_it_left_after(tmp_path):
    out = _ok(
        _run(
            tmp_path,
            """
            import ast, json
            from types import SimpleNamespace as N
            from bamboo.querier import rerun, which_ran

            # arms at 3 and 7, exits at 4 and 5 (updateJobStatus: the write at 242, then the return at 496)
            node = ast.parse("def f():\\n    if a:\\n        x = 1\\n        return 1\\n    return 2\\n    if b:\\n        x = 2\\n").body[0]
            fn = N(node=node, owner="p/j.py::f", name="f")
            run = N(describe=lambda: "the run", token="f < PandaID=1 >")
            one = N(first="2026-10-29 08:23:18,275", last="2026-10-29 08:23:18,288", records=[0, 1], log=N(name="j.log"),
                    found_by=[0])
            rerun.R.function = lambda rel, func, lines: fn
            rerun.R.passes = lambda fn, run, mode: ([one], "1 pass")
            rerun.R.seen = lambda fn, run, mode: None
            rerun.stable_holes = lambda *a, **k: None
            rerun.class_of = lambda fn: None
            verdict = lambda line, kind: which_ran.Verdict(which_ran.Candidate("p/j.py", "f", [line]), line, kind)

            def way(kinds):
                r = object.__new__(rerun.Rerun)
                r.lines, r.hop, r.hand, r.hand_used = [], "", {}, []
                r.opened, r.ends, r.branches = [], [], []
                r.mode, r.decide, r.data, r.args = "map", "positive", {"statuslog": {"states": []}}, N(observed="running")
                r.logs = N(needs_window=False, log=lambda name, key: N(covers=("t0", "t1")))
                r.runs = lambda key, logs, asking, owners: {"j.log": [run]}
                r.base_env = lambda at=None: N(values={})
                rerun.judge = lambda *a, **k: which_ran.Judgement([verdict(l, k) for l, k in kinds.items()], None, "undecided", None)
                arms = [rerun.Arm(3, ["x"], []), rerun.Arm(7, ["y"], [])]
                try:
                    got = r.passes(rerun.Passes("p/j.py::f", ["j.log"], arms, "1", None)).exit_line
                except rerun.Stop as e:
                    got = str(e)
                return [got, [l.strip() for l in r.lines if "the way pass" in l]]

            print(json.dumps({
                "an arm and an exit": way({3: "positive", 4: "positive", 5: "kept", 7: "excluded"}),
                "an arm and two exits": way({3: "positive", 4: "positive", 5: "positive", 7: "kept"}),
                "two arms": way({3: "positive", 4: "positive", 7: "positive"}),
                "exits alone": way({3: "kept", 4: "positive", 5: "positive"}),
                "the arm alone, not chosen": way({3: "positive", 4: "kept"}),
            }))
            """,
        )
    )
    got = json.loads(out.splitlines()[-1])
    stop = ["S2: which way pass 1 went", []]
    assert got == {
        "an arm and an exit": [3, ["the way pass 1 went is arm 3: the positive exit(s) [4] are where it left after"]],
        "an arm and two exits": [3, ["the way pass 1 went is arm 3: the positive exit(s) [4, 5] are where it left after"]],
        "two arms": stop,
        "exits alone": stop,
        "the arm alone, not chosen": stop,
    }



def test_judgement_4_asks_a_late_movers_marks_over_its_stay_and_reads_its_passes_before_the_move(tmp_path):
    out = _ok(
        _run(
            tmp_path,
            """
            import ast, datetime, json
            from types import SimpleNamespace as N
            from bamboo.querier import rerun, which_ran

            # the mover: an arm at 3 (the move), an exit at 4
            node = ast.parse("def f():\\n    if a:\\n        x = 1\\n    return 2\\n").body[0]
            fn = N(node=node, owner="p/j.py::f", name="f")
            mark = "f < PandaID=1 >"

            def pas(first, last, keyed=True):
                return N(first=first, last=last, records=[0], log=N(name="j.log"), found_by=[0] if keyed else [],
                         token=mark if keyed else "f < vo=atlas >", machine="aipanda081")

            rerun.R.function = lambda rel, func, lines: fn
            rerun.R.seen = lambda fn, run, mode: None
            rerun.stable_holes = lambda *a, **k: None
            rerun.class_of = lambda fn: None
            rerun.arms_of = lambda writers, owner, subject, observed: {3: rerun.Arm(3, ["activated"], [])}
            rerun.key_marks = lambda owner, line, cls, key: (mark,) if marked else ()
            left = which_ran.Verdict(which_ran.Candidate("p/j.py", "f", [4]), 4, "positive")
            rerun.judge = lambda cands, runs, **k: which_ran.Judgement([left], left, "positive evidence", None)

            def rerun_(passes, missed=(), holes=None, hand=("j.log",)):
                r = object.__new__(rerun.Rerun)
                r.lines, r.hop, r.hand, r.hand_used, r.unanswered = [], "L2", {}, [], []
                r.opened, r.ends, r.branches = [], [], []
                r.mode, r.decide = "map", "positive"
                r.data, r.args = {"statuslog": {"states": []}}, N(observed="assigned", key="1", asks="asks.jsonl")
                r.logs = N(needs_window=True, names=lambda files: [f for f in files if f in hand])
                r.files_of = lambda owner, today, lines: today
                run = N(token=mark, first="2026-10-29 09:00:00,000", last="2026-10-29 13:33:47,000", describe=lambda: "the run", said=lambda: "the run",
                        machine="aipanda081", found_by=[0])
                r.asked = []
                r._ask = lambda key, names, asking, owners=None: (r.asked.append(asking) or ({"j.log": [run]}, list(missed), holes or {}))
                r.runs = lambda key, logs, asking, owners: {"j.log": [run]}
                rerun.R.passes = lambda fn, run, mode: (passes, f"{len(passes)} pass(es)")
                r.base_env = lambda at=None: N(values={})
                return r

            q = rerun.Missed("JobSpec.jobStatus", "assigned", "activated", datetime.datetime(2026, 10, 29, 9, 6, 9),
                             datetime.datetime(2026, 10, 29, 13, 33, 46), [], "p/j.py::f", None,
                             writers=[N(owner="p/j.py::f", log_files=["j.log"])])
            got = {}
            for name, marked, passes, missed, holes in (
                ("passes before the move", True, [pas("2026-10-29 10:00:00,000", "2026-10-29 10:00:01,000"),
                                                  pas("2026-10-29 13:33:46,500", "2026-10-29 13:33:47,000")], (), None),
                ("the move alone", True, [pas("2026-10-29 13:33:45,900", "2026-10-29 13:33:46,100")], (), None),
                ("a pass without the key", True, [pas("2026-10-29 10:00:00,000", "2026-10-29 10:00:01,000", keyed=False),
                                                  pas("2026-10-29 12:00:00,000", "2026-10-29 12:00:01,000"),
                                                  pas("2026-10-29 13:33:46,500", "2026-10-29 13:33:47,000")], (), None),
                ("no pass holds the key", True, [pas("2026-10-29 10:00:00,000", "2026-10-29 10:00:01,000", keyed=False)], (), None),
                ("no mark holds the key", False, [], (), None),
                ("a hole in the search", True, [], (), {"j.log": ["aipanda081"]}),
                ("an ask not recorded", True, [], [(N(key="a"), "why")], None),
                ("a log not to hand", True, [], (), None),
                ("no arm", True, [], (), None),
                ("the move alone, a caller listed", True, [pas("2026-10-29 13:33:45,900", "2026-10-29 13:33:46,100")], (), None),
            ):
                r = rerun_(passes, missed, holes)
                if name == "a log not to hand":
                    q.writers = [N(owner="p/j.py::f", log_files=["j.log", "k.log"])]
                if name == "the move alone, a caller listed":
                    q.writers = [N(owner="p/j.py::f", log_files=["j.log"], called_from=["p/c.py:9"])]
                arms_of = rerun.arms_of
                if name == "no arm":
                    rerun.arms_of = lambda writers, owner, subject, observed: {}
                try:
                    p = r._late_passes(q)
                    said = [p.since, p.before, p.logs, r.asked[0].window, r.asked[0].strings, str(r.asked[0].why)]
                    try:
                        way = r.passes(p)
                        said.append(["exit", way.exit_line] if isinstance(way, rerun.Blocked)
                                    else ["callers", list(way.window), way.kind, way.entries, way.before])
                    except rerun.Stop as e:
                        said.append(["stop", str(e), e.code])
                except rerun.Stop as e:
                    said = ["stop", str(e), e.code]
                got[name] = said
                got[name + ": lines"] = [l.strip() for l in r.lines if "not read" in l]
                q.writers, rerun.arms_of = [N(owner="p/j.py::f", log_files=["j.log"])], arms_of
            # L2 sends a mover that takes the row by its key to judgement 4, and one with a statement of its own on
            r = rerun_([])
            r._row_of = lambda cls: {"PandaID": 1}
            r._movers_in_window = lambda q, kinds: None
            r._late_passes = lambda q: "judgement 4"
            got["L2, by its key"] = r.missed(q)
            q.statements = [N(owner="p/j.py::f", line=9, order_by=["x"], limits=["<= :n"], crosses_to=[], result="rows")]
            r._passed_over = lambda st, q, took: ("not settled", [])
            r._why_late = lambda q, ordered: None
            got["L2, a statement of its own"] = type(r.missed(q)).__name__
            print(json.dumps(got))
            """,
        )
    )
    got = json.loads(out.splitlines()[-1])
    window = ["2026-10-29 09:06:09", "2026-10-29 13:33:46"]
    asked = ["2026-10-29 09:06:09", "2026-10-29 13:33:46", ["j.log"], window, ["f < PandaID=1 >"],
             "from the stay's start to the move, for 'f < PandaID=1 >'"]
    none = ("S2: a pass of f before the one that moved it on at 2026-10-29 13:33:46: none from 2026-10-29 09:06:09, "
            "asked from the stay's start to the move, for 'f < PandaID=1 >' -- the delay is in what calls f on the row, "
            "not followed here (KNOWN_ISSUES.md K45)")
    open_ = ("S2: a pass of f before the one that moved it on at 2026-10-29 13:33:46: none found from 2026-10-29 09:06:09, "
             "asked from the stay's start to the move, {}, and not ruled out: {} (KNOWN_ISSUES.md K45)")
    # the pass before the move is read, the move's is not; with the move's alone, or none, the delay is upstream
    assert got["passes before the move"] == [*asked, ["exit", 4]]
    assert got["passes before the move: lines"] == ["1 pass(es) ending at 2026-10-29 13:33:46 or later, the one that moved it on among them, not read"]
    assert got["the move alone"] == [*asked, ["stop", none, "K45"]]
    # a pass with no record of the key went its way for other rows: counted, not read (K47); with none other before the
    # move, the delay is upstream as above
    keyless = "1 pass(es) of f in the window print no record of the key, nor run it inside a call: not read"
    assert got["a pass without the key"] == [*asked, ["exit", 4]]
    assert got["a pass without the key: lines"] == [keyless, got["passes before the move: lines"][0]]
    assert got["no pass holds the key"] == [*asked, ["stop", none, "K45"]]
    assert got["no pass holds the key: lines"] == [keyless, got["passes before the move: lines"][0].replace("1 pass", "0 pass")]
    # a mover whose mark holds no key is asked by the key alone (tmaeno: "B"), and none found is not none run; nor
    # where the search left a hole
    alone = "for the key alone (its mark holds none)"
    assert got["no mark holds the key"] == [*asked[:4], [], f"from the stay's start to the move, {alone}",
                                            ["stop", open_.format(alone, "a pass printing no line with the key is not found"), "K45"]]
    assert got["a hole in the search"] == [*asked, ["stop", open_.format("for 'f < PandaID=1 >'", "j.log not searched all the window"), "K45"]]
    assert got["an ask not recorded"] == ["stop", "L2: 1 ask(s) of the logs not in the records (written to asks.jsonl)", "asks"]
    assert got["a log not to hand"] == [*asked, ["stop", open_.format("for 'f < PandaID=1 >'", "k.log not to hand"), "K45"]]
    assert got["no arm"] == ["stop", "L2: an arm of f that writes another value than 'assigned'", None]
    # where the map lists a caller of it, no pass before the move goes to the callers' passes before it (K55)
    assert got["the move alone, a caller listed"] == [*asked, ["callers", window, "before", ["p/c.py:9"], "2026-10-29 13:33:46"]]
    assert got["L2, by its key"] == "judgement 4" and got["L2, a statement of its own"] == "Ordered"


def test_up_reads_the_callers_passes_on_the_row_where_no_pass_of_the_mover_took_it(tmp_path):
    corpus = _corpus(tmp_path)
    (corpus / "pandaserver" / "m.py").write_text(
        textwrap.dedent(
            """\
            class Facade:
                def f(self):
                    return proxy.f()

            class A:
                def a(self):
                    if ready:
                        Facade().f()
                    return 1

            def b():
                Facade().f()

            def c():
                d()

            def d():
                Facade().f()

            class A2:
                def a(self):
                    Facade().f()
            """
        )
    )
    out = _ok(
        _run(
            tmp_path,
            """
            import json
            from types import SimpleNamespace as N
            from bamboo.querier import rerun, which_ran
            from bamboo.querier import render as T

            M = "pandaserver/m.py"
            rerun.prints_nothing = lambda owner, lines: owner.endswith("::f")
            rerun.key_marks = lambda owner, line, cls, key: ("a < PandaID=1 >",) if owner.endswith("::a") else ()
            rerun.prints_key = lambda owner, lines, columns: owner.endswith("::b")
            got = {}
            # the caller of an entry: hop 2 behind a facade that prints nothing, by its marks or the key alone; the innermost
            # that prints where no hop is asked for the key; an entry the map cut names none; two defs of one name, two callers
            entries = [f"{M}:8 -> {M}:3", f"{M}:12 -> {M}:3", f"{M}:15 -> {M}:18", "… 2 more", f"{M}:22 -> {M}:3"]
            got["callers_of"] = [[c.owner, c.lines, c.hop, c.how, list(c.marks), c.between]
                                 for c in rerun.callers_of(entries, "JobSpec", "1", {"JobSpec.PandaID"})]

            def rerun_(passes, chosen=None):
                r = object.__new__(rerun.Rerun)
                r.lines, r.hop, r.hand, r.hand_used, r.unanswered = [], "U2", {}, [], []
                r.opened, r.ends, r.branches = [], [], []
                r.mode, r.decide = "map", "positive"
                r.data, r.args = {"statuslog": {"states": []}}, N(observed="assigned", key="1", asks="asks.jsonl", subject="JobSpec.jobStatus")
                r.logs = N(needs_window=True, names=lambda files: list(files))
                r.files_of = lambda owner, today, lines: today
                r._key_columns = lambda subject, key: {"JobSpec.PandaID"}
                run = N(token="a < PandaID=1 >", first="2026-10-29 09:00:00,000", last="2026-10-29 13:33:47,000",
                        describe=lambda: "the run", machine="aipanda081", found_by=[0])
                r.asked = []
                r._ask = lambda key, names, asking, owners=None: (r.asked.append(asking) or ({"a.log": [run]}, [], {}))
                rerun.R.passes = lambda fn, run, mode: (passes, f"{len(passes)} pass(es)")
                r.base_env = lambda at=None: N(values={})
                verdict = which_ran.Verdict(which_ran.Candidate(M, "a", [chosen]), chosen, "positive")
                rerun.judge = lambda cands, runs, **k: which_ran.Judgement([verdict], verdict, "positive evidence", None)
                return r

            rerun.R._prints_of = lambda owner, lines: N(statements=[N(log_files=["a.log"])])
            functions = rerun.R.functions
            rerun.R.function = lambda rel, func, lines: N(node=functions(rel, func)[0], owner=f"{rel}::{func}", name=func, rel=rel)
            rerun.R.seen = lambda fn, run, mode: None
            rerun.stable_holes = lambda *a, **k: None
            rerun.class_of = lambda fn: None
            rerun.holds_key = lambda fn, p, runs, mode, called_at: p.keyed
            rerun.every_log_asked = lambda owner, lines, asked: {"held": True, "why": T.Said("logs-every-file-asked", {"files": asked})}

            def pas(first, last, keyed=True):
                return N(first=first, last=last, records=[0], log=N(name="a.log"), keyed=keyed)

            up = rerun.Callers(f"{M}::f", [rerun.Arm(3, ["activated"], [])], "JobSpec.jobStatus", "1",
                               ("2026-10-29 09:06:09", "2026-10-29 13:33:46"), "before", entries, before="2026-10-29 13:33:46")
            a = rerun.Caller(f"{M}::a", [8], 2, "marks", ("a < PandaID=1 >",), [f"{M}:3"])
            before = pas("2026-10-29 10:00:00,000", "2026-10-29 10:00:01,000")
            moving = pas("2026-10-29 13:33:46,000", "2026-10-29 13:33:47,000")

            def u2(name, caller, passes, chosen=None, callers=up):
                r = rerun_(passes, chosen)
                try:
                    q = r.caller(rerun.CallerPasses(caller, callers))
                    if isinstance(q, rerun.Blocked):
                        got[name] = ["blocked", q.exit_line, [[x.line, x.outcomes, x.guards] for x in q.arms]]
                    else:
                        got[name] = ["callers", q.mover, q.entries, q.up, list(q.window)]
                except rerun.Stop as e:
                    got[name] = ["stop", str(e), e.code]
                got[name + ": asked"] = [list(r.asked[0].window), list(r.asked[0].strings), str(r.asked[0].why)] if r.asked else []

            # every pass of the caller on the row before the move went another way: S3 reads its call's guard
            u2("another way", a, [before, moving], chosen=9)
            # it called the mover there, and no pass of the mover holding the key was found
            u2("it called", a, [before, moving], chosen=8)
            # no pass of it on the row: at hop 2 nothing further is listed; at hop 1 U1 goes up once
            u2("none, hop 2", a, [moving])
            d = rerun.Caller(f"{M}::d", [18], 1, "marks", ("d < PandaID=1 >",))
            u2("none, hop 1", d, [moving])
            u2("none, hop 1, gone up", d, [moving], callers=rerun.Callers(**{**up.__dict__, "up": f"{M}::x"}))
            # asked by the key alone, none found rules nothing out; a caller printing no key is named
            u2("none, by the key alone", rerun.Caller(f"{M}::b", [12], 2, "key"), [pas("2026-10-29 10:00:00,000", "2026-10-29 10:00:01,000", keyed=False)])
            u2("no key printed", rerun.Caller(f"{M}::d", [18], 1, "none"), [])

            # U1: one caller is read on its own, more each in a branch
            r = rerun_([])
            one = r.callers(rerun.Callers(**{**up.__dict__, "entries": entries[:1]}))
            got["U1, one"] = [type(one).__name__, one.caller.owner]
            many = r.callers(up)
            got["U1, many"] = [type(many).__name__, [str(label) for label in many.labels]]
            try:
                r.callers(rerun.Callers(**{**up.__dict__, "entries": ["… 2 more"]}))
            except rerun.Stop as e:
                got["U1, none"] = [str(e), e.code]
            print(json.dumps(got))
            """,
        )
    )
    got = json.loads(out.splitlines()[-1])
    m = "pandaserver/m.py"
    assert got["callers_of"] == [
        [f"{m}::a", [8], 2, "marks", ["a < PandaID=1 >"], [f"{m}:3"]],
        [f"{m}::b", [12], 2, "key", [], [f"{m}:3"]],
        [f"{m}::d", [18], 1, "none", [], []],
        [f"{m}::a", [22], 2, "marks", ["a < PandaID=1 >"], [f"{m}:3"]],
    ]
    asked = [["2026-10-29 09:06:09", "2026-10-29 13:33:46"], ["a < PandaID=1 >"],
             "from 2026-10-29 09:06:09 to 2026-10-29 13:33:46, for 'a < PandaID=1 >'"]
    # the call's arm carries what the mover's arms write, and its guard the statement call_statements picks
    assert got["another way"] == ["blocked", 9, [[8, ["activated"], ["ready"]]]]
    assert got["another way: asked"] == asked
    assert got["it called"] == ["stop", "U2: f's pass on the row inside a's call at 8: every pass of a on the row called there, "
                                "and no pass of f holding the key was found; its run inside the call is not read (KNOWN_ISSUES.md K55)", "K55"]
    assert got["none, hop 2"] == ["stop", "U2: what made a run on the row: none of its passes from 2026-10-29 09:06:09 to "
                                  "2026-10-29 13:33:46 took it, and what makes it run on the row -- its own selection, or a "
                                  "caller past hop 2 of the map's two -- is not read here (KNOWN_ISSUES.md K55)", "K55"]
    assert got["none, hop 1"] == ["callers", f"{m}::d", [f"{m}:15"], f"{m}::d", ["2026-10-29 09:06:09", "2026-10-29 13:33:46"]]
    assert got["none, hop 1, gone up"][0] == "stop" and "past hop 1" in got["none, hop 1, gone up"][1]
    assert got["none, by the key alone"] == [
        "stop", "U2: a pass of b on the row from 2026-10-29 09:06:09 to 2026-10-29 13:33:46: none found, and not ruled out: "
        "prints: asked by the key alone, a pass of it printing no line with the key is not found (KNOWN_ISSUES.md K55)", "K55"]
    assert got["none, by the key alone: asked"][1:] == [[], "from 2026-10-29 09:06:09 to 2026-10-29 13:33:46, for the key alone (its mark holds none)"]
    assert got["no key printed"] == ["stop", "U2: a run of d (calling at [18]) printing the key: no mark of it holds the key and "
                                     "no row of it prints a key column (KNOWN_ISSUES.md K55)", "K55"]
    assert got["no key printed: asked"] == []
    assert got["U1, one"] == ["CallerPasses", f"{m}::a"]
    assert got["U1, many"] == ["Fork", ["the caller a at [8]", "the caller b at [12]", "the caller d at [18]", "the caller a at [22]"]]
    assert got["U1, none"] == ["U1: a caller of f that called_from lists (KNOWN_ISSUES.md K55)", "K55"]


def test_u3_reads_the_call_that_ran_an_arm_with_no_guard_by_what_it_handed_the_arms_run_mark(tmp_path):
    corpus = _corpus(tmp_path)
    (corpus / "pandaserver" / "m.py").write_text(
        textwrap.dedent(
            """\
            class K:
                def arm(self, job):
                    for x in ids:
                        u.jobStatus = job.jobStatus

                def c1(self, job):
                    if job.jobStatus == "finished":
                        self.arm(job)

                def c2(self, jobSpec):
                    if len(done) > 0:
                        self.arm(jobSpec)

                def c3(self, other):
                    self.arm(other)
            """
        )
    )
    out = _ok(
        _run(
            tmp_path,
            """
            import json
            from types import SimpleNamespace as N
            from bamboo.querier import rerun
            from bamboo.querier import render as T

            from bamboo.querier import mapdump

            M = "pandaserver/m.py"
            mapdump._FRAGMENT[:] = [N(tables=[N(spec_class="JobSpec", identity_attrs=["JobSpec.PandaID"])])]
            # each function's run mark, the expression its hole prints and its rows' printed prefix; c3's prints its own
            # parameter, not the name it hands the arm (getEventRanges's pandaID beside the tmpJobSpec it hands): the same
            # row by the hole's column, whatever the hole spells
            MARKS = {"arm": ("job.PandaID", "arm < PandaID=.*? >"), "c1": ("job.PandaID", "c1 < PandaID=.*? >"),
                     "c2": ("jobSpec.PandaID", "c2 < PandaID=.*? >"), "c3": ("pandaID", "c3 < PandaID=.*? >")}
            identity = [["JobSpec.PandaID"]]
            rerun.R._prints_of = lambda owner, lines: N(
                runs=[N(mark="PandaID={}", key=[N(expression=MARKS[owner.split("::")[-1]][0],
                                                  columns=["JobSpec.PandaID"] if owner.endswith("::arm") else identity[0])])],
                statements=[N(log_files=["a.log"], rows=[N(prefix=MARKS[owner.split("::")[-1]][1])])])
            functions = rerun.R.functions
            rerun.R.function = lambda rel, func, lines: N(node=functions(rel, func)[0], owner=f"{rel}::{func}", name=func, rel=rel)

            def run(token, first, last, machine="m1"):
                return N(token=token, first=f"2026-11-02 {first},000", last=f"2026-11-02 {last},000", machine=machine,
                         describe=lambda: f"the run {token!r}")

            arm = run("arm < PandaID=9 >", "13:43:27", "13:43:29")
            got = {}

            def u3(name, runs, token=None, missed=()):
                r = object.__new__(rerun.Rerun)
                r.lines, r.hop, r.hand, r.hand_used, r.unanswered = [], "U3", {}, [], []
                r.opened, r.ends, r.branches = [], [], []
                r.mode, r.decide, r.data, r.args = "map", "positive", {}, N(asks="asks.jsonl")
                r.logs = N(needs_window=True, names=lambda files: list(files))
                r.files_of = lambda owner, today, lines: today

                async def writers_of(subject):
                    return [N(owner=f"{M}::arm", called_from=[f"{M}:8", f"x.py:1 -> {M}:12", f"{M}:15", "… 2 more"])]

                r._writers_of = writers_of
                asked, held = [], []
                r._ask = lambda key, names, asking, owners=None: (
                    asked.append([key, names, list(asking.window), asking.machine, list(asking.strings), str(asking.why)])
                    or ({"a.log": runs}, list(missed), {}))
                r._held_on = lambda fn, site, guards, run: (
                    held.append([fn.name, site.lineno, guards, run.token])
                    or ([T.Said("guard-held-on", {"guard": g, "on": " (printed)"}) for g in guards], None))
                q = rerun.ArmCaller(rerun.Asked("JobSpec.jobStatus", "closed", "7"), f"{M}::arm", 4,
                                    arm if token is None else run(token, "13:43:27", "13:43:29"))
                try:
                    r.arm_caller(q)
                    got[name] = ["no stop"]
                except rerun.Stop as e:
                    got[name] = [str(e), e.code]
                got[name + ": asked"], got[name + ": held"] = asked, held

            # 4822's way: archiveJob's run holds propagateResultToJEDI's, which holds the arm's; the innermost is the call's
            u3("innermost", [run("c1 < PandaID=9 >", "13:43:20", "13:43:40"), run("c2 < PandaID=9 >", "13:43:25", "13:43:35"), arm])
            # 1783's way: the other caller's run ended before the arm's began
            u3("the one holding it", [run("c1 < PandaID=9 >", "13:43:20", "13:43:40"), run("c2 < PandaID=9 >", "13:43:21", "13:43:24")])
            # a caller's run on another machine, or one the window cut, holds nothing
            u3("none holds it", [run("c1 < PandaID=9 >", "13:43:20", "13:43:40", "m2"), run("c2 < PandaID=9 >", "13:43:28", "13:43:35")])
            u3("asks", [], missed=[(N(key="c1 < PandaID=9 >"), "why")])
            u3("no mark read", [], token="arm < vo=atlas >")
            # the callers' marks hold no row's identity (a hole of another column)
            identity[0] = ["JobSpec.jobsetID"]
            u3("no identity", [])
            print(json.dumps(got))
            """,
        )
    )
    got = json.loads(out.splitlines()[-1])
    # the callers' marks holding the row the arm's mark holds (the job's identity column), filled with its run's value,
    # over its run on its machine; c3's mark prints its own pandaID, not the name it hands, and holds the same row
    asked = [["9", ["a.log"], ["2026-11-02 13:43:27", "2026-11-02 13:43:29"], "m1",
              ["c1 < PandaID=9 >", "c2 < PandaID=9 >", "c3 < PandaID=9 >"],
              "over the arm's run, 2026-11-02 13:43:27 .. 2026-11-02 13:43:29, for 'c1 < PandaID=9 >', 'c2 < PandaID=9 >', "
              "'c3 < PandaID=9 >'"]]
    assert got["innermost: asked"] == asked
    assert got["innermost"] == ["U3: arm:4 ran inside c2's call at 12, which ran under its guards: where the values its guards "
                                "held on come from, its run showing it ran ('len(done) > 0' held (printed)) (KNOWN_ISSUES.md K56)", "K56"]
    assert got["innermost: held"] == [["c2", 12, ["len(done) > 0"], "c2 < PandaID=9 >"]]
    assert got["the one holding it"][0].startswith("U3: arm:4 ran inside c1's call at 8, which ran under its guards")
    assert got["the one holding it: held"] == [["c1", 8, ["job.jobStatus == 'finished'"], "c1 < PandaID=9 >"]]
    assert got["none holds it"] == ["U3: a run of c1, c2, c3 holding arm's run whole (2026-11-02 13:43:27,000 .. 2026-11-02 "
                                    "13:43:29,000, on its machine): 2 run(s) found, none holds it (KNOWN_ISSUES.md K56)", "K56"]
    assert got["asks"] == ["U3: 1 ask(s) of the logs not in the records (written to asks.jsonl)", "asks"]
    assert got["no identity"] == ["U3: a call of arm's whose caller's run marks hold the row its run's mark holds: none "
                                  "of pandaserver/m.py:8, pandaserver/m.py:12, pandaserver/m.py:15 does (KNOWN_ISSUES.md K56)", "K56"]
    assert got["no identity: asked"] == []
    assert got["no mark read"] == ["U3: which row arm's call handed it: no mark the map records for arm reads a row's identity "
                                   "in its run 'arm < vo=atlas >' (KNOWN_ISSUES.md K56)", "K56"]


def test_an_attributes_answer_ends_at_h3_where_its_arms_guards_hold_and_names_those_not_read(tmp_path):
    out = _ok(
        _run(
            tmp_path,
            """
            import json
            from types import SimpleNamespace as N
            from bamboo.querier import evaluate as E
            from bamboo.querier import rerun
            from bamboo.querier.which_ran import Candidate, Verdict

            rerun.R = N(function=lambda rel, func, lines: N(name=func, node=func), statement_at=lambda node, line: (node, line),
                        entries=lambda fn, run, mode: [0], must_rows=lambda node, site: (None, []), seen=lambda fn, run, mode: N(hits={}))
            # a guard's call is read through the pure functions, which E fills from pandaserver at the first call; the
            # dev install (CI) has no pandaserver, and useScout is no pure function either way
            E.PURE.setdefault("<no pure function is read here>", None)
            run = N(log=N(name="panda-TaskRefiner.log"), machine="aipanda084.cern.ch", token="< jediTaskID=7 >", records=[0, 1],
                    first="2026-10-29 04:10:00,100", last="2026-10-29 04:10:01,200", describe=lambda: "the run", said=lambda: "the run")
            readings = [N(file="pandajedi/r.py", owner="pandajedi/r.py::extract")]
            arm = Verdict(Candidate("pandajedi/r.py", "extract", [5]), 5, "positive")
            call = Verdict(Candidate("pandajedi/o.py", "runImpl", [211]), 211, "positive")

            def asked(subject, guards, row=None, via=None, window=("2026-10-29 04:09:50", "2026-10-29 04:10:10"), kind="positive",
                      env=None, unsearched=None):
                r = object.__new__(rerun.Rerun)
                r.lines, r.opened, r.ends, r.touched, r.hop, r.hand, r.hand_used, r.result, r.mode = [], [], [], [], "", {}, [], {}, "outside"
                r.data = {}  # no rows file: no held column (heldcol, K49)
                r.base_env = lambda at=None: E.Env({})
                chosen = Verdict(arm.candidate, arm.line, kind)
                r._which_wrote = lambda *a: rerun.Wrote(readings, run, [], E.Env({}) if env is None else env, chosen, via=via, row=row,
                                                        window=window, unsearched=unsearched or {})
                rerun.pathcond = N(arm_condition=lambda site: guards, path_condition=lambda site: guards,
                                   exits_above_at=lambda site: [], ancestors=lambda node: iter(()))
                try:
                    got = r.asked(rerun.Asked(subject, "A", "7"))
                    said = (None if got is None else [type(got).__name__, got.owner, got.line] if isinstance(got, rerun.ArmCaller)
                            else [type(got).__name__])
                except rerun.Stop as e:
                    said = ["STOP", str(e)]
                return {"said": said, "ends": [f"END {said}" for _, said in r.ends], "wrote": r.result.get("wrote")}

            row = {"status": "defined", "time": "2026-10-29 04:10:00.123456"}
            second = ("2026-10-29 00:49:09", "2026-10-29 00:49:09")
            holes = {"panda-JediDBProxy.log": ["aipanda081.cern.ch: 2026-10-29 00:49:09 .. 2026-10-29 00:49:09",
                                               "aipanda086.cern.ch: 2026-10-29 00:49:09 .. 2026-10-29 00:49:09"]}
            held = E.Env({"task.x": E.Value(True, "hole")})
            # a run whose mark the map's runs of the function read, and one marked by a logger the caller handed in
            marked = [False]
            rerun.mark_identities = lambda owner, line, run: [("JediTaskSpec", "7")] if marked[0] else []
            no_mark = asked("JediTaskSpec.status", [], row)
            marked[0] = True
            print(json.dumps({
                "no guard": asked("JediTaskSpec.transUses", [], row),
                "guards read true, a stand-in": asked("JediTaskSpec.transUses", ["1 == 1"], None, call,
                                                      ("2026-10-29 04:10:00", "2026-10-29 04:10:00")),
                "by elimination": asked("JediTaskSpec.rescueTime", [], None, window=second, kind="kept"),
                "by elimination, not searched": asked("JediTaskSpec.rescueTime", [], None, window=second, kind="kept",
                                                      unsearched=holes),
                "by elimination, a guard the run's values hold": asked("JediTaskSpec.rescueTime", ["task.x"], None, window=second,
                                                                       kind="kept", env=held),
                "by elimination, guards read true": asked("JediTaskSpec.rescueTime", ["1 == 1"], None, window=second, kind="kept"),
                "shown, a guard the run's values hold": asked("JediTaskSpec.rescueTime", ["task.x"], None, window=second, env=held),
                "not read": asked("JediTaskSpec.transUses", ["1 == 1", "task.x", "1 == 2"], row),
                "shown, a guard not read": asked("JediTaskSpec.transUses", ["task.y"], row),
                "shown, one guard read and one not": asked("JediTaskSpec.transUses", ["1 == 1", "task.y"], row),
                "a stand-in, a guard not read": asked("JediTaskSpec.transUses", ["task.y"], None, call,
                                                      ("2026-10-29 04:10:00", "2026-10-29 04:10:00")),
                "through a method": asked("JediTaskSpec.transUses", ["task.useScout()"], row),
                "a status": asked("JediTaskSpec.status", [], row),
                "a status, no guard, no mark the map reads": no_mark,
                "a status, no guard, a stand-in": asked("JediTaskSpec.status", [], None, call, ("2026-10-29 04:10:00", "2026-10-29 04:10:00")),
                "a status, a guard through no method": asked("JediTaskSpec.status", ["task.x < 3"], row),
                "a status, a stand-in": asked("JediTaskSpec.status", ["task.x < 3"], row, call),
            }))
            """,
        )
    )
    got = json.loads(out.splitlines()[-1])
    # an attribute: H2's pass is the answer, written to result.json; an arm with no guard ends at H3
    assert got["no guard"] == {
        "said": None,
        "ends": ["END JediTaskSpec.transUses='A' written last by extract:5 (shown in its run; it has no guard), in the pass of "
                 "the statuslog row 'defined' at 2026-10-29 04:10:00.123456: the run"],
        "wrote": {"owner": "pandajedi/r.py::extract", "line": 5, "shown": True, "via": None,
                  "row": {"status": "defined", "time": "2026-10-29T04:10:00.123456Z"},
                  "window": {"start": "2026-10-29T04:09:50Z", "end": "2026-10-29T04:10:10Z"},
                  "run": {"log": "panda-TaskRefiner.log", "machine": "aipanda084.cern.ch", "token": "< jediTaskID=7 >",
                          "first": "2026-10-29 04:10:00,100", "last": "2026-10-29 04:10:01,200"}},
    }
    # ... and one whose guards hold in the run's values; a writer that prints nothing says the call it is read at
    stood = got["guards read true, a stand-in"]
    assert stood["ends"] == ["END JediTaskSpec.transUses='A' written last by extract:5, called at runImpl:211 (shown in its run; "
                             "its guards hold in the run's values), at 2026-10-29 04:10:00: the run"]
    # an arm no run shows ran, left by elimination (the one writer of a value that is its time): said so, and no run
    # named -- the run the judge read it on is any of the key's at the window -- but the writer's logs not searched there
    left = got["by elimination"]
    assert left["ends"] == ["END JediTaskSpec.rescueTime='A' written last by extract:5 (by elimination: no run shows the "
                            "line ran; it has no guard), at 2026-10-29 00:49:09"]
    assert (left["said"], left["wrote"]["shown"], left["wrote"]["run"]) == (None, False, None)
    assert got["by elimination, not searched"]["ends"] == [
        "END JediTaskSpec.rescueTime='A' written last by extract:5 (by elimination: no run shows the line ran; it has no "
        "guard), at 2026-10-29 00:49:09; not searched there: panda-JediDBProxy.log on aipanda081.cern.ch, aipanda086.cern.ch"]
    # its guards are read in no run's values, and H3 stops there; an arm shown reads them in its run's
    assert got["by elimination, a guard the run's values hold"] == {
        "said": ["STOP", "H3: a run that shows extract:5 ran, to read its guards in (by elimination; its conditions: "
                         "'task.x' not read)"],
        "ends": [], "wrote": left["wrote"]}
    assert got["by elimination, guards read true"]["ends"] == [
        "END JediTaskSpec.rescueTime='A' written last by extract:5 (by elimination: no run shows the line ran; its guards "
        "hold in the rows' values), at 2026-10-29 00:49:09"]
    assert got["shown, a guard the run's values hold"]["ends"] == [
        "END JediTaskSpec.rescueTime='A' written last by extract:5 (shown in its run; its guards hold in the run's values), "
        "at 2026-10-29 00:49:09: the run"]
    assert (stood["said"], stood["wrote"]["via"], stood["wrote"]["row"]) == (None, {"owner": "pandajedi/o.py::runImpl", "line": 211}, None)
    # a condition read false is named where H3 stops, with those not read: the values the guard reads, through no method
    assert got["not read"]["said"] == ["STOP", "H3: the values its guard reads at its run (its conditions: 'task.x' not read, "
                                               "'1 == 2' read false)"]
    assert (got["not read"]["ends"], got["not read"]["wrote"]["line"]) == ([], 5)
    # an arm its run shows ran ran under its guards: one the values leave unread held (K54)
    assert got["shown, a guard not read"]["ends"] == [
        "END JediTaskSpec.transUses='A' written last by extract:5 (shown in its run; its guards hold as its run shows it ran "
        "('task.y')), in the pass of the statuslog row 'defined' at 2026-10-29 04:10:00.123456: the run"]
    assert got["shown, one guard read and one not"]["ends"] == [
        "END JediTaskSpec.transUses='A' written last by extract:5 (shown in its run; its guards hold as its run shows it ran "
        "('task.y'); the others in the run's values), in the pass of the statuslog row 'defined' at 2026-10-29 "
        "04:10:00.123456: the run"]
    # ... not a stand-in's, whose run is its caller's
    assert got["a stand-in, a guard not read"]["said"] == [
        "STOP", "H3: the values its guard reads at its run (its conditions: 'task.y' not read)"]
    # a guard read through a call is read as any other: it held where the run shows the arm ran (K74)
    assert got["through a method"]["said"] is None
    assert got["through a method"]["ends"] == [
        "END JediTaskSpec.transUses='A' written last by extract:5 (shown in its run; its guards hold as its run shows it ran "
        "('task.useScout()')), in the pass of the statuslog row 'defined' at 2026-10-29 04:10:00.123456: the run"]
    # a status: no wrote; with no guard, what made the arm run is its callers': U3 reads the call that ran it, where its
    # run shows it ran and the map reads the run's mark -- one a handed-in logger marks, or a stand-in's run, its
    # caller's, H3 stops on; through no method, its guards held, and what H3 lacks is where the values they held on
    # came from (none printed here); a stand-in's are not read
    assert got["a status"] == {"said": ["ArmCaller", "pandajedi/r.py::extract", 5], "ends": [], "wrote": None}
    no_guard = ["STOP", "H3: extract:5 has no guard of its own to read: what made it run is its callers'"]
    assert got["a status, no guard, no mark the map reads"]["said"] == no_guard
    assert got["a status, no guard, a stand-in"]["said"] == no_guard
    assert got["a status, a guard through no method"]["said"] == [
        "STOP", "H3: where the values its guards held on come from, its run showing it ran ('task.x < 3' held)"]
    assert got["a status, a stand-in"]["said"] == [
        "STOP", "H3: the values its guard reads at its run (its conditions: 'task.x < 3' not read)"]


def test_a_status_arm_whose_value_a_called_method_returns_reads_the_maps_ways_to_it_and_the_one_left_held(tmp_path):
    out = _ok(
        _run(
            tmp_path,
            """
            import json
            from types import SimpleNamespace as N
            from bamboo.querier import evaluate as E
            from bamboo.querier import rerun
            from bamboo.querier.which_ran import Candidate, Verdict

            rerun.R = N(function=lambda rel, func, lines: N(name=func, node=func), statement_at=lambda node, line: (node, line),
                        entries=lambda fn, run, mode: [0], must_rows=lambda node, site: (None, []), seen=lambda fn, run, mode: N(hits={}))
            # a guard's call (spec.is_hpo()) is read through the pure functions, which E fills from pandaserver at the first
            # call; the dev install (CI) has no pandaserver, and the call is no pure function either way
            E.PURE.setdefault("<no pure function is read here>", None)
            rerun.pathcond = N(arm_condition=lambda site: [], path_condition=lambda site: [],
                               exits_above_at=lambda site: [], ancestors=lambda node: iter(()))
            rerun.mark_identities = lambda owner, line, run: []
            run = N(log=N(name="panda-PostProcessor.log"), machine="aipanda086.cern.ch", token="<jediTaskID=7>", records=[0, 1],
                    first="2026-11-02 15:38:55,463", last="2026-11-02 15:38:55,831", describe=lambda: "the run", said=lambda: "the run")
            readings = [N(file="pandajedi/p.py", owner="pandajedi/p.py::post")]
            IN = "  [in final()]"
            # post:5 is `taskSpec.status = self.final(taskSpec)`: the callee's parameter is 'spec', the call hands it
            # 'taskSpec' (the map's call_bindings); 'check' is left to its default and 'stat' and 'n' are the callee's own
            ways = {
                "broken": [["spec.status == 'tobroken'" + IN, "not (spec.is_hpo() and stat)" + IN]],
                "done": [["check and spec.parent_tid not in [None, spec.jediTaskID]" + IN, "parent == 'done'" + IN],
                         ["not (check and spec.parent_tid not in [None, spec.jediTaskID])" + IN, "n == done_n" + IN]],
                "finished": [["n == done_n" + IN, "parent != 'done'" + IN], ["not (n == done_n)" + IN, "not (done_n == 0)" + IN]],
            }
            rerun.mapdump = N(fragment=lambda: N(junctions=[N(
                subject="JediTaskSpec.status", owner="pandajedi/p.py::Post.post",
                branches=[N(line=5, outcome=value, path_condition=way) for value, ws in ways.items() for way in ws]
                         + [N(line=9, outcome="broken", path_condition=["x"])])]))
            bindings = [N(line=5, to="pandajedi/p.py::Post.final", parameter="spec", path="taskSpec"),
                        N(line=5, to="pandajedi/p.py::Post.final", parameter="self", path="self")]

            def asked(value, before, line=5, kind="positive"):
                r = object.__new__(rerun.Rerun)
                r.lines, r.opened, r.ends, r.touched, r.hop, r.hand, r.hand_used, r.result, r.mode = [], [], [], [], "", {}, [], {}, "outside"
                state = {"status": value, "time": "2026-11-02 15:38:55"}
                r.data = {"statuslog": {"states": [{"status": "running", "time": "2026-11-02 14:00:00"},
                                                   {"status": before, "time": "2026-11-02 15:38:44"}, state]}}
                env = E.Env({}, rows={"taskSpec": "task"}, subject="<task>.status")
                env.table["task"] = {"jediTaskID": E.Value(7, "the key"), "parent_tid": E.Value(7, "the task's row")}
                r.base_env = lambda at=None: env
                r._graph = lambda: (None, None, [N(owner="pandajedi/p.py::Post.post", call_bindings=bindings)])
                arm = Verdict(Candidate("pandajedi/p.py", "post", [line]), line, kind)
                r._which_wrote = lambda *a: rerun.Wrote(readings, run, [], env, arm, row=state,
                                                        window=("2026-11-02 15:38:55", "2026-11-02 15:38:55"))
                try:
                    r.asked(rerun.Asked("JediTaskSpec.status", value, "7"))
                except rerun.Stop as e:
                    return str(e)
                return None

            print(json.dumps({
                "the only way": asked("broken", "tobroken"),
                "the only way, read false": asked("broken", "running"),
                "two ways, one read false": asked("done", "prepared"),
                "two ways left": asked("finished", "prepared"),
                "one way left, its run not showing it ran": asked("broken", "tobroken", kind="kept"),
                "no way through a method": asked("broken", "tobroken", line=9),
            }))
            """,
        )
    )
    got = json.loads(out.splitlines()[-1])
    before = "the statuslog row before it, 'tobroken' at 2026-11-02 15:38:44"
    # the one way to the value: its term read in the caller's names -- the row's status as the statuslog row before the
    # question's held it -- and the term no value reads held, since no other way gives the value (K65)
    assert got["the only way"] == (
        "H3: post:5's value 'broken' comes from final(): the one of its 1 way(s) to it the values read leave, its run "
        f"showing it ran (\"spec.status == 'tobroken'  [in final()]\" held on taskSpec.status ({before}); "
        "'not (spec.is_hpo() and stat)  [in final()]' held, the only way to the value)")
    # ... a term read false on the only way is no answer: the run shows the arm ran
    assert got["the only way, read false"] == (
        "H3: post:5's value 'broken' comes from final(): each of its 1 way(s) to it reads a term false (\"spec.status == "
        "'tobroken'  [in final()]\" read false, 'not (spec.is_hpo() and stat)  [in final()]' not read), though its run "
        "shows it ran")
    # the call hands 'spec' the row: the parent is the task itself, so the way through the parent is not the one taken,
    # whatever 'check' was
    only = got["two ways, one read false"]
    assert only.startswith("H3: post:5's value 'done' comes from final(): the one of its 2 way(s) to it the values read leave, ")
    assert "'not (check and spec.parent_tid not in [None, spec.jediTaskID])  [in final()]' held on " in only
    assert "taskSpec.parent_tid (the task's row)" in only and "taskSpec.jediTaskID (the key)" in only
    assert only.endswith("'n == done_n  [in final()]' held, the only way to the value)")
    # two ways the values leave: named, with what they leave unread
    assert got["two ways left"] == (
        "H3: post:5's value 'finished' comes from final(): 2 of its 2 way(s) to it the values read leave ('n == done_n  "
        "[in final()]' not read, \"parent != 'done'  [in final()]\" not read | 'not (n == done_n)  [in final()]' not read, "
        "'not (done_n == 0)  [in final()]' not read)")
    # one way left of an arm no run shows ran: it may not have written, so nothing is held
    assert got["one way left, its run not showing it ran"].startswith(
        "H3: post:5's value 'broken' comes from final(): 1 of its 1 way(s) to it the values read leave (")
    # a branch whose terms are its own: the arm has no guard to read, as before
    assert got["no way through a method"] == "H3: post:9 has no guard of its own to read: what made it run is its callers'"


_FINISHER = """
class Finisher:
    def __init__(self, jobs, time_out):
        self.logger = logger
        self.jobs = jobs
        self.time_out = time_out

    def run(self):
        for job in self.jobs:
            n_tokens = 0
            for f in job.files:
                n_tokens += 1
            n_ok = count(job)
            self.logger.debug(f"tokens n:{n_tokens} ok:{n_ok}")
            if n_tokens <= n_ok:
                self.logger.debug("finish")
            else:
                end = job.endTime
                self.logger.debug(f"end:{str(end)} limit:{str(self.time_out)}")
                if end < self.time_out:
                    diag = reason(job)
                    if diag != "":
                        job.errorDiag = diag
                        self.logger.debug(f"kill with {job.errorDiag}")
                        job.jobStatus = "failed"
"""


def test_a_status_arm_its_run_shows_ran_names_the_values_its_guards_held_on_where_its_run_printed_them(tmp_path):
    (_corpus(tmp_path) / "pandaserver" / "v.py").write_text(_FINISHER)
    lines = _FINISHER.splitlines()
    site = lines.index('                        job.jobStatus = "failed"') + 1
    stamp = "2026-01-01 00:00:0{},000 panda.log.X: DEBUG    <PandaID=1> {}\n"
    tokens, end, kill = "tokens n:1 ok:0", "end:2026-01-01 00:00:00 limit:2026-01-02 00:00:00.500000", "kill with no replica"
    logs = {
        "killed": [tokens, end, kill],
        # the printed counts read the first guard false, which the run says held: a misread, or not this pass
        "contradicted": ["tokens n:0 ok:1", end, kill],
        # the key's records show the loop's row twice: they are not one pass of the loop, and no row in it is read
        "twice": [tokens, tokens, end, kill],
    }
    for name, text in logs.items():
        (tmp_path / f"{name}.log").write_text("".join(stamp.format(n, t) for n, t in enumerate(text)))
    code = """
        import json
        from bamboo.codemap.panda import pathcond
        from bamboo.querier import render as T
        from bamboo.querier import rerun
        from bamboo.querier import rows as R
        from bamboo.querier.records import read_log, runs_for

        R.ROWS_FROM = "walk"
        fn = R.function("pandaserver/v.py", "run", within=[SITE])
        site = R.statement_at(fn.node, SITE)
        guards = pathcond.path_condition(site)
        r = object.__new__(rerun.Rerun)
        r.lines, r.mode = [], "outside"
        got = {"guards": guards}
        for name in ("killed", "contradicted", "twice"):
            run, = runs_for(read_log(f"{name}.log"), "1")
            held, false = r._held_on(fn, site, guards, run)
            got[name] = {"held": [T.text(h) for h in held], "false": None if false is None else [false[0], T.text(false[1])]}
        print(json.dumps(got))
    """
    out = _ok(_run(tmp_path, code.replace("SITE", str(site))))
    got = json.loads(out.splitlines()[-1])
    assert got["guards"] == ["not (n_tokens <= n_ok)", "end < self.time_out", "diag != ''"]
    row = {k: lines.index(next(x for x in lines if f'f"{k}' in x)) + 1 for k in ("tokens", "end", "kill")}
    bound = lines.index("                        job.errorDiag = diag") + 1

    def at(n: int, k: str) -> str:
        return f"row run:{row[k]} at killed.log:{n + 1}"

    # each value said where the run printed it: a count before the guard, a time through str(), and diag through the
    # one assignment after its guard that binds the printed chain to it
    killed = got["killed"]
    assert [h.split(" held")[0] for h in killed["held"]] == ["'not (n_tokens <= n_ok)'", "'end < self.time_out'", '"diag != \'\'"']
    assert killed["held"][0] == f"'not (n_tokens <= n_ok)' held on n_tokens = '1' ({at(0, 'tokens')}), n_ok = '0' ({at(0, 'tokens')})"
    assert killed["held"][1] == (f"'end < self.time_out' held on end = '2026-01-01 00:00:00' ({at(1, 'end')}), "
                                 f"self.time_out = '2026-01-02 00:00:00.500000' ({at(1, 'end')})")
    assert killed["held"][2] == f"\"diag != ''\" held on diag = 'no replica' ({at(2, 'kill')}, as job.errorDiag bound at v.py:{bound})"
    # two times read back as text are not ordered: no guard is read false
    assert killed["false"] is None
    assert got["contradicted"]["false"][0] == "not (n_tokens <= n_ok)"
    assert got["contradicted"]["false"][1].startswith(" on n_tokens = '0' (row run:")
    assert got["twice"] == {"held": ["'not (n_tokens <= n_ok)' held", "'end < self.time_out' held", "\"diag != ''\" held"],
                            "false": None}


_EXITER = """
class Exiter:
    def __init__(self):
        self.logger = logger

    def run(self, job):
        mode = pick(job)
        self.logger.debug(f"mode:{mode}")
        if mode == "skip":
            return
        flag = job.flag
        self.logger.debug(f"flag:{flag}")
        if flag == "none":
            return
        flag = fix(flag)
        if flag:
            job.jobStatus = "failed"
"""


def test_h3_reads_an_arms_early_exits_as_its_guards_each_at_its_own_if(tmp_path):
    (_corpus(tmp_path) / "pandaserver" / "x.py").write_text(_EXITER)
    lines = _EXITER.splitlines()
    site = lines.index('            job.jobStatus = "failed"') + 1
    skip, none = (lines.index(f'        if {t}:') + 1 for t in ('mode == "skip"', 'flag == "none"'))
    stamp = "2026-01-01 00:00:0{},000 panda.log.X: DEBUG    <PandaID=1> {}\n"
    logs = {"ran": ["mode:go", "flag:x"], "skipped": ["mode:skip", "flag:x"], "noned": ["mode:go", "flag:none"]}
    for name, text in logs.items():
        (tmp_path / f"{name}.log").write_text("".join(stamp.format(n, t) for n, t in enumerate(text)))
    code = """
        import json
        from bamboo.codemap.panda import pathcond
        from bamboo.querier import render as T
        from bamboo.querier import rerun
        from bamboo.querier import rows as R
        from bamboo.querier.records import read_log, runs_for

        R.ROWS_FROM = "walk"
        fn = R.function("pandaserver/x.py", "run", within=[SITE])
        site = R.statement_at(fn.node, SITE)
        guards = pathcond.arm_condition(site)
        r = object.__new__(rerun.Rerun)
        r.lines, r.mode = [], "outside"
        got = {"guards": guards}
        for name in ("ran", "skipped", "noned"):
            run, = runs_for(read_log(f"{name}.log"), "1")
            held, false = r._held_on(fn, site, guards, run)
            got[name] = {"held": [T.text(h) for h in held], "false": None if false is None else [false[0], T.text(false[1])]}
        print(json.dumps(got))
    """
    out = _ok(_run(tmp_path, code.replace("SITE", str(site))))
    got = json.loads(out.splitlines()[-1])
    # the if around the write, then the exits above it; flag is given a new value between the second exit and the
    # write, so that exit's test reads it as it is at its own if (KNOWN_ISSUES.md K61)
    assert got["guards"] == ["flag", "not (mode == 'skip')", f"not (flag == 'none')  [at {none}]"]
    row = {k: lines.index(next(x for x in lines if f'f"{k}:' in x)) + 1 for k in ("mode", "flag")}
    # each exit's test is read at its own if, on the values the run printed before it; flag, given a new value after
    # its print, is not read at the write's if
    assert got["ran"]["held"] == [
        "'flag' held",
        f"\"not (mode == 'skip')\" held on mode = 'go' (row run:{row['mode']} at ran.log:1)",
        f"\"not (flag == 'none')  [at {none}]\" held on flag = 'x' (row run:{row['flag']} at ran.log:2)",
    ]
    assert got["ran"]["false"] is None
    # a pass that printed what takes an exit is read false at that exit: a misread, or not this pass
    assert got["skipped"]["false"][0] == "not (mode == 'skip')"
    assert got["noned"]["false"][0] == f"not (flag == 'none')  [at {none}]"
    assert skip < none < site


def test_rule_6_reads_an_early_exit_above_the_write_as_a_guard_on_the_way_to_it(tmp_path):
    corpus = _corpus(tmp_path)
    (corpus / "pandaserver" / "a.py").write_text(
        textwrap.dedent(
            """
            class A:
                def handle(self):
                    if self.job.jobStatus != "failed":
                        return
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
            print(json.dumps(rewriter(fn, node.body[1], "jobStatus", "failed", E.Env({}, {"self.job": "job"}))))
            """,
        )
    )
    # the return above it leaves the write reached only when the job is already failed (KNOWN_ISSUES.md K61)
    assert json.loads(out.splitlines()[-1]).startswith(
        "rule 6: reached only when self.job.jobStatus is already 'failed' (guard \"not (self.job.jobStatus != 'failed')\"")


_NONE_TAKEN = """
class Mover:
    def __init__(self):
        self.logger = logger

    def run(self, job):
        self.logger.debug(f"flag={job.flag}")
        if job.flag:
            self.logger.debug("flagged")
            job.jobStatus = "failed"
        self.logger.debug("done")
"""


def test_a_pass_that_took_none_of_the_candidates_did_not_move_the_value_and_s3_reads_why(tmp_path):
    # setupper_atlas_plugin's run: its one arm 212, which the row 206 its pass printed excludes, is no way the pass
    # went, and S2 stopped "which way pass 1 went"
    (_corpus(tmp_path) / "pandaserver" / "m.py").write_text(_NONE_TAKEN)
    lines = _NONE_TAKEN.splitlines()
    arm = lines.index('            job.jobStatus = "failed"') + 1
    (tmp_path / "x.log").write_text("".join(f"2026-01-01 00:00:0{n},000 panda.log.X: DEBUG    <PandaID=1> {t}\n"
                                            for n, t in enumerate(["flag=False", "done"])))
    code = """
        import json
        from types import SimpleNamespace as N
        from bamboo.querier import evaluate as E
        from bamboo.querier import rerun
        from bamboo.querier import rows as R
        from bamboo.querier.records import read_log, runs_for
        from bamboo.querier.which_ran import class_of, stable_holes

        verdicts = lambda *kinds: N(chosen=None, verdicts=[N(kind=k) for k in kinds])
        took = [rerun.took_none(verdicts("excluded", "excluded")), rerun.took_none(verdicts("excluded", "kept")),
                rerun.took_none(verdicts("excluded", "unexamined")), rerun.took_none(verdicts())]
        R.ROWS_FROM = "walk"
        fn = R.function("pandaserver/m.py", "run", within=[ARM])
        run, = runs_for(read_log("x.log"), "1")
        env = E.Env()
        stable_holes([R.seen(fn, run, "outside")], env, class_of(fn), local_to=fn.owner)
        r = object.__new__(rerun.Rerun)
        r.lines, r.mode, r.opened, r.hand, r.hand_used = [], "outside", [], {}, []
        got = r.blocked(rerun.Blocked(fn, None, [run], [env], [rerun.Arm(ARM, ["failed"], ["job.flag"])]))
        print(json.dumps([took, got.guard, got.line, [x for x in r.lines if x]]))
    """
    out = _ok(_run(tmp_path, code.replace("ARM", str(arm))))
    took, guard, line, said = json.loads(out.splitlines()[-1])
    # every candidate excluded: none taken; one kept or not examined may have been taken
    assert took == [True, False, False, False]
    # no exit's path: the guard of the arm is read false in what the pass printed, and S3 goes on to where it is tested
    assert guard == "job.flag" and line == lines.index("        if job.flag:") + 1
    assert said[1] == "run: every pass took none of its candidates, so no exit's path settles a guard"
    assert said[-1] == f"the nearest arm(s) [{arm}] were kept off by: 'job.flag' (rule 21)"


_QUIET = """
class Mover:
    def __init__(self):
        self.logger = logger

    def jumbo(self, job):
        if len(self.jumbo_jobs) == 0:
            return
        self.logger.debug("start")
        job.jobStatus = "failed"

    def run(self, job):
        try:
            self.logger.debug(f"begin {job.PandaID}")
            self.work()
            self.logger.debug("end")
        except Exception:
            if job.jobStatus not in ["failed", "cancelled"]:
                job.jobStatus = "failed"
            self.logger.error("setupper.run failed")
"""


def test_a_return_has_no_row_after_it_and_an_arm_whose_try_raised_nothing_is_excluded(tmp_path):
    # setup_jumbo_jobs's ``if len(self.jumbo_jobs) == 0: return`` was excluded by the "start" after it, which only
    # the other way prints; setupper_atlas_plugin's run printed "end", its try's last statement, so the arm in its
    # handler did not run, and S3 stopped "a guard the passes made false against every arm"
    (_corpus(tmp_path) / "pandaserver" / "m.py").write_text(_QUIET)
    lines = _QUIET.splitlines()
    ret, start = lines.index("            return") + 1, lines.index('        self.logger.debug("start")') + 1
    arm, handler = lines.index('                job.jobStatus = "failed"') + 1, lines.index("        except Exception:") + 1
    end = lines.index('            self.logger.debug("end")') + 1
    line = "2026-01-01 00:00:0{},000 panda.log.X: DEBUG    <PandaID=1> {}\n"
    (tmp_path / "ended.log").write_text(line.format(0, "begin 1") + line.format(1, "end"))
    (tmp_path / "cut.log").write_text(line.format(0, "begin 1"))
    code = """
        import json
        from types import SimpleNamespace as N
        from bamboo.querier import evaluate as E
        from bamboo.querier import rerun
        from bamboo.querier import rows as R
        from bamboo.querier.records import read_log, runs_for
        from bamboo.querier.which_ran import class_of, stable_holes

        R.ROWS_FROM = "walk"
        jumbo = R.function("pandaserver/m.py", "jumbo", within=[ARM])
        must = [[s.lineno for s in part] for part in R.must_rows(jumbo.node, R.statement_at(jumbo.node, RET))]
        fn = R.function("pandaserver/m.py", "run", within=[ARM])
        got = {}
        for name in ("ended", "cut"):
            run, = runs_for(read_log(f"{name}.log"), "1")
            env = E.Env()
            stable_holes([R.seen(fn, run, "outside")], env, class_of(fn), local_to=fn.owner)
            r = object.__new__(rerun.Rerun)
            r.lines, r.mode, r.opened, r.hand, r.hand_used = [], "outside", [], {}, []
            try:
                r.blocked(rerun.Blocked(fn, None, [run], [env], [rerun.Arm(ARM, ["failed"], ["job.jobStatus not in ['failed', 'cancelled']"])]))
                got[name] = None
            except rerun.Stop as stop:
                got[name] = [type(stop).__name__, str(stop)]
        # a fork: an excluded branch is neither an end nor a stop; every branch excluded is no answer
        class Out:
            pass
        class In:
            pass
        def out(q):
            raise rerun.Excluded("not it")
        def done(q):
            f.ends.append((f.hop, "it"))
        f = object.__new__(rerun.Rerun)
        f.lines, f.hand, f.hand_used, f.hop, f.ends, f.branches = [], {}, [], "S1", [], []
        f.touched, f.row_facts, f.question, f.steps = [], {}, None, {Out: out, In: done}
        f.fork(rerun.Fork([Out(), In()], ["a", "b"]))
        kinds = [b["kind"] for b in f.branches]
        try:
            f.fork(rerun.Fork([Out(), Out()], ["a", "b"]))
            every = None
        except rerun.Stop as stop:
            every = [type(stop).__name__, str(stop)]
        print(json.dumps([must, got, kinds, every]))
    """
    code = code.replace("ARM", str(arm)).replace("RET", str(ret))
    must, got, kinds, every = json.loads(_ok(_run(tmp_path, code)).splitlines()[-1])
    assert start == ret + 1
    # nothing prints after a return: the "start" after the if is the other way's
    assert must == [[], []]
    assert got["ended"] == ["Excluded", f"S3: every arm of run runs only in an except handler (at {handler}), and every pass "
                                        f"printed row {end} with nothing after it in the try that may raise"]
    # the try's last row not printed: it may have raised, and S3 asks a guard of the arm as before
    assert got["cut"] == ["Stop", f"S3: a guard the passes made false against every arm (none shown against arm {arm} (in an except handler))"]
    assert kinds == ["excluded", "end"]
    assert every == ["Stop", "S1: a branch that was not excluded (each of the 2 was)"]


def test_a_guard_over_a_column_the_rows_file_holds_is_read_in_it_by_the_ways_and_the_direction_of_their_changes(tmp_path):
    corpus = _corpus(tmp_path)
    (corpus / "pandajedi" / "base.py").write_text(textwrap.dedent("""\
        class Base:
            def extract(self, m):
                if 'k' in m:
                    self.x = m['k']
                if m['v'] == 3:
                    self.y = 1

            def prepro(self, m):
                if 'pre' not in m:
                    return None
                for k, v in list(m.items()):
                    m[k] = v
        """))
    (corpus / "pandajedi" / "anal.py").write_text(textwrap.dedent("""\
        class Anal:
            def extract(self, m):
                self.log.debug("anal only")
                m['k'] = 1
                Base.extract(self, m)
        """))
    (corpus / "pandajedi" / "gen.py").write_text(textwrap.dedent("""\
        class Gen:
            def extract(self, m):
                del m['k']
                Base.extract(self, m)
        """))
    out = _ok(
        _run(
            tmp_path,
            """
            import json
            from types import SimpleNamespace as N
            from bamboo.querier import heldcol, mapdump, render as T

            B, A, G, RUN = "pandajedi/base.py", "pandajedi/anal.py", "pandajedi/gen.py", "pandajedi/run.py"

            def change(at, kind, key, via=()):
                return N(at=at, kind=kind, key=key, via=list(via))

            def rows(*route, changes=()):
                return N(kind="rows", name="panda:x@9", site_owner="", column="jedi_taskparams.taskparams", steps=[],
                         route=list(route), changes=list(changes), after=[], after_stops=[])

            def param(owner, route, changes=(), after=(), stops=()):
                return N(kind="parameter", name="m", site_owner=owner, column="", steps=[], route=list(route),
                         changes=list(changes), after=list(after), after_stops=[N(kind=k) for k in stops])

            # the '*' store runs only past prepro's early return, read by prepro's own GuardFrom
            star = change(f"{B}:12", "store", "*")
            guard = lambda line, test, after: N(owner=f"{B}::extract", test_line=line, test=test, name="m", arms=[line + 1],
                                                sources=[param(f"{B}::extract", [], after=after, stops=["no_meeting"])])
            mapdump._FRAGMENT[:] = [N(
                guard_froms=[guard(3, "'k' in m", [star]), guard(5, "m['v'] == 3", [star]),
                             N(owner=f"{B}::prepro", test_line=9, test="'pre' not in m", name="m", arms=[10],
                               sources=[param(f"{B}::prepro", [], after=[star])])],
                parameter_froms=[
                    # straight from run.py, through Anal's extract (a store of k before), through Gen's (a delete)
                    N(owner=f"{B}::extract", parameter="m", stops=[N(kind="raises")], sources=[
                        rows(f"{RUN}:20"), param(f"{A}::extract", [f"{A}:5"], changes=[change(f"{A}:4", "store", "'k'")]),
                        param(f"{G}::extract", [f"{G}:4"], changes=[change(f"{G}:3", "delete", "'k'")])]),
                    N(owner=f"{A}::extract", parameter="m", stops=[], sources=[rows(f"{RUN}:20")]),
                    N(owner=f"{G}::extract", parameter="m", stops=[], sources=[rows(f"{RUN}:20")]),
                    N(owner=f"{B}::prepro", parameter="m", stops=[], sources=[rows(f"{RUN}:30")]),
                ])]

            def read(guard, value, arm=4, shown=None):
                data = {"task_params": {"value": value, "as_of": "2026-10-29 14:00:00"}}
                if shown is not None:
                    # the run prints a line of Anal's extract alone
                    heldcol.R.functions = lambda rel, name: [1]
                    heldcol.R.function = lambda rel, name: (rel, name)
                    heldcol.R.seen = lambda fn, run, mode: N(hits={3: [(0, 9, "")]} if fn == (A, "extract") else {})
                got = heldcol.read(f"{B}::extract", guard, arm, data, run=shown, mode="positive")
                return None if got is None else [repr(got.value), [T.text(k) for k in got.keys], got.shown,
                                                 T.text(got.premise([guard]))]

            print(json.dumps({
                "held": read("'k' in m", {"k": 2}),
                "not held": read("not ('k' in m)", {}),
                "held, the store's test held too": read("'k' in m", {"k": 2, "pre": 1}),
                "held, shown": read("'k' in m", {"k": 2}, shown="run"),
                "a value": read("m['v'] == 3", {"v": 3}, arm=6),
                "no GuardFrom": read("'z' in m", {"z": 1}),
            }))
            """,
        )
    )
    got = json.loads(out.splitlines()[-1])
    gone = "store * at base.py:12 did not run: \"'pre' not in m\" at base.py:9 read True"
    premise = "read in the rows file's task_params as fetched at 2026-10-29 14:00:00{}: the column as the pass left it, no " \
              "later write of it read (KNOWN_ISSUES.md K49)"
    # the rows hold k: there at the read unless something may add it after or remove it before -- Gen's delete; the
    # '*' store is not counted where prepro's early return is taken, read by prepro's own GuardFrom
    assert got["held"] == ["UNKNOWN", [f"'k' in m True on run.py:20, anal.py:5 (held as fetched; {gone})",
                                       "'k' in m not read on gen.py:4 (may be changed by delete 'k' at gen.py:3)"],
                           "", "\"'k' in m\" " + premise.format("")]
    # the rows lack k: absent unless something may add it before or remove it after -- Anal's store
    assert got["not held"][:2] == ["UNKNOWN", ["'k' in m False on run.py:20, gen.py:4 (not held as fetched)",
                                               "'k' in m not read on anal.py:5 (may be changed by store 'k' at anal.py:4)"]]
    # where the early return is not taken, the '*' store may add k after the read
    assert got["held, the store's test held too"][:2] == ["UNKNOWN", [
        "'k' in m not read on run.py:20, anal.py:5 (may be changed by store * at base.py:12)",
        "'k' in m not read on gen.py:4 (may be changed by store * at base.py:12, delete 'k' at gen.py:3)"]]
    # the run shows Anal's route by a line only its own function prints: read on it alone
    assert got["held, shown"] == ["True", [f"'k' in m True on anal.py:5 (held as fetched; {gone})"], "anal.py:5",
                                  "\"'k' in m\" " + premise.format(", on anal.py:5, the route its run shows")]
    # a value is read as fetched where nothing changes it on the way
    assert got["a value"][:2] == ["True", [f"m['v'] 3 on run.py:20, anal.py:5, gen.py:4 (held as fetched; {gone})"]]
    # a test over no name whose ways reach a held column is not this reading's
    assert got["no GuardFrom"] is None


def test_another_functions_print_of_the_questions_column_answers_a_statement_only_where_the_map_holds_one_object(tmp_path):
    # KNOWN_ISSUES.md K63: MJE-7394762518's killJob packs a JobSpec of its own (997-998); the adder printed its own
    # object's 'failed' at 444 after killJob ran, so that print of the question's column is not killJob's job's.  The
    # adder's self.job is handed to updateJobs's jobs at adder_gen.py:472, and updateJobs hands archiveJob its job: one
    # object.  A column no write changes in a job's life (prodSourceLabel) is read whichever object printed it.
    out = _ok(
        _run(
            tmp_path,
            """
            import json
            from types import SimpleNamespace as N
            from bamboo.querier import evaluate as E, mapdump

            ADD, TB, JC = ("pandaserver/dataservice/adder_gen.py", "pandaserver/taskbuffer/TaskBuffer.py",
                           "pandaserver/taskbuffer/db_proxy_mods/job_complex_module.py")
            src = lambda kind, name, route=(), site_owner="": N(kind=kind, name=name, route=list(route), site_owner=site_owner)
            mapdump._FRAGMENT[:] = [N(
                guard_froms=[N(owner=f"{TB}::updateJobs", test_line=639, test="job.jobStatus == 'failed'", name="job",
                               arms=[654], sources=[src("parameter", "jobs", site_owner=f"{TB}::updateJobs")])],
                parameter_froms=[
                    N(owner=f"{TB}::updateJobs", parameter="jobs", sources=[src("self_field", "self.job", [f"{ADD}:472"])]),
                    N(owner=f"{JC}::archiveJob", parameter="job", sources=[src("parameter", "jobs", [f"{TB}:667"], f"{TB}::updateJobs")]),
                ],
            )]
            HOLES = {
                "adder": ("self.job.jobStatus", "failed", "hole of process_job_report:444 at a", f"{ADD}::process_job_report"),
                "archive": ("job.jobStatus", "failed", "hole of archiveJob:1268 at b", f"{JC}::archiveJob"),
                "label": ("job.prodSourceLabel", "test", "hole of archiveJob:1268 at b", f"{JC}::archiveJob"),
                "jumbo": ("EventServiceUtils.isJumboJob(self.job)", "False", "hole of parse_job_output_report:873 at a",
                          f"{ADD}::parse_job_output_report"),
                "status": ("self.job_status", "failed", "hole of process_job_report:444 at a", f"{ADD}::process_job_report"),
            }

            def env(*names, table=None, subject="<job>.jobStatus"):
                e = E.Env(rows={"job": "job", "self.job": "job"}, call=None, subject=subject)  # a call is looked up by its text
                e.table["job"] = dict(table or {})
                for n in names:
                    text, value, source, owner = HOLES[n]
                    e.put_hole(text, value, source, [owner])
                return e

            def read(test, names, owner=None, line=0, **kw):
                e = env(*names, **kw)
                at = e.reading(owner, line) if owner else e
                return [repr(E.evaluate(test, at)), at.used]

            def shared():
                e = env("adder", "label")
                at = e.reading(f"{JC}::killJob", 1059)
                E.evaluate("job.jobStatus != 'failed' and job.prodSourceLabel == 'test'", at)
                e.used.update(at.shared())
                return [sorted(at.used), sorted(e.used)]

            KILL, UPD = (f"{JC}::killJob", 1059), (f"{TB}::updateJobs", 654)
            print(json.dumps({
                "killJob": read("job.jobStatus != 'failed'", ["adder"], *KILL),
                "updateJobs": read("job.jobStatus == 'failed'", ["adder"], *UPD),
                "the other way": read("self.job.jobStatus == 'failed'", ["archive"], f"{ADD}::process_job_report", 466),
                "a loop and its callee": read("job.jobStatus == 'failed'", ["archive"], *UPD),
                "the same function": read("self.job.jobStatus == 'failed'", ["adder"], f"{ADD}::process_job_report", 450),
                "the rows file under it": read("job.jobStatus == 'holding'", ["adder"], *KILL,
                                               table={"jobStatus": E.Value("holding", "the rows file")}),
                "not the question's column": read("job.prodSourceLabel == 'test' and not EventServiceUtils.isJumboJob(job)",
                                                  ["label", "jumbo"], *KILL),
                "not about a row": read("self.job_status == 'failed'", ["status"], f"{ADD}::handle_failed_job", 374),
                "no reader": read("job.jobStatus != 'failed'", ["adder"]),
                "no question": read("job.jobStatus != 'failed'", ["adder"], *KILL, subject=""),
                "knows": [env("adder").reading(*KILL).knows("job.jobStatus"), env("adder").reading(*UPD).knows("job.jobStatus")],
                "shared": shared(),
            }))
            """,
        )
    )
    got = json.loads(out.splitlines()[-1])
    adder, archive = "hole of process_job_report:444 at a", "hole of archiveJob:1268 at b"
    # killJob's job is not the adder's object by the map's records: the print of the question's column is not read
    assert got["killJob"] == ["UNKNOWN", {"job.jobStatus": f"unknown: {adder} is another object's"}]
    # updateJobs's job is: its GuardFrom's parameter jobs is handed the adder's self.job at adder_gen.py:472
    assert got["updateJobs"] == ["True", {"job.jobStatus": adder}]
    # the other way round: archiveJob's printed job is handed, through updateJobs's jobs, the adder's self.job
    assert got["the other way"] == ["True", {"self.job.jobStatus": archive}]
    # updateJobs loops over its jobs and hands archiveJob the job: both ways reach updateJobs's jobs
    assert got["a loop and its callee"] == ["True", {"job.jobStatus": archive}]
    # a hole the reader's own function printed is read as before
    assert got["the same function"] == ["True", {"self.job.jobStatus": adder}]
    # a print not read leaves the rows file's value to be read
    assert got["the rows file under it"] == ["True", {"job.jobStatus": "the rows file"}]
    # another column, and a call over the row, are read whichever object printed them
    assert got["not the question's column"] == ["True", {"job.prodSourceLabel": archive,
                                                         "EventServiceUtils.isJumboJob(job)": "hole of parse_job_output_report:873 at a"}]
    # a hole not about a row, a reading of no statement, and a reading for no question are as before
    assert got["not about a row"] == ["True", {"self.job_status": adder}]
    assert got["no reader"] == ["False", {"job.jobStatus": adder}]
    assert got["no question"] == ["False", {"job.jobStatus": adder}]
    assert got["knows"] == [False, True]
    # what a reading did not read stays its own: the env it was read from keeps only what it read
    assert got["shared"] == [["job.jobStatus", "job.prodSourceLabel"], ["job.prodSourceLabel"]]


def test_an_arms_guard_reads_as_the_row_only_the_names_the_maps_records_bind_to_what_the_arm_writes(tmp_path):
    # KNOWN_ISSUES.md K64: killEventServiceConsumers 818 writes dJob, a consumer it packs, while its guard reads job, the
    # parameter it was handed; the rows file spells job as the row, the records do not bind it.  setupper's loop
    # `for job in self.jobs + [None]` binds job once, so its sentinel test reads the row the arm wrote.
    out = _ok(
        _run(
            tmp_path,
            """
            import json, os
            from pathlib import Path
            from types import SimpleNamespace as N
            from bamboo.querier import evaluate as E, mapdump, row_names

            Path(os.environ["PANDA_PIN"], "pandaserver", "tb.py").write_text('''class T:
                def kill(self, job, pandaIDs):
                    for pandaID in pandaIDs:
                        if pandaID == job.PandaID:
                            continue
                        dJob = JobSpec()
                        if dJob.jobsetID != job.jobsetID:
                            continue
                        dJob.jobStatus = "failed"

                def setup(self, jobs):
                    for job in jobs + [None]:
                        if job is None:
                            break
                        job.jobStatus = "assigned"

                def two(self, jobs, other):
                    job = other
                    if job.jobStatus == "x":
                        job = jobs[0]
                        job.jobStatus = "y"

                def refine(self, files):
                    spec = self.taskSpec
                    if spec.status == "x" and self.taskSpec.splitRule:
                        for file in self.taskSpec.Files:
                            if file.type == "input":
                                self.taskSpec.status = "z"

                def objs(self):
                    a = JobSpec()
                    b = JobSpec()
                    c = a
                    if b.x and c.y:
                        a.jobStatus = "z"
            ''')
            TB = "pandaserver/tb.py"
            def src(kind, name, site_owner="", site_line=None, line=None, steps=(), through=(), column=""):
                return N(kind=kind, name=name, site_owner=site_owner, site_line=site_line, line=line, steps=list(steps),
                         through=list(through), column=column, route=[])
            def g(func, test_line, name, arms, *sources):
                return N(owner=f"{TB}::{func}", test_line=test_line, test="", name=name, arms=arms, sources=list(sources))
            def w(func, line, subject, expr, *sources):
                return N(owner=f"{TB}::{func}", line=line, subject=subject, role="object", expression=expr, sources=list(sources))
            par = lambda func, name, *steps: src("parameter", name, f"{TB}::{func}", 2, steps=steps)
            SEL = src("rows", "panda:select@40", f"{TB}::sql", 40)
            mapdump._FRAGMENT[:] = [N(
                write_froms=[
                    w("kill", 9, "JobSpec.jobStatus", "dJob", src("object", "JobSpec", line=6)),
                    w("setup", 15, "JobSpec.jobStatus", "job", par("setup", "jobs", "[*]")),
                    w("two", 21, "JobSpec.jobStatus", "job", par("two", "jobs", "[0]")),
                    w("refine", 28, "JediTaskSpec.status", "self.taskSpec", src("self_field", "self.taskSpec", line=28)),
                    # the map gives an object's way the class it builds as site_owner (SFF), with no site_line
                    w("objs", 35, "JobSpec.jobStatus", "a", src("object", "JobSpec", "pandaserver/JobSpec.py::JobSpec", line=31)),
                ],
                guard_froms=[
                    g("kill", 4, "job", [9], par("kill", "job")),
                    g("kill", 7, "dJob", [9], src("object", "JobSpec", line=6)),
                    g("kill", 7, "job", [9], par("kill", "job")),
                    g("setup", 13, "job", [15], par("setup", "jobs", "[*]"), src("constant", "None", line=12)),
                    g("two", 19, "job", [21], par("two", "other")),
                    g("refine", 25, "spec", [28], src("self_field", "self.taskSpec", line=24)),
                    g("refine", 27, "file", [28], src("self_field", "self.taskSpec", line=26, steps=[".Files", "[*]"])),
                    g("refine", 25, "jobs", [28], src("self_field", "self.jobs", line=24, steps=["[*]"])),
                    g("sql", 41, "job", [43], SEL),
                    g("sql", 41, "status", [43], src("rows", "panda:select@40", f"{TB}::sql", 40, column="jobsDefined4.jobStatus")),
                    g("sql2", 51, "job", [53], SEL),
                    g("sql3", 61, "job", [63], par("sql3", "job")),
                    g("objs", 34, "b", [35], src("object", "JobSpec", "pandaserver/JobSpec.py::JobSpec", line=32)),
                    g("objs", 34, "c", [35], src("object", "JobSpec", "pandaserver/JobSpec.py::JobSpec", line=31)),
                ],
                key_froms=[
                    N(owner=f"{TB}::sql", writes={"JobSpec.jobStatus": [43]},
                      sources=[src("rows", "panda:select@40", f"{TB}::sql", 40, column="jobsDefined4.PandaID")]),
                    N(owner=f"{TB}::sql2", writes={"JobSpec.jobStatus": [53]},
                      sources=[src("rows", "panda:select@50", f"{TB}::sql2", 50), src("parameter", "pandaID", line=52)]),
                    # archiveJob 1851's shape: its job parameter, and two selects its callers filled it from
                    N(owner=f"{TB}::sql3", writes={"JobSpec.jobStatus": [63]},
                      sources=[src("parameter", "job", line=62), src("rows", "panda:a@10", f"{TB}::caller_a", 10),
                               src("rows", "panda:b@20", f"{TB}::caller_b", 20)]),
                ],
            )]
            names = lambda func, line, subject="JobSpec.jobStatus", entity="job": row_names.names(
                f"{TB}::{func}", line, subject, entity, {"Files": "file"})

            def printers():
                # K63's gate reads a printing function's names by the rows file's spellings, not by the arm's
                from bamboo.querier import same_object
                seen = []
                same_object.same = lambda reader, line, names, printer, printed: seen.append([list(names), list(printed)]) or True
                e = E.Env(rows={"job": "job", "self.job": "job"}, call=None, subject="<job>.jobStatus",
                          written="JobSpec.jobStatus")
                e.put_hole("job.jobStatus", "failed", "hole of archive:1268 at a", [f"{TB}::archive"])
                at = e.reading(f"{TB}::kill", 9, arm=(f"{TB}::kill", 9))
                return [repr(E.evaluate("dJob.jobStatus == 'failed'", at)), seen]

            def read(test, func, line, values):
                e = E.Env(rows={"job": "job", "self.job": "job", "file": "file"}, call=None, subject="<job>.jobStatus",
                          written="JobSpec.jobStatus", elements={"Files": "file"})
                e.table["job"] = {k: E.Value(v, "the rows file") for k, v in values.items()}
                at = e.reading(f"{TB}::{func}", line, arm=(f"{TB}::{func}", line))
                return [repr(E.evaluate(test, at)), sorted(at.rows.items())]

            print(json.dumps({
                "kill": names("kill", 9),
                "setup": names("setup", 15),
                "two": names("two", 21),
                "refine": names("refine", 28, "JediTaskSpec.status", "task"),
                "sql": names("sql", 43),
                "sql2": names("sql2", 53),
                "sql3": names("sql3", 63),
                "nothing written": names("plugin", 92),
                "objects": names("objs", 35),
                "read kill": read("job.PandaID == 7", "kill", 9, {"PandaID": 7}),
                "read setup": read("job.jobStatus == 'defined'", "setup", 15, {"jobStatus": "defined"}),
                "read nothing written": read("self.job.jobStatus == 'defined'", "plugin", 92, {"jobStatus": "defined"}),
                "printer's names": printers(),
            }))
            """,
        )
    )
    got = json.loads(out.splitlines()[-1])
    # the written expression is the row; job, the parameter killEventServiceConsumers was handed, is another job
    assert got["kill"] == {"dJob": "job"}
    # one binding of job in setup: the sentinel test reads the row the arm wrote
    assert got["setup"] == {"job": "job"}
    # job at the test is `other`, at the arm `jobs[0]`: not the object the arm writes, so not the row
    assert got["two"] == {}
    # spec ends where self.taskSpec does; file is an element of its Files; jobs, an element of another list, is not
    assert got["refine"] == {"file": "file", "self.taskSpec": "task", "spec": "task"}
    # an SQL arm: the name holding the whole row the one key is read off; a column of it is not the row
    assert got["sql"] == {"job": "job"}
    # a key with two sources in its own function binds nothing
    assert got["sql2"] == {}
    # a parameter of the statement's function, its callers' selects aside: the parameter's name holds the row
    assert got["sql3"] == {"job": "job"}
    # an arm the map records nothing written for (a call on another object): no record to narrow by
    assert got["nothing written"] is None
    # objects of one class are told apart by the line they are built at, though the map names the class on both (SFF):
    # c holds a, the object the arm writes; b is another JobSpec
    assert got["objects"] == {"a": "job", "c": "job"}
    assert got["read nothing written"] == ["True", [["file", "file"], ["job", "job"], ["self.job", "job"]]]
    # the print's names are read as the printing function spells them
    assert got["printer's names"] == ["True", [[["dJob"], ["job"]]]]
    # the rows file's job is not read for killEventServiceConsumers' job: the row's PandaID does not decide its guard
    assert got["read kill"] == ["UNKNOWN", [["dJob", "job"]]]
    # setup's job is the row: its jobStatus is the row's
    assert got["read setup"] == ["True", [["job", "job"]]]


# updateUnmergedJobs, cut down: f writes umJob, an unmerged job, and prints its key in the same pass; g prints it after
# the loop, the last element's; h is f where the map records nothing the arm writes
_UNMERGED = """
def f(self, job, ids):
    self.logger.debug("f updates the unmerged jobs of the merge job")
    for tmpPandaID in ids:
        umJob = JobSpec()
        umJob.jobStatus = job.jobStatus
        self.logger.debug(f"f updated the unmerged PandaID={umJob.PandaID}")
    self.logger.debug("f finished updating the unmerged jobs")


def g(self, job, ids):
    self.logger.debug("g updates the unmerged jobs of the merge job")
    for tmpPandaID in ids:
        umJob = JobSpec()
        umJob.jobStatus = job.jobStatus
    self.logger.debug(f"g updated the unmerged PandaID={umJob.PandaID}")
    self.logger.debug("g finished updating the unmerged jobs")


def h(self, job, ids):
    self.logger.debug("h updates the unmerged jobs of the merge job")
    for tmpPandaID in ids:
        umJob = JobSpec()
        umJob.jobStatus = job.jobStatus
        self.logger.debug(f"h updated the unmerged PandaID={umJob.PandaID}")
    self.logger.debug("h finished updating the unmerged jobs")


def k(self, job, ids):
    self.logger.debug("k updates the unmerged jobs of the merge job")
    for tmpPandaID in ids:
        umJob = JobSpec()
        umJob.jobStatus = job.jobStatus
        self.cur.execute(sql)
        self.logger.debug(f"k updated the unmerged PandaID={umJob.PandaID}")
    self.logger.debug("k finished updating the unmerged jobs")
"""


def test_a_row_printing_the_key_of_what_the_arm_writes_shows_another_key_in_every_record_printed_for_other_objects(tmp_path):
    # KNOWN_ISSUES.md K64: under the merge job's mark, updateUnmergedJobs 1252 printed the unmerged jobs' keys only, so
    # 1236, which writes umJob, did not write the merge job (MJE-7398264739)
    (_corpus(tmp_path) / "pandaserver" / "a.py").write_text(_UNMERGED)
    stamp = "2026-01-01 00:00:0{},000 panda.log.X: DEBUG    <PandaID=1> {}\n"
    def log(func, *keys, after=False):
        lines = [f"{func} updates the unmerged jobs of the merge job"]
        lines += [f"{func} updated the unmerged PandaID={k}" for k in keys]
        lines += [f"{func} finished updating the unmerged jobs"]
        return "".join(stamp.format(i, text) for i, text in enumerate(lines))
    for name, text in {"others": log("f", 5, 6), "mine": log("f", 5, 1), "after": log("g", 6), "none": log("h", 5, 6),
                       "between": log("k", 5, 6)}.items():
        (tmp_path / f"{name}.log").write_text(text)
    out = _ok(
        _run(
            tmp_path,
            """
            import json
            from types import SimpleNamespace as N
            from bamboo.querier import evaluate as E, mapdump
            from bamboo.querier import rows as R
            from bamboo.querier.records import read_log, runs_for
            from bamboo.querier.which_ran import Candidate, judge

            R.ROWS_FROM = "walk"
            A = "pandaserver/a.py"
            def w(func, line):
                return N(owner=f"{A}::{func}", line=line, subject="JobSpec.jobStatus", role="object", expression="umJob",
                         sources=[N(kind="object", name="JobSpec", site_owner="", site_line=None, line=line - 1, steps=[],
                                    through=[], column="", route=[])])
            mapdump._FRAGMENT[:] = [N(write_froms=[w("f", 6), w("g", 15), w("k", 33)], guard_froms=[], key_froms=[],
                                      function_prints=[], handler_marks=[])]
            got = {}
            for name, func, line in (("others", "f", 6), ("mine", "f", 6), ("after", "g", 15), ("none", "h", 24),
                                     ("between", "k", 33)):
                env = E.Env(rows={"job": "job", "self.job": "job"}, call=None, subject="<job>.jobStatus",
                            written="JobSpec.jobStatus", keys=frozenset({"PandaID"}))
                env.table["job"] = {"PandaID": E.Value(1, "the rows file")}
                run, = runs_for(read_log(f"{name}.log"), "1")
                c = Candidate(A, func, [line], [run.log.name], run=run)
                v, = judge([c], {run.log.name: [run]}, question_run=run, env=env).verdicts
                got[name] = [v.kind, v.reasons]
            print(json.dumps(got))
            """,
        )
    )
    got = json.loads(out.splitlines()[-1])
    # every record of 7 holds another unmerged job's key: the pass wrote other objects, not the row
    assert got["others"] == ["excluded", ["rule 4: must row(s) [7] not in the run",
                                          "row 7 printed only for other values of umJob.PandaID (2 time(s)), not '1'"]]
    # one record holds the row's key: a pass may have written it, and 7 proves the arm
    assert got["mine"] == ["positive", ["row(s) [7] print only on a path through it"]], got["mine"]
    # a print after the loop shows the last element's key, whichever pass wrote the row
    assert not any("other values" in r for r in got["after"][1]), got["after"]
    # the map records nothing h's arm writes: umJob is not bound to the row
    assert got["none"][0] != "excluded" and not any("other values" in r for r in got["none"][1]), got["none"]
    # updateUnmergedJobs' own shape: a call between the arm and 35 may leave, so 35 is no must row and rule 4 does not
    # read it; printed for other objects only, it proves nothing either, in no proof (rows around it, printed.py's)
    assert got["between"][0] == "kept", got["between"]


_PRINTED_BEFORE = """
class Proxy:
    def __init__(self):
        self.logger = logger

    def kill(self, job, code, user, hard, late):
        self.logger.debug(f"code={code} user={user} hard={hard}")
        if isinstance(code, int):
            code = str(code)
        user = self.owner_of(job)
        for table in ("a", "b"):
            if code in ["2", "4"]:
                job.jobStatus = "closed"
            elif code == "51":
                job.jobStatus = "closed"
        if hard:
            job.jobStatus = "failed"
        if user == "pilot":
            job.jobStatus = "closed"
        if late:
            job.jobStatus = "cancelled"
        self.logger.debug(f"done late={late}")

    def command(self, job, cmd):
        if cmd in ["kill", "finish"]:
            self.logger.info(f"executing {cmd}")
            if cmd == "kill":
                job.jobStatus = "aborting"
        else:
            self.logger.info(f"executing {cmd}")
            job.jobStatus = "toretry"

    def flagged(self, job, code, flag):
        self.logger.debug(f"code={code} flag={flag}")
        if flag:
            code = str(code)
        if code == 2:
            job.jobStatus = "closed"

    def listed(self, job, items):
        self.logger.debug(f"items={items}")
        items.append(1)
        if items != []:
            job.jobStatus = "closed"
"""


def test_a_local_printed_before_an_arm_on_every_path_and_not_bound_again_is_read_by_the_arms_guards(tmp_path):
    # killJob prints ``code={code}`` at 840 and binds ``code = str(code)`` at 853 under ``isinstance(code, int)``:
    # printed "2", an int takes 853's way and a text the other, so code is "2" at every arm of the chain below, ``code
    # in ['2', '4']`` holds at 1028 and the arms after it are excluded (MJE-7383377087).  What reaches the arm from the
    # print is the map's reading (pathcond.held_since)
    (_corpus(tmp_path) / "pandaserver" / "p.py").write_text(_PRINTED_BEFORE)
    lines = _PRINTED_BEFORE.splitlines()
    at = {name: lines.index(text) + 1 for name, text in {
        "expire": '                job.jobStatus = "closed"', "hard": '            job.jobStatus = "failed"',
        "late": '            job.jobStatus = "cancelled"', "str": "            code = str(code)"}.items()}
    at["jedi"] = at["expire"] + 2
    at["pilot"] = lines.index('        if user == "pilot":') + 2
    kill = lines.index('                job.jobStatus = "aborting"') + 1
    flagged, listed = len(lines) - 6, len(lines)
    stamp = "2026-01-01 00:00:0{},000 panda.log.X: DEBUG    <PandaID=1> {}\n"
    for name, texts in {"one": ["code=2 user=pilot hard=False", "done late=False"],
                        "two": ["code=2 user=pilot hard=False", "code=51 user=pilot hard=False", "done late=False",
                                "done late=False"],
                        "dup": ["executing retry"], "flag": ["code=2 flag=True"], "list": ["items=[]"]}.items():
        (tmp_path / f"{name}.log").write_text("".join(stamp.format(i, t) for i, t in enumerate(texts)))
    out = _ok(
        _run(
            tmp_path,
            """
            import json
            from bamboo.querier import evaluate as E
            from bamboo.querier import rows as R
            from bamboo.querier.records import read_log, runs_for
            from bamboo.querier.which_ran import Candidate, judge

            R.ROWS_FROM = "walk"
            got = {}
            for name in ("one", "two"):
                run, = runs_for(read_log(f"{name}.log"), "1")
                c = Candidate("pandaserver/p.py", "kill", ARMS, [run.log.name], run=run)
                verdicts = judge([c], {run.log.name: [run]}, question_run=run, env=E.Env()).verdicts
                got[name] = {v.line: [v.kind, v.reasons] for v in verdicts}
            run, = runs_for(read_log("dup.log"), "1")
            c = Candidate("pandaserver/p.py", "command", [KILL], [run.log.name], run=run)
            got["dup"] = {v.line: [v.kind, v.reasons] for v in judge([c], {run.log.name: [run]}, question_run=run, env=E.Env()).verdicts}
            run, = runs_for(read_log("flag.log"), "1")
            c = Candidate("pandaserver/p.py", "flagged", [FLAGGED], [run.log.name], run=run)
            got["flag"] = {v.line: [v.kind, v.reasons] for v in judge([c], {run.log.name: [run]}, question_run=run, env=E.Env()).verdicts}
            run, = runs_for(read_log("list.log"), "1")
            c = Candidate("pandaserver/p.py", "listed", [LISTED], [run.log.name], run=run)
            got["list"] = {v.line: [v.kind, v.reasons] for v in judge([c], {run.log.name: [run]}, question_run=run, env=E.Env()).verdicts}
            print(json.dumps(got))
            """.replace("ARMS", str(sorted(v for k, v in at.items() if k != "str"))).replace("KILL", str(kill))
            .replace("FLAGGED", str(flagged)).replace("LISTED", str(listed)),
        )
    )
    got = {name: {int(k): v for k, v in arms.items()} for name, arms in json.loads(out.splitlines()[-1]).items()}
    one, two = got["one"], got["two"]

    def step5(verdict: list) -> list[str]:
        return [r for r in verdict[1] if r.startswith("step 5")]

    # code is put as the text printed, a string: whichever type it had, the way it took makes it "2"
    assert not step5(one[at["expire"]]), one[at["expire"]]
    jedi = step5(one[at["jedi"]])
    assert one[at["jedi"]][0] == "excluded" and len(jedi) == 1, one[at["jedi"]]
    assert jedi[0].startswith("step 5: guard \"not (code in ['2', '4'])\" is false"), jedi
    assert f"through code = str(code) at {at['str']}" in jedi[0], jedi
    # a local bound nowhere after the print is put as the literal it reads as: "False" is False, not a string
    assert one[at["hard"]][0] == "excluded" and step5(one[at["hard"]])[0].startswith("step 5: guard 'hard' is false"), one[at["hard"]]
    # user is bound again after the print, and late is printed only after its arm: neither is read
    assert not step5(one[at["pilot"]]) and not step5(one[at["late"]]), (one[at["pilot"]], one[at["late"]])
    # two records of the print with two values of code: which one ran the arm is not known, so code is not read
    assert not step5(two[at["jedi"]]) and not step5(two[at["expire"]]), (two[at["jedi"]], two[at["expire"]])
    assert step5(two[at["hard"]]), two[at["hard"]]
    # "executing retry" is read alike by the print before the kill arm and by the one in the other arm: no record of
    # the first, so its cmd is not read (TaskCommando's runImpl 133 and 235)
    assert not step5(got["dup"][kill]), got["dup"][kill]
    # ``if flag: code = str(code)``: flag is not known where the way is chosen, so an int 2 may reach the arm as 2 and the
    # text as "2" -- not one value, so code is not read (read as "2", ``code == 2`` would exclude the arm)
    assert not step5(got["flag"][flagged]), got["flag"][flagged]
    # a list is changed in place by the append after the print (the map's changes.object_changes): "[]" is not read
    # (read, ``items != []`` would exclude the arm)
    assert not step5(got["list"][listed]), got["list"][listed]


_PASSES_OF = """
class Setupper:
    def __init__(self):
        self.logger = logger

    def run(self):
        self.logger.debug("running jumbo")
        self.setup_jumbo()
        self.logger.debug("end")

    def setup_jumbo(self):
        if len(self.jumbo) == 0:
            return
        self.logger.debug("start jumbo")
        self.status = "failed"


class Broker:
    def __init__(self):
        self.logger = logger

    def broke(self):
        self.logger.debug("brokerage done")
"""


def test_a_run_holding_none_of_a_functions_rows_is_its_pass_only_where_a_callers_row_after_its_call_shows_it_ran(tmp_path):
    # setupper's run prints "end" at 206 only after calling setup_jumbo_jobs at 202, which returned at 1967 printing
    # nothing: the run is its pass (MJ-assigned-stuck).  TaskBrokerThread's runImpl prints under the time token
    # TaskCheckerThread's does, and 085's post-processing stopped before calling doBasicPostProcess: no record says
    # the function ran, so the run is not its pass (MT-assigning-late, MT-prepared-late; KNOWN_ISSUES.md K69)
    (_corpus(tmp_path) / "pandaserver" / "p.py").write_text(_PASSES_OF)
    lines = _PASSES_OF.splitlines()
    call, end = lines.index("        self.setup_jumbo()") + 1, lines.index('        self.logger.debug("end")') + 1
    stamp = "2026-01-01 00:00:0{},000 panda.log.X: DEBUG    <PandaID=1> {}\n"
    for name, texts in {"called": ["running jumbo", "end"], "before": ["running jumbo"], "other": ["brokerage done"],
                        "own": ["running jumbo", "start jumbo", "end"], "shared": ["brokerage done", "start jumbo"]}.items():
        (tmp_path / f"{name}.log").write_text("".join(stamp.format(i, t) for i, t in enumerate(texts)))
    out = _ok(
        _run(
            tmp_path,
            """
            import json
            from bamboo.querier import rows as R
            from bamboo.querier.records import read_log, runs_for
            from bamboo.querier.which_ran import passes_of

            R.ROWS_FROM = "walk"
            # in shared.log the map reads "start jumbo" as Broker's too (TaskBroker's two runImpl both print "getting Impl")
            both = frozenset({"pandaserver/p.py::Setupper.setup_jumbo", "pandaserver/p.py::Broker.broke"})
            R.line_writers = lambda record, log: both if log.startswith("shared") and "start jumbo" in record.text else frozenset()
            fn = R.function("pandaserver/p.py", "setup_jumbo")
            got = {}
            for name in ("called", "before", "other", "own", "shared"):
                run, = runs_for(read_log(f"{name}.log"), "1")
                parts, how = passes_of(fn, run, "outside")
                got[name] = [len(parts), how]
            print(json.dumps(got))
            """,
        )
    )
    got = json.loads(out.splitlines()[-1])
    # the caller's "end" comes after its call on every way: the run is the callee's one pass, and says why
    assert got["called"][0] == 1 and f"its call shown by run:{end} after its call(s) at [{call}]" in got["called"][1], got
    # "running jumbo" is printed before the call: it does not say the call ran
    assert got["before"][0] == 0 and "not shown to be a pass of it" in got["before"][1], got
    # another class's rows under the same mark say nothing of the function
    assert got["other"][0] == 0 and "not shown to be a pass of it" in got["other"][1], got
    # a record of the function's own row: the run is read as before
    assert got["own"][0] == 1 and "no record" not in got["own"][1], got
    # a record another function's print can have written too says nothing of which ran
    assert got["shared"][0] == 0 and "not shown to be a pass of it" in got["shared"][1], got


def test_a_line_its_logger_writes_with_no_mark_is_read_from_its_first_word_and_is_a_run_of_its_own(tmp_path):
    # ContentsFeederThread's self.logger is the module's logger, not wrapped yet where the task's spec was not got:
    # "failed to get taskSpec for jediTaskID=..." (ContentsFeeder.py:157) is cut as the mark "failed" and the message
    # "to get taskSpec ...", which the row's pattern does not find; and the "failed ..." lines of every task and pass
    # of a machine's log were one run (Tr; KNOWN_ISSUES.md K70)
    (tmp_path / "panda-x.log").write_text(
        "2026-01-01 00:00:00,000 panda.log.X: DEBUG    failed to get taskSpec for jediTaskID=5\n"
        "2026-01-01 00:00:01,000 panda.log.X: DEBUG    <jediTaskID=5 gshare=G> start loop\n"
        "2026-01-01 00:00:02,000 panda.log.X: DEBUG    failed to get taskSpec for jediTaskID=6\n"
        "2026-01-01 00:00:03,000 panda.log.X: DEBUG    <jediTaskID=5 gshare=G> done\n"
        "2026-01-01 01:00:00,000 panda.log.X: DEBUG    failed to get taskSpec for jediTaskID=5\n"
        "2026-01-01 01:00:01,000 panda.log.X: DEBUG    failed with nothing for jediTaskID=5\n"
    )
    out = _ok(
        _run(
            tmp_path,
            """
            import ast, json, re
            from types import SimpleNamespace as N
            from bamboo.querier import mapdump
            from bamboo.querier import rows as R
            from bamboo.querier.records import head_word, read_log, runs_for

            bare = re.escape("failed to get taskSpec for jediTaskID=") + ".*?"
            marked = re.escape("start loop")
            owner = "p/f.py::Feeder.feed"
            mapdump._FRAGMENT[:] = [N(function_prints=[N(owner=owner, runs=[], statements=[
                N(log_files_basis="logger", runs=[], handed_runs=[], log_files=["panda-x.log"],
                  rows=[N(pattern=bare, prefix=None)]),
                N(log_files_basis="logger", runs=[], handed_runs=[], log_files=["panda-x.log"],
                  rows=[N(pattern=marked, prefix="<jediTaskID={} gshare={}>")]),
            ])])]
            runs = runs_for(read_log("panda-x.log"), "5")
            # as a recorded search reads them: on a machine, with ends of what was searched for each mark
            log = runs[0].log
            for x in log.records:
                x.machine = "m.cern.ch"
            log.edges = lambda machine, token, first, last: "its records lie outside what was searched on m"
            with_edges = [[r.alone, r.describe()] for r in runs]
            log.edges = None
            for x in log.records:
                x.machine = ""
            cut = runs[0].records[0]
            stmt = ast.parse("pass").body[0]
            row = R.Row("p/f.py", "feed", stmt, "DEBUG", 'f"failed to get taskSpec for jediTaskID={jediTaskID}"',
                        map_patterns=[bare])
            prefixed = R.Row("p/f.py", "feed", stmt, "DEBUG", 'f"failed to get taskSpec for jediTaskID={jediTaskID}"',
                             map_patterns=[bare], map_prefixes=["<jediTaskID={} gshare={}>"])
            print(json.dumps({
                "runs": [[r.token, [x.stamp[11:19] for x in r.records], r.alone, r.would_print("DEBUG")] for r in runs],
                "said": runs[0].describe(),
                "with_edges": with_edges,
                "cut": [cut.token, cut.message],
                "writers": sorted(R.line_writers(cut, "panda-x.log")),
                "matches": row.matches(cut, "outside")[0] > 0,
                "prefixed": prefixed.texts(cut),
                "heads": [head_word(p) for p in (bare, marked, ".*?" + re.escape(" x"), re.escape("a-b c"), "start")],
            }))
            """,
        )
    )
    got = json.loads(out.splitlines()[-1])
    # each line of the row is a run of its own, whose missing rows say nothing; the marked lines are one run as
    # before; another task's line is not the key's; a line whose first word is the row's but whose text is not stays
    # under that word
    assert got["runs"] == [
        ["failed", ["00:00:00"], True, False],
        ["<jediTaskID=5 gshare=G>", ["00:00:01", "00:00:03"], False, True],
        ["failed", ["01:00:00"], True, False],
        ["failed", ["01:00:01"], False, True],
    ], got
    assert "read alone: its logger writes no mark, so the token is the line's first word" in got["said"], got
    # a line read alone is all its run: what was searched for its first word cuts nothing of it, while a marked run
    # near an end of what was searched is read as cut there
    assert [[alone, "read as cut" in said] for alone, said in got["with_edges"]] == [
        [True, False], [False, True], [True, False], [False, True]], got
    assert all("read alone" in said for alone, said in got["with_edges"] if alone), got
    # the cut is the parser's still; the row and the map's writers read the line as written
    assert got["cut"] == ["failed", "to get taskSpec for jediTaskID=5"], got
    assert got["writers"] == ["p/f.py::Feeder.feed"], got
    assert got["matches"] is True, got
    # a row the map gives a prefix reads the message only
    assert got["prefixed"] == ["to get taskSpec for jediTaskID=5"], got
    assert got["heads"] == ["failed", "start", "", "a-b", ""], got


_ROW_SPEC = """\
class JobSpec:
    _attributes = ("PandaID", "jobStatus")

    def __init__(self):
        for attr in self._attributes:
            object.__setattr__(self, attr, None)

    def pack(self, values):
        for i in range(len(self._attributes)):
            object.__setattr__(self, self._attributes[i], values[i])

    @classmethod
    def columnNames(cls):
        return ",".join(cls._attributes)
"""

_ROW_PROXY = """\
from pandaserver.spec import JobSpec


class Proxy:
    def killJob(self, pandaID, code):
        sql2 = f"SELECT {JobSpec.columnNames()} FROM ATLAS_PANDA.jobsActive4 WHERE PandaID=:PandaID"
        varMap = {}
        varMap[":PandaID"] = pandaID
        self.cur.execute(sql2, varMap)
        res = self.cur.fetchall()
        if len(res) == 0:
            return
        job = JobSpec()
        job.pack(res[0])
        if job.jobStatus != "failed":
            if code == "2":
                job.jobStatus = "closed"
            if job.jobStatus != "closed":
                job.jobStatus = "cancelled"
"""


def test_a_guard_reading_the_questions_column_the_pass_filled_from_the_row_reads_the_statuslogs_state_before_it(tmp_path):
    # killJob reads job.jobStatus at 1017 after job.pack(res[0]) at 998, res the rows of 992's SELECT WHERE
    # PandaID=:PandaID: the value is the row's as the pass read it, the statuslog's state before the pass
    # (MJE-7383377087: 'merging'); at the guard after the stores the stores reach it too (H2 step 2 (iii);
    # KNOWN_ISSUES.md K71)
    corpus = _corpus(tmp_path)
    (corpus / "pandaserver" / "__init__.py").write_text("")
    (corpus / "pandaserver" / "spec.py").write_text(_ROW_SPEC)
    (corpus / "pandaserver" / "proxy.py").write_text(_ROW_PROXY)
    (tmp_path / "x.log").write_text(
        "2026-01-01 00:00:05,000 panda.log.X: DEBUG    < PandaID=1 > start\n"
        "2026-01-01 00:00:05,100 panda.log.X: DEBUG    < PandaID=1 > done\n"
    )
    out = _ok(
        _run(
            tmp_path,
            """
            import ast, json
            from bamboo.querier import evaluate as E
            from bamboo.querier import rows as R
            from bamboo.querier.records import Log, Record, Run, read_log, runs_for
            from bamboo.querier.which_ran import row_state, state_before

            R.ROWS_FROM = "walk"
            fn = R.function("pandaserver/proxy.py", "killJob")
            closed, cancelled = (next(s for s in ast.walk(fn.node) if isinstance(s, ast.stmt) and ast.unparse(s) == t)
                                 for t in ("job.jobStatus = 'closed'", "job.jobStatus = 'cancelled'"))
            run, = runs_for(read_log("x.log"), "1")

            def at(status, time, host):
                return {"status": status, "time": time[:19], "time_extended": time, "host": host}

            before = at("merging", "2026-01-01 00:00:00.000000", "m1")
            after = at("closed", "2026-01-01 00:00:30.000000", "m2")
            inside = at("holding", "2026-01-01 00:00:05.050000", "m2")

            def env(states, **known):
                e = E.Env(rows={"job": "job"}, subject="<job>.jobStatus", keys=frozenset({"PandaID"}), statuslog=states)
                for k, v in known.items():
                    e.put(k, v, "a print")
                return e

            got = {
                "closed": row_state(fn, closed, run, env([before, after])),
                "cancelled": row_state(fn, cancelled, run, env([before, after])),
                "inside": row_state(fn, closed, run, env([before, inside, after])),
                "known": row_state(fn, closed, run, env([before, after], **{"job.jobStatus": "running"})),
                "none_before": row_state(fn, closed, run, env([after])),
            }
            e = env([before, after])
            row_state(fn, closed, run, e)
            got["source"] = e.values[e.canonical("job.jobStatus")].source
            # five minutes before a pass: past a second of the server's clock, within the windows' slack on JEDI's
            # (fifteen minutes before it, the first row is past both)
            records = [Record(file="y.log", lineno=1, stamp="2026-01-01 00:15:00,000", logger="panda.log.Y",
                              level="DEBUG", token="<t>", message="m", machine="j1")]
            near = at("ready", "2026-01-01 00:10:00.000000", "m2")
            jedi = Run(Log(path="y.log", records=records, jedi=frozenset({"j1"})), "<t>", records, records)
            server = Run(Log(path="y.log", records=records), "<t>", records, records)
            got["jedi"] = state_before(jedi, [before, near])[1]
            got["server"] = state_before(server, [before, near])[0]["status"]
            print(json.dumps(got))
            """,
        )
    )
    got = json.loads(out.splitlines()[-1])
    # the guard's read is filled only from the key's row: the state before the pass
    assert got["closed"] == {"job.jobStatus": "merging"}, got
    assert got["source"] == "the statuslog row before the pass, 'merging' at 2026-01-01 00:00:00.000000", got
    # the second guard is reached by the store of 'closed' too: the text is not put for the arm
    assert got["cancelled"] == {}, got
    # another machine's row within the pass: the read may have come after it
    assert got["inside"] == {}, got
    # a value a print showed is kept; no row before the pass, nothing
    assert got["known"] == {} and got["none_before"] == {}, got
    assert got["jedi"] == "statuslog row(s) ['ready'] of another machine within the pass", got
    assert got["server"] == "ready", got


def test_where_h2_decides_nothing_each_arm_left_walks_on_and_those_not_excluded_are_the_answer(tmp_path):
    out = _ok(
        _run(
            tmp_path,
            """
            import json
            from types import SimpleNamespace as N
            from bamboo.querier import evaluate as E
            from bamboo.querier import rerun
            from bamboo.querier.which_ran import Candidate, Verdict

            rerun.R = N(function=lambda rel, func, lines: N(name=func, node=func), statement_at=lambda node, line: (node, line),
                        entries=lambda fn, run, mode: [0], must_rows=lambda node, site: (None, []), seen=lambda fn, run, mode: N(hits={}))
            guards = {("extract", 5): ["task.x"], ("gen", 9): ["task.y"], ("old", 3): ["task.z"], ("rec", 99): [], ("put", 7): []}
            rerun.pathcond = N(arm_condition=lambda site: guards[site], path_condition=lambda site: guards[site],
                               exits_above_at=lambda site: [])
            new, old = {"status": "running", "time": "2026-10-29 11:38:00"}, {"status": "scouting", "time": "2026-10-29 10:06:00"}
            # task.x is false at the newer row, true at the older; task.z false at both; task.y read at neither
            envs = {new["time"]: E.Env({"task.x": E.Value(False, "row"), "task.z": E.Value(False, "row")}),
                    old["time"]: E.Env({"task.x": E.Value(True, "row"), "task.z": E.Value(False, "row")})}
            readings = [N(file=f"pandajedi/{f}.py", owner=f"pandajedi/{f}.py::{func}") for f, func in (("r", "extract"), ("g", "gen"), ("o", "old"), ("c", "rec"), ("s", "put"))]
            unshown = rerun.Premise("no line of a.log in the records holds 7: its runs are not shown to name it", rerun.NOT_READ)
            kept = rerun.Premise("its run does not show the line ran (kept)", rerun.NOT_IN_DATA)

            def may(f, func, line, row, premise):
                branch = rerun.Wrote(readings, None, [], None, Verdict(Candidate(f"pandajedi/{f}.py", func, [line]), line, "unexamined", []),
                                     row=row, window=(row["time"], row["time"]))
                return rerun.Possible(f"{f}.py::{func}:{line}", row, False, [premise], branch=branch)

            a = [may("r", "extract", 5, new, unshown), may("r", "extract", 5, old, unshown)]
            b, c, d = may("g", "gen", 9, new, kept), may("o", "old", 3, new, unshown), may("c", "rec", 99, new, unshown)

            def walk(left):
                r = object.__new__(rerun.Rerun)
                r.lines, r.opened, r.ends, r.touched, r.row_facts, r.hop, r.hand, r.hand_used, r.result, r.mode = [], [], [], [], {}, "H2", {}, [], {}, "outside"
                r.data = {}  # no rows file: no held column (heldcol, K49)
                r.question, r.branches, r.end_code, r._walking = None, [], None, None
                r.base_env = lambda at=None: envs.get(at, E.Env({}))
                r._which_wrote = lambda *x: rerun.OneOf(left, 2)

                def step(q):
                    if q.wrote.chosen.line == 99:  # a walk that asks the logs for what the records do not hold
                        raise rerun.Stop("H4: the asks the records do not answer", "asks")
                    return r.wrote_by(q)

                r.steps = {rerun.WroteBy: step}
                q = r.asked(rerun.Asked("JediTaskSpec.transUses", "A", "7"))
                try:
                    while q is not None:
                        q = r.steps.get(type(q), getattr(r, "arms_left"))(q)
                    said = None
                except rerun.Stop as e:
                    said = ["STOP", str(e), e.code]
                # the row is a record (version 3); rendered, as the fork's label says it
                branches = [{k: e[k] for k in ("arm", "kind", "step", "code", "kinds")} | {"wrote": bool(e["result"])}
                            | {"row": rerun.T.text(e["row"]) if e["row"] is not None else None}
                            | {"text": rerun.T.text(e["record"]) if e["record"] else None}
                            for e in r.branches]
                return {"said": said, "ends": [f"END {said}" for _, said in r.ends], "code": r.end_code, "branches": branches,
                        "heads": [line for line in r.lines if line.startswith("====") or "its next row" in line],
                        "wrote": r.result.get("wrote") is not None, "possibilities": len(r.result.get("possibilities") or [])}

            # an arm with no guard: walked at its newest row, its older row named
            x = [may("s", "put", 7, new, unshown), may("s", "put", 7, old, unshown)]
            # an arm shown to write, at a row an excluded one was newer than
            run = N(log=N(name="a.log"), machine="m", token="< jediTaskID=7 >", records=[0], first="2026-10-29 10:06:00,000",
                    last="2026-10-29 10:06:01,000", describe=lambda: "the run", said=lambda: "the run")
            shown = rerun.Wrote(readings, run, [], E.Env({}), Verdict(Candidate("pandajedi/s.py", "put", [7]), 7, "positive"), row=old,
                                window=(old["time"], old["time"]))
            y = [c, rerun.Possible("s.py::put:7", old, True, [], shown, branch=shown)]
            print(json.dumps({"one of": walk(a + [b, c]), "one ended": walk(a + [c]), "one open": walk([b, c]), "excluded": walk([c]),
                              "asks": walk(a + [d]), "none": walk([]), "older": walk(x + [c]), "shown": walk(y)}))
            """,
        )
    )
    got = json.loads(out.splitlines()[-1])
    end_a = ("END JediTaskSpec.transUses='A' written last by extract:5 (no run shows the line ran; its guards hold in the rows' "
             "values), in the pass of the statuslog row 'scouting' at 2026-10-29 10:06:00")
    stop_b = ("STOP H3: a run that shows gen:9 ran, to read its guards in (it may have: its run does not show the line ran (kept); "
              "its conditions: 'task.y' not read)")
    excluded_c = "excluded H3: old:3's guard 'task.z' read false in the rows' values: it did not write at any row it may have written at"
    # an arm walks on at its newest row; a guard read false there sends it to the next; one ends, one stops, one is
    # excluded: the answer is the two not excluded, the data not deciding which (K38)
    one = got["one of"]
    assert one["said"] is None and one["code"] == "one-of"
    assert one["ends"] == ["END JediTaskSpec.transUses='A' written last by one of the 2 arm(s) left, which the data does not "
                           "decide: 1 r.py::extract:5: ended; 2 g.py::gen:9: open (not in the data, not read); the other 1 arm(s) "
                           "left excluded (3) (KNOWN_ISSUES.md K38)"]
    assert one["branches"] == [
        {"arm": "r.py::extract:5", "row": "'scouting' 2026-10-29 10:06:00", "kind": "end", "step": "H3", "text": end_a, "code": None,
         "kinds": ["not read"], "wrote": True},
        {"arm": "g.py::gen:9", "row": "'running' 2026-10-29 11:38:00", "kind": "stop", "step": "H3", "text": stop_b, "code": None,
         "kinds": ["not in the data", "not read"], "wrote": True},
        {"arm": "o.py::old:3", "row": "'running' 2026-10-29 11:38:00", "kind": "excluded", "step": "H3", "text": excluded_c,
         "code": None, "kinds": ["not read"], "wrote": True},
    ]
    assert one["heads"] == [
        "==== 1 of 3: r.py::extract:5 (may have at 'running' 2026-10-29 11:38:00, may have at 'scouting' 2026-10-29 10:06:00)",
        "extract:5's guard 'task.x' read false in the rows' values at 'running' 2026-10-29 11:38:00: it did not write there; its "
        "next row, 'scouting' 2026-10-29 10:06:00",
        "==== 1: reached the end",
        "==== 2 of 3: g.py::gen:9 (may have at 'running' 2026-10-29 11:38:00)", f"==== 2: {stop_b}",
        "==== 3 of 3: o.py::old:3 (may have at 'running' 2026-10-29 11:38:00)", f"==== 3: {excluded_c}",
    ]
    # what a branch wrote to result.json is its own; H2's possibilities stay the run's
    assert (one["wrote"], one["possibilities"]) == (False, 3)
    # one ended, the rest excluded: its END, with what it rests on, and its wrote
    assert got["one ended"]["ends"] == [end_a + "; it rests on: no line of a.log in the records holds 7: its runs are not shown to "
                                                "name it; the other 1 arm(s) left excluded (2)"]
    assert (got["one ended"]["code"], got["one ended"]["wrote"]) == ("one-of", True)
    # the one left shown to write, the newer excluded: decided, no code
    assert got["shown"]["ends"] == ["END JediTaskSpec.transUses='A' written last by put:7 (shown in its run; it has no guard), in the "
                                    "pass of the statuslog row 'scouting' at 2026-10-29 10:06:00: the run; the other 1 arm(s) left "
                                    "excluded (1)"]
    assert (got["shown"]["code"], got["shown"]["wrote"]) == (None, True)
    assert got["older"]["ends"] == [
        "END JediTaskSpec.transUses='A' written last by put:7 (no run shows the line ran; it has no guard), in the pass of the "
        "statuslog row 'running' at 2026-10-29 11:38:00; it rests on: no line of a.log in the records holds 7: its runs are not "
        "shown to name it; or at an older row it may have written at: 'scouting' 2026-10-29 10:06:00; the other 1 arm(s) left "
        "excluded (2)"]
    # one left that stops: the answer is still that arm, open
    assert got["one open"]["ends"] == ["END JediTaskSpec.transUses='A' written last by the one arm left, which the data does not "
                                       "decide: 1 g.py::gen:9: open (not in the data, not read); the other 1 arm(s) left excluded (2) "
                                       "(KNOWN_ISSUES.md K38)"]
    # none left: no arm could have
    assert got["excluded"]["said"] == ["STOP", "H2: an arm that could have written JediTaskSpec.transUses='A': each of the 1 left "
                                               f"is excluded ({excluded_c}; KNOWN_ISSUES.md K38)", "one-of"]
    assert got["excluded"]["ends"] == [] and got["excluded"]["code"] is None
    # an arm stopped on asks the records do not answer: the run stops on them, every arm walked first
    assert got["asks"]["said"] == ["STOP", "H2: 1 of the 2 arm(s) left stopped on what the records do not answer yet: 2 (STOP H4: "
                                           "the asks the records do not answer)", "asks"]
    assert got["asks"]["ends"] == [] and [b["kind"] for b in got["asks"]["branches"]] == ["end", "stop"]
    # H2 leaves no arm: its stop, as before
    assert got["none"]["said"] == ["STOP", "H2: a pass that wrote JediTaskSpec.transUses='A' at the 2 statuslog row(s) its writers' "
                                           "passes can write (none shows a writer's pass: KNOWN_ISSUES.md K38)", "one-of"]


def test_a_late_question_goes_on_with_the_one_arm_h2_left_and_every_end_rests_on_it(tmp_path):
    out = _ok(
        _run(
            tmp_path,
            """
            import json
            from types import SimpleNamespace as N
            from bamboo.querier import rerun
            from bamboo.querier.which_ran import Candidate, Verdict

            row = {"status": "finished", "time": "2026-10-29 13:47:28"}
            readings = [N(file="pandaserver/a.py", owner="pandaserver/a.py::fin"), N(file="pandaserver/b.py", owner="pandaserver/b.py::put")]
            run = N(log=N(name="a.log"), machine="m", token="7", first="2026-10-29 13:47:26,958", last="2026-10-29 13:47:28,324")

            def may(f, func, line, guard, on_run=True):
                branch = rerun.Wrote(readings, run if on_run else None, [], None, Verdict(Candidate(f"pandaserver/{f}.py", func, [line]), line, "positive", []),
                                     row=row, window=(row["time"], row["time"]))
                said = rerun.T.Said("its-condition-not-read", {"guard": guard})
                return rerun.Possible(f"{f}.py::{func}:{line}", row, False, [rerun.Premise(said, rerun.NOT_READ)], branch=branch)

            def walk(left):
                r = object.__new__(rerun.Rerun)
                r.lines, r.ends, r.hop, r.result, r.end_code = [], [], "H2", {}, None
                try:
                    got = r._the_arm_left("JobSpec.jobStatus", "finished", rerun.OneOf(left, 1))
                    said = [got.reading.owner, got.chosen.line, got.question is run]
                except rerun.Stop as e:
                    said = ["STOP", str(e), e.code]
                r.hop = "S2"
                r.end(rerun.T.Said("ends", {"ends": ["moved on at last"]}))
                return {"said": said, "lines": r.lines[:-1], "ends": [str(w) for _, w in r.ends], "code": r.end_code,
                        "rests": [rerun.T.to_json(s) for s in r.rests]}

            one = may("a", "fin", 304, "not (x.failed)")
            # the same arm called twice in the pass: one arm left
            twice = may("a", "fin", 304, "not (x.merging)")
            print(json.dumps({"one": walk([one, twice]), "two": walk([one, may("b", "put", 9, "y")]),
                              "no branch": walk([rerun.Possible("a.py::fin:304", row, False, [], branch=None)]),
                              "no run": walk([may("a", "fin", 304, "z", on_run=False)])}))
            """,
        )
    )
    got = json.loads(out.splitlines()[-1])
    arm = "the one arm H2 left moved it on, a.py::fin:304 (its condition 'not (x.failed)' not read; KNOWN_ISSUES.md K38)"
    # one arm left: the walk goes on with it, on the run H2 read it on, and every END after says what it rests on
    assert got["one"]["said"] == ["pandaserver/a.py::fin", 304, True]
    assert got["one"]["lines"] == [f"{arm}: the late question goes on with it"]
    assert got["one"]["ends"] == [f"END moved on at last; it rests on: {arm}"]
    assert got["one"]["code"] == "one-of" and got["one"]["rests"][0]["kind"] == "the-one-arm-left-moved-it-on"
    # two arms left: which moved it on is not walked for each -- H2's stop, as before, and an END rests on nothing
    assert got["two"]["said"] == ["STOP", "H2: one arm that wrote JobSpec.jobStatus='finished' last, of 2 left: a.py::fin:304 (may "
                                          "have at 'finished' 2026-10-29 13:47:28); b.py::put:9 (may have at 'finished' 2026-10-29 "
                                          "13:47:28) (KNOWN_ISSUES.md K38)", "one-of"]
    assert got["two"]["ends"] == ["END moved on at last"] and got["two"]["code"] is None and got["two"]["rests"] == []
    # an arm with no walk to go on with stops as before
    assert got["no branch"]["said"][0] == "STOP" and got["no branch"]["rests"] == []
    # one with no run goes on: L1 says it has no pass to read (K47)
    assert got["no run"]["said"] == ["pandaserver/a.py::fin", 304, False]


def test_a_condition_an_arm_left_rests_on_is_dropped_where_h3_reads_it_holding(tmp_path):
    out = _ok(
        _run(
            tmp_path,
            """
            import json
            from types import SimpleNamespace as N
            from bamboo.querier import evaluate as E
            from bamboo.querier import rerun
            from bamboo.querier.which_ran import Candidate, Verdict

            rerun.R = N(function=lambda rel, func, lines: N(name=func, node=func), statement_at=lambda node, line: (node, line),
                        entries=lambda fn, run, mode: [0], must_rows=lambda node, site: (None, []), seen=lambda fn, run, mode: N(hits={}))
            guards = {("extract", 5): ["task.x"], ("gen", 9): ["task.x", "task.y"]}
            rerun.pathcond = N(arm_condition=lambda site: guards[site], path_condition=lambda site: guards[site],
                               exits_above_at=lambda site: [])
            row = {"status": "defined", "time": "2026-10-29 07:42:23"}
            # task.x holds in the rows' values; task.y is read nowhere
            envs = {row["time"]: E.Env({"task.x": E.Value(True, "row")})}
            readings = [N(file=f"pandajedi/{f}.py", owner=f"pandajedi/{f}.py::{func}") for f, func in (("r", "extract"), ("g", "gen"))]

            def cond(g):
                return rerun.Premise(rerun.T.Said("its-condition-not-read", {"guard": g}), rerun.NOT_READ)

            between = rerun.Premise(rerun.T.Said("functions-between-not-read", {"n": 1}), rerun.NOT_READ)

            def may(f, func, line, premises):
                branch = rerun.Wrote(readings, None, [], None, Verdict(Candidate(f"pandajedi/{f}.py", func, [line]), line, "unexamined", []),
                                     row=row, window=(row["time"], row["time"]))
                return rerun.Possible(f"{f}.py::{func}:{line}", row, False, list(premises), branch=branch)

            def walk(left):
                r = object.__new__(rerun.Rerun)
                r.lines, r.opened, r.ends, r.touched, r.row_facts, r.hop, r.hand, r.hand_used, r.result, r.mode = [], [], [], [], {}, "H2", {}, [], {}, "outside"
                r.data = {}  # no rows file: no held column (heldcol, K49)
                r.question, r.branches, r.end_code, r._walking = None, [], None, None
                # another writer of the value H2 dropped by a guard: the arm left is not the only way to it (K66)
                r._dropped_by_guard = frozenset({"x.py::other:1"})
                r.base_env = lambda at=None: envs.get(at, E.Env({}))
                r._which_wrote = lambda *x: rerun.OneOf(left, 1)
                r.steps = {rerun.WroteBy: r.wrote_by}
                q = r.asked(rerun.Asked("JediTaskSpec.transUses", "A", "7"))
                while q is not None:
                    q = r.steps.get(type(q), getattr(r, "arms_left"))(q)
                return {"ends": [f"END {said}" for _, said in r.ends], "code": r.end_code,
                        "branches": [{"kind": e["kind"], "premises": [str(p) for p in e["premises"]], "kinds": e["kinds"]}
                                     for e in r.branches]}

            print(json.dumps({"read": walk([may("r", "extract", 5, [cond("task.x")])]),
                              "between": walk([may("r", "extract", 5, [cond("task.x"), between])]),
                              "one unread": walk([may("g", "gen", 9, [cond("task.x"), cond("task.y")])])}))
            """,
        )
    )
    got = json.loads(out.splitlines()[-1])
    end = ("END JediTaskSpec.transUses='A' written last by extract:5 (no run shows the line ran; its guards hold in the rows' "
           "values), in the pass of the statuslog row 'defined' at 2026-10-29 07:42:23")
    # the condition H2 left the arm resting on is read holding at H3: the END rests on that reading, not on it (K62)
    assert got["read"] == {"ends": [end], "code": None, "branches": [{"kind": "end", "premises": [], "kinds": []}]}
    # what H3 does not read stays: the functions between the call and the arm
    between = "the 1 function(s) between the call and it not read"
    assert got["between"]["ends"] == [f"{end}; it rests on: {between}"] and got["between"]["code"] == "one-of"
    assert got["between"]["branches"] == [{"kind": "end", "premises": [between], "kinds": ["not read"]}]
    # an arm H3 stops on keeps the condition it did not read, and only that one
    assert got["one unread"]["branches"] == [{"kind": "stop", "premises": ["its condition 'task.y' not read"], "kinds": ["not read"]}]
    assert got["one unread"]["code"] == "one-of"


def test_an_attributes_one_writer_left_none_dropped_by_a_guard_is_the_only_way_to_the_value_its_guards_held(tmp_path):
    out = _ok(
        _run(
            tmp_path,
            """
            import json
            from types import SimpleNamespace as N
            from bamboo.querier import evaluate as E
            from bamboo.querier import rerun
            from bamboo.querier.which_ran import Candidate, Verdict

            rerun.R = N(function=lambda rel, func, lines: N(name=func, node=func), statement_at=lambda node, line: (node, line),
                        entries=lambda fn, run, mode: [0], must_rows=lambda node, site: (None, []), seen=lambda fn, run, mode: N(hits={}))
            # a guard's call (task.useScout()) is read through the pure functions, which E fills from pandaserver at the first
            # call; the dev install (CI) has no pandaserver, and the call is no pure function either way
            E.PURE.setdefault("<no pure function is read here>", None)
            guards = {("gen", 9): ["task.x", "task.y", "not (task.useScout())"], ("put", 7): ["task.z"]}
            rerun.pathcond = N(arm_condition=lambda site: guards[site], path_condition=lambda site: guards[site],
                               exits_above_at=lambda site: [])
            new, old = {"status": "running", "time": "2026-10-29 11:38:00"}, {"status": "defined", "time": "2026-10-29 07:42:23"}
            # task.x holds in the rows' values; task.y and the method are read nowhere; task.z is false at the newer row
            envs = {new["time"]: E.Env({"task.x": E.Value(True, "row"), "task.z": E.Value(False, "row")}),
                    old["time"]: E.Env({"task.x": E.Value(True, "row")})}
            readings = [N(file=f"pandajedi/{f}.py", owner=f"pandajedi/{f}.py::{func}") for f, func in (("g", "gen"), ("s", "put"))]
            unshown = rerun.Premise("no line of a.log in the records holds 7: its runs are not shown to name it", rerun.NOT_READ)
            cond = rerun.Premise(rerun.T.Said("its-condition-not-read", {"guard": "task.y"}), rerun.NOT_READ)

            def may(f, func, line, row, premises):
                branch = rerun.Wrote(readings, None, [], None, Verdict(Candidate(f"pandajedi/{f}.py", func, [line]), line, "unexamined", []),
                                     row=row, window=(row["time"], row["time"]))
                return rerun.Possible(f"{f}.py::{func}:{line}", row, False, list(premises), branch=branch)

            def walk(left, value="A", dropped=()):
                r = object.__new__(rerun.Rerun)
                r.lines, r.opened, r.ends, r.touched, r.row_facts, r.hop, r.hand, r.hand_used, r.result, r.mode = [], [], [], [], {}, "H2", {}, [], {}, "outside"
                r.data = {}  # no rows file: no held column (heldcol, K49)
                r.question, r.branches, r.end_code, r._walking = None, [], None, None
                if dropped:
                    r._dropped_by_guard = frozenset(dropped)
                r.base_env = lambda at=None: envs.get(at, E.Env({}))
                r._which_wrote = lambda *x: rerun.OneOf(left, 2)
                r.steps = {rerun.WroteBy: r.wrote_by}
                q = r.asked(rerun.Asked("JediTaskSpec.transUses", value, "7"))
                try:
                    while q is not None:
                        q = r.steps.get(type(q), getattr(r, "arms_left"))(q)
                    said = None
                except rerun.Stop as e:
                    said = str(e)
                return {"said": said, "ends": [f"END {said}" for _, said in r.ends], "code": r.end_code,
                        "branches": [{"kind": e["kind"], "premises": [str(p) for p in e["premises"]]} for e in r.branches]}

            gen = [may("g", "gen", 9, old, [unshown, cond])]
            put = [may("s", "put", 7, new, [unshown]), may("s", "put", 7, old, [unshown])]
            # what H2's judge drops by a guard: a candidate's own arm, or the arms a caller stands in for; not rule 4's
            r = object.__new__(rerun.Rerun)
            r._stand_ins = {("pandajedi/o.py::runImpl", 211): [N(arm="pandajedi/r.py::extract:5")]}
            dropped = sorted(r._guard_dropped([
                Verdict(Candidate("pandajedi/o.py", "runImpl", [211]), 211, "excluded", ["step 5: guard 'x' is false ({})"]),
                Verdict(Candidate("pandajedi/g.py", "gen", [9]), 9, "excluded", ["rule 6: reached only when x is already 'A'"]),
                Verdict(Candidate("pandajedi/s.py", "put", [7]), 7, "excluded", ["rule 4: must row(s) [3] not in the run"]),
            ]))
            print(json.dumps({
                "dropped": dropped,
                "the only way": walk(gen),
                "itself dropped at a newer row": walk(gen, dropped={"g.py::gen:9"}),
                "another dropped by a guard": walk(gen, dropped={"o.py::old:3"}),
                "a default": walk(gen, value="0"),
                "its guard false at the newer row": walk(put),
                "two left": walk(gen + put),
            }))
            """,
        )
    )
    got = json.loads(out.splitlines()[-1])
    assert got["dropped"] == ["g.py::gen:9", "r.py::extract:5"]
    end = ("END JediTaskSpec.transUses='A' written last by gen:9 (the one writer of the value left, none dropped by a guard: no "
           "run shows the line ran; its guards held, the only way to the value ('task.y', 'not (task.useScout())'); the others "
           "in the rows' values), in the pass of the statuslog row 'defined' at 2026-10-29 07:42:23")
    # the one writer left, no other dropped by a guard: its guards the values leave unread held -- a method's too -- and
    # nothing it rested on is open of who wrote it: an END, no one-of (K66)
    assert got["the only way"] == {"said": None, "ends": [end], "code": None, "branches": [{"kind": "end", "premises": []}]}
    # the arm itself dropped at another row by its guard is no other way to the value
    assert got["itself dropped at a newer row"] == got["the only way"]
    # another writer dropped by a guard, or a value a row holds with no write: not the only way, as before (K38, K62)
    for name in ("another dropped by a guard", "a default"):
        assert got[name]["code"] == "one-of" and got[name]["branches"][0]["kind"] == "stop", name
    # a guard read false still decides: the next row, where it is the only way
    assert got["its guard false at the newer row"]["ends"] == [
        "END JediTaskSpec.transUses='A' written last by put:7 (the one writer of the value left, none dropped by a guard: no run "
        "shows the line ran; its guards held, the only way to the value ('task.z')), in the pass of the statuslog row 'defined' "
        "at 2026-10-29 07:42:23"]
    # two left: neither is the only way
    assert got["two left"]["code"] == "one-of" and [b["kind"] for b in got["two left"]["branches"]] == ["stop", "stop"]


def test_an_attribute_end_says_what_it_leaves_unread_of_its_value_reaching_the_row(tmp_path):
    out = _ok(
        _run(
            tmp_path,
            """
            import ast, json
            from types import SimpleNamespace as N
            from bamboo.querier import mapdump, rerun

            src = '''
            def set_status(self, task_id, status):
                tmpLog = self.create_tagged_logger(comment, f"jediTaskID={task_id}")
                try:
                    varMap = {}
                    varMap[":status"] = status
                    self.cur.execute(sql, varMap)
                    return True
                except Exception:
                    self.dump_error_message(tmpLog)
                    return False

            def set_bare(self, task_id, status):
                varMap = {}
                varMap[":status"] = status
                self.cur.execute(sql, varMap)
            '''
            import textwrap
            from bamboo.codemap.panda import pathcond
            tree = ast.parse(textwrap.dedent(src))
            pathcond.attach_parents(tree)
            fns = {f.name: N(node=f, name=f.name) for f in tree.body}
            sql_arm = [N(why="not_an_attribute_arm")]
            mapdump._FRAGMENT[:] = [N(junctions=[
                N(subject="T.status", owner="p/m.py::DB.set_status", branches=[N(line=6, rewrites_open=sql_arm)]),
                N(subject="T.status", owner="p/m.py::DB.set_bare", branches=[N(line=15, rewrites_open=sql_arm)]),
                N(subject="T.attr", owner="p/r.py::Refiner.extract", branches=[N(line=9, rewrites_open=[N(why="row_key")])]),
            ])]

            def said(subject, owner, line, fn, via):
                got = rerun._store_unread(subject, owner, line, fns.get(fn), via)
                return rerun.T.text(got) if got else ""

            print(json.dumps({
                "attribute": said("T.attr", "p/r.py::extract", 9, None, None),
                "sql own run": said("T.status", "p/m.py::set_status", 6, "set_status", None),
                "sql via, caught": said("T.status", "p/m.py::set_status", 6, "set_status", "a call"),
                "sql via, not caught": said("T.status", "p/m.py::set_bare", 15, "set_bare", "a call"),
                "no branch": said("T.attr", "p/r.py::extract", 10, None, None),
            }))
            """,
        )
    )
    got = json.loads(out.splitlines()[-1])
    # an attribute arm's row is written by a later statement or call, which the END does not read (K41)
    assert got["attribute"] == "; its row is written by a later statement or call, which is not read (KNOWN_ISSUES.md K41)"
    # an SQL arm on its own run: its path is read with the statement's raise apart
    assert got["sql own run"] == ""
    # an SQL arm read on its caller's run, whose function catches its exception and returns: its failure is not read
    assert got["sql via, caught"] == ("; its statement's failure is caught in set_status (except at 9) and the call returns: "
                                      "that it wrote is not read (KNOWN_ISSUES.md K41)")
    # its exception leaves the function: the caller's rows after the call do not print on it
    assert got["sql via, not caught"] == ""
    # no branch of the arm in the map: nothing said
    assert got["no branch"] == ""


_TURNS = """
class Proxy:
    def __init__(self):
        self.logger = logger

    def finish(self, task_ids, status_map):
        for jediTaskID in task_ids:
            taskStatus = status_map[jediTaskID]
            self.logger.debug(f"start jediTaskID={jediTaskID} status={taskStatus}")
            toSkip = False
            if self.locked(jediTaskID):
                toSkip = True
            if not toSkip:
                self.logger.debug(f"read jediTaskID={jediTaskID} status={taskStatus}")
                if self.scouting(jediTaskID):
                    newTaskStatus = "scouted"
                elif self.running(jediTaskID):
                    if self.unprocessed(jediTaskID):
                        toSkip = True
                    if not toSkip:
                        n = self.mutable(jediTaskID)
                        self.logger.debug(f"jediTaskID={jediTaskID} has {n} mutable datasets")
                        if n > 0:
                            newTaskStatus = "defined"
                        elif self.ready(jediTaskID):
                            newTaskStatus = "registered"
                        else:
                            newTaskStatus = "prepared"
                else:
                    toSkip = True
                if not toSkip:
                    self.update(jediTaskID, newTaskStatus)
                    self.logger.debug(f"done new status={newTaskStatus} for jediTaskID={jediTaskID}")

    def once(self, task):
        self.logger.debug(f"begin status={task.status}")
        if self.scouting(task):
            newTaskStatus = "scouted"
        elif self.ready(task):
            newTaskStatus = "registered"
        else:
            newTaskStatus = "prepared"
        self.logger.debug(f"end new status={newTaskStatus}")

    def bare(self, task):
        if self.ready(task):
            newTaskStatus = "registered"
        else:
            newTaskStatus = "prepared"
        self.logger.debug(f"end new status={newTaskStatus}")

    def maybe(self, task):
        self.logger.debug(f"begin status={task.status}")
        if self.ready(task):
            newTaskStatus = "registered"
        elif self.done(task):
            newTaskStatus = "prepared"
        self.logger.debug(f"end new status={newTaskStatus}")
"""


def test_a_kept_arm_binding_a_local_to_the_value_a_row_of_the_keys_turn_printed_is_positive_where_only_it_gives_it(tmp_path):
    # prepareTasksToBeFinished_JEDI's batched run (MT-running-late): the turn of the question's task printed "start"
    # (3854), the mutable datasets (4025) and "done new status=prepared" (4074).  Every way from 4025 to 4074 binds
    # newTaskStatus, and of the bindings run since, only 4061 gives 'prepared' (P1).  From 3907 a way skipping 4015's
    # body (open only when toSkip is True, which the map's reaching does not read) reaches 4074 unbound: a turn that
    # printed no row after 3907 says nothing
    (_corpus(tmp_path) / "pandaserver" / "p.py").write_text(_TURNS)
    lines = _TURNS.splitlines()
    at = {v: lines.index(f'{" " * n}newTaskStatus = "{v}"') + 1
          for v, n in (("scouted", 20), ("defined", 28), ("registered", 28), ("prepared", 28))}
    done = lines.index('                    self.logger.debug(f"done new status={newTaskStatus} for jediTaskID={jediTaskID}")') + 1
    has = lines.index('                        self.logger.debug(f"jediTaskID={jediTaskID} has {n} mutable datasets")') + 1
    start = lines.index('            self.logger.debug(f"start jediTaskID={jediTaskID} status={taskStatus}")') + 1
    # the 'prepared' arm of each one-call function
    one = {func: lines.index('            newTaskStatus = "prepared"', lines.index(f"    def {func}(self, task):")) + 1
           for func in ("once", "bare", "maybe")}
    stamp = "2026-01-01 00:00:{:02d},000 panda.log.X: DEBUG    <label=user> {}\n"

    def turn(key: int, status: str, n: int | None) -> list[str]:
        return ([f"start jediTaskID={key} status=running", f"read jediTaskID={key} status=running"]
                + ([f"jediTaskID={key} has {n} mutable datasets"] if n is not None else [])
                + [f"done new status={status} for jediTaskID={key}"])

    logs = {
        # the question's task 2: its turn printed 'prepared'; the turns around it printed other values
        "turns": turn(1, "defined", 3) + turn(2, "prepared", 0) + turn(3, "registered", 0),
        # task 2's turn printed nothing after "read": from there a way reaches "done" with nothing bound
        "scout": turn(1, "prepared", 0) + turn(2, "scouted", None),
        # task 2 went round twice: which turn is the question's is not read
        "again": turn(2, "prepared", 0) + turn(2, "prepared", 0),
        # 'prepared' printed in task 1's turn, not in task 2's
        "other": turn(1, "prepared", 0) + turn(2, "defined", 3),
        "once": ["begin status=running", "end new status=prepared"],
        "bare": ["end new status=prepared"],
        "maybe": ["begin status=running", "end new status=prepared"],
    }
    for name, texts in logs.items():
        (tmp_path / f"{name}.log").write_text("".join(stamp.format(i, t) for i, t in enumerate(texts)))
    out = _ok(
        _run(
            tmp_path,
            """
            import json
            from bamboo.querier import evaluate as E
            from bamboo.querier import rows as R
            from bamboo.querier import which_ran as W
            from bamboo.querier.records import read_log, runs_for
            from bamboo.querier.which_ran import Candidate, judge

            R.ROWS_FROM = "walk"
            env = E.Env(subject="<task>.status", keys=frozenset({"jediTaskID"}), table={"task": {"jediTaskID": E.Value(2, "the rows file")}},
                        call=None)  # the guards' calls are not run: the corpus here has no helpers
            got = {}
            for name, observed in (("turns", "prepared"), ("scout", "scouted"), ("again", "prepared"), ("other", "prepared")):
                run, = runs_for(read_log(f"{name}.log"), "2")
                c = Candidate("pandaserver/p.py", "finish", ARMS, [run.log.name], run=run)
                got[name] = {v.line: [v.kind, v.reasons] for v in judge([c], {run.log.name: [run]}, observed=observed, env=env).verdicts}
            # one call, no loop: P1's own reading, as the judge reads it after P
            for func, line in ONE.items():
                run, = runs_for(read_log(f"{func}.log"), "prepared")
                fn = R.function("pandaserver/p.py", func, [line])
                got[func] = W.printed_local(fn, R.seen(fn, run, "outside"), R.statement_at(fn.node, line), "prepared", None, set())
            print(json.dumps(got))
            """.replace("ARMS", str(sorted(at.values()))).replace("ONE", repr(one)),
        )
    )
    got = json.loads(out.splitlines()[-1])
    turns, scout, again, other = ({int(k): v for k, v in got[name].items()} for name in ("turns", "scout", "again", "other"))

    def p1(verdict: list) -> list[str]:
        return [r for r in verdict[1] if r.startswith("P1:")]

    # the key's turn printed 'prepared'; from the mutable datasets' row every way binds newTaskStatus, and of the
    # bindings run since, defined and registered give other values
    assert turns[at["prepared"]][0] == "positive", turns[at["prepared"]]
    assert p1(turns[at["prepared"]]) == [
        f"P1: row {done} printed newTaskStatus='prepared' in the turn of jediTaskID=2 (row {start} at turns.log:5), at "
        f"turns.log:8; every way from row {has} (at turns.log:7) to it binds newTaskStatus, and of the bindings run since "
        f"({[at['defined'], at['registered'], at['prepared']]}) only {at['prepared']} gives 'prepared'"
    ], turns[at["prepared"]]
    assert not p1(turns[at["defined"]]) and not p1(turns[at["registered"]]), turns
    # no record of the turn from which every way to "done" binds the name: nothing said
    assert scout[at["scouted"]][0] == "kept" and not p1(scout[at["scouted"]]), scout[at["scouted"]]
    # two turns of the key, or the key's turn showing another value: nothing said
    assert again[at["prepared"]][0] == "kept" and not p1(again[at["prepared"]]), again[at["prepared"]]
    assert other[at["prepared"]][0] == "kept" and not p1(other[at["prepared"]]), other[at["prepared"]]
    # one call: from the print before it, or from the entry where every way binds the name
    assert got["once"] == (
        f"P1: row {one['once'] + 1} printed newTaskStatus='prepared' in the call, at once.log:2; every way from row "
        f"{one['once'] - 6} (at once.log:1) to it binds newTaskStatus, and of the bindings run since "
        f"({[one['once'] - 4, one['once'] - 2, one['once']]}) only {one['once']} gives 'prepared'"
    ), got["once"]
    assert got["bare"] == (
        f"P1: row {one['bare'] + 1} printed newTaskStatus='prepared' in the call, at bare.log:1; every way from the call's "
        f"entry to it binds newTaskStatus, and of the bindings run since ({[one['bare'] - 2, one['bare']]}) only "
        f"{one['bare']} gives 'prepared'"
    ), got["bare"]
    # a way binding nothing reaches the row: neither the print before it nor the entry binds it on every way
    assert got["maybe"] == "", got["maybe"]


_J_PROXY = """\
from pandaserver.spec import JobSpec


class Proxy:
    def killJob(self, pandaID, code):
        self.logger.debug(f"code={code}")
        if isinstance(code, int):
            code = str(code)
        try:
            flagKilled = False
            for attempt in range(2):
                sql2 = f"SELECT {JobSpec.columnNames()} FROM ATLAS_PANDA.jobsActive4 WHERE PandaID=:PandaID"
                varMap = {}
                varMap[":PandaID"] = pandaID
                self.cur.execute(sql2, varMap)
                res = self.cur.fetchall()
                if len(res) == 0:
                    continue
                job = JobSpec()
                job.pack(res[0])
                if job.jobStatus != "failed":
                    if code in ["2", "4"]:
                        job.jobStatus = "closed"
                    elif code == "51":
                        job.jobStatus = "closed"
                    if job.jobStatus != "closed":
                        job.jobStatus = "cancelled"
                flagKilled = True
                break
            self.logger.debug(f"kill={flagKilled}")
            return flagKilled
        except Exception as e:
            if self.no_wait(e):
                self.logger.debug("skipped with NOWAIT")
            self.logger.debug("rolled back")
            return False
"""


def test_a_kept_arm_is_positive_where_its_call_ran_the_head_if_entered_no_handler_and_every_test_down_to_it_holds(tmp_path):
    # killJob's arms under 1017 (MJE-7397556704, code=50): the call printed kill=True, which only the binding after
    # the if gives, so the call ran the if; the handler's own print on every way is not in the run; the tests from the
    # if down hold with the state before the pass and the printed code, the stores of 'closed' under code tests false
    # (J; KNOWN_ISSUES.md K73).  code=None is None or "None": the tests are false under both
    corpus = _corpus(tmp_path)
    (corpus / "pandaserver" / "__init__.py").write_text("")
    (corpus / "pandaserver" / "spec.py").write_text(_ROW_SPEC)
    (corpus / "pandaserver" / "proxy.py").write_text(_J_PROXY)
    lines = _J_PROXY.splitlines()
    at = {t: lines.index(t) + 1 for t in (
        '                        job.jobStatus = "cancelled"', '                if job.jobStatus != "failed":',
        "        except Exception as e:", '                self.logger.debug("skipped with NOWAIT")',
        '            self.logger.debug("rolled back")')}
    closed = lines.index('                        job.jobStatus = "closed"') + 1
    stamp = "2026-01-01 00:00:{:02d},000 panda.log.X: DEBUG    < PandaID=1 > {}\n"
    logs = {
        "fifty": ["code=50", "kill=True"],
        "none": ["code=None", "kill=True"],
        # the handler printed: it was entered
        "entered": ["code=50", "kill=True", "rolled back"],
        # no row shows the if ran
        "short": ["code=50"],
        # two calls in the run
        "twice": ["code=50", "kill=True", "code=50", "kill=True"],
        # a store of 'closed' runs
        "two": ["code=2", "kill=True"],
    }
    for name, texts in logs.items():
        (tmp_path / f"{name}.log").write_text("".join(stamp.format(5 + i, t) for i, t in enumerate(texts)))
    handler = {"owner": "pandaserver/proxy.py::Proxy.killJob", "line": at["        except Exception as e:"],
               "catches": "Exception", "catch_all": True, "try_first": lines.index("            flagKilled = False") + 1,
               "try_last": at["        except Exception as e:"] - 1, "kind": "marked",
               "prints": [at['                self.logger.debug("skipped with NOWAIT")'], at['            self.logger.debug("rolled back")']]}
    got = {}
    for covered in (True, False):
        # one dump a process (mapdump.install): the handler covered by its own prints, then not
        out = _ok(
            _run(
                tmp_path,
                f"""
                import json
                from bamboo.codemap.models import HandlerMark, MapFragment
                from bamboo.querier import evaluate as E
                from bamboo.querier import mapdump
                from bamboo.querier import rows as R
                from bamboo.querier import which_ran as W
                from bamboo.querier.records import read_log, runs_for

                R.ROWS_FROM = "walk"
                before = {{"status": "merging", "time": "2026-01-01 00:00:00", "time_extended": "2026-01-01 00:00:00.000000",
                           "host": "m1"}}
                fragment = MapFragment(map_id="panda", derived_from="source-root:{_corpus(tmp_path)}",
                                       handler_marks=[HandlerMark(**{handler!r}, prints_on_every_way={covered})])
                with open("map.json", "w") as f:
                    f.write(fragment.model_dump_json())
                mapdump.install("map.json")
                fn = R.function("pandaserver/proxy.py", "killJob")
                got = {{}}
                for name in {sorted(logs) if covered else ["fifty"]!r}:
                    run, = runs_for(read_log(f"{{name}}.log"), "1")
                    sn = R.seen(fn, run, "outside")
                    for line in ({at['                        job.jobStatus = "cancelled"']}, {closed}):
                        env = E.Env(rows={{"job": "job"}}, subject="<job>.jobStatus", keys=frozenset({{"PandaID"}}),
                                    statuslog=[before], call=None)
                        got[f"{{name}} {{line}} {covered}"] = W.ran_down_to(fn, sn, sn, R.statement_at(fn.node, line), env, set())
                print(json.dumps(got))
                """,
            )
        )
        got.update(json.loads(out.splitlines()[-1]))
    cancelled, head = at['                        job.jobStatus = "cancelled"'], at['                if job.jobStatus != "failed":']
    code_row, kill_row = lines.index('        self.logger.debug(f"code={code}")') + 1, lines.index('            self.logger.debug(f"kill={flagKilled}")') + 1
    flag, handler_line = lines.index("                flagKilled = True") + 1, at["        except Exception as e:"]
    second = lines.index('                        job.jobStatus = "closed"', closed) + 1
    state = "the statuslog row before the pass, 'merging' at 2026-01-01 00:00:00.000000"

    def ran(name: str, arm: int) -> str:
        return (f"J: the call ran {head} (row {kill_row} at {name}.log:2 printed flagKilled='True', which of the bindings run "
                f"since row {code_row} (at {name}.log:1) only [{flag}] can give, every way to them passing {head}); it entered "
                f"none of the handlers [{handler_line}] around {head}..{arm} (their own prints not in the run); the tests from "
                f"{head} down to {arm} hold: ")

    # the print after the loop shows flagKilled True, which only the binding after the if gives; 1161's own prints are
    # not in the call; 1071's read: the stores of 'closed' under code tests false with "50", the state before the pass
    off = f"""{closed} under "code in ['2', '4']" False, {second} under "code == '51'" False"""
    assert got[f"fifty {cancelled} True"] == ran("fifty", cancelled) + (
        f""""job.jobStatus != 'closed'" True ({ {'job.jobStatus': f'{state}; the other writes reaching it do not run: {off}'} }); """
        f""""job.jobStatus != 'failed'" True ({ {'job.jobStatus': state} })"""
    ), got[f"fifty {cancelled} True"]
    # code=None is None or "None": the stores' tests are false under both
    both = "code read as each of ['None', None]"
    off = f"""{closed} under "code in ['2', '4']" False ({both}), {second} under "code == '51'" False ({both})"""
    assert got[f"none {cancelled} True"] == ran("none", cancelled) + (
        f""""job.jobStatus != 'closed'" True ({ {'job.jobStatus': f'{state}; the other writes reaching it do not run: {off}'} }); """
        f""""job.jobStatus != 'failed'" True ({ {'job.jobStatus': state} })"""
    ), got[f"none {cancelled} True"]
    # the store of 'closed' under code in ['2', '4'] does not run with code 50
    assert got[f"fifty {closed} True"] == "", got
    # the handler is not covered by its own prints, entered, or no row shows the if ran, or two calls
    assert got[f"fifty {cancelled} False"] == "", got
    assert got[f"entered {cancelled} True"] == "" and got[f"short {cancelled} True"] == "", got
    assert got[f"twice {cancelled} True"] == "", got
    # with code 2 the store of 'closed' runs: the arm of 'closed' is read, 'cancelled' is not
    assert got[f"two {closed} True"].startswith(ran("two", closed) + "\"code in ['2', '4']\" True"), got
    assert got[f"two {cancelled} True"] == "", got


def test_s_reads_a_parameter_up_its_callers_to_constants_and_an_apis_argument(tmp_path):
    # killJob's code: '' from one caller, and through killJobs' own code '60' from another and whatever an API hands
    out = _ok(_run(tmp_path, """
        import json
        from types import SimpleNamespace as N
        from bamboo.querier import mapdump
        from bamboo.querier import sources as SR

        src = lambda kind, name="", route=(), site_owner="", paths=(): N(kind=kind, name=name, route=list(route),
                                                                           site_owner=site_owner, paths=list(paths))
        # where a part of the rows sits in the value: no way of the value itself
        part = src("rows", "panda:p/r.py::ids@9", ["p/r.py:9"], paths=["[*]"])
        mapdump._FRAGMENT[:] = [N(parameter_froms=[
            N(owner="p/k.py::killJob", parameter="code", line=812, stops=[],
              sources=[src("constant", "''", ["p/t.py:697"]), src("parameter", "code", ["p/t.py:1190"], "p/t.py::killJobs")]),
            N(owner="p/t.py::killJobs", parameter="code", line=1138, stops=[],
              sources=[src("constant", "'60'", ["p/a.py:226"]), part, src("parameter", "code", ["p/api.py:572"], "p/api.py::kill")]),
            N(owner="p/api.py::kill", parameter="code", line=513, sources=[], stops=[N(kind="no_caller")]),
            # a parameter handed back to itself is read once
            N(owner="p/c.py::loop", parameter="x", line=1, stops=[], sources=[src("parameter", "x", ["p/c.py:9"], "p/c.py::loop")]),
            N(owner="p/q.py::parts", parameter="ids", line=5, stops=[], sources=[part]),
        ], entered_froms=[])]
        ways = [[w.end, w.value, w.why, list(w.route)] for w in SR.param_ways("p/k.py::killJob", "code")]
        print(json.dumps({"ways": ways, "loop": [[w.end, w.why] for w in SR.param_ways("p/c.py::loop", "x")],
                          "none": [[w.end, w.why] for w in SR.param_ways("p/k.py::other", "code")],
                          "parts": [[w.end, w.why, list(w.route)] for w in SR.param_ways("p/q.py::parts", "ids")],
                          "same": [SR.same_value("'60'", "60"), SR.same_value("None", "None"), SR.same_value("2", "2"),
                                   SR.same_value("'2'", "3")]}))
    """))
    got = json.loads(out.splitlines()[-1])
    assert got["ways"] == [
        ["constant", "''", "constant", ["p/t.py:697"]],
        # the outermost call first: the constant's own, then each caller's call into the next
        ["constant", "'60'", "constant", ["p/a.py:226", "p/t.py:1190"]],
        ["api", "", "no_caller", ["p/api.py:513", "p/api.py:572", "p/t.py:1190"]],
    ]
    assert got["loop"] == [["stop", "cycle"]]
    assert got["none"] == [["stop", "no-parameter-from"]]
    # killJobs' part of the rows is not among killJob's ways above; a parameter with only parts stops, named
    assert got["parts"] == [["stop", "parts-only", ["p/q.py:5"]]]
    # a print shows a value's str: '60' printed is 60, None is None
    assert got["same"] == [True, True, True, False]


_KILLER = """
class Killer:
    def kill(self, job, code):
        if not self._commit():
            raise RuntimeError("commit")
        res = self.cur.fetchone()
        if res is None:
            return
        if code == "60":
            job.jobStatus = "closed"
"""


def test_s_ends_each_guard_term_where_the_maps_ways_end_and_a_call_by_its_runnable_record(tmp_path):
    (_corpus(tmp_path) / "pandaserver" / "k.py").write_text(_KILLER)
    lines = _KILLER.splitlines()
    site = lines.index('            job.jobStatus = "closed"') + 1
    commit, res, code = (lines.index(f"        if {t}:") + 1 for t in ("not self._commit()", "res is None", 'code == "60"'))
    out = _ok(_run(tmp_path, f"""
        import json
        from types import SimpleNamespace as N
        from bamboo.codemap.panda import pathcond
        from bamboo.querier import mapdump, rerun
        from bamboo.querier import rows as R
        from bamboo.querier import sources as SR

        R.ROWS_FROM = "walk"
        owner = "pandaserver/k.py::kill"
        src = lambda kind, name, line, **kw: N(kind=kind, name=name, line=line, changes=kw.pop("changes", []), site_owner="",
                                               site_line=None, **kw)
        mapdump._FRAGMENT[:] = [N(
            guard_froms=[
                N(owner=owner, test_line={commit}, name="self", sources=[src("self_field", "self", {commit})], stops=[]),
                # a part of other rows sitting in res is no way of res
                N(owner=owner, test_line={res}, name="res", sources=[src("rows", "panda:" + owner + "@{res - 2}", {res - 2}),
                                                                   src("rows", "panda:other@1", 1, paths=["[0]"])],
                  stops=[]),
                N(owner=owner, test_line={code}, name="code", sources=[src("parameter", "code", 3)], stops=[]),
                # a name the map holds only parts for stops, named
                N(owner=owner, test_line={code}, name="cpart", sources=[src("self_field", "self.jobs", 1, paths=["[0]", "[1]"])],
                  stops=[]),
            ],
            runnables=[
                # the commit reaches the database: its answer ends the term its receiver is
                N(owner="pandaserver/b.py::_commit", runnable=False, asked_at=["pandaserver/k.py:{commit}"],
                  stops=[N(kind="database", at="pandaserver/b.py:194")]),
            ])]
        fn = R.function("pandaserver/k.py", "kill", within=[{site}])
        stmt = R.statement_at(fn.node, {site})
        guards = pathcond.arm_condition(stmt)
        ifs = rerun._guard_ifs(fn, stmt, guards)
        terms = SR.terms(fn, owner, list(zip(guards, ifs)))
        print(json.dumps([[t.term, t.ended, [[e.end, e.why, e.at, e.param] for e in t.ends]] for t in terms]))
    """))
    got = json.loads(out.splitlines()[-1])
    assert got == [
        ["code", False, [["stop", "parameter", "pandaserver/k.py:3", "code"]]],
        ["cpart", False, [["stop", "parts-only", f"pandaserver/k.py:{code}", ""]]],
        ["self", True, [["database", "database", "pandaserver/b.py:194", ""]]],
        ["res", True, [["database", "rows", f"pandaserver/k.py:{res - 2}", ""]]],
    ], got


def test_a_row_fill_is_not_a_change_on_the_way_it_ends(tmp_path):
    # killJob's job, built at 997 and filled by job.pack(res[0]) at 998 from the key's row: the map records pack's
    # setattr (JobSpec 423) as a change made by 998; the fills reading has read that fill, so the term ends at the row.
    # A change at another statement stays on the way, and the term does not end (G4).
    corpus = _corpus(tmp_path)
    (corpus / "pandaserver" / "__init__.py").write_text("")
    (corpus / "pandaserver" / "spec.py").write_text(_ROW_SPEC)
    (corpus / "pandaserver" / "proxy.py").write_text(_ROW_PROXY)
    out = _ok(_run(tmp_path, """
        import json
        from types import SimpleNamespace as N
        from bamboo.codemap.panda import pathcond
        from bamboo.querier import mapdump, rerun
        from bamboo.querier import rows as R
        from bamboo.querier import sources as SR

        import ast
        from bamboo.querier import asked
        from bamboo.querier import printed as PR
        from bamboo.querier import which_ran as WR

        R.ROWS_FROM = "walk"
        owner = "pandaserver/proxy.py::killJob"
        fill = N(at="pandaserver/spec.py:10", kind="method", key="*", via=["pandaserver/proxy.py:14"])
        other = N(at="pandaserver/proxy.py:99", kind="store", key=".jobStatus", via=[])
        got = {}
        # the name job and the field job.jobStatus start at one place: each its own answer, whichever is read first
        fn = R.function("pandaserver/proxy.py", "killJob", within=[17])
        test = R.statement_at(fn.node, 15).test
        calls = PR._calls_index(asked._types())
        name = next(n for n in ast.walk(test) if isinstance(n, ast.Name) and n.id == "job")
        field = next(n for n in ast.walk(test) if isinstance(n, ast.Attribute))
        held = [WR._row_held(calls, fn.rel, fn.node, x, ()) for x in (name, field)]
        got["held"] = [[[f.site.lineno for f in h.fills], [o.kind for o in h.others]] for h in held]
        for name, changes in (("the fill", [fill]), ("another", [fill, other])):
            mapdump._FRAGMENT[:] = [N(guard_froms=[
                N(owner=owner, test_line=15, name="job", stops=[], sources=[
                    N(kind="object", name="JobSpec()", line=13, changes=changes, site_owner="", site_line=None, paths=[],
                      steps=[], through=[], route=[])]),
                N(owner=owner, test_line=16, name="code", stops=[], sources=[
                    N(kind="constant", name="'2'", line=1, changes=[], site_owner="", site_line=None, paths=[], steps=[],
                      through=[], route=[])])], runnables=[], field_froms=[])]
            SR._index.cache_clear()
            fn = R.function("pandaserver/proxy.py", "killJob", within=[17])
            stmt = R.statement_at(fn.node, 17)
            guards = pathcond.arm_condition(stmt)
            terms = SR.terms(fn, owner, list(zip(guards, rerun._guard_ifs(fn, stmt, guards))))
            got[name] = [[t.term, t.ended, [[e.end, e.why, [c.at for c in e.changes]] for e in t.ends]] for t in terms]
        print(json.dumps(got))
    """))
    got = json.loads(out.splitlines()[-1])
    assert got["held"] == [[[], ["binding"]], [[14], []]], got
    assert got["the fill"][0] == ["job", True, [["row", "object", []]]], got
    assert got["another"][0] == ["job", False, [["row", "object", ["pandaserver/proxy.py:99"]]]], got


def test_f1_reads_the_entry_by_the_call_a_callers_run_shows_where_no_printed_argument_chose_it(tmp_path):
    # killUnusedEventServiceConsumers < PandaID=7398538282 >: the job archiveJob archives, which it hands the consumer
    # killer at three calls of one run; another caller hands a name its marks do not print
    src = textwrap.dedent(
        """\
        class K:
            def arm(self, job):
                job.jobStatus = "closed"

            def c1(self, job):
                self.arm(job)
                if job.x:
                    self.arm(job)
                self.arm(job)

            def c2(self, other):
                self.arm(other)
        """
    )
    (_corpus(tmp_path) / "pandaserver" / "m.py").write_text(src)
    lines = src.splitlines()
    c1_calls = [i + 1 for i, t in enumerate(lines) if t.strip() == "self.arm(job)"]
    c2_call = lines.index("        self.arm(other)") + 1
    out = _ok(_run(tmp_path, f"""
        import ast
        import json
        from types import SimpleNamespace as N
        from bamboo.querier import rerun
        from bamboo.querier import sources as SR

        from bamboo.querier import mapdump

        M = "pandaserver/m.py"
        mapdump._FRAGMENT[:] = [N(tables=[N(spec_class="JobSpec", identity_attrs=["JobSpec.PandaID"])])]
        # c2's mark prints its own pandaID, not the name it hands: the same row by the hole's column
        MARKS = {{"arm": ("job.PandaID", "arm < PandaID=.*? >"), "c1": ("job.PandaID", "c1 < PandaID=.*? >"),
                 "c2": ("pandaID", "c2 < PandaID=.*? >")}}
        identity = [True]

        def prints(owner, lines):
            name = owner.split("::")[-1].split(".")[-1]
            expr, prefix = MARKS[name]
            columns = ["JobSpec.PandaID"] if name != "arm" or identity[0] else ["JobSpec.jobsetID"]
            return N(runs=[N(mark="PandaID={{}}", key=[N(expression=expr, columns=columns)])],
                     statements=[N(log_files=["a.log"], rows=[N(prefix=prefix)])])

        rerun.R._prints_of = prints
        rerun.owners_of = lambda ids: {{}}
        rec =N(sites=[N(owner=f"{{M}}::K.c1", line=ln) for ln in {c1_calls}] + [N(owner=f"{{M}}::K.c2", line={c2_call})], open=[])
        SR.entered = lambda owner, line: rec
        picked = [{c1_calls[1]}]
        rerun.judge = lambda cands, logs, **kw: N(verdicts=[
            N(line=c.lines[0], kind="positive" if c.lines[0] in picked else "excluded", describe=lambda c=c: f"judged {{c.lines[0]}}")
            for c in cands])

        def run(token, first, last, machine="m1"):
            return N(token=token, first=f"2026-11-02 {{first}},000", last=f"2026-11-02 {{last}},000", machine=machine,
                     describe=lambda: f"the run {{token!r}}")

        arm = run("arm < PandaID=9 >", "13:43:27", "13:43:29")
        fn = N(node=ast.parse("def arm(self):\\n    pass").body[0], rel=M, name="arm")
        ended = [SR.Term("g", "x", (SR.End("x", SR.CONSTANT, "constant", f"{{M}}:1"),))]
        stopped = [SR.Term("g", "y", (SR.End("y", SR.STOP, "self_field", f"{{M}}:1"),))]
        got = {{}}

        def entry(name, terms, runs):
            r = object.__new__(rerun.Rerun)
            r.lines, r.hop, r.hand, r.hand_used, r.unanswered = [], "H3", {{}}, [], []
            r.opened, r.ends, r.branches = [], [], []
            r.mode, r.decide, r.data, r.args = "map", "positive", {{}}, N(asks="asks.jsonl", key="9")
            r.logs = N(needs_window=True, names=lambda files: list(files))
            r.files_of = lambda owner, today, lines: today
            asked = []
            r._ask = lambda key, names, asking, owners=None, whole=False: (
                asked.append([key, list(asking.strings), str(asking.why)]) or ({{"a.log": runs}}, [], {{}}))
            e = r._entry_end(fn, f"{{M}}::arm", 3, arm, ("2026-11-02 13:43:27", "2026-11-02 13:43:29"), terms)
            got[name] = [e.end, e.why, e.at]
            got[name + ": asked"] = asked

        caller = run("c1 < PandaID=9 >", "13:43:20", "13:43:40")
        # one caller's run holds the arm's and three of its calls: H2 judges each in that run
        entry("judged", ended, [caller])
        # a term that stops: the callers are not asked
        entry("a term stops", stopped, [caller])
        # no caller's run holds the arm's (another machine)
        entry("none holds it", ended, [run("c1 < PandaID=9 >", "13:43:20", "13:43:40", "m2")])
        # a mark holding no row's identity: each caller's logs by the key
        identity[0] = False
        picked[:] = [{c2_call}]
        entry("by the key", ended, [caller])
        print(json.dumps(got))
    """))
    got = json.loads(out.splitlines()[-1])
    assert got["judged"] == ["caller", "entered-by", f"pandaserver/m.py:{c1_calls[1]}"]
    # the callers' marks holding the arm's row, with the run's value: c1's and c2's, whatever their holes spell
    assert got["judged: asked"] == [["9", ["c1 < PandaID=9 >", "c2 < PandaID=9 >"],
                                     "over the arm's run, 2026-11-02 13:43:27 .. 2026-11-02 13:43:29, "
                                     "for 'c1 < PandaID=9 >', 'c2 < PandaID=9 >'"]]
    assert got["a term stops"] == ["stop", "entry-unchosen", ""]
    assert got["a term stops: asked"] == []
    assert got["none holds it"] == ["stop", "entry-unchosen", ""]
    assert got["by the key"] == ["caller", "entered-by", f"pandaserver/m.py:{c2_call}"]
    assert [a[2] for a in got["by the key: asked"]] == [
        f"for the call at pandaserver/m.py:{ln} that may have entered it" for ln in [*c1_calls, c2_call]]


def test_owi_ends_an_entry_at_the_one_call_the_map_holds_into_it_where_nothing_else_may_enter(tmp_path):
    # process_job_report (adder_gen 382): the map's EnteredFrom holds run:114 alone and nothing open, so the arm's
    # function was entered there -- read without asking a caller's run, even where a term stops (F4's gate); adder's
    # execute, one site and a hole open, stays with F1
    src = textwrap.dedent(
        """\
        class K:
            def arm(self):
                self.status = "closed"

            def c1(self):
                self.arm()

            def c2(self):
                self.arm()
        """
    )
    (_corpus(tmp_path) / "pandaserver" / "m.py").write_text(src)
    lines = src.splitlines()
    c1_call, c2_call = [i + 1 for i, t in enumerate(lines) if t.strip() == "self.arm()"]
    out = _ok(_run(tmp_path, f"""
        import ast
        import json
        from types import SimpleNamespace as N
        from bamboo.querier import rerun
        from bamboo.querier import sources as SR

        M = "pandaserver/m.py"
        one = [N(owner=f"{{M}}::K.c1", line={c1_call})]
        two = [*one, N(owner=f"{{M}}::K.c2", line={c2_call})]
        hole = [N(kind="hole", at="pandaserver/n.py:7", text="x.arm")]
        arm = N(token="arm", first="2026-11-02 13:43:27,000", last="2026-11-02 13:43:29,000", machine="m1",
                describe=lambda: "the run 'arm'")
        fn = N(node=ast.parse("def arm(self):\\n    pass").body[0], rel=M, name="arm")
        ended = [SR.Term("g", "x", (SR.End("x", SR.CONSTANT, "constant", f"{{M}}:1"),))]
        stopped = [SR.Term("g", "y", (SR.End("y", SR.STOP, "self_field", f"{{M}}:1"),))]
        got = {{}}

        def entry(name, sites, open_, terms):
            SR.entered = lambda owner, line: N(sites=sites, open=open_)
            r = object.__new__(rerun.Rerun)
            r.lines = []
            asked = []
            r._entered_by = lambda owner, line, run, window, rec: asked.append(owner)
            e = r._entry_end(fn, f"{{M}}::K.arm", 2, arm, ("2026-11-02 13:43:27", "2026-11-02 13:43:29"), terms)
            got[name] = [e.end, e.why, e.at, len(asked), r.lines]

        entry("one site, a term stops", one, [], stopped)
        entry("one site, every term ends", one, [], ended)
        entry("one site and a hole", one, hole, ended)
        entry("one site and a hole, a term stops", one, hole, stopped)
        entry("two sites", two, [], ended)
        print(json.dumps(got))
    """))
    got = json.loads(out.splitlines()[-1])
    said = [f"  entry: the map holds one call into arm, at pandaserver/m.py:{c1_call}, and no other way in"]
    assert got["one site, a term stops"] == ["caller", "only-way-in", f"pandaserver/m.py:{c1_call}", 0, said]
    assert got["one site, every term ends"] == ["caller", "only-way-in", f"pandaserver/m.py:{c1_call}", 0, said]
    # something open: F1 asks where every term ends, else the entry stops as before
    assert got["one site and a hole"] == ["stop", "entry-unchosen", "", 1, []]
    assert got["one site and a hole, a term stops"] == ["stop", "entry-unchosen", "", 0, []]
    assert got["two sites"] == ["stop", "entry-unchosen", "", 1, []]


def test_ebv_ends_an_entry_at_the_one_call_the_values_its_run_printed_at_its_entry_leave(tmp_path):
    # killJob (job_complex_module 812), MJE-7397556704: four calls in, nothing open; its run printed code '50' at its
    # entry, so the calls handing "51" (task_complex_module 5732) and "" / "55" are not it, and TaskBuffer 1190, whose
    # ways go on to '50' and a stop, is left.  MJE-7397817316: code '51' leaves 5732 too; user 'JEDI' rules it out.
    src = textwrap.dedent(
        """\
        class K:
            def arm(self, code, user):
                self.status = "closed"

            def c1(self):
                self.arm("51", "reassign")

            def c2(self):
                self.arm("", "x")

            def c3(self, code, user):
                self.arm(code, user)
        """
    )
    (_corpus(tmp_path) / "pandaserver" / "m.py").write_text(src)
    lines = src.splitlines()
    c1, c2, c3 = [f"pandaserver/m.py:{i + 1}" for i, t in enumerate(lines) if t.strip().startswith("self.arm(")]
    out = _ok(_run(tmp_path, f"""
        import ast
        import json
        from types import SimpleNamespace as N
        from bamboo.querier import rerun
        from bamboo.querier import sources as SR

        M = "pandaserver/m.py"
        c1, c2, c3 = {c1!r}, {c2!r}, {c3!r}
        sites = [N(owner=f"{{M}}::K.{{c}}", line=int(at.rpartition(":")[2])) for c, at in (("c1", c1), ("c2", c2), ("c3", c3))]
        arm = N(token="arm", first="2026-11-02 13:43:27,000", last="2026-11-02 13:43:29,000", machine="m1",
                describe=lambda: "the run 'arm'")
        fn = N(node=ast.parse("def arm(self, code, user):\\n    pass").body[0], rel=M, name="arm")
        ended = [SR.Term("g", "x", (SR.End("x", SR.CONSTANT, "constant", f"{{M}}:1"),))]
        stopped = [SR.Term("g", "y", (SR.End("y", SR.STOP, "self_field", f"{{M}}:1"),))]
        W, C = SR.Way, SR.CONSTANT
        base = {{"code": [W(C, '"51"', "constant", (c1,)), W(C, '""', "constant", (c2,)),
                          W(C, "'50'", "constant", ("pandaserver/n.py:4", c3)),
                          W(SR.STOP, "", "missed,no_caller", ("pandaserver/api.py:9", c3))],
                 "user": [W(C, '"reassign"', "constant", (c1,)), W(C, '"x"', "constant", (c2,)),
                          W(SR.STOP, "", "unknown", (c3,))]}}
        got = {{}}

        def entry(name, printed, terms=stopped, ways=base, open_=(), f1=None):
            SR.entered = lambda owner, line: N(sites=sites, open=list(open_))
            SR.param_ways = lambda owner, p, **kw: ways[p]
            r = object.__new__(rerun.Rerun)
            r.lines = []
            asked = []
            r._entered_by = lambda owner, line, run, window, rec: asked.append(owner) or f1
            r._entry_value = lambda fn, run, p: printed.get(p)
            r._param_end = lambda term, owner, p, fn, run, window, **kw: (SR.End(term, SR.STOP, "parameter-unchosen", ""),)
            e = r._entry_end(fn, f"{{M}}::K.arm", 2, arm, ("2026-11-02 13:43:27", "2026-11-02 13:43:29"), terms)
            got[name] = [e.end, e.why, e.at, len(asked), r.lines]

        entry("code rules out two", {{"code": "50"}})
        entry("code leaves two, user rules one out", {{"code": "51", "user": "JEDI"}})
        entry("code leaves two", {{"code": "51"}})
        entry("every term ends: F1 first", {{"code": "50"}}, terms=ended)
        entry("F1 shows a call", {{"code": "50"}}, terms=ended, f1=SR.End("entry", SR.CALLER, "entered-by", c2))
        entry("nothing printed", {{}})
        entry("something open", {{"code": "50"}}, open_=[N(kind="hole", at="pandaserver/n.py:7", text="x.arm")])
        entry("a part of a literal", {{"code": "50"}}, ways={{**base, "code": [W(C, "", SR.THROUGH, (c1,)), *base["code"][1:]]}})
        entry("a way stops at the function", {{"code": "50"}}, ways={{**base, "code": [*base["code"], W(SR.STOP, "", "missed", (f"{{M}}:2",))]}})
        entry("a call no way comes by", {{"code": "50"}}, ways={{**base, "code": [base["code"][0], *base["code"][2:]]}})
        print(json.dumps(got))
    """))
    got = json.loads(out.splitlines()[-1])
    head = "  entry: its run's prints at its entry rule out the call(s) into arm at "
    two = sorted([(c1, "code \"51\", printed '50'"), (c2, "code \"\", printed '50'")])
    assert got["code rules out two"] == ["caller", "by-values", c3, 0, [
        head + "; ".join(f"{c} ({w})" for c, w in two) + f": {c3} is left"]]
    by_user = sorted([(c1, "user \"reassign\", printed 'JEDI'"), (c2, "code \"\", printed '51', user \"x\", printed 'JEDI'")])
    assert got["code leaves two, user rules one out"] == ["caller", "by-values", c3, 0, [
        head + "; ".join(f"{c} ({w})" for c, w in by_user) + f": {c3} is left"]]
    assert got["code leaves two"] == ["stop", "entry-unchosen", "", 0, []]
    # F1 is read first where every term ends; EBV only where it shows none
    assert got["every term ends: F1 first"][:4] == ["caller", "by-values", c3, 1]
    assert got["F1 shows a call"] == ["caller", "entered-by", c2, 1, []]
    assert got["nothing printed"] == ["stop", "entry-unchosen", "", 0, []]
    assert got["something open"] == ["stop", "entry-unchosen", "", 0, []]
    # a literal a value is only part of is not the value: c1 stays
    assert got["a part of a literal"] == ["stop", "entry-unchosen", "", 0, []]
    # a way in by no call the map holds: the calls are not every way in
    assert got["a way stops at the function"] == ["stop", "entry-unchosen", "", 0, []]
    # c2 hands code no way: it stays beside c3
    assert got["a call no way comes by"] == ["stop", "entry-unchosen", "", 0, []]


_GEN = """
from pandaserver.srvcore.CoreUtils import naive_utcnow


class Plugin:
    def __init__(self):
        self.result = Result()


class Gen:
    def __init__(self, job, status, retry):
        self.job = job
        self.job_status = status
        self.retry = retry
        self.started = naive_utcnow()
        self.plugin = Plugin()
        self.result = None

    def report(self):
        self.result = self.plugin.result
        if self.job_status == "failed" and self.retry and self.started and self.extra and self.result:
            self.job.jobStatus = "failed"
"""


def test_s4a_reads_a_field_of_self_on_through_the_maps_field_records(tmp_path):
    # AdderGen.process_job_report 441's terms: a field bound from __init__'s parameter, one bound to what the plugin it
    # built holds, a clock; no binding is chosen over another here (S4b's), so each is a way
    (_corpus(tmp_path) / "pandaserver" / "g.py").write_text(_GEN)
    lines = _GEN.splitlines()
    pres, status, retry, started, plugin, none, bound, site = (lines.index(t) + 1 for t in (
        "        self.result = Result()", "        self.job_status = status", "        self.retry = retry",
        "        self.started = naive_utcnow()", "        self.plugin = Plugin()", "        self.result = None",
        "        self.result = self.plugin.result", '            self.job.jobStatus = "failed"'))
    test = site - 1
    out = _ok(_run(tmp_path, f"""
        import json
        from types import SimpleNamespace as N
        from bamboo.codemap.panda import pathcond
        from bamboo.querier import mapdump, rerun
        from bamboo.querier import rows as R
        from bamboo.querier import sources as SR

        R.ROWS_FROM = "walk"
        G, P, INIT, ARM = "pandaserver/g.py::Gen", "pandaserver/g.py::Plugin", "pandaserver/g.py::__init__", "pandaserver/g.py::report"
        src = lambda kind, name, line, **kw: N(kind=kind, name=name, line=line, changes=kw.pop("changes", []),
                                               site_owner=kw.pop("site_owner", ""), site_line=None, **kw)
        bind = lambda line, owner, *sources: N(at=f"pandaserver/g.py:{{line}}", owner=owner, how="assign",
                                               sources=list(sources), stops=[])
        mapdump._FRAGMENT[:] = [N(
            guard_froms=[
                # the test's plain self: each field it reads with no record of its own at the line
                N(owner=ARM, test_line={test}, name="self", sources=[src("self_field", "self", {test}, site_owner=G)], stops=[]),
                N(owner=ARM, test_line={test}, name="self.result",
                  sources=[src("self_field", "self.result", {test}, site_owner=G)], stops=[]),
            ],
            runnables=[],
            field_froms=[
                N(owner=G, field="job_status", changes=[], stops=[],
                  bindings=[bind({status}, INIT, src("parameter", "status", {status}, site_owner=INIT))]),
                N(owner=G, field="retry", stops=[], changes=[N(at="pandaserver/g.py:99", kind="store", key=".x", what="", via=[])],
                  bindings=[bind({retry}, INIT, src("parameter", "retry", {retry}, site_owner=INIT))]),
                N(owner=G, field="started", changes=[], stops=[], bindings=[bind({started}, INIT, src("call", "naive_utcnow", {started}))]),
                N(owner=G, field="plugin", changes=[], stops=[],
                  bindings=[bind({plugin}, INIT, src("object", "Plugin()", {plugin}, site_owner=P))]),
                N(owner=G, field="result", changes=[], stops=[], bindings=[
                    bind({none}, INIT, src("constant", "None", {none})),
                    bind({bound}, ARM, src("self_field", "self.plugin", {bound}, site_owner=G, steps=[".result"])),
                    # where a part of the job sits in it: no way of the value
                    bind(98, "pandaserver/g.py::other", src("self_field", "self.job", 98, site_owner=G, paths=[".job"])),
                ]),
                N(owner=P, field="result", changes=[], bindings=[bind({pres}, INIT, src("call", "Result", {pres}))],
                  stops=[N(kind="computed", at=["pandaserver/g.py::Plugin:3", "pandaserver/g.py::Plugin:4"])]),
            ])]
        fn = R.function("pandaserver/g.py", "report", within=[{site}])
        stmt = R.statement_at(fn.node, {site})
        guards = pathcond.arm_condition(stmt)
        terms = SR.terms(fn, ARM, list(zip(guards, rerun._guard_ifs(fn, stmt, guards))))
        print(json.dumps([[t.term, t.ended, [[e.end, e.why, e.at, e.param, e.of, list(e.via), [c.at for c in e.changes]]
                                              for e in t.ends]] for t in terms]))
    """))
    got = json.loads(out.splitlines()[-1])
    g = "pandaserver/g.py"
    assert got == [
        # a parameter of the binding's method: read up that method's callers next (rerun's _with_callers)
        ["self.job_status", False, [["stop", "parameter", f"{g}:{status}", "status", f"{g}::__init__", [], []]]],
        # the record's change is on the binding's way, not a way of its own (G4)
        ["self.retry", False, [["stop", "parameter", f"{g}:{retry}", "retry", f"{g}::__init__", [], [f"{g}:99"]]]],
        ["self.started", True, [["clock", "naive_utcnow", f"{g}:{started}", "", "", [], []]]],
        ["self.extra", False, [["stop", "no-field-from", f"{g}::Gen.extra", "", "", [], []]]],
        # the plugin's result read on in the plugin's own record, through the bindings at the read's and the plugin's
        ["self.result", False, [["constant", "None", f"{g}:{none}", "", "", [], []],
                                ["stop", "call", f"{g}:{pres}", "", "", [f"{g}:{bound}", f"{g}:{plugin}"], []],
                                ["stop", "computed", f"{g}::Plugin:3, {g}::Plugin:4", "", "", [f"{g}:{bound}", f"{g}:{plugin}"], []],
                                ["stop", "parts-only", f"{g}:98", "", "", [], []]]],
    ], got


_GEN5 = """
class Gen:
    def __init__(self, tb):
        self.job = tb.peek()
        self.result = Res()

    def report(self, tb):
        status = tb.get_status()
        if self.job is None and self.result.code and status:
            fail()
"""


def test_g2_a_way_end_keeps_a_changes_calls_and_a_returned_values_route_and_line_and_g3_its_bare_place(tmp_path):
    # case A's self.job at adder_gen 111 (out/gm/gm5-design.md G2): its sources came back through calls, from lines of
    # callees; AdderAtlasPlugin.result's set_fatal at adder_result 55, made through two calls, on the binding's way
    # (G4); a local that came back through a call.  Each is told apart by what the map's record holds, and only an end equal in all of it is one.
    # G3: result.json keeps the place as file:line, " at " in the template.
    (_corpus(tmp_path) / "pandaserver" / "g5.py").write_text(_GEN5)
    lines = _GEN5.splitlines()
    job, res, status, site = (lines.index(t) + 1 for t in (
        "        self.job = tb.peek()", "        self.result = Res()", "        status = tb.get_status()", "            fail()"))
    test = site - 1
    out = _ok(_run(tmp_path, f"""
        import json
        from types import SimpleNamespace as N
        from bamboo.codemap.panda import pathcond
        from bamboo.querier import mapdump, rerun
        from bamboo.querier import render as T
        from bamboo.querier import rows as R
        from bamboo.querier import sources as SR

        R.ROWS_FROM = "walk"
        g = "pandaserver/g5.py"
        G, INIT, ARM = f"{{g}}::Gen", f"{{g}}::__init__", f"{{g}}::report"
        src = lambda kind, name, line, **kw: N(kind=kind, name=name, line=line, changes=[], site_owner=kw.pop("site_owner", ""),
                                               site_line=None, **kw)
        back = lambda kind, name, line, *route: src(kind, name, line, route=list(route))
        bind = lambda line, *sources: N(at=f"{{g}}:{{line}}", owner=INIT, how="assign", sources=list(sources), stops=[])
        change = lambda via: N(at="pandaserver/res.py:55", kind="inner", key=".code", what="", via=[via])
        mapdump._FRAGMENT[:] = [N(
            guard_froms=[
                N(owner=ARM, test_line={test}, name="self.job", sources=[src("self_field", "self.job", {test}, site_owner=G)], stops=[]),
                N(owner=ARM, test_line={test}, name="self.result",
                  sources=[src("self_field", "self.result", {test}, site_owner=G)], stops=[]),
                # the local came back through get_status, from its callee's 1029, and through 1022's callee's 478;
                # the second record is the first again
                N(owner=ARM, test_line={test}, name="status", stops=[], sources=[
                    back("constant", "None", 1029, f"{{g}}:{status}"), back("constant", "None", 1029, f"{{g}}:{status}"),
                    back("constant", "None", 478, "pandaserver/tb.py:1022", f"{{g}}:{status}")]),
            ],
            runnables=[],
            field_froms=[
                N(owner=G, field="job", changes=[], stops=[], bindings=[bind({job},
                    back("unknown", "x", 978, "pandaserver/tb.py:1025", f"{{g}}:{job}"),
                    back("constant", "None", 478, "pandaserver/tb.py:1022", f"{{g}}:{job}"),
                    back("constant", "None", 1029, f"{{g}}:{job}"))]),
                N(owner=G, field="result", stops=[], bindings=[bind({res}, src("call", "Res", {res}))],
                  changes=[change(f"{{g}}:91"), change(f"{{g}}:92")]),
            ])]
        fn = R.function(g, "report", within=[{site}])
        stmt = R.statement_at(fn.node, {site})
        guards = pathcond.arm_condition(stmt)
        terms = SR.terms(fn, ARM, list(zip(guards, rerun._guard_ifs(fn, stmt, guards))))
        print(json.dumps({{
            "ends": [[t.term, [[e.end, e.why, e.at, [[c.at, list(c.made_by)] for c in e.changes], list(e.returned),
                                e.returned_line] for e in t.ends]] for t in terms],
            "said": [[str(rerun._way_said(e)) for e in t.ends] for t in terms],
            "text": [[rerun._way_text(e) for e in t.ends] for t in terms],
            "record": T.to_json(rerun._way_said(terms[1].ends[0])),
        }}))
    """))
    got = json.loads(out.splitlines()[-1])
    g, tb = "pandaserver/g5.py", "pandaserver/tb.py"
    J, S = f"{g}:{job}", f"{g}:{status}"
    assert got["ends"] == [
        # each placed at the binding, told apart by the calls it came back by and its callee's line
        ["self.job", [["stop", "unknown", J, [], [f"{tb}:1025", J], 978], ["constant", "None", J, [], [f"{tb}:1022", J], 478],
                      ["constant", "None", J, [], [J], 1029]]],
        # the two changes at res 55 on the binding's way (G4), told apart by the call each was made through
        ["self.result", [["stop", "call", f"{g}:{res}", [["pandaserver/res.py:55", [f"{g}:91"]], ["pandaserver/res.py:55", [f"{g}:92"]]],
                          [], 0]]],
        # the local placed at its call in report, not at its callee's line in g5; the repeated record said once
        ["status", [["constant", "None", S, [], [S], 1029], ["constant", "None", S, [], [f"{tb}:1022", S], 478]]],
    ], got["ends"]
    assert got["said"][0][1] == f"constant (None) at {J}, returned from callee line 478 by {tb}:1022 -> {J}"
    changed = f"changed between it and the read: pandaserver/res.py:55 made by {g}:91; pandaserver/res.py:55 made by {g}:92"
    assert got["said"][1][0] == f"stop (call) at {g}:{res}, {changed}"
    assert got["text"][0][1] == f"constant (None at {J}) returned from callee line 478 by {tb}:1022 -> {J}"
    assert got["text"][1][0] == f"stop (call at {g}:{res}), {changed}"
    # G3: the record holds the place as it is; its words are the template's
    facts = got["record"]["facts"]
    assert facts["at"] == {"said": {"kind": "way-end-at", "facts": {"at": f"{g}:{res}"}}}
    assert facts["changes"]["said"]["facts"]["changes"][0] == {"said": {"kind": "a-change", "facts": {
        "at": "pandaserver/res.py:55", "key": ".code", "made": {"said": {"kind": "made-by", "facts": {"via": [f"{g}:91"]}}}}}}
    assert facts["returned"] == ""


_GEN7 = """
class Res:
    def set_fatal(self):
        self.code = 2


class Plugin:
    def __init__(self):
        self.result = Res()

    def run(self):
        self.result.set_fatal()


class Gen:
    def go(self):
        self.plugin = Plugin()
        self.plugin.run()
        self.plugin.result.set_fatal()
        self.out = self.plugin.result
        errors = {}
        errors["a"] = "x"
        if self.out.code and errors["a"]:
            fail()
"""


def test_g4_a_change_is_on_the_way_it_changes_at_the_level_of_the_object_it_changes(tmp_path):
    # case A's add_result (out/gm/gm5-design.md G4): Gen.plugin's record names the plugin's set_fatal (inner .result,
    # through run) and Gen's own set_fatal on the plugin's result; Plugin.result's record names the first as the store
    # it makes.  The way reads .result on in Plugin.result's record: Gen.plugin's changes of what .result holds are that
    # object's, the first one change with the record's own, the second (no record of Plugin's holds it) kept, its key not
    # said.  A local's changes are on its own end.
    (_corpus(tmp_path) / "pandaserver" / "g7.py").write_text(_GEN7)
    lines = _GEN7.splitlines()
    ln = {k: lines.index(t) + 1 for k, t in {
        "code": "        self.code = 2", "pres": "        self.result = Res()", "run": "        self.result.set_fatal()",
        "plugin": "        self.plugin = Plugin()", "go_run": "        self.plugin.run()",
        "own": "        self.plugin.result.set_fatal()", "out": "        self.out = self.plugin.result",
        "errors": "        errors = {}", "store": '        errors["a"] = "x"', "fail": "            fail()"}.items()}
    test = ln["fail"] - 1
    out = _ok(_run(tmp_path, f"""
        import json
        from types import SimpleNamespace as N
        from bamboo.codemap.panda import pathcond
        from bamboo.querier import mapdump, rerun
        from bamboo.querier import rows as R
        from bamboo.querier import sources as SR

        R.ROWS_FROM = "walk"
        g = "pandaserver/g7.py"
        LN = {ln!r}
        G, P, ARM = f"{{g}}::Gen", f"{{g}}::Plugin", f"{{g}}::go"
        src = lambda kind, name, line, **kw: N(kind=kind, name=name, line=line, changes=kw.pop("changes", []),
                                               site_owner=kw.pop("site_owner", ""), site_line=None, **kw)
        bind = lambda line, owner, *sources: N(at=f"{{g}}:{{LN[line]}}", owner=f"{{g}}::{{owner}}", how="assign",
                                               sources=list(sources), stops=[])
        change = lambda at, kind, key, *via: N(at=at, kind=kind, key=key, what="", via=[f"{{g}}:{{LN[v]}}" for v in via], value=[])
        mapdump._FRAGMENT[:] = [N(
            guard_froms=[
                N(owner=ARM, test_line={test}, name="self.out", sources=[src("self_field", "self.out", {test}, site_owner=G)],
                  stops=[]),
                N(owner=ARM, test_line={test}, name="errors", stops=[], sources=[
                    src("unknown", "{{}}", LN["errors"], changes=[change(f"{{g}}:{{LN['store']}}", "store", "*")])]),
            ],
            runnables=[],
            field_froms=[
                N(owner=G, field="out", changes=[], stops=[], bindings=[
                    bind("out", "go", src("self_field", "self.plugin", LN["out"], site_owner=G, steps=[".result"]))]),
                N(owner=G, field="plugin", stops=[], bindings=[
                    bind("plugin", "go", src("object", "Plugin()", LN["plugin"], site_owner=P))],
                  changes=[change(f"{{g}}:{{LN['run']}}", "inner", ".result", "go_run"),
                           change(f"{{g}}:{{LN['own']}}", "inner", ".result")]),
                N(owner=P, field="result", stops=[], bindings=[bind("pres", "__init__", src("call", "Res", LN["pres"]))],
                  changes=[change(f"{{g}}:{{LN['code']}}", "store", ".code", "run")]),
            ])]
        fn = R.function(g, "go", within=[LN["fail"]])
        stmt = R.statement_at(fn.node, LN["fail"])
        guards = pathcond.arm_condition(stmt)
        terms = SR.terms(fn, ARM, list(zip(guards, rerun._guard_ifs(fn, stmt, guards))))
        print(json.dumps([[t.term, t.ended, [[e.end, e.why, e.at, list(e.via), [[c.at, c.key, list(c.made_by)] for c in e.changes]]
                                              for e in t.ends]] for t in terms]))
    """))
    got = json.loads(out.splitlines()[-1])
    at = {k: f"pandaserver/g7.py:{v}" for k, v in ln.items()}
    assert got == [
        # the local's own end kept, its change on it
        ["errors", False, [["stop", "unknown", at["errors"], [], [[at["store"], "*", []]]]]],
        # one way, the Res() built in Plugin.__init__, with the store set_fatal makes through run, and Gen's own call on
        # the plugin's result, which no record of Plugin's holds
        ["self.out", False, [["stop", "call", at["pres"], [at["out"], at["plugin"]],
                              [[at["code"], ".code", [at["run"]]], [at["own"], "*", []]]]]],
    ], got


_GEN8 = """
class Res:
    fatal = 2
    temporary = 1

    def __init__(self):
        self.code = None

    def set_fatal(self):
        self.code = self.fatal

    def set_temporary(self):
        self.code = self.temporary

    def ok(self):
        return self.code == 0


class Plugin:
    def __init__(self, logger):
        self.logger = logger
        self.result = Res()

    def register(self):
        self.logger.debug("register")
        is_fatal = False
        try:
            work()
        except Exception as e:
            is_fatal = e.fatal
        if is_fatal:
            self.result.set_fatal()
        else:
            self.result.set_temporary()

    def subscribe(self):
        self.logger.debug("subscribe")
        self.result.set_fatal()


class Gen:
    def execute(self):
        self.plugin = Plugin(self.logger)
        self.plugin.register()
        self.out = self.plugin.result
        self.logger.debug(f"plugin done with {self.out.code}")

    def finalize(self):
        self.execute()
        if self.out is None or not self.out.ok():
            fail()
"""


def test_s4c_reads_the_change_a_runs_print_and_must_rows_choose_as_an_arm(tmp_path):
    # case A's add_result after G4 (out/gm/gm5-design.md S4c): the way to self.out holds three stores of .code, through
    # register's two arms and subscribe.  V: the run printed 2 of self.out.code after its binding, so set_temporary's 1
    # is not what the read reads.  H2: register's rows are in the run, subscribe's are not, so the change register made
    # is the one; read as an arm, its guard's term is bound in an except handler (S3's).  Where subscribe's rows are in
    # the run too, two are left and named.
    (_corpus(tmp_path) / "pandaserver" / "g8.py").write_text(_GEN8)
    lines = _GEN8.splitlines()
    ln = {k: lines.index(t) + 1 for k, t in {
        "fatal": "        self.code = self.fatal", "temp": "        self.code = self.temporary",
        "rfatal": "    fatal = 2", "rtemp": "    temporary = 1", "pres": "        self.result = Res()",
        "init_fatal": "        is_fatal = False", "except": "        except Exception as e:", "in_handler": "            is_fatal = e.fatal",
        "if_fatal": "        if is_fatal:", "reg_fatal": "            self.result.set_fatal()",
        "reg_temp": "            self.result.set_temporary()", "sub_fatal": "        self.result.set_fatal()",
        "plugin": "        self.plugin = Plugin(self.logger)", "out": "        self.out = self.plugin.result",
        "fail": "            fail()"}.items()}
    ln["test"] = ln["fail"] - 1
    stamp = "2026-01-01 00:00:0{},000 panda.log.X: DEBUG    <PandaID=1> {}\n"
    for name, text in {"one": ["register", "plugin done with 2"], "two": ["register", "subscribe", "plugin done with 2"]}.items():
        (tmp_path / f"{name}.log").write_text("".join(stamp.format(n, t) for n, t in enumerate(text)))
    out = _ok(_run(tmp_path, f"""
        import json
        from dataclasses import replace
        from types import SimpleNamespace as N
        from bamboo.codemap.panda import pathcond
        from bamboo.querier import mapdump, rerun
        from bamboo.querier import render as T
        from bamboo.querier import rows as R
        from bamboo.querier import sources as SR
        from bamboo.querier.records import read_log, runs_for

        R.ROWS_FROM = "walk"
        g = "pandaserver/g8.py"
        LN = {ln!r}
        RES = f"{{g}}::Res"
        src = lambda kind, name, line, **kw: N(kind=kind, name=name, line=line, changes=[], site_owner=kw.pop("site_owner", ""),
                                               site_line=None, paths=[], steps=[], through=[], route=[], **kw)
        runs = {{name: runs_for(read_log(f"{{name}}.log"), "1")[0] for name in ("one", "two")}}
        # the map's prints of register and subscribe land in the runs' log; set_fatal prints nothing
        printing = {{f"{{g}}::{{f}}": N(statements=[N(log_files=[runs["one"].log.name, runs["two"].log.name],
                                                       log_files_basis="logger")]) for f in ("register", "subscribe")}}
        R._prints_of = lambda owner, lines=(): printing.get(owner)
        klass = lambda field, line, value: N(owner=RES, field=field, changes=[], stops=[], bindings=[N(
            at=f"{{g}}:{{LN[line]}}", owner=RES, how="class", stops=[], sources=[src("constant", value, LN[line])])])
        mapdump._FRAGMENT[:] = [N(
            guard_froms=[N(owner=f"{{g}}::register", test_line=LN["if_fatal"], name="is_fatal", stops=[], sources=[
                src("constant", "False", LN["init_fatal"]), src("attribute", "e.fatal", LN["in_handler"])])],
            runnables=[N(owner=f"{{g}}::ok", line=0, bound=[RES], asked=["condition"], asked_at=[f"{{g}}:{{LN['test']}}"],
                         runnable=True, reads=["self.code"], stops=[])],
            field_froms=[klass("fatal", "rfatal", "2"), klass("temporary", "rtemp", "1")], function_prints=[])]
        store = lambda at, field, via: SR.Changed(
            f"{{g}}:{{LN[at]}}", "store", ".code", (f"{{g}}:{{LN[via]}}",),
            N(kind="store", value=[src("self_field", f"self.{{field}}", LN[at], site_owner=RES)], value_stops=[]),
            f"{{g}}:{{LN['out']}}")
        way = SR.End("self.out", "stop", "call", f"{{g}}:{{LN['pres']}}", via=(f"{{g}}:{{LN['out']}}", f"{{g}}:{{LN['plugin']}}"),
                     changes=(store("fatal", "fatal", "reg_fatal"), store("temp", "temporary", "reg_temp"),
                              store("fatal", "fatal", "sub_fatal")))
        fn = R.function(g, "finalize", within=[LN["fail"]])
        site = R.statement_at(fn.node, LN["fail"])
        guards = pathcond.arm_condition(site)
        term = SR.Term(guards[0], "self.out", (way,), LN["test"])
        got = {{"keys": sorted(SR.read_keys(R.statement_at(fn.node, LN["test"]).test, "self.out", g, LN["test"])),
               "stored": [SR.stored(c.record) for c in way.changes]}}
        for name, run in runs.items():
            r = object.__new__(rerun.Rerun)
            r.lines, r.mode, r.decide = [], "outside", "positive"
            printed = {{}}
            r._held_on(fn, site, guards, run, printed)
            t = r._s4c(term, fn, printed, run, (run.first[:19], run.last[:19]), frozenset(), "  ")
            e = t.ends[0]
            m = e.made
            got[name] = {{
                "left": [[c.at, list(c.made_by)] for c in e.changes],
                "made": None if m is None else [m.site, m.fn, m.how, list(m.guards),
                                                [[u.term, [[x.end, x.why, x.at, x.handler] for x in u.ends]] for u in m.terms],
                                                [m.entry.end, m.entry.why]],
                "stop": T.text(rerun._first_stop(t)), "said": T.text(rerun._way_said(e)), "lines": r.lines}}
            # one change of a part the run printed nothing of: its rows in the run say it may have run, not that it made
            # the value rather than the way's end
            lone = replace(way, changes=(replace(way.changes[0], key=".flag"),))
            u = r._s4c(SR.Term(guards[0], "self.out", (lone,), LN["test"]), fn, printed, run,
                       (run.first[:19], run.last[:19]), frozenset(), "  ")
            got[name]["lone"] = [[[c.at, c.key] for c in u.ends[0].changes], u.ends[0].made is None]
        # an arm read on no run (an attribute's, no window): S4c, which reads the run, is not asked
        r = object.__new__(rerun.Rerun)
        r.lines, r.mode, r.decide = [], "outside", "positive"
        r._s4c = lambda *a: (_ for _ in ()).throw(AssertionError("S4c read on no run"))
        rfn = R.function(g, "register", within=[LN["reg_fatal"]])
        rsite = R.statement_at(rfn.node, LN["reg_fatal"])
        rguards = pathcond.arm_condition(rsite)
        terms, _entry = r._arm_terms(rfn, f"{{g}}::register", LN["reg_fatal"], rguards,
                                     rerun._guard_ifs(rfn, rsite, rguards), {{}}, None, None)
        got["no run"] = [[u.term, len(u.ends)] for u in terms]
        print(json.dumps(got))
    """))
    got = json.loads(out.splitlines()[-1])
    assert got["no run"] == [["is_fatal", 2]]
    at = {k: f"pandaserver/g8.py:{v}" for k, v in ln.items()}
    # what ok() reads, by the map's Runnable; what each store stores, read to the class's literal
    assert got["keys"] == ["code"]
    assert got["stored"] == [["2"], ["1"], ["2"]]
    one = got["one"]
    assert any(f"V: {at['temp']} made by {at['reg_temp']} stores 1, not the '2' printed of self.out.code" in x
               for x in one["lines"]), one["lines"]
    # register's change: its rows are in the run (register's print, which both its arms pass, says it may have; none
    # says it did) and subscribe's are not, so it is left by elimination, read as an arm at its call in register;
    # is_fatal bound in the handler
    assert one["left"] == [[at["fatal"], [at["reg_fatal"]]]], one
    assert one["made"] == [at["reg_fatal"], "pandaserver/g8.py::register", "left", ["is_fatal"],
                           [["is_fatal", [["constant", "False", at["init_fatal"], 0],
                                          ["stop", "attribute", at["in_handler"], ln["except"]]]]],
                           ["stop", "no-entered-from"]], [x for x in one["lines"] if "g8.py:" in x]
    assert one["stop"] == (f"it stops at 'is_fatal' (in 'is_fatal'), in the arm that made the change at {at['reg_fatal']}, "
                           f"bound in the except handler at {ln['except']} (S3: which exception)")
    # Q7's (b): what the way's end held of .code at its start is not read (a call's end: no class), so the change is
    # not the end of .code, and the way says so
    assert one["said"] == (
        f"stop (call) at {at['pres']}, through the field bindings at {at['out']} -> {at['plugin']}, changed between it and "
        f"the read by {at['fatal']} made by {at['reg_fatal']}, left by elimination (it stores what the run printed, its "
        f"function's rows are in the run, and every other change is shown not to have run or to store another value), "
        f"but what the object held of .code at its start is not read; "
        f"read as an arm at {at['reg_fatal']}: its "
        f"terms is_fatal -> constant (False) at {at['init_fatal']}, stop (attribute) at {at['in_handler']}, in the except "
        f"handler at {ln['except']} (S3: which exception); its entry: stop (no-entered-from)")
    # subscribe ran too: two changes of 2 are left, none chosen, both named
    two = got["two"]
    assert two["left"] == [[at["fatal"], [at["reg_fatal"]]], [at["fatal"], [at["sub_fatal"]]]] and two["made"] is None, two
    assert two["stop"] == "it stops at 'self.out' (in 'self.out is None or not self.out.ok()')"
    # one change, kept, of a part not printed: not chosen, named
    assert one["lone"] == [[[at["fatal"], ".flag"]], True], one["lone"]


_GEN6 = """
class Gen:
    def report(self):
        if self.job is None and self.ok:
            fail()
"""

_GEN9 = """
from pandacommon.pandautils.PandaUtils import naive_utcnow


def caller():
    now = naive_utcnow()
    target(now)


def target(t):
    if t:
        pass


class Gen:
    def check(self):
        stamp = naive_utcnow()
        made = other()
        if stamp > made:
            fail()
"""


def test_a_way_ending_at_a_call_the_map_reads_as_a_clock_ends_at_the_clock(tmp_path):
    # S2's (b): a guard's local bound to naive_utcnow() (add_main 97's timeNow) and a parameter a caller hands from it
    # end at the clock, as a field's way does; another call stops, and so does a clock in a callee the value came back
    # from (its line is in a file the record does not name)
    (_corpus(tmp_path) / "pandaserver" / "g9.py").write_text(_GEN9)
    out = _ok(_run(tmp_path, """
        import json
        from types import SimpleNamespace as N
        from bamboo.codemap.panda import pathcond
        from bamboo.querier import mapdump, rerun
        from bamboo.querier import rows as R
        from bamboo.querier import sources as SR

        R.ROWS_FROM = "walk"
        g = "pandaserver/g9.py"
        src = lambda name, line, route=(): N(kind="call", name=name, line=line, route=list(route), changes=[], site_owner="",
                                             site_line=None, steps=[], through=[], paths=[])
        mapdump._FRAGMENT[:] = [N(
            guard_froms=[N(owner=f"{g}::check", test_line=19, name="stamp", stops=[], sources=[src("naive_utcnow", 17)]),
                         N(owner=f"{g}::check", test_line=19, name="made", stops=[], sources=[src("other", 18)])],
            parameter_froms=[N(owner=f"{g}::target", line=10, parameter="t", stops=[], sources=[
                src("naive_utcnow", 6, [f"{g}:7"]), src("naive_utcnow", 6, [f"{g}:30", f"{g}:7"])])],
            entered_froms=[], runnables=[], field_froms=[])]
        fn = R.function(g, "check", within=[20])
        stmt = R.statement_at(fn.node, 20)
        guards = pathcond.arm_condition(stmt)
        terms = SR.terms(fn, f"{g}::check", list(zip(guards, rerun._guard_ifs(fn, stmt, guards))))
        ways = SR.param_ways(f"{g}::target", "t")
        print(json.dumps({"terms": [[t.term, [[e.end, e.why, e.at] for e in t.ends], t.ended] for t in terms],
                          "ways": [[w.end, w.why, list(w.route)] for w in ways]}))
    """))
    got = json.loads(out.splitlines()[-1])
    g = "pandaserver/g9.py"
    assert got["terms"] == [["made", [["stop", "call", f"{g}:18"]], False],
                            ["stamp", [["clock", "naive_utcnow", f"{g}:17"]], True]]
    assert got["ways"] == [["clock", "naive_utcnow", [f"{g}:7"]], ["stop", "call", [f"{g}:30", f"{g}:7"]]]


_GEN10 = """
from pandacommon.pandautils.PandaUtils import naive_utcnow


class Thread:
    def run(self):
        timeNow = naive_utcnow()
        while True:
            one = self.reports.pop()
            if not one:
                break
            stamp = one[1]
            try:
                modTime = stamp
                if (timeNow - modTime) > 24:
                    flag = False
                else:
                    flag = True
                gen = Gen(one, flag)
            except Exception:
                pass


class Gen:
    def __init__(self, job, ignore):
        self.ignore = ignore
"""


def test_s2a_reads_the_if_telling_apart_the_constants_one_call_hands(tmp_path):
    # add_main 127 (S2's (a)): ignoreTmpError is False at 130 and True at 134, both handed at 136.  Their caller's
    # run is asked which binding ran (Q5); where it does not tell, the if they sit apart in is read by the map's
    # records -- timeNow a clock (b), modTime from the pop -- the stop goes down to its first term that does not end,
    # and the split says what the run answered.  The guard both bindings share ('one') is not read.  No record at the
    # if: it stops there.  Not read where a constant is bound in a callee it came back from, or where another way is
    # left beside the constants.
    (_corpus(tmp_path) / "pandaserver" / "g10.py").write_text(_GEN10)
    out = _ok(_run(tmp_path, """
        import json
        from types import SimpleNamespace as N
        from bamboo.querier import mapdump, rerun
        from bamboo.querier import render as T
        from bamboo.querier import rows as R
        from bamboo.querier import sources as SR

        R.ROWS_FROM = "walk"
        g = "pandaserver/g10.py"
        INIT = f"{g}::__init__"
        src = lambda kind, name, line, route=(): N(kind=kind, name=name, line=line, route=list(route), changes=[],
                                                   site_owner="", site_line=None, steps=[], through=[], paths=[])
        flag = [src("constant", "False", 16, [f"{g}:19"]), src("constant", "True", 18, [f"{g}:19"])]
        tested = [N(owner=f"{g}::run", test_line=15, name="timeNow", stops=[], sources=[src("call", "naive_utcnow", 7)]),
                  N(owner=f"{g}::run", test_line=15, name="modTime", stops=[], sources=[src("call", "self.reports.pop", 9)]),
                  N(owner=f"{g}::run", test_line=10, name="one", stops=[], sources=[src("call", "self.reports.pop", 9)])]
        got = {}

        def walk(name, sources, guards, shown=None, no_log=False, answer="none-shown", ask=True):
            mapdump._FRAGMENT[:] = [N(guard_froms=guards, runnables=[], field_froms=[], entered_froms=[],
                                      parameter_froms=[N(owner=INIT, parameter="ignore", line=25, stops=[], sources=sources)])]
            SR._calls_in.cache_clear()
            SR._index.cache_clear()
            r = object.__new__(rerun.Rerun)
            r.lines = []
            asked = []
            prints_of = R._prints_of
            if no_log:
                # Q5 itself, with no log of the caller to hand: nothing asked
                R._prints_of = lambda owner, lines: None
                r.files_of = lambda owner, today, lines: []
            else:
                # Q5: the caller's run asked which binding ran -- here it shows the one at *shown*, or tells none
                # (*answer*)
                def binding_shown(consts, window, machine):
                    asked.append([w.bound_at for w in consts])
                    way = next((w for w in consts if w.bound_at == shown), None)
                    return way, "" if way is not None else answer
                r._binding_shown = binding_shown
            t = SR.Term("g", "self.ignore", (SR.End("self.ignore", SR.STOP, SR.PARAMETER, f"{g}:26", param="ignore", of=INIT),))
            try:
                t = r._with_callers(t, "the arm's function", f"{g}::Gen.report", N(machine="m1"), ("a", "b"), ask=ask)
            finally:
                R._prints_of = prints_of
            e = t.ends[0]
            s = e.split
            got[name] = {"end": [e.end, e.why, e.at, list(e.route)], "lines": r.lines, "stop": T.text(rerun._first_stop(t)),
                         "said": T.text(rerun._way_said(e)), "asked": asked, "split_asked": None if s is None else s.asked,
                         "split": None if s is None else [s.call, s.fn, list(s.ifs), list(s.guards),
                                                          [[u.term, [[x.end, x.why, x.at] for x in u.ends]] for u in s.terms]]}

        walk("shown", flag, tested, shown=18)
        walk("record", flag, tested)
        walk("no log", flag, tested, no_log=True)
        walk("unanswered", flag, tested, answer="unexamined")
        walk("several", flag, tested, answer="several-shown")
        walk("not asked", flag, tested, ask=False)
        # a part of each constant: no literal left (Q2), Q5 not reached
        part = [src("constant", "False", 16, [f"{g}:19"]), src("constant", "True", 18, [f"{g}:19"])]
        for s in part:
            s.steps = ["[0]"]
        walk("through", part, tested)
        walk("no record", flag, tested[2:])
        walk("in a callee", [flag[0], src("constant", "True", 3, [f"{g}:19", f"{g}:19"])], tested)
        walk("another way", [*flag, src("call", "other", 12, [f"{g}:19"])], tested)
        print(json.dumps(got))
    """))
    got = json.loads(out.splitlines()[-1])
    g = "pandaserver/g10.py"
    test = "timeNow - modTime > 24"  # as ast.unparse spells the if's test
    # Q5: the caller's run asked first; where it shows one binding, the constant it binds is the end, no split read
    assert got["shown"]["asked"] == [[16, 18]] and got["record"]["asked"] == [[16, 18]]
    assert got["shown"]["end"] == ["constant", "True", f"{g}:18", [f"{g}:19"]] and got["shown"]["split"] is None
    assert any(f"all come by {g}:19: which binding ran, by its caller's run (Q5)" in x for x in got["shown"]["lines"])
    # no log of the caller: said so, nothing asked, the if read as without Q5
    assert got["no log"]["asked"] == [] and got["no log"]["split"] == got["record"]["split"]
    assert any(f"all come by {g}:19: which binding ran, by its caller's run (Q5)" in x for x in got["no log"]["lines"])
    assert any(x.strip() == f"Q5: {g}:19 (run): no log of it to hand (none): its caller's run not asked"
               for x in got["no log"]["lines"]), got["no log"]["lines"]
    # no literal left: each way read where it ends (Q2), not asked, and not said to be
    assert got["through"]["asked"] == [] and got["through"]["end"][:2] == ["constant", "constant-through"], got["through"]
    assert any(f"all come by {g}:19: its caller's run not asked" in x for x in got["through"]["lines"]), got["through"]["lines"]
    assert not any("Q5" in x for x in got["through"]["lines"]), got["through"]["lines"]
    assert got["record"]["end"][:3] == ["stop", "parameter-unchosen", ""]
    assert got["record"]["split"] == [f"{g}:19", f"{g}::run", [f"{g}:15"], [test], [
        ["modTime", [["stop", "call", f"{g}:9"]]], ["timeNow", [["clock", "naive_utcnow", f"{g}:7"]]]]]
    assert any(f"told apart by the if at {g}:15 (run; the bindings at {g}:16, {g}:18)" in x for x in got["record"]["lines"])
    assert got["record"]["stop"] == (f"it stops at 'modTime' (in {test!r}), in the if at {g}:15 that tells apart the "
                                     f"constants {g}:19 hands")
    # the split says what the caller's run said of which binding ran (Q5), or that it was not asked
    said = (f"stop (parameter-unchosen), the constants {g}:19 hands told apart by the if at {g}:15 ({test!r}), read by the "
            "map's records ({}): its terms modTime -> stop (call) at")
    for name, asked, text in (
            ("record", "none-shown", "its caller's run asked which binding ran (Q5): it shows none"),
            ("unanswered", "unexamined", "its caller's run asked which binding ran (Q5), not answered"),
            ("several", "several-shown", "its caller's run asked which binding ran (Q5): it shows more than one"),
            ("no log", "no-log", "its caller's run not asked which binding ran (Q5): no log of it to hand"),
            ("not asked", "", "its run not asked")):
        assert got[name]["split_asked"] == asked and got[name]["split"] == got["record"]["split"], name
        assert got[name]["said"].startswith(said.format(text)), (name, got[name]["said"])
        assert any(x.strip().endswith(f"read by the map's records: {text}") for x in got[name]["lines"]), name
    assert got["not asked"]["asked"] == [] and not any("Q5" in x for x in got["not asked"]["lines"])
    assert got["no record"]["split"][4] == [[test, [["stop", "no-record", f"{g}:15"]]]]
    assert got["no record"]["stop"].startswith(f"it stops at {test!r} (in {test!r}), in the if at {g}:15 ")
    for name in ("in a callee", "another way"):
        assert got[name]["end"][:2] == ["stop", "parameter-unchosen"] and got[name]["split"] is None, name
        assert got[name]["stop"] == "it stops at 'self.ignore' (in 'g')", name


_Q5_ADDM = """
import datetime

from pandacommon.pandalogger.LogWrapper import LogWrapper
from pandacommon.pandalogger.PandaLogger import PandaLogger
from pandacommon.pandautils.PandaUtils import naive_utcnow

_logger = PandaLogger().getLogger("add_main")


def main(taskBuffer=None):
    tmpLog = LogWrapper(_logger, "<pid=1>")
    tmpLog.debug("start")

    class AdderThread:
        def run(self):
            timeNow = naive_utcnow()
            uniq_pid = self.get_pid()
            tmpLog.debug(f"pid={uniq_pid} : run")
            while True:
                one_jor = self.job_output_reports.pop()
                if not one_jor:
                    break
                panda_id, job_status, attempt_nr, time_stamp = one_jor
                token_str = f"pid={uniq_pid} : job={panda_id}.{attempt_nr}"
                tmpLog.debug(f"{token_str} to lock timestamp={time_stamp}")
                try:
                    modTime = time_stamp
                    if (timeNow - modTime) > datetime.timedelta(hours=24):
                        tmpLog.debug(f"{token_str} last add st={job_status}")
                        ignoreTmpError = False
                    else:
                        tmpLog.debug(f"{token_str} add st={job_status}")
                        ignoreTmpError = True
                    adder_gen = AdderGen(taskBuffer, panda_id, job_status, attempt_nr, ignore_tmp_error=ignoreTmpError)
                    adder_gen.run()
                    tmpLog.debug(f"{token_str} done")
                except Exception as e:
                    tmpLog.error(f"pid={uniq_pid} : failed {e}")
            tmpLog.debug(f"pid={uniq_pid} : processed")

        def each(self, jobs):
            for panda_id in jobs:
                token_str = f"each job={panda_id}."
                if panda_id > 0:
                    tmpLog.debug(f"{token_str} kept")
                    kept = True
                else:
                    tmpLog.debug(f"{token_str} dropped")
                    kept = False
"""


def test_q5_a_loop_turn_printed_for_another_element_shows_nothing_of_the_keys_binding(tmp_path):
    # add_main 106-134 (out/gm/qk-design.md Q5): one mark for every turn of the while that pops a job; 129 'last add
    # st=' printed for another job made 130 positive beside 134.  The name the key's records print the key as
    # (panda_id, through token_str) is bound to it, and a row printed only for another value is not the key's turn's.
    # A for loop's element through a template is read alike; both turns of the key itself still leave both.
    (_corpus(tmp_path) / "pandaserver" / "addm.py").write_text(_Q5_ADDM)
    line = "2026-01-01 {} panda.log.add_main: DEBUG    <pid=1> {}\n"
    def turn(j: int, last: bool) -> list[str]:
        return [f"pid=9 : job={j}.1 to lock timestamp=x", f"pid=9 : job={j}.1 {'last add' if last else 'add'} st=failed",
                f"pid=9 : job={j}.1 done"]

    whole =["pid=9 : run", *turn(777, False), *turn(555, True), "pid=9 : processed", "each job=555. kept", "each job=777. dropped"]
    logs = {
        "whole": [line.format(f"13:00:00,{i:03d}", t) for i, t in enumerate(whole)],
        # the key's own lines, from two cycles, one through each binding
        "both": ["# extract by 777: lines holding it\n"] + [line.format(f"12:00:00,{i:03d}", t) for i, t in enumerate(turn(777, True))]
        + [line.format(f"13:00:00,{i:03d}", t) for i, t in enumerate(turn(777, False))],
    }
    for name, said in logs.items():
        (tmp_path / name).mkdir()
        (tmp_path / name / "panda-add_main.log").write_text("".join(said))
    out = _ok(_run(tmp_path, """
        import json
        from bamboo.querier import evaluate as E
        from bamboo.querier import rows as R
        from bamboo.querier.records import read_log, runs_for
        from bamboo.querier.which_ran import Candidate, judge, key_env

        R.ROWS_FROM = "walk"
        # a guard's call (datetime.timedelta) is read through the pure functions, which E fills from pandaserver at
        # the first call; the dev install (CI) has no pandaserver, and none of them is read here
        E.PURE.setdefault("<no pure function is read here>", None)
        f = "pandaserver/addm.py"
        text = open("corpus/pandaserver/addm.py").read().splitlines()
        at_line = lambda s: next(i for i, x in enumerate(text, 1) if s in x)
        no, yes, kept, dropped = (at_line(s) for s in ("ignoreTmpError = False", "ignoreTmpError = True", "kept = True", "kept = False"))
        at = ("2026-01-01 13:00:00", "2026-01-01 13:00:00")
        log = "panda-add_main.log"
        got = {"lines": [no, yes, kept, dropped]}
        for name, func, lines in (("whole", "run", [no, yes]), ("both", "run", [no, yes]), ("for", "each", [kept, dropped])):
            runs = {log: runs_for(read_log(f"{'whole' if name == 'for' else name}/{log}"), "777")}
            c = Candidate(f, func, lines, [log])
            env = key_env(c, runs, "777")
            got[name] = {"env": sorted(env.values) if env else None,
                         "with": [[v.line, v.kind, v.reasons] for v in judge([c], runs, window=at, env=env).verdicts],
                         "without": [[v.line, v.kind] for v in judge([c], runs, window=at).verdicts]}
        print(json.dumps(got))
    """))
    got = json.loads(out.splitlines()[-1])
    no, yes, kept, dropped = got["lines"]
    for name in ("whole", "both", "for"):
        assert got[name]["env"] == ["panda_id"], (name, got[name])
    # today's judge, no element bound: both bindings positive
    assert got["whole"]["without"] == [[no, "positive"], [yes, "positive"]]
    assert [x[:2] for x in got["whole"]["with"]] == [[no, "excluded"], [yes, "positive"]], got["whole"]
    assert any("printed only for other values of panda_id" in r and "not '777'" in r for r in got["whole"]["with"][0][2])
    # the key's own turns through both: still both
    assert [x[:2] for x in got["both"]["with"]] == [[no, "positive"], [yes, "positive"]], got["both"]
    # a for loop's element through a template
    assert got["for"]["without"] == [[kept, "positive"], [dropped, "positive"]]
    assert [x[:2] for x in got["for"]["with"]] == [[kept, "excluded"], [dropped, "positive"]], got["for"]


def test_a_field_record_with_no_way_stops_its_term_rather_than_drops_it(tmp_path):
    # the map's PCS build made field records with no binding, change or stop: such a term said nothing, and the others
    # ending would read as every value ending
    (_corpus(tmp_path) / "pandaserver" / "g6.py").write_text(_GEN6)
    out = _ok(_run(tmp_path, """
        import json
        from types import SimpleNamespace as N
        from bamboo.codemap.panda import pathcond
        from bamboo.querier import mapdump, rerun
        from bamboo.querier import rows as R
        from bamboo.querier import sources as SR

        R.ROWS_FROM = "walk"
        g = "pandaserver/g6.py"
        G, ARM = f"{g}::Gen", f"{g}::report"
        src = lambda name: N(kind="self_field", name=name, line=4, changes=[], site_owner=G, site_line=None)
        mapdump._FRAGMENT[:] = [N(
            guard_froms=[N(owner=ARM, test_line=4, name="self.job", sources=[src("self.job")], stops=[]),
                         N(owner=ARM, test_line=4, name="self.ok", sources=[src("self.ok")], stops=[])],
            runnables=[],
            field_froms=[N(owner=G, field="job", bindings=[], changes=[], stops=[]),
                         N(owner=G, field="ok", changes=[], stops=[], bindings=[
                             N(at=f"{g}:9", owner=f"{g}::__init__", how="assign", stops=[],
                               sources=[N(kind="constant", name="True", line=9, changes=[], site_owner="", site_line=None)])])])]
        fn = R.function(g, "report", within=[5])
        stmt = R.statement_at(fn.node, 5)
        guards = pathcond.arm_condition(stmt)
        terms = SR.terms(fn, ARM, list(zip(guards, rerun._guard_ifs(fn, stmt, guards))))
        print(json.dumps([[t.term, t.ended, [[e.end, e.why, e.at] for e in t.ends]] for t in terms]))
    """))
    assert json.loads(out.splitlines()[-1]) == [
        ["self.job", False, [["stop", "no-way", "pandaserver/g6.py::Gen.job"]]],
        ["self.ok", True, [["constant", "True", "pandaserver/g6.py:9"]]]]


def test_a_guards_local_item_of_a_literal_ends_at_the_literal_not_as_its_value(tmp_path):
    # register_files' attempt_number (for ... in range(max_attempt), max_attempt = 3): the map's source is the constant
    # 3 with a step and range() through; the way ends at the literal, its value not 3 (as a parameter's way says it)
    (_corpus(tmp_path) / "pandaserver" / "g6.py").write_text(_GEN6)
    out = _ok(_run(tmp_path, """
        import json
        from types import SimpleNamespace as N
        from bamboo.codemap.panda import pathcond
        from bamboo.querier import mapdump, rerun
        from bamboo.querier import rows as R
        from bamboo.querier import sources as SR

        R.ROWS_FROM = "walk"
        g = "pandaserver/g6.py"
        ARM = f"{g}::report"
        src = lambda name, **kw: N(kind="constant", name=name, line=9, changes=[], site_owner="", site_line=None, **kw)
        mapdump._FRAGMENT[:] = [N(
            guard_froms=[N(owner=ARM, test_line=4, name="self.job", stops=[], sources=[src("None", steps=[], through=[])]),
                         N(owner=ARM, test_line=4, name="self.ok", stops=[], sources=[
                             src("3", steps=["[*]"], through=[N(call="range", at=f"{g}:10", gives="computed")])])],
            runnables=[], field_froms=[])]
        fn = R.function(g, "report", within=[5])
        stmt = R.statement_at(fn.node, 5)
        guards = pathcond.arm_condition(stmt)
        terms = SR.terms(fn, ARM, list(zip(guards, rerun._guard_ifs(fn, stmt, guards))))
        print(json.dumps([[t.term, [[e.end, e.why] for e in t.ends]] for t in terms]))
    """))
    assert json.loads(out.splitlines()[-1]) == [
        ["self.job", [["constant", "None"]]], ["self.ok", [["constant", "constant-through"]]]]


def test_s4a_reads_a_field_bindings_parameter_up_its_callers_without_the_arms_prints(tmp_path):
    # AdderGen.__init__'s job_status: handed at two calls; the arm's entry prints say nothing of __init__'s parameter
    out = _ok(_run(tmp_path, """
        import json
        from types import SimpleNamespace as N
        from bamboo.querier import mapdump, rerun
        from bamboo.querier import sources as SR

        INIT = "pandaserver/g.py::__init__"
        src = lambda kind, name, route: N(kind=kind, name=name, route=list(route), site_owner="", paths=[])
        mapdump._FRAGMENT[:] = [N(parameter_froms=[
            N(owner=INIT, parameter="status", line=5, stops=[],
              sources=[src("constant", "'failed'", ["pandaserver/a.py:10"]), src("constant", "'finished'", ["pandaserver/a.py:20"])]),
        ], entered_froms=[])]
        r = object.__new__(rerun.Rerun)
        r.lines = []

        def no_print(*a):
            raise AssertionError("the arm's prints read for another method's parameter")

        r._entry_value = no_print
        r._call_shown = lambda site, window, machine, why="": site == "pandaserver/a.py:20"
        t = SR.Term("g", "self.job_status", (
            SR.End("self.job_status", SR.STOP, SR.PARAMETER, "pandaserver/g.py:7", param="status", of=INIT,
                   via=("pandaserver/g.py:30",)),
            SR.End("self.job_status", SR.CONSTANT, "'failed'", "pandaserver/g.py:40")))
        got = r._with_callers(t, "the arm's function", "pandaserver/g.py::report", N(machine="m1"), ("a", "b"))
        print(json.dumps({"ends": [[e.end, e.why, e.at, list(e.via)] for e in got.ends], "said": str(rerun._way_said(got.ends[0])),
                          "lines": r.lines}))
    """))
    got = json.loads(out.splitlines()[-1])
    # both constants are left (no print read); the one a caller's run shows is the way, the field's binding kept
    assert got["ends"] == [["constant", "'finished'", "pandaserver/a.py:20", ["pandaserver/g.py:30"]],
                           ["constant", "'failed'", "pandaserver/g.py:40", []]]
    assert got["said"] == "constant ('finished') at pandaserver/a.py:20, through the field bindings at pandaserver/g.py:30"
    assert "pandaserver/g.py::__init__ is not the arm's: no print read" in got["lines"][0]


_GEN2 = """
class Res:
    def set_fatal(self):
        self.code = 2


class Gen:
    def __init__(self, status):
        self.job = None
        self.status = status
        self.result = None

    def run(self):
        self.job = fetch()
        self.report()

    def other(self):
        self.job = fetch2()

    def plug(self):
        try:
            self.result = Res()
            self.result.set_fatal()
            log("done")
        except Exception:
            log("failed")
            self.result = None

    def report(self):
        if self.job is None:
            return
        self.result = None
        self.plug()
        if self.status == "x" and self.result.ok():
            self.job.jobStatus = "failed"
"""


def test_s4b_takes_the_last_binding_on_the_runs_path_by_the_callers_chain_callees_and_init(tmp_path):
    # AdderGen.process_job_report's reads (out/gm/s4b-order.md): self.job bound by run before its call (a), not by a
    # method off the path (d); self.add_result by execute_plugin's try, not its handler (b, the run's prints); a field
    # only __init__ binds (c)
    (_corpus(tmp_path) / "pandaserver" / "g2.py").write_text(_GEN2)
    lines = _GEN2.splitlines()
    ln = {t: lines.index(t) + 1 for t in (
        "        self.job = None", "        self.status = status", "        self.result = None", "        self.job = fetch()",
        "        self.report()", "        self.job = fetch2()", "            self.result = Res()", '            log("done")',
        "            self.result.set_fatal()",
        "            self.result = None", "        self.plug()", "        if self.job is None:",
        '        if self.status == "x" and self.result.ok():')}
    rep_none = [i + 1 for i, t in enumerate(lines) if t == "        self.result = None"][1]
    out = _ok(_run(tmp_path, f"""
        import json
        from types import SimpleNamespace as N
        from bamboo.querier import field_order as FO, mapdump
        from bamboo.querier import sources as SR

        F = "pandaserver/g2.py"
        LN = {ln!r}
        b = lambda line, owner, *src: N(at=f"{{F}}:{{line}}", owner=f"{{F}}::{{owner}}", how="assign", sources=list(src), stops=[])
        call = lambda to, line: N(to=f"{{F}}::Gen.{{to}}", lines=[line], line=line, thread=False)
        reach = lambda extra=(), holes=(): [
            N(owner=f"{{F}}::Gen.run", calls=[call("report", LN["        self.report()"])], holes=[]),
            N(owner=f"{{F}}::Gen.report", calls=[call("plug", LN["        self.plug()"])], holes=list(holes)),
            N(owner=f"{{F}}::Gen.plug", calls=[], holes=[]),
            N(owner=f"{{F}}::Gen.__init__", calls=[], holes=[]),
            N(owner=f"{{F}}::Gen.other", calls=list(extra), holes=[]),
        ]
        G = f"{{F}}::Gen"
        ffs = [
            N(owner=G, field="job", changes=[], stops=[], bindings=[
                b(LN["        self.job = None"], "__init__"),
                # what fetch() returns, its jobStatus changed on the way
                b(LN["        self.job = fetch()"], "run", N(kind="call", name="fetch", line=LN["        self.job = fetch()"],
                                                           route=[], site_owner="", steps=[], paths=[], through=[],
                                                           changes=[N(at=f"{{F}}:99", kind="store", key=".jobStatus", what="", via=[])])),
                b(LN["        self.job = fetch2()"], "other")]),
            N(owner=G, field="status", changes=[], stops=[], bindings=[b(LN["        self.status = status"], "__init__")]),
            N(owner=G, field="result", stops=[], bindings=[
                b(LN["        self.result = None"], "__init__"), b(LN["            self.result = Res()"], "plug"),
                b(LN["            self.result = None"], "plug"), b({rep_none}, "report")],
              # set_fatal's store, made through plug's self.result
              changes=[N(at=f"{{F}}:4", kind="store", key=".code", what="", via=[f"{{F}}:{{LN['            self.result.set_fatal()']}}"])]),
        ]
        got = {{}}

        def pick(name, field, test, keys, run=None):
            ff = SR._field_froms()[(G, field)]
            c = FO.choose(ff, F, LN[test], run, "map", keys)
            got[name] = [[x.at.rpartition(":")[2] for x in c.bindings], [x.at.rpartition(":")[2] for x in c.changes],
                         [[at.rpartition(":")[2], why] for at, why in c.dropped], c.entry]

        def install(**kw):
            mapdump._FRAGMENT[:] = [N(field_froms=ffs, entered_froms=[], parameter_froms=[], handler_marks=[], **kw)]
            for cached in (SR._field_index, SR._calls_in, FO._reach_index, FO._closure_in, FO._callers_in):
                cached.cache_clear()

        install(reach_graph=reach())
        pick("job", "job", "        if self.job is None:", frozenset())
        pick("result", "result", '        if self.status == "x" and self.result.ok():', None)
        pick("status", "status", '        if self.status == "x" and self.result.ok():', None)
        # the binding's way read on: a change on it counts where the read reads what it changes
        choose = lambda ff, rel, line, keys, of_self: FO.choose(ff, rel, line, None, "map", keys, of_self)
        at = (F, LN["        if self.job is None:"])
        for name, keys in (("ways, which object", frozenset()), ("ways, jobStatus", frozenset({{"jobStatus"}}))):
            got[name] = [[w.end, w.why, [c.at for c in w.changes]] for w in SR.field_ways(G, "job", choose=choose, at=at, keys=keys)]
        # the run printed plug's "done": its handler's binding did not run
        real = FO._Path._narrowing
        FO._Path._narrowing = lambda self, node, owner: (
            ([s for s in node.body[0].body if s.lineno == LN['            log("done")']], []) if node.name == "plug" else ([], None))
        pick("result, printed", "result", '        if self.status == "x" and self.result.ok():', None)
        FO._Path._narrowing = real
        # report entered from two calls: the chain stops, nothing is off the path
        install(reach_graph=reach(extra=[call("report", LN["        self.job = fetch2()"])]))
        pick("two callers", "job", "        if self.job is None:", frozenset())
        # a call on the path that may go anywhere: other's binding may have run
        install(reach_graph=reach(holes=[N(line=1, callee="x()", why="unplaced", for_classes=[])]))
        pick("a hole", "job", "        if self.job is None:", frozenset())
        print(json.dumps(got))
    """))
    got = json.loads(out.splitlines()[-1])
    L = {k: str(v) for k, v in ln.items()}
    # (a) run binds before it calls report; __init__'s is rebound; other is off the run's path; which object: no change
    assert got["job"] == [[L["        self.job = fetch()"]], [],
                          [[L["        self.job = None"], "rebound-before-the-read"], [L["        self.job = fetch2()"], "off-the-run-s-path"]], ""]
    # (b) plug binds on both its ways; report's own None is rebound by the call; the change through plug counts
    assert got["result"] == [[L["            self.result = Res()"], L["            self.result = None"]], ["4"],
                             [[L["        self.result = None"], "rebound-before-the-read"], [str(rep_none), "rebound-before-the-read"]], ""]
    assert got["result, printed"][0] == [L["            self.result = Res()"]]
    assert [L["            self.result = None"], "not-run"] in got["result, printed"][2]
    # (c) only __init__ binds it
    assert got["status"] == [[L["        self.status = status"]], [], [], ""]
    # run's binding read on: its jobStatus changed on the way is on a read of jobStatus (G4: the call's own end kept), not
    # on one of which object it is
    assert got["ways, which object"] == [["stop", "call", []]]
    assert got["ways, jobStatus"] == [["stop", "call", ["pandaserver/g2.py:99"]]]
    # two callers: neither run's nor other's binding is chosen past report's entry
    assert got["two callers"][0] == [L["        self.job = None"], L["        self.job = fetch()"], L["        self.job = fetch2()"]]
    assert got["two callers"][3] == "report: 2 calls enter it"
    assert got["a hole"][0] == [L["        self.job = fetch()"], L["        self.job = fetch2()"]]


_GEN4 = """
class Res:
    def set_fatal(self):
        self.code = 2

    def set_ok(self):
        self.code = 0


class Plugin:
    def __init__(self, bad):
        self.bad = bad
        self.result = Res()

    def execute(self):
        if self.bad:
            log("bad")
            self.result.set_fatal()
            return
        self.update()
        log("done")

    def update(self):
        try:
            register()
            self.result.set_ok()
            log("registered")
        except Exception:
            log("failed")
            self.result.set_fatal()


class Gen:
    def run(self):
        self.plugin = Plugin(False)
        self.plugin.execute()
        if self.plugin.result.code == 2:
            fail()
"""


def test_g1_a_change_placed_at_its_first_call_is_read_further_in_by_each_functions_printing(tmp_path):
    # AdderGen.adder_plugin's changes of .result (out/gm/gm5-design.md G1): each is placed at run's call of execute,
    # which ran; the set_fatal of execute's early return and of update's handler did not, by those functions' prints.
    # A function the run does not narrow (called twice, or no prints) drops nothing.
    (_corpus(tmp_path) / "pandaserver" / "g4.py").write_text(_GEN4)
    lines = _GEN4.splitlines()
    first = lambda t: lines.index(t) + 1  # noqa: E731
    ln = {
        "bind": first("        self.plugin = Plugin(False)"),
        "execute": first("        self.plugin.execute()"),
        "read": first("        if self.plugin.result.code == 2:"),
        "bad": first("            self.result.set_fatal()"),
        "update": first("        self.update()"),
        "ok": first("            self.result.set_ok()"),
        "failed": [i + 1 for i, t in enumerate(lines) if t == "            self.result.set_fatal()"][1],
        "done": first('        log("done")'),
        "registered": first('            log("registered")'),
    }
    out = _ok(_run(tmp_path, f"""
        import json
        from types import SimpleNamespace as N
        from bamboo.querier import field_order as FO, mapdump
        from bamboo.querier import sources as SR

        F = "pandaserver/g4.py"
        LN = {ln!r}
        call = lambda to, line: N(to=f"{{F}}::{{to}}", lines=[line], line=line, thread=False)
        reach = [
            N(owner=f"{{F}}::Gen.run", calls=[call("Plugin.execute", LN["execute"])], holes=[]),
            N(owner=f"{{F}}::Plugin.execute", calls=[call("Plugin.update", LN["update"])], holes=[]),
            N(owner=f"{{F}}::Plugin.update", calls=[], holes=[]),
        ]
        change = lambda line, *via: N(at=f"{{F}}:{{LN[line]}}", kind="inner", key=".result", what="",
                                      via=[f"{{F}}:{{LN[v]}}" for v in via])
        ff = N(owner=f"{{F}}::Gen", field="plugin", stops=[],
               bindings=[N(at=f"{{F}}:{{LN['bind']}}", owner=f"{{F}}::run", how="assign", sources=[], stops=[])],
               changes=[change("bad", "execute"), change("ok", "execute", "update"),
                        change("failed", "execute", "update")])
        mapdump._FRAGMENT[:] = [N(field_froms=[ff], entered_froms=[], parameter_froms=[], handler_marks=[],
                                  reach_graph=reach)]
        for cached in (SR._field_index, SR._calls_in, FO._reach_index, FO._closure_in, FO._callers_in):
            cached.cache_clear()
        got = {{}}
        read = []

        def pick(name, prints):
            def narrowing(self, node, owner):
                read.append(owner)
                if node.name not in prints:
                    return [], None
                return [FO.R.statement_at(node, ln) for ln in prints[node.name]], []
            FO._Path._narrowing = narrowing
            read.clear()
            c = FO.choose(ff, F, LN["read"], None, "map", frozenset({{"result"}}))
            got[name] = [[x.at.rpartition(":")[2] for x in c.changes],
                         [[at.rpartition(":")[2], why] for at, why in c.dropped], len(read) == len(set(read))]

        pick("no prints", {{}})
        pick("printed", {{"execute": [LN["done"]], "update": [LN["registered"]]}})
        # update called twice in the run's window: its prints narrow nothing
        pick("update twice", {{"execute": [LN["done"]]}})
        print(json.dumps(got))
    """))
    got = json.loads(out.splitlines()[-1])
    L = {k: str(v) for k, v in ln.items()}
    # without prints every change placed at the call of execute that ran is between
    assert got["no prints"] == [[L["bad"], L["ok"], L["failed"]], [], True]
    # execute printed "done": its early return's set_fatal did not run; update printed "registered" and entered no
    # handler: its handler's set_fatal did not run; each function's prints read once
    assert got["printed"] == [[L["ok"]], [[L["bad"], "not-run"], [L["failed"], "not-run"]], True]
    assert got["update twice"] == [[L["ok"], L["failed"]], [[L["bad"], "not-run"]], True]


_GEN3 = """
class Gen:
    def __init__(self, status):
        self.status = status

    def run(self):
        self.report()

    def report(self):
        if self.status == "x":
            fail()


def thread_run():
    gen = Gen(pop())
    gen.run()


def thread_again(fresh):
    if fresh:
        gen = Gen(pop())
    else:
        gen = Gen(peek())
    gen.run()


def api(status):
    gen = Gen(status)
    gen.dump()


class Holder:
    def go(self):
        self.gen = Gen(pop())
        self.gen.run()
"""


def test_s4b_reads_a_bindings_parameter_by_the_call_the_runs_path_entered_its_function_by(tmp_path):
    # AdderGen.__init__'s attempt_nr (out/gm/s4b-site.md): the path built the object at add_main 136, so pilot_api 700's
    # ways in are left out; where the path entered at two calls both are read; under (c) or an open path, every caller's
    # (e); the call that built the object read, by the map's records, open path or not (e2)
    (_corpus(tmp_path) / "pandaserver" / "g3.py").write_text(_GEN3)
    lines = _GEN3.splitlines()

    def one(t, k=0):
        return [i + 1 for i, x in enumerate(lines) if x == t][k]

    L = {"bind": one("        self.status = status"), "read": one('        if self.status == "x":'),
         "report": one("        self.report()"), "pop": one("    gen = Gen(pop())"), "run1": one("    gen.run()"),
         "again_pop": one("        gen = Gen(pop())"), "again_peek": one("        gen = Gen(peek())"),
         "run2": one("    gen.run()", 1), "api": one("    gen = Gen(status)"), "def": one("    def __init__(self, status):"),
         "go_build": one("        self.gen = Gen(pop())"), "go_run": one("        self.gen.run()")}
    out = _ok(_run(tmp_path, f"""
        import json
        from types import SimpleNamespace as N
        from bamboo.querier import field_order as FO, mapdump, rerun
        from bamboo.querier import sources as SR

        F = "pandaserver/g3.py"
        L = {L!r}
        G, INIT, API = f"{{F}}::Gen", f"{{F}}::__init__", f"{{F}}::api"
        at = lambda k: f"{{F}}:{{L[k]}}"
        call = lambda to, k: N(to=f"{{F}}::{{to}}", lines=[L[k]], line=L[k], thread=False)
        src = lambda kind, name, k, owner="": N(kind=kind, name=name, line=L[k], route=[at(k)], site_owner=owner,
                                                 steps=[], paths=[], through=[], changes=[])
        ffs = [N(owner=G, field="status", changes=[], stops=[], bindings=[
            N(at=at("bind"), owner=INIT, how="assign", stops=[], sources=[
                N(kind="parameter", name="status", line=L["def"], route=[], site_owner=INIT, steps=[], paths=[],
                  through=[], changes=[])])])]
        pfs = [N(owner=INIT, parameter="status", line=L["def"], stops=[], sources=[
                   src("call", "pop", "pop"), src("call", "pop", "again_pop"), src("call", "peek", "again_peek"),
                   src("parameter", "status", "api", API), src("call", "pop", "go_build")]),
               N(owner=API, parameter="status", line=L["api"] - 1, sources=[], stops=[N(kind="no_caller")])]

        def reach(callers, holes=()):
            return [N(owner=f"{{F}}::Gen.run", calls=[call("Gen.report", "report")], holes=[],
                      call_bindings=[N(line=L["report"], to=f"{{F}}::Gen.report", parameter="self", path="self")]),
                    N(owner=f"{{F}}::Gen.report", calls=[], holes=list(holes)),
                    N(owner=f"{{F}}::Gen.__init__", calls=[], holes=[]),
                    N(owner=API, calls=[call("Gen.__init__", "api")], holes=[]), *callers]

        choose = lambda ff, rel, line, keys, of_self: FO.choose(ff, rel, line, None, "map", keys, of_self)
        got = {{}}

        def pick(name, graph, of_self=False):
            mapdump._FRAGMENT[:] = [N(field_froms=ffs, entered_froms=[], parameter_froms=pfs, handler_marks=[],
                                      reach_graph=graph)]
            for cached in (SR._field_index, SR._calls_in, FO._reach_index, FO._closure_in, FO._callers_in):
                cached.cache_clear()
            c = choose(SR._field_froms()[(G, "status")], F, L["read"], None, of_self)
            ways = SR.field_ways(G, "status", choose=choose, at=(F, L["read"]), of_self=of_self)
            got[name] = {{"taken": [b.at for b in c.bindings], "entered": [[b, list(s)] for b, s in c.entered],
                          "ways": [[w.why, w.of, list(w.entered)] for w in ways],
                          "param": [[w.end, w.why, list(w.route)] for w in SR.param_ways(INIT, "status", entered=ways[0].entered)],
                          "off": SR.calls_off_the_path(INIT, "status", ways[0].entered)}}

        thread_run = lambda *c: N(owner=f"{{F}}::thread_run", calls=list(c), holes=[])
        pick("one call", reach([thread_run(call("Gen.__init__", "pop"), call("Gen.run", "run1"))]))
        pick("two calls", reach([N(owner=f"{{F}}::thread_again", holes=[], calls=[
            N(to=f"{{F}}::Gen.__init__", lines=[L["again_pop"], L["again_peek"]], line=L["again_pop"], thread=False),
            call("Gen.run", "run2")])]))
        # nothing the map holds calls run: the chain stops before the object is built, __init__ is placed first (c)
        pick("(c)", reach([thread_run(call("Gen.__init__", "pop"))]))
        pick("open", reach([thread_run(call("Gen.__init__", "pop"), call("Gen.run", "run1"))],
                           holes=[N(line=1, callee="x()", why="unplaced", for_classes=[])]))

        # (e2): a call in the caller may go anywhere; the object run is its local, what Gen(...) returned
        hole = [N(line=1, callee="lock()", why="unplaced", for_classes=[])]
        built = lambda k, how="result": N(line=L[k], name="gen", path="" if how == "result" else "other", how=how,
                                          to=[f"{{F}}::Gen.__init__"] if how == "result" else [])
        hands = lambda k, path="gen": N(line=L[k], to=f"{{F}}::Gen.run", parameter="self", path=path)
        local = N(owner=f"{{F}}::thread_run", holes=hole, calls=[call("Gen.__init__", "pop"), call("Gen.run", "run1")],
                  bindings=[built("pop")], call_bindings=[hands("run1")])
        pick("built, open", reach([local]), of_self=True)
        pick("built, not self's", reach([local]), of_self=False)
        again = lambda how: N(owner=f"{{F}}::thread_again", holes=hole, calls=[
            N(to=f"{{F}}::Gen.__init__", lines=[L["again_pop"], L["again_peek"]], line=L["again_pop"], thread=False),
            call("Gen.run", "run2")], bindings=[built("again_pop"), built("again_peek", how)], call_bindings=[hands("run2")])
        pick("built twice", reach([again("result")]), of_self=True)
        pick("built and an alias", reach([again("alias")]), of_self=True)
        # the object run is held in a field: not told, the path is open
        held = lambda holes: N(owner=f"{{F}}::Holder.go", holes=holes, calls=[call("Gen.__init__", "go_build"), call("Gen.run", "go_run")],
                               bindings=[], call_bindings=[hands("go_run", "self.gen")])
        pick("a field's, open", reach([held(hole)]), of_self=True)
        pick("a field's", reach([held([])]), of_self=True)

        # the parameter's end on the arm's read: a call left out is not asked of its caller's logs
        mapdump._FRAGMENT[:] = [N(parameter_froms=[
            N(owner=INIT, parameter="status", line=L["def"], stops=[], sources=[
                N(kind="constant", name="'failed'", route=["pandaserver/a.py:10"], site_owner="", paths=[]),
                N(kind="constant", name="'finished'", route=["pandaserver/a.py:20"], site_owner="", paths=[])])],
            entered_froms=[])]
        SR._calls_in.cache_clear()
        r = object.__new__(rerun.Rerun)
        r.lines = []

        def shown(site, window, machine, why=""):
            assert site == "pandaserver/a.py:20", site
            return True

        r._call_shown = shown
        t = SR.Term("g", "self.status", (SR.End("self.status", SR.STOP, SR.PARAMETER, at("bind"), param="status", of=INIT,
                                                  entered=("pandaserver/a.py:20",)),))
        end = r._with_callers(t, "the arm's function", f"{{F}}::Gen.report", N(machine="m1"), ("a", "b")).ends[0]
        got["end"] = [end.end, end.why, end.at, r.lines]
        # the one way left is where it ends: what a call outside the corpus returned, by the call that built it
        mapdump._FRAGMENT[:] = [N(parameter_froms=[
            N(owner=INIT, parameter="code", line=L["def"], stops=[], sources=[
                N(kind="constant", name="'failed'", route=["pandaserver/a.py:10"], site_owner="", paths=[]),
                N(kind="call", name="pop", route=["pandaserver/a.py:20"], site_owner="", paths=[])])],
            entered_froms=[])]
        SR._calls_in.cache_clear()
        t = SR.Term("g", "self.code", (SR.End("self.code", SR.STOP, SR.PARAMETER, at("bind"), param="code", of=INIT,
                                                via=(at("bind"),), entered=("pandaserver/a.py:20",)),))
        end = r._with_callers(t, "the arm's function", f"{{F}}::Gen.report", N(machine="m1"), ("a", "b")).ends[0]
        got["one left"] = [end.end, end.why, end.at, list(end.via)]
        # two constants left, both by one call: its caller's run cannot tell them apart and is not asked (shown raises)
        mapdump._FRAGMENT[:] = [N(parameter_froms=[
            N(owner=INIT, parameter="flag", line=L["def"], stops=[], sources=[
                N(kind="constant", name="False", route=["pandaserver/a.py:30"], site_owner="", paths=[]),
                N(kind="constant", name="True", route=["pandaserver/a.py:30"], site_owner="", paths=[])])],
            entered_froms=[])]
        SR._calls_in.cache_clear()
        r.lines = []
        t = SR.Term("g", "self.flag", (SR.End("self.flag", SR.STOP, SR.PARAMETER, at("bind"), param="flag", of=INIT),))
        end = r._with_callers(t, "the arm's function", f"{{F}}::Gen.report", N(machine="m1"), ("a", "b")).ends[0]
        got["same call"] = [end.end, end.why, r.lines]
        print(json.dumps(got))
    """))
    got = json.loads(out.splitlines()[-1])
    F = "pandaserver/g3.py"
    at = {k: f"{F}:{v}" for k, v in L.items()}
    # the path built the object at thread_run's call: only the way in by it is read
    assert got["one call"]["taken"] == [at["bind"]]
    assert got["one call"]["entered"] == [[at["bind"], [at["pop"]]]]
    assert got["one call"]["ways"] == [["parameter", f"{F}::__init__", [at["pop"]]]]
    assert got["one call"]["param"] == [["stop", "call", [at["pop"]]]]
    assert got["one call"]["off"] == [at["again_pop"], at["again_peek"], at["api"], at["go_build"]]
    # entered at either of two calls: both are read
    assert got["two calls"]["entered"] == [[at["bind"], sorted([at["again_pop"], at["again_peek"]])]]
    assert got["two calls"]["param"] == [["stop", "call", [at["again_pop"]]], ["stop", "call", [at["again_peek"]]]]
    assert got["two calls"]["off"] == [at["pop"], at["api"], at["go_build"]]
    # (c) and an open path tell no call: every caller's way is read; so does a field's object on an open path, and a
    # local one binding of which reaching the call is not what Gen(...) returned
    for name in ("(c)", "open", "built, not self's", "built and an alias", "a field's, open"):
        assert got[name]["taken"] == [at["bind"]], name
        assert got[name]["entered"] == [] and got[name]["off"] == [], name
        assert [w[0] for w in got[name]["param"]] == ["stop", "stop", "stop", "api", "stop"], name
    # (e2) the object read was built at the call whose result the caller's local holds, a call there open or not
    assert got["built, open"]["entered"] == [[at["bind"], [at["pop"]]]]
    assert got["built, open"]["param"] == [["stop", "call", [at["pop"]]]]
    assert got["built, open"]["off"] == [at["again_pop"], at["again_peek"], at["api"], at["go_build"]]
    assert got["built twice"]["entered"] == [[at["bind"], sorted([at["again_pop"], at["again_peek"]])]]
    # a field's object on a path with no hole: (e)'s call the path entered __init__ by
    assert got["a field's"]["entered"] == [[at["bind"], [at["go_build"]]]]
    assert got["end"][:3] == ["constant", "'finished'", "pandaserver/a.py:20"]
    assert "left out: 1 by pandaserver/a.py:10" in got["end"][3][0]
    assert got["one left"] == ["stop", "call", "pandaserver/a.py:20", [at["bind"]]]
    assert got["same call"][:2] == ["stop", "parameter-unchosen"]
    assert "all come by pandaserver/a.py:30: its caller's run not asked" in got["same call"][2][0]


def test_q1_reads_a_container_made_empty_through_its_puts_and_q2_carries_the_reads_steps_up_fields_and_callers(tmp_path):
    # case A's attempt_nr: AdderGen.__init__'s parameter is handed self.reports[*][2] at 136; the field is bound at 88
    # from T.__init__'s parameter and popped at 108; that one is handed a WeightedLists made empty and filled by add
    out = _ok(_run(tmp_path, """
        import json
        from types import SimpleNamespace as N
        from bamboo.codemap import models as M
        from bamboo.querier import mapdump
        from bamboo.querier import sources as SR

        G, T, TI = "p/g.py::__init__", "p/m.py::main.T", "p/m.py::__init__"
        put = M.HeldChange(at="p/m.py:206", kind="method", key="*", what="add", value=[
            M.HeldSource(kind="rows", name="panda:p/j.py::list@2399", line=2432, steps=["[*]"],
                         route=["p/t.py:2578", "p/m.py:199"], at=["p/j.py:2432"]),
            M.HeldSource(kind="constant", name="None", line=2371, steps=["[*]"], route=["p/t.py:2578", "p/m.py:199"],
                         at=["p/j.py:2371"])])
        lists = M.HeldSource(kind="call", name="WeightedLists", line=196, route=["p/m.py:231"], made_empty="collection",
                             changes=[put], at=["p/m.py:196"])
        # a put the map does not follow: the container stays a way
        unread = lists.model_copy(update={"changes": [put, M.HeldChange(at="p/m.py:210", kind="escaped", key="*")]})
        # case A's zip_files: a {} filled by a store of another {}, read whole
        inner = M.HeldSource(kind="unknown", name="{}", line=316, made_empty="mapping", at=["p/r.py:316"])
        zips = M.HeldSource(kind="unknown", name="{}", line=272, made_empty="mapping", route=["p/r.py:677"],
                            at=["p/r.py:272"], changes=[M.HeldChange(at="p/r.py:316", kind="store", key="*", value=[inner])])
        empty = M.HeldSource(kind="unknown", name="[]", line=5, made_empty="collection", route=["p/r.py:9"], at=["p/r.py:5"])
        # case A's cont_zip_map: a None put in it is an element, not the value
        nones = M.HeldSource(kind="unknown", name="{}", line=274, made_empty="mapping", route=["p/r.py:677"], at=["p/r.py:274"],
                             changes=[M.HeldChange(at="p/r.py:334", kind="store", key="*", value=[
                                 M.HeldSource(kind="constant", name="None", line=334, at=["p/r.py:334"])])])
        pf = lambda owner, name, *sources: N(owner=owner, parameter=name, line=1, stops=[], sources=list(sources))
        mapdump._FRAGMENT[:] = [N(
            parameter_froms=[
                pf(G, "attempt_nr", M.HeldSource(kind="self_field", name="self.reports", line=108, site_owner=T,
                                                 steps=["[*]", "[2]"], route=["p/m.py:136"])),
                pf(TI, "reports", lists),
                pf("p/u.py::f", "reports", unread),
                pf("p/r.py::register", "zip_files", zips),
                pf("p/r.py::register", "nothing", empty),
                pf("p/r.py::register", "nones", nones),
            ],
            field_froms=[N(owner=T, field="reports", stops=[],
                           changes=[N(at="p/m.py:108", kind="method", key="*", what="pop", via=[])],
                           bindings=[N(at="p/m.py:88", owner=TI, how="assign", stops=[], sources=[
                               M.HeldSource(kind="parameter", name="reports", line=84, site_owner=TI)])])],
            entered_froms=[])]
        SR._calls_in.cache_clear()
        SR._field_index.cache_clear()
        way = lambda w: [w.end, w.value, w.why, list(w.route), w.end_at]
        print(json.dumps({
            "attempt_nr": [way(w) for w in SR.param_ways(G, "attempt_nr")],
            "element": [way(w) for w in SR.param_ways(TI, "reports", steps=("[*]", "[3]"))],
            "unread": [way(w) for w in SR.param_ways("p/u.py::f", "reports", steps=("[*]",))],
            "whole": [way(w) for w in SR.param_ways("p/r.py::register", "zip_files")],
            "nothing put": [way(w) for w in SR.param_ways("p/r.py::register", "nothing")],
            "a None put": [way(w) for w in SR.param_ways("p/r.py::register", "nones")],
            "field, element": [[w.why, list(w.steps), [c.at for c in w.changes]] for w in SR.field_ways(T, "reports", ("[*]", "[2]"))],
            "field, whole": [[w.why, list(w.steps), [c.at for c in w.changes]] for w in SR.field_ways(T, "reports")],
        }))
    """))
    got = json.loads(out.splitlines()[-1])
    back = ["p/t.py:2578", "p/m.py:199", "p/m.py:231"]
    # Q2 (a) and (c): the field read on with [*][2], its binding's parameter up T.__init__'s callers; Q1: the element is
    # what add put -- a row's part, and a part of the None (no literal's value) -- each placed where it ends
    assert got["attempt_nr"] == [["database", "", "rows", [*back, "p/m.py:88", "p/m.py:84", "p/m.py:136"], "p/j.py:2432"],
                                 ["constant", "", "constant-through", [*back, "p/m.py:88", "p/m.py:84", "p/m.py:136"], "p/j.py:2371"]]
    assert got["element"] == [["database", "", "rows", back, "p/j.py:2432"], ["constant", "", "constant-through", back, "p/j.py:2371"]]
    assert got["unread"] == [*got["element"], ["stop", "", "container", ["p/m.py:231"], "p/m.py:196"]]
    # read whole: the empty start, each value put read whole in turn; a mapping stays (no put's value holds its keys),
    # a container nothing is done to holds only its start
    assert got["whole"] == [["constant", "", "made-empty", ["p/r.py:677"], "p/r.py:272"],
                            ["constant", "", "made-empty", ["p/r.py:677"], "p/r.py:316"],
                            ["stop", "", "container", ["p/r.py:677"], "p/r.py:272"]]
    assert got["nothing put"] == [["constant", "", "made-empty", ["p/r.py:9"], "p/r.py:5"]]
    assert got["a None put"] == [["constant", "", "made-empty", ["p/r.py:677"], "p/r.py:274"],
                                 ["constant", "", "constant-through", ["p/r.py:677"], "p/r.py:334"],
                                 ["stop", "", "container", ["p/r.py:677"], "p/r.py:274"]]
    # an element read is not changed by the pop that takes it out; the whole value is
    assert got["field, element"] == [["parameter", ["[*]", "[2]"], []]]
    assert got["field, whole"] == [["parameter", [], ["p/m.py:108"]]]


def test_q2_a_parameter_with_no_literal_left_lists_each_way_and_a_splits_reading_asks_no_run(tmp_path):
    # modTime's ways: a statement's rows by two calls, a part of a None, a + not read -- no literal a run could tell
    # apart: each way where it ends or stops, a place reached by two routes said once; a literal left keeps the stop
    out = _ok(_run(tmp_path, """
        import json
        from types import SimpleNamespace as N
        from bamboo.querier import mapdump, rerun
        from bamboo.querier import sources as SR

        INIT = "pandaserver/g.py::__init__"
        src = lambda kind, name, route, **kw: N(kind=kind, name=name, route=list(route), site_owner="", paths=[], **kw)
        mapdump._FRAGMENT[:] = [N(parameter_froms=[
            N(owner=INIT, parameter="t", line=5, stops=[], sources=[
                src("rows", "panda:p/j.py::l@9", ["p/t.py:1", "pandaserver/a.py:10"], at=["p/j.py:9"]),
                src("rows", "panda:p/j.py::l@9", ["p/t.py:1", "pandaserver/a.py:20"], at=["p/j.py:9"]),
                src("constant", "None", ["p/t.py:1", "pandaserver/a.py:10"], at=["p/j.py:3"], steps=["[*]"]),
                src("unknown", "a + b", ["p/t.py:1", "pandaserver/a.py:10"], at=["p/j.py:7"])]),
            N(owner=INIT, parameter="u", line=5, stops=[], sources=[
                src("rows", "panda:p/j.py::l@9", ["pandaserver/a.py:10"], at=["p/j.py:9"]),
                src("constant", "'x'", ["pandaserver/a.py:20"], at=["pandaserver/a.py:20"])]),
        ], entered_froms=[])]
        SR._calls_in.cache_clear()
        r = object.__new__(rerun.Rerun)
        r.lines = []

        def shown(*a, **k):
            raise AssertionError("a caller's run asked")

        r._call_shown = shown
        end = lambda e: [e.end, e.why, e.at, list(e.route), list(e.via)]
        t = SR.Term("g", "self.t", (SR.End("self.t", SR.STOP, SR.PARAMETER, "pandaserver/g.py:7", param="t", of=INIT,
                                           via=("pandaserver/g.py:30",)),))
        u = SR.Term("g", "self.u", (SR.End("self.u", SR.STOP, SR.PARAMETER, "pandaserver/g.py:8", param="u", of=INIT),))
        print(json.dumps({
            "t": [end(e) for e in r._with_callers(t, None, "pandaserver/g.py::report", None, None, ask=False).ends],
            "u": [end(e) for e in r._with_callers(u, None, "pandaserver/g.py::report", None, None, ask=False).ends],
            "lines": r.lines}))
    """))
    got = json.loads(out.splitlines()[-1])
    by = ["p/t.py:1", "pandaserver/a.py:10"]
    assert got["t"] == [["database", "rows", "p/j.py:9", by, ["pandaserver/g.py:30"]],
                        ["constant", "constant-through", "p/j.py:3", by, ["pandaserver/g.py:30"]],
                        ["stop", "unknown", "p/j.py:7", by, ["pandaserver/g.py:30"]]]
    assert any("t: 4 way(s) end at 3 place(s)" in line for line in got["lines"])
    # 'x' is a literal a caller's run could choose: not asked here (a split's terms), so it stops as before
    assert got["u"] == [["stop", "parameter-unchosen", "", [], []]]


_LISTER = """
class Proxy:
    def list_reports(self, only_unlocked=False):
        if only_unlocked:
            ret = self.cur.fetchall()
        else:
            ret = self.cur.fetchall()
        return ret
"""


def test_cro_a_callees_test_false_on_its_calls_constant_rules_a_way_out(tmp_path):
    # listJobOutputReport's else branch (2420): both of add_main's calls hand only_unlocked=True through the task
    # buffer, so its way is none of theirs; another caller's default False keeps it, and a value not one constant too
    (_corpus(tmp_path) / "pandaserver" / "j.py").write_text(_LISTER)
    lines = _LISTER.splitlines()
    test, locked, unlocked = (lines.index(t) + 1 for t in (
        "        if only_unlocked:", "            ret = self.cur.fetchall()", "        else:"))
    unlocked += 1
    out = _ok(_run(tmp_path, f"""
        import json
        from types import SimpleNamespace as N
        from bamboo.codemap import models as M
        from bamboo.querier import mapdump
        from bamboo.querier import sources as SR

        P, TB = "pandaserver/j.py::list_reports", "pandaserver/t.py::list_reports"
        mapdump._FRAGMENT[:] = [N(
            guard_froms=[N(owner=P, test_line={test}, name="only_unlocked", test="only_unlocked", arms=[], stops=[],
                           calls=[{locked}, {unlocked}],
                           sources=[M.HeldSource(kind="parameter", name="only_unlocked", line={test}, site_owner=P)])],
            runnables=[],
            parameter_froms=[
                N(owner=P, parameter="only_unlocked", line=3, stops=[], sources=[
                    M.HeldSource(kind="parameter", name="only_unlocked", line=20, site_owner=TB, route=["pandaserver/t.py:20"])]),
                N(owner=TB, parameter="only_unlocked", line=18, stops=[], sources=[
                    M.HeldSource(kind="constant", name="True", line=200, route=["pandaserver/m.py:199"]),
                    M.HeldSource(kind="constant", name="True", line=208, route=["pandaserver/m.py:207"]),
                    # a default, placed by the call it is the default of
                    M.HeldSource(kind="constant", name="False", line=18, route=["pandaserver/c.py:369"]),
                    M.HeldSource(kind="parameter", name="flag", line=40, site_owner="pandaserver/x.py::f",
                                 route=["pandaserver/x.py:41"])]),
                N(owner="pandaserver/x.py::f", parameter="flag", line=39, stops=[], sources=[
                    M.HeldSource(kind="constant", name="True", line=1, route=["pandaserver/y.py:1"]),
                    M.HeldSource(kind="constant", name="False", line=2, route=["pandaserver/y.py:2"])]),
            ],
            entered_froms=[])]
        SR._calls_in.cache_clear()
        way = lambda line, *route: M.HeldSource(kind="database", name="self.cur.fetchall", line=line, route=list(route),
                                                at=[f"pandaserver/j.py:{{line}}"])
        got = {{
            "else by 199": SR._cut(way({unlocked}, "pandaserver/t.py:20", "pandaserver/m.py:199")),
            "if by 199": SR._cut(way({locked}, "pandaserver/t.py:20", "pandaserver/m.py:199")),
            "else by the default": SR._cut(way({unlocked}, "pandaserver/t.py:20", "pandaserver/c.py:369")),
            "else by two values": SR._cut(way({unlocked}, "pandaserver/t.py:20", "pandaserver/x.py:41")),
            "else by a call not read": SR._cut(way({unlocked}, "pandaserver/t.py:20", "pandaserver/z.py:9")),
            "no route": SR._cut(way({unlocked})),
        }}
        got["said"] = SR.take_cuts()
        got["said again"] = SR.take_cuts()
        print(json.dumps(got))
    """))
    got = json.loads(out.splitlines()[-1])
    assert got["else by 199"] == f"'not (only_unlocked)' false at pandaserver/j.py:{unlocked} (only_unlocked=True by pandaserver/t.py:20)"
    for kept in ("if by 199", "else by the default", "else by two values", "else by a call not read", "no route"):
        assert got[kept] == "", kept
    assert got["said"] == [f"pandaserver/j.py:{unlocked} by pandaserver/t.py:20 -> pandaserver/m.py:199: {got['else by 199']}"]
    assert got["said again"] == []


_PEEK = """\
from pandaserver.spec import JobSpec


class Proxy:
    def peekJob(self, pandaID):
        sql = f"SELECT {JobSpec.columnNames()} FROM ATLAS_PANDA.jobsActive4 WHERE PandaID=:PandaID"
        varMap = {}
        varMap[":PandaID"] = pandaID
        self.cur.execute(sql, varMap)
        res = self.cur.fetchall()
        if len(res) != 0:
            job = JobSpec()
            job.pack(res[0])
            return job
        return None

    def peekChanged(self, pandaID):
        sql = f"SELECT {JobSpec.columnNames()} FROM ATLAS_PANDA.jobsActive4 WHERE PandaID=:PandaID"
        varMap = {}
        varMap[":PandaID"] = pandaID
        self.cur.execute(sql, varMap)
        res = self.cur.fetchall()
        job = JobSpec()
        job.pack(res[0])
        job.jobStatus = "closed"
        return job
"""


def test_q4_an_object_a_callee_filled_from_one_row_and_returned_ends_at_the_row(tmp_path):
    # AdderGen's self.job = peekJobs(...)[0]: peekJob builds a JobSpec at 518, packs res[0] into it and returns it; a
    # test reading a field of it, or which object it is (its table's identity), ends at the row; a field the callee
    # writes after the fill does not
    corpus = _corpus(tmp_path)
    (corpus / "pandaserver" / "__init__.py").write_text("")
    (corpus / "pandaserver" / "spec.py").write_text(_ROW_SPEC)
    (corpus / "pandaserver" / "peek.py").write_text(_PEEK)
    lines = _PEEK.splitlines()
    built, changed = (i + 1 for i, t in enumerate(lines) if t.strip() == "job = JobSpec()")
    out = _ok(_run(tmp_path, f"""
        import json
        from types import SimpleNamespace as N
        from bamboo.codemap import models as M
        from bamboo.querier import mapdump
        from bamboo.querier import sources as SR

        mapdump._FRAGMENT[:] = [N(tables=[N(spec_class="JobSpec", identity_attrs=["JobSpec.PandaID"])],
                                  guard_froms=[], parameter_froms=[], field_froms=[], runnables=[], entered_froms=[])]
        src = lambda line, route=("pandaserver/tb.py:5", "pandaserver/g.py:9"): M.HeldSource(
            kind="object", name="JobSpec", line=line, site_owner="pandaserver/spec.py::JobSpec", route=list(route),
            at=[f"pandaserver/peek.py:{{line}}"])
        said = lambda got: None if got is None else [c.at for c in got]
        print(json.dumps({{
            "a field": said(SR._returned_row(src({built}), frozenset({{"jobStatus"}}))),
            "which object": said(SR._returned_row(src({built}), frozenset())),
            "whole": said(SR._returned_row(src({built}), None)),
            "not returned through a call": said(SR._returned_row(src({built}, ()), frozenset({{"jobStatus"}}))),
            "a field written after": said(SR._returned_row(src({changed}), frozenset({{"jobStatus"}}))),
            "its identity, not written after": said(SR._returned_row(src({changed}), frozenset())),
        }}))
    """))
    got = json.loads(out.splitlines()[-1])
    assert got == {"a field": [], "which object": [], "whole": None, "not returned through a call": None,
                   "a field written after": None, "its identity, not written after": []}, got


_DISPS = """\
class Setupper:
    def subscribe(self, disps):
        disp_error = {}
        for disp in disps:
            disp_error[disp] = ""
            if self.bad(disp):
                disp_error[disp] = "could not register"
        disp_error.pop(None, None)
        out = []
        for disp in disps:
            if disp_error[disp] != "":
                out.append(disp)
        return out
"""


def test_q1_an_element_the_test_takes_is_read_by_s4c_on_the_arms_run_and_by_its_puts_where_no_run_is(tmp_path):
    # MF-7398155528's disp_error[disp]: on the arm's run the puts stay changes on the container's way, for S4c to tell
    # apart by the run (it chose setupper 1098 by elimination); where no run is read (a split's terms) the element is a
    # value a put put; the pop that takes an element out is on neither; a field read by element the same way
    (_corpus(tmp_path) / "pandaserver" / "s.py").write_text(_DISPS)
    lines = _DISPS.splitlines()
    test, site = lines.index('            if disp_error[disp] != "":') + 1, lines.index("                out.append(disp)") + 1
    out = _ok(_run(tmp_path, f"""
        import ast, json
        from types import SimpleNamespace as N
        from bamboo.codemap import models as M
        from bamboo.codemap.panda import pathcond
        from bamboo.querier import mapdump, rerun
        from bamboo.querier import rows as R
        from bamboo.querier import sources as SR

        R.ROWS_FROM = "walk"
        owner, rel = "pandaserver/s.py::Setupper.subscribe", "pandaserver/s.py"
        put = lambda line, text: M.HeldChange(at=f"{{rel}}:{{line}}", kind="store", key="*", value=[
            M.HeldSource(kind="constant", name=text, line=line, at=[f"{{rel}}:{{line}}"])])
        errors = M.HeldSource(kind="unknown", name="{{}}", line=3, made_empty="mapping", at=[f"{{rel}}:3"], changes=[
            put(5, '""'), put(7, "'could not register'"), M.HeldChange(at=f"{{rel}}:8", kind="method", key="*", what="pop")])
        T = "p/m.py::main.T"
        jobs = M.HeldSource(kind="unknown", name="[]", line=147, made_empty="collection", at=["p/m.py:147"], changes=[
            M.HeldChange(at="p/m.py:149", kind="method", key="*", what="append", value=[
                M.HeldSource(kind="parameter", name="job", line=149, site_owner="p/m.py::run")])])
        mapdump._FRAGMENT[:] = [N(
            guard_froms=[N(owner=owner, test_line={test}, name="disp_error", sources=[errors], stops=[])],
            parameter_froms=[], entered_froms=[], runnables=[],
            field_froms=[N(owner=T, field="jobs", stops=[], changes=[], bindings=[
                N(at="p/m.py:151", owner="p/m.py::run", how="assign", stops=[], sources=[jobs])])])]
        SR._calls_in.cache_clear()
        SR._field_index.cache_clear()
        fn = R.function(rel, "subscribe", within=[{site}])
        stmt = R.statement_at(fn.node, {site})
        guards = pathcond.arm_condition(stmt)
        pairs = list(zip(guards, rerun._guard_ifs(fn, stmt, guards)))
        choose = lambda ff, *a: N(bindings=ff.bindings, changes=ff.changes, entered_by=lambda at: ())
        end = lambda e: [e.end, e.why, e.at, [c.at for c in e.changes]]
        ends = lambda ts: [end(e) for t in ts if t.term == "disp_error" for e in t.ends]
        way = lambda w: [w.end, w.why, [c.at for c in w.changes]]
        t = lambda s: ast.parse(s, mode="eval").body
        print(json.dumps({{
            "on the run": ends(SR.terms(fn, owner, pairs, choose=choose)),
            "no run": ends(SR.terms(fn, owner, pairs)),
            "field, on the run": [way(w) for w in SR.field_ways(T, "jobs", ("[*]",), choose=choose, at=("p/m.py", 160))],
            "field, no run": [way(w) for w in SR.field_ways(T, "jobs", ("[*]",))],
            "steps": [SR.item_step(t(s), n) for s, n in [("d[k] != ''", "d"), ("d['total'] > 0", "d"), ("d['a'] or d['b']", "d"),
                                                         ("not d", "d"), ("d.x", "d"), ("self.m[0]", "self.m")]],
        }}))
    """))
    got = json.loads(out.splitlines()[-1])

    def at(line: int) -> str:
        return f"pandaserver/s.py:{line}"

    assert got["on the run"] == [["stop", "unknown", at(3), [at(5), at(7)]]], got
    assert got["no run"] == [["constant", '""', at(5), []], ["constant", "'could not register'", at(7), []],
                             ["stop", "container", at(3), []]], got
    assert got["field, on the run"] == [["stop", "unknown", ["p/m.py:149"]]], got
    assert got["field, no run"] == [["stop", "parameter", []]], got
    assert got["steps"] == ["[*]", "['total']", "[*]", None, None, "[0]"], got


def test_a_call_tells_a_constant_apart_only_where_every_way_it_comes_by_ends_there(tmp_path):
    # adder_atlas_plugin 677 hands cont_zip_map, a {} whose put is read as a part: the call carries the container's
    # start and stop too, so its run showing it ran chooses none of them; a literal alone on its call is chosen as
    # before; a constant-through alone on its call keeps its why; a call handing True or False, the run printing True
    # (job_api 556's prodManager), is asked: the print leaves the True alone on it
    out = _ok(_run(tmp_path, """
        import json
        from types import SimpleNamespace as N
        from bamboo.codemap import models as M
        from bamboo.querier import mapdump, rerun
        from bamboo.querier import sources as SR

        F = "pandaserver/f.py::f"
        nones = M.HeldSource(kind="unknown", name="{}", line=274, made_empty="mapping", route=["pandaserver/a.py:10"],
                             at=["pandaserver/r.py:274"], changes=[M.HeldChange(at="pandaserver/r.py:334", kind="store", key="*",
                             value=[M.HeldSource(kind="constant", name="None", line=334, at=["pandaserver/r.py:334"])])])
        src = lambda kind, name, route, **kw: M.HeldSource(kind=kind, name=name, line=1, route=[route], at=[route], **kw)
        pf = lambda name, *sources: N(owner=F, parameter=name, line=1, stops=[], sources=list(sources))
        mapdump._FRAGMENT[:] = [N(parameter_froms=[
            pf("c", nones),
            pf("d", src("constant", "'x'", "pandaserver/a.py:20"), src("unknown", "g()", "pandaserver/a.py:30")),
            pf("e", src("constant", "'y'", "pandaserver/a.py:40", steps=["[*]"]), src("unknown", "g()", "pandaserver/a.py:50")),
            pf("f", src("constant", "True", "pandaserver/a.py:60"), src("constant", "False", "pandaserver/a.py:60"),
               src("unknown", "g()", "pandaserver/a.py:70")),
        ], entered_froms=[], guard_froms=[])]
        SR._calls_in.cache_clear()
        r = object.__new__(rerun.Rerun)
        r.lines = []
        asked = []

        def shown(site, *a, **k):
            asked.append(site)
            return True

        r._call_shown = shown
        r._entry_value = lambda fn, run, param: "True"
        end = lambda e: [e.end, e.why, e.at]
        got = {}
        for p in "cdef":
            asked.clear()
            got[p] = [[end(e) for e in r._param_end(p, F, p, N() if p == "f" else None, None, None)], list(asked)]
        print(json.dumps(got))
    """))
    got = json.loads(out.splitlines()[-1])
    assert got["c"] == [[["constant", "made-empty", "pandaserver/r.py:274"],
                         ["constant", "constant-through", "pandaserver/r.py:334"],
                         ["stop", "container", "pandaserver/r.py:274"]], []], got
    assert got["d"] == [[["constant", "'x'", "pandaserver/a.py:20"]], ["pandaserver/a.py:20"]], got
    assert got["e"] == [[["constant", "constant-through", "pandaserver/a.py:40"]], ["pandaserver/a.py:40"]], got
    assert got["f"] == [[["constant", "True", "pandaserver/a.py:60"]], ["pandaserver/a.py:60"]], got


_K_ADDER = """\
from pandaserver.exc import RegistrationError


class AdderPlugin:
    def register(self, files):
        for attempt in range(3):
            is_fatal = False
            try:
                out = self.api.add(files)
            except RegistrationError as e:
                is_fatal = e.fatal
            except Exception as e:
                is_fatal = "too large" in str(e)
            if is_fatal:
                self.logger.error("failed")
                return 1
        return 0
"""
_K_DDM = """\
from pandaserver.exc import RegistrationError


def add(files):
    try:
        files[0]
    except KeyError as exc:
        raise RegistrationError("no file", fatal=True) from exc


def check(files):
    raise RegistrationError("not verified")
"""
_K_EXC = """\
class RegistrationError(Exception):
    def __init__(self, message="", fatal=False):
        super().__init__(message)
        self.fatal = fatal
"""


def test_k_a_field_of_the_exception_a_handler_names_is_read_up_the_raise_its_runs_traceback_shows(tmp_path):
    # case A's is_fatal = e.fatal (adder_atlas_plugin 758): the map's object of the class caught at the handler (EXF)
    # read by its field up to the raises that hand it; the run's traceback (the last chain of a raise ... from) names
    # the exception, the handler it entered and the raise; a way that cannot reach the guard through that handler goes
    # (one in the other handler; the False bound before the try, which the handler binds again)
    corpus = _corpus(tmp_path) / "pandaserver"
    for name, text in (("a.py", _K_ADDER), ("ddm.py", _K_DDM), ("exc.py", _K_EXC)):
        (corpus / name).write_text(text)
    lines = _K_ADDER.splitlines()
    at = {k: lines.index(v) + 1 for k, v in {
        "false": "            is_fatal = False", "body": "                out = self.api.add(files)",
        "handler": "            except RegistrationError as e:", "other": '                is_fatal = "too large" in str(e)',
        "test": "            if is_fatal:", "site": '                self.logger.error("failed")'}.items()}
    pkg = "/opt/panda/lib/python3.11/site-packages/pandaserver"
    tb = (f'Traceback (most recent call last):\n  File "{pkg}/ddm.py", line 6, in add\n    files[0]\nKeyError: 0\n\n'
          "The above exception was the direct cause of the following exception:\n\n"
          f'Traceback (most recent call last):\n  File "{pkg}/a.py", line {at["body"]}, in register\n'
          "    out = self.api.add(files)\n"
          f'  File "{pkg}/ddm.py", line 8, in add\n    raise RegistrationError("no file", fatal=True) from exc\n'
          "pandaserver.exc.RegistrationError: no file")
    value = (f'Traceback (most recent call last):\n  File "{pkg}/a.py", line {at["body"]}, in register\n'
             "    out = self.api.add(files)\nValueError: too large")
    out = _ok(_run(tmp_path, f"""
        import json
        from types import SimpleNamespace as N
        from bamboo.codemap import models as M
        from bamboo.codemap.panda import pathcond
        from bamboo.querier import mapdump, rerun
        from bamboo.querier import records as REC
        from bamboo.querier import rows as R
        from bamboo.querier import sources as SR

        R.ROWS_FROM = "walk"
        owner, rel = "pandaserver/a.py::AdderPlugin.register", "pandaserver/a.py"
        cls, init = "pandaserver/exc.py::RegistrationError", "pandaserver/exc.py::__init__"
        src = lambda **k: M.HeldSource(**k)
        mapdump._FRAGMENT[:] = [N(
            guard_froms=[N(owner=owner, test_line={at["test"]}, name="is_fatal", stops=[], sources=[
                src(kind="constant", name="False", line={at["false"]}, at=[f"{{rel}}:{at["false"]}"]),
                src(kind="object", name="RegistrationError", line={at["handler"]}, site_owner=cls, steps=[".fatal"],
                    at=[f"{{rel}}:{at["handler"]}"]),
                src(kind="unknown", name="'too large' in str(e)", line={at["other"]}, at=[f"{{rel}}:{at["other"]}"])])],
            parameter_froms=[N(owner=init, parameter="fatal", line=2, stops=[], sources=[
                src(kind="constant", name="True", line=8, route=["pandaserver/ddm.py:8"], at=["pandaserver/ddm.py:8"]),
                src(kind="constant", name="False", line=2, route=["pandaserver/ddm.py:12"], at=["pandaserver/exc.py:2"])])],
            entered_froms=[], runnables=[],
            field_froms=[N(owner=cls, field="fatal", stops=[], changes=[], bindings=[
                N(at="pandaserver/exc.py:4", owner=init, how="assign", stops=[], sources=[
                    src(kind="parameter", name="fatal", line=2, site_owner=init, site_line=2)])])])]
        SR._calls_in.cache_clear()
        SR._field_index.cache_clear()
        fn = R.function(rel, "register", within=[{at["site"]}])
        stmt = R.statement_at(fn.node, {at["site"]})
        guards = pathcond.arm_condition(stmt)
        pairs = list(zip(guards, rerun._guard_ifs(fn, stmt, guards)))
        choose = lambda ff, *a: N(bindings=ff.bindings, changes=ff.changes, entered_by=lambda at: ())
        r = object.__new__(rerun.Rerun)
        r.lines = []
        r._call_shown = lambda *a, **k: False
        r._entry_value = lambda fn, run, param: None
        end = lambda e: [e.end, e.why, e.at, e.caught, list(e.entered)]
        term = next(t for t in SR.terms(fn, owner, pairs, choose=choose) if t.term == "is_fatal")
        record = lambda tb: REC.Record("adder.log", 1, "2026-09-28 10:25:08,110", "panda.log.adder", "ERROR", "1",
                                       "failed", tb.splitlines())
        run = N(records=[record(json.loads({json.dumps(tb)!r}))])
        k = r._raised(term, fn, run, "")
        said = list(r.lines)
        print(json.dumps({{
            "terms": [end(e) for e in term.ends],
            "raised": [end(e) for e in k.ends],
            "said": said,
            "ended": [end(e) for e in r._with_callers(k, fn, owner, run, None).ends],
            "not traced": [end(e) for e in r._with_callers(r._raised(term, fn, run, ""), fn, owner, N(records=[]),
                                                           None).ends],
            "no run": [end(e) for e in r._raised(term, fn, None, "  ").ends],
            "a ValueError": [end(e) for e in r._raised(term, fn, N(records=[record(json.loads({json.dumps(value)!r}))]),
                                                      "  ").ends],
            "handler line": rerun._handler_line(fn, SR.End("is_fatal", "stop", "object", f"{{rel}}:{at["handler"]}")),
        }}))
    """))
    got = json.loads(out.splitlines()[-1])

    def a(line: int) -> str:
        return f"pandaserver/a.py:{line}"

    caught = ["stop", "parameter", "pandaserver/exc.py:2", at["handler"]]
    assert got["terms"] == [["constant", "False", a(at["false"]), 0, []], [*caught, []],
                            ["stop", "unknown", a(at["other"]), 0, []]], got
    assert got["raised"] == [[*caught, ["pandaserver/ddm.py:8"]]], got
    assert got["said"] == [f"  K: term 'is_fatal' of 'is_fatal': its run's traceback: pandaserver.exc.RegistrationError "
                           f"raised at pandaserver/ddm.py:8, "
                           f"caught by the except handler at {a(at['handler'])}; a way that cannot reach the guard "
                           f"through it goes: {a(at['false'])}, {a(at['other'])}"], got
    assert got["ended"] == [["constant", "True", "pandaserver/ddm.py:8", at["handler"], []]], got
    # the call by the path, no frame of the run read standing at it: its caller's run is asked, as before
    assert got["not traced"] == [["stop", "parameter-unchosen", "", at["handler"], []]], got
    assert got["no run"] == got["terms"], got
    assert got["a ValueError"] == [["stop", "unknown", a(at["other"]), 0, []]], got
    assert got["handler line"] == at["handler"], got


def test_k_an_arm_in_a_handler_whose_terms_end_stops_at_the_raise_its_runs_traceback_shows(tmp_path):
    # K 7339032371's set_fatal at adder_atlas_plugin 170, in 'except Exception' at 165 with no guard: its run's
    # traceback (KeyError raised in decompose_id_map) names the raise the handler took; the arm stops there, no end
    corpus = _corpus(tmp_path) / "pandaserver"
    for name, text in (("a.py", _K_ADDER), ("ddm.py", _K_DDM), ("exc.py", _K_EXC)):
        (corpus / name).write_text(text)
    lines = _K_ADDER.splitlines()
    body = lines.index("                out = self.api.add(files)") + 1
    arm = lines.index('                is_fatal = "too large" in str(e)') + 1
    pkg = "/opt/panda/lib/python3.11/site-packages/pandaserver"
    value = (f'Traceback (most recent call last):\n  File "{pkg}/a.py", line {body}, in register\n'
             "    out = self.api.add(files)\n"
             f'  File "{pkg}/ddm.py", line 6, in add\n    files[0]\nKeyError: 0')
    out = _ok(_run(tmp_path, f"""
        import json
        from types import SimpleNamespace as N
        from bamboo.querier import rerun, render
        from bamboo.querier import records as REC
        from bamboo.querier import rows as R
        from bamboo.querier import sources as SR

        R.ROWS_FROM = "walk"
        fn = R.function("pandaserver/a.py", "register", within=[{arm}])
        r = object.__new__(rerun.Rerun)
        r.lines = []
        record = REC.Record("adder.log", 1, "2026-09-28 10:25:08,110", "panda.log.adder", "DEBUG", "1", "execute()",
                            json.loads({json.dumps(value)!r}).splitlines())
        raised = r._raise_into(fn, {arm}, N(records=[record]), "")
        said = list(r.lines)
        # the run's code another version at the raising frame: the pin holds other text at ddm.py:6
        old = REC.Record("adder.log", 1, "2026-09-28 10:25:08,110", "panda.log.adder", "DEBUG", "1", "execute()",
                         json.loads({json.dumps(value)!r}).replace("files[0]", "files[1]").splitlines())
        other = r._raise_into(fn, {arm}, N(records=[old]), "")
        change = SR.Changed("pandaserver/r.py:55", "method", "*")
        term = lambda made: SR.Term("self.add_result is None", "self.add_result", (SR.End(
            "self.add_result", "stop", "object", "pandaserver/b.py:45", changes=(change,), made=made),))
        made = lambda raised: SR.Made(change, "pandaserver/a.py:{arm}", "pandaserver/a.py::register", **raised)
        print(json.dumps({{
            "raised": raised, "said": said, "other": other,
            "stop": render.text(rerun._first_stop(term(made(raised)))),
            "stop, other": render.text(rerun._first_stop(term(made(other)))),
            "no raise": render.text(rerun._first_stop(term(SR.Made(change, "pandaserver/a.py:{arm}", "x")))),
            "no run": r._raise_into(fn, {arm}, None, "  "),
        }}))
    """))
    got = json.loads(out.splitlines()[-1])
    handler = lines.index("            except Exception as e:") + 1
    caught = {"raised_type": "KeyError", "raised_at": "pandaserver/ddm.py:6", "caught_at": f"pandaserver/a.py:{handler}"}
    assert got["raised"] == {**caught, "raised_other": ""}, got
    assert got["other"] == {**caught, "raised_other": "add: files[1]"}, got
    assert got["said"] == [f"  K: its run's traceback: KeyError raised at pandaserver/ddm.py:6, caught by the except "
                           f"handler at pandaserver/a.py:{handler}"], got
    stop = (f"it stops at 'self.add_result' (in 'self.add_result is None'), in the arm that made the change at "
            f"pandaserver/a.py:{arm}, at the KeyError its run raised at pandaserver/ddm.py:6{{}}, caught by the except "
            f"handler at pandaserver/a.py:{handler} (K: its run's traceback)")
    assert got["stop"] == stop.format(""), got
    assert got["stop, other"] == stop.format(" (add: files[1]: its run's code, not the pin's)"), got
    assert got["no raise"] == "it stops at 'self.add_result' (in 'self.add_result is None')", got
    assert got["no run"] == {}, got


_TRUTHS = """\
class Adder:
    def register(self, files):
        found = []
        for f in files:
            found.append(f)
        found.sort()
        if not found and not files:
            return None
        self.done = True
"""


def test_q6_a_test_reading_a_container_made_empty_for_its_truth_alone_reads_its_start_and_its_puts(tmp_path):
    # case A's 737 'not dest_id_map and not zip_files and not cont_zip_map': whether a put ran is what the test reads,
    # not what it put -- a parameter's way up its callers, a field's, a local's: the empty start and each put (an
    # element's inner change and a reorder leave it as empty as it was); a change that takes out, or one not followed,
    # leaves the container a way; a test reading how many it holds, or an element, is no such read
    (_corpus(tmp_path) / "pandaserver" / "t.py").write_text(_TRUTHS)
    lines = _TRUTHS.splitlines()
    test, site = lines.index("        if not found and not files:") + 1, lines.index("        self.done = True") + 1
    out = _ok(_run(tmp_path, f"""
        import ast, json
        from types import SimpleNamespace as N
        from bamboo.codemap import models as M
        from bamboo.codemap.panda import pathcond
        from bamboo.querier import mapdump, rerun
        from bamboo.querier import rows as R
        from bamboo.querier import sources as SR

        R.ROWS_FROM = "walk"
        owner, rel = "pandaserver/t.py::Adder.register", "pandaserver/t.py"
        found = M.HeldSource(kind="unknown", name="[]", line=3, made_empty="collection", at=[f"{{rel}}:3"], changes=[
            M.HeldChange(at=f"{{rel}}:5", kind="method", key="*", what="append", value=[
                M.HeldSource(kind="parameter", name="f", line=5, site_owner=owner)]),
            M.HeldChange(at=f"{{rel}}:6", kind="method", key="*", what="sort")])
        # case A's dest_id_map: made empty in a callee and returned (674), handed at 677; two stores of a {{}} into it,
        # and the inner store into what one holds
        inner = M.HeldSource(kind="unknown", name="{{}}", line=1039, made_empty="mapping", at=["p/r.py:1039"])
        dest = M.HeldSource(kind="unknown", name="{{}}", line=1031, made_empty="mapping", route=["p/r.py:674", "p/r.py:677"],
                            at=["p/r.py:1031"], changes=[
                                M.HeldChange(at="p/r.py:1039", kind="store", key="tmp_dest", value=[inner]),
                                M.HeldChange(at="p/r.py:1040", kind="inner", key="*"),
                                M.HeldChange(at="p/r.py:1048", kind="store", key="tmp_dest", value=[inner])])
        popped = dest.model_copy(update={{"changes": [*dest.changes, M.HeldChange(at="p/r.py:1050", kind="method", key="*", what="pop")]}})
        gone = dest.model_copy(update={{"changes": [*dest.changes, M.HeldChange(at="p/r.py:1051", kind="escaped", key="*")]}})
        pf = lambda name, s: N(owner="p/r.py::register", parameter=name, line=1, stops=[], sources=[s])
        T = "p/m.py::main.T"
        mapdump._FRAGMENT[:] = [N(
            guard_froms=[N(owner=owner, test_line={test}, name="found", sources=[found], stops=[])],
            parameter_froms=[pf("dest", dest), pf("popped", popped), pf("gone", gone)], entered_froms=[], runnables=[],
            field_froms=[N(owner=T, field="jobs", stops=[], bindings=[
                N(at="p/m.py:151", owner="p/m.py::run", how="assign", stops=[], sources=[
                    M.HeldSource(kind="unknown", name="[]", line=151, made_empty="collection", at=["p/m.py:151"])])],
                changes=[N(at="p/m.py:160", kind="method", key="*", what="append", via=["p/m.py:158"], value=[])])])]
        SR._calls_in.cache_clear()
        SR._field_index.cache_clear()
        fn = R.function(rel, "register", within=[{site}])
        stmt = R.statement_at(fn.node, {site})
        guards = pathcond.arm_condition(stmt)
        pairs = list(zip(guards, rerun._guard_ifs(fn, stmt, guards)))
        t = lambda s: ast.parse(s, mode="eval").body
        way = lambda w: [w.end, w.why, list(w.route), w.end_at]
        end = lambda e: [e.end, e.why, e.at, list(e.route), [c.at for c in e.changes], e.truth]
        print(json.dumps({{
            "truth": [SR.truth_only(t(s), "d") for s in [
                "not d", "d", "not d and not z", "x or not d", "bool(d)", "len(d)", "len(d) > 0", "0 < len(d)",
                "len(d) == 0", "a if d else b", "len(d) > 3", "d[k]", "k in d", "d is None", "d.items()",
                "not d or d[k]", "a if b else d", "z"]],
            "parameter": [way(w) for w in SR.param_ways("p/r.py::register", "dest", truth=True)],
            "whole": sorted({{w.why for w in SR.param_ways("p/r.py::register", "dest")}}),
            "popped": [way(w) for w in SR.param_ways("p/r.py::register", "popped", truth=True)],
            "gone": [way(w) for w in SR.param_ways("p/r.py::register", "gone", truth=True)],
            "field": [way(w) for w in SR.field_ways(T, "jobs", truth=True)],
            "field, whole": [[w.end, w.why, [c.at for c in w.changes]] for w in SR.field_ways(T, "jobs")],
            "local": [end(e) for tm in SR.terms(fn, owner, pairs) if tm.term == "found" for e in tm.ends],
        }}))
    """))
    got = json.loads(out.splitlines()[-1])
    assert got["truth"] == [True] * 10 + [False] * 8, got
    by = ["p/r.py:674", "p/r.py:677"]
    puts = [["constant", "put", by, "p/r.py:1039"], ["constant", "put", by, "p/r.py:1048"]]
    assert got["parameter"] == [["constant", "made-empty", by, "p/r.py:1031"], *puts], got
    # read whole (Q1), as before: the values put, and the mapping's keys a stop
    assert got["whole"] == ["container", "made-empty"], got
    assert got["popped"] == [*got["parameter"], ["stop", "container", by, "p/r.py:1031"]], got
    assert got["gone"] == got["popped"], got
    # a field's: the binding's empty start, and the record's put placed at the put through the call it was made by
    assert got["field"] == [["constant", "made-empty", ["p/m.py:151"], "p/m.py:151"],
                            ["constant", "put", ["p/m.py:151", "p/m.py:158", "p/m.py:160"], "p/m.py:160"]], got
    # read whole (Q1), as before: the record's put a change on the start, for S4c to read on the run (G4)
    assert got["field, whole"] == [["constant", "made-empty", ["p/m.py:160"]]], got
    assert got["local"] == [["constant", "made-empty", "pandaserver/t.py:3", [], [], False],
                            ["constant", "put", "pandaserver/t.py:5", [], [], False]], got


def test_q6_a_container_its_run_printed_is_its_start_or_its_puts_and_a_term_an_earlier_operand_decided_is_not_read(tmp_path):
    # (b): what the run printed at the test (V's) chooses -- empty, the start (and the container, where a change may
    # have taken out); not empty, the puts; no print, or none of a container, as it was.  (c): Python reads 'not files'
    # only where 'not found' was True: where the run printed found not empty, files is not read, one end naming the
    # operand; an operand whose value is not known stops the reading there
    (_corpus(tmp_path) / "pandaserver" / "t.py").write_text(_TRUTHS)
    lines = _TRUTHS.splitlines()
    test = lines.index("        if not found and not files:") + 1
    out = _ok(_run(tmp_path, f"""
        import ast, json
        from bamboo.querier import rerun, render
        from bamboo.querier import rows as R
        from bamboo.querier import sources as SR

        R.ROWS_FROM = "walk"
        rel = "pandaserver/t.py"
        fn = R.function(rel, "register", within=[{test}])
        r = object.__new__(rerun.Rerun)
        r.lines = []
        guard = "not (not found and (not files))"
        start = SR.End("found", SR.CONSTANT, SR.MADE_EMPTY, f"{{rel}}:3")
        put = SR.End("found", SR.CONSTANT, SR.PUT, f"{{rel}}:5")
        taken = SR.End("found", SR.STOP, SR.CONTAINER, f"{{rel}}:3")
        files = SR.End("files", SR.STOP, "parameter", "", param="files")
        terms = [SR.Term(guard, "found", (start, put, taken), {test}), SR.Term(guard, "files", (files,), {test})]

        def read(printed):
            got = r._read_by_its_run(fn, terms, {{({test}, k): (v, "row x") for k, v in printed.items()}}, "  ")
            return [[u.term, [rerun._way_text(e) for e in u.ends], u.ended] for u in got]

        t = lambda s: ast.parse(s, mode="eval").body
        unread = SR.End("files", SR.UNREAD, "short-circuit", f"{{rel}}:{test}", decided="not found")
        said = rerun._way_said(unread)
        print(json.dumps({{
            "filled": read({{"found": "['a.root']"}}),
            "empty": read({{"found": "[]"}}),
            "no print": read({{}}),
            "not a container": read({{"found": "None"}}),
            "short": [rerun._short_circuit(t(s), n, {{k: (v, "row x") for k, v in p.items()}}) for s, n, p in [
                ("not a and not b", "b", {{"a": "{{'x': 1}}"}}),
                ("not a and not b", "b", {{"a": "{{}}"}}),
                ("a or b", "b", {{"a": "set()"}}),
                ("a or b", "b", {{"a": "[1]"}}),
                ("a and b or c", "b", {{"a": "0"}}),
                ("a and b or c", "c", {{"a": "0"}}),
                ("not a and not b", "a", {{"a": "{{'x': 1}}"}}),
                ("x and b", "b", {{}}),
                ("not a and (b or c)", "c", {{"a": "1"}}),
                ("not a and b and a", "a", {{"a": "1"}})]],
            "empty?": [rerun._printed_empty(v) for v in
                       ["{{}}", "[]", "()", "set()", "{{'a': 1}}", "[<x>]", "None", "3", "{{'a': datetime(1)}}"]],
            "said": render.text(said),
            "said, back": render.text(render.from_json(render.to_json(said))),
            "ended": SR.Term(guard, "files", (unread,), {test}).ended,
            "lines": r.lines,
        }}))
    """))
    got = json.loads(out.splitlines()[-1])
    at = f"pandaserver/t.py:{test}"
    unread = [f"unread (short-circuit at {at}): 'not found' decided the test on its run"]
    assert got["filled"] == [["found", ["constant (put at pandaserver/t.py:5)"], True], ["files", unread, True]], got
    assert got["empty"] == [["found", ["constant (made-empty at pandaserver/t.py:3)", "stop (container at pandaserver/t.py:3)"], False],
                            ["files", ["stop (parameter)"], False]], got
    as_it_was = [["found", ["constant (made-empty at pandaserver/t.py:3)", "constant (put at pandaserver/t.py:5)",
                            "stop (container at pandaserver/t.py:3)"], False], ["files", ["stop (parameter)"], False]]
    assert got["no print"] == as_it_was, got
    assert got["not a container"] == as_it_was, got
    assert got["short"] == ["not a", None, None, "a", "a", None, None, None, "not a", None], got
    assert got["empty?"] == [True, True, True, True, False, False, None, None, False], got
    assert got["said"] == f"unread (short-circuit) at {at}: 'not found' decided the test on its run", got
    assert got["said, back"] == got["said"], got
    assert got["ended"] is True, got
    assert f"  term 'files' of {'not (not found and (not files))'!r}: its run's test did not read it: 'not found' decided the test" in got["lines"], got
    assert any(x.startswith("  term 'found' of ") and "its run printed it not empty at its test (row x)" in x for x in got["lines"]), got


def test_q7_a_part_the_test_takes_ends_at_the_change_left_where_the_objects_start_held_another_value(tmp_path):
    # Q7 (out/gm/qk-design.md) on S4c's case A shape: the test takes .code off self.out, a Res() made at Plugin's
    # __init__; the run printed 2; register's set_fatal is left by elimination.  (b): where the map's FieldFrom of
    # Res.code says __init__ binds None, the start is not what the run printed, so .code ends at the 2 set_fatal
    # stores, read as an arm as before (its term is_fatal stops, so the term does not end); where __init__ binds 2, the
    # run does not tell the change from the start; where the map holds no FieldFrom of Res.code, the start is not read.
    # Both stop where they did and say why.
    (_corpus(tmp_path) / "pandaserver" / "g8.py").write_text(_GEN8)
    lines = _GEN8.splitlines()
    ln = {k: lines.index(t) + 1 for k, t in {
        "fatal": "        self.code = self.fatal", "temp": "        self.code = self.temporary",
        "rfatal": "    fatal = 2", "rtemp": "    temporary = 1", "pres": "        self.result = Res()",
        "init_code": "        self.code = None", "if_fatal": "        if is_fatal:",
        "init_fatal": "        is_fatal = False", "except": "        except Exception as e:", "in_handler": "            is_fatal = e.fatal",
        "reg_fatal": "            self.result.set_fatal()", "reg_temp": "            self.result.set_temporary()",
        "sub_fatal": "        self.result.set_fatal()", "plugin": "        self.plugin = Plugin(self.logger)",
        "out": "        self.out = self.plugin.result", "fail": "            fail()"}.items()}
    ln["test"] = ln["fail"] - 1
    stamp = "2026-01-01 00:00:0{},000 panda.log.X: DEBUG    <PandaID=1> {}\n"
    (tmp_path / "one.log").write_text("".join(stamp.format(n, t) for n, t in enumerate(["register", "plugin done with 2"])))
    out = _ok(_run(tmp_path, f"""
        import json
        from types import SimpleNamespace as N
        from bamboo.codemap.panda import pathcond
        from bamboo.querier import mapdump, rerun
        from bamboo.querier import render as T
        from bamboo.querier import rows as R
        from bamboo.querier import sources as SR
        from bamboo.querier.records import read_log, runs_for

        R.ROWS_FROM = "walk"
        g = "pandaserver/g8.py"
        LN = {ln!r}
        RES = f"{{g}}::Res"
        src = lambda kind, name, line, **kw: N(kind=kind, name=name, line=line, changes=[], site_owner=kw.pop("site_owner", ""),
                                               site_line=None, paths=[], steps=[], through=[], route=[], **kw)
        run = runs_for(read_log("one.log"), "1")[0]
        printing = {{f"{{g}}::{{f}}": N(statements=[N(log_files=[run.log.name], log_files_basis="logger")])
                     for f in ("register", "subscribe")}}
        R._prints_of = lambda owner, lines=(): printing.get(owner)
        klass = lambda field, line, value: N(owner=RES, field=field, changes=[], stops=[], bindings=[N(
            at=f"{{g}}:{{LN[line]}}", owner=RES, how="class", stops=[], sources=[src("constant", value, LN[line])])])
        init = lambda value: N(owner=RES, field="code", changes=[], stops=[], bindings=[N(
            at=f"{{g}}:{{LN['init_code']}}", owner=f"{{g}}::__init__", how="assign", stops=[],
            sources=[src("constant", value, LN["init_code"])])])
        store = lambda at, field, via: SR.Changed(
            f"{{g}}:{{LN[at]}}", "store", ".code", (f"{{g}}:{{LN[via]}}",),
            N(kind="store", value=[src("self_field", f"self.{{field}}", LN[at], site_owner=RES)], value_stops=[]),
            f"{{g}}:{{LN['out']}}")
        way = SR.End("self.out", "stop", "object", f"{{g}}:{{LN['pres']}}", via=(f"{{g}}:{{LN['out']}}", f"{{g}}:{{LN['plugin']}}"),
                     changes=(store("fatal", "fatal", "reg_fatal"), store("temp", "temporary", "reg_temp"),
                              store("fatal", "fatal", "sub_fatal")), made_of=RES)
        fn = R.function(g, "finalize", within=[LN["fail"]])
        site = R.statement_at(fn.node, LN["fail"])
        guards = pathcond.arm_condition(site)
        term = SR.Term(guards[0], "self.out", (way,), LN["test"])
        got = {{}}
        for name, code in (("start None", [init("None")]), ("start 2", [init("2")]), ("no record", [])):
            mapdump._FRAGMENT[:] = [N(
                guard_froms=[N(owner=f"{{g}}::register", test_line=LN["if_fatal"], name="is_fatal",
                               stops=[], sources=[src("constant", "False", LN["init_fatal"]),
                                                  src("attribute", "e.fatal", LN["in_handler"])])],
                runnables=[N(owner=f"{{g}}::ok", line=0, bound=[RES], asked=["condition"], asked_at=[f"{{g}}:{{LN['test']}}"],
                             runnable=True, reads=["self.code"], stops=[])],
                field_froms=[klass("fatal", "rfatal", "2"), klass("temporary", "rtemp", "1"), *code], function_prints=[])]
            r = object.__new__(rerun.Rerun)
            r.lines, r.mode, r.decide = [], "outside", "positive"
            printed = {{}}
            r._held_on(fn, site, guards, run, printed)
            t = r._s4c(term, fn, printed, run, (run.first[:19], run.last[:19]), frozenset(), "  ")
            e = t.ends[0]
            m = e.made
            said = rerun._way_said(e)
            got[name] = {{
                "end": [e.end, e.why, e.at, e.part, e.made_at, [c.at for c in e.changes]],
                "made": None if m is None else [m.how, m.part, list(m.start), m.printed_start, m.no_record],
                "ended": t.ended, "stop": T.text(rerun._first_stop(t)), "said": T.text(said),
                "back": T.text(T.from_json(T.to_json(said))) == T.text(said),
                "q7": [x for x in r.lines if x.strip().startswith("Q7:")]}}
        print(json.dumps(got))
    """))
    got = json.loads(out.splitlines()[-1])
    at = {k: f"pandaserver/g8.py:{v}" for k, v in ln.items()}
    elim = ("left by elimination (it stores what the run printed, its function's rows are in the run, and every other "
            "change is shown not to have run or to store another value)")
    arm = (f"read as an arm at {at['reg_fatal']}: its terms is_fatal -> constant (False) at {at['init_fatal']}, stop "
           f"(attribute) at {at['in_handler']}, in the except handler at {ln['except']} (S3: which exception); its entry: "
           f"stop (no-entered-from)")
    stop = (f"it stops at 'is_fatal' (in 'is_fatal'), in the arm that made the change at {at['reg_fatal']}, bound in the "
            f"except handler at {ln['except']} (S3: which exception)")
    one = got["start None"]
    assert one["end"] == ["constant", "2", at["fatal"], ".code", at["pres"], []], one
    assert one["made"] == ["left", ".code", ["None"], False, ""], one
    # the arm's term stops, so the term does not end: the walk goes down into the arm as before
    assert one["ended"] is False and one["stop"] == stop, one
    assert one["said"] == (
        f"constant (2) at {at['fatal']}: its .code as written there, of the object made at {at['pres']}, through the "
        f"field bindings at {at['out']} -> {at['plugin']}, changed between it and the read by {at['fatal']} made by "
        f"{at['reg_fatal']}, {elim}, and what the object held of .code at its start (None) is not what the run printed; "
        f"{arm}"), one["said"]
    assert one["back"] is True, one
    assert one["q7"] == [f"    Q7: self.out.code last written by {at['fatal']} made by {at['reg_fatal']}: 2 (its start "
                         f"held None, not the printed '2')"], one
    two = got["start 2"]
    assert two["end"] == ["stop", "object", at["pres"], "", "", [at["fatal"]]], two
    assert two["made"] == ["left", ".code", ["2"], True, ""] and two["ended"] is False and two["stop"] == stop, two
    assert (f", {elim}, but what the object held of .code at its start (2) is what the run printed too: the run does "
            f"not tell the change from the start; {arm}") in two["said"], two["said"]
    none = got["no record"]
    assert none["end"] == ["stop", "object", at["pres"], "", "", [at["fatal"]]], none
    assert none["made"] == ["left", ".code", [], False, "pandaserver/g8.py::Res.code"], none
    assert (f", {elim}, but what the object held of .code at its start is not read (the map holds no FieldFrom of "
            f"pandaserver/g8.py::Res.code); {arm}") in none["said"], none["said"]
    assert none["back"] is True and two["back"] is True, got


def test_q7_an_object_filled_from_a_row_ends_at_the_row_where_its_held_test_rules_every_later_write_out(tmp_path):
    # Q7's (a) and (f) on killJob's shape: job = JobSpec() at 13 and job.pack(res[0]) at 14 fill job.jobStatus from the
    # row; 17 writes "closed" after it, so the fill is not the only write (the row-fill reading alone says no), and the
    # test at 18 reads job.jobStatus != "closed", which the arm at 19 says its run held.  The fill is the start, 17 the
    # change on it; "closed" would turn the held test false, so 17 is not the last writer, and the term ends at the
    # row.  (f) on literals (K75's premise 3): parse_result < 2 held, so not 2; a test another unknown name can decide
    # does not tell.  (d): a way at a change ends where its arm's terms and entry end and no raise is shown.
    corpus = _corpus(tmp_path)
    (corpus / "pandaserver" / "__init__.py").write_text("")
    (corpus / "pandaserver" / "spec.py").write_text(_ROW_SPEC)
    (corpus / "pandaserver" / "proxy.py").write_text(_ROW_PROXY)
    lines = _ROW_PROXY.splitlines()
    assert [lines[i - 1].strip() for i in (13, 14, 17, 18, 19)] == [
        "job = JobSpec()", "job.pack(res[0])", 'job.jobStatus = "closed"', 'if job.jobStatus != "closed":',
        'job.jobStatus = "cancelled"']
    out = _ok(_run(tmp_path, """
        import json
        from dataclasses import replace
        from types import SimpleNamespace as N
        from bamboo.codemap.panda import pathcond
        from bamboo.querier import mapdump, rerun
        from bamboo.querier import rows as R
        from bamboo.querier import sources as SR

        R.ROWS_FROM = "walk"
        rel = "pandaserver/proxy.py"
        owner = f"{rel}::killJob"
        s = lambda kind, name, line, changes=(): N(kind=kind, name=name, line=line, changes=list(changes), site_owner="",
                                                   site_line=None, paths=[], steps=[], through=[], route=[])
        fill = N(at="pandaserver/spec.py:10", kind="method", key="*", via=[f"{rel}:14"])
        closed = N(at=f"{rel}:17", kind="store", key=".jobStatus", via=[], value=[s("constant", '"closed"', 17)], value_stops=[])
        mapdump._FRAGMENT[:] = [N(guard_froms=[N(owner=owner, test_line=18, name="job", stops=[], sources=[
            s("object", "JobSpec()", 13, [fill, closed])])], runnables=[], field_froms=[], function_prints=[])]
        SR._index.cache_clear()
        fn = R.function(rel, "killJob", within=[19])
        stmt = R.statement_at(fn.node, 19)
        guards = pathcond.arm_condition(stmt)
        terms = SR.terms(fn, owner, list(zip(guards, rerun._guard_ifs(fn, stmt, guards))))
        t = next(u for u in terms if u.term == "job" and u.line == 18)
        e = t.ends[0]
        r = object.__new__(rerun.Rerun)
        r.lines = []
        u = r._s4c(t, fn, {}, None, ("", ""), frozenset(), "  ")
        held = rerun._held_test
        c = lambda v, at: SR.End("parse_result", SR.CONSTANT, v, f"x.py:{at}")
        two = (c("0", 1), c("2", 2))
        drops = lambda guard, ends=two, term="parse_result": [x.why for x in r._held_drops(SR.Term(guard, term, ends, 5), "  ").ends]
        ch = SR.Changed("a.py:5", "store", ".code")
        done = SR.Term("x", "x", (SR.End("x", SR.CONSTANT, "1", "a.py:1"),), 1)
        stopped = SR.Term("x", "y", (SR.End("y", SR.STOP, "call", "a.py:2"),), 1)
        entry = SR.End("f", SR.CALLER, "entered-by", "a.py:9")
        made = lambda **kw: SR.Made(ch, "a.py:7", "a.py::f", "left", **kw)
        way = lambda m: SR.Term("g", "o", (SR.End("o", SR.CONSTANT, "2", "a.py:5", made=m, part=".code", made_at="a.py:3"),), 1)
        print(json.dumps({
            "guard": t.guard,
            "way": [e.end, e.why, list(e.filled), [x.at for x in e.changes], t.ended],
            "s4c": [[x.end, x.why, x.at, [y.at for y in x.changes]] for x in u.ends], "ended": u.ended,
            "said": [x for x in r.lines if x.strip().startswith("Q7:")],
            "rules out": [rerun._held_rules_out(held(g), n, v) for g, n, v in [
                ("job.jobStatus != 'closed'", "job.jobStatus", ('"closed"',)),
                ("job.jobStatus != 'closed'", "job.jobStatus", ('"cancelled"',)),
                ("job.jobStatus != 'closed' or code", "job.jobStatus", ('"closed"',)),
                ("code and job.jobStatus != 'closed'", "job.jobStatus", ('"closed"',)),
                ("not (job is None)  [at 387]", "job", ("None",)),
                ("x.f()", "x", ("1",)),
                ("a < 2", "a", ("2", "0")),
                ("a < 2", "a", ("f(1)",))]],
            "drops": [drops("parse_result < 2"), drops("parse_result < 2 or other"), drops("parse_result < 5"),
                      drops("parse_result > 5"), drops("not status", (c("True", 1), c("False", 2)), "status"),
                      drops("parse_result < 2", (*two, SR.End("parse_result", SR.CONSTANT, SR.THROUGH, "x.py:3")))],
            "arm": {
                "ends": way(made(terms=(done,), entry=entry)).ended,
                "a term stops": way(made(terms=(done, stopped), entry=entry)).ended,
                "its entry stops": way(made(terms=(done,), entry=SR.End("f", SR.STOP, "entry-unchosen", ""))).ended,
                "its entry not read": way(made(terms=(done,))).ended,
                "a raise": way(made(terms=(done,), entry=entry, raised_at="a.py:8")).ended,
                "none": SR.Term("g", "o", (SR.End("o", SR.CONSTANT, "2", "a.py:5"),), 1).ended},
            "lines": r.lines,
        }))
    """))
    got = json.loads(out.splitlines()[-1])
    assert got["guard"] == "job.jobStatus != 'closed'", got
    # the fill is the start, 17 the change after it
    assert got["way"] == ["stop", "object", ["pandaserver/proxy.py:14"], ["pandaserver/proxy.py:17"], False], got
    assert got["s4c"] == [["row", "object", "pandaserver/proxy.py:13", []]] and got["ended"] is True, got
    assert any("F: pandaserver/proxy.py:17 stores \"closed\": job.jobStatus so would turn \"job.jobStatus != 'closed'\", "
               "which its run held, the other way (Q7's (f))" in x for x in got["lines"]), got["lines"]
    assert got["said"] == ["    Q7: every change ruled out: what the test takes is the row pandaserver/proxy.py:14 put"], got
    assert got["rules out"] == [True, False, False, True, True, False, False, False], got
    assert got["drops"] == [["0"], ["0", "2"], ["0", "2"], ["0", "2"], ["False"], ["0", "constant-through"]], got
    assert any("term 'parse_result' of 'parse_result < 2': its run held the test, so not 2 at x.py:2 (Q7's (f))" in x
               for x in got["lines"]), got["lines"]
    assert got["arm"] == {"ends": True, "a term stops": False, "its entry stops": False, "its entry not read": False,
                          "a raise": False, "none": True}, got


def test_snd_a_way_across_a_send_carries_the_send_and_the_endpoint_it_entered(tmp_path):
    # SND: an endpoint's parameter the map marks sent (the client's call last on its route) marks every way read up
    # through it with that send and the endpoint; a way not across a send is not marked.  The sends into an endpoint,
    # the calls into a function by its qualified name (a method of the same bare name is another function), and a
    # def's line by its ParameterFrom records
    out = _ok(_run(tmp_path, """
        import json
        from types import SimpleNamespace as N
        from bamboo.codemap import models as M
        from bamboo.querier import mapdump
        from bamboo.querier import sources as SR

        A, C, T = "pandaserver/a.py::kill", "pandaserver/c.py::kill_jobs", "pandaserver/t.py::killJobs"
        mapdump._FRAGMENT[:] = [N(
            guard_froms=[], runnables=[], field_froms=[], entered_froms=[], tables=[],
            parameter_froms=[
                N(owner=T, parameter="code", line=3, stops=[], sources=[
                    M.HeldSource(kind="parameter", name="code", line=10, site_owner=A, route=["pandaserver/a.py:12"]),
                    M.HeldSource(kind="constant", name="'60'", line=9, route=["pandaserver/g.py:9"])]),
                N(owner=A, parameter="code", line=10, stops=[], sources=[
                    M.HeldSource(kind="parameter", name="code", line=5, site_owner=C, route=["pandaserver/c.py:7"], sent=True),
                    M.HeldSource(kind="constant", name="None", line=10, route=["pandaserver/c.py:7"], sent=True,
                                 at=["pandaserver/a.py:10"])]),
                N(owner=C, parameter="code", line=5, stops=[], sources=[
                    M.HeldSource(kind="constant", name="'4'", line=20, route=["pandaserver/d.py:20"])]),
            ],
            reach_graph=[N(owner="pandaserver/d.py::main", calls=[N(to=C, line=20, lines=[20, 30])]),
                         N(owner="pandaserver/d.py::Daemon.run#2", calls=[N(to="pandaserver/c.py::Client.kill_jobs", line=40, lines=[])]),
                         N(owner="pandaserver/e.py::other", calls=[N(to="pandaserver/z.py::f", line=1, lines=[])])])]
        SR._calls_in.cache_clear()
        SR._sends_in.cache_clear()
        got = {
            "ways": sorted([w.value, list(w.route), w.sent, w.sent_into] for w in SR.param_ways(T, "code")),
            "sends into the endpoint": list(SR.sends_into(A)),
            "sends into the mover's caller": list(SR.sends_into(T)),
            "calls into the client": [[o, list(n)] for o, n in SR.calls_into("pandaserver/c.py", "kill_jobs")],
            "calls into the method": [[o, list(n)] for o, n in SR.calls_into("pandaserver/c.py", "Client.kill_jobs")],
            "def line": [SR.def_line(A), SR.def_line("pandaserver/z.py::f")],
        }
        print(json.dumps(got))
    """))
    got = json.loads(out.splitlines()[-1])
    assert got["ways"] == [
        ["'4'", ["pandaserver/d.py:20", "pandaserver/c.py:7", "pandaserver/a.py:12"], "pandaserver/c.py:7", "pandaserver/a.py::kill"],
        ["'60'", ["pandaserver/g.py:9"], "", ""],
        ["None", ["pandaserver/c.py:7", "pandaserver/a.py:12"], "pandaserver/c.py:7", "pandaserver/a.py::kill"],
    ], got
    assert got["sends into the endpoint"] == ["pandaserver/c.py:7"]
    assert got["sends into the mover's caller"] == []
    assert got["calls into the client"] == [["pandaserver/d.py::main", [20, 30]]]
    assert got["calls into the method"] == [["pandaserver/d.py::Daemon.run", [40]]]
    assert got["def line"] == [10, 0]


def test_hop_reader_asks_a_caller_across_a_send_on_every_machine_from_slack_before_to_the_entrys_first_line(tmp_path):
    # HOP's reader (out/gm/qk-design.md HOP (2)): a way not across a send is asked as before, on the arm's window and
    # machine; one across a send on every machine, from windows.SLACK before the arm's window to the first line of the
    # endpoint's pass that holds the key and began last in that window, else to the window's end; the endpoint asked
    # once per window.  A mark many calls share is one run of the day, cut into the endpoint's passes: a pass of other
    # jobs in the window, or of the key outside it, closes nothing.  Where the window before the send shows no call, the
    # caller's rows after its call are asked from the pass's first line to its last and windows.SLACK more (HOP (a′))
    (_corpus(tmp_path) / "pandaserver" / "a.py").write_text(textwrap.dedent('''
        def kill(req, job_ids):
            tmp_logger = LogWrapper(_logger, "kill")
            tmp_logger.debug(f"Start job_ids: {job_ids}")
            ret = global_task_buffer.killJobs(job_ids)
            tmp_logger.debug(f"Done with ret: {ret}")
            return ret
    ''').lstrip())
    line = "2026-11-02 {} panda.log.api_job: DEBUG    kill {}\n"
    calls = [("08:00:00,000", "[777]"), ("15:45:00,000", "[5]"), ("15:51:58,491", "[777, 8]"), ("15:52:30,000", "[777]")]
    (tmp_path / "panda-api_job.log").write_text("".join(
        line.format(at, f"Start job_ids: {ids}") + line.format(at[:-3] + "900", "Done with ret: [True]") for at, ids in calls))
    # a run of the key's one line, as an ask by the key alone finds it: one pass, begun at its entry row
    (tmp_path / "one.log").write_text(line.format("15:51:58,491", "Start job_ids: [777, 8]"))
    # a run of the key with no entry row: no pass of it begins in these records
    (tmp_path / "done.log").write_text(line.format("15:51:58,900", "Done with ret: [True] job_ids: [777]"))
    out = _ok(_run(tmp_path, """
        import json
        from dataclasses import replace
        from types import SimpleNamespace as N
        from bamboo.querier import rerun
        from bamboo.querier import rows as R
        from bamboo.querier import sources as SR
        from bamboo.querier.records import read_log, runs_for

        R.ROWS_FROM = "walk"
        r = object.__new__(rerun.Rerun)
        r.lines = []
        r.say = r.lines.append
        r.args = N(key="777")
        r.mode = "outside"
        r.logs = N(names=lambda files: list(files))
        r.files_of = lambda owner, today, lines: today
        r._keep_unanswered = lambda unanswered: None
        rerun.owners_of = lambda ids: {}
        shown, asked = [], []

        shows = {"before": True}

        def call_shown(site, window, machine, why="for-a-call-handing-a-value", facts=None, after_call=False):
            shown.append([site, list(window), machine, why, facts, after_call])
            return True if after_call else shows["before"]

        r._call_shown = call_shown
        rerun.R._prints_of = lambda owner, lines: N(statements=[N(log_files=["panda-api_job.log"])])
        SR.def_line = lambda owner: 1
        # one run of the mark's day on the machine, the key on three of its lines
        entry = {"panda-api_job.log": runs_for(read_log("panda-api_job.log"), "777")}
        whole = [[len(x.records), len(x.found_by)] for x in entry["panda-api_job.log"]]

        def ask(key, names, asking, owners=None, whole=False):
            asked.append([list(asking.window), asking.machine, str(asking.why)])
            return entry, [], {}

        r._ask = ask
        arm = N(machine="aipanda090")
        window = ("2026-11-02 15:51:59", "2026-11-02 15:51:59")
        sent = SR.Way("constant", "'10'", "constant", ("pandaserver/d.py:634", "pandaserver/c.py:7", "pandaserver/a.py:4"),
                      sent="pandaserver/c.py:7", sent_into="pandaserver/a.py::kill")
        r._caller_shows(SR.Way("constant", "'60'", "constant", ("pandaserver/g.py:9",)), window, arm)
        r._caller_shows(sent, window, arm)
        r._caller_shows(replace(sent, route=("pandaserver/d.py:904", *sent.route[1:])), window, arm)
        # the window before the send shows no call: its rows after the call, asked after the pass (no new ask of the entry)
        shows["before"] = False
        r._caller_shows(sent, window, arm)
        shows["before"] = True
        # in this window only a pass of other jobs began (15:45); the key's pass at 08:00 is outside it
        r._caller_shows(sent, ("2026-11-02 15:50:00", "2026-11-02 15:50:00"), arm)
        entry = {"panda-api_job.log": runs_for(read_log("one.log"), "777")}
        r._caller_shows(sent, ("2026-11-02 15:52:00", "2026-11-02 15:52:00"), arm)
        entry = {"panda-api_job.log": runs_for(read_log("done.log"), "777")}
        r._caller_shows(sent, ("2026-11-02 15:52:01", "2026-11-02 15:52:01"), arm)
        entry = {}
        # no pass of the entry: nothing after it to ask
        shows["before"] = False
        r._caller_shows(sent, ("2026-11-02 16:34:45", "2026-11-02 16:34:45"), arm)
        print(json.dumps({"whole": whole, "shown": shown, "asked": asked, "lines": r.lines}))
    """))
    got = json.loads(out.splitlines()[-1])
    assert got["whole"] == [[8, 3]], got
    across = ["2026-11-02 15:41:59", "2026-11-02 15:51:58,491"]
    send = {"send": "pandaserver/c.py:7"}
    after = ["2026-11-02 15:51:58,491", "2026-11-02 16:01:58"]
    assert got["shown"] == [
        ["pandaserver/g.py:9", ["2026-11-02 15:51:59", "2026-11-02 15:51:59"], "aipanda090", "for-a-call-handing-a-value", None, False],
        ["pandaserver/d.py:634", across, None, "for-a-call-across-a-send", send, False],
        ["pandaserver/d.py:904", across, None, "for-a-call-across-a-send", send, False],
        ["pandaserver/d.py:634", across, None, "for-a-call-across-a-send", send, False],
        ["pandaserver/d.py:634", after, None, "for-the-rows-after-a-call-across-a-send", send, True],
        ["pandaserver/d.py:634", ["2026-11-02 15:40:00", "2026-11-02 15:50:00"], None, "for-a-call-across-a-send", send, False],
        ["pandaserver/d.py:634", ["2026-11-02 15:42:00", "2026-11-02 15:51:58,491"], None, "for-a-call-across-a-send", send, False],
        ["pandaserver/d.py:634", ["2026-11-02 15:42:01", "2026-11-02 15:52:01"], None, "for-a-call-across-a-send", send, False],
        ["pandaserver/d.py:634", ["2026-11-02 16:24:45", "2026-11-02 16:34:45"], None, "for-a-call-across-a-send", send, False],
    ], got
    why = "for the run of kill the send at pandaserver/c.py:7 entered, its first line closing the caller's window"
    assert got["asked"] == [[["2026-11-02 15:41:59", "2026-11-02 15:51:59"], None, why],
                            [["2026-11-02 15:40:00", "2026-11-02 15:50:00"], None, why],
                            [["2026-11-02 15:42:00", "2026-11-02 15:52:00"], None, why],
                            [["2026-11-02 15:42:01", "2026-11-02 15:52:01"], None, why],
                            [["2026-11-02 16:24:45", "2026-11-02 16:34:45"], None, why]], got
    none = "the window's end: no pass of the entry holding the key began in it"
    assert got["lines"] == [
        "    across the send at pandaserver/c.py:7 into kill: its callers asked from 2026-11-02 15:41:59 to "
        "2026-11-02 15:51:58,491 (the first line of the entry's pass that holds the key), on every machine",
        "    and a caller's rows after its call from 2026-11-02 15:51:58,491 to 2026-11-02 16:01:58 (the pass's last line, "
        "2026-11-02 15:51:58,900, and 10 min more), where the window before shows no call",
        "    across the send at pandaserver/c.py:7 into kill: its callers asked from 2026-11-02 15:40:00 to "
        f"2026-11-02 15:50:00 ({none}), on every machine",
        "    across the send at pandaserver/c.py:7 into kill: its callers asked from 2026-11-02 15:42:00 to "
        "2026-11-02 15:51:58,491 (the first line of the entry's pass that holds the key), on every machine",
        "    and a caller's rows after its call from 2026-11-02 15:51:58,491 to 2026-11-02 16:01:58 (the pass's last line, "
        "2026-11-02 15:51:58,491, and 10 min more), where the window before shows no call",
        "    across the send at pandaserver/c.py:7 into kill: its callers asked from 2026-11-02 15:42:01 to "
        f"2026-11-02 15:52:01 ({none}), on every machine",
        "    across the send at pandaserver/c.py:7 into kill: its callers asked from 2026-11-02 16:24:45 to "
        f"2026-11-02 16:34:45 ({none}), on every machine",
    ], got


def test_u2x_reads_a_caller_across_a_send_and_sets_aside_by_its_way_the_pass_that_moved_it(tmp_path):
    # U2X (out/gm/qk-design.md U2X): (A) an entry whose outermost hop is an endpoint a send enters goes on once to the
    # sender's callers, the lines across the send named; (B) a pass of such a caller ending within windows.SLACK of the
    # move that took a call across the send moved it on: set aside, and with no pass before it the caller is read as
    # absent up to it.  Not by time alone: a pass in that span that went another way, or one before the span that took
    # the call, is read as before; a caller with no send is read as before
    corpus = _corpus(tmp_path)
    files = {
        "t.py": "class TaskBuffer:\n    def killJobs(self, ids, code):\n        return proxy.killJob(ids, code)\n",
        "a.py": "def kill(req, job_ids, code=None):\n    tb = TaskBuffer()\n    return tb.killJobs(job_ids, code)\n",
        "c.py": "def kill_jobs(ids, code=None):\n    data = {'job_ids': ids, 'code': code}\n    return HttpClient().post(url, data)\n",
        "d.py": ("def main():\n    if pending:\n        log('killJobs for Pending')\n        Client.kill_jobs(jobs, 4)\n"
                 "    taskBuffer.killJobs(others, 2)\n    return 0\n"),
    }
    for name, text in files.items():
        (corpus / "pandaserver" / name).write_text(text)
    out = _ok(_run(tmp_path, """
        import json
        from types import SimpleNamespace as N
        from bamboo.codemap import models as M
        from bamboo.querier import mapdump, rerun, which_ran
        from bamboo.querier import render as T
        from bamboo.querier import sources as SR

        P = "pandaserver/"
        mapdump._FRAGMENT[:] = [N(
            guard_froms=[], runnables=[], field_froms=[], entered_froms=[], tables=[],
            parameter_froms=[N(owner=P + "a.py::kill", parameter="code", line=1, stops=[], sources=[
                M.HeldSource(kind="parameter", name="code", line=1, site_owner=P + "c.py::kill_jobs", route=[P + "c.py:3"], sent=True)])],
            reach_graph=[N(owner=P + "d.py::main", calls=[N(to=P + "c.py::kill_jobs", line=4, lines=[4])]),
                         N(owner=P + "x.py::Other.run", calls=[N(to=P + "c.py::Other.kill_jobs", line=9, lines=[9])])])]
        SR._calls_in.cache_clear()
        SR._sends_in.cache_clear()
        rerun.prints_nothing = lambda owner, lines: False
        rerun.key_marks = lambda owner, line, cls, key: ()
        rerun.prints_key = lambda owner, lines, columns: owner.endswith("d.py::main")
        entries = [P + "t.py:3", P + "a.py:3 -> " + P + "t.py:3", P + "d.py:5 -> " + P + "t.py:3"]
        got = {"across": rerun.across_sends(entries),
               "callers_of": [[c.owner, c.lines, c.hop, c.how, c.between, c.across, c.sent]
                              for c in rerun.callers_of(entries, "JobSpec", "1", {"JobSpec.PandaID"})]}

        def rerun_(passes, chosen):
            r = object.__new__(rerun.Rerun)
            r.lines, r.hop, r.hand, r.hand_used, r.unanswered = [], "U2", {}, [], []
            r.opened, r.ends, r.branches = [], [], []
            r.mode, r.decide = "map", "positive"
            r.data, r.args = {"statuslog": {"states": []}}, N(observed="pending", key="1", asks="asks.jsonl", subject="JobSpec.jobStatus")
            r.logs = N(needs_window=True, names=lambda files: list(files))
            r.files_of = lambda owner, today, lines: today
            r._key_columns = lambda subject, key: {"JobSpec.PandaID"}
            run = N(token="killJobs for Pending", first="2026-10-29 09:31:36,000", last="2026-10-29 10:33:15,000",
                    describe=lambda: "the run", machine="aipanda093", found_by=[0])
            r._ask = lambda key, names, asking, owners=None: ({"a.log": [run]}, [], {})
            rerun.R.passes = lambda fn, run, mode: (passes, f"{len(passes)} pass(es)")
            r.base_env = lambda at=None: N(values={})
            verdict = which_ran.Verdict(which_ran.Candidate(P + "d.py", "main", [chosen]), chosen, "positive")
            rerun.judge = lambda cands, runs, **k: which_ran.Judgement([verdict], verdict, "positive evidence", None)
            return r

        rerun.R._prints_of = lambda owner, lines: N(statements=[N(log_files=["a.log"])])
        functions = rerun.R.functions
        rerun.R.function = lambda rel, func, lines: N(node=functions(rel, func)[0], owner=f"{rel}::{func}", name=func, rel=rel)
        rerun.R.seen = lambda fn, run, mode: None
        rerun.stable_holes = lambda *a, **k: None
        rerun.class_of = lambda fn: None
        rerun.holds_key = lambda fn, p, runs, mode, called_at: True
        rerun.every_log_asked = lambda owner, lines, asked: {"held": True, "why": T.Said("logs-every-file-asked", {"files": asked})}

        def pas(first, last):
            return N(first=first, last=last, records=[0], log=N(name="a.log"))

        up = rerun.Callers(P + "p.py::killJob", [rerun.Arm(9, ["closed"], [])], "JobSpec.jobStatus", "1",
                           ("2026-10-29 09:31:36", "2026-10-29 10:33:15"), "before", entries, before="2026-10-29 10:33:15")
        main = rerun.Caller(P + "d.py::main", [4, 5], 2, "key", (), [], [4], [P + "c.py:3"])
        moved = pas("2026-10-29 10:33:04,808", "2026-10-29 10:33:04,808")

        def u2(name, caller, passes, chosen):
            r = rerun_(passes, chosen)
            try:
                q = r.caller(rerun.CallerPasses(caller, up))
                got[name] = ["blocked", q.exit_line] if isinstance(q, rerun.Blocked) else ["callers", list(q.window)]
            except rerun.Stop as e:
                got[name] = ["stop", str(e), e.code]
            got[name + ": set aside"] = [x for x in r.lines if "the one that moved it on, not read as a pass before it" in x]

        u2("moved across", main, [moved], 4)
        u2("another way in the span", main, [moved], 6)
        u2("across, before the span", main, [pas("2026-10-29 10:00:00,000", "2026-10-29 10:00:01,000")], 4)
        u2("no send", rerun.Caller(P + "d.py::main", [4, 5], 2, "key"), [moved], 4)
        print(json.dumps(got))
    """))
    got = json.loads(out.splitlines()[-1])
    p = "pandaserver/"
    assert got["across"] == {f"{p}d.py:4 -> {p}c.py:3 -> {p}a.py:3 -> {p}t.py:3": f"{p}c.py:3"}, got
    assert got["callers_of"] == [
        [f"{p}t.py::killJobs", [3], 1, "none", [], [], []],
        [f"{p}d.py::main", [4, 5], 2, "key", [f"{p}a.py:3", f"{p}c.py:3", f"{p}t.py:3"], [4], [f"{p}c.py:3"]],
    ], got
    assert got["moved across"][0] == "stop" and got["moved across"][2] == "K55", got
    assert got["moved across"][1].startswith(
        "U2: a pass of main on the row from 2026-10-29 09:31:36 to 2026-10-29 10:33:04,808: none found"), got
    assert got["moved across: set aside"] == [
        f"pass 2026-10-29 10:33:04,808 .. 2026-10-29 10:33:04,808 took main:4 (positive), a call across the send at {p}c.py:3, "
        "and ended within 10 min of the move at 2026-10-29 10:33:15: the one that moved it on, not read as a pass before it"]
    assert got["another way in the span"] == ["blocked", 6], got
    assert got["another way in the span: set aside"] == []
    for kept in ("across, before the span", "no send"):
        assert got[kept][0] == "stop" and got[kept][1].startswith("U2: killJob's pass on the row inside main's call at 4"), got
        assert got[kept + ": set aside"] == [], got


def test_the_record_door_asks_no_mark_for_a_line_its_logger_writes_with_no_mark(tmp_path):
    # UMK (out/gm/qk-design.md): copyArchive's module logger writes no mark, so the first word of "reassignJobs for JEDI
    # at Nq/Nr overshoot site ... ([...])" (copyArchive.py:633) is cut as the mark; runs_for reads such a line as a run
    # of its own (K70), so asking that "mark" whole brings nothing its run reads -- and unanswered, the ask kept the
    # whole log from the reader (MJE-7398283030).  A mark its logger writes (LogWrapper's "kill") is asked as before
    out = _ok(
        _run(
            tmp_path,
            textwrap.dedent(_RECORDED)
            + textwrap.dedent("""
            import re
            from types import SimpleNamespace as N
            from bamboo.querier import mapdump

            bare = re.escape("reassignJobs for JEDI at site ") + "[^ ]*" + re.escape(" (") + ".*" + re.escape(")")
            mapdump._FRAGMENT[:] = [N(function_prints=[N(owner="p/c.py::main", runs=[], statements=[
                N(log_files_basis="logger", runs=[], handed_runs=[], log_files=["panda-x.log"], rows=[N(pattern=bare, prefix="")]),
                N(log_files_basis="logger", runs=[], handed_runs=[], log_files=["panda-x.log"],
                  rows=[N(pattern=re.escape("Start job_ids: ") + ".*", prefix="kill ")]),
            ])])]
            W = ("2026-09-28 08:30:00", "2026-09-28 08:40:00")
            bare_line = {**line("08:35:00", "for JEDI at site S1 ([5, 777])", "reassignJobs"), "offset": 10}
            kill_line = {**line("08:35:01", "Start job_ids: [777]", "kill"), "offset": 20}
            got = {}
            for name, recs in (("bare", [bare_line]), ("kill", [kill_line])):
                log, asks = ask_of([record(Ask("panda-x.log", ("777",), W), {M1: answered({**TODAY, "records": recs})})],
                                   "panda-x.log", "777", W)
                got[name] = [asks, None if log is None else
                             [[r.token, r.alone, [x.stamp for x in r.records]] for r in runs_for(log, "777")]]
            print(json.dumps(got))
            """),
        )
    )
    got = json.loads(out.splitlines()[-1])
    # the line with no mark: no ask, the log answered from the key's record, the line a run of its own
    assert got["bare"] == [[], [["reassignJobs", True, ["2026-09-28 08:35:00,000"]]]], got
    # a mark its logger writes: asked whole on its machine, and the log unanswered until then, as before
    asks, log = got["kill"]
    assert log is None, got
    assert [(a["strings"], a["machine"], why) for a, why in asks] == [
        (["kill"], "aipanda093.cern.ch", "the records of the key's runs whose mark does not hold the key")], got


def test_a_callers_rows_after_its_call_show_it_after_the_send_and_rows_before_it_do_not(tmp_path):
    # HOP (a′) (out/gm/qk-design.md): copyArchive 881 calls Client.kill_jobs(jobs, 2) and prints 882 "killJobs for Anal
    # Active (...)" once it returns, after the endpoint's pass (MJE-7383377087).  Over a window after the send, a must
    # row after the call shows it ran; one before its call printed then is a later call's, and does not
    (_corpus(tmp_path) / "pandaserver" / "c.py").write_text(textwrap.dedent('''
        def main(jobs, other, n):
            tmp_log = LogWrapper(_logger, f"main <{n}>")
            if jobs:
                tmp_log.debug(f"killJobs for Pending ({jobs})")
                Client.kill_jobs(jobs, 4)
            if other:
                Client.kill_jobs(other, 2)
                tmp_log.debug(f"killJobs for Anal Active ({other})")
    ''').lstrip())
    # the call's pass, printing after 7's call; and a later pass, printing before it calls 5
    line = "2026-11-02 {} panda.log.c: DEBUG    main <{}> {}\n"
    (tmp_path / "after.log").write_text(line.format("15:14:55,289", 1, "killJobs for Anal Active ([777])"))
    (tmp_path / "later.log").write_text(line.format("15:14:56,000", 2, "killJobs for Pending ([777])"))
    out = _ok(_run(tmp_path, """
        import json
        from types import SimpleNamespace as N
        from bamboo.querier import rerun
        from bamboo.querier import rows as R
        from bamboo.querier.records import read_log, runs_for

        R.ROWS_FROM = "walk"
        r = object.__new__(rerun.Rerun)
        r.lines = []
        r.say = r.lines.append
        r.args = N(key="777")
        r.mode, r.decide = "outside", "positive"
        r.logs = N(names=lambda files: list(files))
        r.files_of = lambda owner, today, lines: today
        r._keep_unanswered = lambda unanswered: None
        rerun.owners_of = lambda ids: {}
        rerun.R._prints_of = lambda owner, lines: N(statements=[N(log_files=["panda-c.log"])])
        runs = {}
        r._ask = lambda key, names, asking, owners=None, whole=False: (runs, [], {})
        after = ("2026-11-02 15:14:54,493", "2026-11-02 15:24:55")
        send = {"send": "pandaserver/userinterface/Client.py:273"}
        why = "for-the-rows-after-a-call-across-a-send"
        got = {}
        runs["panda-c.log"] = runs_for(read_log("after.log"), "777")
        got["after the call"] = r._call_shown("pandaserver/c.py:7", after, None, why, send, after_call=True)
        runs["panda-c.log"] = runs_for(read_log("later.log"), "777")
        got["before the call"] = r._call_shown("pandaserver/c.py:5", after, None, why, send, after_call=True)
        got["as before"] = r._call_shown("pandaserver/c.py:5", after, None, "for-a-call-across-a-send", send)
        got["lines"] = r.lines
        print(json.dumps(got))
    """))
    got = json.loads(out.splitlines()[-1])
    # 8 prints only after 7's call returned: shown
    assert got["after the call"] is True, got
    # 4 prints before 5's call: printed after the send, it is a later call's
    assert got["before the call"] is False, got
    assert "    pandaserver/c.py:5: not read after its call: row(s) [4] print before the call, ending at 5" in got["lines"], got
    # the same verdict where the window is not after the call: positive, as H2 reads it
    assert got["as before"] is True, got


def test_an_agent_s_cycles_tell_the_selection_a_pass_belongs_to_and_its_lock_the_other_selections(tmp_path):
    # (C) out/gm/qk-design.md: the caller process() is handed parts of Worker.taskList (runImpl), given at __init__ by a
    # ListWithLock made in Agent.start of what getTasks returned.  Its cycles (MsgWrapper time tokens): the row became
    # selectable at 00:00:10; m1's user cycle took it (got 6 at 00:00:20) and its pass on it began at 00:15:00; m2's
    # selections went on every minute and took it after the 10-min lock, at 00:10:30, its pass beginning at 00:12:00
    corpus = _corpus(tmp_path)
    (corpus / "pandajedi" / "jediorder").mkdir(parents=True)
    (corpus / "pandajedi" / "jediorder" / "Agent.py").write_text(textwrap.dedent('''\
        from pandajedi.jedicore.MsgWrapper import MsgWrapper
        from pandajedi.jedicore.ThreadUtils import ListWithLock

        logger = None


        class Worker:
            def __init__(self, taskList):
                self.taskList = taskList

            def runImpl(self):
                while True:
                    task_list = self.taskList.get(10)
                    if not task_list:
                        return
                    self.process(task_list)

            def process(self, task_list):
                for t in task_list:
                    logger.info(f"done {t}")


        class Agent:
            def start(self):
                while True:
                    tmp_log = MsgWrapper(logger)
                    tmp_log.info("start")
                    for vo in self.vos:
                        for prod_source_label in self.labels:
                            tmp_log.info(f"preparing tasks to be finished for vo={vo} label={prod_source_label}")
                            tmp_log.info("getting tasks to be finished")
                            task_list = self.tb.getTasks(vo, prod_source_label)
                            if task_list is None:
                                tmp_log.error("failed to get tasks to be finished")
                                continue
                            tmp_log.info(f"got {len(task_list)} tasks")
                            locked = ListWithLock(task_list)
                            thr = Worker(locked)
                            thr.start()
        '''))
    (corpus / "pandaserver" / "taskbuffer" / "db_proxy_mods").mkdir(parents=True)
    (corpus / "pandaserver" / "taskbuffer" / "db_proxy_mods" / "mod.py").write_text(textwrap.dedent('''\
        import datetime


        class M:
            def getTasks(self, vo, label):
                sql = "SELECT id FROM T WHERE status=:s AND (lockedBy IS NULL OR lockedTime<:timeLimit) "
                lock = "UPDATE T SET lockedBy=:lockedBy,lockedTime=CURRENT_DATE WHERE id=:id "
                varMap = {}
                timeLimit = naive_utcnow() - datetime.timedelta(minutes=10)
                varMap[":timeLimit"] = timeLimit
                return self.run(sql, lock, varMap)
        '''))

    def cycle(mark, label, t, n, getting=None):
        g = getting or t
        return (f"2026-01-01 {t[:8]},000 panda.log.PostProcessor: INFO     <2026-01-01/{mark}> preparing tasks to be finished for vo=None label={label}\n"
                f"2026-01-01 {g[:8]},500 panda.log.PostProcessor: INFO     <2026-01-01/{mark}> getting tasks to be finished\n"
                f"2026-01-01 {t[9:]},000 panda.log.PostProcessor: INFO     <2026-01-01/{mark}> got {n} tasks\n")
    m1 = (cycle("00:00:00.000001", "user", "00:00:00 00:00:20", 6, "00:00:01")
          + cycle("00:00:05.000001", "managed", "00:00:05 00:00:07", 0)
          + cycle("00:21:00.000001", "user", "00:21:00 00:21:02", 0))
    m2 = "".join(cycle(f"00:{mm:02d}:30.000001", "user", f"00:{mm:02d}:30 00:{mm:02d}:31", 0) for mm in range(1, 10))
    m2 += cycle("00:10:29.000001", "user", "00:10:29 00:10:30", 29, "00:10:29")
    (tmp_path / "m1.log").write_text(m1)
    (tmp_path / "m2.log").write_text(m2)
    out = _ok(_run(tmp_path, """
        import datetime, json
        from types import SimpleNamespace as N
        from bamboo.querier import agents as AG
        from bamboo.querier import mapdump
        from bamboo.querier import rows as R
        from bamboo.querier.records import read_log

        R.ROWS_FROM = "walk"
        rel = "pandajedi/jediorder/Agent.py"
        db = "pandaserver/taskbuffer/db_proxy_mods/mod.py::getTasks"
        mapdump._FRAGMENT[:] = [N(
            entered_froms=[N(owner=f"{rel}::process", line=18, sites=[N(owner=f"{rel}::runImpl", line=16)], open=[]),
                           N(owner=db, line=5, sites=[], open=[])],
            field_froms=[N(owner=f"{rel}::Worker", field="taskList", bindings=[N(sources=[
                N(kind="parameter", name="taskList", site_owner=f"{rel}::__init__")])])],
            parameter_froms=[N(owner=f"{rel}::__init__", parameter="taskList", sources=[
                N(kind="object", name="ListWithLock", at=[f"{rel}:37"], route=[f"{rel}:38"]),
                N(kind="constant", name="None", at=["pandajedi/jedimsgprocessor/m.py:9"], route=[])])],
        )]
        sel, why = AG.selection_of(f"{rel}::process", 18)
        crows = AG.cycle_rows(sel)
        records = []
        for name, machine in (("m1.log", "m1"), ("m2.log", "m2")):
            for r in read_log(name).records:
                r.machine = machine
                records.append(r)
        picks, spans = AG.selections(records, crows, "outside")
        since = "2026-01-01 00:00:10"
        user = {m: [p for p in ps if p.label == "user"] for m, ps in picks.items()}
        b1, w1 = AG.batch_of(user["m1"], "2026-01-01 00:15:00,000", "user", since)
        b2, w2 = AG.batch_of(user["m2"], "2026-01-01 00:12:00,000", "user", since)
        early, why_early = AG.batch_of(user["m1"], "2026-01-01 00:15:00,000", "user", "2026-01-01 00:00:30")
        lock = AG.lock_of(sel)
        between = [q for qs in user.values() for q in qs if b1.at < q.at < b2.after]
        other = lambda minutes: AG.Lock(datetime.timedelta(minutes=minutes), lock.term, lock.bound, lock.sets)
        print(json.dumps({
            "sel": [sel.worker, sel.field, sel.agent, sel.made, sel.handed, sel.call, sel.callee, list(sel.others), why],
            "not_taken": list(AG.selection_of(f"{rel}::start", 24)),
            "rows": [[r.line for r in crows.before], [r.line for r in crows.after], crows.label],
            "literals": AG.literals(crows.after[-1].map_patterns[0]),
            "strings": AG.ask_strings(list(crows.rows)), "one": AG.ask_strings([crows.after[-1]]),
            "labels": sorted({(m, p.label, p.count) for m, ps in picks.items() for p in ps if p.count != "0"}),
            "b1": [b1.mark, b1.after[11:19], b1.at[11:19], b1.count], "b2": [b2.mark, b2.at[11:19], b2.count],
            "early": [early, why_early],
            "one_process": [AG.one_process(user["m1"], spans["m1"]), AG.one_process(user["m2"], spans["m2"])],
            "overlap": AG.one_process(picks["m1"], {**spans["m1"], "<2026-01-01/00:00:05.000001>": ("2026-01-01 00:00:05,000", "2026-01-01 00:21:01,000")}),
            "lock": [str(lock.span), lock.term, lock.bound, lock.sets],
            "kept": AG.kept_off(b1, b2, between, lock, since), "n_between": len(between),
            "short": AG.kept_off(b1, b2, between, other(5), since), "long": AG.kept_off(b1, b2, between, other(15), since),
        }))
    """))
    got = json.loads(out.splitlines()[-1])
    # C1: process() <- runImpl's self.taskList.get <- Worker.__init__'s taskList <- the ListWithLock made at 37 of what
    # getTasks returned at 32, handed at 38; the message processor's None is another way in, said
    assert got["sel"] == ["pandajedi/jediorder/Agent.py::Worker", "taskList", "pandajedi/jediorder/Agent.py::Agent.start",
                          37, 38, 32, "getTasks", ["pandajedi/jedimsgprocessor/m.py:9"], ""], got
    # a function that takes no part of a worker's list, and no call into it hands one: C1 says what it looked for
    assert got["not_taken"] == [None, "start takes no part of a worker's list (self.<field>.get()), nor is handed one by the "
                                      "map's calls into it"], got
    # C2: the rows in the innermost loop before and after the call, the label the loop's element; the "start" before
    # the loops is not one; the literal they all hold
    assert got["rows"] == [[30, 31], [34, 36], "prod_source_label"], got
    assert got["literals"] == ["got ", " tasks"], got
    assert got["strings"] == [" tasks"] and got["one"] == [" tasks"], got
    # C3: each selection carries its cycle's label, the managed one apart; the pass at 00:15 belongs to m1's 00:00:20
    # selection (its SELECT ran after the row became selectable, 00:00:10), the one at 00:12 to m2's 00:10:30; had the row
    # become selectable at 00:00:30, after m1's selection ended, the pass would belong to none
    assert got["labels"] == [["m1", "user", "6"], ["m2", "user", "29"]], got
    assert got["b1"] == ["<2026-01-01/00:00:00.000001>", "00:00:01", "00:00:20", "6"], got
    assert got["b2"] == ["<2026-01-01/00:10:29.000001>", "00:10:30", "29"], got
    assert got["early"][0] is None and "before the row could be selected" in got["early"][1], got
    # cycles of the label that do not overlap; a cycle spanning another's does
    assert got["one_process"] == [True, True] and got["overlap"] is False, got
    # C4: getTasks' lock term, its limit's binding and the take's lock
    assert got["lock"] == ["0:10:00", "lockedTime<:timeLimit", "timeLimit = naive_utcnow() - datetime.timedelta(minutes=10)",
                           "lockedTime=CURRENT_DATE"], got
    # m2's 9 selections between the takes began within the lock from 00:00:20 (to 00:10:20), and the take at 00:10:30
    # came after its earliest end (00:10:10): kept off; a 5-min lock leaves selections after it, a 15-min one ends
    # after the next take
    assert got["n_between"] == 9 and got["kept"] == [True, ""], got
    assert got["short"][0] is False and "after the lock had ended" in got["short"][1], got
    assert got["long"][0] is False and "before the lock" in got["long"][1], got
