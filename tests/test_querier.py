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
    "callers", "census", "census_p", "evaluate", "false_guard", "handlers", "mapdump", "outbound", "parts",
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
                     "it hands its rows on", "the map could not read all of it"]


_MOVERS_MAP = """
    import json
    from types import SimpleNamespace as N
    from bamboo.querier.rerun import arms_of, movers_of

    def branch(line, outcome, pre=()):
        return N(line=line, outcome=outcome, path_condition=[], row_precondition=list(pre))

    def writer(owner, *branches):
        return N(owner=owner, branches=list(branches), log_files=[])

    def stmt(owner, line, selects, region, crosses_to=()):
        return N(owner=owner, line=line, selects=selects, region=region, crosses_to=list(crosses_to))

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
        stmt("d/metrics.py::count", 109, ["JobSpec.jobStatus=running", "JobSpec.VO=atlas"], [100, 120]),
    ])
    print(json.dumps(movers_of(follow, writers, "JobSpec.jobStatus", "running")))
    """,
        )
    )
    # a counting statement keeps its rows (a mover with no arm of its own); the guarded writer comes last
    assert json.loads(out.splitlines()[-1]) == {
        "d/metrics.py::count": "count@109 selects the value",
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
