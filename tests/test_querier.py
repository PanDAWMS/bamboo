"""The querier's package -- ``bamboo.querier``.

Its modules read their environment when imported: ``PANDA_PIN`` names the
corpus (the PanDA source a map was built from) and goes on ``sys.path``, and
``BAMBOO_TREE`` must agree with where ``bamboo`` came from.  So every check
here runs in a child interpreter, with a temporary corpus, and leaves this
process as it was.  The case runs against production logs and rows are kept
out of the repository; these tests need neither.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MODULES = [
    "callers", "census", "census_p", "evaluate", "false_guard", "handlers", "mapdump", "outbound", "parts",
    "printed", "raised", "records", "reproduce", "rerun", "rows", "value_diff", "which_ran",
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
