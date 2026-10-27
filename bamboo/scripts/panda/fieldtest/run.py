"""Run the querier on cases held as data, and record how far each run got with no one choosing.

    python -m bamboo.scripts.panda.fieldtest.run --home <dir> run <case.json>... [--jobs N] [--pair NAME] [--batch DIR]
        [--record [--rounds N]]
    python -m bamboo.scripts.panda.fieldtest.run --home <dir> summary <batch dir>

A case is a JSON file (:func:`load_case`): the question (kind, subject, observed
value, key), its inputs (a rows file, or the answers to build one from, and log
files or a records directory), the querier's modes, the pin whose code wrote the
logs, and, for a replayed case, the output it must reproduce.  Nothing here
knows a case: every step reads its arguments from the case file.

**Rounds.**  A run is one round of the querier in ``<case>/r1/``.  With
``--record``, a round that stops on asks (``code`` ``asks``) is followed by the
recording step, which answers them from production into the case's records
directory, and the querier runs again in ``r<n+1>/``, until it asks nothing,
the recording step exits other than 0 (3: a day file still being compressed;
4: every ask deferred, a grep or a list having failed), or ``--rounds`` are
done.  An ask the recording step defers is asked again by the next round.  The
recording step runs from the tree this module is in, where PanDA's credentials
load (the home's ``cwd``).  Each case runs into ``<batch>/<name>/``, so the
cases of a batch have distinct names.

**Which code runs.**  A case names its pin, not a map: the map moves on that
pin as the map session lands, and a case should not be edited for it.
``pairs/current.json`` names the pair each pin runs on now; ``--pair`` names
another, to compare with the map before.  A case whose pin is ``unsettled``
(the code that wrote its logs is not known) runs only on a pair ``--pair``
names, and its run says so.  A pair is ``PANDA_PIN`` (the corpus),
``BAMBOO_TREE`` (a run tree: the map's tree with ``bamboo.querier`` laid over
it, whose ``QUERIER`` file names the querier's sha) and ``MAP_DUMP`` with its
sha256.  :func:`load_pair` checks a pair the way the querier's
``run_baselines.sh`` does: the dump's sha256, a run tree named after the dump's
bamboo sha, and the dump's ``derived_from`` naming the pin.  The pair is written
into every command, never taken from the querier's defaults.

The querier runs as ``python -m bamboo.querier.rerun`` from the run tree.  It
installs ``MAP_DUMP`` itself, and writes how the run ended and what it read to
``result.json`` (version 1, since the querier's b02b589): the dump it installed
with its whole sha256, the run tree and the ``QUERIER`` label in it, the rows
and the logs with theirs.  Each run checks these against what it handed
(:func:`read_as_handed`).  A run is never given a Neo4j it can reach
(:data:`NO_GRAPH`), so a querier that reads the map store instead of the dump
-- one older than e2a9fff, or an entry that does not install -- fails rather
than answering from whatever map the store holds at the time.

**What is scored.**  No ``--hand``: the steps before the first ``STOP`` are the
ones that ran with no judgement.  :func:`score` reads from ``result.json`` the
steps the run opened, how it ended (``STOP`` or ``END``, at which step, and the
``code`` that names the class of what a stop needs) and the hands it used.  The
querier writes ``result.json`` last and :func:`run_case` removes an earlier one
first, so a run without one did not finish.  A case with a baseline also gets
the number of lines (``#`` lines aside) its ``out.txt`` differs from it.
"""

from __future__ import annotations

import argparse
import datetime
import difflib
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

QUERIER_MODULE = "bamboo.querier.rerun"
KINDS = ("default", "late", "stuck")

#: how a case's pin starts when the code that wrote its logs is not known: it runs only on a pair named for the run
UNSETTLED = "unsettled"
#: the tree this module is in, which the recording step runs from (bamboo/scripts/panda/fieldtest/run.py)
HERE = Path(__file__).resolve().parents[4]
#: rounds of querier and recording step a run with --record takes at most, by default
ROUNDS = 3
#: the files of a case's records directory the querier reads, in this order
RECORDS_GLOB = "records-r*.jsonl"

#: the code that builds a dump and reads it: a dump goes with a run tree whose commit holds this code as the
#: dump's own commit does
MAP_CODE = ("bamboo/codemap", "bamboo/scripts/build_map.py")

DERIVED = re.compile(r'"derived_from":\s*"source-root:([^"]+)"')

#: the shape of result.json read here (agreed with the querier, 2026-10-15)
RESULT_VERSION = 1
#: the files a run writes beside its out.txt
RESULT, ASKS = "result.json", "asks.jsonl"

#: where a run's Neo4j is: a closed port (bamboo's settings take the environment over .env)
NO_GRAPH = {"NEO4J_URI": "bolt://127.0.0.1:1"}


class FieldTestError(Exception):
    """A case, a pair or the home is not what a run needs; nothing was run for it."""


@dataclass(frozen=True)
class Home:
    """The machine-local directory a field test reads its cases from and writes its runs to."""

    root: Path
    python: str
    cwd: Path

    @classmethod
    def load(cls, root: str | os.PathLike | None) -> Home:
        named = root or os.environ.get("FIELDTEST_HOME")
        if not named:
            raise FieldTestError("no home: pass --home or set FIELDTEST_HOME")
        home = Path(named).expanduser().resolve()
        config_file = home / "fieldtest.json"
        if not config_file.exists():
            raise FieldTestError(f"{config_file} does not exist")
        config = json.loads(config_file.read_text())
        return cls(
            root=home,
            python=config.get("python") or sys.executable,
            cwd=Path(config.get("cwd") or home).expanduser(),
        )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_case(path: Path) -> dict:
    case = json.loads(path.read_text())
    lacking = [
        k for k in ("name", "question", "inputs", "modes", "pin") if k not in case
    ]
    if lacking:
        raise FieldTestError(f"{path} lacks {lacking}")
    if case["question"].get("kind") not in KINDS:
        raise FieldTestError(f"{path}: question.kind must be one of {KINDS}")
    if not ({"rows", "answers"} & set(case["inputs"])):
        raise FieldTestError(f"{path}: inputs need rows, or answers to build them from")
    if case["inputs"].get("records") and case["inputs"]["logs"]:
        raise FieldTestError(
            f"{path}: inputs hand logs or a records directory, not both (the querier reads one door)"
        )
    return case


def distinct_names(paths: list[Path]) -> list[str]:
    """The cases' names, each its run's directory in the batch: two cases of one name would write into one."""
    names: dict[str, list[str]] = {}
    for p in paths:
        names.setdefault(load_case(p)["name"], []).append(str(p))
    shared = {n: ps for n, ps in names.items() if len(ps) > 1}
    if shared:
        raise FieldTestError(
            "cases of one name would run into one directory: "
            + "; ".join(f"{n}: {', '.join(ps)}" for n, ps in sorted(shared.items()))
        )
    return list(names)


def pair_name(home: Home, case: dict, override: str = "") -> str:
    """The pair a case runs on: *override*, else the one ``pairs/current.json`` names for the case's pin."""
    if override:
        return override
    current = json.loads((home.root / "pairs" / "current.json").read_text())
    if case["pin"] not in current:
        raise FieldTestError(
            f"pairs/current.json names no pair for pin {case['pin']} (case {case['name']})"
        )
    return current[case["pin"]]


def same_map_code(built: str, archived: str) -> bool:
    """Whether commits *built* and *archived* hold the same :data:`MAP_CODE`, by the git of the tree this module is in.

    A dump is named for the commit that built it and a run tree for the commit it was archived from, so the two
    differ when master moves by files the map does not read: the querier's tree of 9c4019c (PanDAWMS's docs and
    fieldtest over 1a8e694) reads the dump 1a8e694 built (2026-10-26).  A commit git does not know is not the same.
    """
    return (
        subprocess.run(
            [
                "git",
                "-C",
                str(HERE),
                "diff",
                "--quiet",
                built,
                archived,
                "--",
                *MAP_CODE,
            ],
            capture_output=True,
        ).returncode
        == 0
    )


def load_pair(home: Home, name: str) -> dict:
    """The pair *name*, checked: the dump's sha256, the run tree it is read with, and the pin it was built from."""
    pair = json.loads((home.root / "pairs" / f"{name}.json").read_text())
    dump = Path(pair["MAP_DUMP"])
    got = sha256(dump)
    if got != pair["MAP_DUMP_SHA256"]:
        raise FieldTestError(
            f"{dump} has sha256 {got[:16]}, the pair says {pair['MAP_DUMP_SHA256'][:16]}"
        )
    tree = Path(pair["BAMBOO_TREE"])
    if m := re.search(r"/map-.*-([0-9a-f]{7,40})\.json$", str(dump)):
        t = re.fullmatch(r"tree-([0-9a-f]{7,40})(-q[0-9a-f]+)?", tree.name)
        if not t or (t[1] != m[1] and not same_map_code(m[1], t[1])):
            raise FieldTestError(
                f"{dump} is read with tree-{m[1]}[-q<querier>] or the tree of a commit with the same "
                f"{', '.join(MAP_CODE)}, not {tree}"
            )
    marker = tree / "QUERIER"
    if not marker.exists():
        raise FieldTestError(
            f"{tree} has no QUERIER file: not a run tree with bamboo.querier laid over the map"
        )
    querier = marker.read_text().strip()
    if (m := re.search(r"-q([0-9a-f]+)$", tree.name)) and not querier.startswith(m[1]):
        raise FieldTestError(
            f"{tree} is named for querier {m[1]}, its QUERIER file says {querier}"
        )
    with dump.open() as f:
        derived = DERIVED.search(f.read(4096))
    if not derived or Path(derived[1]) != Path(pair["PANDA_PIN"]):
        raise FieldTestError(
            f"{dump} is derived from {derived and derived[1]}, not PANDA_PIN={pair['PANDA_PIN']}"
        )
    return pair | {"QUERIER_SHA": querier}


def build_rows(answers: Path, pair: dict, out: Path) -> Path:
    """Lay the fetch record *answers* out as a rows file at *out*, spelled by the pair's pin and dump."""
    # here, not at the top: rows imports fetch, which imports this module
    from bamboo.scripts.panda.fieldtest import rows  # noqa: PLC0415

    spellings = rows.column_spellings(Path(pair["PANDA_PIN"]), Path(pair["MAP_DUMP"]))
    out.write_text(
        json.dumps(rows.build(json.loads(answers.read_text()), spellings), indent=1)
        + "\n"
    )
    return out


def command(home: Home, case: dict, out: Path) -> list[str]:
    """The querier's command for *case*: the output *out*, and the result and the asks beside it, named here rather
    than left to the querier's defaults."""
    q, i, m = case["question"], case["inputs"], case["modes"]
    args = [home.python, "-m", QUERIER_MODULE]
    if q["kind"] != "default":
        args.append(f"--{q['kind']}")
    args += [
        "--subject",
        q["subject"],
        "--observed",
        q["observed"],
        "--key",
        str(q["key"]),
    ]
    if i.get("records"):
        args += ["--records", i["records"]]
    for log in i["logs"]:
        args += ["--log", log]
    args += [
        "--rows",
        i["rows"],
        "--patterns",
        m["patterns"],
        "--decide",
        m["decide"],
        "--out",
        str(out),
        "--result",
        str(out.with_name(RESULT)),
        "--asks",
        str(out.with_name(ASKS)),
    ]
    return args


def score(result: dict | None) -> dict:
    """How far a run got, from its result.json: the steps it opened, how it ended, and the hands it used.

    ``terminal`` is ``STOP`` or ``END``; ``code`` names the class of what a stop
    needs (``asks``, a KNOWN_ISSUES number, or None where none is named yet);
    ``text`` is the ``STOP`` line or the ``END`` lines.  A run with no result,
    or one of another version, is ``NONE`` and ``why`` says which.
    """
    got: dict = {
        "terminal": "NONE",
        "why": None,
        "step": None,
        "code": None,
        "text": None,
        "steps": [],
        "hand": None,
        "hand_used": None,
        "asks": None,
    }
    if result is None:
        return got | {"why": "no result.json: the run did not finish"}
    if result.get("version") != RESULT_VERSION:
        return got | {
            "why": f"result.json version {result.get('version')}, this reads {RESULT_VERSION}"
        }
    o = result["outcome"]
    return got | {
        "terminal": o["kind"].upper(),
        "step": o["step"],
        "code": o["code"],
        "text": o["text"],
        "steps": result["steps"],
        "hand": len(o["hand_used"]),
        "hand_used": o["hand_used"],
        "asks": result["asks"],
    }


def lines_differ(baseline: str, text: str) -> int:
    """Lines (``#`` lines aside) that one output has and the other lacks."""
    a, b = (
        [x for x in t.splitlines() if not x.startswith("#")] for t in (baseline, text)
    )
    return sum(1 for x in difflib.ndiff(a, b) if x[:2] in ("- ", "+ "))


def _real(path: str) -> str:
    return str(Path(path).resolve())


def read_as_handed(result: dict | None, pair: dict, handed: dict) -> dict:
    """Whether the run read what it was handed, by its result.json: the pair's dump (path and whole sha256), run tree
    and ``QUERIER`` label, and *handed*'s rows, logs and record files (path and sha256 each).  ``differ`` names each
    that does not match."""
    if result is None or result.get("version") != RESULT_VERSION:
        return {"ok": False, "differ": [RESULT], "bamboo_py_sha256": None}
    i, q = result["inputs"], result["inputs"]["querier"]
    dump = i["map_dump"]
    want = {
        "map_dump": {
            "path": _real(pair["MAP_DUMP"]),
            "sha256": pair["MAP_DUMP_SHA256"],
        },
        "tree": _real(pair["BAMBOO_TREE"]),
        "label": pair["QUERIER_SHA"],
        "rows": handed["rows"],
        "logs": handed["logs"] or None,
        "records": handed["records"],
    }
    records = i["records"]
    got = {
        "map_dump": dump and {"path": _real(dump["path"]), "sha256": dump["sha256"]},
        "tree": _real(q["tree"]),
        "label": q.get("label"),
        "rows": i["rows"],
        "logs": i["logs"],
        "records": records
        and [
            {"path": _real(f["path"]), "sha256": f["sha256"]} for f in records["files"]
        ],
    }
    differ = [k for k in want if want[k] != got[k]]
    return {
        "ok": not differ,
        "differ": differ,
        "bamboo_py_sha256": q.get("bamboo_py_sha256"),
    }


def handed_of(case: dict) -> dict:
    """What a round hands the querier, with each file's sha256: the rows, the logs, and the record files its records
    directory holds now (None for a case without one)."""
    i = case["inputs"]
    records = i.get("records")
    return {
        "rows": {"path": i["rows"], "sha256": sha256(Path(i["rows"]))},
        "logs": [{"path": p, "sha256": sha256(Path(p))} for p in i["logs"]],
        "records": records
        and [
            {"path": _real(str(f)), "sha256": sha256(f)}
            for f in sorted(Path(records).glob(RECORDS_GLOB))
        ],
    }


def run_round(home: Home, case: dict, pair: dict, env: dict, where: Path) -> dict:
    """One run of the querier into *where*, scored from its result.json."""
    where.mkdir(parents=True, exist_ok=True)
    out = where / "out.txt"
    # a run that does not finish is not scored from an earlier run's files in the same place
    for f in (out, out.with_name(RESULT), out.with_name(ASKS)):
        f.unlink(missing_ok=True)
    args = command(home, case, out)
    handed = handed_of(case)
    t0 = time.time()
    with (where / "stdout.log").open("w") as log:
        rc = subprocess.run(
            args,
            cwd=home.cwd,
            env=os.environ | env,
            stdout=log,
            stderr=subprocess.STDOUT,
        ).returncode
    result_file = out.with_name(RESULT)
    result = json.loads(result_file.read_text()) if result_file.exists() else None
    return {
        "exit": rc,
        "seconds": round(time.time() - t0, 2),
        "argv": args,
        "out": str(out),
        "result": (
            {"path": str(result_file), "sha256": sha256(result_file)}
            if result
            else None
        ),
        "score": score(result),
        "read": read_as_handed(result, pair, handed),
    }


def record_command(home: Home, asks: Path, records: Path) -> list[str]:
    """The recording step's command: run by the home's python from the tree this module is in, which it asserts."""
    code = (
        "import sys, bamboo; assert bamboo.__file__.startswith(sys.argv[1]), bamboo.__file__; "
        "from bamboo.scripts.panda.fieldtest import record; sys.exit(record.main(sys.argv[2:]))"
    )
    return [
        home.python,
        "-c",
        code,
        str(HERE),
        "--asks",
        str(asks),
        "--records",
        str(records),
    ]


def record_round(home: Home, asks: Path, records: Path, where: Path) -> dict:
    """Answer a round's *asks* from production into *records*, where PanDA's credentials load (the home's cwd)."""
    before = set(records.glob(RECORDS_GLOB))
    args = record_command(home, asks, records)
    env = {"PYTHONPATH": str(HERE), "PYTHONDONTWRITEBYTECODE": "1"}
    t0 = time.time()
    with (where / "record.log").open("w") as log:
        rc = subprocess.run(
            args,
            cwd=home.cwd,
            env=os.environ | env,
            stdout=log,
            stderr=subprocess.STDOUT,
        ).returncode
    return {
        "exit": rc,
        "seconds": round(time.time() - t0, 2),
        "argv": args,
        "written": sorted(str(f) for f in set(records.glob(RECORDS_GLOB)) - before),
    }


def run_case(
    home: Home,
    path: Path,
    batch: Path,
    override: str = "",
    record: bool = False,
    rounds: int = ROUNDS,
) -> dict:
    """Run *path*'s case into *batch*: one round, or with *record* up to *rounds* of the querier, each but the last
    followed by the recording step answering its asks from production, until the querier asks nothing.

    A case whose pin is unsettled runs only on the pair *override* names."""
    case = load_case(path)
    settled = not case["pin"].startswith(UNSETTLED)
    if not settled and not override:
        raise FieldTestError(
            f"case {case['name']}: its pin is {case['pin']!r}; name the pair it runs on with --pair"
        )
    name = pair_name(home, case, override)
    pair = load_pair(home, name)
    if settled and Path(pair["PANDA_PIN"]).name != f"pin-{case['pin']}":
        raise FieldTestError(
            f"pair {name} runs {pair['PANDA_PIN']}, case {case['name']} was written by pin-{case['pin']}"
        )
    records = case["inputs"].get("records")
    if record and not records:
        raise FieldTestError(
            f"case {case['name']}: --record needs inputs.records, the directory its records go to"
        )
    where = batch / case["name"]
    where.mkdir(parents=True, exist_ok=True)
    built = None
    if "rows" not in case["inputs"]:
        built = build_rows(Path(case["inputs"]["answers"]), pair, where / "rows.json")
        case = case | {"inputs": case["inputs"] | {"rows": str(built)}}
    env = {k: pair[k] for k in ("PANDA_PIN", "BAMBOO_TREE", "MAP_DUMP")}
    env |= {"PYTHONPATH": pair["BAMBOO_TREE"], "PYTHONDONTWRITEBYTECODE": "1"}
    env |= NO_GRAPH
    done: list[dict] = []
    ended = "one round (no --record)"
    for n in range(1, (rounds if record else 1) + 1):
        one = run_round(home, case, pair, env, where / f"r{n}")
        done.append(one)
        if not record:
            break
        if one["score"]["code"] != "asks":
            ended = "the querier asks nothing more"
            break
        if n == rounds:
            ended = f"still asking after {rounds} round(s)"
            break
        one["record"] = record_round(
            home, Path(one["score"]["asks"]["path"]), Path(records), where / f"r{n}"
        )
        if one["record"]["exit"] != 0:
            ended = f"the recording step exited {one['record']['exit']}"
            break
    last = done[-1]
    run = {
        "case": str(path),
        "case_sha256": sha256(path),
        "name": case["name"],
        "pin": case["pin"],
        "pin_settled": settled,
        "pair": name,
        "querier_sha": pair["QUERIER_SHA"],
        "env": env,
        "cwd": str(home.cwd),
        "inputs_sha256": {
            p: sha256(Path(p))
            for p in [
                case["inputs"]["rows"],
                *case["inputs"]["logs"],
                *([case["inputs"]["answers"]] if built else []),
            ]
        },
        "rounds": done,
        "ended": ended,
        "exit": last["exit"],
        "seconds": round(
            sum(r["seconds"] + r.get("record", {}).get("seconds", 0) for r in done), 2
        ),
        "result": last["result"],
        "score": last["score"],
        "read": last["read"],
    }
    if base := case.get("baseline"):
        out = Path(last["out"])
        run["baseline"] = {
            "path": base,
            "sha256": sha256(Path(base)),
            "lines_differ": lines_differ(
                Path(base).read_text(), out.read_text() if out.exists() else ""
            ),
        }
    (where / "run.json").write_text(json.dumps(run, indent=1) + "\n")
    return run


def line(r: dict) -> str:
    """One line per run, for the terminal."""
    if "read" not in r:
        return f"{r['name']:<8} {r['pair']}  scored from out.txt, before result.json: see its run.json"
    s = r["score"]
    if s["terminal"] == "NONE":
        where = s["why"]
    else:
        said = s["text"] or f"{s['terminal']} {s['step']} (no END line)"
        where = said.splitlines()[-1][:80] + (f"  [{s['code']}]" if s["code"] else "")
    extra = f"  asks {s['asks']['count']}" if s["asks"] else ""
    if "baseline" in r:
        extra += f"  baseline diff {r['baseline']['lines_differ']}"
    if not r["read"]["ok"]:
        extra += f"  NOT READ AS HANDED: {', '.join(r['read']['differ'])}"
    if not r.get("pin_settled", True):
        extra += "  pin unsettled"
    if r.get("rounds") and r["ended"] != "one round (no --record)":
        extra += f"  rounds {len(r['rounds'])}: {r['ended']}"
    last = s["steps"][-1] if s["steps"] else None
    return (
        f"{r['name']:<8} {r['pair']} q{r.get('querier_sha', '?')[:7]}  exit {r['exit']}  {r['seconds']:>6}s  "
        f"steps {len(s['steps']):>2} (last {last})  hand {s['hand']}  {where}{extra}"
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m bamboo.scripts.panda.fieldtest.run",
        description=__doc__.split("\n\n")[0],
    )
    ap.add_argument(
        "--home",
        default="",
        help="the field test's directory (default: $FIELDTEST_HOME)",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run cases")
    r.add_argument("cases", nargs="+")
    r.add_argument("--jobs", type=int, default=1)
    r.add_argument(
        "--batch",
        default="",
        help="where the runs go (default: <home>/runs/<time>)",
    )
    r.add_argument(
        "--pair",
        default="",
        help="a pair under pairs/ instead of the one pairs/current.json names (needed for a case whose pin is unsettled)",
    )
    r.add_argument(
        "--record",
        action="store_true",
        help="answer the querier's asks from production and run it again, up to --rounds",
    )
    r.add_argument("--rounds", type=int, default=ROUNDS)
    s = sub.add_parser("summary", help="one line per run of a batch")
    s.add_argument("batch")
    a = ap.parse_args(argv)
    if a.cmd == "summary":
        for p in sorted(Path(a.batch).glob("*/run.json")):
            print(line(json.loads(p.read_text())))
        return 0
    try:
        home = Home.load(a.home)
        distinct_names([Path(p) for p in a.cases])
        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        batch = Path(a.batch) if a.batch else home.root / "runs" / stamp
        batch.mkdir(parents=True, exist_ok=True)
        print(f"# batch {batch}", flush=True)
        with ThreadPoolExecutor(a.jobs) as pool:
            for rec in pool.map(
                lambda p: run_case(home, Path(p), batch, a.pair, a.record, a.rounds),
                a.cases,
            ):
                print(line(rec), flush=True)
    except FieldTestError as e:
        print(f"ABORT: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
