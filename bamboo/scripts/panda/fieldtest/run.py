"""Run the querier on cases held as data, and record how far each run got with no one choosing.

    python -m bamboo.scripts.panda.fieldtest.run --home <dir> run <case.json>... [--jobs N] [--pair NAME] [--batch DIR]
    python -m bamboo.scripts.panda.fieldtest.run --home <dir> summary <batch dir>

A case is a JSON file (:func:`load_case`): the question (kind, subject, observed
value, key), its inputs (a rows file and log files), the querier's modes, the
pin whose code wrote the logs, and, for a replayed case, the output it must
reproduce.  Nothing here knows a case: every step reads its arguments from the
case file.

**Which code runs.**  A case names its pin, not a map: the map moves on that
pin as the map session lands, and a case should not be edited for it.
``pairs/current.json`` names the pair each pin runs on now; ``--pair`` names
another, to compare with the map before.  A pair is ``PANDA_PIN`` (the corpus),
``BAMBOO_TREE`` (a run tree: the map's tree with ``bamboo.querier`` laid over
it, whose ``QUERIER`` file names the querier's sha) and ``MAP_DUMP`` with its
sha256.  :func:`load_pair` checks a pair the way the querier's
``run_baselines.sh`` does: the dump's sha256, a run tree named after the dump's
bamboo sha, and the dump's ``derived_from`` naming the pin.  The pair is written
into every command, never taken from the querier's defaults.

The querier runs as ``python -m bamboo.querier.rerun`` from the run tree.  It
installs ``MAP_DUMP`` itself and says in its ``# map:`` line which dump it
installed (since the querier's e2a9fff); each run checks that line against the
pair.  A run is never given a Neo4j it can reach (:data:`NO_GRAPH`), so a
querier that reads the map store instead of the dump -- one older than e2a9fff,
or an entry that does not install -- fails rather than answering from whatever
map the store holds at the time.

**What is scored.**  No ``--hand``: the steps before the first ``STOP`` are the
ones that ran with no judgement.  :func:`score` reads the hops the run opened,
its ``STOP`` or ``END``, and how many times it used a hand; a case with a
baseline also gets the number of lines (``#`` lines aside) that differ from it.
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

HOP = re.compile(r"^== (?P<hop>[A-Z]\d+) ")
STOP = re.compile(
    r"^STOP (?P<hop>[A-Z]\d+): (?P<need>.*?); HAND used (?P<hand>\d+) time"
)
END = re.compile(r"^END (?P<what>.*)")
REACHED = re.compile(r"^== reached the end; HAND used (?P<hand>\d+) time")
DERIVED = re.compile(r'"derived_from":\s*"source-root:([^"]+)"')
MAP = re.compile(r"^# map: (?P<map>.*)$", re.M)

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
    return case


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
        if not re.fullmatch(rf"tree-{m[1]}(-q[0-9a-f]+)?", tree.name):
            raise FieldTestError(
                f"{dump} is read with tree-{m[1]}[-q<querier>], not {tree}"
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


def command(home: Home, case: dict, out: Path) -> list[str]:
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
    ]
    return args


def score(text: str) -> dict:
    """How far a run got: the hops it opened, where it ended, and how many times it used a hand."""
    lines = text.splitlines()
    hops = [m["hop"] for line in lines if (m := HOP.match(line))]
    got: dict = {
        "hops": hops,
        "last_hop": hops[-1] if hops else None,
        "end": None,
        "stop": None,
        "hand": None,
    }
    for line in lines:
        if m := STOP.match(line):
            got.update(stop={"hop": m["hop"], "need": m["need"]}, hand=int(m["hand"]))
        elif m := END.match(line):
            got["end"] = m["what"]
        elif m := REACHED.match(line):
            got["hand"] = int(m["hand"])
    got["terminal"] = (
        "STOP" if got["stop"] else "END" if got["end"] is not None else "NONE"
    )
    return got


def lines_differ(baseline: str, text: str) -> int:
    """Lines (``#`` lines aside) that one output has and the other lacks."""
    a, b = (
        [x for x in t.splitlines() if not x.startswith("#")] for t in (baseline, text)
    )
    return sum(1 for x in difflib.ndiff(a, b) if x[:2] in ("- ", "+ "))


def map_said(text: str, pair: dict) -> dict:
    """The run's ``# map:`` line, and whether it names the pair's dump and its sha256."""
    m = MAP.search(text)
    said = m["map"] if m else None
    want = f"dump {pair['MAP_DUMP']} (sha256 {pair['MAP_DUMP_SHA256'][:16]})"
    return {"said": said, "ok": said == want}


def run_case(home: Home, path: Path, batch: Path, override: str = "") -> dict:
    case = load_case(path)
    name = pair_name(home, case, override)
    pair = load_pair(home, name)
    if Path(pair["PANDA_PIN"]).name != f"pin-{case['pin']}":
        raise FieldTestError(
            f"pair {name} runs {pair['PANDA_PIN']}, case {case['name']} was written by pin-{case['pin']}"
        )
    where = batch / case["name"]
    where.mkdir(parents=True, exist_ok=True)
    out = where / "out.txt"
    args = command(home, case, out)
    env = {k: pair[k] for k in ("PANDA_PIN", "BAMBOO_TREE", "MAP_DUMP")}
    env |= {"PYTHONPATH": pair["BAMBOO_TREE"], "PYTHONDONTWRITEBYTECODE": "1"}
    env |= NO_GRAPH
    t0 = time.time()
    with (where / "stdout.log").open("w") as log:
        rc = subprocess.run(
            args,
            cwd=home.cwd,
            env=os.environ | env,
            stdout=log,
            stderr=subprocess.STDOUT,
        ).returncode
    text = out.read_text() if out.exists() else ""
    record = {
        "case": str(path),
        "case_sha256": sha256(path),
        "name": case["name"],
        "exit": rc,
        "seconds": round(time.time() - t0, 2),
        "pair": name,
        "querier_sha": pair["QUERIER_SHA"],
        "env": env,
        "cwd": str(home.cwd),
        "argv": args,
        "inputs_sha256": {
            p: sha256(Path(p))
            for p in [case["inputs"]["rows"], *case["inputs"]["logs"]]
        },
        "score": score(text),
        "map": map_said(text, pair),
    }
    if base := case.get("baseline"):
        record["baseline"] = {
            "path": base,
            "sha256": sha256(Path(base)),
            "lines_differ": lines_differ(Path(base).read_text(), text),
        }
    (where / "run.json").write_text(json.dumps(record, indent=1) + "\n")
    return record


def line(r: dict) -> str:
    """One line per run, for the terminal."""
    s = r["score"]
    if s["terminal"] == "END":
        where = f"END {s['end'][:70]}"
    elif s["stop"]:
        where = f"STOP {s['stop']['hop']}: {s['stop']['need'][:70]}"
    else:
        where = "no terminal line"
    base = f"  baseline diff {r['baseline']['lines_differ']}" if "baseline" in r else ""
    if not r.get("map", {"ok": True})["ok"]:
        base += f"  MAP NOT THE PAIR'S: {r['map']['said']}"
    return (
        f"{r['name']:<8} {r['pair']} q{r.get('querier_sha', '?')[:7]}  exit {r['exit']}  {r['seconds']:>6}s  "
        f"hops {len(s['hops']):>2} (last {s['last_hop']})  hand {s['hand']}  {where}{base}"
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
        help="a pair under pairs/ instead of the one pairs/current.json names",
    )
    s = sub.add_parser("summary", help="one line per run of a batch")
    s.add_argument("batch")
    a = ap.parse_args(argv)
    if a.cmd == "summary":
        for p in sorted(Path(a.batch).glob("*/run.json")):
            print(line(json.loads(p.read_text())))
        return 0
    try:
        home = Home.load(a.home)
        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        batch = Path(a.batch) if a.batch else home.root / "runs" / stamp
        batch.mkdir(parents=True, exist_ok=True)
        print(f"# batch {batch}", flush=True)
        with ThreadPoolExecutor(a.jobs) as pool:
            for rec in pool.map(
                lambda p: run_case(home, Path(p), batch, a.pair),
                a.cases,
            ):
                print(line(rec), flush=True)
    except FieldTestError as e:
        print(f"ABORT: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
