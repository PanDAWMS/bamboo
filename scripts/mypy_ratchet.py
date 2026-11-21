"""Keep mypy's findings from growing: its error count per file against a baseline kept in the repository.

The whole package once held about 1300 mypy errors that nothing checked.  This holds each file to the count in
``scripts/mypy_baseline.json``:

* a file with more errors than its baseline count fails, and its errors are printed (a file not in the baseline
  has a count of 0, so one new error there fails);
* a file with fewer errors also fails, until the baseline is lowered: a commit that fixes errors runs
  ``update`` and commits the lower counts with the fix, so the gain cannot be spent later.

The counts depend on mypy's version and on the packages installed (with ``ignore_missing_imports`` an absent
package is ``Any``), so the baseline is taken as CI runs it: the ``dev`` extra only, and the mypy version named in
the baseline, which CI installs.  Locally, point mypy at such an environment with ``--python-executable``.

usage:
    python scripts/mypy_ratchet.py check  [-- MYPY_ARGS...]
    python scripts/mypy_ratchet.py update [--allow-rise] [-- MYPY_ARGS...]
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASELINE = ROOT / "scripts" / "mypy_baseline.json"
TARGET = "bamboo/"
ERROR = re.compile(r"^(?P<file>[^:\s]+\.py):\d+: error: ")


def run_mypy(extra: list[str]) -> tuple[str, list[str]]:
    """mypy's version and its error lines over the package, as the pyproject configures it."""
    version = subprocess.run(
        [sys.executable, "-m", "mypy", "--version"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()[1]
    with tempfile.TemporaryDirectory() as cache:
        out = subprocess.run(
            [
                sys.executable,
                "-m",
                "mypy",
                TARGET,
                "--no-color-output",
                "--no-error-summary",
                "--cache-dir",
                cache,
                *extra,
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
    if out.returncode not in (0, 1):  # 2: mypy itself failed, not the code
        sys.exit(f"mypy failed (rc {out.returncode}):\n{out.stderr or out.stdout}")
    return version, [ln for ln in out.stdout.splitlines() if ERROR.match(ln)]


def counts_of(errors: list[str]) -> dict[str, int]:
    return dict(
        sorted(Counter(m["file"] for ln in errors if (m := ERROR.match(ln))).items())
    )


def load() -> dict:
    data: dict = json.loads(BASELINE.read_text())
    return data


def check(extra: list[str]) -> int:
    base = load()
    version, errors = run_mypy(extra)
    if version != base["mypy"]:
        print(
            f"note: mypy {version} here, the baseline was taken with {base['mypy']}: counts may differ"
        )
    now, was = counts_of(errors), base["counts"]
    rose = {f: (was.get(f, 0), n) for f, n in now.items() if n > was.get(f, 0)}
    fell = {f: (n, now.get(f, 0)) for f, n in was.items() if now.get(f, 0) < n}
    for f, (a, b) in rose.items():
        print(f"{f}: {a} -> {b} errors")
        for ln in errors:
            if ln.startswith(f + ":"):
                print("    " + ln)
    for f, (a, b) in fell.items():
        print(
            f"{f}: {a} -> {b} errors: lower the baseline (python scripts/mypy_ratchet.py update)"
        )
    total = sum(now.values())
    print(
        f"mypy {version}: {total} errors in {len(now)} files; the baseline holds {sum(was.values())}"
    )
    return 1 if rose or fell else 0


def update(extra: list[str], allow_rise: bool) -> int:
    was = load()["counts"] if BASELINE.exists() else {}
    version, errors = run_mypy(extra)
    now = counts_of(errors)
    rose = {f: n for f, n in now.items() if n > was.get(f, 0)}
    if was and rose and not allow_rise:
        for f, n in rose.items():
            print(f"{f}: {was.get(f, 0)} -> {n} errors")
        print("not written: a count rose (fix it, or pass --allow-rise)")
        return 1
    BASELINE.write_text(
        json.dumps({"mypy": version, "target": TARGET, "counts": now}, indent=1) + "\n"
    )
    print(
        f"{BASELINE.relative_to(ROOT)}: mypy {version}, {sum(now.values())} errors in {len(now)} files"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    extra = args[args.index("--") + 1 :] if "--" in args else []
    args = args[: args.index("--")] if "--" in args else args
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("action", choices=["check", "update"])
    ap.add_argument(
        "--allow-rise",
        action="store_true",
        help="update: write a baseline with a higher count",
    )
    a = ap.parse_args(args)
    return check(extra) if a.action == "check" else update(extra, a.allow_rise)


if __name__ == "__main__":
    sys.exit(main())
