"""Size each class of input the querier asks PanDA for, over the whole corpus, from one map dump.

    python -m bamboo.fieldtest.census <map dump.json> [--json out.json] [--tables N] [--exclude T ...]

The classes are where a value lives, as the map tells them apart: a database
row (H, a status history; R, a row read by its columns), a view PanDA builds in
memory from the database (the SiteMapper), another system (Rucio, harvester,
the pilot), and a log line.  An API part answers a class, not a case; the
cases only check the classes, and the sizes come from here.

``--tables N`` ranks the tables by how many of the code's read sites read them
and reports how many read sites a whitelist of the first N reads in full -- the
measure the read API's table whitelist starts from.  It counts statements, so
a table the code reads through one shared accessor (``CONFIG`` behind
``getConfigValue``, four statements and many callers) ranks lower than its use.

Read only.  Nothing here imports the querier or the map's code: the dump is
plain JSON, and the one piece of PanDA knowledge it needs -- which table keeps
the history of which subject -- is written below with where it comes from.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

#: the history PanDA keeps, per subject: base_module.py recordStatusChange writes jobs_StatusLog (jobStatus),
#: record_task_status_change writes TASKS_STATUSLOG (status) -- the two bamboo.querier.parts names too
HISTORIES = {
    "JobSpec.jobStatus": "jobs_StatusLog",
    "JediTaskSpec.status": "TASKS_STATUSLOG",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def histories(d: dict) -> dict:
    """H: the subjects the map has writers for, and whether PanDA keeps a history of each."""
    rows = []
    for s in sorted(
        (s for s in d["subjects"] if s["updated_by"]),
        key=lambda s: (s["spec_class"] or "", s["attribute"]),
    ):
        name = f"{s['spec_class']}.{s['attribute']}"
        rows.append(
            {
                "subject": name,
                "values_written": len(s["updated_by"]),
                "values_selected": len(s["selected_by"]),
                "history": HISTORIES.get(name),
            }
        )
    return {
        "written_subjects": len(rows),
        "with_history": sum(1 for r in rows if r["history"]),
        "rows": rows,
    }


def _canonical(d: dict) -> dict[str, str]:
    """Each spelling of an entity's table, lower case -> the entity's first spelling."""
    return {t.lower(): e["tables"][0] for e in d["entities"] for t in e["tables"]}


def table_reads(d: dict) -> Counter:
    """How many read sites read each table (an entity's spellings counted as one table)."""
    canon = _canonical(d)
    per: Counter = Counter()
    for r in d["read_sites"]:
        for t in {canon.get(t.lower(), t) for t in r["tables"]}:
            per[t] += 1
    return per


def covered(d: dict, allowed: list[str]) -> int:
    """Read sites every table of which is in *allowed* (a site reading two tables needs both)."""
    canon = _canonical(d)
    allow = {canon.get(t.lower(), t).lower() for t in allowed}
    return sum(
        1
        for r in d["read_sites"]
        if all(canon.get(t.lower(), t).lower() in allow for t in r["tables"])
    )


def ranked(d: dict, exclude: list[str]) -> list[str]:
    """The tables by reads, less *exclude*; among tables read equally often, the one that
    lets the most read sites be read in full with the tables before it comes first.

    Without the second key the order among equals is the dump's order, and a whitelist cut
    inside such a group would hold whichever table the dump happened to list first.
    """
    ex = {t.lower() for t in exclude}
    by_reads: dict[int, list[str]] = defaultdict(list)
    for t, k in table_reads(d).items():
        if t.lower() not in ex:
            by_reads[k].append(t)
    out: list[str] = []
    for k in sorted(by_reads, reverse=True):
        group = sorted(by_reads[k])
        while group:
            best = max(group, key=lambda t: covered(d, [*out, t]))
            out.append(best)
            group.remove(best)
    return out


def whitelist(d: dict, n: int, exclude: list[str]) -> dict:
    """The first *n* tables of :func:`ranked`, and how many read sites they read in full."""
    ranked_ = ranked(d, exclude)
    return {
        "n": n,
        "excluded": exclude,
        "tables": ranked_[:n],
        "covered": covered(d, ranked_[:n]),
        "read_sites": len(d["read_sites"]),
    }


def reads(d: dict) -> dict:
    """R: the statements the code runs, by the tables they read and the columns they bind."""
    rs = d["read_sites"]
    ents = {t.lower(): e for e in d["entities"] for t in e["tables"]}
    by_class: Counter = Counter()
    not_entities: Counter = Counter()
    for r in rs:
        for t in r["tables"]:
            e = ents.get(t.lower())
            if e is None:
                not_entities[t] += 1
            by_class[
                (e["spec_class"] or f"(no class) {e['name']}")
                if e
                else "(not an entity)"
            ] += 1
    keys = Counter(tuple(r["keys"]) for r in rs)
    return {
        "read_sites": len(rs),
        "entities": len(d["entities"]),
        "entities_with_class": sum(1 for e in d["entities"] if e["spec_class"]),
        "classes": sorted({e["spec_class"] for e in d["entities"] if e["spec_class"]}),
        "tables": len(table_reads(d)),
        "tables_per_read": dict(sorted(Counter(len(r["tables"]) for r in rs).items())),
        "no_bound_column": keys.get((), 0),
        "bound_column_sets": len(keys),
        "top_bound_columns": [[list(k), n] for k, n in keys.most_common(15)],
        "reads_by_class": by_class.most_common(),
        "tables_not_entities": not_entities.most_common(),
        "reading_stops": Counter(x["reason"] for x in d["reading_stops"]).most_common(),
    }


def logs(d: dict) -> dict:
    """L: the log files the code's printed lines land in, and which package prints to each."""
    files: dict[str, Counter] = defaultdict(Counter)
    basis: Counter = Counter()
    for fp in d["function_prints"]:
        package = fp["owner"].split("/", 1)[0]
        for st in fp["statements"]:
            basis[st["log_files_basis"]] += 1
            for f in st["log_files"]:
                files[f][package] += 1
    return {
        "log_files": len(files),
        "printed_statements_by_basis": dict(basis),
        "files_printed_from_both_packages": sorted(
            f for f, c in files.items() if len(c) > 1
        ),
        "statements_per_file": sorted(
            ((f, dict(c)) for f, c in files.items()), key=lambda x: -sum(x[1].values())
        ),
    }


def outside(d: dict) -> dict:
    """The boundaries the map names, by the system on the other side."""
    return {
        "boundaries": len(d["boundaries"]),
        "by_system": Counter(b["system"] for b in d["boundaries"]).most_common(),
        "by_kind": Counter(b["kind"] for b in d["boundaries"]).most_common(),
    }


def census(d: dict) -> dict:
    return {
        "derived_from": d.get("derived_from"),
        "H": histories(d),
        "R": reads(d),
        "L": logs(d),
        "outside": outside(d),
    }


def report(out: dict, w=None) -> None:
    w = w or sys.stdout.write
    h = out["H"]
    w(
        f"H  subjects the map has writers for: {h['written_subjects']}; PanDA keeps a history of {h['with_history']}\n"
    )
    for r in h["rows"]:
        w(
            f"   {r['subject']:<34} values written {r['values_written']:>2}  selected {r['values_selected']:>2}  "
            f"history {r['history'] or '-'}\n"
        )
    r = out["R"]
    w(
        f"\nR  read sites {r['read_sites']} over {r['tables']} tables ({r['entities']} entities, "
        f"{r['entities_with_class']} with a spec class)\n"
    )
    w(
        f"   tables per read {r['tables_per_read']}; reads binding no column {r['no_bound_column']}; "
        f"distinct bound-column sets {r['bound_column_sets']}\n"
    )
    w(f"   classes: {', '.join(r['classes'])}\n")
    w("   reads by class (a read of n tables counts n times):\n")
    for c, n in r["reads_by_class"]:
        w(f"     {n:>4}  {c}\n")
    w(f"   tables read that are not entities: {r['tables_not_entities'] or 'none'}\n")
    w("   top bound-column sets:\n")
    for k, n in r["top_bound_columns"]:
        w(f"     {n:>4}  {', '.join(k) or '(none)'}\n")
    w("   reading stops (statements the map reads only in part):\n")
    for k, n in r["reading_stops"]:
        w(f"     {n:>4}  {k}\n")
    lg = out["L"]
    w(
        f"\nL  log files {lg['log_files']}; printed statements by how the map placed their file "
        f"{lg['printed_statements_by_basis']}\n"
    )
    w(
        f"   files printed to from both pandaserver and pandajedi code: {lg['files_printed_from_both_packages']}\n"
    )
    for f, c in lg["statements_per_file"][:25]:
        w(f"     {sum(c.values()):>5}  {f}  {c}\n")
    o = out["outside"]
    w(
        f"\nX  boundaries {o['boundaries']}: by system {o['by_system']}; by kind {o['by_kind']}\n"
    )
    if wl := out.get("whitelist"):
        w(
            f"\nW  the first {wl['n']} tables by reads (less {wl['excluded'] or 'none'}) read "
            f"{wl['covered']}/{wl['read_sites']} read sites in full: {', '.join(wl['tables'])}\n"
        )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m bamboo.fieldtest.census", description=__doc__.split("\n\n")[0]
    )
    ap.add_argument("dump")
    ap.add_argument("--json", default="", help="also write the census as JSON here")
    ap.add_argument(
        "--tables",
        type=int,
        default=0,
        help="report a whitelist of the first N tables by reads",
    )
    ap.add_argument(
        "--exclude", nargs="*", default=[], help="tables the whitelist leaves out"
    )
    a = ap.parse_args(argv)
    path = Path(a.dump)
    d = json.loads(path.read_text())
    out = {"dump": str(path), "dump_sha256": sha256(path)} | census(d)
    if a.tables:
        out["whitelist"] = whitelist(d, a.tables, a.exclude)
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1) + "\n")
    sys.stdout.write(
        f"# dump {path} sha256 {out['dump_sha256'][:16]} ({out['derived_from']})\n\n"
    )
    report(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
