"""Throwaway diagnostic: draw symptoms at random and read what the map says.

Not production code.  It exists because the plan named it for six rounds and
it was never there -- every draw until now was done by hand, which is the same
failure that retired the sealed holdout: a procedure a document names and no
artefact performs.  What it does is small and the point is that it is written
down.

**The draw is recorded before the answers are read.**  That ordering is the whole
test: a sample looked at first and recorded afterwards can always be called
unrepresentative once its answers are in.  So ``--out`` is written before anything
about the drawn symptoms is asked.  (It used to ask ``derive-strategy`` itself,
with ``--report``; that command is gone, and the drawn keys are read by hand.)

**The population is the map's vocabulary** -- the set of questions the map
claims it can answer -- rather than the values production was seen to hold.
Measured when that was decided: the two symptoms that had found the most were
both absent from the production side, because one is a transient the sample
never caught and the other belongs to a table the evidence only holds failures
for.  Drawing from what the map advertises is what makes this a test of the
advertisement.

``--population`` draws from a plain list of keys instead, one per line, for
categories that are not vocabulary entries -- the junctions with no resolvable
caller, the tables holding no spec.  Those are read by hand; the tool only
draws and records them.
"""

from __future__ import annotations

import asyncio
import json
import random
from pathlib import Path
from typing import Optional

import click

from bamboo.codemap.lookup import CodeMap


async def _vocabulary(map_id: str, dump: Path) -> list[dict]:
    from bamboo.codemap.store import dump_graph

    db = await dump_graph(dump)
    try:
        terms = await CodeMap(db, map_id=map_id).vocabulary()
    finally:
        await db.close()
    # Sorted by key so the draw depends on the map's content and not on the
    # order the store happened to return.
    return [
        {
            "kind": term.kind,
            "key": term.key,
            "subject": term.symptom.subject,
            "observed": term.symptom.observed or "",
        }
        for term in sorted(terms, key=lambda t: t.key)
    ]


@click.command("review-symptoms")
@click.option("--seed", type=int, required=True, help="The draw is this and the population.")
@click.option("-k", "--count", default=3, show_default=True, help="How many to draw.")
@click.option("--map-id", default="panda", show_default=True)
@click.option(
    "--kind",
    default=None,
    help="Restrict the population to one kind (value, cut, step, chain).",
)
@click.option(
    "--population",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Draw from this list of keys, one per line, instead of the vocabulary.",
)
@click.option(
    "--out",
    type=click.Path(path_type=Path),
    default=None,
    help="Write the draw here.  Required by --report, and written before it.",
)
@click.option(
    "--dump",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="The map whose vocabulary is drawn from, as build-map --dump wrote it.",
)
def main(
    seed: int,
    count: int,
    map_id: str,
    kind: Optional[str],
    population: Optional[Path],
    out: Optional[Path],
    dump: Optional[Path],
) -> None:
    """Draw *count* symptoms with *seed* and record them."""

    if population is not None:
        rows = [
            {"kind": "-", "key": line.strip(), "subject": "", "observed": ""}
            for line in population.read_text().splitlines()
            if line.strip() and not line.startswith("#")
        ]
        where = str(population)
    else:
        if dump is None:
            raise click.UsageError("drawing from the vocabulary reads the map: give --dump, or --population")
        rows = asyncio.run(_vocabulary(map_id, dump))
        if kind:
            rows = [row for row in rows if row["kind"] == kind]
        where = f"{map_id} vocabulary" + (f" ({kind})" if kind else "")

    if count > len(rows):
        raise click.UsageError(f"asked for {count} of a population of {len(rows)}")
    drawn = random.Random(seed).sample(rows, count)

    click.echo(f"population {len(rows)} from {where}   seed {seed}   n={count}")
    for i, row in enumerate(drawn, 1):
        click.echo(f"  {i}  {row['kind']:<6} {row['key']}")

    if out is not None:
        out.write_text(
            json.dumps(
                {"seed": seed, "population": len(rows), "where": where, "drawn": drawn},
                indent=1,
            )
        )
        click.echo(f"\ndraw recorded in {out}")


if __name__ == "__main__":
    main()
