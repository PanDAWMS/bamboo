"""``bamboo bind-maps`` — say what the boundaries of two or more Code Maps reach in one another.

Each map is built from its own code and states its boundaries without the far
side (``build-map --dump``).  This reads the dumps together
(:func:`bamboo.codemap.bind.bind`) and reports which function of one map
reaches which endpoint, handler, table or column of another, what is left
unbound and why, and which message types a receiver takes that no map given
sends.  Any number of maps, of any systems: binding reads the fields every
map has.

Usage::

    bamboo build-map --dump panda.json
    bamboo build-map --map-id swf-epicprod --source-root <checkout> --dump epicprod.json
    bamboo build-map --map-id swf-monitor  --source-root <checkout> --dump monitor.json
    bamboo bind-maps panda.json epicprod.json monitor.json --dump bindings.json
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Optional

import click

from bamboo.codemap.bind import Binding, Bindings, bind
from bamboo.codemap.store import load_dump


def _report(found: Bindings, top: int) -> None:
    click.echo("maps:")
    for map_id, version in sorted(found.maps.items()):
        click.echo(f"  {map_id}: {version}")

    channels: dict[tuple[str, str, str], list[Binding]] = {}
    for b in found.bound:
        channels.setdefault((b.source_map, b.target_map, b.kind), []).append(b)
    click.echo(f"\nbound: {len(found.bound)} over {len(channels)} channel(s)")
    for (source, target, kind), crossings in sorted(channels.items()):
        whole = [b for b in crossings if not b.part]
        parts = [b for b in crossings if b.part]
        click.echo(
            f"  {source} -> {target} [{kind}]: {len(whole)} bound"
            + (f", {len(parts)} column(s)" if parts else "")
        )
        for b in sorted(whole, key=lambda b: (b.key, b.target_node))[:top]:
            columns = sorted(
                {
                    p.part
                    for p in parts
                    if p.key == b.key and p.source_node == b.source_node
                }
            )
            click.echo(
                f"      {b.key}: {len(b.source_functions)} function(s) -> {b.target_node}"
                + (f"  [{', '.join(columns)}]" if columns else "")
            )
            for note in b.notes:
                click.echo(f"          note: {note}")
        if len(whole) > top:
            click.echo(f"      … {len(whole) - top} more")
        for note, count in sorted(
            Counter(n for p in parts for n in p.notes if "here" not in n).items()
        ):
            click.echo(f"      note ({count}): {note}")

    click.echo(f"\nunbound: {len(found.unbound)}")
    for (map_id, reason), count in sorted(
        Counter((u.map_id, u.reason) for u in found.unbound).items()
    ):
        click.echo(f"  {map_id}: {count} × {reason}")
        rows = [u for u in found.unbound if (u.map_id, u.reason) == (map_id, reason)]
        for u in rows[:top]:
            detail = u.key or u.part
            click.echo(
                f"      {u.node}"
                + (f"  [{detail}]" if detail and detail not in u.node else "")
            )
        if len(rows) > top:
            click.echo(f"      … {len(rows) - top} more")

    click.echo(f"\ntaken and sent by no map given: {len(found.unreached)}")
    for node, count in sorted(Counter(u.node for u in found.unreached).items()):
        click.echo(f"  {node}: {count} type(s)")


@click.command("bind-maps")
@click.argument(
    "dumps",
    nargs=-1,
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--dump",
    "dump_to",
    default=None,
    metavar="PATH",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Write the bindings there as JSON.  Without it the command only reports.",
)
@click.option("--top", default=10, show_default=True, help="Rows per listing.")
def main(dumps: tuple[Path, ...], dump_to: Optional[Path], top: int) -> None:
    """Bind the boundaries of the maps in DUMPS (``build-map --dump``) to one another."""
    if len(dumps) < 2:
        raise click.ClickException("binding needs at least two maps")
    fragments = [load_dump(path) for path in dumps]
    ids = [fragment.map_id for fragment in fragments]
    if len(set(ids)) != len(ids):
        raise click.ClickException(
            f"the dumps hold the same map more than once: {sorted(ids)}"
        )
    found = bind(fragments)
    _report(found, top)
    if dump_to is not None:
        dump_to.write_text(found.model_dump_json(indent=1))
        click.echo(f"\nwrote {dump_to}")


if __name__ == "__main__":
    main()
