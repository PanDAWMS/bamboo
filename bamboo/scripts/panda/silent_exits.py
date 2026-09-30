"""List the decisions PanDA takes without saying so, where the test reads configuration or another system.

An ``if`` one of whose branches prints nothing is a decision a log cannot
show: ``setupper_atlas_plugin.setup_destination`` makes a fake vuid instead
of registering a ``_sub`` dataset on 803-812, and ``worker_module`` skips a
job type on 552 with its log line commented out.  Where the test reads a
configuration value or another system's answer, the outcome depends on
something the code does not hold, and a reader has neither the value nor a
line -- which is what makes the place worth one line of logging.

Each ``if`` in the corpus is read: which of its branches prints nothing, and
whether that branch leaves (``continue``, ``break``, ``return``, ``raise``).
Its test is walked, narrowed at the ``if`` and with a small allowance beyond
the function, and the ``if`` is listed when the walk stops at a configuration
value, a value another system supplies or one the database supplies -- each
with what it was asked with.

    python -m bamboo.scripts.panda.silent_exits --source-root <tree> [--only config,external]
"""

from __future__ import annotations

import ast
import collections
from pathlib import Path
from typing import Optional

import click

from bamboo.codemap import trace
from bamboo.codemap.models import STOP_CONFIG, STOP_EXTERNAL, STOP_UPSTREAM
from bamboo.codemap.panda import pathcond
from bamboo.codemap.panda.plugin import PandaCodeMapPlugin
from bamboo.codemap.panda.provenance import classify
from bamboo.codemap.panda.recognizers.emit import logging_arguments

#: What the test's walk ends in, by the word ``--only`` takes.
_KINDS = {"config": STOP_CONFIG, "external": STOP_EXTERNAL, "database": STOP_UPSTREAM}
#: Little beyond the function: the list is read per ``if``, over the corpus.
_BUDGET = trace.Budget(crossed_files=6, crossed_functions=12, crossed_steps=60, crossed_depth=4)
_LEAVES = (ast.Continue, ast.Break, ast.Return, ast.Raise)


def _prints(block: list[ast.stmt], logged: set[int]) -> bool:
    return any(id(node) in logged for statement in block for node in ast.walk(statement))


def _leaves(block: list[ast.stmt]) -> bool:
    return bool(block) and isinstance(block[-1], _LEAVES)


def silent_ifs(func: ast.AST) -> list[tuple[ast.If, str, bool]]:
    """``(if, the silent branch, whether it leaves)`` for each ``if`` with a branch that prints nothing."""
    logged = {id(argument) for argument in logging_arguments(func)}
    found: list[tuple[ast.If, str, bool]] = []
    for node in ast.walk(func):
        if not isinstance(node, ast.If):
            continue
        printed_body = _prints(node.body, logged)
        printed_else = _prints(node.orelse, logged)
        if not printed_body and (printed_else or _leaves(node.body)):
            found.append((node, "body", _leaves(node.body)))
        elif node.orelse and not printed_else and (printed_body or _leaves(node.orelse)):
            found.append((node, "else", _leaves(node.orelse)))
    return found


def scan(root: Path, kinds: set[str]) -> list[dict]:
    roots = PandaCodeMapPlugin._resolve_roots(root)
    wanted = {_KINDS[kind] for kind in kinds}
    listed: list[dict] = []
    for package, base in sorted(roots.items()):
        for path in sorted(base.rglob("*.py")):
            file = f"{package}/{path.relative_to(base)}"
            if "/test" in file:
                continue
            try:
                tree = ast.parse(path.read_text())
            except SyntaxError:
                continue
            pathcond.attach_parents(tree)
            for func, _cls in pathcond.functions_with_owner(tree):
                for node, branch, leaves in silent_ifs(func):
                    walked = trace.walk(
                        roots,
                        file=file,
                        owner=f"{file}::{func.name}",
                        lines=[node.lineno],
                        classify=classify,
                        budget=_BUDGET,
                        reaching=True,
                    )
                    reads = collections.OrderedDict()
                    for step in walked.steps:
                        if step.terminal in wanted:
                            reads.setdefault(f"{step.terminal} -- {step.detail}", step.line)
                    if reads:
                        listed.append(
                            {
                                "at": f"{file}:{node.lineno}",
                                "function": func.name,
                                "branch": branch,
                                "leaves": leaves,
                                "test": ast.unparse(node.test)[:160],
                                "reads": list(reads),
                            }
                        )
    return listed


@click.command()
@click.option(
    "--source-root",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="The tree to read.",
)
@click.option(
    "--only",
    default="config,external",
    show_default=True,
    help="Which stops make an if worth listing: config, external, database.",
)
@click.option("--json", "as_json", type=click.Path(path_type=Path), default=None, help="Also write the list here.")
def main(source_root: Path, only: str, as_json: Optional[Path]) -> None:
    """List the silent decisions whose test reads configuration or another system."""
    kinds = {kind.strip() for kind in only.split(",") if kind.strip()}
    listed = scan(source_root, kinds)
    by_kind = collections.Counter(
        kind for entry in listed for kind in {read.split(" -- ")[0] for read in entry["reads"]}
    )
    click.echo(f"{len(listed)} silent decision(s) whose test reads {only}")
    for kind, count in by_kind.most_common():
        click.echo(f"   {count:>5}  {kind}")
    for entry in listed:
        leaves = ", and leaves" if entry["leaves"] else ""
        click.echo(f"\n{entry['at']}  {entry['function']}  (the {entry['branch']} prints nothing{leaves})")
        click.echo(f"    if {entry['test']}")
        for read in entry["reads"][:6]:
            click.echo(f"    reads  {read}")
    if as_json is not None:
        import json

        as_json.write_text(json.dumps(listed, indent=1))


if __name__ == "__main__":
    main()
