"""The nodes CPython's parser hands one object of to every tree -- ``pathcond.SHARED_NODES`` -- keep no parent and no
line the map writes: ``pathcond.attach_parents``, and the copies ``sql.at_key`` makes of a loop's put."""

from __future__ import annotations

import ast

from bamboo.codemap.panda import pathcond, sql


def test_a_parent_is_not_written_on_a_node_every_tree_shares() -> None:
    one = ast.parse("def f(job):\n    return job.status + 1\n")
    other = ast.parse("x = 1\n\n\ndef g(task):\n    y = task.status - 2\n")
    shared = {
        id(node): node
        for node in ast.walk(one)
        if isinstance(node, pathcond.SHARED_NODES)
    }
    assert any(id(node) in shared for node in ast.walk(other)), (
        "the parser shares Load between the two trees"
    )
    pathcond.attach_parents(one)
    pathcond.attach_parents(other)
    assert all(getattr(node, "parent", None) is None for node in shared.values())
    statements = {id(node) for node in ast.walk(one) if isinstance(node, ast.stmt)}
    for node in ast.walk(one):
        if hasattr(node, "lineno"):
            assert id(pathcond.site_of(node)) in statements


def test_a_loop_puts_copy_gives_its_line_to_its_own_nodes_only() -> None:
    """``param[key]`` put for the key ``transExitCode``: the copy is on the put's line, the Load it shares with every
    tree is not."""
    tree = ast.parse("for key in keys:\n    varMap[':' + key] = param[key]\n")
    pathcond.attach_parents(tree)
    value = tree.body[0].body[0].value  # type: ignore[attr-defined]
    copied = sql.at_key(value, "key", "transExitCode")
    assert ast.unparse(copied) == "param['transExitCode']"
    assert getattr(value.ctx, "lineno", None) is None
    assert getattr(value.ctx, "parent", None) is None
    assert (copied.lineno, copied.value.lineno) == (2, 2)  # type: ignore[attr-defined]
