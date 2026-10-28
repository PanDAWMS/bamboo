"""bamboo.codemap.astwalk: ast.walk held for each tree it is asked of."""

import ast

from bamboo.codemap.astwalk import changing, walk

SOURCE = """
def f(a: int, b) -> dict:
    x = [a + 1 for _ in range(b)]
    if x:
        return {"k": x}
    return {}
"""


def _held(node: ast.AST):
    return getattr(node, "_walked", None)


def test_a_walk_to_its_end_is_held_and_handed_back_in_ast_walks_order():
    tree = ast.parse(SOURCE)
    first = list(walk(tree))
    assert first == list(ast.walk(tree))
    assert _held(tree) == tuple(first)
    assert list(walk(tree)) == first
    # A subtree is a tree of its own.
    func = tree.body[0]
    assert _held(func) is None
    assert list(walk(func)) == list(ast.walk(func))


def test_a_walk_left_early_is_not_held():
    tree = ast.parse(SOURCE)
    assert next(walk(tree)) is tree
    assert (
        next((n for n in walk(tree) if isinstance(n, ast.FunctionDef)), None)
        is tree.body[0]
    )
    assert _held(tree) is None


def test_a_tree_changed_in_a_window_is_walked_afresh_and_its_order_is_right_again_after():
    tree = ast.parse(SOURCE)
    func = tree.body[0]
    before = list(walk(func))
    saved = func.returns
    with changing():
        func.returns = None
        inside = list(walk(func))
        assert saved not in inside
        assert inside == list(ast.walk(func))
        fresh = ast.parse("y = 1")
        list(walk(fresh))
        assert _held(fresh) is None
        func.returns = saved
    assert list(walk(func)) == before
    assert saved in before


def test_a_walk_running_across_a_window_is_not_held():
    tree = ast.parse(SOURCE)
    walking = walk(tree)
    next(walking)
    with changing():
        pass
    rest = list(walking)
    assert [tree, *rest] == list(ast.walk(tree))
    assert _held(tree) is None
    list(walk(tree))
    assert _held(tree) is not None
