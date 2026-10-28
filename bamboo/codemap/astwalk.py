"""``ast.walk``, held for each tree it is asked of.

The map walks the same trees again and again: a build of the PanDA pin asks ``ast.walk`` 3.04 million times of
194,198 trees, and 99.6% of the nodes those walks yield are of a tree walked before -- about a third of the build's
time.  A tree does not change while the map reads it, so the order one walk finds is the order every later walk would
find, and :func:`walk` hands it back.

The order is held on the node walked from, as the readers hold what they derive from a tree (``_reaching_once``,
``_rebinds``): it lives as long as the tree does.  Keyed by ``id`` instead, a tree parsed for a moment and dropped
would leave its order to the next tree at the same address.  Only a walk that runs to its end is held, so a walk its
caller leaves early costs what it did.

The one reader that changes a tree in place -- the annotation audit takes an annotation out and puts it back -- does
it inside :func:`changing`: a walk there is fresh and not held, and a walk running across it is not held.
"""

from __future__ import annotations

import ast
from collections import deque
from contextlib import contextmanager
from typing import Iterator

_HELD = "_walked"

#: How many :func:`changing` windows are open, and how many have been opened: a walk is held only where none opened
#: while it ran.
_open = 0
_opened = 0


def walk(node: ast.AST) -> Iterator[ast.AST]:
    """*node* and every node below it, in :func:`ast.walk`'s order."""
    if _open:
        return ast.walk(node)
    held = getattr(node, _HELD, None)
    if held is not None:
        return iter(held)
    return _walking(node)


def _walking(node: ast.AST) -> Iterator[ast.AST]:
    opened = _opened
    seen: list[ast.AST] = []
    todo = deque([node])
    while todo:
        one = todo.popleft()
        todo.extend(ast.iter_child_nodes(one))
        seen.append(one)
        yield one
    if _opened == opened and not _open:
        setattr(node, _HELD, tuple(seen))


@contextmanager
def changing() -> Iterator[None]:
    """A window in which a tree is changed in place and then put back as it was.

    What was held before the window is right again after it, since the tree is the same again.
    """
    global _open, _opened
    _open += 1
    _opened += 1
    try:
        yield
    finally:
        _open -= 1
