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
from typing import Callable, Iterator, TypeVar, cast

_HELD = "_walked"
#: The nodes of each kinds asked of a walk (:func:`of_kinds`), held beside the order.
_KINDS = "_walked_kinds"
#: What :func:`derived` finds no value held as.
_NONE = object()
_T = TypeVar("_T")

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


def of_kinds(node: ast.AST, kinds: tuple[type[ast.AST], ...]) -> tuple[ast.AST, ...]:
    """The nodes :func:`walk` finds from *node* that are of *kinds*, in its order, held on *node* for *kinds* as the
    order is.  A reader looking for a few kinds -- the assignments a bind is read from, the functions holding a
    line -- asks ``isinstance`` of each node it walks past otherwise: ``sql.scan_binds`` asked whether the run rules
    out each of 15.9 million nodes before seeing that only its statements could fill a bind."""
    if _open:
        return tuple(one for one in ast.walk(node) if isinstance(one, kinds))
    by_kinds: dict[tuple[type[ast.AST], ...], tuple[ast.AST, ...]] | None = getattr(node, _KINDS, None)
    if by_kinds is not None and kinds in by_kinds:
        return by_kinds[kinds]
    opened = _opened
    found = tuple(one for one in walk(node) if isinstance(one, kinds))
    if _opened == opened and not _open:
        if by_kinds is None:
            by_kinds = {}
            setattr(node, _KINDS, by_kinds)
        by_kinds[kinds] = found
    return found


def derived(node: ast.AST, name: str, make: Callable[[], _T]) -> _T:
    """What *make* reads off the tree under *node*, held on *node* as the attribute *name*, as :func:`walk` holds its
    order: read again inside a :func:`changing` window, and not held when one opened while it was read.  For a value
    a reader derives from a tree alone and asks of again: a scope's ``global`` and ``nonlocal`` names."""
    if _open:
        return make()
    held = getattr(node, name, _NONE)
    if held is not _NONE:
        return cast(_T, held)
    opened = _opened
    found = make()
    if _opened == opened and not _open:
        setattr(node, name, found)
    return found


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
