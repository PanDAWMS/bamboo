"""Memos for readings that can ask for what they are still reading: Tarjan's lowlink over the keys open.

A reading that asks again for a key it is still reading -- ``a`` runs ``b``, which runs ``a`` -- is handed what
reading that key once more would add to its own reading: nothing new.  That is right for the key that opened the
cycle, its root, whose reading goes on to add all the cycle reaches.  It is not right for what is read under the
root: read there, a value lacks the root's own part, and kept, it would hand that short value to whoever asks for it
later from elsewhere -- an answer that depends on which was read first.  So a value read while a key it reached is
still open is not kept: it is handed back, its lowest open key is told to the one asking, and it is read again when
asked after that key is done -- as Tarjan's lowlink keeps a component open until its root closes.  A root keeps its
value.

One :class:`Lowlink` serves every memo whose readings recurse through one another (the changes reading's
``own_changes``, ``handed_changes`` and ``by_field``): a cut met under any of them reaches the keys of all.
"""

from __future__ import annotations

from typing import Callable, Hashable, MutableMapping, Optional, TypeVar

K = TypeVar("K", bound=Hashable)
V = TypeVar("V")


class Lowlink:
    """The keys still being read by one reading, over the memos it keeps, and the lowest of them each reading under
    them has reached."""

    def __init__(self) -> None:
        self._open: dict[tuple[int, Hashable], int] = {}
        self._low: list[int] = []

    def read(
        self,
        memo: MutableMapping[K, V],
        key: K,
        compute: Callable[[], V],
        again: Optional[V] = None,
        settle: Optional[Callable[[V], V]] = None,
    ) -> V:
        """*memo*'s value for *key*, else ``compute()``, kept in *memo* only where nothing it read was cut at a key
        still open under which it was read.

        *again*: what a key asked for while it is being read hands back -- nothing new to its own reading.  ``None``
        reads it again inside instead: a reading that only ever reaches itself through the keys of other memos, which
        do the cutting.  *settle*: applied to a value as it is kept (``reach``'s binding that reads only itself)."""
        if key in memo:
            return memo[key]
        mark = (id(memo), key)
        index = len(self._low)
        if again is not None:
            if mark in self._open:
                self._low[-1] = min(self._low[-1], self._open[mark])
                return again
            self._open[mark] = index
        self._low.append(index)
        try:
            found = compute()
        finally:
            low = self._low.pop()
            if again is not None:
                del self._open[mark]
        if low < index:
            # Read under a key still open: kept by none, told to the one asking.
            self._low[-1] = min(self._low[-1], low)
            return found
        if settle is not None:
            found = settle(found)
        memo[key] = found
        return found
