"""lowlink.Lowlink: a value read under a key still being read is kept only once that key is done."""

from __future__ import annotations

import pytest

from bamboo.codemap.lowlink import Lowlink

#: Each key: its own part, and the keys its reading asks for.  ``a`` and ``b`` ask for each other; ``c`` asks for
#: ``b`` from outside the cycle.
GRAPH = {
    "a": ("x", ["b"]),
    "b": ("y", ["a"]),
    "c": ("z", ["b"]),
}


class _Reader:
    """Reads a key of :data:`GRAPH` as its own part and all its asks read, through one memo."""

    def __init__(self) -> None:
        self.lowlink = Lowlink()
        self.memo: dict[str, frozenset[str]] = {}
        self.reads: list[str] = []

    def of(self, key: str) -> frozenset[str]:
        return self.lowlink.read(
            self.memo, key, lambda: self._of(key), again=frozenset()
        )

    def _of(self, key: str) -> frozenset[str]:
        self.reads.append(key)
        own, asks = GRAPH[key]
        found = {own}
        for one in asks:
            found |= self.of(one)
        return frozenset(found)


@pytest.mark.parametrize(
    "order", [["a", "b", "c"], ["b", "a", "c"], ["c", "b", "a"], ["c", "a", "b"]]
)
def test_every_key_of_a_cycle_reads_the_whole_cycle_whichever_is_asked_first(order):
    reader = _Reader()
    assert {key: reader.of(key) for key in order} == {
        "a": {"x", "y"},
        "b": {"x", "y"},
        "c": {"x", "y", "z"},
    }


def test_only_the_root_of_a_cycle_keeps_its_value_what_was_read_under_it_is_read_again_when_asked():
    reader = _Reader()
    reader.of("a")
    # ``b`` met ``a`` still open: its {y} lacked ``a``'s part and was not kept.
    assert reader.memo == {"a": {"x", "y"}}
    reader.of("b")
    assert reader.memo == {"a": {"x", "y"}, "b": {"x", "y"}}
    assert reader.reads == ["a", "b", "b"]


def test_a_key_asked_for_while_it_is_read_hands_back_again():
    lowlink = Lowlink()
    memo: dict[str, str] = {}
    inner: list[str] = []

    def compute() -> str:
        inner.append(lowlink.read(memo, "k", compute, again="again"))
        return "done"

    assert lowlink.read(memo, "k", compute, again="again") == "done"
    assert inner == ["again"]
    assert memo == {"k": "done"}


def test_a_memo_with_no_again_reads_its_key_again_inside_and_a_cut_in_another_memo_reaches_through_it():
    # As the changes reading's own_changes (no again) and handed_changes (cuts) on one lowlink:
    # own k -> handed m -> own k, read again inside -> handed m, cut.
    lowlink = Lowlink()
    owned: dict[str, tuple[str, ...]] = {}
    handed: dict[str, tuple[str, ...]] = {}

    def own() -> tuple[str, ...]:
        return lowlink.read(owned, "k", lambda: ("k",) + hand())

    def hand() -> tuple[str, ...]:
        return lowlink.read(handed, "m", lambda: ("m",) + own(), again=())

    assert own() == ("k", "m", "k")
    # The inner reading of k was cut at m: kept by none.  m, the root of the cut, and the outer k are kept.
    assert owned == {"k": ("k", "m", "k")}
    assert handed == {"m": ("m", "k")}


def test_settle_is_applied_to_a_value_as_it_is_kept():
    lowlink = Lowlink()
    memo: dict[str, str] = {}

    def settle(value: str) -> str:
        return "unknown" if value == "nothing" else value

    assert (
        lowlink.read(memo, "k", lambda: "nothing", again="nothing", settle=settle)
        == "unknown"
    )
    assert memo == {"k": "unknown"}


def test_a_reading_that_raises_leaves_its_key_closed():
    lowlink = Lowlink()
    memo: dict[str, int] = {}

    def fails() -> int:
        raise ValueError("no")

    with pytest.raises(ValueError):
        lowlink.read(memo, "k", fails, again=0)
    assert lowlink.read(memo, "k", lambda: 1, again=0) == 1
    assert memo == {"k": 1}
