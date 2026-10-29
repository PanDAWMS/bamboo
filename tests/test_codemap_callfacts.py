"""What a call does where no body of the corpus says it, and the one order a call is read in: ``codemap.callfacts``."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from bamboo.codemap import callfacts
from bamboo.codemap import calls as calls_mod

PACKAGE = "pkg"


def _tree(tmp_path: Path, **files: str) -> dict[str, Path]:
    root = tmp_path / PACKAGE
    root.mkdir(parents=True, exist_ok=True)
    (root / "__init__.py").write_text("")
    for name, text in files.items():
        (root / f"{name}.py").write_text(text)
    return {PACKAGE: root}


def _call(text: str) -> ast.Call:
    call = ast.parse(text, mode="eval").body
    assert isinstance(call, ast.Call)
    return call


def _puts(name: str, text: str) -> list[tuple[str, str, str]]:
    method = callfacts.by_name(name)
    assert method is not None
    return [
        (ast.unparse(value), put.into, ast.unparse(key) if key is not None else keyword or "")
        for value, put, key, keyword in callfacts.put_arguments(method, _call(text))
    ]


def test_a_builtin_method_is_said_once_by_its_type_and_by_its_name_for_every_type_having_it():
    """``setdefault`` is a dict's: its key and the value under it.  ``update`` by its name is a dict's merge and a
    set's elements together; ``pop`` puts nothing and changes its receiver."""
    setdefault = callfacts.builtin_method("dict", "setdefault")
    assert setdefault is not None and [put.into for put in setdefault.puts] == [callfacts.KEY, callfacts.VALUE]
    assert {put.into for put in callfacts.by_name("update").puts} == {
        callfacts.MERGE,
        callfacts.VALUE,
        callfacts.ELEMENTS,
    }
    assert callfacts.by_name("pop") == callfacts.Method((), True)
    assert callfacts.by_name("format") is None
    assert {"append", "add", "insert", "extend", "update", "setdefault"} <= callfacts.PUTTING
    assert "pop" not in callfacts.PUTTING and {"pop", "remove", "clear", "sort"} <= callfacts.CHANGING


@pytest.mark.parametrize(
    ("name", "text", "expected"),
    [
        ("insert", "x.insert(i, v)", [("v", callfacts.ELEMENT, "")]),
        ("setdefault", "d.setdefault(k, v)", [("k", callfacts.KEY, ""), ("v", callfacts.VALUE, "k")]),
        ("update", "d.update(m)", [("m", callfacts.MERGE, "")]),
        ("update", "s.update(a, b)", [("b", callfacts.ELEMENTS, ""), ("a", callfacts.MERGE, "")]),
        ("update", "d.update(k=v)", [("v", callfacts.VALUE, "k")]),
    ],
    ids=["insert-puts-its-second", "setdefault-a-key-and-its-value", "update-merges", "update-of-two", "keyword"],
)
def test_what_a_call_puts_is_its_arguments_where_the_method_puts_them(name, text, expected):
    """An argument merged is not put as its elements too: what holds it whole holds each element's."""
    assert sorted(_puts(name, text)) == sorted(expected)


_MODULE = (
    "import json\n"
    "from pandacommon.pandalogger.LogWrapper import LogWrapper\n"
    "from pandacommon.pandautils.base import SpecBase\n"
    "from pandacommon.pandautils.thread_utils import WeightedLists\n\n\n"
    "class Taker:\n"
    "    def append(self, x):\n"
    "        pass\n\n\n"
    "class Holder:\n"
    "    pass\n\n\n"
    "class Spec(SpecBase):\n"
    "    pass\n\n\n"
    "def declares(lists: WeightedLists, log: LogWrapper) -> None:\n"
    "    pass\n\n\n"
    "def fills(d, v):\n"
    "    d['k'] = v\n\n\n"
    "def reads(v, lock, text, other):\n"
    "    plain = []\n"
    "    plain.append(v)\n"
    "    lists = WeightedLists(lock)\n"
    "    lists.add(3, v)\n"
    "    log = LogWrapper(lock)\n"
    "    log.debug(v)\n"
    "    loaded = json.loads(text)\n"
    "    loaded.update(v)\n"
    "    taker = Taker()\n"
    "    taker.append(v)\n"
    "    holder = Holder()\n"
    "    holder.add(v)\n"
    "    spec = Spec()\n"
    "    spec.pack(v)\n"
    "    spec.__setstate__(other.__getstate__())\n"
    "    json.dumps(v)\n"
    "    fills(plain, v)\n"
)


@pytest.mark.parametrize(
    ("callee", "kind", "assumption"),
    [
        ("plain.append", callfacts.BUILTIN, None),
        ("lists.add", callfacts.LIBRARY, "library_keeps_what_it_is_handed"),
        ("log.debug", callfacts.LOGGER, "logger_keeps_nothing"),
        ("loaded.update", callfacts.ASSUMED, "builtin_by_name"),
        ("taker.append", callfacts.DEFINITION, None),
        ("holder.add", callfacts.UNKNOWN, None),
        ("spec.pack", callfacts.LIBRARY, "library_keeps_what_it_is_handed"),
        ("spec.__setstate__", callfacts.PROTOCOL, "pickle_round_trip"),
        ("json.dumps", callfacts.UNKNOWN, None),
        ("fills", callfacts.DEFINITION, None),
    ],
    ids=[
        "a-builtin-container",
        "a-library-class-the-corpus-annotates",
        "a-logger-s-method-of-a-library-class",
        "a-value-a-call-outside-the-corpus-made",
        "a-definition-of-the-corpus",
        "a-class-of-the-corpus-with-no-such-method",
        "a-base-outside-the-corpus",
        "the-pickle-round-trip",
        "a-module-s-function",
        "a-function-of-the-corpus",
    ],
)
def test_a_call_is_read_in_one_order(tmp_path, callee, kind, assumption):
    """The protocol first, then the definition the call lands at, a logger's method, a library object, a builtin
    container, the builtin of the name where the class cannot be said: ``WeightedLists(lock)`` is a library class's
    object because the corpus declares a parameter so -- ``json.loads(text)`` reads the same and is not."""
    calls = calls_mod.Calls(calls_mod.Types(_tree(tmp_path, mod=_MODULE), {}))
    file = f"{PACKAGE}/mod.py"
    func = next(
        node for node in ast.walk(calls.types.tree(file)) if isinstance(node, ast.FunctionDef) and node.name == "reads"
    )
    call = next(node for node in ast.walk(func) if isinstance(node, ast.Call) and ast.unparse(node.func) == callee)

    meant = callfacts.meaning(calls, file, func, call)

    assert (meant.kind, meant.assumption) == (kind, assumption)
    if kind == callfacts.PROTOCOL:
        assert ast.unparse(meant.copied_from) == "other"
