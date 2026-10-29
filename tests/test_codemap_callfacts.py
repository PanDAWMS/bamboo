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
    set's elements together; ``pop`` puts nothing, changes its receiver and hands back one of what it holds."""
    setdefault = callfacts.builtin_method("dict", "setdefault")
    assert setdefault is not None and [put.into for put in setdefault.puts] == [callfacts.KEY, callfacts.VALUE]
    assert {put.into for put in callfacts.by_name("update").puts} == {
        callfacts.MERGE,
        callfacts.VALUE,
        callfacts.ELEMENTS,
    }
    assert callfacts.by_name("pop") == callfacts.Method((), True, callfacts.Give(callfacts.AT, 0, 1))
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


@pytest.mark.parametrize(
    ("name", "type_name", "expected"),
    [
        ("get", "dict", callfacts.Give(callfacts.AT, 0, 1)),
        ("pop", "list", callfacts.Give(callfacts.AT, 0)),
        ("pop", None, callfacts.Give(callfacts.AT, 0, 1)),
        ("popleft", None, callfacts.Give(callfacts.AT)),
        ("popitem", "dict", callfacts.Give(callfacts.ITEM)),
        ("items", "dict", callfacts.Give(callfacts.ITEMS)),
        ("keys", None, callfacts.Give(callfacts.KEYS)),
        ("copy", "set", callfacts.Give(callfacts.SELF)),
        ("get", "list", callfacts.Give(callfacts.AT, 0, 1)),
        ("append", "list", None),
    ],
    ids=[
        "get-the-value-at-its-key-or-the-default",
        "a-list-s-pop-at-its-index",
        "pop-by-its-name",
        "popleft-any-one",
        "popitem-one-pair",
        "items-a-view-of-pairs",
        "keys-a-view-of-keys",
        "copy-another-of-its-make",
        "a-type-without-the-method-takes-the-name-s",
        "append-hands-back-nothing",
    ],
)
def test_what_a_method_hands_back_of_its_receiver_is_said_once(name, type_name, expected):
    """By its type where the reader knows it and the type has the method, by the name for every type having it
    otherwise: ``pop`` is one of what it holds for each, at the key a dict's is handed, with a default beside it; a
    receiver read as a list and handed ``get`` takes the name's."""
    assert callfacts.taken(name, type_name) == expected
    assert ({name} <= callfacts.TAKING) is (expected is not None)


_SHADOWED = (
    "from itertools import zip_longest as zip\n\n\n"
    "def filter(x):\n"
    "    return x\n\n\n"
    "def reads(list, xs, it, a, b):\n"
    "    max = 3\n"
    "    list(xs)\n"
    "    max(xs)\n"
    "    zip(xs)\n"
    "    filter(xs)\n"
    "    sorted(xs)\n"
    "    next(it, None)\n"
    "    min(a, b)\n"
    "    getattr(a, 'x', b)\n"
    "    len(xs)\n"
)


@pytest.mark.parametrize(
    ("callee", "expected"),
    [
        ("list", None),
        ("max", None),
        ("zip", None),
        ("filter", None),
        ("sorted", callfacts.Function(callfacts.COPY)),
        ("next", callfacts.Function(callfacts.ONE, 1)),
        ("min", callfacts.Function(callfacts.ONE)),
        ("getattr", callfacts.Function(callfacts.ATTRIBUTE, 2)),
        ("len", None),
    ],
    ids=[
        "a-parameter",
        "a-local-name",
        "an-import",
        "a-module-s-function",
        "the-builtin-copies",
        "the-builtin-picks-with-a-default",
        "the-builtin-picks",
        "the-builtin-takes-an-attribute",
        "a-builtin-whose-value-holds-nothing-handed",
    ],
)
def test_a_builtin_function_is_read_where_no_scope_binds_its_name(tmp_path, callee, expected):
    """``list(xs)`` with a parameter ``list``, ``max`` bound in the function, ``zip`` imported, ``filter`` defined
    in the module: none is the builtin, whose facts say what its value holds of what it is handed."""
    types = calls_mod.Types(_tree(tmp_path, mod=_SHADOWED), {})
    file = f"{PACKAGE}/mod.py"
    func = next(
        node for node in ast.walk(types.tree(file)) if isinstance(node, ast.FunctionDef) and node.name == "reads"
    )
    call = next(
        node
        for node in ast.walk(func)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == callee
    )
    assert callfacts.builtin_function(types, file, func, call) == expected
    assert types.is_builtin(file, func, callee) is (callee in ("sorted", "next", "min", "getattr", "len"))


@pytest.mark.parametrize(
    ("text", "elements", "whole"),
    [
        ("next(it, default)", ["it"], ["default"]),
        ("next(it)", ["it"], []),
        ("max(xs)", ["xs"], []),
        ("max(a, b)", [], ["a", "b"]),
        ("max(*xs, a)", ["xs"], ["a"]),
    ],
    ids=["next-or-its-default", "next", "one-of-the-elements", "one-of-the-arguments", "starred"],
)
def test_a_picked_value_is_an_element_of_the_first_argument_or_an_argument_itself(text, elements, whole):
    """``next(it, default)`` is one of ``it``'s elements or the default -- not ``it`` itself, as ``max(a, b)`` is
    ``a`` or ``b``."""
    call = _call(text)
    got = callfacts.picked(callfacts.BUILTIN_FUNCTIONS[call.func.id], call)
    assert ([ast.unparse(one) for one in got[0]], [ast.unparse(one) for one in got[1]]) == (elements, whole)


_MODULE = (
    "import json\n"
    "import smtplib\n"
    "from rucio.client import Client as RucioClient\n"
    "from pandacommon.pandamsgbkr import msg_processor\n"
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
    "class Mailer(smtplib.SMTP):\n"
    "    pass\n\n\n"
    "class Agent(msg_processor.MsgProcAgentBase):\n"
    "    pass\n\n\n"
    "def declares(lists: WeightedLists, log: LogWrapper, rucio: RucioClient) -> None:\n"
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
    "    rucio = RucioClient()\n"
    "    rucio.add_files_to_dataset(v)\n"
    "    mailer = Mailer()\n"
    "    mailer.sendmail(v)\n"
    "    agent = Agent()\n"
    "    agent.start_passive_mode(v)\n"
    "    json.dumps(v)\n"
    "    fills(plain, v)\n"
)


@pytest.mark.parametrize(
    ("callee", "kind", "assumption"),
    [
        ("plain.append", callfacts.BUILTIN, None),
        ("lists.add", callfacts.LIBRARY, "library_keeps_what_it_is_handed"),
        ("log.debug", callfacts.LOGGER, "logger_keeps_nothing"),
        ("rucio.add_files_to_dataset", callfacts.CLIENT, "other_systems_keep_nothing"),
        ("mailer.sendmail", callfacts.CLIENT, "other_systems_keep_nothing"),
        ("agent.start_passive_mode", callfacts.CLIENT, "other_systems_keep_nothing"),
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
        "another-system-s-client",
        "a-base-another-system-s-client",
        "a-base-of-a-module-imported-from-its-package",
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
    """The protocol first, then the definition the call lands at, a logger's method, another system's client (Rucio's
    ``Client``, a class built on ``smtplib.SMTP`` or on ``msg_processor.MsgProcAgentBase``) keeping nothing, a library
    object, a builtin container, the builtin of the name where the class cannot be said: ``WeightedLists(lock)`` is
    a library class's object because the corpus declares a parameter so -- ``json.loads(text)`` reads the same and is
    not."""
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
