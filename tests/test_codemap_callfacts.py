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
        ("sorted", (callfacts.COPY, None)),
        ("next", (callfacts.ONE, 1)),
        ("min", (callfacts.ONE, None)),
        ("getattr", (callfacts.ATTRIBUTE, 2)),
        ("len", (callfacts.COMPUTED, None)),
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
        "the-builtin-computes",
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
    function = callfacts.builtin_function(types, file, func, call)
    assert (function.gives, function.default) == expected if expected is not None else function is None
    assert callfacts.builtin_function_at(call) == function
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



@pytest.mark.parametrize(
    ("text", "argument", "expected"),
    [
        ("isinstance(x, C)", "C", True),
        ("str(C)", "C", True),
        ("getattr(C, name)", "C", True),
        ("getattr(x, name, C)", "C", False),
        ("super(C, self)", "C", True),
        ("super(C, self)", "self", False),
        ("sorted(xs, key=C)", "C", False),
        ("list(C)", "C", False),
    ],
    ids=["a-test", "a-text", "an-attribute-of-it", "a-default", "super-s-class", "super-s-object", "a-key", "a-copy"],
)
def test_a_builtin_builds_nothing_it_neither_calls_nor_holds_as_it_is(text, argument, expected):
    """``isinstance``, ``str``, ``getattr(C, name)`` and ``super``'s class read a class and run no constructor of it;
    ``getattr``'s default may be the value, ``sorted`` calls its key, and ``list`` holds what iterating it gives."""
    call = _call(text)
    handed = [*call.args, *(keyword.value for keyword in call.keywords)]
    one = next(node for node in handed if ast.unparse(node) == argument)
    assert callfacts.builds_nothing(callfacts.BUILTIN_FUNCTIONS[call.func.id], call, one) is expected


_CONVERTS = (
    "def plain(x, spec):\n"
    "    a = str(int(x))\n"
    "    b = float(x)\n"
    "    c = repr(x)\n"
    "    d = format(x, spec)\n\n\n"
    "def bound(x, str):\n"
    "    e = str(x)\n"
)


def test_a_conversion_hands_one_value_on_where_the_name_is_the_builtin(tmp_path):
    """``str(int(x))``, ``float(x)`` and ``repr(x)`` are ``x`` as another type; ``format(x, spec)`` writes it as the
    spec says, and a ``str`` the function binds is not the builtin.  Read off the call's parents
    (:func:`calls.is_builtin_at`), the scopes say the same; a tree with no parents has none to read."""
    types = calls_mod.Types(_tree(tmp_path, mod=_CONVERTS), {})
    file = f"{PACKAGE}/mod.py"
    got = {}
    for func in ast.walk(types.tree(file)):
        if isinstance(func, ast.FunctionDef):
            for node in ast.walk(func):
                if isinstance(node, ast.Assign):
                    got[node.targets[0].id] = ast.unparse(callfacts.unconverted(types, file, func, node.value))
                    called = node.value.func.id
                    assert calls_mod.is_builtin_at(node.value, called) is types.is_builtin(file, func, called)
    assert got == {"a": "x", "b": "x", "c": "x", "d": "format(x, spec)", "e": "str(x)"}
    with pytest.raises(ValueError):
        calls_mod.is_builtin_at(_call("str(x)"), "str")

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


@pytest.mark.parametrize(
    ("method", "level"),
    [("debug", "debug"), ("warn", "warning"), ("exception", "error"), ("log", None), ("sendMsg", None)],
    ids=["debug", "warn-is-warning", "exception-writes-an-error", "log-has-no-level-of-its-own", "sendMsg-no-file"],
)
def test_every_reader_of_a_print_reads_its_level_from_one_table(method, level):
    """A print is a logger's method writing one line to its file at a level of its own (callfacts.LOG_LEVELS):
    selection's log_level (emit's prints, the selection and progress slices), loopcut's logging calls and
    score_trace's audit read the one table -- ``exception`` writes at ``error``, ``warn`` is ``warning``.  ``log``
    takes its level as an argument, and ``sendMsg`` sends to the HTTP logger: no print here."""
    from bamboo.codemap.panda.pathcond import attach_parents
    from bamboo.codemap.panda.recognizers import emit, loopcut, selection
    from bamboo.scripts.panda import score_trace

    tree = ast.parse(f"def f(log, rows):\n    for row in rows:\n        log.{method}('skip ' + row)\n        continue")
    attach_parents(tree)
    func = tree.body[0]
    (call,) = [node for node in ast.walk(func) if isinstance(node, ast.Call) and node.func.attr == method]

    assert selection.log_level(call.args[0]) == level
    assert emit.logging_arguments(func) == ([call.args[0]] if level else [])
    assert list(loopcut._logging_calls(func.body[0])) == ([call] if level else [])
    assert score_trace._logging_calls(func) == ({3: level} if level else {})


@pytest.mark.parametrize(
    ("text", "placed"),
    [
        ("self.cur.fetchone()", {("[2]",)}),
        ("self.cur.fetchall()", {("[*]", "[2]")}),
        ("self.cur.fetchmany(100)", {("[*]", "[2]")}),
        ("self.taskBufferIF.querySQL(sql, var_map)", {("[*]", "[2]")}),
        ("self.taskBuffer.querySQLS(sql, var_map)", {("[1]", "[*]", "[2]")}),
        ("self.cur.execute(sql, var_map)", set()),
    ],
    ids=["fetchone-one-row", "fetchall-the-rows", "fetchmany-the-rows", "querySQL-the-rows", "querySQLS-status-rows",
         "execute-binds-nothing"],
)
def test_a_call_giving_a_statements_rows_says_where_a_select_item_sits(text, placed):
    """The cursor's fetches and the task buffer's runners hand back one row or the rows, ``querySQLS`` inside its
    ``(status, rows)`` (callfacts.FETCHES, TASK_BUFFER_RUNS); ``execute`` binds nothing at the call."""
    from bamboo.codemap.panda.recognizers import sqlwrite

    assert sqlwrite._item_in(_call(text), 2) == placed
    assert sqlwrite._fetches(_call(text)) == text.startswith("self.cur.fetch")
    assert sqlwrite._fetches(ast.parse("self.cur.rowcount", mode="eval").body)


@pytest.mark.parametrize(
    ("text", "driver_class", "driver"),
    [
        ("self.conn.commit()", None, True),
        ("cur.fetchall()", None, True),
        ("self.cur.execute(sql, var_map)", None, True),
        ("self.proxy.fetchall()", "WrappedCursor", True),
        ("tbuf.querySQL(sql, var_map)", None, False),
    ],
    ids=["a-connection-by-its-name", "a-cursor-local", "a-cursor-field", "a-driver-by-its-class", "the-task-buffer"],
)
def test_a_call_on_the_driver_goes_only_to_the_database(text, driver_class, driver):
    """A call on the driver -- held as ``cur`` / ``conn``, or placed at a method of a driver class
    (callfacts.CURSOR_NAMES, CONNECTION_NAMES, DRIVER_CLASSES) -- is the database's.  A value read off a cursor is a
    statement's rows, off a connection none: provenance's stop names the cursor and the task buffer, not the
    connection."""
    from types import SimpleNamespace

    from bamboo.codemap.panda import plugin, provenance

    targets = [SimpleNamespace(cls=SimpleNamespace(name=driver_class))] if driver_class else []
    assert plugin._on_driver(_call(text), SimpleNamespace(targets=targets)) is driver
    assert {"cur", "taskBuffer", "taskBufferIF"} <= provenance.DATABASE_INTERFACES
    assert "conn" not in provenance.DATABASE_INTERFACES


def test_the_runners_where_no_corpus_is_read_are_the_cursors_and_the_task_buffers():
    """A reading with no corpus runs statements at the cursor's ``execute`` / ``executemany`` (its binds one map per
    row) and the task buffer's ``querySQL`` / ``querySQLS``; sql's runners and sqleval's are the same names."""
    from bamboo.codemap.panda import sql, sqleval

    runners = sql.runners()
    assert {name: runner.many for name, runner in runners.items()} == {
        "execute": False,
        "executemany": True,
        "querySQL": False,
        "querySQLS": False,
    }
    assert sqleval._RUNNERS == frozenset(runners)
    assert sql.CURSOR_RUNNERS == {"execute", "executemany"}


@pytest.mark.parametrize(
    ("name", "mapping", "collection", "record", "made", "ordered", "generator"),
    [
        ("Counter", True, False, False, True, False, False),
        ("DefaultDict", True, False, False, False, False, False),
        ("Collection", False, True, False, False, False, False),
        ("frozenset", False, True, False, True, False, False),
        ("deque", False, True, False, True, True, False),
        ("Iterable", False, True, False, False, False, True),
        ("Tuple", False, False, True, False, True, False),
    ],
)
def test_every_reader_of_a_container_name_reads_it_from_one_table(name, mapping, collection, record, made, ordered,
                                                                   generator):
    """What a container holds is said once (callfacts.CONTAINERS) and each reader asks it its own question: calls'
    loops and subscripts, reach's builtin values, sqlwrite's dicts and sequences, trace's empty constructors and
    attribution's records and contextmanager returns."""
    from bamboo.codemap import calls, reach, trace
    from bamboo.codemap.panda import attribution
    from bamboo.codemap.panda.recognizers import sqlwrite

    assert (name in calls._MAPPINGS, name in sqlwrite._DICT_TYPES) == (mapping, mapping)
    assert (name in calls._COLLECTIONS) is collection
    assert (name in calls._RECORDS, name in attribution._RECORD_WRAPPERS) == (record, record)
    assert (name in reach._BUILTIN_ANNOTATIONS) is (mapping or collection or record)
    assert (name in trace._EMPTY_CALLS, name in sqlwrite._DICT_MAKERS) == (made, made and mapping)
    assert (sqlwrite._annotation_kind(ast.parse(name, mode="eval").body) == sqlwrite._SEQUENCE) is ordered
    assert (name in calls._YIELDING) is generator


@pytest.mark.parametrize(
    ("constant", "folds", "named", "reads"),
    [
        ("'ABC'.lower()", ["abc"], "status.lower()", {"status"}),
        ("'abc'.capitalize()", ["Abc"], "status.capitalize()", {"status"}),
        ("'abc'.removesuffix('c')", None, "status.removesuffix(x)", {"status", "x"}),
        ("'abc'.startswith('a')", None, "status.startswith(x)", set()),
    ],
    ids=["lower", "capitalize", "removesuffix", "startswith"],
)
def test_every_reader_of_a_str_method_reads_it_from_one_table(constant, folds, named, reads):
    """What a ``str`` method's value is is said once (callfacts.TEXT_METHODS): the trace folds a constant through one
    changing the receiver and handed nothing, and sqleval reads the text a text method's value is made of -- not a
    test's (``startswith``).  The evaluator runs methods of the table only."""
    from bamboo.codemap import trace
    from bamboo.codemap.panda import sqleval

    assert trace._fold(ast.parse(constant, mode="eval").body, {}) == folds
    assert sqleval._text_reads(ast.parse(named, mode="eval").body) == reads
    assert sqleval._STR_METHODS <= set(callfacts.TEXT_METHODS)
