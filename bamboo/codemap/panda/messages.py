"""Which function a message a call sends is processed by: :class:`~bamboo.codemap.models.MessageEdge`.

``ContentsFeeder`` hands a task on with ``self.taskBufferIF.push_task_trigger_message("jedi_job_generator",
jediTaskID, ...)``; the proxy puts ``msg_type`` and ``taskid`` into a JSON message for the queue the type names, and
``JediJobGeneratorMsgProcPlugin.process`` takes it back -- ``msg_type = msg_dict["msg_type"]``, refusing any other
type, ``task_id = int(msg_dict["taskid"])``.  No call joins the two: the message crosses into another process, which
reads it in its own run.  So the edge is no edge of reach (:data:`bamboo.codemap.reach.PREMISES`
``cross_run_not_reach``: what a message carries is another run's input), but a record of its own beside the runs, as
a row one run writes and another reads is joined by its writer and its read site.

**A sender** is a function that puts one of its parameters into a dict under the key ``"msg_type"``
(``msg_dict.update({"msg_type": msg_type, "taskid": jedi_task_id, ...})``), each other key filled from a parameter
being the message's payload; or one that hands its own parameter on to a sender's type, the rest as a call of it
binds them (``JediTaskBuffer.push_task_trigger_message`` to the proxy's).  **A send** is a call placed at a sender
(:meth:`Calls.callees`) handing its type a literal.  **A receiver** is a function that reads ``X["msg_type"]`` of a
dict into a name and leaves -- raises or returns -- under ``if name != "<type>":``: it processes that type only.
Which receiver listens to which queue is the deployment's; the edge says only that this receiver takes no other type.
"""

from __future__ import annotations

import ast
from typing import Iterable, Optional

from bamboo.codemap import calls as C
from bamboo.codemap import pathcond
from bamboo.codemap.models import KEY_ROWS, MessageEdge, MessageRead, SourceModule

#: The key a message's type is put under.
TYPE_KEY = "msg_type"


def _functions(modules: Iterable[SourceModule]):
    for module in modules:
        for node in ast.walk(module.tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                yield module.rel_path, node


def _parameters(func: ast.AST) -> list[str]:
    arguments = func.args  # type: ignore[attr-defined]
    return [one.arg for one in (*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs)]


def _built(func: ast.AST) -> Optional[tuple[str, dict[str, str]]]:
    """``(type parameter, {payload key: parameter})`` where *func* puts a parameter into a dict under
    :data:`TYPE_KEY`."""
    names = set(_parameters(func))
    for node in C.own_nodes(func):
        if not isinstance(node, ast.Dict):
            continue
        entries = {
            key.value: value.id
            for key, value in zip(node.keys, node.values, strict=True)
            if isinstance(key, ast.Constant) and isinstance(key.value, str) and isinstance(value, ast.Name)
            and value.id in names
        }
        if TYPE_KEY in entries:
            kind = entries.pop(TYPE_KEY)
            return kind, entries
    return None


def _senders(modules: list[SourceModule], calls: C.Calls) -> dict[int, tuple[str, dict[str, str]]]:
    """``{id(func): (type parameter, {payload key: parameter})}`` for each sender: the builders, and the functions
    handing a parameter of their own on to a sender's type, until no more are found."""
    found: dict[int, tuple[str, dict[str, str]]] = {}
    functions = list(_functions(modules))
    for _file, func in functions:
        built = _built(func)
        if built is not None:
            found[id(func)] = built
    while True:
        names = {func.name for _file, func in functions if id(func) in found}  # type: ignore[attr-defined]
        more = 0
        for file, func in functions:
            if id(func) in found:
                continue
            own = set(_parameters(func))
            for call in _named_calls(func, names):
                for target, bound in _placed(calls, file, func, call, found):
                    kind_param, payload = found[id(target.func)]
                    handed = bound.get(kind_param)
                    if not (isinstance(handed, ast.Name) and handed.id in own):
                        continue
                    found[id(func)] = (
                        handed.id,
                        {
                            key: one.id
                            for key, param in payload.items()
                            if isinstance(one := bound.get(param), ast.Name) and one.id in own
                        },
                    )
                    more += 1
                    break
                if id(func) in found:
                    break
        if not more:
            return found


def _named_calls(func: ast.AST, names: set[str]) -> list[ast.Call]:
    """*func*'s own calls of a function or method of one of *names*: the ones a placement is asked of."""
    return [
        node
        for node in C.own_nodes(func)
        if isinstance(node, ast.Call)
        and (
            (isinstance(node.func, ast.Attribute) and node.func.attr in names)
            or (isinstance(node.func, ast.Name) and node.func.id in names)
        )
    ]


def _placed(calls: C.Calls, file: str, func: ast.AST, call: ast.Call, senders: dict[int, tuple[str, dict[str, str]]]):
    """``(target, {parameter: expression})`` for each sender *call* is placed at."""
    try:
        placed = calls.callees(file, func, call)
    except Exception:  # noqa: BLE001 -- a call the corpus cannot place sends nothing read here
        return []
    return [
        (target, C.bind(call, target, placed.on_class))
        for target in placed.targets
        if id(target.func) in senders
    ]


def _receivers(modules: list[SourceModule]) -> dict[str, list[tuple[str, ast.AST, list[MessageRead]]]]:
    """``{type: [(file, func, its reads)]}``: each function processing messages of that one type."""
    found: dict[str, list[tuple[str, ast.AST, list[MessageRead]]]] = {}
    for file, func in _functions(modules):
        taken = _taken(func)
        if taken is None:
            continue
        dict_name, kind = taken
        reads = [
            MessageRead(key=key, line=node.lineno, name=target.id)
            for node in C.own_nodes(func)
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(target := node.targets[0], ast.Name)
            for key in _keys_read(node.value, dict_name)
            if key != TYPE_KEY
        ]
        found.setdefault(kind, []).append((file, func, reads))
    return found


def _keys_read(value: ast.expr, dict_name: str) -> list[str]:
    """The keys of *dict_name* *value* reads whole or through one call (``int(msg_dict["taskid"])``)."""
    node = value.args[0] if isinstance(value, ast.Call) and len(value.args) == 1 and not value.keywords else value
    if (
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name)
        and node.value.id == dict_name
        and isinstance(node.slice, ast.Constant)
        and isinstance(node.slice.value, str)
    ):
        return [node.slice.value]
    return []


def _taken(func: ast.AST) -> Optional[tuple[str, str]]:
    """``(the message dict's name, the one type taken)`` where *func* reads the type into a name and leaves under
    ``if name != "<type>":``."""
    typed: dict[str, str] = {}
    for node in C.own_nodes(func):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            value = node.value
            if (
                isinstance(value, ast.Subscript)
                and isinstance(value.value, ast.Name)
                and isinstance(value.slice, ast.Constant)
                and value.slice.value == TYPE_KEY
            ):
                typed[node.targets[0].id] = value.value.id
    for node in C.own_nodes(func):
        if not (isinstance(node, ast.If) and isinstance(node.test, ast.Compare) and len(node.test.ops) == 1):
            continue
        if not isinstance(node.test.ops[0], ast.NotEq):
            continue
        sides = [node.test.left, node.test.comparators[0]]
        names = [one.id for one in sides if isinstance(one, ast.Name) and one.id in typed]
        literals = [one.value for one in sides if isinstance(one, ast.Constant) and isinstance(one.value, str)]
        if len(names) == 1 and len(literals) == 1 and _leaves(node.body):
            return typed[names[0]], literals[0]
    return None


def _leaves(body: list[ast.stmt]) -> bool:
    """Whether *body* ends by raising or returning."""
    return bool(body) and isinstance(body[-1], (ast.Raise, ast.Return))


def edges(modules: list[SourceModule], calls: Optional[C.Calls], reader=None) -> list[MessageEdge]:
    """Every send of a literal type joined to each receiver taking only that type (see the module), with what the
    send hands each payload key, the columns the value is where its ways end at rows (*reader*:
    :func:`bamboo.codemap.panda.heldfrom.value_reader`), and where the receiver binds each key."""
    if calls is None:
        return []
    senders = _senders(modules, calls)
    receivers = _receivers(modules)
    if not senders or not receivers:
        return []
    names = {
        func.name for _file, func in _functions(modules) if id(func) in senders  # type: ignore[attr-defined]
    }
    found: list[MessageEdge] = []
    for file, func in _functions(modules):
        if id(func) in senders:
            continue
        for call in _named_calls(func, names):
            for target, bound in _placed(calls, file, func, call, senders):
                kind_param, payload = senders[id(target.func)]
                handed = bound.get(kind_param)
                if not (isinstance(handed, ast.Constant) and isinstance(handed.value, str)):
                    continue
                sent = {key: ast.unparse(one) for key, param in payload.items() if (one := bound.get(param)) is not None}
                columns = {
                    key: _columns(reader, one, call, file, func)
                    for key, param in payload.items()
                    if (one := bound.get(param)) is not None
                }
                for receiver_file, receiver, reads in receivers.get(handed.value, []):
                    found.append(
                        MessageEdge(
                            sender=f"{file}::{func.name}",  # type: ignore[attr-defined]
                            line=call.lineno,
                            msg_type=handed.value,
                            sent=sent,
                            columns={key: held for key, held in columns.items() if held},
                            receiver=f"{receiver_file}::{receiver.name}",  # type: ignore[attr-defined]
                            reads=reads,
                        )
                    )
                break
    unique = {(one.sender, one.line, one.receiver): one for one in found}
    return [unique[key] for key in sorted(unique)]


def _columns(reader, value: ast.expr, call: ast.Call, file: str, func: ast.AST) -> list[str]:
    """The columns, ``Class.attribute``, the ways of *value* end at as a statement's rows' item, by *reader*."""
    if reader is None:
        return []
    try:
        ways = reader.read(value, pathcond.site_of(call), file, func)
    except Exception:  # noqa: BLE001 -- a value the reader cannot follow names no column
        return []
    return sorted({way.attr for way in ways if way.kind == KEY_ROWS and way.attr})
