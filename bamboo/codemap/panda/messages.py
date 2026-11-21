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
being the message's payload -- and the keywords of a call where it merges its ``**kwargs`` into that dict
(``make_message``'s ``msg_dict.update(kwargs)``); or one that hands its own parameter on to a sender's type, the
rest as a call of it binds them (``JediTaskBuffer.push_task_trigger_message`` to the proxy's).  **A send**
(:func:`sends`) is a call placed at a sender (:meth:`Calls.callees`) handing its type anything but a parameter of
its own, or a dict built with its type written in place (``{"msg_type": "job_status", ...}``): of a literal type,
of a pattern (``f"{command_str}_task"``), or of a type not read here -- an open send, which may be any.  **A
receiver** of an edge is a function that reads ``X["msg_type"]`` of a dict into a name and leaves -- raises or
returns -- under ``if name != "<type>":``: it processes that type only.  Which receiver listens to which queue is the
deployment's; the edge says only that this receiver takes no other type.

**Every receiver** (:func:`receivers`) is read apart from the edges: each corpus override of the message processor
plugins' ``process`` (:data:`PLUGIN_BASE`), with the types its tests take and the sends of each.  A type no send of
the corpus carries, and no open send could, came from outside PanDA: an inbound boundary.

**What a key is received as** (:func:`attach_received_as`) is read from the records the build has made, not from
the code again: a :class:`~bamboo.codemap.models.KeyFrom`'s parameter source is followed back through the
:class:`~bamboo.codemap.models.ParameterFrom` of each function handed the whole value on, to the receiver's
parameter with the payload key as the last of the way's steps.
"""

from __future__ import annotations

import ast
import re
from typing import Iterable, NamedTuple, Optional

from bamboo.codemap import calls as C
from bamboo.codemap import pathcond
from bamboo.codemap.models import (
    KEY_PARAMETER,
    KEY_ROWS,
    THROUGH_CONVERTED,
    THROUGH_COPY,
    THROUGH_DECODED,
    Anchor,
    BoundaryNode,
    HeldSource,
    KeyFrom,
    MessageEdge,
    MessageRead,
    MessageReceiver,
    ParameterFrom,
    SourceModule,
)

#: The key a message's type is put under.
TYPE_KEY = "msg_type"
#: The head of a sender's payload entry for a parameter it merges into the message whole (``**kwargs``, a dict).
SPREAD = "**"
#: The base of a message processor's plugins, ``(module, class)``, outside the corpus as the broker's package is
#: (:data:`bamboo.codemap.callfacts.OTHER_SYSTEMS`).  A thread taking messages off a queue hands each to its plugin's
#: :data:`ENTRY` (``SimpleMsgProcThread.run``: ``self.plugin.process(_msg_obj)``), so every override of it in the
#: corpus is a receiver; which plugin listens to which queue is the deployment's.
PLUGIN_BASE = ("pandacommon.pandamsgbkr.msg_processor", "SimpleMsgProcPluginBase")
ENTRY = "process"
#: The transport a message boundary is crossed by.
TRANSPORT = "message"


def _functions(modules: Iterable[SourceModule]):
    for module in modules:
        for node in ast.walk(module.tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                yield module.rel_path, node


def _parameters(func: ast.AST) -> list[str]:
    arguments = func.args  # type: ignore[attr-defined]
    return [one.arg for one in (*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs)]


def _typed(func: ast.AST, key: str) -> list[tuple[ast.AST, ast.expr, dict[str, ast.expr]]]:
    """``(node, type expression, {payload key: expression})``, by place, for each dict *func* builds holding *key* --
    a literal's entry (``{"msg_type": msg_type, ...}``, inside an ``update`` too), the literal's other constant keys
    its payload -- and each store of it (``msg_dict["msg_type"] = ...``), with no payload read."""
    found: list[tuple[ast.AST, ast.expr, dict[str, ast.expr]]] = []
    for node in C.own_nodes(func):
        if isinstance(node, ast.Dict):
            entries = {
                one.value: value
                for one, value in zip(node.keys, node.values, strict=True)
                if isinstance(one, ast.Constant) and isinstance(one.value, str)
            }
            if key in entries:
                kind = entries.pop(key)
                found.append((node, kind, entries))
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Subscript) and isinstance(target.slice, ast.Constant) and target.slice.value == key:
                    found.append((node, node.value, {}))
    return sorted(found, key=lambda one: (one[0].lineno, one[0].col_offset))  # type: ignore[attr-defined]


def _merged(func: ast.AST, built: ast.AST) -> list[str]:
    """*func*'s parameters it merges into the dict *built*, in order: ``{..., **kwargs}`` in it, or ``x.update(p)``
    on a name holding it -- bound to it (``msg_dict: dict = {"msg_type": msg_type}``) or updated with it
    (``msg_dict.update({"msg_type": msg_type, ...})``)."""
    names = set(_parameters(func))
    spread = func.args.kwarg  # type: ignore[attr-defined]
    if spread is not None:
        names.add(spread.arg)
    found: list[str] = []
    if isinstance(built, ast.Dict):
        found.extend(
            value.id
            for one, value in zip(built.keys, built.values, strict=True)
            if one is None and isinstance(value, ast.Name) and value.id in names
        )
    holders: set[str] = set()
    updates: list[ast.Call] = []
    for node in C.own_nodes(func):
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is built:
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            holders.update(one.id for one in targets if isinstance(one, ast.Name))
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "update"
            and isinstance(node.func.value, ast.Name)
            and len(node.args) == 1
            and not node.keywords
        ):
            updates.append(node)
            if node.args[0] is built:
                holders.add(node.func.value.id)
    for node in sorted(updates, key=lambda one: (one.lineno, one.col_offset)):
        handed = node.args[0]
        if node.func.value.id in holders and isinstance(handed, ast.Name) and handed.id in names:  # type: ignore[attr-defined]
            found.append(handed.id)
    return list(dict.fromkeys(found))


def _built(func: ast.AST, key: str = TYPE_KEY) -> Optional[tuple[str, dict[str, str]]]:
    """``(type parameter, {payload key: parameter})`` where *func* puts a parameter into a dict under *key*; each
    parameter it merges into that dict under :data:`SPREAD` and its name (``"**kwargs"``)."""
    names = set(_parameters(func))
    for node, kind, entries in _typed(func, key):
        if isinstance(kind, ast.Name) and kind.id in names:
            payload = {
                one: value.id for one, value in entries.items() if isinstance(value, ast.Name) and value.id in names
            }
            payload.update({f"{SPREAD}{one}": one for one in _merged(func, node)})
            return kind.id, payload
    return None


def _senders(
    modules: list[SourceModule], calls: C.Calls, key: str = TYPE_KEY
) -> dict[int, tuple[str, dict[str, str]]]:
    """``{id(func): (type parameter, {payload key: parameter})}`` for each sender of messages typed under *key*: the
    builders, and the functions handing a parameter of their own on to a sender's type, until no more are found."""
    found: dict[int, tuple[str, dict[str, str]]] = {}
    functions = list(_functions(modules))
    for _file, func in functions:
        built = _built(func, key)
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
                            one_key: one.id
                            for one_key, one in _payload(payload, bound).items()
                            if isinstance(one, ast.Name) and one.id in own
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


class Send(NamedTuple):
    """A message the corpus sends: a call placed at a sender handing its type anything but a parameter of its own,
    or a dict built with its type written in place."""

    file: str
    func: ast.AST
    #: The call, or the dict or store naming the type.
    node: ast.AST
    #: What the type is handed or written as: a literal, an f-string, anything else; ``None`` where it is not said.
    kind: Optional[ast.expr]
    #: Each payload key, the expression the send hands it.
    payload: dict[str, ast.expr]

    @property
    def place(self) -> str:
        return f"{self.file}::{C.def_name(self.func)}:{self.node.lineno}"  # type: ignore[attr-defined]

    @property
    def literal(self) -> Optional[str]:
        """The type, where the send writes it out."""
        kind = self.kind
        return kind.value if isinstance(kind, ast.Constant) and isinstance(kind.value, str) else None

    @property
    def pattern(self) -> Optional[re.Pattern[str]]:
        """The types an f-string type may be: its text, each hole any text."""
        if not isinstance(self.kind, ast.JoinedStr):
            return None
        parts = [
            re.escape(one.value) if isinstance(one, ast.Constant) and isinstance(one.value, str) else ".+"
            for one in self.kind.values
        ]
        return re.compile("".join(parts))

    @property
    def open(self) -> bool:
        """Whether the type is not read here, so the send may carry any."""
        return self.literal is None and self.pattern is None

    def carries(self, kind: str) -> bool:
        """Whether the send is of the type *kind*: written out so, or an f-string it matches."""
        pattern = self.pattern
        return self.literal == kind or (pattern is not None and pattern.fullmatch(kind) is not None)


def _payload(payload: dict[str, str], bound: dict[str, Optional[ast.expr]]) -> dict[str, ast.expr]:
    """``{payload key: expression}`` a call binding *bound* hands a sender's *payload*: a merged parameter's
    (:data:`SPREAD`) dict -- a literal handed it, or the keywords a ``**`` parameter holds -- each key under its own
    name."""
    found: dict[str, ast.expr] = {}
    for key, param in payload.items():
        one = bound.get(param)
        if one is None:
            continue
        if not key.startswith(SPREAD):
            found[key] = one
        elif isinstance(one, ast.Dict):
            found.update(
                {
                    name.value: value
                    for name, value in zip(one.keys, one.values, strict=True)
                    if isinstance(name, ast.Constant) and isinstance(name.value, str)
                }
            )
    return found


def sends(modules: list[SourceModule], calls: Optional[C.Calls], key: str = TYPE_KEY) -> list[Send]:
    """Every send in the corpus of a message typed under *key* (see the module), by place.  A call is read at the
    first sender it is placed at handing it a literal type, else an f-string, else at the first one: an open send,
    unless what it hands is a parameter of its own -- a sender's hand-on, no send."""
    if calls is None:
        return []
    senders = _senders(modules, calls, key)
    names = {
        func.name for _file, func in _functions(modules) if id(func) in senders  # type: ignore[attr-defined]
    }
    found: list[Send] = []
    for file, func in _functions(modules):
        own = set(_parameters(func))
        for node, kind, entries in _typed(func, key):
            if not (isinstance(kind, ast.Name) and kind.id in own):
                found.append(Send(file, func, node, kind, entries))
        for call in _named_calls(func, names):
            placed = [
                Send(file, func, call, bound.get(senders[id(target.func)][0]), _payload(senders[id(target.func)][1], bound))
                for target, bound in _placed(calls, file, func, call, senders)
            ]
            placed = [one for one in placed if not (isinstance(one.kind, ast.Name) and one.kind.id in own)]
            chosen = (
                next((one for one in placed if one.literal is not None), None)
                or next((one for one in placed if one.pattern is not None), None)
                or next(iter(placed), None)
            )
            if chosen is not None:
                found.append(chosen)
    return sorted(found, key=lambda one: (one.file, one.node.lineno, one.node.col_offset))  # type: ignore[attr-defined]


def edges(
    modules: list[SourceModule], calls: Optional[C.Calls], reader=None, sent: Optional[list[Send]] = None
) -> list[MessageEdge]:
    """Every send of a literal type (*sent*, else :func:`sends`) joined to each receiver taking only that type (see
    the module), with what the send hands each payload key, the columns the value is where its ways end at rows
    (*reader*: :func:`bamboo.codemap.panda.heldfrom.value_reader`), and where the receiver binds each key."""
    if calls is None:
        return []
    receivers = _receivers(modules)
    if not receivers:
        return []
    found: list[MessageEdge] = []
    for send in sends(modules, calls) if sent is None else sent:
        kind = send.literal
        if kind is None or kind not in receivers:
            continue
        columns = {
            key: _columns(reader, one, send.node, send.file, send.func) for key, one in send.payload.items()
        }
        for receiver_file, receiver, reads in receivers[kind]:
            found.append(
                MessageEdge(
                    sender=f"{send.file}::{C.def_name(send.func)}",
                    line=send.node.lineno,  # type: ignore[attr-defined]
                    msg_type=kind,
                    sent={key: ast.unparse(one) for key, one in send.payload.items()},
                    columns={key: held for key, held in columns.items() if held},
                    receiver=f"{receiver_file}::{C.def_name(receiver)}",
                    reads=reads,
                )
            )
    unique = {(one.sender, one.line, one.receiver): one for one in found}
    return [unique[key] for key in sorted(unique)]


def _columns(reader, value: ast.expr, call: ast.AST, file: str, func: ast.AST) -> list[str]:
    """The columns, ``Class.attribute``, the ways of *value* end at as a statement's rows' item, by *reader*."""
    if reader is None:
        return []
    try:
        ways = reader.read(value, pathcond.site_of(call), file, func)
    except Exception:  # noqa: BLE001 -- a value the reader cannot follow names no column
        return []
    return sorted({way.attr for way in ways if way.kind == KEY_ROWS and way.attr})


#: What a call outside the corpus may do to a received value on its way to a term and leave it the value sent: read
#: it out of the message's text, convert it (``int``), copy it.
_UNCHANGED = frozenset({THROUGH_DECODED, THROUGH_CONVERTED, THROUGH_COPY})


def _payload_key(step: str) -> Optional[str]:
    """The key of an item step spelled ``['taskid']``; ``None`` for any other step."""
    if not (step.startswith("[") and step.endswith("]")):
        return None
    try:
        key = ast.literal_eval(step[1:-1])
    except (ValueError, SyntaxError):
        return None
    return key if isinstance(key, str) else None


def _unchanged(source: HeldSource) -> bool:
    """Whether *source*'s way hands on the value it ends at, whole and as it was: a parameter's, sitting at no part of
    the value, no field of it, changed in place nowhere, through nothing but :data:`_UNCHANGED`."""
    return (
        source.kind == KEY_PARAMETER
        and not source.paths
        and not source.attr
        and not source.changes
        and all(one.gives in _UNCHANGED for one in source.through)
    )


def attach_received_as(
    edges: Iterable[MessageEdge], key_froms: Iterable[KeyFrom], parameter_froms: Iterable[ParameterFrom]
) -> None:
    """Fill each edge's :attr:`~MessageEdge.received_as` from the records: the columns of the terms each payload key
    names rows by in the receiver's run.

    A :class:`KeyFrom` whose term has a column of a class (*attr*) and whose source is a parameter, the bind's value
    itself, points at that parameter's :class:`ParameterFrom` (*site_owner*, *site_line*); each source there that
    is a caller's parameter, the whole value handed on, points at the caller's -- a chain read to its end.  Where a
    source's way ends at a receiver's parameter, the last of its steps the item of a payload key and every call it
    went through one that leaves the value as sent (:data:`_UNCHANGED`), the term names rows by that key:
    ``JediTaskBuffer.getTaskWithID_JEDI``'s ``jediTaskID`` is ``msg_obj.data['taskid']`` of
    ``JediJobGeneratorMsgProcPlugin.process`` through ``json.loads`` and ``int``, handed whole to the proxy's,
    whose ``jediTaskID=:jediTaskID`` reads ``JEDI_Tasks``.  A value sitting at a part of what is handed on
    (``target_tasks=[task_id]``), a field of it, one changed in place, or one through text or a computation names
    no row by the key."""
    edges = list(edges)
    receivers = {edge.receiver for edge in edges}
    if not receivers:
        return
    parameter_froms = list(parameter_froms)
    # Each ParameterFrom, the (receiver, payload key) pairs its parameter is the received value of.
    received: dict[tuple[str, Optional[int], str], set[tuple[str, str]]] = {}
    for one in parameter_froms:
        here = received.setdefault((one.owner, one.line, one.parameter), set())
        for source in one.sources:
            if source.site_owner in receivers and source.steps and _unchanged(source):
                key = _payload_key(source.steps[-1])
                if key is not None:
                    here.add((source.site_owner, key))
    grown = True
    while grown:
        grown = False
        for one in parameter_froms:
            here = received[(one.owner, one.line, one.parameter)]
            for source in one.sources:
                if source.steps or source.site_owner in receivers or not _unchanged(source):
                    continue
                further = received.get((source.site_owner, source.site_line, source.name))
                if further and not further <= here:
                    here |= further
                    grown = True
    columns: dict[tuple[str, str], set[str]] = {}
    for key_from in key_froms:
        if not key_from.attr:
            continue
        for bound in key_from.sources:
            parts = [path for path in bound.paths if path != "()"]
            if bound.kind != KEY_PARAMETER or bound.attr or ("()" not in bound.paths and parts):
                continue
            for pair in received.get((bound.site_owner, bound.site_line, bound.name), ()):
                columns.setdefault(pair, set()).add(key_from.attr)
    for edge in edges:
        edge.received_as = {
            key: sorted(columns[(edge.receiver, key)]) for key in sorted(edge.sent) if (edge.receiver, key) in columns
        }


def receivers(
    modules: list[SourceModule],
    calls: Optional[C.Calls],
    map_id: str,
    derived_from: str,
    sent: Optional[dict[str, list[Send]]] = None,
) -> tuple[list[MessageReceiver], list[BoundaryNode]]:
    """Every receiver -- :data:`PLUGIN_BASE`'s :data:`ENTRY` overridden in the corpus -- with each type its tests
    take, the functions it hands a message of the type to, and the sends of it (:func:`sends` under the key the
    receiver tells types by; *sent* holds those already read, by key); and an inbound boundary for each receiver
    taking a type no send carries and no open send could.

    A receiver's messages are its parameters and the names it binds to one of them or to a call handed one, or one's
    attribute, alone (``msg_dict = json.loads(msg_obj.data)``, ``yaml.safe_load``, ``copy.deepcopy(msg_dict)``).  A
    type is a literal an ``if`` compares a value read off a message under a key with -- ``==`` or ``in``, or ``!=`` /
    ``not in`` over a body that leaves, alone or in an ``and``.  PanDA's own senders put the type under
    :data:`TYPE_KEY`, so a receiver reading that key tells types by it, its tests of other keys choosing within a
    type (``relation_type``); one reading none tells them by the keys it does test, the DDM callback's
    ``event_type``.  A receiver testing none takes whatever its queue holds: one row with no key or type, no sends
    joined.  A body that only passes or raises (the base's own) receives nothing."""
    if calls is None:
        return [], []
    by_file = {module.rel_path: module for module in modules}
    sent = dict(sent or {})
    rows: list[MessageReceiver] = []
    boundaries: list[BoundaryNode] = []
    for where, cls in _plugins(calls, modules):
        for func in cls.body:
            if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)) or func.name != ENTRY or _abstract(func):
                continue
            receiver = f"{where}::{C.def_name(func)}"
            messages = _messages(func)
            taken = _type_tests(calls, where, func, messages)
            if not taken:
                rows.append(MessageReceiver(receiver=receiver))
                continue
            here: list[MessageReceiver] = []
            for key, kind, handled in taken:
                if key not in sent:
                    sent[key] = sends(modules, calls, key)
                here.append(
                    MessageReceiver(
                        receiver=receiver,
                        key=key,
                        msg_type=kind,
                        handled_by=handled,
                        senders=[one.place for one in sent[key] if one.carries(kind)],
                        senders_open=[one.place for one in sent[key] if one.open],
                    )
                )
            unsent = [one for one in here if not one.senders and not one.senders_open]
            if unsent:
                boundary = BoundaryNode.make_name(map_id, "", receiver)
                module = by_file.get(where)
                boundaries.append(
                    BoundaryNode(
                        map_id=map_id,
                        derived_from=derived_from,
                        name=boundary,
                        system="",
                        kind="transports_causation",
                        transport=TRANSPORT,
                        interface=receiver,
                        carried_values=_keys_off(func, messages),
                        anchor=Anchor(
                            package=module.package if module is not None else "",
                            file=where,
                            line_start=func.lineno,
                            line_end=func.end_lineno,
                            blob_sha=module.blob_sha if module is not None else None,
                        ),
                    )
                )
                for one in unsent:
                    one.boundary = boundary
            rows.extend(here)
    return rows, boundaries


def _plugins(calls: C.Calls, modules: list[SourceModule]) -> list[tuple[str, ast.ClassDef]]:
    """Every corpus class with :data:`PLUGIN_BASE` among the bases outside the corpus its order reads, by place."""
    found: list[tuple[str, ast.ClassDef]] = []
    for module in modules:
        for node in ast.walk(module.tree):
            if isinstance(node, ast.ClassDef) and PLUGIN_BASE in (calls.types.outside_bases(module.rel_path, node) or ()):
                found.append((module.rel_path, node))
    return sorted(found, key=lambda one: (one[0], one[1].lineno))


def _abstract(func: ast.AST) -> bool:
    """Whether *func*'s body, its docstring aside, only passes or raises."""
    body = list(func.body)  # type: ignore[attr-defined]
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        body = body[1:]
    return all(
        isinstance(one, (ast.Pass, ast.Raise))
        or (isinstance(one, ast.Expr) and isinstance(one.value, ast.Constant) and one.value.value is Ellipsis)
        for one in body
    )


def _root(expr: ast.expr) -> Optional[str]:
    """The name *expr* reads an attribute or item of, or is."""
    while isinstance(expr, (ast.Attribute, ast.Subscript)):
        expr = expr.value
    return expr.id if isinstance(expr, ast.Name) else None


def _messages(func: ast.AST) -> set[str]:
    """The names holding a message in *func*: its parameters but the object, and, in order, each name bound to one or
    to a call handed one alone."""
    found = set(_parameters(func)[1:])
    bindings = sorted(
        (node for node in C.own_nodes(func) if isinstance(node, ast.Assign) and len(node.targets) == 1),
        key=lambda node: (node.lineno, node.col_offset),
    )
    for node in bindings:
        target, value = node.targets[0], node.value
        if not isinstance(target, ast.Name):
            continue
        if isinstance(value, ast.Call) and len(value.args) == 1 and not value.keywords:
            value = value.args[0]
        if _root(value) in found and not isinstance(value, ast.Subscript):
            found.add(target.id)
    return found


def _key_off(expr: ast.expr, messages: set[str], keyed: dict[str, str]) -> Optional[str]:
    """The key *expr* reads a message's value under: ``m["k"]``, ``m.get("k")``, or a name bound to one."""
    if (
        isinstance(expr, ast.Subscript)
        and isinstance(expr.value, ast.Name)
        and expr.value.id in messages
        and isinstance(expr.slice, ast.Constant)
        and isinstance(expr.slice.value, str)
    ):
        return expr.slice.value
    if (
        isinstance(expr, ast.Call)
        and isinstance(expr.func, ast.Attribute)
        and expr.func.attr == "get"
        and isinstance(expr.func.value, ast.Name)
        and expr.func.value.id in messages
        and expr.args
        and isinstance(expr.args[0], ast.Constant)
        and isinstance(expr.args[0].value, str)
    ):
        return expr.args[0].value
    if isinstance(expr, ast.Name):
        return keyed.get(expr.id)
    return None


def _keys_off(func: ast.AST, messages: set[str]) -> list[str]:
    """Every key *func* reads a message's value under."""
    return sorted(
        {
            key
            for node in C.own_nodes(func)
            if isinstance(node, (ast.Subscript, ast.Call)) and (key := _key_off(node, messages, {})) is not None
        }
    )


def _compared(test: ast.expr, messages: set[str], keyed: dict[str, str]) -> Optional[tuple[str, list[str], bool]]:
    """``(key, literals, taking)`` where *test* compares a value read off a message under a key with literals:
    taking for ``==`` / ``in``, not for ``!=`` / ``not in``."""
    if not (isinstance(test, ast.Compare) and len(test.ops) == 1):
        return None
    op, left, right = test.ops[0], test.left, test.comparators[0]
    if isinstance(op, (ast.Eq, ast.NotEq)):
        for side, other in ((left, right), (right, left)):
            key = _key_off(side, messages, keyed)
            if key is not None and isinstance(other, ast.Constant) and isinstance(other.value, str):
                return key, [other.value], isinstance(op, ast.Eq)
        return None
    if isinstance(op, (ast.In, ast.NotIn)) and isinstance(right, (ast.Tuple, ast.List, ast.Set)) and right.elts:
        key = _key_off(left, messages, keyed)
        literals = [one.value for one in right.elts if isinstance(one, ast.Constant) and isinstance(one.value, str)]
        if key is not None and len(literals) == len(right.elts):
            return key, literals, isinstance(op, ast.In)
    return None


def _type_tests(calls: C.Calls, file: str, func: ast.AST, messages: set[str]) -> list[tuple[str, str, list[str]]]:
    """``(key, type, the functions a branch taking it hands a message to)`` for each type *func*'s tests take (see
    :func:`receivers`), by key and type."""
    keyed: dict[str, str] = {}
    for binding in sorted(
        (node for node in C.own_nodes(func) if isinstance(node, ast.Assign) and len(node.targets) == 1),
        key=lambda node: (node.lineno, node.col_offset),
    ):
        if isinstance(binding.targets[0], ast.Name) and (key := _key_off(binding.value, messages, keyed)) is not None:
            keyed[binding.targets[0].id] = key
    found: dict[tuple[str, str], set[str]] = {}
    for node in C.own_nodes(func):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        parts = test.values if isinstance(test, ast.BoolOp) and isinstance(test.op, ast.And) else [test]
        for part in parts:
            compared = _compared(part, messages, keyed)
            if compared is None:
                continue
            key, literals, taking = compared
            if not taking and not _leaves(node.body):
                continue
            handed = _handed(calls, file, func, node.body, messages) if taking else set()
            for kind in literals:
                found.setdefault((key, kind), set()).update(handed)
    if any(key == TYPE_KEY for key, _kind in found):
        found = {pair: handed for pair, handed in found.items() if pair[0] == TYPE_KEY}
    return [(key, kind, sorted(handed)) for (key, kind), handed in sorted(found.items())]


def _handed(calls: C.Calls, file: str, func: ast.AST, body: list[ast.stmt], messages: set[str]) -> set[str]:
    """The corpus functions, ``file::name``, a call in *body* handing a message (or a part of one) is placed at."""
    found: set[str] = set()
    for statement in body:
        for node in ast.walk(statement):
            if not isinstance(node, ast.Call):
                continue
            handed = [*node.args, *(one.value for one in node.keywords)]
            if not any(_root(one) in messages for one in handed if not isinstance(one, ast.Starred)):
                continue
            try:
                placed = calls.callees(file, func, node)
            except Exception:  # noqa: BLE001 -- a call the corpus cannot place hands the message to nothing read
                continue
            found.update(target.owner for target in placed.targets)
    return found
