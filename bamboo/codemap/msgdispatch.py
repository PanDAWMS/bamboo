"""Messages told apart by a type the message carries, as the code that sends and receives them states them.

One convention, used by every project built on ``swf-common-lib``'s agents and
by anything written the same way: a message is a dictionary whose ``msg_type``
says what it asks for, it is sent with ``<sender>.send_message(<destination>,
..)`` to a ``/queue/..`` or ``/topic/..``, and a receiver hands it to the
method named for its type, ``getattr(self, f"_handle_{msg_type}")``.  The names
are a :class:`Convention`, so code with other names is read by the same rule.

The recognizers are apart from any plugin, for each map's plugin to use on its
own code.  What they record is in the fields a map is made of, the ones
PanDA's own message reading fills, so two maps read by this rule -- or one
read by it and one by another -- bind without knowing of each other
(:mod:`bamboo.codemap.bind`):

* :func:`visit_send` records a send as an outbound ``message`` boundary named
  by its destination, the types sent in ``handed_over``, and a
  :class:`~bamboo.codemap.models.MessageEdge` per type with the sending
  function and line.
* :func:`visit_receivers` records a class that dispatches by type as an
  inbound ``message`` boundary named by the queue its module declares, and a
  :class:`~bamboo.codemap.models.MessageReceiver` per type with its handler.
* :func:`close`, once every module is read, gives each send whose receiver is
  in the same map that receiver's handler, and each receiver its senders.

Read from the code alone, so a destination is what the code states: a queue an
environment variable may rename (``EPICPROD_OPS_QUEUE``) is recorded under its
default with the variable in ``metadata["env"]``, and one built at run time is
:data:`RUNTIME_DESTINATION`.  A message a caller builds and hands to the
function that sends it leaves no edge: the function that sends writes no type.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from typing import Any, Iterator, Optional, Protocol

from bamboo.codemap.models import BoundaryNode, MessageEdge, MessageReceiver

#: The destination of a send whose first argument the code does not settle.
RUNTIME_DESTINATION = "<runtime destination>"

#: name -> (destination, the environment variable that may rename it or "")
Queues = dict[str, tuple[str, str]]

_DESTINATION = re.compile(r"^/(queue|topic)/")


@dataclass(frozen=True)
class Convention:
    """The names one way of sending typed messages uses."""

    #: The system a message crosses into: the broker, which keeps nothing of a message nobody took.
    broker: str = "activemq"
    #: The method that sends.
    send: str = "send_message"
    #: The class whose instances send through the broker; a send on anything else is one only where its
    #: destination reads as a queue or a topic.
    sender_class: str = "ActiveMQConnectionManager"
    #: The key a message's type is written under and told by.
    type_key: str = "msg_type"
    #: What a handler's name starts with; the rest is the type it takes.
    handler_prefix: str = "_handle_"


SWF = Convention()


class Recorder(Protocol):
    """What a plugin gives the recognizers to record on."""

    message_edges: list[MessageEdge]
    message_receivers: list[MessageReceiver]
    boundaries: dict[str, BoundaryNode]

    def boundary(
        self,
        owner: str,
        system: str,
        interface: str,
        transport: str,
        rel: str,
        line: int,
        *,
        kind: str = ...,
        direction: str = ...,
        verb: str = ...,
        metadata: Optional[dict[str, Any]] = ...,
    ) -> BoundaryNode: ...


def _dotted(node: ast.AST) -> Optional[str]:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def _own_nodes(func: ast.AST) -> Iterator[ast.AST]:
    """Every node of *func*'s body, not those of functions or classes nested in it."""
    stack = list(reversed(getattr(func, "body", [])))
    while stack:
        node = stack.pop()
        yield node
        if isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
        ):
            continue
        stack.extend(reversed(list(ast.iter_child_nodes(node))))


def _queue_default(value: ast.AST) -> Optional[tuple[str, str]]:
    """``'/queue/x'`` or ``os.environ.get('VAR', '/queue/x')`` / ``os.getenv(..)`` -> (destination, variable)."""
    if isinstance(value, ast.Constant) and isinstance(value.value, str):
        return (value.value, "") if _DESTINATION.match(value.value) else None
    if (
        isinstance(value, ast.Call)
        and (_dotted(value.func) or "") in ("os.environ.get", "os.getenv")
        and len(value.args) == 2
    ):
        variable, default = (
            a.value if isinstance(a, ast.Constant) else None for a in value.args
        )
        if (
            isinstance(variable, str)
            and isinstance(default, str)
            and _DESTINATION.match(default)
        ):
            return default, variable
    return None


def module_queues(tree: ast.Module) -> Queues:
    """The queues and topics a module's module- and class-level names declare."""
    queues: Queues = {}

    def visit(body: list[ast.stmt]) -> None:
        for stmt in body:
            if isinstance(stmt, ast.ClassDef):
                visit(stmt.body)
            elif (
                isinstance(stmt, ast.Assign)
                and len(stmt.targets) == 1
                and isinstance(stmt.targets[0], ast.Name)
            ):
                queue = _queue_default(stmt.value)
                if queue is not None:
                    queues[stmt.targets[0].id] = queue

    visit(tree.body)
    return queues


def _text(node: ast.AST) -> Optional[str]:
    """A string literal, or an f-string with ``{}`` for each part the code fills at run time."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        return "".join(
            str(v.value) if isinstance(v, ast.Constant) else "{}" for v in node.values
        )
    return None


def _assigned(func: ast.AST, name: str) -> list[ast.expr]:
    """What *func*'s own statements assign to *name*."""
    return [
        node.value
        for node in _own_nodes(func)
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)
    ]


def destination(
    node: ast.AST, func: ast.AST, constants: dict[str, list[str]], queues: Queues
) -> tuple[Optional[str], str]:
    """``(destination, environment variable)`` of a send's first argument; ``(None, "")`` where the code does
    not settle one: a parameter, a name assigned twice, a call."""
    text = _text(node)
    if text is not None:
        return text, ""
    if isinstance(node, ast.Name):
        local = _assigned(func, node.id)
        if local:
            text = _text(local[0]) if len(local) == 1 else None
            return text, ""
        name = node.id
    elif isinstance(node, ast.Attribute):  # self.BRAINS_QUEUE, a class-level name
        name = node.attr
    else:
        return None, ""
    if name in queues:
        return queues[name]
    values = constants.get(name, [])
    return (values[0], "") if len(values) == 1 else (None, "")


def message_types(func: ast.AST, convention: Convention = SWF) -> list[str]:
    """The types *func* writes: ``{'msg_type': 'x'}``, ``dict(msg_type='x')``, ``m['msg_type'] = 'x'``."""
    found: list[str] = []
    key_name = convention.type_key

    def add(value: Optional[ast.AST]) -> None:
        if (
            isinstance(value, ast.Constant)
            and isinstance(value.value, str)
            and value.value not in found
        ):
            found.append(value.value)

    for node in _own_nodes(func):
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values, strict=True):
                if isinstance(key, ast.Constant) and key.value == key_name:
                    add(value)
        elif isinstance(node, ast.Call) and _dotted(node.func) == "dict":
            for kw in node.keywords:
                if kw.arg == key_name:
                    add(kw.value)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if (
                    isinstance(target, ast.Subscript)
                    and isinstance(target.slice, ast.Constant)
                    and target.slice.value == key_name
                ):
                    add(node.value)
    return found


def _sends_through_broker(
    sender: ast.AST, func: ast.AST, convention: Convention
) -> bool:
    """True where *sender* is the broker's connection: ``ActiveMQConnectionManager()`` or a name bound to one."""

    def makes(value: ast.AST) -> bool:
        return isinstance(value, ast.Call) and (_dotted(value.func) or "").endswith(
            convention.sender_class
        )

    if makes(sender):
        return True
    return isinstance(sender, ast.Name) and any(
        makes(value) for value in _assigned(func, sender.id)
    )


def visit_send(
    recorder: Recorder,
    node: ast.AST,
    func: ast.AST,
    owner: str,
    rel: str,
    constants: dict[str, list[str]],
    queues: Queues,
    convention: Convention = SWF,
) -> None:
    """Record *node* on *recorder* where it is a send through the broker; another ``send_message`` (mail, a
    chat post) is left."""
    if not (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == convention.send
    ):
        return
    first = (
        node.args[0]
        if node.args
        else next((kw.value for kw in node.keywords if kw.arg == "destination"), None)
    )
    if first is None:
        return
    where, env = destination(first, func, constants, queues)
    if where is None or not _DESTINATION.match(where):
        if where is not None or not _sends_through_broker(
            node.func.value, func, convention
        ):
            return
        where, env = RUNTIME_DESTINATION, ""
    boundary = recorder.boundary(
        owner, convention.broker, where, "message", rel, node.lineno
    )
    if env:
        boundary.metadata["env"] = env
    for sent in message_types(func, convention):
        if sent not in boundary.handed_over:
            boundary.handed_over.append(sent)
        recorder.message_edges.append(
            MessageEdge(sender=owner, line=node.lineno, msg_type=sent, receiver="")
        )


def _dispatches(cls: ast.ClassDef, convention: Convention) -> bool:
    """True where the class looks a handler up by the message's type: ``getattr(self, f"_handle_{msg_type}", ..)``."""
    for node in ast.walk(cls):
        if (
            isinstance(node, ast.Call)
            and _dotted(node.func) == "getattr"
            and len(node.args) >= 2
            and (_text(node.args[1]) or "").startswith(convention.handler_prefix + "{")
        ):
            return True
    return False


def visit_receivers(
    recorder: Recorder,
    tree: ast.Module,
    rel: str,
    queues: Queues,
    convention: Convention = SWF,
) -> None:
    """A class that hands each message to ``_handle_<msg_type>`` receives those types from the queue its module
    declares; where the module declares none or several, the boundary is named by the class.
    """
    declared = {queue for queue in queues.values() if queue[0].startswith("/queue/")}
    prefix = convention.handler_prefix
    for cls in tree.body:
        if not isinstance(cls, ast.ClassDef) or not _dispatches(cls, convention):
            continue
        handlers = {
            stmt.name[len(prefix) :]: f"{rel}::{cls.name}.{stmt.name}"
            for stmt in cls.body
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef))
            and stmt.name.startswith(prefix)
        }
        if not handlers:
            continue
        receiver = f"{rel}::{cls.name}"
        queue, env = (
            next(iter(declared)) if len(declared) == 1 else (f"<{receiver}>", "")
        )
        boundary = recorder.boundary(
            receiver,
            convention.broker,
            queue,
            "message",
            rel,
            cls.lineno,
            direction="inbound",
        )
        if env:
            boundary.metadata["env"] = env
        for msg_type, handler in sorted(handlers.items()):
            if msg_type not in boundary.carried_values:
                boundary.carried_values.append(msg_type)
            recorder.message_receivers.append(
                MessageReceiver(
                    receiver=receiver,
                    key=convention.type_key,
                    msg_type=msg_type,
                    handled_by=[handler],
                    boundary=boundary.name,
                )
            )


def close(recorder: Recorder) -> None:
    """Give each send whose destination a class of the same map receives from that class's handler for the
    type, and each receiver the sends of its type, ``file::name:line``."""
    destination_of = {
        site: node.interface
        for node in recorder.boundaries.values()
        if node.transport == "message" and node.direction == "outbound"
        for site in node.call_sites
    }
    receivers: dict[tuple[str, str], MessageReceiver] = {}
    for row in recorder.message_receivers:
        inbound = recorder.boundaries.get(row.boundary)
        if inbound is not None:
            receivers[(inbound.interface, row.msg_type)] = row
    for edge in recorder.message_edges:
        site = f"{edge.sender.partition('::')[0]}:{edge.line}"
        taking = receivers.get((destination_of.get(site, ""), edge.msg_type))
        if taking is None:
            continue
        edge.receiver = (taking.handled_by or [taking.receiver])[0]
        place = f"{edge.sender}:{edge.line}"
        if place not in taking.senders:
            taking.senders.append(place)
