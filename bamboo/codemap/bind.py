"""Bind maps to one another: where one map's code crosses into another system, the node of that system's map it reaches.

A map is built from one distribution's code and states its boundaries without
the far side, which is another distribution at a version of its own
(:mod:`bamboo.codemap.base`).  Binding reads two or more maps together and says
what each crossing reaches.  It needs nothing of the plugins that built the
maps: it reads the fields every map is made of, so a map of any system binds
as soon as its plugin fills them.

**Ports.**  A map *uses* what another system has and *offers* what it has
itself.  A :class:`Port` is one of either, read from the map
(:func:`ports`), and a use binds to an offer of the same kind and key:

``http``
    *Uses*: an outbound boundary whose far side is a URL path
    (``/api/v1/creds/get_access_token``).  *Offers*: the map's endpoints
    (:attr:`MapFragment.endpoints`, ``/api/<version>/<route>``), each with the
    inbound boundary of its function where the map has one, which carries the
    conditions a request is turned away by.

``call``
    *Uses*: an outbound boundary whose far side is a callable of a package
    (``rucio.client.Client.add_dataset``, ``pandaclient.Client.submitJobs``).
    *Offers*: every function a map knows, ``file::name`` spelled as the import
    path.  A client library is a map like any other: its functions are offers
    and the requests they make are ``http`` uses, so a call binds to the
    library and the library to the server.

``table``
    *Uses*: a ``shared_table`` boundary -- rows this map's code reads
    (``carried_values``) or writes (``handed_over``) in a table another system
    declares, named ``schema.table`` or by the model class the code imports
    (``ai.models.Proposal``).  *Offers*: the tables of a map's entities and
    table records, and its subjects, column by column.  The schema is the
    deployment's (``ATLAS_PANDA`` in one, ``doma_panda`` in another), so a
    table binds by its own name and the binding says both spellings.

``message``
    *Uses*: an outbound ``message`` boundary, named by its destination, and
    the types sent to it (:attr:`MapFragment.message_edges` placed at its
    call sites, or its ``handed_over``).  *Offers*: the types a receiver takes
    (:attr:`MapFragment.message_receivers`) and the inbound boundary each
    crosses by.  A type binds to the receiver's handler for it where both
    sides state the same destination, or where the receiver states none -- which
    receiver listens to which queue is then the deployment's, and the binding
    says so.

In an outbound boundary a name spelled ``file::name`` is a function of the map
itself and any other name is the far side's, whichever of ``interface`` and
``calls_out`` holds it: PanDA's map names the boundary by its own method and
lists the outside methods called, a map of code that calls out from anywhere
names it by the outside interface and lists its own functions.

**What is left.**  A use nothing offers is :class:`Unbound` with the reason --
no map given covers the system, the map that covers it has no such node, the
receiver has no handler for the type -- so the list of what is not connected
is as complete as the list of what is.  A message type a receiver takes that
nothing sends is listed apart (:attr:`Bindings.unreached`).

**The result is not a change to either map.**  A binding holds between two
versions and a map's dump is one version's, so :class:`Bindings` is an
artifact of its own, stamped with each map's version, and
``BoundaryNode.resolution`` is not written; :meth:`Bindings.resolution`
answers it for the maps that were bound.

A new kind of crossing is a reader of ports and, where the key alone does not
decide, a matcher (:func:`register_reader`, :func:`register_matcher`); the
binding itself does not change.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Callable, Iterable, Iterator, Optional

from pydantic import BaseModel, Field

from bamboo.codemap.models import BoundaryNode, MapFragment

HTTP = "http"
CALL = "call"
TABLE = "table"
MESSAGE = "message"

USES = "uses"
OFFERS = "offers"

#: A written value the code leaves to run time; not one to hold against a vocabulary.
_RUNTIME = "runtime("
_URL = re.compile(r"^(?:[A-Z]+\s+)?(?:https?://[^/]+)?(/[^?#\s]*)")


class Port(BaseModel):
    """One thing a map uses of another system, or offers to one."""

    map_id: str
    role: str = Field(description="``uses`` | ``offers``")
    kind: str = Field(
        description="``http`` | ``call`` | ``table`` | ``message``: what the key names."
    )
    key: str = Field(
        description="What a use and an offer must share, in one spelling for both sides."
    )
    part: str = Field(
        default="",
        description="The column of a table, lowered; empty for the table itself and for every other kind.",
    )
    spelling: str = Field(default="", description="The key as this map writes it.")
    qualifier: str = Field(
        default="",
        description=(
            "What narrows a key without being part of it: a message's destination, a table's schema.  Empty "
            "where the map states none."
        ),
    )
    node: str = Field(
        description="The node of the map at this port: a boundary, a subject, an entity, a function."
    )
    functions: list[str] = Field(
        default_factory=list,
        description="The functions of this map at the port, ``file::name``: the callers, the handler, the writers.",
    )
    system: str = Field(
        default="", description="A use: the far system as the map names it."
    )
    boundary: str = Field(
        default="",
        description="An offer: the inbound boundary the crossing arrives by, if any.",
    )
    values: list[str] = Field(
        default_factory=list,
        description="A use: the values written.  An offer: the values the column declares.",
    )
    access: str = Field(default="", description="A table use: ``read`` | ``written``.")
    senders: list[str] = Field(
        default_factory=list,
        description="A message offer: the sends of the type its own map already places at it, ``file::name:line``.",
    )


class Binding(BaseModel):
    """A use bound: the port of one map and the port of another (or, for a message, the same) it reaches."""

    kind: str
    key: str
    part: str = ""
    source_map: str
    source_node: str = Field(description="The boundary the crossing leaves by.")
    source_functions: list[str] = Field(default_factory=list)
    target_map: str
    target_node: str = Field(
        description="What is reached: an endpoint's or handler's function, an entity, a subject."
    )
    target_boundary: str = Field(
        default="",
        description="The inbound boundary it arrives by, where the map has one.",
    )
    target_functions: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class Unbound(BaseModel):
    """A use, or a part of one, that nothing among the maps given offers."""

    map_id: str
    node: str = Field(description="The boundary.")
    kind: str = ""
    key: str = ""
    part: str = ""
    reason: str


class Bindings(BaseModel):
    """What binding a set of maps found."""

    maps: dict[str, str] = Field(
        description="map id -> the version each map was derived from."
    )
    bound: list[Binding] = Field(default_factory=list)
    unbound: list[Unbound] = Field(default_factory=list)
    unreached: list[Unbound] = Field(
        default_factory=list,
        description=(
            "A message type a receiver takes that no function of the maps given sends.  Not a defect in itself "
            "-- a script, a timer or a system with no map may send it -- but it is where the maps stop being "
            "able to say why a handler ran."
        ),
    )

    def resolution(self, boundary: str) -> str:
        """``bound`` where every use of *boundary* is bound, ``partial`` where some is, else ``unresolved``."""
        some = any(b.source_node == boundary for b in self.bound)
        left = any(u.node == boundary for u in self.unbound)
        if some and not left:
            return "bound"
        return "partial" if some else "unresolved"


# --------------------------------------------------------------------------- reading a map's ports

Reader = Callable[[MapFragment], Iterable[Port]]
_READERS: list[Reader] = []


def register_reader(reader: Reader) -> Reader:
    """Add a reader of ports; every map is read by all of them."""
    _READERS.append(reader)
    return reader


def ports(fragment: MapFragment) -> list[Port]:
    """Everything *fragment* uses and offers."""
    return [port for reader in _READERS for port in reader(fragment)]


def table_key(name: str) -> tuple[str, str]:
    """``(table, schema)`` of ``"ATLAS_PANDA"."jobsActive4"``: lowered, unquoted, the schema apart."""
    schema, _, table = name.replace('"', "").replace("`", "").rpartition(".")
    return table.lower(), schema


def call_key(owner: str) -> str:
    """``pkg/mod.py::Class.method`` as an import spells it: ``pkg.mod.Class.method``."""
    file, _, name = owner.partition("::")
    module = file[:-3] if file.endswith(".py") else file
    module = module.removesuffix("/__init__").replace("/", ".")
    return f"{module}.{name}" if name else module


def _own(names: Iterable[str]) -> list[str]:
    return [name for name in names if "::" in name]


def _far(boundary: BoundaryNode) -> list[str]:
    return [
        name
        for name in [boundary.interface, *boundary.calls_out]
        if name and "::" not in name
    ]


def _functions(boundary: BoundaryNode) -> list[str]:
    """The functions of the map at an outbound boundary: those of its names spelled ``file::name``."""
    return _own([boundary.interface, *boundary.calls_out])


@register_reader
def _calls_out(fragment: MapFragment) -> Iterator[Port]:
    """``http`` and ``call`` uses: the far names of each outbound boundary that is neither a table nor a message."""
    for boundary in fragment.boundaries:
        if boundary.direction != "outbound" or boundary.transport in (
            "shared_table",
            "message",
        ):
            continue
        for name in _far(boundary):
            url = _URL.match(name)
            yield Port(
                map_id=fragment.map_id,
                role=USES,
                kind=HTTP if url else CALL,
                key=url.group(1).rstrip("/") if url else name,
                spelling=name,
                node=boundary.name,
                functions=_functions(boundary),
                system=boundary.system,
            )


@register_reader
def _endpoints(fragment: MapFragment) -> Iterator[Port]:
    """``http`` offers: each endpoint, with the inbound boundary of its function where there is one."""
    inbound = {
        boundary.interface: boundary.name
        for boundary in fragment.boundaries
        if boundary.direction == "inbound" and boundary.transport == "http"
    }
    for endpoint in fragment.endpoints:
        file, _, name = endpoint.function.partition("::")
        stem = file.rpartition("/")[2].removesuffix(".py")
        yield Port(
            map_id=fragment.map_id,
            role=OFFERS,
            kind=HTTP,
            key=f"/api/{endpoint.version}/{endpoint.route}",
            spelling=endpoint.route,
            node=endpoint.function,
            functions=[endpoint.function],
            boundary=inbound.get(f"{stem}::{name}", ""),
        )


def known_functions(fragment: MapFragment) -> set[str]:
    """Every function *fragment* names, ``file::name``."""
    found: set[str] = set()
    found.update(node.owner for node in fragment.reach_graph)
    found.update(junction.owner for junction in fragment.junctions)
    found.update(endpoint.function for endpoint in fragment.endpoints)
    for boundary in fragment.boundaries:
        found.update(_own([boundary.interface, *boundary.calls_out]))
    for receiver in fragment.message_receivers:
        found.update([receiver.receiver, *receiver.handled_by])
    for entity in fragment.entities:
        found.update(_own([*entity.created_by, *entity.updated_by, *entity.read_by]))
    return found


@register_reader
def _callables(fragment: MapFragment) -> Iterator[Port]:
    """``call`` offers: every function the map names, as an import spells it."""
    for owner in sorted(known_functions(fragment)):
        yield Port(
            map_id=fragment.map_id, role=OFFERS, kind=CALL, key=call_key(owner), spelling=owner, node=owner,
            functions=[owner],
        )  # fmt: skip


@register_reader
def _table_uses(fragment: MapFragment) -> Iterator[Port]:
    """``table`` uses: a ``shared_table`` boundary, and each column it reads or writes."""
    for boundary in fragment.boundaries:
        if boundary.transport != "shared_table":
            continue
        key, schema = table_key(boundary.interface)
        if "." in boundary.interface and not schema.isidentifier():
            key, schema = boundary.interface, ""  # a class path: ai.models.Proposal
        whole = Port(
            map_id=fragment.map_id, role=USES, kind=TABLE, key=key, spelling=boundary.interface, qualifier=schema,
            node=boundary.name, functions=_own(boundary.calls_out), system=boundary.system,
        )  # fmt: skip
        yield whole
        values = boundary.metadata.get("values")
        written: dict[str, list[str]] = values if isinstance(values, dict) else {}
        for access, columns in (
            ("read", boundary.carried_values),
            ("written", boundary.handed_over),
        ):
            for column in columns:
                yield whole.model_copy(
                    update={
                        "part": column.lower(),
                        "access": access,
                        "values": list(written.get(column, [])),
                    }
                )


@register_reader
def _table_offers(fragment: MapFragment) -> Iterator[Port]:
    """``table`` offers: an entity's tables, a table record's, and each subject as a column of its class's tables.

    A subject that says the module declaring its class (the ORM recognizers') is also offered under the class
    path, which is how code of another distribution names the table: by importing the model.
    """
    tables: dict[str, set[str]] = defaultdict(set)  # class -> its tables, as keys
    for entity in fragment.entities:
        offered: set[str] = set()
        for table in entity.tables:
            key, schema = table_key(table)
            if entity.spec_class:
                tables[entity.spec_class].add(key)
            if key in offered:
                continue  # one table the corpus spells two ways: jedi_tasks, JEDI_Tasks
            offered.add(key)
            yield Port(
                map_id=fragment.map_id, role=OFFERS, kind=TABLE, key=key, spelling=table, qualifier=schema,
                node=entity.name, functions=sorted({*entity.created_by, *entity.updated_by}),
            )  # fmt: skip
    entity_of = {
        entity.spec_class: entity.name
        for entity in fragment.entities
        if entity.spec_class
    }
    for record in fragment.tables:
        if record.rows_of and record.name not in tables[record.rows_of]:
            tables[record.rows_of].add(record.name)
            yield Port(
                map_id=fragment.map_id, role=OFFERS, kind=TABLE, key=record.name,
                spelling=(record.spellings or [record.name])[0], qualifier=",".join(record.schema_names),
                node=entity_of.get(record.rows_of, record.rows_of),
            )  # fmt: skip
    writers: dict[str, list[str]] = defaultdict(list)
    for junction in fragment.junctions:
        writers[junction.subject].append(junction.owner)
    paths: set[str] = set()
    for subject in fragment.subjects:
        keys = set(tables.get(subject.spec_class, ()))
        declared = subject.metadata.get("table")
        if isinstance(declared, str) and declared:
            keys.add(table_key(declared)[0])
        module = subject.metadata.get("module")
        path = (
            f"{module}.{subject.spec_class}"
            if isinstance(module, str) and module
            else ""
        )
        for key in sorted(keys) + ([path] if path else []):
            if key == path and path not in paths:
                paths.add(path)
                yield Port(
                    map_id=fragment.map_id, role=OFFERS, kind=TABLE, key=path, spelling=path,
                    node=entity_of.get(subject.spec_class, subject.spec_class),
                )  # fmt: skip
            yield Port(
                map_id=fragment.map_id, role=OFFERS, kind=TABLE, key=key, part=subject.attribute.lower(),
                spelling=subject.name, node=subject.name, functions=sorted(set(writers.get(subject.name, []))),
                values=list(subject.vocabulary),
            )  # fmt: skip


@register_reader
def _message_uses(fragment: MapFragment) -> Iterator[Port]:
    """``message`` uses: per outbound ``message`` boundary, each type sent to it and the functions sending it.

    The sends are the message edges placed at the boundary's call sites; where the map has none, the types are
    the boundary's ``handed_over`` and any of its functions may send any of them.  A function that sends and
    writes no type is a use with an empty key: its caller builds the message."""
    for boundary in fragment.boundaries:
        if boundary.transport != "message" or boundary.direction != "outbound":
            continue
        sites = set(boundary.call_sites)
        senders: dict[str, list[str]] = defaultdict(list)
        for edge in fragment.message_edges:
            if (
                f"{edge.sender.partition('::')[0]}:{edge.line}" in sites
                and edge.sender not in senders[edge.msg_type]
            ):
                senders[edge.msg_type].append(edge.sender)
        own = _own(boundary.calls_out)
        if not senders:
            senders = defaultdict(list, dict.fromkeys(boundary.handed_over, own))
        typed = {sender for of_type in senders.values() for sender in of_type}
        if any(sender not in typed for sender in own) or not senders:
            senders[""] = [sender for sender in own if sender not in typed]
        for msg_type, functions in sorted(senders.items()):
            yield Port(
                map_id=fragment.map_id, role=USES, kind=MESSAGE, key=msg_type, spelling=msg_type,
                qualifier=boundary.interface, node=boundary.name, functions=functions, system=boundary.system,
            )  # fmt: skip


@register_reader
def _message_offers(fragment: MapFragment) -> Iterator[Port]:
    """``message`` offers: each type a receiver takes, at its handler, with the destination its boundary names.

    A boundary named by a function and not a destination states no queue: PanDA's receivers are plugins the
    deployment's configuration puts on queues."""
    queues = {
        boundary.name: boundary.interface
        for boundary in fragment.boundaries
        if boundary.transport == "message"
        and boundary.direction == "inbound"
        and "::" not in boundary.interface
    }
    for receiver in fragment.message_receivers:
        if not receiver.msg_type:
            continue
        yield Port(
            map_id=fragment.map_id, role=OFFERS, kind=MESSAGE, key=receiver.msg_type, spelling=receiver.msg_type,
            qualifier=queues.get(receiver.boundary, ""), node=(receiver.handled_by or [receiver.receiver])[0],
            functions=receiver.handled_by or [receiver.receiver], boundary=receiver.boundary,
            senders=[*receiver.senders, *receiver.senders_open],
        )  # fmt: skip


# --------------------------------------------------------------------------- matching

#: ``(use, the offers under its kind and key in every map) -> (the offers it binds to, a note per offer)``.
Matcher = Callable[[Port, list[Port]], list[tuple[Port, list[str]]]]
_MATCHERS: dict[str, Matcher] = {}


def register_matcher(kind: str) -> Callable[[Matcher], Matcher]:
    """Set how a use of *kind* chooses among the offers under its key; without one, every offer of another map."""

    def register(matcher: Matcher) -> Matcher:
        _MATCHERS[kind] = matcher
        return matcher

    return register


def _other_maps(use: Port, offers: list[Port]) -> list[tuple[Port, list[str]]]:
    return [(offer, []) for offer in offers if offer.map_id != use.map_id]


@register_matcher(TABLE)
def _same_column(use: Port, offers: list[Port]) -> list[tuple[Port, list[str]]]:
    """A table to the entity holding its rows, a column to its subject; a value the subject does not declare and
    a schema spelled otherwise are noted."""
    found: list[tuple[Port, list[str]]] = []
    for offer in offers:
        if offer.map_id == use.map_id or offer.part != use.part:
            continue
        notes = [
            f"writes '{value}', which {offer.node} does not declare"
            for value in use.values
            if offer.values
            and not value.startswith(_RUNTIME)
            and value not in offer.values
        ]
        if (
            not use.part
            and use.qualifier
            and offer.qualifier
            and use.qualifier.lower() not in offer.qualifier.lower()
        ):
            notes.append(
                f"schema {use.qualifier} here, {offer.qualifier} there: the deployment's"
            )
        found.append((offer, notes))
    return found


@register_matcher(MESSAGE)
def _same_destination(use: Port, offers: list[Port]) -> list[tuple[Port, list[str]]]:
    """A type to its handler where the destinations agree; to a receiver that states none, with a note.  The
    receiver may be the sender's own map: a message goes to another run, not another distribution.
    """
    found: list[tuple[Port, list[str]]] = []
    for offer in offers:
        if offer.qualifier == use.qualifier:
            found.append((offer, []))
        elif not offer.qualifier:
            found.append(
                (
                    offer,
                    [
                        f"the receiver names no queue; whether it listens to {use.qualifier} is the deployment's"
                    ],
                )
            )
    return found


def _covering(
    use: Port, fragments: list[MapFragment], offered: dict[tuple[str, str], list[Port]]
) -> Optional[str]:
    """The map that covers what *use* names though it offers no such node, if one does: for a call, the map
    holding the package called; else the map of the system the use names."""
    if use.kind == CALL:
        package = use.key.split(".")[0] + "."
        for (kind, key), there in offered.items():
            if (
                kind == CALL
                and key.startswith(package)
                and there[0].map_id != use.map_id
            ):
                return there[0].map_id
    for fragment in fragments:
        if fragment.map_id == use.system and (
            fragment.map_id != use.map_id or use.kind == MESSAGE
        ):
            return fragment.map_id
    return None


def bind(fragments: list[MapFragment]) -> Bindings:
    """What the uses of *fragments* reach among their offers."""
    found = Bindings(
        maps={fragment.map_id: fragment.derived_from for fragment in fragments}
    )
    uses: list[Port] = []
    offered: dict[tuple[str, str], list[Port]] = defaultdict(list)
    for fragment in fragments:
        for port in ports(fragment):
            if port.role == USES:
                uses.append(port)
            else:
                offered[(port.kind, port.key)].append(port)

    reached: set[tuple[str, str, str]] = set()
    tables_bound: set[tuple[str, str]] = set()
    for use in uses:
        if use.kind == MESSAGE and not use.key:
            found.unbound.append(
                Unbound(
                    map_id=use.map_id, node=use.node, kind=use.kind, part=", ".join(use.functions),
                    reason="sends a message whose type the sending function does not write itself",
                )  # fmt: skip
            )
            continue
        offers = offered.get((use.kind, use.key), [])
        matched = _MATCHERS.get(use.kind, _other_maps)(use, offers)
        for offer, notes in matched:
            reached.add((offer.map_id, offer.node, offer.key))
            if use.kind == TABLE and not use.part:
                tables_bound.add((use.node, use.key))
            found.bound.append(
                Binding(
                    kind=use.kind, key=use.key, part=use.part, source_map=use.map_id, source_node=use.node,
                    source_functions=use.functions, target_map=offer.map_id, target_node=offer.node,
                    target_boundary=offer.boundary, target_functions=offer.functions,
                    notes=([f"{use.access} here"] if use.access else []) + notes,
                )  # fmt: skip
            )
        if matched:
            continue
        if use.kind == TABLE and use.part:
            continue  # a column no subject stands for: said once on the table's binding, below
        covering = _covering(use, fragments, offered)
        if use.kind == MESSAGE and any(
            offer.qualifier == use.qualifier
            for group in offered.values()
            for offer in group
            if offer.kind == MESSAGE
        ):
            reason = "its receiver has no handler for this type"
        elif use.kind == MESSAGE:
            reason = f"no map given receives from {use.qualifier}"
        elif covering is not None:
            reason = f"the map of {covering} has no node for it"
        else:
            reason = f"no map given covers {use.system or use.key.split('.')[0]}"
        found.unbound.append(
            Unbound(
                map_id=use.map_id,
                node=use.node,
                kind=use.kind,
                key=use.spelling or use.key,
                reason=reason,
            )
        )

    # a table bound whose columns have no subject: the columns are named on the table's binding
    loose: dict[tuple[str, str], list[str]] = defaultdict(list)
    column_bound = {
        (b.source_node, b.key, b.part)
        for b in found.bound
        if b.kind == TABLE and b.part
    }
    for use in uses:
        if use.kind == TABLE and use.part and (use.node, use.key) in tables_bound:
            if (
                use.node,
                use.key,
                use.part,
            ) not in column_bound and use.part not in loose[(use.node, use.key)]:
                loose[(use.node, use.key)].append(use.part)
    for binding in found.bound:
        if (
            binding.kind == TABLE
            and not binding.part
            and loose.get((binding.source_node, binding.key))
        ):
            columns = loose[(binding.source_node, binding.key)]
            binding.notes.append(
                f"{len(columns)} column(s) named here are no subject there: {', '.join(sorted(columns))}"
            )

    for (kind, _key), group in sorted(offered.items()):
        if kind != MESSAGE:
            continue
        for offer in group:
            if (
                offer.map_id,
                offer.node,
                offer.key,
            ) not in reached and not offer.senders:
                found.unreached.append(
                    Unbound(
                        map_id=offer.map_id, node=offer.boundary or offer.node, kind=MESSAGE, key=offer.key,
                        reason="no function of the maps given sends this type",
                    )  # fmt: skip
                )
    found.unbound.sort(key=lambda u: (u.map_id, u.node, u.key, u.part))
    found.unreached.sort(key=lambda u: (u.map_id, u.node, u.key))
    return found
