"""The API's endpoints, and the statements reading rows each one's calls reach (:class:`~bamboo.codemap.models.Endpoint`).

The live-case harness's EP (to-map-33, 10-28; to-map-44, 10-31): a reader asking PanDA for an object's rows through
its API, rather than the database, needs per endpoint the rows it reads -- their tables, the terms naming them, and
what of the request fills those.

An endpoint is a function the server runs for a URL ``/api/<version>/<module>/<method>``.  The server's routing table
names the module for each ``<module>`` -- ``module_mapping`` in :data:`SERVER`, a literal mapping per version, its
``"module"`` a name the server imports (``from pandaserver.api.v1 import credential_management_api as cred_api_v1``
under ``"creds"``) -- and every public function defined in that module is a method, but ``init_task_buffer``
(``extract_allowed_methods``): undecorated ones too, ``workflow/submit_workflow_raw_request``.  The legacy
``/server/panda/<method>``, run from the server's own globals, is not read.

From each, the reach record's closure (:meth:`~bamboo.codemap.reach.ReachGraph.closure`, as a mover's reach walks
it), and each read site whose statement a function of it holds -- placed by its anchor, so a method's owner spelled
with or without its class does not matter -- with its read terms (:attr:`KeyFrom.reads`).  A statement past a hole of
the closure is not there: :attr:`Endpoint.holes` says where one may be.
"""

from __future__ import annotations

import ast
import logging
from typing import TYPE_CHECKING, Iterator, Optional

from bamboo.codemap import callfacts, reach
from bamboo.codemap import calls as C
from bamboo.codemap.models import (
    BIND_ALIAS,
    BIND_ELEMENT,
    BIND_FILL,
    BIND_RESULT,
    FILL_CONSTANT,
    FILL_CUT,
    FILL_NOT_SAID,
    FILL_OTHER,
    FILL_RESULT,
    FILL_SELF,
    FILL_TERM,
    FILL_UNREAD,
    FILL_VALUE,
    HOLE_CLOSED,
    HOLE_MAY_CHANGE,
    HOLE_OPEN,
    KEY_CONSTANT,
    KEY_PARAMETER,
    KEY_ROWS,
    KEY_SELF_FIELD,
    Endpoint,
    EndpointChange,
    EndpointFill,
    EndpointHole,
    EndpointRead,
    EndpointTerm,
    EndpointWrite,
    KeyFrom,
    MapFragment,
    ReachCall,
    ReachHole,
    ReadSiteNode,
    SourceModule,
)
from bamboo.codemap.panda import objects, pathcond, sql, sqltext

if TYPE_CHECKING:
    from bamboo.codemap.panda.recognizers import sqlwrite

logger = logging.getLogger(__name__)

#: The file holding the server's routing table, and the function holding it.
SERVER = "pandaserver/server/panda.py"
ROUTING = "module_mapping"
#: A module's public functions the server does not route (``extract_allowed_methods``).
NOT_ROUTED = frozenset({"init_task_buffer"})
#: The file holding the async requests' daemon's table of handlers, and the table: a module's name bound to a dict of
#: request types to functions.  :attr:`Endpoint.version` of a handler's record, and its route's head.
HANDLING = "pandaserver/asyncprocess/processor.py"
HANDLERS = "HANDLERS"
ASYNC = "async"


def routes(modules: list[SourceModule]) -> list[tuple[str, str, str]]:
    """``(version, route, owner)`` per endpoint, ``route`` ``<module>/<method>``; none where :data:`SERVER`'s
    routing table is not read."""
    by_path = {module.rel_path: module for module in modules}
    server = by_path.get(SERVER)
    if server is None:
        return []
    imported = _imported(server.tree)
    found = []
    for version, name, held in _table(server.tree):
        module = by_path.get(imported.get(held, ""))
        if module is None:
            continue
        for node in module.tree.body:
            if (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and not node.name.startswith("_")
                and node.name not in NOT_ROUTED
            ):
                found.append(
                    (version, f"{name}/{node.name}", f"{module.rel_path}::{node.name}")
                )
    return list(dict.fromkeys(found))


def _imported(tree: ast.Module) -> dict[str, str]:
    """``{name: file}`` for each module the file imports by name: ``from a.b import c as d`` gives ``d: a/b/c.py``."""
    found = {}
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module and not node.level:
            for alias in node.names:
                found[alias.asname or alias.name] = (
                    f"{node.module}.{alias.name}".replace(".", "/") + ".py"
                )
    return found


def _table(tree: ast.Module) -> Iterator[tuple[str, str, str]]:
    """``(version, module name, the name its module is held by)`` per entry of :data:`ROUTING`'s mapping: the dict
    literal of versions, each a dict literal of module names, each a dict literal whose ``"module"`` is a name."""
    for node in tree.body:
        if not (isinstance(node, ast.FunctionDef) and node.name == ROUTING):
            continue
        for held in ast.walk(node):
            if not isinstance(held, ast.Dict):
                continue
            for version, entries in zip(held.keys, held.values, strict=True):
                if not (_text(version) and isinstance(entries, ast.Dict)):
                    continue
                for name, entry in zip(entries.keys, entries.values, strict=True):
                    module = _value(entry, "module") if _text(name) else None
                    if isinstance(module, ast.Name):
                        yield version.value, name.value, module.id  # type: ignore[union-attr]


def _text(node: Optional[ast.expr]) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, str)


def _value(node: ast.expr, key: str) -> Optional[ast.expr]:
    if isinstance(node, ast.Dict):
        for one, value in zip(node.keys, node.values, strict=True):
            if isinstance(one, ast.Constant) and one.value == key:
                return value
    return None


def attach(
    fragment: MapFragment,
    modules: list[SourceModule],
    calls: C.Calls,
    returned: Optional[dict[str, sqlwrite.Returned]] = None,
) -> int:
    """Set *fragment*'s :attr:`MapFragment.endpoints` and :attr:`MapFragment.handlers`; returns how many statements
    the endpoints read, counted per endpoint.

    With *returned* -- :func:`~bamboo.codemap.panda.recognizers.sqlwrite.attach_crossings`' walks asked for every
    endpoint's answer -- each read says where its rows are in what the endpoint returns, and the holes on the way in
    the endpoint's closure (:attr:`EndpointRead.returned`, :attr:`EndpointRead.returned_open`)."""
    record = reach.ReachGraph(fragment.reach_graph)
    sites: dict[str, list[ReadSiteNode]] = {}
    for site in fragment.read_sites:
        if site.anchor is None:
            continue
        holder = record.function_at(site.anchor.file, site.anchor.line_start)
        if holder is not None:
            sites.setdefault(holder, []).append(site)
    terms: dict[tuple[str, int], list[KeyFrom]] = {}
    for key_from in fragment.key_froms:
        if key_from.reads:
            terms.setdefault((key_from.owner, key_from.statement), []).append(key_from)
    code = _Code(calls)
    read = _Reader(record, code, sites, terms, _Bounds(record, code))
    fragment.endpoints = [
        one
        for version, route, owner in routes(modules)
        if (one := read.endpoint(version, route, owner, returned)) is not None
    ]
    fragment.handlers = [
        one
        for request_type, owner in handlers(modules)
        if (one := read.endpoint(ASYNC, f"{ASYNC}/{request_type}", owner)) is not None
    ]
    return sum(len(endpoint.reads) for endpoint in fragment.endpoints)


class _Reader:
    """An :class:`Endpoint` from its function: the closure and what it reads, writes, changes and leaves open."""

    def __init__(
        self,
        record: reach.ReachGraph,
        code: _Code,
        sites: dict[str, list[ReadSiteNode]],
        terms: dict[tuple[str, int], list[KeyFrom]],
        bounds: _Bounds,
    ):
        self.record, self.code, self.sites, self.terms, self.bounds = (
            record,
            code,
            sites,
            terms,
            bounds,
        )

    def endpoint(
        self,
        version: str,
        route: str,
        owner: str,
        returned: Optional[dict[str, sqlwrite.Returned]] = None,
    ) -> Optional[Endpoint]:
        record, code = self.record, self.code
        if owner not in record.rows:
            logger.warning(
                "endpoints: %s's function %s is no row of the reach record",
                route,
                owner,
            )
            return None
        came = record.closure([owner])
        shut = _shut(record, code, came)
        if shut:
            came = record.closure(
                [owner], cut=lambda node, call, shut=shut: (node, id(call)) in shut
            )
        holes = {}
        reached: dict[str, tuple[ReadSiteNode, list[tuple[str, str]]]] = {}
        for node in came:
            function, cls = node
            for hole in record.rows[function].holes:
                if not hole.for_classes or cls in hole.for_classes:
                    holes.setdefault((function, hole.line, hole.callee), hole)
            for site in self.sites.get(function, ()):
                reached.setdefault(site.name, (site, []))[1].append(node)
        fills = _Fills(record, code, came, owner, shut)
        inside = {function for function, _cls in came}
        reads = []
        for site, nodes in reached.values():
            answer = returned.get(site.name) if returned is not None else None
            reads.append(
                EndpointRead(
                    site=site.name,
                    function=nodes[0][0],
                    tables=sorted({table.lower() for table in site.tables}),
                    terms=[
                        EndpointTerm(
                            term=key_from.term,
                            table=key_from.table,
                            attr=key_from.attr,
                            fills=fills.term(nodes, key_from, set(reached)),
                        )
                        for key_from in self.terms.get((site.owner, _line(site)), ())
                    ],
                    way=reach.way(came, nodes[0]),
                    returned=answer.paths.get(owner, []) if answer is not None else [],
                    returned_open=sorted(
                        {
                            f"{function}:{line}"
                            for function, line, _callee in answer.holes
                            if function in inside
                        }
                    )
                    if answer is not None
                    else [],
                )
            )
        functions = list(dict.fromkeys(function for function, _cls in came))
        return Endpoint(
            route=route,
            version=version,
            function=owner,
            closure=len(functions),
            holes=list(
                dict.fromkeys(f"{function}:{line}" for function, line, _callee in holes)
            ),
            shut=sorted(
                {
                    f"{node[0].partition('::')[0]}:{line}"
                    for node, call in _calls_shut(record, came, shut)
                    for line in call.lines or [call.line]
                }
            ),
            reads=reads,
            writes=[one for function in functions for one in code.writes(function)],
            changes=[one for function in functions for one in code.changes(function)],
            hole_bounds=[
                self.bounds.of(function, hole)
                for (function, _line, _callee), hole in holes.items()
            ],
        )


Node = tuple[str, str]


class _Code:
    """The functions of the corpus by their owners, and what is read off each once: its names' versions, the calls
    on each of its lines, where each lands."""

    def __init__(self, calls: C.Calls):
        self.calls = calls
        self.types = calls.types
        self.names = reach._Names(calls.types)
        self.funcs = {owner: (file, func) for file, func, owner in self.names.functions}
        self._versions: dict[str, objects._Versions] = {}
        self._lines: dict[str, dict[int, list[ast.AST]]] = {}
        self._writes: dict[str, list[EndpointWrite]] = {}
        self._changes: dict[str, list[EndpointChange]] = {}
        self._bound: dict[
            tuple[str, int, str], list[tuple[ast.Call, dict[str, Optional[ast.expr]]]]
        ] = {}

    def writes(self, owner: str) -> list[EndpointWrite]:
        """The statements writing rows *owner* runs, one per statement and kind."""
        found = self._writes.get(owner)
        if found is None:
            found = self._writes[owner] = []
            held = self.funcs.get(owner)
            for run in sql.executions(held[1]) if held is not None else []:  # type: ignore[arg-type]
                for kind, read in _WRITES:
                    tables = sorted({table.lower() for table in read(run.reading)})
                    one = EndpointWrite(
                        function=owner, line=run.call.lineno, kind=kind, tables=tables
                    )
                    if tables and one not in found:
                        found.append(one)
        return found

    def changes(self, owner: str) -> list[EndpointChange]:
        """The calls *owner* makes that change something beside PanDA's tables (:func:`callfacts.change`)."""
        found = self._changes.get(owner)
        if found is None:
            found = self._changes[owner] = []
            file, func = self.funcs.get(owner, ("", None))
            for node in C.own_nodes(func) if func is not None else []:
                got = (
                    callfacts.change(self.types, file, func, node)
                    if isinstance(node, ast.Call)
                    else None
                )
                if got is not None:
                    kind, system, callee = got
                    found.append(
                        EndpointChange(
                            kind=kind,
                            system=system,
                            callee=callee,
                            function=owner,
                            line=node.lineno,
                        )
                    )
        return found

    def versions(self, owner: str) -> objects._Versions:
        found = self._versions.get(owner)
        if found is None:
            func = self.funcs[owner][1]
            found = self._versions[owner] = objects._Versions(
                func, objects._local_names(func)
            )
        return found

    def at(self, owner: str, line: int) -> list[ast.AST]:
        """The nodes of *owner*'s own body on *line*, in the order a walk meets them."""
        lines = self._lines.get(owner)
        if lines is None:
            lines = self._lines[owner] = {}
            for node in C.own_nodes(self.funcs[owner][1]):
                lines.setdefault(getattr(node, "lineno", 0), []).append(node)
        return lines.get(line, [])

    def bound(
        self, caller: str, call: ReachCall, callee: str
    ) -> list[tuple[ast.Call, dict[str, Optional[ast.expr]]]]:
        """Each call of *call*'s lines in *caller* landing in *callee*, with what it hands each parameter
        (:func:`calls.bind`: ``None`` where it does not say)."""
        key = (caller, id(call), callee)
        found = self._bound.get(key)
        if found is not None:
            return found
        file, func = self.funcs[caller]
        found = self._bound[key] = []
        for line in call.lines or [call.line]:
            for node in self.at(caller, line):
                if not isinstance(node, ast.Call):
                    continue
                try:
                    placed = self.calls.callees(file, func, node)
                except Exception:  # noqa: BLE001 -- the reach record has the hole
                    continue
                for target in placed.targets:
                    if self.names.of(target.func) == callee:
                        found.append((node, C.bind(node, target, placed.on_class)))
        return found

    def handed(
        self, caller: str, call: ReachCall, callee: str, parameter: str
    ) -> list[Optional[ast.expr]]:
        """What each call of *call*'s lines in *caller* landing in *callee* hands its *parameter*; ``None`` where the
        call does not say."""
        return [
            given.get(parameter) for _node, given in self.bound(caller, call, callee)
        ]


class _Fills:
    """What fills each term of the statements one endpoint reads, followed up its closure's calls to its own
    parameters (:class:`EndpointFill`)."""

    def __init__(
        self,
        record: reach.ReachGraph,
        code: _Code,
        came: reach.Came,
        endpoint: str,
        shut: frozenset[tuple[Node, int]] = frozenset(),
    ):
        self.record, self.code, self.endpoint = record, code, endpoint
        self.into = _into(record, came, shut)
        # Each call between functions of the closure, ``file:line``: a way rows of a statement can come by here.
        self.calls = {
            f"{caller[0].partition('::')[0]}:{line}"
            for edges in self.into.values()
            for caller, call in edges
            for line in call.lines or [call.line]
        }
        self._read: dict[tuple[Node, str, str], list[EndpointFill]] = {}
        self._reading: set[tuple[Node, str, str]] = set()

    def term(
        self, nodes: list[Node], key_from: KeyFrom, reads: set[str]
    ) -> list[EndpointFill]:
        found: list[EndpointFill] = []
        for source in key_from.sources:
            if source.kind == KEY_PARAMETER:
                for node in nodes:
                    found.extend(
                        one.model_copy(update={"attr": source.attr})
                        for one in self.parameter(node, source.name, "", 0)
                    )
            elif source.kind == KEY_ROWS:
                # Rows this endpoint reads, come by its own calls: not rows a call elsewhere hands down.
                if source.name in reads and all(
                    one in self.calls for one in source.via
                ):
                    found.append(EndpointFill(rows=source.name, attr=source.attr))
            elif source.kind == KEY_CONSTANT:
                found.append(EndpointFill(why=FILL_CONSTANT))
            elif source.kind == KEY_SELF_FIELD:
                found.append(EndpointFill(why=FILL_SELF))
            else:
                found.append(EndpointFill(why=FILL_OTHER))
        if not key_from.sources:
            found.append(EndpointFill(why=FILL_TERM))
        unique = {one.model_dump_json(): one for one in found}
        return [unique[key] for key in sorted(unique)]

    def parameter(
        self, node: Node, name: str, steps: str, depth: int
    ) -> list[EndpointFill]:
        """What fills *name*, a parameter of *node*'s function, at *steps* inside it: the endpoint's own, or what each
        call into the function hands it."""
        if node[0] == self.endpoint:
            return [EndpointFill(parameter=name + steps)]
        key = (node, name, steps)
        held = self._read.get(key)
        if held is not None:
            return held
        self._read[key] = []  # a way round a loop of calls adds nothing
        found: list[EndpointFill] = []
        for caller, call in self.into.get(node, ()):
            at = f"{caller[0].partition('::')[0]}:{call.line}"
            if depth >= MAX_STEPS:
                found.append(EndpointFill(why=FILL_CUT, at=at))
                continue
            if call.how != "call":
                found.append(EndpointFill(why=FILL_NOT_SAID, at=at))
                continue
            for value in self.code.handed(caller[0], call, node[0], name) or [None]:
                if value is None:
                    found.append(EndpointFill(why=FILL_NOT_SAID, at=at))
                else:
                    found.extend(self.value(caller, value, steps, depth + 1))
        self._read[key] = found
        return found

    def value(
        self, node: Node, value: ast.expr, steps: str, depth: int
    ) -> list[EndpointFill]:
        """What fills *steps* of *value*, an expression of *node*'s function."""
        owner = node[0]
        file, func = self.code.funcs[owner]
        at = f"{file}:{getattr(value, 'lineno', 0)}"
        value = callfacts.unconverted(self.code.types, file, func, value)
        if isinstance(value, ast.Constant):
            return [EndpointFill(why=FILL_CONSTANT, at=at)]
        if isinstance(value, (ast.List, ast.Tuple, ast.Set)):
            if not value.elts:
                return []  # nothing in it to fill a term with
            if steps.startswith(ELEMENT):
                return [
                    one
                    for element in value.elts
                    for one in self.value(node, element, steps[len(ELEMENT) :], depth)
                ]
            return [EndpointFill(why=FILL_VALUE, at=at)]
        if isinstance(value, ast.BinOp) and isinstance(value.op, ast.Add):
            if steps.startswith(ELEMENT):
                # a + b holds the elements of a and of b
                return [
                    one
                    for side in (value.left, value.right)
                    for one in self.value(node, side, steps, depth)
                ]
            return [EndpointFill(why=FILL_VALUE, at=at)]
        if isinstance(value, ast.Call):
            kept = self._kept(file, func, value)
            if kept is not None:
                return self.value(node, kept, steps, depth)
            try:
                placed = self.code.calls.callees(file, func, value).targets
            except Exception:  # noqa: BLE001 -- the reach record has the hole
                placed = []
            if placed:
                return [EndpointFill(why=FILL_RESULT, at=at)]
            return [EndpointFill(why=FILL_VALUE, at=at)]
        paths = self.code.versions(owner).paths(value)
        if not paths:
            return [EndpointFill(why=FILL_VALUE, at=at)]
        return [one for path in paths for one in self.path(node, path, steps, depth)]

    def _kept(self, file: str, func: ast.AST, call: ast.Call) -> Optional[ast.expr]:
        """The argument whose elements *call*'s value holds, one by one, where it is a builtin keeping them:
        ``list(xs)``, ``dict.fromkeys(xs)`` (:data:`callfacts.KEYED_BY_ELEMENTS`)."""
        if len(call.args) != 1 or isinstance(call.args[0], ast.Starred):
            return None
        function = callfacts.builtin_function(self.code.types, file, func, call)
        if function is not None:
            return call.args[0] if function.gives == callfacts.COPY else None
        called = call.func
        if (
            isinstance(called, ast.Attribute)
            and isinstance(called.value, ast.Name)
            and (called.value.id, called.attr) in callfacts.KEYED_BY_ELEMENTS
            and self.code.types.is_builtin(file, func, called.value.id)
        ):
            return call.args[0]
        return None

    def path(self, node: Node, path: str, steps: str, depth: int) -> list[EndpointFill]:
        """What fills *steps* of *path*, a path of *node*'s function's names (``jobID@1082``, ``jobIDs[]``)."""
        key = (node, path, steps)
        if key in self._reading:
            return []  # a binding read again inside its own reading adds nothing
        self._reading.add(key)
        try:
            return self._path(node, path, steps, depth)
        finally:
            self._reading.discard(key)

    def _path(
        self, node: Node, path: str, steps: str, depth: int
    ) -> list[EndpointFill]:
        owner = node[0]
        file, func = self.code.funcs[owner]
        root, rest = _split(path)
        steps = rest.replace("[]", ELEMENT) + steps
        if root in ("self", "cls"):
            return [EndpointFill(why=FILL_SELF, at=f"{file}:{func.lineno}")]  # type: ignore[attr-defined]
        if "@" not in root and root in _parameters(func):
            return self.parameter(node, root, steps, depth)
        found: list[EndpointFill] = []
        bound = False
        for binding in self.record.rows[owner].bindings:
            at = f"{file}:{binding.line}"
            if binding.name == root:
                bound = True
                if binding.how in (BIND_ALIAS, BIND_ELEMENT) and binding.path:
                    found.extend(self.path(node, binding.path, steps, depth))
                elif binding.how == BIND_RESULT and not binding.to:
                    found.extend(
                        self._statement(node, root, binding.line, steps, depth)
                    )
                elif binding.how == BIND_RESULT:
                    found.append(EndpointFill(why=FILL_RESULT, at=at))
            elif (
                binding.name == root + "[]"
                and binding.how == BIND_FILL
                and binding.path
            ):
                bound = True
                if steps.startswith(ELEMENT):
                    found.extend(
                        self.path(node, binding.path, steps[len(ELEMENT) :], depth)
                    )
                else:
                    found.append(EndpointFill(why=FILL_VALUE, at=at))
        if not bound:
            name, _sep, line = root.partition("@")
            found.extend(
                self._statement(
                    node, name if line else root, int(line) if line else 0, steps, depth
                )
            )
        return found

    def _statement(
        self, node: Node, name: str, line: int, steps: str, depth: int
    ) -> list[EndpointFill]:
        """What fills *steps* of what *name* is bound to by the assignment on *line* (by its only assignment, for 0)
        of *node*'s function: where no binding of the record says it, or its value is a builtin's."""
        owner = node[0]
        file, func = self.code.funcs[owner]
        name = name.partition("@")[0]
        values = [
            one.value
            for one in C.own_nodes(func)
            if isinstance(one, (ast.Assign, ast.AnnAssign))
            and one.value is not None
            and (not line or one.lineno == line)
            and any(
                isinstance(target, ast.Name) and target.id == name
                for target in (
                    one.targets if isinstance(one, ast.Assign) else [one.target]
                )
            )
        ]
        if len(values) != 1:
            return [EndpointFill(why=FILL_UNREAD, at=f"{file}:{line or func.lineno}")]  # type: ignore[attr-defined]
        return self.value(node, values[0], steps, depth)


def _into(
    record: reach.ReachGraph,
    came: reach.Came,
    shut: frozenset[tuple[Node, int]] = frozenset(),
) -> dict[Node, list[tuple[Node, ReachCall]]]:
    """Each function of the closure *came* as a class, with every call into it from one of the closure, as the closure
    walks them; not one of *shut*."""
    found: dict[Node, list[tuple[Node, ReachCall]]] = {}
    for node in came:
        owner, cls = node
        for call in record.rows[owner].calls:
            if call.for_classes and cls not in call.for_classes:
                continue
            if (node, id(call)) in shut:
                continue
            entered = record.rows.get(call.to)
            if entered is None:
                continue
            for one in call.runs_as or entered.classes or [""]:
                if (call.to, one) in came:
                    found.setdefault((call.to, one), []).append((node, call))
    return found


def _calls_shut(
    record: reach.ReachGraph, came: reach.Came, shut: frozenset[tuple[Node, int]]
) -> Iterator[tuple[Node, ReachCall]]:
    """Each call of *shut* a function of the closure *came* makes."""
    for node in came:
        for call in record.rows[node[0]].calls:
            if (node, id(call)) in shut:
                yield node, call


def _shut(
    record: reach.ReachGraph, code: _Code, came: reach.Came
) -> frozenset[tuple[Node, int]]:
    """The calls of the closure *came* no way into it makes: ``(the function and class making it, id(its call))``.

    A parameter of a function of the closure holds a constant where every call into the function, on the ways still
    open, hands it the same one -- a literal, the default the call leaves it to (:func:`calls.bind`), a parameter of
    the caller holding one --; the endpoint's own parameters are the request's and hold none.  A call is shut where
    every call of its lines is ruled out (:func:`pathcond.ruled_out_at`) by such a parameter of the function making
    it: ``get_task_details_json`` calls ``_resolve_parent_task_id`` under ``if resolve_parent:``, and
    ``task/get_detailed_info``'s way leaves ``resolve_parent`` to its default, ``False``.  Read again without the
    calls shut, until no more is.
    """
    starts = [node for node, came_from in came.items() if came_from is None]
    shut: set[tuple[Node, int]] = set()
    while True:
        into = _into(record, came, frozenset(shut))
        open_ = _reached(starts, into)
        held: dict[Node, dict[str, tuple[type, object]]] = {node: {} for node in starts}
        changed = True
        while changed:
            changed = False
            for node in open_:
                if node in starts:
                    continue
                found = _constants(
                    code,
                    node,
                    [one for one in into.get(node, ()) if one[0] in open_],
                    held,
                )
                if found != held.get(node):
                    held[node] = found
                    changed = True
        more = set()
        for callee, edges in into.items():
            for caller, call in edges:
                values = held.get(caller)
                if caller not in open_ or not values or call.how != "call":
                    continue
                runs = code.bound(caller[0], call, callee[0])
                func = code.funcs[caller[0]][1]
                if runs and all(
                    any(
                        pathcond.ruled_out_at(node, func, name, value)
                        for name, (_kind, value) in values.items()
                    )
                    for node, _given in runs
                ):
                    more.add((caller, id(call)))
        if more <= shut:
            return frozenset(shut)
        shut |= more


def _reached(
    starts: list[Node], into: dict[Node, list[tuple[Node, ReachCall]]]
) -> set[Node]:
    """The nodes *starts* reach by the calls of *into*."""
    out: dict[Node, list[Node]] = {}
    for callee, edges in into.items():
        for caller, _call in edges:
            out.setdefault(caller, []).append(callee)
    found = set(starts)
    pending = list(starts)
    while pending:
        for callee in out.get(pending.pop(), ()):
            if callee not in found:
                found.add(callee)
                pending.append(callee)
    return found


def _constants(
    code: _Code,
    node: Node,
    edges: list[tuple[Node, ReachCall]],
    held: dict[Node, dict[str, tuple[type, object]]],
) -> dict[str, tuple[type, object]]:
    """``{parameter: (type, value)}`` of *node*'s function for each parameter every call of *edges* hands the same
    constant."""
    func = code.funcs[node[0]][1]
    found: dict[str, tuple[type, object]] = {}
    for name in sorted(_parameters(func) - {"self", "cls"}):
        values = []
        for caller, call in edges:
            runs = code.bound(caller[0], call, node[0]) if call.how == "call" else []
            if not runs:
                values.append(None)
            for _node, given in runs:
                values.append(_constant(code, caller, given.get(name), held))
        if (
            values
            and all(one is not None for one in values)
            and len(set(map(repr, values))) == 1
        ):
            found[name] = values[0]  # type: ignore[assignment]
    return found


def _constant(
    code: _Code,
    caller: Node,
    value: Optional[ast.expr],
    held: dict[Node, dict[str, tuple[type, object]]],
) -> Optional[tuple[type, object]]:
    """``(type, value)`` of the constant *value*, an argument *caller* hands, holds; ``None`` where it holds none."""
    if isinstance(value, ast.Constant):
        return type(value.value), value.value
    if isinstance(value, ast.Name) and code.versions(caller[0]).paths(value) == [
        value.id
    ]:
        return held.get(caller, {}).get(value.id)
    return None


class _Bounds:
    """Each hole bounded by its callee's name (:class:`EndpointHole`), by the reach record's rule for a mover's
    (:class:`reach._ByName`): a function of the corpus of the name may write a table or change something
    (:meth:`_Code.writes`, :meth:`_Code.changes`), or its closure does, or leaves a hole no name bounds, or makes a
    call whose name may -- the least fixpoint over names."""

    def __init__(self, record: reach.ReachGraph, code: _Code):
        self.record = record
        changing = [
            owner for owner in record.rows if code.writes(owner) or code.changes(owner)
        ]
        self.destination = reach.Destination({owner: [] for owner in changing}, [])
        self.names = reach._ByName(record, self.destination)
        self._through: dict[str, list[str]] = {}

    def of(self, function: str, hole: ReachHole) -> EndpointHole:
        at = f"{function}:{hole.line}"
        name = reach._named_call(f"{at} {hole.callee} -- {hole.why}")
        through: list[str] = []
        if name is None:
            bound = HOLE_OPEN
        elif self.names.can(name):
            bound, through = HOLE_MAY_CHANGE, self.through(name)
        else:
            bound = HOLE_CLOSED
        return EndpointHole(
            at=at, callee=hole.callee, why=hole.why, bound=bound, through=through
        )

    def through(self, name: str) -> list[str]:
        """The functions named *name* that may: each whose closure reaches one that writes or changes, leaves a hole
        no name bounds, or makes a call whose name may."""
        found = self._through.get(name)
        if found is None:
            found = self._through[name] = []
            for owner in self.record.functions_named(name):
                answer = self.record.reach([owner], self.destination, by_name=False)
                inner = {reach._named_call(hole) for hole in answer.holes}
                if (
                    answer.writes
                    or None in inner
                    or any(self.names.can(one) for one in inner if one is not None)
                ):
                    found.append(owner)
        return found


def handlers(modules: list[SourceModule]) -> list[tuple[str, str]]:
    """``(request type, owner)`` per entry of the async requests' daemon's table (:data:`HANDLING`'s
    :data:`HANDLERS`); none where it is not read.

    A dict literal, its keys a text, a module's name bound to one or a text made of a comprehension's names, its
    values a function of the module or ``functools.partial`` of one, and ``**`` another module's table of the same
    name -- or a dict comprehension over a module's tuple of texts.  What the reading cannot say is logged and
    left out."""
    by_path = {module.rel_path: module for module in modules}
    found: list[tuple[str, str]] = []
    _handlers_in(by_path, HANDLING, found, set())
    return list(dict.fromkeys(found))


def _handlers_in(
    by_path: dict[str, SourceModule],
    file: str,
    found: list[tuple[str, str]],
    seen: set[str],
) -> None:
    module = by_path.get(file)
    if module is None or file in seen:
        return
    seen.add(file)
    table = _bound_value(module.tree, HANDLERS)
    imported = _imported(module.tree)
    if isinstance(table, ast.Dict):
        for key, value in zip(table.keys, table.values, strict=True):
            if key is not None:
                _handler(module, imported, key, value, {}, found)
            elif (
                isinstance(value, ast.Attribute)
                and value.attr == HANDLERS
                and isinstance(value.value, ast.Name)
                and value.value.id in imported
            ):
                _handlers_in(by_path, imported[value.value.id], found, seen)
            else:
                logger.warning(
                    "endpoints: %s's %s spreads %s", file, HANDLERS, ast.unparse(value)
                )
    elif (
        isinstance(table, ast.DictComp)
        and len(table.generators) == 1
        and isinstance(table.generators[0].target, ast.Name)
        and not table.generators[0].ifs
    ):
        loop = table.generators[0]
        for one in _texts(module.tree, loop.iter):
            _handler(
                module, imported, table.key, table.value, {loop.target.id: one}, found
            )  # type: ignore[union-attr]
    else:
        logger.warning("endpoints: %s's %s is not read", file, HANDLERS)


def _handler(
    module: SourceModule,
    imported: dict[str, str],
    key: ast.expr,
    value: ast.expr,
    names: dict[str, str],
    found: list[tuple[str, str]],
) -> None:
    request_type = _text_of(module.tree, key, names)
    owner = _function_of(module, imported, value)
    if request_type is None or owner is None:
        logger.warning(
            "endpoints: %s's handler %s: %s is not read",
            module.rel_path,
            ast.unparse(key),
            ast.unparse(value),
        )
        return
    found.append((request_type, owner))


def _bound_value(tree: ast.Module, name: str) -> Optional[ast.expr]:
    """The value the module's own body last binds *name* to."""
    value = None
    for node in tree.body:
        targets = (
            node.targets
            if isinstance(node, ast.Assign)
            else [node.target]
            if isinstance(node, ast.AnnAssign)
            else []
        )
        if any(isinstance(one, ast.Name) and one.id == name for one in targets):
            value = node.value  # type: ignore[union-attr]
    return value


def _text_of(tree: ast.Module, node: ast.expr, names: dict[str, str]) -> Optional[str]:
    """The text *node* is: a literal, a name the comprehension or the module binds to one, an f-string of them."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        if node.id in names:
            return names[node.id]
        value = _bound_value(tree, node.id)
        return (
            value.value
            if isinstance(value, ast.Constant) and isinstance(value.value, str)
            else None
        )
    if isinstance(node, ast.JoinedStr):
        parts = []
        for part in node.values:
            if isinstance(part, ast.FormattedValue):
                if part.conversion != -1 or part.format_spec is not None:
                    return None
                part = part.value
            text = _text_of(tree, part, names)
            if text is None:
                return None
            parts.append(text)
        return "".join(parts)
    return None


def _texts(tree: ast.Module, node: ast.expr) -> list[str]:
    """The texts of a tuple or a list *node* is, or the module binds its name to."""
    if isinstance(node, ast.Name):
        node = _bound_value(tree, node.id)  # type: ignore[assignment]
    if not isinstance(node, (ast.Tuple, ast.List)):
        return []
    return [
        one.value
        for one in node.elts
        if isinstance(one, ast.Constant) and isinstance(one.value, str)
    ]


def _function_of(
    module: SourceModule, imported: dict[str, str], node: ast.expr
) -> Optional[str]:
    """The owner of the function *node* names: the module's own, a module's it imports, or ``functools.partial``'s
    first argument."""
    if isinstance(node, ast.Call) and node.args:
        called = node.func
        if (isinstance(called, ast.Name) and called.id == "partial") or (
            isinstance(called, ast.Attribute)
            and called.attr == "partial"
            and isinstance(called.value, ast.Name)
            and called.value.id == "functools"
        ):
            return _function_of(module, imported, node.args[0])
        return None
    if isinstance(node, ast.Name) and any(
        isinstance(one, (ast.FunctionDef, ast.AsyncFunctionDef)) and one.name == node.id
        for one in module.tree.body
    ):
        return f"{module.rel_path}::{node.id}"
    if (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id in imported
    ):
        return f"{imported[node.value.id]}::{node.attr}"
    return None


#: The kinds of statement writing rows, each with what reads its tables off its text.
_WRITES = (
    ("insert", sqltext.creates),
    ("update", sqltext.updates),
    ("delete", sqltext.deletes),
    ("merge", sqltext.merges),
)
#: How an element of a value is spelt in :attr:`EndpointFill.parameter`, as :attr:`KeySource.paths` spells it.
ELEMENT = "[*]"
#: How many calls up a fill is followed.
MAX_STEPS = 12


def _line(site: ReadSiteNode) -> int:
    return site.anchor.line_start  # type: ignore[union-attr] -- only sites with an anchor are placed


def _split(path: str) -> tuple[str, str]:
    """A path's root name and the rest: ``jobIDs[]`` is ``jobIDs``, ``[]``."""
    for index, char in enumerate(path):
        if char in ".[":
            return path[:index], path[index:]
    return path, ""


def _parameters(func: ast.AST) -> set[str]:
    args = func.args  # type: ignore[attr-defined]
    return {
        one.arg
        for one in [
            *args.posonlyargs,
            *args.args,
            *args.kwonlyargs,
            args.vararg,
            args.kwarg,
        ]
        if one is not None
    }
