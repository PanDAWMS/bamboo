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
from typing import Iterator, Optional

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
    KEY_CONSTANT,
    KEY_PARAMETER,
    KEY_ROWS,
    KEY_SELF_FIELD,
    Endpoint,
    EndpointFill,
    EndpointRead,
    EndpointTerm,
    KeyFrom,
    MapFragment,
    ReachCall,
    ReadSiteNode,
    SourceModule,
)
from bamboo.codemap.panda import objects

logger = logging.getLogger(__name__)

#: The file holding the server's routing table, and the function holding it.
SERVER = "pandaserver/server/panda.py"
ROUTING = "module_mapping"
#: A module's public functions the server does not route (``extract_allowed_methods``).
NOT_ROUTED = frozenset({"init_task_buffer"})


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


def attach(fragment: MapFragment, modules: list[SourceModule], calls: C.Calls) -> int:
    """Set *fragment*'s :attr:`MapFragment.endpoints`; returns how many statements they read, counted per endpoint."""
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
    found = []
    for version, route, owner in routes(modules):
        if owner not in record.rows:
            logger.warning(
                "endpoints: %s's function %s is no row of the reach record",
                route,
                owner,
            )
            continue
        came = record.closure([owner])
        holes = []
        reached: dict[str, tuple[ReadSiteNode, list[tuple[str, str]]]] = {}
        for node in came:
            function, cls = node
            holes.extend(
                f"{function}:{hole.line}"
                for hole in record.rows[function].holes
                if not hole.for_classes or cls in hole.for_classes
            )
            for site in sites.get(function, ()):
                reached.setdefault(site.name, (site, []))[1].append(node)
        fills = _Fills(record, code, came, owner)
        reads = []
        for site, nodes in reached.values():
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
                        for key_from in terms.get((site.owner, _line(site)), ())
                    ],
                    way=reach.way(came, nodes[0]),
                )
            )
        found.append(
            Endpoint(
                route=route,
                version=version,
                function=owner,
                closure=len({function for function, _cls in came}),
                holes=list(dict.fromkeys(holes)),
                reads=reads,
            )
        )
    fragment.endpoints = found
    return sum(len(endpoint.reads) for endpoint in found)


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

    def handed(
        self, caller: str, call: ReachCall, callee: str, parameter: str
    ) -> list[Optional[ast.expr]]:
        """What each call of *call*'s lines in *caller* landing in *callee* hands its *parameter*; ``None`` where the
        call does not say."""
        file, func = self.funcs[caller]
        found: list[Optional[ast.expr]] = []
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
                        found.append(
                            C.bind(node, target, placed.on_class).get(parameter)
                        )
        return found


class _Fills:
    """What fills each term of the statements one endpoint reads, followed up its closure's calls to its own
    parameters (:class:`EndpointFill`)."""

    def __init__(
        self, record: reach.ReachGraph, code: _Code, came: reach.Came, endpoint: str
    ):
        self.record, self.code, self.endpoint = record, code, endpoint
        self.into: dict[Node, list[tuple[Node, ReachCall]]] = {}
        for node in came:
            owner, cls = node
            for call in record.rows[owner].calls:
                if call.for_classes and cls not in call.for_classes:
                    continue
                entered = record.rows.get(call.to)
                if entered is None:
                    continue
                for one in call.runs_as or entered.classes or [""]:
                    if (call.to, one) in came:
                        self.into.setdefault((call.to, one), []).append((node, call))
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
