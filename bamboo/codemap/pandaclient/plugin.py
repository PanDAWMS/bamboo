"""panda-client Code Map plugin: a client library's functions and the requests each one makes.

Code that asks PanDA for something rarely writes the request itself: it calls
``pandaclient.Client.submitJobs`` and the library makes it.  A map of the
calling code can say only that it calls the library; PanDA's map says only
which function runs for a URL.  The library is a distribution with a version
of its own between the two, so it is a map of its own, and a small one: per
function, the requests it makes.

panda-client declares them.  ``@http_request_decorator(endpoint="job/kill",
method="post")`` on a function says it sends that verb to
``<server_base_path_ssl>/job/kill``, and ``server_base_path_ssl`` is built
ending ``/api/v1``.  The public functions mostly wrap a decorated one
(``getJobStatus`` calls ``getJobStatus_internal``), so a function's requests
are those of every decorated function its calls inside the package reach.

One outbound ``http`` boundary per function that makes a request, named by the
function, ``calls_out`` the requests as ``POST /api/v1/job/kill``.  That is
all the map holds; :mod:`bamboo.codemap.bind` binds a caller's
``pandaclient.Client.submitJobs`` to the function and each request to the
server's endpoint.

Not read: the requests written out by hand rather than declared (the legacy
``/server/panda/<method>`` calls through ``_Curl``), and calls placed through
a name the code builds.
"""

from __future__ import annotations

import ast
import logging
import re
from importlib.metadata import PackageNotFoundError, version
from importlib.util import find_spec
from pathlib import Path
from typing import Optional

from bamboo.codemap.base import CodeMapPlugin
from bamboo.codemap.gitsource import blob_sha, describe
from bamboo.codemap.models import Anchor, BoundaryNode, MapFragment

logger = logging.getLogger(__name__)

MAP_ID = "panda-client"
PACKAGE = "pandaclient"
DISTRIBUTION = "panda-client"
#: The system the requests go to.
SERVER = "panda"
#: The decorator that declares a function's request.
DECORATOR = "http_request_decorator"
#: The path the library's base URL ends with where the code does not say: ``<scheme>://<host>/api/v1``.
DEFAULT_BASE = "/api/v1"

_BASE = re.compile(r"(/api/v\d+)/?$")


def _declared(func: ast.AST) -> Optional[tuple[str, str]]:
    """``(VERB, endpoint)`` of the request *func*'s decorator declares, if it declares one."""
    for decorator in getattr(func, "decorator_list", []):
        if not isinstance(decorator, ast.Call):
            continue
        name = decorator.func
        if (
            name.id if isinstance(name, ast.Name) else getattr(name, "attr", "")
        ) != DECORATOR:
            continue
        given = {kw.arg: kw.value for kw in decorator.keywords if kw.arg}
        endpoint = given.get("endpoint", decorator.args[0] if decorator.args else None)
        method = given.get(
            "method", decorator.args[1] if len(decorator.args) > 1 else None
        )
        if isinstance(endpoint, ast.Constant) and isinstance(endpoint.value, str):
            verb = (
                method.value
                if isinstance(method, ast.Constant) and isinstance(method.value, str)
                else "post"
            )
            return verb.upper(), endpoint.value.strip("/")
    return None


def _base_path(tree: ast.Module) -> Optional[str]:
    """The path ``server_base_path*`` is built ending with, ``/api/v1``, where the module builds one."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(t, ast.Name) and t.id.startswith("server_base_path")
            for t in node.targets
        ):
            continue
        value = node.value
        text = (
            "".join(str(v.value) for v in value.values if isinstance(v, ast.Constant))
            if isinstance(value, ast.JoinedStr)
            else (
                value.value
                if isinstance(value, ast.Constant) and isinstance(value.value, str)
                else ""
            )
        )
        found = _BASE.search(text)
        if found:
            return found.group(1)
    return None


class PandaClientCodeMapPlugin(CodeMapPlugin):
    """Builds the panda-client fragment from a checkout or the installed distribution."""

    def __init__(self) -> None:
        self._package: Optional[Path] = None
        self._stamp = ""

    @property
    def map_id(self) -> str:
        return MAP_ID

    def prepare(self, source_root=None, schema_root=None, instance_record=None) -> str:
        if source_root is not None:
            package = Path(source_root).resolve() / PACKAGE
            if not package.is_dir():
                raise FileNotFoundError(
                    f"{Path(source_root).resolve()} holds no {PACKAGE}/"
                )
            stamp = describe(package.parent) or "unversioned"
        else:
            spec = find_spec(PACKAGE)
            if spec is None or not spec.submodule_search_locations:
                raise FileNotFoundError(
                    f"{DISTRIBUTION} is not installed: pass --source-root <checkout>"
                )
            package = Path(next(iter(spec.submodule_search_locations)))
            try:
                stamp = version(DISTRIBUTION)
            except PackageNotFoundError:
                stamp = "unversioned"
        self._package = package
        self._stamp = f"{DISTRIBUTION} {stamp}"
        return self._stamp

    def run(self) -> MapFragment:
        if self._package is None:
            raise RuntimeError("prepare() must be called before run()")
        modules: dict[str, tuple[str, ast.Module, str]] = (
            {}
        )  # module name -> (file, tree, sha)
        for path in sorted(self._package.glob("*.py")):
            text = path.read_text(encoding="utf-8", errors="replace")
            try:
                tree = ast.parse(text, filename=path.name)
            except SyntaxError as exc:
                logger.warning("skipping %s: %s", path.name, exc)
                continue
            modules[path.stem] = (f"{PACKAGE}/{path.name}", tree, blob_sha(text))

        base = next(
            (
                found
                for _file, tree, _sha in modules.values()
                if (found := _base_path(tree)) is not None
            ),
            DEFAULT_BASE,
        )
        functions: dict[tuple[str, str], ast.AST] = {}
        for module, (_file, tree, _sha) in modules.items():
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    functions[(module, node.name)] = node
        requests = {
            key: request
            for key, func in functions.items()
            if (request := _declared(func)) is not None
        }

        def callees(module: str, func: ast.AST) -> set[tuple[str, str]]:
            """The package's functions *func* calls by name: ``f(..)`` of its module, ``Client.f(..)`` of another."""
            found: set[tuple[str, str]] = set()
            for node in ast.walk(func):
                if not isinstance(node, ast.Call):
                    continue
                called = node.func
                if isinstance(called, ast.Name):
                    key = (module, called.id)
                elif isinstance(called, ast.Attribute) and isinstance(
                    called.value, ast.Name
                ):
                    key = (called.value.id, called.attr)
                else:
                    continue
                if key in functions:
                    found.add(key)
            return found

        calls = {key: callees(key[0], func) for key, func in functions.items()}
        boundaries: list[BoundaryNode] = []
        for key in sorted(functions):
            reached, todo = {key}, [key]
            while todo:
                for callee in calls[todo.pop()]:
                    if callee not in reached:
                        reached.add(callee)
                        todo.append(callee)
            made = sorted(
                {requests[one] for one in reached if one in requests},
                key=lambda r: (r[1], r[0]),
            )
            if not made:
                continue
            module, name = key
            file, _tree, sha = modules[module]
            func = functions[key]
            interface = f"{file}::{name}"
            boundaries.append(
                BoundaryNode(
                    name=BoundaryNode.make_name(MAP_ID, SERVER, interface),
                    map_id=MAP_ID,
                    derived_from=self._stamp,
                    system=SERVER,
                    kind="transports_causation",
                    transport="http",
                    interface=interface,
                    direction="outbound",
                    verbs=sorted({verb for verb, _endpoint in made}),
                    calls_out=[f"{verb} {base}/{endpoint}" for verb, endpoint in made],
                    anchor=Anchor(
                        package=PACKAGE,
                        file=file,
                        line_start=func.lineno,  # type: ignore[attr-defined]
                        line_end=getattr(func, "end_lineno", None),
                        blob_sha=sha,
                    ),
                )
            )
        return MapFragment(
            map_id=MAP_ID, derived_from=self._stamp, boundaries=boundaries
        )
