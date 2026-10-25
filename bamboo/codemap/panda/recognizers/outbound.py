"""Where PanDA calls out to another system, and what it asks that system to do.

The other direction from :mod:`boundary`, which reads only the receiving side
by design: an endpoint another system calls, a table another system fills.
What PanDA *sends* -- a dataset registered in Rucio, a replication rule, an
erasure -- left no node, so a setupper that returned without registering a
``_sub`` dataset and one that registered it looked the same in the map.

One node per method of an interface class (:data:`provenance.INTERFACE_CLASSES`)
that calls an object of a class from the package the system is reached through
(:data:`provenance.OUTSIDE_PACKAGES`).  The object is found by what it is
declared to be -- ``client = self._get_rucio_client()`` returns a
``RucioClient``, imported from ``rucio`` -- not by its name.  Each node carries:

- ``calls_out``: the outside methods it calls, the far side's own API;
- ``verbs``: what those calls do, **read from the outside method's name**
  (``add_*_to_*`` attaches, ``add_*``/``create_*`` creates, ``set_*`` changes,
  ``delete_*``/``erase_*``/``detach_*`` removes, ``list_*``/``get_*`` asks).
  The name is the far side's and not PanDA's, and it is still a name: stated
  here so that a reader weighs it as one;
- ``call_sites``: the calls in PanDA that :meth:`Calls.callers` places at the
  method.
"""

from __future__ import annotations

import ast
from typing import Optional

from bamboo.codemap.calls import Calls, Target, own_nodes
from bamboo.codemap.models import Anchor, BoundaryNode, SourceModule
from bamboo.codemap.panda.provenance import INTERFACE_CLASSES, OUTSIDE_PACKAGES

#: ``(prefix, verb)`` in the order they are tried: ``add_files_to_dataset``
#: attaches before ``add_`` would call it a creation.
_VERBS = (
    ("attach", "attaches"),
    ("add_files_to", "attaches"),
    ("add_datasets_to", "attaches"),
    ("add_replicas", "creates"),
    ("add_", "creates"),
    ("create", "creates"),
    ("register", "creates"),
    ("set_", "changes"),
    ("update", "changes"),
    ("delete", "removes"),
    ("erase", "removes"),
    ("detach", "removes"),
    ("list_", "asks"),
    ("get_", "asks"),
)
#: What changes the far side, as opposed to asking it.
_CAUSING = frozenset({"attaches", "creates", "changes", "removes"})


def verb_of(method: str) -> str:
    """What an outside method does, by its name -- see the module docstring."""
    for prefix, verb in _VERBS:
        if method.startswith(prefix):
            return verb
    return "other"


def _outside_class(calls: Calls, file: str, func: ast.AST, receiver: ast.expr) -> Optional[str]:
    """The outside class *receiver* is declared to be, as ``module.Name``, if it is one."""
    types = calls.types
    annotation: Optional[ast.expr] = None
    if isinstance(receiver, ast.Name):
        for argument in getattr(func.args, "args", []) + getattr(func.args, "kwonlyargs", []):  # type: ignore[attr-defined]
            if argument.arg == receiver.id and argument.annotation is not None:
                annotation = argument.annotation
        for node in own_nodes(func):
            if annotation is not None:
                break
            if not isinstance(node, (ast.Assign, ast.AnnAssign)) or node.value is None:
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if not any(isinstance(target, ast.Name) and target.id == receiver.id for target in targets):
                continue
            if isinstance(node, ast.AnnAssign):
                annotation = node.annotation
            elif isinstance(node.value, ast.Call):
                called = node.value.func
                if isinstance(called, ast.Name) and types.imported_from_outside(file, called.id):
                    annotation = called
                else:
                    # A call placed open may land somewhere returning another
                    # class: its targets' annotation is not the receiver's.
                    placed = calls.callees(file, func, node.value)
                    for target in placed.targets if not placed.unplaced else ():
                        annotation = getattr(target.func, "returns", None)
                        if annotation is not None:
                            file = target.file
                            break
    if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
        try:
            annotation = ast.parse(annotation.value, mode="eval").body
        except SyntaxError:
            return None
    if not isinstance(annotation, ast.Name) or not types.imported_from_outside(file, annotation.id):
        return None
    module, attribute = types.imports(file)[annotation.id]
    return f"{module}.{attribute or annotation.id}"


def extract(
    modules: list[SourceModule], calls: Calls, map_id: str, derived_from: str
) -> list[BoundaryNode]:
    """One outbound boundary per interface method that calls another system."""
    by_file = {module.rel_path: module for module in modules}
    nodes: list[BoundaryNode] = []
    for key, system in sorted(INTERFACE_CLASSES.items()):
        file, _, class_name = key.partition("::")
        tree = calls.types.tree(file)
        cls = next(
            (node for node in (tree.body if tree is not None else []) if isinstance(node, ast.ClassDef) and node.name == class_name),
            None,
        )
        packages = OUTSIDE_PACKAGES.get(system, ())
        if cls is None or not packages:
            continue
        for method in cls.body:
            if not isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            outside: set[str] = set()
            outside_calls: set[str] = set()
            for node in own_nodes(method):
                if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                    continue
                held = _outside_class(calls, file, method, node.func.value)
                if held is not None and held.split(".")[0] in packages:
                    outside.add(node.func.attr)
                    outside_calls.add(f"{held}.{node.func.attr}")
            if not outside:
                continue
            verbs = sorted({verb_of(name) for name in outside})
            sites = sorted({calls.hop(site) for site in calls.callers(Target(file, method, cls))})
            interface = f"{file}::{class_name}.{method.name}"
            module = by_file.get(file)
            nodes.append(
                BoundaryNode(
                    map_id=map_id,
                    derived_from=derived_from,
                    name=BoundaryNode.make_name(map_id, system, interface),
                    system=system,
                    kind="transports_causation" if _CAUSING & set(verbs) else "reports_state",
                    transport="http",
                    interface=interface,
                    direction="outbound",
                    verbs=verbs,
                    calls_out=sorted(outside_calls),
                    call_sites=sites,
                    anchor=Anchor(
                        package=file.partition("/")[0],
                        file=file,
                        line_start=method.lineno,
                        line_end=method.end_lineno or method.lineno,
                        blob_sha=module.blob_sha if module is not None else "",
                    ),
                )
            )
    return nodes
