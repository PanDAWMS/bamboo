"""Outside part 6: every call through one outbound client, grouped by what it does to the object.

The verb is read off the method name and nothing else (make / attach / remove /
ask / other), so the grouping is a question for the reader, not a verdict.
The client is found from the module a traceback's boundary frame is in: the
module-level instance that module builds (``rucioAPI = RucioAPI()`` in
``ddm.py``) is the name the rest of the corpus calls it by.
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass

from . import rows as R
from .census import corpus_files, parsed

VERBS = [
    # ``register_dataset`` alone: the ``_location`` and ``_subscription`` calls attach to a dataset that exists
    ("make", r"^(register_dataset$|register_container|create|make|new)"),
    ("attach", r"^(register_files|register_zip|add_|attach|register_dataset_location|register_dataset_subscription)"),
    ("remove", r"^(erase|delete|remove|close|detach|set_metadata.*expir)"),
    ("ask", r"^(get|list|check|is_|has_|exists|find)"),
]


@dataclass
class Call:
    rel: str
    func: str
    line: int
    method: str
    verb: str
    first_arg: str


def instances_in(rel: str) -> list[str]:
    """Module-level names *rel* binds to an instance of a class it defines."""
    tree = parsed(rel)
    if tree is None:
        return []
    classes = {n.name for n in tree.body if isinstance(n, ast.ClassDef)}
    return [
        t.id
        for n in tree.body
        if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name) and n.value.func.id in classes
        for t in n.targets
        if isinstance(t, ast.Name)
    ]


def calls_through(receiver: str) -> list[Call]:
    out = []
    for rel in corpus_files():
        tree = parsed(rel)
        if tree is None:
            continue
        for fn, _ in R.pathcond.functions_with_owner(tree):
            for n in R.own_nodes(fn):
                if (
                    isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Attribute)
                    and ast.unparse(n.func.value).split(".")[-1] == receiver
                ):
                    verb = next((v for v, rx in VERBS if re.search(rx, n.func.attr)), "other")
                    arg = ast.unparse(n.args[0]) if n.args else ""
                    out.append(Call(rel, fn.name, n.lineno, n.func.attr, verb, arg))
    return sorted(out, key=lambda c: (c.rel, c.line))
