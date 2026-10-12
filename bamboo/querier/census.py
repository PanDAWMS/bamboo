"""Outside part 4, from a guard's atom to the calls that could have set what it reads.

A guard that reads a value through a method -- ``not self.add_result.is_succeeded()``
at ``adder_gen.py:440`` -- has no assignment for a walk to reach: the value is
set by other methods of the same object, called from anywhere.  One rule, and
nothing about the adder in it:

1. the class is the one class in the corpus that defines the method; more than
   one and this stops rather than guess;
2. the attributes the method reads on ``self``, and the methods that assign
   them (the mutators), each with the value it assigns where that is a
   constant of the class;
3. every attribute bound to ``CLASS()`` anywhere in the corpus (the holders),
   and every call ``self.<holder>.<mutator>()``;
4. a run-time dispatch the map records (``Reading.dispatch``) is settled by the
   run's own line that announces the class, and calls in the other
   alternatives are dropped;
5. **the value match**: a printed hole spelling ``<x>.<attribute>`` in the run,
   after every record of the calls' functions and before the guard's own row,
   is the value the guard read; a mutator that assigns another value did not
   make it.
"""
from __future__ import annotations

import ast
import functools
import re
from dataclasses import dataclass, field
from pathlib import Path

from . import rows as R
from .records import CORPUS, Run

PACKAGES = ("pandaserver", "pandajedi")


@functools.cache
def corpus_files() -> tuple[str, ...]:
    return tuple(
        str(p.relative_to(CORPUS))
        for package in PACKAGES
        for p in sorted((CORPUS / package).rglob("*.py"))
    )


@functools.cache
def parsed(rel: str) -> ast.Module | None:
    try:
        return R.tree(rel)
    except (SyntaxError, UnicodeDecodeError):
        return None


def _self_attrs(node: ast.AST, ctx: type) -> set[str]:
    return {
        n.attr
        for n in ast.walk(node)
        if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == "self" and isinstance(n.ctx, ctx)
    }


@dataclass
class Site:
    rel: str
    func: str
    cls: str
    line: int
    mutator: str
    assigned: object
    kept: bool = True
    why: str = ""


@dataclass
class Census:
    method: str
    cls: str = ""
    cls_file: str = ""
    reads: list[str] = field(default_factory=list)
    mutators: dict[str, object] = field(default_factory=dict)
    holders: list[str] = field(default_factory=list)
    sites: list[Site] = field(default_factory=list)
    dispatch: str = ""
    printed: tuple[str, str, str] | None = None  # (attribute, value, where)
    stop: str = ""

    def kept(self) -> list[Site]:
        return [s for s in self.sites if s.kept]

    def report(self) -> str:
        out = [f"method {self.method}: class {self.cls or '?'} ({self.cls_file}) reads {self.reads}; mutators {self.mutators}"]
        if self.stop:
            out.append(f"STOP: {self.stop}")
        out.append(f"holders {self.holders}; {len(self.sites)} call(s)")
        if self.dispatch:
            out.append(f"dispatch: {self.dispatch}")
        if self.printed:
            out.append(f"value match: {self.printed[0]} printed {self.printed[1]!r} at {self.printed[2]}")
        for s in self.sites:
            out.append(f"  {'kept   ' if s.kept else 'dropped'} {Path(s.rel).name}:{s.line} {s.cls}.{s.func} {s.mutator}() assigns {s.assigned!r}  {s.why}")
        return "\n".join(out)


def _class_constant(cls: ast.ClassDef, name: str) -> object:
    for node in cls.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            try:
                return ast.literal_eval(node.value)
            except ValueError:
                return None
    return None


def census(method: str) -> Census:
    out = Census(method)
    owners = []
    for rel in corpus_files():
        tree = parsed(rel)
        if tree is None:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and any(
                isinstance(m, ast.FunctionDef) and m.name == method for m in node.body
            ):
                owners.append((rel, node))
    if len(owners) != 1:
        out.stop = f"{len(owners)} classes define {method}: {[f'{r}:{c.name}' for r, c in owners]}"
        return out
    rel, cls = owners[0]
    out.cls, out.cls_file = cls.name, rel
    methods = {m.name: m for m in cls.body if isinstance(m, ast.FunctionDef)}
    reads = _self_attrs(methods[method], ast.Load)
    out.reads = sorted(reads)
    for name, fn in methods.items():
        written = _self_attrs(fn, ast.Store) & reads
        if not written:
            continue
        assigned: object = None
        for node in ast.walk(fn):
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Attribute) and t.attr in written for t in node.targets
            ):
                v = node.value
                if isinstance(v, ast.Attribute) and isinstance(v.value, ast.Name) and v.value.id == "self":
                    assigned = _class_constant(cls, v.attr)
                else:
                    try:
                        assigned = ast.literal_eval(v)
                    except ValueError:
                        assigned = ast.unparse(v)
        out.mutators[name] = assigned
    holders = set()
    for rel2 in corpus_files():
        tree = parsed(rel2)
        if tree is None:
            continue
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Assign)
                and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name)
                and node.value.func.id == cls.name
            ):
                for t in node.targets:
                    if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) and t.value.id == "self":
                        holders.add(t.attr)
    out.holders = sorted(holders)
    for rel2 in corpus_files():
        tree = parsed(rel2)
        if tree is None:
            continue
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in out.mutators
                and node.func.attr != "__init__"
                and isinstance(node.func.value, ast.Attribute)
                and node.func.value.attr in holders
                and isinstance(node.func.value.value, ast.Name)
                and node.func.value.value.id == "self"
            ):
                continue
            fn = R.pathcond.enclosing_function(node)
            owner = next((a.name for a in R.pathcond.ancestors(node) if isinstance(a, ast.ClassDef)), "")
            out.sites.append(Site(rel2, fn.name if fn else "<module>", owner, node.lineno, node.func.attr, out.mutators[node.func.attr]))
    out.sites.sort(key=lambda s: (s.rel, s.line))
    return out


def settle_dispatch(c: Census, dispatch: list, run: Run) -> None:
    """Drop the calls in the classes a recorded dispatch did not choose."""
    for fan in dispatch:
        if not fan.announced_by or not fan.candidates:
            continue
        pattern = re.compile(re.escape(fan.announced_by) + r"\W*(\w+)")
        chosen = next((m[1] for r in run.records if (m := pattern.search(r.message)) and m[1] in fan.candidates), None)
        if chosen is None:
            continue
        c.dispatch = f"{fan.selector or 'dispatch'} chose {chosen} (the run printed {fan.announced_by!r}); alternatives {fan.candidates}"
        bases = {chosen, fan.base} if fan.base else {chosen}
        for s in c.sites:
            if s.cls in fan.candidates and s.cls not in bases:
                s.kept, s.why = False, f"in {s.cls}, which the dispatch did not choose"


def match_value(c: Census, run: Run, mode: str, before_record: int) -> None:
    """The value match (rule 5 of the docstring)."""
    kept = c.kept()
    last_site_record = -1
    # one def per site's line: a name two defs of a file share is read in the def the site is in
    for rel, func, at in {(s.rel, s.func, R.def_at(s.rel, s.func, [s.line]).lineno) for s in kept}:
        sn = R.seen(R.function(rel, func, within=[at]), run, mode)
        for hits in sn.hits.values():
            last_site_record = max([last_site_record, *(i for i, _, _ in hits)])
    hole = re.compile(rf"\.({'|'.join(map(re.escape, c.reads))})$")
    found = []
    for rel in corpus_files():
        tree = parsed(rel)
        if tree is None:
            continue
        for fn_node, _ in R.pathcond.functions_with_owner(tree):
            if not any(
                isinstance(n, ast.FormattedValue) and hole.search(ast.unparse(n.value)) for n in R.own_nodes(fn_node)
            ):
                continue
            fn = R.function(rel, fn_node.name, within=[fn_node.lineno])
            sn = R.seen(fn, run, mode)
            for row in fn.rows:
                for i, _, _ in sn.hits.get(row.line, ()):
                    for expr, value in row.capture(run.records[i]).items():
                        attr = expr.rsplit(".", 1)[-1]
                        if attr in c.reads and last_site_record < i < before_record:
                            found.append((i, attr, value.strip(), f"{Path(rel).name}:{row.line} at {run.records[i].span}"))
    if not found:
        return
    i, attr, value, where = max(found)
    c.printed = (attr, value, where)
    try:
        printed = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        printed = value
    for s in c.kept():
        if s.assigned != printed:
            s.kept, s.why = False, f"assigns {s.assigned!r}, the run printed {attr}={value!r} ({where})"
