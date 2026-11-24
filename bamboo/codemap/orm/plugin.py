"""The walk a Code Map plugin for a Django code base makes, over the shared ORM recognizers.

:class:`DjangoCodeMapPlugin` reads the packages a subclass names, takes the
models, write forms, values and conditions from :mod:`bamboo.codemap.orm`, and
assembles the fragment.  A subclass states

* its map id (:attr:`MAP_ID`) and where each package lies in a checkout
  (:attr:`PACKAGES`),
* the code bases installed beside it (:attr:`NEIGHBOURS`) and the database
  schemas other systems own (:attr:`SCHEMAS`),
* its own boundaries: :meth:`module_context` reads what a module says once
  (its imports, its constants), :meth:`visit_node` is handed every node of
  every function with that, :meth:`visit_module` the module itself, and
  :meth:`finish` runs once every module is read.  :meth:`boundary` records a
  crossing.

What the walk itself reads of raw SQL and of rows other systems declare:

* ``UPDATE <table> SET <col> = ...`` on a table one of the corpus's models
  declares is a write of that model's field, ``certain``.
* A write of a model imported from a neighbour (``from ai.models import
  Proposal``) is a row the neighbour declares: an outbound ``shared_table``
  boundary named by the class as the import spells it, with the fields written
  (``handed_over``).
* A statement's text that names a table of a schema another system owns
  (``FROM "{PANDA_SCHEMA}"."jobsactive4"``) is an outbound ``shared_table``
  boundary named ``schema.table``, with the SQL verbs used on it
  (``operations``) and the column names the text quotes (``carried_values``).
  The boundary is placed where the text is, which for a statement a helper
  builds is the helper and not the function that runs it.

Every boundary is stated in the fields a map is made of, so
:mod:`bamboo.codemap.bind` binds it to the map of the system it names with no
knowledge of this walk.  In an outbound boundary ``interface`` is the far
side's name and ``calls_out`` the functions here that cross, ``file::name``.

A file is spelled ``<package>/<path inside it>`` whatever directory the
checkout keeps the package in, as the shared call reader spells it
(:class:`bamboo.codemap.calls.Types`), so an owner is
``monitor_app/views.py::f`` although the file is ``src/monitor_app/views.py``.
"""

from __future__ import annotations

import ast
import logging
import re
from pathlib import Path
from typing import Any, Iterator, Optional

from bamboo.codemap import pathcond
from bamboo.codemap.base import CodeMapPlugin
from bamboo.codemap.calls import Calls, Types
from bamboo.codemap.gitsource import blob_sha, describe
from bamboo.codemap.models import (
    BoundaryNode,
    MapFragment,
    MessageEdge,
    MessageReceiver,
)
from bamboo.codemap.orm.assemble import FieldWrite, assemble, write_condition
from bamboo.codemap.orm.resolve import Resolved, resolve, settler
from bamboo.codemap.orm.schema import (
    DjangoModel,
    models_declared_elsewhere,
    module_constants,
    read_models,
)
from bamboo.codemap.orm.writes import CREATING, OrmWrite, scan_function

logger = logging.getLogger(__name__)

_SQL_UPDATE = re.compile(
    r"\bUPDATE\s+\"?(\w+)\"?\s+(?:AS\s+\w+\s+|\w+\s+)?SET\s+(.*?)(?:\bWHERE\b|\bFROM\b|\bRETURNING\b|$)",
    re.I | re.S,
)
_SQL_ASSIGN = re.compile(
    r"(?:^|,)\s*\"?(\w+)\"?\s*=\s*('(?:[^']*)'|%s|%\(\w+\)s|:\w+|[\w.]+)", re.S
)
#: ``FROM "{PANDA_SCHEMA}"."jobsactive4"``: the keyword, the schema as a name or a filled-in part, the table.
_SQL_TABLE = re.compile(
    r"\b(FROM|JOIN|UPDATE|INTO)\s+\"?(\{[\w.]+\}|\w+)\"?\s*\.\s*\"?(\w+)\"?", re.I
)
_SQL_QUOTED = re.compile(r"\"(\w+)\"")
_SQL_VERB = {"FROM": "SELECT", "JOIN": "SELECT", "UPDATE": "UPDATE", "INTO": "INSERT"}


def own_nodes(func: ast.AST) -> Iterator[ast.AST]:
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


def _statement_text(node: ast.AST) -> Optional[tuple[str, str]]:
    """A string's text twice over: with ``%s`` and with ``{name}`` for each part an f-string fills in.

    None for a node that is no string, and for a literal part of an f-string, which is read with the whole.
    """
    if isinstance(node, ast.JoinedStr):
        binds = "".join(
            str(v.value) if isinstance(v, ast.Constant) else "%s" for v in node.values
        )
        named = "".join(
            (
                str(v.value)
                if isinstance(v, ast.Constant)
                else "{%s}" % ast.unparse(getattr(v, "value", v))
            )
            for v in node.values
        )
        return binds, named
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        if isinstance(getattr(node, "parent", None), ast.JoinedStr):
            return None
        return node.value, node.value
    return None


class DjangoCodeMapPlugin(CodeMapPlugin):
    """Builds a Django code base's fragment from a checkout."""

    #: The map's id; also the name its version stamp starts with.
    MAP_ID = ""
    #: Top-level package -> its directory in a checkout, in the order they are walked.
    PACKAGES: dict[str, str] = {}
    #: Top-level package of another code base installed beside this one -> the system it is.  A model this
    #: code imports from one of them is a row that system declares and this one writes
    #: (:meth:`_shared_row`), not a subject of this map.
    NEIGHBOURS: dict[str, str] = {}
    #: Database schema, lowered -> the system that owns its tables (``{"doma_panda": "panda"}``).  A
    #: statement naming a table of one is a boundary to that system (:meth:`_foreign_tables`).
    SCHEMAS: dict[str, str] = {}

    def __init__(self) -> None:
        self._root: Optional[Path] = None
        self._stamp = ""
        self.models: dict[str, DjangoModel] = {}
        self.writes: list[FieldWrite] = []
        self.selected: dict[tuple[str, str], set[str]] = {}
        self.boundaries: dict[str, BoundaryNode] = {}
        self.message_edges: list[MessageEdge] = []
        self.message_receivers: list[MessageReceiver] = []
        self.unresolved_writes = 0
        self._calls: Optional[Calls] = None
        #: module-level name -> the values the corpus's modules give it
        self._everywhere: dict[str, set[str]] = {}

    @property
    def map_id(self) -> str:
        return self.MAP_ID

    def prepare(self, source_root=None, schema_root=None, instance_record=None) -> str:
        if source_root is None:
            raise FileNotFoundError(
                f"{self.MAP_ID} is not an installed distribution: pass --source-root <checkout>"
            )
        root = Path(source_root).resolve()
        if not any((root / where).is_dir() for where in self.PACKAGES.values()):
            raise FileNotFoundError(
                f"{root} holds none of {tuple(self.PACKAGES.values())}"
            )
        self._root = root
        self._stamp = f"{self.MAP_ID} {describe(root) or 'unversioned'}"
        return self._stamp

    # ----------------------------------------------------------------- sources

    def _roots(self) -> dict[str, Path]:
        assert self._root is not None
        return {
            package: self._root / where
            for package, where in self.PACKAGES.items()
            if (self._root / where).is_dir()
        }

    def _sources(self) -> Iterator[tuple[str, str, ast.Module, str]]:
        for package, base in self._roots().items():
            for path in sorted(base.rglob("*.py")):
                rel = f"{package}/{path.relative_to(base).as_posix()}"
                parts = rel.split("/")
                if (
                    "migrations" in parts
                    or "tests" in parts
                    or parts[-1].startswith("test")
                ):
                    continue
                text = path.read_text(encoding="utf-8", errors="replace")
                try:
                    tree = ast.parse(text, filename=rel)
                except SyntaxError as exc:
                    logger.warning("skipping %s: %s", rel, exc)
                    continue
                module = rel[:-3].replace("/", ".")
                if module.endswith(".__init__"):
                    module = module[: -len(".__init__")]
                yield rel, module, tree, text

    # ----------------------------------------------------------------- the subclass's part

    def module_context(self, tree: ast.Module, constants: dict[str, list[str]]) -> Any:
        """What :meth:`visit_node` needs of a module, read once: its imports, its constants."""
        return None

    def visit_module(self, tree: ast.Module, rel: str, context: Any) -> None:
        """A module, before its functions: for boundaries a class declares rather than a call makes."""

    def visit_node(
        self,
        node: ast.AST,
        func: ast.FunctionDef | ast.AsyncFunctionDef,
        owner: str,
        rel: str,
        context: Any,
    ) -> None:
        """One node of *func*, the function *owner* names: record the boundary it crosses, if any."""

    def finish(self) -> None:
        """Every module is read: tie together what was recorded apart."""

    def boundary(
        self,
        owner: str,
        system: str,
        interface: str,
        transport: str,
        rel: str,
        line: int,
        *,
        kind: str = "transports_causation",
        direction: str = "outbound",
        verb: str = "",
        metadata: Optional[dict[str, Any]] = None,
    ) -> BoundaryNode:
        """The boundary *interface* of *system*, made on first sight, with this crossing added.

        One node per (direction, system, interface).  Outbound, the functions that cross it are ``calls_out``
        and each crossing is a ``call_sites`` entry; inbound, the caller says what receives.
        """
        name = (
            f"{self.MAP_ID}->{system}::{interface}"
            if direction == "outbound"
            else f"{system}->{self.MAP_ID}::{interface}"
        )
        node = self.boundaries.get(name)
        if node is None:
            node = BoundaryNode(
                name=name,
                map_id=self.MAP_ID,
                derived_from=self._stamp,
                system=system,
                kind=kind,
                transport=transport,
                interface=interface,
                direction=direction,
                metadata=dict(metadata or {}),
            )
            self.boundaries[name] = node
        if direction == "outbound":
            if owner not in node.calls_out:
                node.calls_out.append(owner)
            node.call_sites.append(f"{rel}:{line}")
        if verb and verb not in node.verbs:
            node.verbs.append(verb)
        return node

    # ----------------------------------------------------------------- one function

    def _scan_function(
        self,
        func: ast.FunctionDef | ast.AsyncFunctionDef,
        qual: str,
        cls: Optional[str],
        rel: str,
        constants: dict[str, list[str]],
        context: Any,
        models: dict[str, DjangoModel],
    ) -> None:
        """*models* is the corpus's and those the module imports from a neighbour."""
        owner = f"{rel}::{qual}"
        settle = settler(models, constants, self._calls)
        scan = scan_function(func, cls, models, constants)
        for key, vals in scan.selected.items():
            self.selected.setdefault(key, set()).update(vals)
        for w in scan.writes:
            if w.model is not None and models[w.model].elsewhere:
                arms = resolve(w.value, w.at, func, settle, models, model=w.model)
                self._shared_row(models[w.model], w, arms, owner, rel)
                continue
            self.writes.append(
                FieldWrite(
                    model=w.model,
                    field=w.field,
                    owner=owner,
                    rel_path=rel,
                    line=w.line,
                    form=w.form,
                    attribution=w.attribution,
                    arms=resolve(w.value, w.at, func, settle, models, model=w.model),
                    path_condition=write_condition(w.at),
                    structural=w.structural,
                )
            )
        for node in own_nodes(func):
            self.visit_node(node, func, owner, rel, context)
            text = _statement_text(node)
            if text is not None:
                binds, named = text
                self._visit_sql(binds, node, owner, rel)
                if self.SCHEMAS:
                    self._foreign_tables(named, node, owner, rel, constants)

    def _shared_row(
        self,
        model: DjangoModel,
        write: OrmWrite,
        arms: list[Resolved],
        owner: str,
        rel: str,
    ) -> None:
        """A write of a field of a model a neighbour declares: an outbound ``shared_table`` boundary named by
        the class as the import spells it, ``ai.models.Proposal``.

        The table is the neighbour's to say, so the boundary carries the fields written (``handed_over``),
        what the writes do to a row (``operations``) and, beside them, each field's values
        (``metadata["values"]``); the neighbour's map has the subject they bind to."""
        node = self.boundary(
            owner,
            self.NEIGHBOURS[model.module.split(".")[0]],
            f"{model.module}.{model.name}",
            "shared_table",
            rel,
            write.line,
            kind="reports_state",
            metadata={"values": {}},
        )
        if write.field not in node.handed_over:
            node.handed_over.append(write.field)
        operation = "INSERT" if write.form in CREATING else "UPDATE"
        if operation not in node.operations:
            node.operations.append(operation)
        values = node.metadata["values"].setdefault(write.field, [])
        values.extend(arm.outcome for arm in arms if arm.outcome not in values)

    def _visit_sql(self, text: str, at: ast.AST, owner: str, rel: str) -> None:
        for m in _SQL_UPDATE.finditer(text):
            table, sets = m.group(1), m.group(2)
            model = next(
                (x.name for x in self.models.values() if x.db_table == table), None
            )
            if model is None:
                continue
            for col, raw_value in _SQL_ASSIGN.findall(sets):
                if col not in self.models[model].state_fields():
                    continue
                arm = (
                    Resolved(raw_value.strip("'"), 1, [])
                    if raw_value.startswith("'")
                    else Resolved(f"runtime({raw_value})", 2, [])
                )
                self.writes.append(
                    FieldWrite(
                        model=model,
                        field=col,
                        owner=owner,
                        rel_path=rel,
                        line=at.lineno,  # type: ignore[attr-defined]
                        form="sql.update",
                        attribution="certain",
                        arms=[arm],
                        path_condition=write_condition(at),
                    )
                )

    def _schema(self, spelled: str, constants: dict[str, list[str]]) -> Optional[str]:
        """The schema a statement names: the word itself, or the one value the name an f-string fills in has
        -- in the module, else anywhere in the corpus (``PANDA_SCHEMA``, imported from the module of
        constants).  None where the name has none or several."""
        if not spelled.startswith("{"):
            return spelled
        name = spelled.strip("{}").rpartition(".")[2]
        values = set(constants.get(name, ())) or self._everywhere.get(name, set())
        return next(iter(values)) if len(values) == 1 else None

    def _foreign_tables(
        self,
        text: str,
        at: ast.AST,
        owner: str,
        rel: str,
        constants: dict[str, list[str]],
    ) -> None:
        """Each table of a schema another system owns that *text* names: an outbound ``shared_table`` boundary
        ``schema.table`` to that system, with the verb used on it and, for a read, the names the text quotes.
        """
        found = list(_SQL_TABLE.finditer(text))
        if not found:
            return
        quoted = list(dict.fromkeys(_SQL_QUOTED.findall(text)))
        for m in found:
            schema = self._schema(m.group(2), constants)
            system = self.SCHEMAS.get((schema or "").lower())
            if schema is None or system is None:
                continue
            keyword, table = m.group(1).upper(), m.group(3)
            verb = _SQL_VERB[keyword]
            if keyword == "FROM" and re.search(
                r"\bDELETE\s+$", text[: m.start()], re.I
            ):
                verb = "DELETE"
            node = self.boundary(
                owner,
                system,
                f"{schema}.{table}",
                "shared_table",
                rel,
                at.lineno,  # type: ignore[attr-defined]
                kind="reports_state",
            )
            if verb not in node.operations:
                node.operations.append(verb)
            if verb == "SELECT":
                tables = {t.group(3) for t in found}
                node.carried_values.extend(
                    name
                    for name in quoted
                    if name not in tables
                    and name != schema
                    and name not in node.carried_values
                )

    # ----------------------------------------------------------------- run

    def run(self) -> MapFragment:
        if self._root is None:
            raise RuntimeError("prepare() must be called before run()")
        parsed = list(self._sources())
        shas = {rel: blob_sha(text) for rel, _m, _t, text in parsed}
        for _rel, _m, tree, _text in parsed:
            pathcond.attach_parents(tree)
        self.models = read_models(
            (rel, module, tree) for rel, module, tree, _t in parsed
        )
        self._calls = Calls(
            Types(
                self._roots(),
                {rel: (shas[rel], tree) for rel, _m, tree, _t in parsed},
            )
        )
        constants_of = {rel: module_constants(tree) for rel, _m, tree, _t in parsed}
        for held in constants_of.values():
            for name, values in held.items():
                if len(values) == 1:
                    self._everywhere.setdefault(name, set()).add(values[0])

        for rel, _module, tree, _text in parsed:
            constants = constants_of[rel]
            context = self.module_context(tree, constants)
            self.visit_module(tree, rel, context)
            # the corpus's own class wins a name both declare
            models = {
                **models_declared_elsewhere(tree, self.NEIGHBOURS),
                **self.models,
            }

            def functions(body, prefix: str, cls: Optional[str]):
                for stmt in body:
                    if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        qual = f"{prefix}{stmt.name}"
                        yield stmt, qual, cls
                        yield from functions(stmt.body, f"{qual}.", cls)
                    elif isinstance(stmt, ast.ClassDef):
                        yield from functions(
                            stmt.body, f"{prefix}{stmt.name}.", stmt.name
                        )

            for func, qual, cls in functions(tree.body, "", None):
                self._scan_function(func, qual, cls, rel, constants, context, models)

        self.finish()
        subjects, entities, junctions, self.unresolved_writes = assemble(
            self.writes, self.models, self.MAP_ID, self._stamp, shas, self.selected
        )
        return MapFragment(
            map_id=self.MAP_ID,
            derived_from=self._stamp,
            subjects=subjects,
            entities=entities,
            junctions=junctions,
            boundaries=sorted(self.boundaries.values(), key=lambda b: b.name),
            message_edges=self.message_edges,
            message_receivers=self.message_receivers,
        )
