"""The prints that run before an arm on every path: :attr:`Branch.must_prints` and :attr:`Branch.must_names`.

A run whose log lacks such a line did not reach the arm, and a run whose line
names the written row is the run that wrote it.  The querier reads it that
way round: of the runs a function's log shows, which one set the value.  A
line that may or may not come before the arm says nothing about a run that
lacks it.

**The rule.**  P must precede arm A when no path from the function's entry
reaches A without passing P: ``reaching(A, [func.args], killers=[P])`` is
empty while ``reaching(A, [func.args])`` is not.  The flow is
:func:`~bamboo.codemap.panda.reaching.reaching`'s -- both ways at every
branch, every loop any number of times including none, the state at every
statement that can raise to its handlers, ``finally`` on every way out -- so
a print inside a ``try`` whose handler goes on to the arm is not one, nor is
one in a loop the arm comes after.  Taking paths that cannot run is the safe
side: it can only drop a must print, never invent one.  What the flow leaves
out (a ``with`` that swallows an exception, a call that never returns) it
leaves out here too.

**The owner's own prints only.**  A function every path to the arm calls
prints too, and its lines would prove as much; that is a question of reach,
left until it is asked.

**Lines, not copies.**  The level, the files and the rows of a must print
are its statement in the owner's :class:`FunctionPrintsNode`, and
:func:`must_prints` joins them for a reader holding only a dump.  The names
are not: whether a hole holds the written row's column is a question of
which binding the print and the arm see, which needs the syntax tree.

**Names are read off the message, for an attribute write.**  The holes of
the line itself, on the object the arm writes to.  Not read yet: the
logger's mark -- ``create_tagged_logger(comment, f"PandaID={pandaID}")``
folds its tag to a hole whose expression is gone -- which is the run's key
and is asked with it; and the row an SQL statement lands on, whose key binds
are paired with its run and not with the deciding statement a branch's line
is.  Such an arm's names are ``None``, with why.

**Every write of a branch.**  One branch can stand for several writes
(:attr:`Branch.other_lines`), and a print before the first need not come
before the others: a branch has what all of them have.
"""

from __future__ import annotations

import ast
from typing import NamedTuple, Optional

from pydantic import BaseModel, Field

from bamboo.codemap.astwalk import walk
from bamboo.codemap.models import (
    LOG_FILES_FOLLOWED,
    MUST_ROW_WRITTEN,
    Branch,
    FunctionPrintsNode,
    JunctionNode,
    MustName,
    PrintRow,
    SourceModule,
)
from bamboo.codemap.panda import pathcond
from bamboo.codemap.panda.reaching import reaching

_FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef)

HOLE_NO_LINE = "the branch says no line"
HOLE_NO_MODULE = "the owner's file is not a module of this build"
HOLE_MODULE_BODY = (
    "the write is in no function: a module's or a class's body, run at import"
)
HOLE_NOT_REACHED = (
    "the arm is not reached from the function's entry, as this reading follows it"
)
WHY_REBOUND = (
    "between the print and the arm the object or the attribute can be bound again,"
    " or the object handed to a call"
)
NAMES_NOT_ATTRIBUTE = (
    "names: the arm does not write the subject as an attribute of an object"
)


class _Site(NamedTuple):
    """One statement a branch writes its value at, in the function holding it."""

    func: ast.AST
    #: The prints' owner of *func*: ``file::main.FinisherThr.run``.
    owner: str
    statement: ast.stmt


def attach(
    junctions: list[JunctionNode],
    function_prints: list[FunctionPrintsNode],
    modules: list[SourceModule],
    declarations: dict[str, set[str]],
) -> int:
    """Read every branch's must prints and names into it.  Returns how many branches have a must print.

    A branch standing for several writes (:attr:`Branch.other_lines`) has the
    prints that come before every one of them, and the names every one of
    them has.
    """
    trees = {module.rel_path: module.tree for module in modules}
    printed: dict[str, list[int]] = {}
    holes: dict[tuple[str, int], list[str]] = {}
    for node in function_prints:
        for statement in node.statements:
            printed.setdefault(node.owner, []).append(statement.line)
            holes[(node.owner, statement.line)] = [
                hole.expression for row in statement.rows for hole in row.holes
            ]
    located: list[tuple[JunctionNode, Branch, list[_Site]]] = []
    by_function: dict[int, tuple[ast.AST, str, dict[int, ast.stmt]]] = {}
    for junction in junctions:
        for branch in junction.branches:
            branch.must_prints = None
            branch.must_names = None
            branch.must_hole = ""
            sites = _sites(junction, branch, trees)
            if isinstance(sites, str):
                branch.must_hole = sites
                continue
            located.append((junction, branch, sites))
            for site in sites:
                held = by_function.setdefault(
                    id(site.func), (site.func, site.owner, {})
                )
                held[2][id(site.statement)] = site.statement
    must: dict[int, Optional[list[int]]] = {}
    statements: dict[int, dict[int, Optional[ast.stmt]]] = {}
    for func, owner, arms in by_function.values():
        lines = sorted(
            line
            for line in set(printed.get(owner, ()))
            if func.lineno <= line <= (func.end_lineno or func.lineno)  # type: ignore[attr-defined]
        )
        statements[id(func)] = {line: _statement_at(func, line) for line in lines}
        entry = [func.args]  # type: ignore[attr-defined]
        for key, arm in arms.items():
            must[key] = [] if reaching(arm, entry) else None
        # One flow per print, asked of every arm in the function: reaching
        # keeps the flow on the scope for the set of writes it was asked with.
        for line, statement in statements[id(func)].items():
            if statement is None:
                continue
            for key, arm in arms.items():
                held = must[key]
                if held is not None and not reaching(arm, entry, [statement]):
                    held.append(line)
    found = 0
    for junction, branch, sites in located:
        reached = [site for site in sites if must[id(site.statement)] is not None]
        if not reached:
            branch.must_hole = HOLE_NOT_REACHED
            continue
        first = must[id(reached[0].statement)] or []
        lines = [
            line
            for line in first
            if all(line in (must[id(site.statement)] or ()) for site in reached[1:])
        ]
        branch.must_prints = lines
        found += bool(lines)
        written = [
            _written_object(site.statement, junction.subject) for site in reached
        ]
        if any(one is None for one in written):
            branch.must_hole = NAMES_NOT_ATTRIBUTE
            continue
        each = [
            _names(
                site,
                one,  # type: ignore[arg-type]
                lines,
                statements[id(site.func)],
                holes,
                junction.subject,
                declarations,
            )
            for site, one in zip(reached, written, strict=True)
        ]
        branch.must_names = _agreed(each)
    return found


def _sites(
    junction: JunctionNode, branch: Branch, trees: dict[str, ast.AST]
) -> list[_Site] | str:
    """The statements *branch* writes at, or why one of them cannot be read."""
    file, _sep, qualname = junction.owner.partition("::")
    tree = trees.get(file)
    if tree is None:
        return HOLE_NO_MODULE
    if branch.line is None:
        return HOLE_NO_LINE
    _parented(tree)
    sites: list[_Site] = []
    for line in (branch.line, *branch.other_lines):
        func = _function_at(tree, line)
        if func is None:
            return HOLE_MODULE_BODY
        # A junction's owner need not spell the nesting -- ``datasetManager.py::run``
        # is any of seven -- and the prints' owner does: ``main.FinisherThr.run``.
        nested = pathcond.qualified_name(func)
        if nested != qualname and not nested.endswith("." + qualname):
            return f"line {line} is in {nested}, not {qualname}"
        statement = _statement_at(func, line)
        if statement is None:
            return f"no statement of {nested} holds line {line}"
        sites.append(_Site(func, f"{file}::{nested}", statement))
    return sites


def _agreed(each: list[list[MustName]]) -> list[MustName]:
    """The names every write of a branch has, said to hold the row only where every one says so."""
    first, rest = each[0], each[1:]
    agreed: list[MustName] = []
    for name in first:
        key = (name.line, name.expression)
        others = [
            [held for held in names if (held.line, held.expression) == key]
            for names in rest
        ]
        if not all(others):
            continue
        if name.row and all(held[0].row for held in others):
            agreed.append(name)
        else:
            agreed.append(
                name.model_copy(update={"row": "", "why": name.why or WHY_REBOUND})
            )
    return agreed


def _names(
    site: _Site,
    written: ast.expr,
    lines: list[int],
    statements: dict[int, Optional[ast.stmt]],
    holes: dict[tuple[str, int], list[str]],
    subject: str,
    declarations: dict[str, set[str]],
) -> list[MustName]:
    """The holes of the must prints that read an attribute of *written*, each with whether it holds the arm's row."""
    cls = subject.rpartition(".")[0]
    columns = declarations.get(cls, set())
    spelled = ast.unparse(written)
    names: list[MustName] = []
    for line in lines:
        seen: set[str] = set()
        for expression in holes.get((site.owner, line), ()):
            if expression in seen:
                continue
            seen.add(expression)
            attribute = _attribute_of(expression, spelled)
            if attribute is None or attribute not in columns:
                continue
            statement = statements.get(line)
            same = statement is not None and _unbound_since(
                site.func, site.statement, statement, written, attribute
            )
            names.append(
                MustName(
                    line=line,
                    expression=expression,
                    column=f"{cls}.{attribute}",
                    row=MUST_ROW_WRITTEN if same else "",
                    why="" if same else WHY_REBOUND,
                )
            )
    return names


def _attribute_of(expression: str, written: str) -> Optional[str]:
    """``PandaID`` for ``job.PandaID`` when *written* is ``job``; ``None`` for anything else."""
    try:
        parsed = ast.parse(expression, mode="eval").body
    except SyntaxError:
        return None
    if isinstance(parsed, ast.Attribute) and ast.unparse(parsed.value) == written:
        return parsed.attr
    return None


def _unbound_since(
    func: ast.AST, arm: ast.stmt, printed: ast.stmt, written: ast.expr, attribute: str
) -> bool:
    """Whether, on every path into *arm*, *printed* ran after *written* and its *attribute* were last bound.

    The writes are the name's bindings in the function, every store to the
    chain ``job``, ``job.PandaID`` (``self.job`` and ``self.job.PandaID`` for
    ``self.job``), every call made on an object of the chain or handed one --
    ``self.ppEventServiceJob(job, ...)`` between ``archiveJob``'s first print
    and its arms may set any of ``job``'s attributes, and ``self.retry()``
    may rebind ``self.job`` -- and the call's own binding of the parameters;
    the print kills them.  None of them reaching the arm means the print saw
    what the arm sees.  A store through another name for the object (``j =
    job; j.PandaID = 3``) is not seen.
    """
    chain = [ast.unparse(written) + "." + attribute]
    root: ast.expr = written
    while isinstance(root, ast.Attribute):
        chain.append(ast.unparse(root))
        root = root.value
    if not isinstance(root, ast.Name):
        return False
    sites: list[ast.AST] = [func.args]  # type: ignore[attr-defined]
    for binding in pathcond.bindings(func, root.id):
        if not binding.nested:
            site = binding.site
            sites.append(
                site
                if isinstance(site, (ast.stmt, ast.ExceptHandler))
                else _statement_of(site)
            )
    stored = set(chain)
    # The objects whose state holds the attribute: ``job``; ``self.job`` and ``self``.
    holding = {*chain[1:], root.id}
    for node in walk(func):
        if isinstance(node, ast.Attribute) and isinstance(
            node.ctx, (ast.Store, ast.Del)
        ):
            if ast.unparse(node) in stored:
                sites.append(_statement_of(node))
        elif isinstance(node, ast.Call) and _given(node, holding):
            sites.append(_statement_of(node))
    return not reaching(
        arm,
        [site for site in sites if site is not None and site is not printed],
        [printed],
    )


def _given(call: ast.Call, holding: set[str]) -> bool:
    """Whether *call* is made on one of *holding* or is handed one of them."""
    if isinstance(call.func, ast.Attribute) and ast.unparse(call.func.value) in holding:
        return True
    given = [*call.args, *(keyword.value for keyword in call.keywords)]
    return any(
        ast.unparse(value.value if isinstance(value, ast.Starred) else value) in holding
        for value in given
    )


def _written_object(statement: ast.stmt, subject: str) -> Optional[ast.expr]:
    """``job`` for ``job.jobStatus = ...`` when *subject* is ``JobSpec.jobStatus``: the object the arm writes to."""
    attribute = subject.rpartition(".")[2]
    if isinstance(statement, ast.Assign):
        targets = statement.targets
    elif isinstance(statement, (ast.AnnAssign, ast.AugAssign)):
        targets = [statement.target]
    else:
        return None
    found = [
        target.value
        for target in targets
        if isinstance(target, ast.Attribute)
        and target.attr == attribute
        and _chain(target.value)
    ]
    return found[0] if len(found) == 1 else None


def _chain(node: ast.expr) -> bool:
    """A name, or attributes on one: ``job``, ``self.job``."""
    while isinstance(node, ast.Attribute):
        node = node.value
    return isinstance(node, ast.Name)


def _parented(tree: ast.AST) -> None:
    body = getattr(tree, "body", None)
    if body and not hasattr(body[0], "parent"):
        pathcond.attach_parents(tree)


def _function_at(tree: ast.AST, line: int) -> Optional[ast.AST]:
    """The innermost ``def`` holding *line*."""
    best = None
    for node in walk(tree):
        if isinstance(node, _FUNCTIONS) and node.lineno <= line <= (
            node.end_lineno or node.lineno
        ):
            if best is None or node.lineno >= best.lineno:
                best = node
    return best


def _statement_at(func: ast.AST, line: int) -> Optional[ast.stmt]:
    """The innermost statement of *func* holding *line*, *func* itself left out."""
    best: Optional[ast.stmt] = None
    for node in walk(func):
        if not isinstance(node, ast.stmt) or node is func:
            continue
        if node.lineno <= line <= (node.end_lineno or node.lineno):
            if best is None or (
                node.lineno >= best.lineno
                and (node.end_lineno or node.lineno) <= (best.end_lineno or best.lineno)
            ):
                best = node
    return best


def _statement_of(node: ast.AST) -> Optional[ast.stmt]:
    current: Optional[ast.AST] = node
    while current is not None and not isinstance(current, ast.stmt):
        current = getattr(current, "parent", None)
    return current  # type: ignore[return-value]


# -- reading it from a dump -------------------------------------------------- #


class MustPrint(BaseModel):
    """One must print as a reader needs it: the owner's statement at the line, and its names."""

    line: int
    level: str
    log_files: list[str] = Field(default_factory=list)
    log_files_basis: str = ""
    rows: list[PrintRow] = Field(default_factory=list)
    names: list[MustName] = Field(
        default_factory=list,
        description="The holes said to hold the written row's columns.",
    )
    holes: list[str] = Field(
        default_factory=list,
        description="Why the line may not be where or what a reader expects, and the names not said.",
    )


class MustPrints(BaseModel):
    """:func:`must_prints`'s answer: the prints, or ``None`` with why."""

    prints: Optional[list[MustPrint]] = None
    why: str = ""


def must_prints(
    function_prints: list[FunctionPrintsNode], junction: JunctionNode, branch: Branch
) -> MustPrints:
    """The must prints of *branch*, each joined with its statement in the owner's :class:`FunctionPrintsNode`.

    A line's files are said only where they were followed to the logger
    (:data:`LOG_FILES_FOLLOWED`); otherwise they are the module's, a logger
    handed in from outside or a plugin's, and the line is held with that as a
    hole.  So is a row whose prefix is empty and may not be
    (:attr:`PrintRow.prefix_hole`): ``self.logger`` built in ``__init__``
    writes its mark before every message though the row says none, and a
    reader looking for the run by its mark has to know the mark was not read.
    ``None`` when the branch's prints were not read, with why.
    """
    if branch.must_prints is None:
        return MustPrints(
            why=branch.must_hole
            or "not read: the map was built before must prints were"
        )
    statements = {
        statement.line: statement
        for statement in _printing(function_prints, junction, branch).statements
    }
    names = branch.must_names or []
    prints: list[MustPrint] = []
    for line in branch.must_prints:
        statement = statements.get(line)
        if statement is None:
            return MustPrints(
                why=f"line {line} is no print statement of {junction.owner} in this dump"
            )
        held = (
            [
                f"files read off the module ({statement.log_files_basis}): the logger was not followed to them"
            ]
            if statement.log_files_basis != LOG_FILES_FOLLOWED
            else []
        )
        held.extend(
            f"{name.expression}: {name.why}"
            for name in names
            if name.line == line and not name.row
        )
        held.extend(
            f"row refused: {row.refused}" for row in statement.rows if row.refused
        )
        held.extend(
            f"prefix not read: {why}"
            for why in dict.fromkeys(
                row.prefix_hole for row in statement.rows if row.prefix_hole
            )
        )
        prints.append(
            MustPrint(
                line=line,
                level=statement.level,
                log_files=list(statement.log_files),
                log_files_basis=statement.log_files_basis,
                rows=list(statement.rows),
                names=[name for name in names if name.line == line and name.row],
                holes=held,
            )
        )
    if branch.must_names is None and branch.must_hole:
        for printed in prints:
            printed.holes.append(branch.must_hole)
    return MustPrints(prints=prints)


def _printing(
    function_prints: list[FunctionPrintsNode], junction: JunctionNode, branch: Branch
) -> FunctionPrintsNode:
    """The innermost function holding the branch's line that prints, or an empty one.

    By position, because the junction's owner need not spell the nesting the
    prints' owner does.
    """
    file = junction.owner.partition("::")[0]
    line = branch.line or 0
    holding = [
        node
        for node in function_prints
        if node.anchor is not None
        and node.anchor.file == file
        and node.anchor.line_start
        <= line
        <= (node.anchor.line_end or node.anchor.line_start)
    ]
    if not holding:
        return FunctionPrintsNode(
            name="", map_id="", derived_from="", owner="", function="", count=0
        )
    return max(holding, key=lambda node: node.anchor.line_start)  # type: ignore[union-attr]
