"""The prints that run before an arm on every path: :attr:`Branch.must_prints` and :attr:`Branch.must_names`.

A run whose log lacks such a line did not reach the arm, and a run whose line
names the written row is the run that wrote it.  The querier reads it that
way round: of the runs a function's log shows, which one set the value.  A
line that may or may not come before the arm says nothing about a run that
lacks it.

**The rule.**  P must precede arm A when no path from the function's entry
reaches A without passing P: ``reaching(A, [func.args], killers=[P])`` is
empty while ``reaching(A, [func.args])`` is not.  The flow is
:func:`~bamboo.codemap.reaching.reaching`'s -- both ways at every
branch, every loop any number of times including none, the state at every
statement that can raise to its handlers, ``finally`` on every way out -- so
a print inside a ``try`` whose handler goes on to the arm is not one, nor is
one in a loop the arm comes after.  Taking paths that cannot run is the safe
side: it can only drop a must print, never invent one.  What the flow leaves
out (a ``with`` that swallows an exception, a call that never returns) it
leaves out here too.

**An SQL arm writes where its statement runs.**  Its line fills a bind
(``varMap[":status"] = ...``) or sets a column in the statement's text, and
the row changes at ``self.cur.execute(...)``.  So P is a must print too
where it is on every path from the arm to each run of the statement
(:attr:`KeyFrom.statement`): ``reaching(run, [A], killers=[P])`` empty --
``tmp_log.debug(sql + comment + str(varMap))`` before the execute.  A print
before the arm on every path stays one: a branch filling the bind with its
own value is passed on the way to the run by every run writing that value.

**The owner's own prints only.**  A function every path to the arm calls
prints too, and its lines would prove as much; that is a question of reach,
left until it is asked.

**Lines, not copies.**  The level, the files and the rows of a must print
are its statement in the owner's :class:`FunctionPrintsNode`, and
:func:`must_prints` joins them for a reader holding only a dump.  The names
are not: whether a hole holds the written row's column is a question of
which binding the print and the arm see, which needs the syntax tree.

**Names are read off the message and the logger's mark.**  The holes of the
line itself, on the object the arm writes to; and those holding a key of the
written row -- its identity, or for an SQL arm each term naming its rows --
in the message or in the mark of the logger it writes through, which is
taken where the logger is made: ``tmp_log =
self.create_tagged_logger(comment, f"PandaID={tmp_id}")`` after ``tmp_id =
job.PandaID``.  An SQL arm is a line its statement writes the subject on
(:attr:`KeyFrom.writes`), and the key is what each place filling the term's
bind fills it with (:attr:`KeySource.line`): ``varMap[":jediTaskID"] =
jediTaskID``.  A name is said where the hole holds what the key holds where
the write counts (:func:`_holds_key`).  An arm writing the subject neither
way has ``None``, with why.

**Every write of a branch.**  One branch can stand for several writes
(:attr:`Branch.other_lines`), and a print before the first need not come
before the others: a branch has what all of them have.
"""

from __future__ import annotations

import ast
import re
from typing import TYPE_CHECKING, Callable, NamedTuple, Optional, Sequence

from pydantic import BaseModel, Field

from bamboo.codemap import pathcond
from bamboo.codemap.astwalk import walk
from bamboo.codemap.callfacts import LOG_METHODS
from bamboo.codemap.calls import spells
from bamboo.codemap.models import (
    EXIT_LINE,
    LOG_FILES_FOLLOWED,
    MUST_PART_BIND,
    MUST_PART_MARK,
    MUST_ROW_WRITTEN,
    AfterPrint,
    Branch,
    FunctionPrintsNode,
    JunctionNode,
    KeyFrom,
    MadeBefore,
    MapEntry,
    MustName,
    PrintRow,
    SourceModule,
    TableRecord,
    TakenBack,
)
from bamboo.codemap.panda import changes, flow, mapentries, order
from bamboo.codemap.panda.recognizers import progress
from bamboo.codemap.reaching import reaching, reaching_end

if TYPE_CHECKING:
    from bamboo.codemap.calls import Calls

_FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef)

#: Whether an object's attribute holds from one statement to another: ``(func, arm, printed, written, attribute)``.
_Kept = Callable[[ast.AST, ast.stmt, ast.stmt, ast.expr, str], bool]

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
    " or the object handed to a call that may store the attribute"
)
WHY_KEY_UNSHOWN = (
    "the hole reads the written row's key, but this reading cannot show it holds the value the write counts:"
    " the key or the name holding it can be bound again, or changed by a call, between them"
)
NAMES_NOT_ATTRIBUTE = (
    "names: the arm does not write the subject as an attribute of an object"
)
NAMES_NO_KEY = (
    "names: no term of the arm's statement naming its rows is a bind this reading"
    " pairs with the value filling it"
)


class _Site(NamedTuple):
    """One statement a branch writes its value at, in the function holding it."""

    func: ast.AST
    #: The prints' owner of *func*: ``file::main.FinisherThr.run``.
    owner: str
    statement: ast.stmt
    #: The branch's line it stands for: :attr:`Branch.line` or one of :attr:`Branch.other_lines`.
    line: int = 0


class _Key(NamedTuple):
    """A column of the row an arm writes, and the expression holding its value where the write counts."""

    #: As :attr:`MustName.column` spells it: ``JobSpec.PandaID``.
    column: str
    expression: ast.expr
    #: Where the value counts: the arm, or the statement an SQL arm's bind is run in.
    at: ast.stmt
    #: The statements filling an SQL arm's key bind, each of which must hold what *at* holds.
    filled: tuple[ast.stmt, ...] = ()
    #: An SQL arm's key bind, ``:jediTaskID``, and the name of the bind map every fill fills it in: ``varMap``.
    bind: str = ""
    map: str = ""


def attach(
    junctions: list[JunctionNode],
    function_prints: list[FunctionPrintsNode],
    modules: list[SourceModule],
    declarations: dict[str, set[str]],
    key_froms: Sequence[KeyFrom] = (),
    tables: Sequence[TableRecord] = (),
    calls: Optional["Calls"] = None,
) -> int:
    """Read every branch's must prints and names into it.  Returns how many branches have a must print.

    A branch standing for several writes (:attr:`Branch.other_lines`) has the
    prints that come before every one of them, and the names every one of
    them has.  With *key_froms* and *tables*, the names of the row's key too:
    held in a hole of the logger's mark, or by a name bound to it, and an SQL
    arm's (:func:`_key_names`).  With *calls*, a call between a print and the
    arm changes a value only where :func:`changes.may_change` says it may;
    without, every call on the object or handed it may.
    """
    global _ACROSS
    files = {
        id(node): module.rel_path
        for module in modules
        for node in ast.walk(module.tree)
        if isinstance(node, _FUNCTIONS)
    }
    _ACROSS = (calls, files) if calls is not None else None
    # What was kept outside an attach was read without the corpus's calls.
    for kept in _KEPT:
        kept.clear()
    try:
        return _attach(
            junctions, function_prints, modules, declarations, key_froms, tables
        )
    finally:
        _ACROSS = None
        _ASSIGNMENTS.clear()
        _SINGLES.clear()
        for kept in _KEPT:
            kept.clear()


def _attach(
    junctions: list[JunctionNode],
    function_prints: list[FunctionPrintsNode],
    modules: list[SourceModule],
    declarations: dict[str, set[str]],
    key_froms: Sequence[KeyFrom],
    tables: Sequence[TableRecord],
) -> int:
    trees = {module.rel_path: module.tree for module in modules}
    initial = progress.spec_initial_values(modules)
    printed: dict[str, list[int]] = {}
    holes: dict[tuple[str, int], list[str]] = {}
    maps_read: dict[tuple[str, int], dict[str, Optional[_MapRead]]] = {}
    marks: dict[tuple[str, int], list[str]] = {}
    unread: dict[tuple[str, int], bool] = {}
    for node in function_prints:
        for statement in node.statements:
            printed.setdefault(node.owner, []).append(statement.line)
            holes[(node.owner, statement.line)] = [
                hole.expression for row in statement.rows for hole in row.holes
            ]
            maps_read[(node.owner, statement.line)] = _maps_read(statement.rows)
            marks[(node.owner, statement.line)] = [
                hole.expression for row in statement.rows for hole in row.prefix_holes
            ]
            # A row whose message, or whose logger's mark, this reading has no text of.
            unread[(node.owner, statement.line)] = any(
                row.refused or (not row.prefix and row.prefix_hole)
                for row in statement.rows
            )
    identity: dict[str, set[str]] = {}
    for table in tables:
        if table.rows_of:
            identity.setdefault(table.rows_of, set()).update(
                attr.partition(".")[2] for attr in table.identity_attrs
            )
    writing: dict[tuple[str, int], list[KeyFrom]] = {}
    for record in key_froms:
        # A statement's own terms: not a read's, nor an object's flush through a method, whose terms are the method's.
        if record.reads or record.object:
            continue
        for lines_written in record.writes.values():
            for line_written in lines_written:
                writing.setdefault((record.owner, line_written), []).append(record)
    located: list[tuple[JunctionNode, Branch, list[_Site]]] = []
    by_function: dict[int, tuple[ast.AST, str, dict[int, ast.stmt]]] = {}
    for junction in junctions:
        for branch in junction.branches:
            branch.must_prints = None
            branch.must_names = None
            branch.after_prints = None
            branch.made_before = None
            branch.taken_back = None
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
        for line, printing in statements[id(func)].items():
            if printing is None:
                continue
            for key, arm in arms.items():
                missed = must[key]
                if missed is not None and not reaching(arm, entry, [printing]):
                    missed.append(line)
    found = 0
    # Each junction's own arms, for what may run between an object's construction and an arm of it (made_before).
    own_arms: dict[int, list[ast.stmt]] = {}
    for junction, _branch, sites in located:
        own_arms.setdefault(id(junction), []).extend(site.statement for site in sites)
    for junction, branch, sites in located:
        reached = [site for site in sites if must[id(site.statement)] is not None]
        if not reached:
            branch.must_hole = HOLE_NOT_REACHED
            continue
        sql = _written_object(reached[0].statement, junction.subject) is None
        before = {
            id(site.statement): sorted(
                {
                    *(must[id(site.statement)] or ()),
                    *(
                        _on_the_way(
                            site,
                            writing.get((junction.owner, site.line), ()),
                            statements[id(site.func)],
                        )
                        if sql
                        else ()
                    ),
                }
            )
            for site in reached
        }
        first = before[id(reached[0].statement)]
        lines = [
            line
            for line in first
            if all(line in before[id(site.statement)] for site in reached[1:])
        ]
        branch.must_prints = lines
        found += bool(lines)
        made = [
            _made_before(site, written, junction.subject, initial)
            for site in reached
            for written in [_written_object(site.statement, junction.subject)]
        ]
        if made and made[0] is not None and all(one == made[0] for one in made[1:]):
            kept = [
                _made_held(site, written, junction.subject, own_arms[id(junction)])
                for site in reached
                for written in [_written_object(site.statement, junction.subject)]
            ]
            branch.made_before = made[0].model_copy(
                update={
                    "held": all(one[0] for one in kept),
                    "arms_between": sorted({line for one in kept for line in one[1]}),
                }
            )
        backs = [
            _taken_back(site, written, junction.subject, statements[id(site.func)])
            for site in reached
            for written in [_written_object(site.statement, junction.subject)]
        ]
        if backs and all(one is not None for one in backs):
            opened = {line for one in backs for line in one[1]}  # type: ignore[index]
            branch.taken_back = TakenBack(
                resets=sorted({line for one in backs for line in one[0]}),  # type: ignore[index]
                open=sorted(opened),
                open_prints=sorted(set.intersection(*(set(one[2]) for one in backs))) if opened else [],  # type: ignore[index]
            )
        each: list[list[MustName]] = []
        afters: list[dict[int, AfterPrint]] = []
        counted: dict[int, str] = {}
        for one in branch.count_prints:
            counted.setdefault(one.line, one.rows)
        whole = True
        for site in reached:
            written = _written_object(site.statement, junction.subject)
            run_by: Sequence[KeyFrom] = ()
            if written is not None:
                names = _names(
                    site,
                    written,
                    lines,
                    statements[id(site.func)],
                    holes,
                    junction.subject,
                    declarations,
                )
                keys = _attribute_keys(site, written, junction.subject, identity)
            else:
                names = []
                terms = writing.get((junction.owner, site.line))
                if not terms:
                    branch.must_hole = NAMES_NOT_ATTRIBUTE
                    break
                keys = _statement_keys(site, terms, junction.subject, declarations)
                if not keys:
                    branch.must_hole = NAMES_NO_KEY
                    break
                run_by = terms
            names += _key_names(
                site,
                keys,
                written is not None,
                lines,
                statements[id(site.func)],
                holes,
                marks,
                maps_read,
            )
            each.append(names)
            afters.append(
                _after_prints(
                    site,
                    written,
                    keys,
                    run_by,
                    statements[id(site.func)],
                    holes,
                    marks,
                    maps_read,
                    junction.subject,
                    declarations,
                    counted,
                )
            )
            whole = whole and _read_whole(
                site,
                keys,
                written is None,
                lines,
                statements[id(site.func)],
                holes,
                marks,
                unread,
                maps_read,
            )
        else:
            branch.must_names = _agreed(each)
            branch.must_names_whole = whole
            branch.after_prints = _after_agreed(afters)
    return found


def _after_prints(
    site: _Site,
    written: Optional[ast.expr],
    keys: list[_Key],
    terms: Sequence[KeyFrom],
    statements: dict[int, Optional[ast.stmt]],
    holes: dict[tuple[str, int], list[str]],
    marks: dict[tuple[str, int], list[str]],
    maps_read: dict[tuple[str, int], dict[str, Optional[_MapRead]]],
    subject: str,
    declarations: dict[str, set[str]],
    counted: dict[int, str],
) -> dict[int, AfterPrint]:
    """The prints after one write of a branch, by line (:attr:`Branch.after_prints`): those a way on from the write
    reaches in one turn of each loop holding both (:func:`flow.turn_loops`) -- for an SQL arm, from a run of its
    statement the arm reaches so -- that the arm's own way does not rule out (:func:`flow.ruled_out_after`); each
    with its names read from the write to it, whether the object is bound again between, and whether every way on
    from the write passes it (:func:`flow.passes_after`)."""
    starts = [site.statement] if written is not None else _runs_after(site, terms)
    found: dict[int, AfterPrint] = {}
    for line, printed in statements.items():
        if printed is None or printed is site.statement or any(printed is start for start in starts):
            continue
        after = [start for start in starts if reaching(printed, [start], once=flow.turn_loops(start, printed))]
        if not after or flow.ruled_out_after(site.statement, printed):
            continue
        names = (
            _names(site, written, [line], statements, holes, subject, declarations, kept=_kept_after)
            if written is not None
            else []
        )
        names += _key_names(site, keys, written is not None, [line], statements, holes, marks, maps_read, after=True)
        # The object written, or for an SQL arm what holds each key's value: the name, or the attribute's owner.
        objects = (
            [written]
            if written is not None
            else [key.expression.value if isinstance(key.expression, ast.Attribute) else key.expression for key in keys]
        )
        found[line] = AfterPrint(
            line=line,
            names=names,
            rebound=any(
                flow.changed_between(start, printed, _object_sites(site.func, one))
                for start in after
                for one in objects
            ),
            every_way=all(flow.passes_after(start, printed) for start in after),
            rows=counted.get(line, ""),
        )
    return found


def _made_before(
    site: _Site, written: Optional[ast.expr], subject: str, initial: dict[tuple[str, str], str]
) -> Optional[MadeBefore]:
    """The construction making the object *written* that is the only binding of its name reaching the arm --
    ``tmpTaskSpec = JediTaskSpec()``, a class called with nothing -- and the written field's value as its constructor
    left it (*initial*, :func:`progress.spec_initial_values`); None otherwise."""
    made = _construction(site, written)
    if made is None:
        return None
    called = made.value.func  # type: ignore[attr-defined]
    cls = called.id if isinstance(called, ast.Name) else called.attr if isinstance(called, ast.Attribute) else ""
    left = initial.get((cls, subject.rpartition(".")[2]))
    return MadeBefore(line=made.lineno, cls=cls, value=left) if left is not None else None


def _construction(site: _Site, written: Optional[ast.expr]) -> Optional[ast.Assign]:
    """``tmpTaskSpec = JediTaskSpec()``: the assignment of a class called with nothing to the name *written*, where it
    is the only binding of the name reaching the arm; None otherwise."""
    if not isinstance(written, ast.Name) or not isinstance(site.func, _FUNCTIONS):
        return None
    held = pathcond.held_at(site.func, written.id, site.statement)
    if len(held.sites) != 1:
        return None
    (made,) = held.sites
    value = getattr(made, "value", None)
    if (
        isinstance(made, ast.Assign)
        and len(made.targets) == 1
        and isinstance(made.targets[0], ast.Name)
        and made.targets[0].id == written.id
        and isinstance(value, ast.Call)
        and not value.args
        and not value.keywords
    ):
        return made
    return None


def _made_held(
    site: _Site, written: Optional[ast.expr], subject: str, own: list[ast.stmt]
) -> tuple[bool, list[int]]:
    """Whether the field the arm writes holds what its construction left up to the arm but for the junction's own
    arms (*own*), and the lines of those that can run between: nothing else that may change it
    (:func:`_changes_of`) runs on a way from the construction to the arm in one turn (:func:`flow.changed_between`)."""
    made = _construction(site, written)
    if made is None or written is None:
        return False, []
    attribute = subject.rpartition(".")[2]
    sites = _changes_of(site.func, written, attribute)
    if sites is None:
        return False, []
    spelled = ast.unparse(written)
    same = [
        one
        for one in own
        if one is not site.statement
        and pathcond.enclosing_function(one) is site.func
        and (object_ := _written_object(one, subject)) is not None
        and ast.unparse(object_) == spelled
    ]
    mine = {id(one) for one in same}
    others = [one for one in sites if one is not made and id(one) not in mine]
    between = sorted({one.lineno for one in same if flow.changed_between(made, site.statement, [one])})
    return not flow.changed_between(made, site.statement, others), between


def _taken_back(
    site: _Site, written: Optional[ast.expr], subject: str, prints: dict[int, Optional[ast.stmt]]
) -> Optional[tuple[list[int], list[int], list[int]]]:
    """The lines of the calls taking the arm's change back on its object that a way from the arm reaches in one turn
    (:func:`order.reset_columns`); the lines a way from the arm reaches without passing one: each ``return``, the
    function's end (:data:`EXIT_LINE`), each call made on the object or handed it; and the lines of the *prints* every
    such way passes first.  None where no reset is reached."""
    if written is None or not isinstance(site.func, _FUNCTIONS):
        return None
    arm = site.statement
    attribute = subject.rpartition(".")[2]
    spelled = ast.unparse(written)
    resets: list[ast.stmt] = []
    handing: list[ast.stmt] = []
    returns: list[ast.stmt] = []
    for node in walk(site.func):
        if isinstance(node, ast.Return) and pathcond.enclosing_function(node) is site.func:
            returns.append(node)
            continue
        if not isinstance(node, ast.Call) or pathcond.enclosing_function(node) is not site.func:
            continue
        statement = _statement_of(node)
        if statement is None or statement is arm:
            continue
        called = node.func
        if isinstance(called, ast.Attribute) and ast.unparse(called.value) == spelled:
            columns = order.reset_columns(node)
            if columns is not None and (attribute in columns or "*" in columns):
                resets.append(statement)
                continue
        if _given(node, {spelled}):
            handing.append(statement)
    resets = [one for one in resets if reaching(one, [arm], entered=[arm], once=flow.turn_loops(arm, one))]
    if not resets:
        return None
    targets = [
        one
        for one in [*returns, *handing]
        if all(one is not reset for reset in resets) and reaching(one, [arm], resets, entered=[arm, *resets])
    ]
    ends = bool(reaching_end(site.func, [arm], resets, entered=[arm, *resets]))
    open_ = {one.lineno for one in targets} | ({EXIT_LINE} if ends else set())

    def covers(printed: ast.stmt) -> bool:
        killers = [*resets, printed]
        return all(not reaching(one, [arm], killers, entered=[arm, *killers]) for one in targets) and not (
            ends and reaching_end(site.func, [arm], killers, entered=[arm, *killers])
        )

    covering = sorted(
        line
        for line, printed in prints.items()
        if printed is not None
        and printed is not arm
        and open_
        and reaching(printed, [arm], entered=[arm])
        and covers(printed)
    )
    return sorted({one.lineno for one in resets}), sorted(open_), covering


def _runs_after(site: _Site, terms: Sequence[KeyFrom]) -> list[ast.stmt]:
    """The runs of the statements an SQL arm fills (*terms*) that a way on from the arm reaches in one turn of each
    loop holding both: where its row is written."""
    runs: list[ast.stmt] = []
    for record in terms:
        ran = _statement_at(site.func, record.statement)
        if (
            ran is not None
            and ran is not site.statement
            and all(ran is not one for one in runs)
            and reaching(ran, [site.statement], entered=[site.statement], once=flow.turn_loops(site.statement, ran))
        ):
            runs.append(ran)
    return runs


#: :func:`_object_sites`' answers while :func:`attach` runs, by the function's id and the object as spelled -- 573
#: objects asked 100124 times on pin s; each with its function, as :data:`_ASSIGNMENTS` keeps it.
_OBJECT_SITES: dict[tuple[int, str], tuple[ast.AST, tuple[ast.AST, ...]]] = {}


def _object_sites(func: ast.AST, held: ast.expr) -> list[ast.AST]:
    """Where the object *held* -- a name, ``tmpTaskSpec``, or an attribute chain, ``self.job`` -- can be bound again
    in *func*: the bindings of the chain's root name, every store to the chain.  Read once for each function and
    object while :func:`attach` runs: the answer is the chain's, by its names, not the node's."""
    key = (id(func), ast.unparse(held))
    kept = _OBJECT_SITES.get(key)
    if kept is None:
        kept = _OBJECT_SITES[key] = (func, tuple(_read_object_sites(func, held)))
    return list(kept[1])


def _read_object_sites(func: ast.AST, held: ast.expr) -> list[ast.AST]:
    chain: set[str] = set()
    root: ast.expr = held
    while isinstance(root, ast.Attribute):
        chain.add(ast.unparse(root))
        root = root.value
    if not isinstance(root, ast.Name):
        return []
    sites: list[ast.AST] = []
    for binding in pathcond.bindings(func, root.id):  # type: ignore[arg-type]
        if binding.nested or isinstance(binding.site, ast.arg):
            continue
        site = binding.site
        statement = site if isinstance(site, (ast.stmt, ast.ExceptHandler)) else _statement_of(site)
        if statement is not None:
            sites.append(statement)
    for node in walk(func):
        if isinstance(node, ast.Attribute) and isinstance(node.ctx, (ast.Store, ast.Del)) and ast.unparse(node) in chain:
            statement = _statement_of(node)
            if statement is not None:
                sites.append(statement)
    return sites


def _after_agreed(each: list[dict[int, AfterPrint]]) -> list[AfterPrint]:
    """The prints after every write of a branch, each with the names every one has (:func:`_agreed`), bound again
    where after one, and every way's only where so after each."""
    lines = sorted(set(each[0]).intersection(*each[1:])) if each else []
    return [
        each[0][line].model_copy(
            update={
                "names": _agreed([one[line].names for one in each]),
                "rebound": any(one[line].rebound for one in each),
                "every_way": all(one[line].every_way for one in each),
            }
        )
        for line in lines
    ]


def _read_whole(
    site: _Site,
    keys: list[_Key],
    sql: bool,
    lines: list[int],
    statements: dict[int, Optional[ast.stmt]],
    holes: dict[tuple[str, int], list[str]],
    marks: dict[tuple[str, int], list[str]],
    unread: dict[tuple[str, int], bool],
    maps_read: dict[tuple[str, int], dict[str, Optional[_MapRead]]],
) -> bool:
    """Whether the names of *site* were read over every must print whole (:attr:`Branch.must_names_whole`): each
    print's message and mark have a text, a mark with holes is made in the function, and every map printed is the
    one an SQL arm's statement runs with or one whose every entry the print's record reads (:func:`_read_all`),
    which :func:`_key_names` reads."""
    maps = {key.map for key in keys if key.map} if sql else set()
    for line in lines:
        where = (site.owner, line)
        printed = statements.get(line)
        if printed is None or unread.get(where, True):
            return False
        if marks.get(where) and _logger_made(site.func, printed) is None:
            return False
        for expression in holes.get(where, ()):
            name = _printed_map(expression)
            if (
                name
                and name not in maps
                and _is_map(site.func, name)
                and not _read_all(maps_read.get(where, {}).get(expression))
            ):
                return False
    return True


class _MapRead(NamedTuple):
    """The entries a hole printing a map whole may hold where it is printed, over the rows of its print, and the
    lines where it may get entries the source does not spell (:attr:`SkeletonHole.entries`, ``entries_open``)."""

    entries: list[MapEntry]
    open: list[int]


def _maps_read(rows: Sequence[PrintRow]) -> dict[str, Optional[_MapRead]]:
    """Each hole of a print's *rows* whose record reads the map it prints whole, by expression: ``None`` where a
    row's record of the same hole does not read it."""
    found: dict[str, _MapRead] = {}
    unread: set[str] = set()
    for row in rows:
        for hole in row.holes:
            if hole.entries is None:
                unread.add(hole.expression)
                continue
            held = found.setdefault(hole.expression, _MapRead([], []))
            held.entries.extend(hole.entries)
            held.open.extend(hole.entries_open)
    return {
        expression: None if expression in unread else read
        for expression, read in found.items()
    }


def _read_all(read: Optional[_MapRead]) -> bool:
    """Whether the record reads every entry a printed map may hold: each key spelled, none a spec's ``valuesMap()``
    (a whole object, its key not read inside it), and none the source does not spell."""
    return (
        read is not None
        and not read.open
        and all(not entry.computed and not entry.row for entry in read.entries)
    )


def _is_map(func: ast.AST, name: str) -> bool:
    """Whether *name* holds a mapping in *func*: bound to a dict's literal or ``dict()``, or an item stored in it."""
    for node in _assignments(func):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            bound = any(
                isinstance(target, ast.Name) and target.id == name for target in targets
            )
            value = node.value
            if bound and (
                isinstance(value, ast.Dict)
                or (
                    isinstance(value, ast.Call)
                    and isinstance(value.func, ast.Name)
                    and value.func.id == "dict"
                )
            ):
                return True
            if any(
                isinstance(target, ast.Subscript)
                and isinstance(target.value, ast.Name)
                and target.value.id == name
                for target in targets
            ):
                return True
    return False


def _on_the_way(
    site: _Site, terms: Sequence[KeyFrom], statements: dict[int, Optional[ast.stmt]]
) -> list[int]:
    """The prints on every path from an SQL arm's line to each run of the statements it fills (*terms*): the row is
    written where the statement runs."""
    runs: list[ast.stmt] = []
    for record in terms:
        ran = _statement_at(site.func, record.statement)
        # A run the arm reaches: on no path, every print would be on every one.
        if (
            ran is not None
            and ran is not site.statement
            and all(ran is not one for one in runs)
            and reaching(ran, [site.statement])
        ):
            runs.append(ran)
    if not runs:
        return []
    return [
        line
        for line, statement in statements.items()
        if statement is not None
        and all(not reaching(run, [site.statement], [statement]) for run in runs)
    ]


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
        # A junction's owner spells the nesting only where the module repeats the name
        # (:func:`calls.spells`), and the prints' owner always does: ``main.FinisherThr.run``.
        nested = pathcond.qualified_name(func)
        if not spells(func, qualname):
            return f"line {line} is in {nested}, not {qualname}"
        statement = _statement_at(func, line)
        if statement is None:
            return f"no statement of {nested} holds line {line}"
        sites.append(_Site(func, f"{file}::{nested}", statement, line))
    return sites


def _agreed(each: list[list[MustName]]) -> list[MustName]:
    """The names every write of a branch has, said to hold the row only where every one says so."""
    first, rest = each[0], each[1:]
    agreed: list[MustName] = []
    for name in first:
        key = (name.line, name.expression, name.column, name.part)
        others = [
            [
                held
                for held in names
                if (held.line, held.expression, held.column, held.part) == key
            ]
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
    kept: Optional["_Kept"] = None,
) -> list[MustName]:
    """The holes of the must prints that read an attribute of *written*, each with whether it holds the arm's row:
    *kept* says whether the attribute holds between the print and the arm, :func:`_unbound_since` by default (a
    print before the arm); :func:`_kept_after` for a print after it."""
    held_since = kept or _unbound_since
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
            same = statement is not None and held_since(
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


def _attribute_keys(
    site: _Site, written: ast.expr, subject: str, identity: dict[str, set[str]]
) -> list[_Key]:
    """The written object's identity attributes, as the arm spells the object: ``job.PandaID``."""
    cls = subject.rpartition(".")[0]
    spelled = ast.unparse(written)
    return [
        _Key(
            f"{cls}.{attribute}",
            ast.parse(f"{spelled}.{attribute}", mode="eval").body,
            site.statement,
        )
        for attribute in sorted(identity.get(cls, ()))
    ]


_TERM = re.compile(r"^\s*(\w+)\s*=\s*(:\w+)\s*$")


def _statement_keys(
    site: _Site,
    held: list[KeyFrom],
    subject: str,
    declarations: dict[str, set[str]],
) -> list[_Key]:
    """The terms naming the rows of the statement an SQL arm fills a bind of, each with the value its bind is
    filled with: ``taskID=:jediTaskID`` and ``varMap[":jediTaskID"] = taskSpec.jediTaskID``.

    A term is one where every place the bind is filled (:attr:`KeySource.line`) fills it with the same expression
    -- a store into the bind map or its literal's entry -- and its column is one of the subject's class, or of its
    table where the subject is a table's (``async_results.status``).  Its value counts where the statement runs.
    """
    cls = subject.rpartition(".")[0]
    columns = {column.lower(): column for column in declarations.get(cls, ())}
    keys: list[_Key] = []
    for record in held:
        term = _TERM.match(record.term)
        column = columns.get(term.group(1).lower()) if term else None
        if (
            term
            and not columns
            and record.table.rpartition(".")[2].lower() == cls.lower()
        ):
            column = term.group(1)
        ran = _statement_at(site.func, record.statement)
        if term is None or column is None or ran is None or not record.sources:
            continue
        filled: list[ast.stmt] = []
        values: list[ast.expr] = []
        for source in record.sources:
            statement = (
                _statement_at(site.func, source.line)
                if source.line is not None
                else None
            )
            value = (
                _bound_value(statement, term.group(2))
                if statement is not None
                else None
            )
            if value is None:
                break
            filled.append(statement)  # type: ignore[arg-type]
            values.append(value)
        else:
            if len({ast.unparse(value) for value in values}) == 1:
                maps = {_map_filled(statement) for statement in filled}
                keys.append(
                    _Key(
                        f"{cls}.{column}",
                        values[0],
                        ran,
                        tuple(filled),
                        term.group(2),
                        maps.pop() if len(maps) == 1 else "",
                    )
                )
    return keys


def _map_filled(statement: ast.stmt) -> str:
    """The bind map *statement* fills an entry of: ``varMap`` for ``varMap[":jediTaskID"] = ...`` or ``varMap =
    {...}``."""
    target = (
        statement.targets[0]
        if isinstance(statement, ast.Assign)
        else getattr(statement, "target", None)
    )
    if isinstance(target, ast.Subscript):
        target = target.value
    return target.id if isinstance(target, ast.Name) else ""


def _bound_value(statement: ast.stmt, bind: str) -> Optional[ast.expr]:
    """What *statement* fills *bind* with: ``varMap[":jediTaskID"] = jediTaskID``, or the entry of a bind map's
    literal ``varMap = {":jediTaskID": jediTaskID}``, annotated or not."""
    if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
        target = statement.targets[0]
    elif isinstance(statement, ast.AnnAssign) and statement.value is not None:
        target = statement.target
    else:
        return None
    if isinstance(target, ast.Subscript):
        key = target.slice
        if isinstance(key, ast.Constant) and key.value == bind:
            return statement.value
        return None
    if isinstance(target, ast.Name) and isinstance(statement.value, ast.Dict):
        for entry, value in zip(
            statement.value.keys, statement.value.values, strict=True
        ):
            if isinstance(entry, ast.Constant) and entry.value == bind:
                return value
    return None


def _key_names(
    site: _Site,
    keys: list[_Key],
    by_attribute: bool,
    lines: list[int],
    statements: dict[int, Optional[ast.stmt]],
    holes: dict[tuple[str, int], list[str]],
    marks: dict[tuple[str, int], list[str]],
    maps_read: dict[tuple[str, int], dict[str, Optional[_MapRead]]],
    after: bool = False,
) -> list[MustName]:
    """The holes of the must prints holding a key of the written row (*keys*), each where the value is.

    A hole of the message, as it spells the key -- :func:`_names` reads those of an attribute arm already --
    or a name bound to it; a hole of the logger's mark the same, the mark being taken where the logger is made
    (:func:`_logger_made`): ``tmp_log = self.create_tagged_logger(comment, f"PandaID={tmp_id}")`` after ``tmp_id
    = job.PandaID``.  Each said to hold the row only where it holds the value the key holds where the write counts
    (:func:`_holds_key`); a hole spelling the key, or a name bound to it, that this reading cannot show holds it is
    said with why (:data:`WHY_KEY_UNSHOWN`), so that a branch with no name holds no hole spelling the key.

    A map printed whole other than the one the statement runs with is read by the entries its print's record holds
    (:attr:`SkeletonHole.entries`): an entry with a spelled key whose value spells the key is a name of the entry
    (part ``bind``, the entry's key as *bind*), said with why -- the entry is not followed from where it is filled
    to the write.

    *after*: the prints come after the write (:attr:`AfterPrint.names`), and a message's hole holds the key where
    nothing changes it from the write to the print (:func:`_holds_key_after`, :func:`_map_holds_key_after`); a
    mark's hole is read as before, where the logger is made.
    """
    holds_key = _holds_key_after if after else _holds_key
    map_holds_key = _map_holds_key_after if after else _map_holds_key
    names: list[MustName] = []
    for line in lines:
        printed = statements.get(line)
        if printed is None:
            continue
        made = _logger_made(site.func, printed)
        for key in keys:
            spelled = ast.unparse(key.expression)
            for expression in dict.fromkeys(holes.get((site.owner, line), ())):
                if by_attribute and expression == spelled:
                    continue
                if key.map and _printed_map(expression) == key.map:
                    held = map_holds_key(site.func, key, printed)
                    names.append(
                        MustName(
                            line=line,
                            expression=expression,
                            column=key.column,
                            row=MUST_ROW_WRITTEN if held else "",
                            why="" if held else WHY_KEY_UNSHOWN,
                            part=MUST_PART_BIND,
                            bind=key.bind,
                        )
                    )
                    continue
                read = maps_read.get((site.owner, line), {}).get(expression)
                if read is not None:
                    for bind in dict.fromkeys(
                        entry.key
                        for entry in read.entries
                        if not entry.computed
                        and not entry.row
                        and _spells_key(site.func, key, entry.expression)
                    ):
                        names.append(
                            MustName(
                                line=line,
                                expression=expression,
                                column=key.column,
                                why=WHY_KEY_UNSHOWN,
                                part=MUST_PART_BIND,
                                bind=bind,
                            )
                        )
                    continue
                if holds_key(site.func, key, printed, expression):
                    names.append(
                        MustName(
                            line=line,
                            expression=expression,
                            column=key.column,
                            row=MUST_ROW_WRITTEN,
                        )
                    )
                elif _spells_key(site.func, key, expression):
                    names.append(
                        MustName(
                            line=line,
                            expression=expression,
                            column=key.column,
                            why=WHY_KEY_UNSHOWN,
                        )
                    )
            if made is None:
                continue
            statement, call = made
            for expression in dict.fromkeys(marks.get((site.owner, line), ())):
                if not any(ast.unparse(node) == expression for node in walk(call)):
                    continue
                if _holds_key(site.func, key, statement, expression):
                    names.append(
                        MustName(
                            line=line,
                            expression=expression,
                            column=key.column,
                            row=MUST_ROW_WRITTEN,
                            part=MUST_PART_MARK,
                        )
                    )
                elif _spells_key(site.func, key, expression):
                    names.append(
                        MustName(
                            line=line,
                            expression=expression,
                            column=key.column,
                            why=WHY_KEY_UNSHOWN,
                            part=MUST_PART_MARK,
                        )
                    )
    # Two keys can name one column -- two terms of the statements a line fills, the same column -- and give one
    # hole the same name twice.
    unique: dict[tuple[int, str, str, str, str, str, str], MustName] = {}
    for name in names:
        unique.setdefault((name.line, name.expression, name.column, name.row, name.why, name.part, name.bind), name)
    return list(unique.values())


def _printed_map(expression: str) -> str:
    """The name of the map a hole prints whole: ``varMap`` for ``str(varMap)``, ``repr(varMap)`` or ``varMap``."""
    try:
        read = ast.parse(expression, mode="eval").body
    except SyntaxError:
        return ""
    return mapentries.printed_map(read)


def _map_holds_key(func: ast.AST, key: _Key, printed: ast.stmt) -> bool:
    """Whether the bind map printed at *printed* is the one an SQL arm's statement runs with, holding the key's
    entry there: the map not bound again between the print and the run, which hands it over; on every path to the
    print, an entry filled after the map was last bound; and none filled again before the run."""
    if not any(
        isinstance(node, ast.Name) and node.id == key.map for node in walk(key.at)
    ):
        return False
    if not _holds(func, key.at, printed, ast.Name(id=key.map, ctx=ast.Load())):
        return False
    fills = list(key.filled)
    bound = []
    for binding in pathcond.bindings(func, key.map):  # type: ignore[arg-type]
        if binding.nested:
            continue
        site = binding.site
        statement = (
            site
            if isinstance(site, (ast.stmt, ast.ExceptHandler))
            else _statement_of(site)
        )
        if statement is not None and all(statement is not fill for fill in fills):
            bound.append(statement)
    entry = [func.args]  # type: ignore[attr-defined]
    present = not reaching(printed, [*entry, *bound], fills)
    return present and not reaching(key.at, fills, [printed])


def _kept_after(func: ast.AST, arm: ast.stmt, printed: ast.stmt, written: ast.expr, attribute: str) -> bool:
    """Whether *written*'s *attribute* holds at *printed*, after *arm*, what the arm left it: nothing that may change
    it (:func:`_changes_of`) between them in the turn (:func:`flow.changed_between`)."""
    sites = _changes_of(func, written, attribute)
    return sites is not None and not flow.changed_between(arm, printed, sites)


def _holds_after(func: ast.AST, write: ast.stmt, printed: ast.stmt, expression: ast.expr) -> bool:
    """:func:`_holds` from a write to a print after it: an attribute chain as :func:`_kept_after` reads one, a name
    where none of its bindings is between them; anything else is not read."""
    if isinstance(expression, ast.Attribute):
        return _kept_after(func, write, printed, expression.value, expression.attr)
    if isinstance(expression, ast.Name):
        sites = _rebindings(func, expression.id, printed)
        return not flow.changed_between(write, printed, sites)
    return False


def _holds_key_after(func: ast.AST, key: _Key, at: ast.stmt, expression: str) -> bool:
    """:func:`_holds_key` for a hole read at *at*, after the write: the key itself unchanged from the write to *at*;
    a name bound to it once, that binding the only one reaching *at* and taking what the key holds at the write --
    before the write, the key unchanged from it to the write, after, from the write to it; what the key is bound
    to once, as :func:`_bound_from` reads it either way round.  And every fill of an SQL arm's bind holding it
    unchanged to the run."""
    try:
        read = ast.parse(expression, mode="eval").body
    except SyntaxError:
        return False
    spelled = ast.unparse(key.expression)
    if ast.unparse(read) == spelled:
        held = _holds_after(func, key.at, at, key.expression)
    elif isinstance(read, ast.Name) and _single_bindings(func, read.id, spelled):
        held = _bound_to_after(func, key.at, at, read.id, key.expression)
    elif isinstance(key.expression, (ast.Name, ast.Attribute)):
        held = _bound_from(func, key, at, read)
    else:
        return False
    # Each fill comes before the run: what it filled lasts to the run where nothing between changes it -- not on
    # every way into the run, which another arm's fill of the same bind reaches.
    return held and all(_holds_after(func, one, key.at, key.expression) for one in key.filled)


def _bound_to_after(func: ast.AST, write: ast.stmt, at: ast.stmt, name: str, expression: ast.expr) -> bool:
    """:func:`_bound_to` for *name* read at *at* after *write*: bound to *expression* once, that binding the only one
    reaching *at*, and the binding taking what *expression* holds at the write."""
    spelled = ast.unparse(expression)
    sites = _rebindings(func, name, write, entry=True)
    made = [
        site
        for site in sites
        if isinstance(site, ast.Assign)
        and len(site.targets) == 1
        and isinstance(site.targets[0], ast.Name)
        and site.targets[0].id == name
        and ast.unparse(site.value) == spelled
    ]
    if len(made) != 1:
        return False
    (bound,) = made
    if reaching(at, [site for site in sites if site is not bound], [bound]):
        return False
    if reaching(bound, [write], entered=[write], once=flow.turn_loops(write, bound)):
        return _holds_after(func, write, bound, expression)
    return _holds(func, write, bound, expression)


def _map_holds_key_after(func: ast.AST, key: _Key, printed: ast.stmt) -> bool:
    """:func:`_map_holds_key` for a print after the run: the bind map the statement ran with, neither bound again nor
    any of the key's fills run again between the run and the print."""
    if not any(isinstance(node, ast.Name) and node.id == key.map for node in walk(key.at)):
        return False
    sites: list[ast.AST] = list(key.filled)
    for binding in pathcond.bindings(func, key.map):  # type: ignore[arg-type]
        if binding.nested:
            continue
        site = binding.site
        statement = site if isinstance(site, (ast.stmt, ast.ExceptHandler)) else _statement_of(site)
        if statement is not None:
            sites.append(statement)
    return not flow.changed_between(key.at, printed, sites)


def _spells_key(func: ast.AST, key: _Key, expression: str) -> bool:
    """Whether a hole *expression* reads *key*'s expression: spelled so, a name bound to it somewhere in *func*
    (``tmp_id = job.PandaID``), or what the key, a name or an attribute chain, is bound to (``pandaID = job.PandaID``
    filling ``:pandaID``, the hole ``job.PandaID``; ``tmpTaskSpec.jediTaskID = jediTaskID``, the hole
    ``jediTaskID``)."""
    try:
        read = ast.parse(expression, mode="eval").body
    except SyntaxError:
        return False
    spelled = ast.unparse(key.expression)
    if ast.unparse(read) == spelled:
        return True
    if isinstance(read, ast.Name) and _single_bindings(func, read.id, spelled):
        return True
    return isinstance(key.expression, (ast.Name, ast.Attribute)) and bool(
        _single_bindings(func, ast.unparse(key.expression), ast.unparse(read))
    )


#: The assignments of each function read while :func:`attach` runs, by its id; each with its function, so that the
#: id is not reused while the entry lives.
_ASSIGNMENTS: dict[int, tuple[ast.AST, list[ast.stmt]]] = {}


def _assignments(func: ast.AST) -> list[ast.stmt]:
    """Every assignment in *func*, read once."""
    if id(func) not in _ASSIGNMENTS:
        _ASSIGNMENTS[id(func)] = (
            func,
            [
                node
                for node in walk(func)
                if isinstance(node, (ast.Assign, ast.AnnAssign))
            ],
        )
    return _ASSIGNMENTS[id(func)][1]


#: Each function's assignments of one name or attribute chain, by the target and the value as spelled, read while
#: :func:`attach` runs; kept with the function as :data:`_ASSIGNMENTS` is.
_SINGLES: dict[int, tuple[ast.AST, dict[tuple[str, str], list[ast.Assign]]]] = {}


def _single_bindings(func: ast.AST, target: str, spelled: str) -> list[ast.Assign]:
    """The statements binding *target*, a name or an attribute chain, alone to an expression spelled *spelled*:
    ``tmp_id = job.PandaID``, ``tmpTaskSpec.jediTaskID = jediTaskID``.  A hole is asked of every key, so each
    function's are spelled once."""
    if id(func) not in _SINGLES:
        found: dict[tuple[str, str], list[ast.Assign]] = {}
        for node in _assignments(func):
            if (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], (ast.Name, ast.Attribute))
            ):
                found.setdefault(
                    (ast.unparse(node.targets[0]), ast.unparse(node.value)), []
                ).append(node)
        _SINGLES[id(func)] = (func, found)
    return list(_SINGLES[id(func)][1].get((target, spelled), ()))


def _logger_made(
    func: ast.AST, printed: ast.stmt
) -> Optional[tuple[ast.stmt, ast.Call]]:
    """Where the logger *printed* writes through is made, when it is a local this function binds once by a call:
    the statement and the call, whose arguments hold the mark's holes."""
    for node in walk(printed):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in LOG_METHODS
            and isinstance(node.func.value, ast.Name)
        ):
            bound = pathcond.single_definition(func, node.func.value.id)  # type: ignore[arg-type]
            statement = _statement_of(bound) if isinstance(bound, ast.Call) else None
            if statement is not None:
                return statement, bound  # type: ignore[return-value]
    return None


#: :func:`_holds_key`'s answers while :func:`attach` runs, by the function, the key's expression as spelled, where it
#: counts and the statements filling it, the statement read at and the expression read -- 12092 asked 218634 times
#: on pin s; each with the nodes its ids are of, so that no id is reused while the entry lives.
_HOLDS_KEY: dict[tuple, tuple[tuple[ast.AST, ...], bool]] = {}


def _holds_key(func: ast.AST, key: _Key, at: ast.stmt, expression: str) -> bool:
    """Whether *expression*, read at *at*, holds what *key*'s expression holds where the write counts -- and so
    does every statement filling an SQL arm's bind with it: the key itself, a name bound to it once
    (:func:`_bound_to`), or what the key, a name or an attribute chain, is bound to once (:func:`_bound_from`).
    Read once for each such question while :func:`attach` runs: of the key, only its expression's spelling, where
    it counts and its fills are read."""
    asked = (
        id(func), ast.unparse(key.expression), id(key.at), tuple(id(one) for one in key.filled), id(at), expression
    )
    kept = _HOLDS_KEY.get(asked)
    if kept is None:
        kept = _HOLDS_KEY[asked] = ((func, key.at, *key.filled, at), _read_holds_key(func, key, at, expression))
    return kept[1]


def _read_holds_key(func: ast.AST, key: _Key, at: ast.stmt, expression: str) -> bool:
    try:
        read = ast.parse(expression, mode="eval").body
    except SyntaxError:
        return False
    if ast.unparse(read) == ast.unparse(key.expression):
        held = _holds(func, key.at, at, key.expression)
    elif isinstance(read, ast.Name) and _single_bindings(
        func, read.id, ast.unparse(key.expression)
    ):
        held = _bound_to(func, key.at, at, read.id, key.expression)
    elif isinstance(key.expression, (ast.Name, ast.Attribute)):
        held = _bound_from(func, key, at, read)
    else:
        return False
    return held and all(_holds(func, key.at, one, key.expression) for one in key.filled)


def _holds(
    func: ast.AST, reference: ast.stmt, at: ast.stmt, expression: ast.expr
) -> bool:
    """Whether, on every path into *reference*, *expression* holds there what it held at *at*: an attribute chain
    as :func:`_unbound_since` reads one, a name where none of its bindings reaches *reference* without passing
    *at*; anything else is not read."""
    if isinstance(expression, ast.Attribute):
        return _unbound_since(
            func, reference, at, expression.value, expression.attr, entered=False
        )
    if isinstance(expression, ast.Name):
        return not reaching(
            reference, _rebindings(func, expression.id, reference, at), [at]
        )
    return False


def _bound_to(
    func: ast.AST, reference: ast.stmt, at: ast.stmt, name: str, expression: ast.expr
) -> bool:
    """Whether *name*, read at *at*, holds what *expression* holds at *reference*: bound to it once -- ``tmp_id =
    job.PandaID`` -- that binding the only one reaching *at*, and the expression holding the same from it to
    *reference*.  What *name* is bound to after *at* does not matter: the print, or the mark the logger was made
    with, has taken the value."""
    spelled = ast.unparse(expression)
    sites = _rebindings(func, name, reference, entry=True)
    made = [
        site
        for site in sites
        if isinstance(site, ast.Assign)
        and len(site.targets) == 1
        and isinstance(site.targets[0], ast.Name)
        and site.targets[0].id == name
        and ast.unparse(site.value) == spelled
    ]
    if len(made) != 1:
        return False
    (bound,) = made
    others = [site for site in sites if site is not bound]
    return not reaching(at, others, [bound]) and _holds(
        func, reference, bound, expression
    )


def _bound_from(func: ast.AST, key: _Key, at: ast.stmt, read: ast.expr) -> bool:
    """Whether *read*, at *at*, holds what *key*'s expression holds where the write counts, the expression bound to
    *read* once -- ``pandaID = job.PandaID`` under the mark reading ``job.PandaID``; ``tmpTaskSpec.jediTaskID =
    jediTaskID`` under ``< jediTaskID={jediTaskID} >`` -- that binding the only one reaching the write (for a chain,
    the chain unchanged from it to the write), and *read* holding the same at the binding and at *at*: from whichever
    of the two runs first to the other, nothing binds it.  Both ways would ask the impossible of a loop that binds
    *read* and holds the binding and the mark, as ``TaskCommando.runImpl``'s ``for jediTaskID, ... in taskList``."""
    made = _single_bindings(func, ast.unparse(key.expression), ast.unparse(read))
    if len(made) != 1:
        return False
    (bound,) = made
    if not (_holds(func, bound, at, read) or _holds(func, at, bound, read)):
        return False
    if isinstance(key.expression, ast.Name):
        others = [
            site
            for site in _rebindings(func, key.expression.id, key.at, entry=True)
            if site is not bound
        ]
        return not reaching(key.at, others, [bound])
    return _holds(func, key.at, bound, key.expression)


def _rebindings(
    func: ast.AST,
    name: str,
    reference: ast.stmt,
    at: Optional[ast.stmt] = None,
    entry: bool = False,
) -> list[ast.AST]:
    """Where *name* is bound in *func* but at *at* -- with *entry*, its entry too for a parameter -- less a binding
    that cannot be on one run with *reference* (:func:`pathcond.compatible`): ``tmp_id = None`` under ``if job is
    None:``, before an arm the same test leaves at.  The entry binds before everything, so whether a value read at
    *at* is still the one at *reference* asks only what is bound after it."""
    arguments = func.args  # type: ignore[attr-defined]
    parameters = {
        one.arg
        for one in (
            *arguments.posonlyargs,
            *arguments.args,
            *arguments.kwonlyargs,
            *filter(None, (arguments.vararg, arguments.kwarg)),
        )
    }
    sites: list[ast.AST] = [arguments] if entry and name in parameters else []
    for binding in pathcond.bindings(func, name):
        if binding.nested or isinstance(binding.site, ast.arg):
            continue
        site = (
            binding.site
            if isinstance(binding.site, (ast.stmt, ast.ExceptHandler))
            else _statement_of(binding.site)
        )
        if (
            site is not None
            and site is not at
            and pathcond.compatible(site, reference, current=True)
        ):
            sites.append(site)
    return sites


def _unbound_since(
    func: ast.AST,
    arm: ast.stmt,
    printed: ast.stmt,
    written: ast.expr,
    attribute: str,
    entered: bool = True,
) -> bool:
    """Whether, on every path into *arm*, *printed* ran after *written* and its *attribute* were last bound.

    The writes are the name's bindings in the function, every store to the
    chain ``job``, ``job.PandaID`` (``self.job`` and ``self.job.PandaID`` for
    ``self.job``), every call made on an object of the chain or handed one
    that may change the value (:func:`_may_change`) --
    ``self.ppEventServiceJob(job, ...)`` between ``archiveJob``'s first print
    and its arms may set ``job``'s, and ``self.retry()`` may rebind
    ``self.job``; ``job.valuesMap()`` sets none -- and the call's own binding
    of the parameters; the print kills them.  None of them reaching the arm
    means the print saw what the arm sees.  A store through another name for
    the object (``j = job; j.PandaID = 3``) in this function is not seen.
    Without *entered*, *printed* need not be on every path: only whether,
    where it ran, the value lasts to the arm.
    """
    sites = _changes_of(func, written, attribute)
    if sites is None:
        return False
    if entered:
        sites = [func.args, *sites]  # type: ignore[attr-defined]
    return not reaching(arm, [site for site in sites if site is not printed], [printed])


#: :func:`_changes_of`'s answers while :func:`attach` runs, by the function's id, the chain as spelled and the
#: attribute -- 744 chains asked 32340 times on pin s; each with its function, as :data:`_ASSIGNMENTS` keeps it.
_CHANGES: dict[tuple[int, str, str], tuple[ast.AST, Optional[tuple[ast.AST, ...]]]] = {}


def _changes_of(func: ast.AST, written: ast.expr, attribute: str) -> Optional[list[ast.AST]]:
    """Where *written*'s *attribute* can change in *func*, as :func:`_unbound_since` reads it: the bindings of the
    chain's root name, every store to the chain, every call on an object of the chain or handed one that may change
    the value.  None where the chain's root is no name.  Read once for each function, chain and attribute while
    :func:`attach` runs: the answer is the chain's, by its names -- :func:`changes.may_change` reads it so too."""
    key = (id(func), ast.unparse(written), attribute)
    kept = _CHANGES.get(key)
    if kept is None:
        found = _read_changes(func, written, attribute)
        kept = _CHANGES[key] = (func, None if found is None else tuple(found))
    return None if kept[1] is None else list(kept[1])


def _read_changes(func: ast.AST, written: ast.expr, attribute: str) -> Optional[list[ast.AST]]:
    chain = [ast.unparse(written) + "." + attribute]
    value = ast.Attribute(value=written, attr=attribute, ctx=ast.Load())
    root: ast.expr = written
    while isinstance(root, ast.Attribute):
        chain.append(ast.unparse(root))
        root = root.value
    if not isinstance(root, ast.Name):
        return None
    sites: list[Optional[ast.AST]] = []
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
        elif (
            isinstance(node, ast.Call)
            and _given(node, holding)
            and _may_change(func, node, value)
        ):
            sites.append(_statement_of(node))
    return [site for site in sites if site is not None]


def _given(call: ast.Call, holding: set[str]) -> bool:
    """Whether *call* is made on one of *holding* or is handed one of them."""
    if isinstance(call.func, ast.Attribute) and ast.unparse(call.func.value) in holding:
        return True
    given = [*call.args, *(keyword.value for keyword in call.keywords)]
    return any(
        ast.unparse(value.value if isinstance(value, ast.Starred) else value) in holding
        for value in given
    )


#: While :func:`attach` reads with the corpus's calls: them, and the file of each function of the build.
_ACROSS: Optional[tuple["Calls", dict[int, str]]] = None


def _may_change(func: ast.AST, call: ast.Call, value: ast.expr) -> bool:
    """Whether *call*, in *func*, may change what *value* holds (:func:`changes.may_change`); every call may where
    the corpus's calls are not read."""
    if _ACROSS is None:
        return True
    calls, files = _ACROSS
    file = files.get(id(func))
    return (
        file is None or changes.may_change(calls, file, func, call, value) is not None
    )


def _written_object(statement: ast.stmt, subject: str) -> Optional[ast.expr]:
    """``job`` for ``job.jobStatus = ...`` when *subject* is ``JobSpec.jobStatus``: the object the arm writes to."""
    attribute = subject.rpartition(".")[2]
    if isinstance(statement, ast.Assign):
        targets = list(statement.targets)
    elif isinstance(statement, (ast.AnnAssign, ast.AugAssign)):
        targets = [statement.target]
    else:
        return None
    # ``(original_task_spec.userName,) = tmp_res``: each element of a tuple or list target is a target.
    flat: list[ast.expr] = []
    while targets:
        target = targets.pop(0)
        if isinstance(target, (ast.Tuple, ast.List)):
            targets[:0] = target.elts
        else:
            flat.append(target)
    found = [
        target.value
        for target in flat
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


#: The answers :func:`attach` keeps while it runs (:data:`_CHANGES`, :data:`_OBJECT_SITES`, :data:`_HOLDS_KEY`),
#: emptied as it starts and ends: :func:`_changes_of` reads the corpus's calls only inside it.
_KEPT: tuple[dict, ...] = (_CHANGES, _OBJECT_SITES, _HOLDS_KEY)
