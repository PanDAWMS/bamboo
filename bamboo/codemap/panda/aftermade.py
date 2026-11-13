"""Arms that cannot leave a value other than the row's on a row made before them (AMW, the querier's PW-v).

A reader comparing the value a row holds with the value an arm sets asks which arms can have changed the row after
it was made.  An arm is not counted (:attr:`~bamboo.codemap.models.AfterMade.counted` false) where the records prove
one of these; every other arm read is counted, with its reasons:

* **No row yet** (:data:`~bamboo.codemap.models.AFTER_MADE_NO_ROW_YET`), an arm assigning on an object.  At each
  statement of the arm:

  - the object is a local its function binds to a new object of the subject's class at every binding of the name
    (:func:`bamboo.codemap.panda.offrow._made_here`) -- not a row read, a parameter or a copy;
  - no place of the function that may make the object's row can come before the arm
    (:meth:`bamboo.codemap.objects._Order.meets`, a loop's new object read as new): a write of a table of the class
    taking the object, whole or by a column of it; a call handed the object whose run may write one, or is not read
    to its end; a hole the object's follow meets there;
  - nothing gives the object a key: no assignment of a key attribute of the class (:attr:`TableRecord.identity_attrs`)
    or of every attribute (``*``) where the object is followed (:attr:`~bamboo.codemap.objects.ObjectReach.assigned`,
    ``pack`` from a row among them; a constructor's own defaults are not).

  So the row the value goes to is made after the arm.  It rests on two things of the class, read over the corpus
  once; where one breaks, no arm of the class is "no row yet"
  (:data:`~bamboo.codemap.models.AFTER_MADE_KEY_ELSEWHERE`, the breaking places in *elsewhere*):

  - K1, a new object gets no old key later: every arm of a key attribute on an object not made new in its function
    sets a new key -- the object is one an ``INSERT`` of that function makes (:attr:`KeyFrom.new_row`), or the value
    is a sequence's (:data:`~bamboo.codemap.models.KEY_SEQUENCE`), read there or returned by the function a call
    binding the value's name lands at -- or ``None``;
  - K2, a whole object is written to its own row: every ``UPDATE`` writing an object of the class whole, or by its
    changed attributes, names its row by that object's key (a KeyFrom of the statement on it).

* **New row** (:data:`~bamboo.codemap.models.AFTER_MADE_NEW_ROW`), an arm with no object (an SQL arm's bind): every
  write of its function of a table of the class naming the column, or not read, is an ``INSERT`` making its row
  (:attr:`DbWrite.new_row`).
* **Same value** (:data:`~bamboo.codemap.models.AFTER_MADE_SAME_VALUE_WHERE`): the branch's row precondition asks the
  column for the very value the arm sets.

An arm ``off_row`` holds for is not read (its row is never written from it).  Nor is an arm of a class no table holds
(:func:`~bamboo.codemap.panda.objects.column_classes`): the reach record keeps no assignments of it, so "no key is
set" cannot be seen.

**What it rests on.**  What :mod:`bamboo.codemap.panda.offrow` and :func:`bamboo.codemap.objects.follow` rest on; and
that a value written after the arm on a row made after it is the object's value then, which differs from the arm's
only where another arm changed the object -- one running on an object with a row, counted here.
"""

from __future__ import annotations

import ast
import re
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Optional

from bamboo.codemap import calls as C
from bamboo.codemap import objects
from bamboo.codemap.models import (
    AFTER_MADE_KEY_ELSEWHERE,
    AFTER_MADE_KEY_SET,
    AFTER_MADE_MADE_BEFORE,
    AFTER_MADE_MOVED,
    AFTER_MADE_NEW_ROW,
    AFTER_MADE_NO_ROW_YET,
    AFTER_MADE_OPEN,
    AFTER_MADE_ROW_OBJECT,
    AFTER_MADE_SAME_VALUE_WHERE,
    AFTER_MADE_UPDATE,
    BIND_RESULT,
    KEY_SEQUENCE,
    AfterMade,
    Branch,
    DbWrite,
    JunctionNode,
    MapFragment,
    SourceModule,
)
from bamboo.codemap.panda import mustprints, offrow

if TYPE_CHECKING:
    from bamboo.codemap.reach import ReachGraph

#: How a constructor's own assignments are recorded: the defaults of a new object, not a key given to it.
_CONSTRUCT = "construct"
#: Every attribute, as an assignment by a computed name records it.
_EVERY = "*"


def attach(
    fragment: MapFragment,
    modules: list[SourceModule],
    calls: C.Calls,
    record: Optional[ReachGraph],
) -> int:
    """Set :attr:`Branch.after_made` on every branch of *fragment* of a class a table holds
    (:func:`~bamboo.codemap.panda.objects.column_classes`) that ``off_row`` does not hold for; returns how many."""
    reader = _Reader(fragment, modules, calls, record)
    read = 0
    for junction in fragment.junctions:
        cls = junction.subject.rpartition(".")[0]
        if not reader.tables_of(cls):
            continue
        for branch in junction.branches:
            if branch.off_row is not None:
                continue
            branch.after_made = reader.after_made(junction, branch)
            read += 1
    return read


class _Reader:
    def __init__(
        self,
        fragment: MapFragment,
        modules: list[SourceModule],
        calls: C.Calls,
        record: Optional[ReachGraph],
    ):
        self.base = offrow._Reader(fragment, modules, calls)
        self.rows = self.base.rows
        self.calls, self.record = calls, record
        self.junctions = fragment.junctions
        self.keys: dict[str, set[str]] = defaultdict(set)
        for table in fragment.tables:
            for one in table.identity_attrs:
                cls, _sep, attr = one.rpartition(".")
                if cls == table.spec_class:
                    self.keys[cls].add(attr)
        #: ``(file, function name)`` -> its KeyFroms: how the corpus's KeyFroms name their function.
        self.key_froms: dict[tuple[str, str], list] = defaultdict(list)
        for key_from in fragment.key_froms:
            file, _sep, name = key_from.owner.partition("::")
            self.key_froms[(file, name.rpartition(".")[2])].append(key_from)
        self._broken: dict[str, list[str]] = {}
        #: sqlwrite's ``_Filled`` per function and its ``_Reading``, made on the first sequence asked.
        self._filled: dict[int, Any] = {}
        self._reading: Any = None
        self._follows: dict[tuple[str, str], objects.ObjectReach] = {}

    def tables_of(self, cls: str) -> set[str]:
        return {table for table, of in self.base.classes.items() if of == cls}

    def follow(self, owner: str, path: str) -> objects.ObjectReach:
        if (owner, path) not in self._follows:
            self._follows[(owner, path)] = objects.follow(self.rows, owner, path)
        return self._follows[(owner, path)]

    def _of_class(self, write: DbWrite, tables: set[str]) -> bool:
        return write.table.rpartition(".")[2].lower() in tables

    # -- one arm -------------------------------------------------------------------------------------------------- #

    def after_made(self, junction: JunctionNode, branch: Branch) -> AfterMade:
        cls, _sep, attr = junction.subject.rpartition(".")
        tables = self.tables_of(cls)
        sites = mustprints._sites(junction, branch, self.base.trees)
        if isinstance(sites, str) or not sites or branch.line is None:
            return AfterMade(counted=True, why=[AFTER_MADE_OPEN])
        why: set[str] = set()
        lines: set[int] = set()
        sql_rows = []
        objects_seen = False
        for site in sites:
            owner = self.base.names.of(site.func)
            row = self.rows.get(owner) if owner is not None else None
            if owner is None or row is None:
                return AfterMade(counted=True, why=[AFTER_MADE_OPEN])
            span = range(
                site.statement.lineno,
                (site.statement.end_lineno or site.statement.lineno) + 1,
            )
            paths = sorted(
                {
                    one.path
                    for one in row.assigns
                    if one.line in span and attr in one.attrs
                }
            )
            if not paths:
                sql_rows.append(row)
                continue
            objects_seen = True
            for path in paths:
                w, ls = self._no_row_yet(
                    owner, row, path, branch.line, cls, attr, tables
                )
                why |= w
                lines |= ls
        if objects_seen and sql_rows:
            why.add(AFTER_MADE_OPEN)
        same = _same_value(branch, attr)
        if not objects_seen:
            kinds = [
                write
                for row in sql_rows
                for write in row.writes
                if self._of_class(write, tables)
                and (
                    write.unread
                    or any(
                        column.column.lower() == attr.lower()
                        for column in write.columns
                    )
                )
            ]
            if kinds and all(
                write.kind == "insert" and write.new_row for write in kinds
            ):
                return AfterMade(counted=False, by=AFTER_MADE_NEW_ROW)
            if same:
                return AfterMade(counted=False, by=AFTER_MADE_SAME_VALUE_WHERE)
            for write in kinds:
                why.add(
                    AFTER_MADE_UPDATE if write.kind != "insert" else AFTER_MADE_MOVED
                )
                lines.update(write.lines[:1])
            if not kinds:
                why.add(AFTER_MADE_OPEN)
            return AfterMade(counted=True, why=sorted(why), lines=sorted(lines))
        if not why:
            broken = self.broken(cls)
            if not broken:
                return AfterMade(counted=False, by=AFTER_MADE_NO_ROW_YET)
            return AfterMade(
                counted=True, why=[AFTER_MADE_KEY_ELSEWHERE], elsewhere=broken
            )
        if same:
            return AfterMade(counted=False, by=AFTER_MADE_SAME_VALUE_WHERE)
        return AfterMade(counted=True, why=sorted(why), lines=sorted(lines))

    def _no_row_yet(
        self,
        owner: str,
        row,
        path: str,
        line: int,
        cls: str,
        attr: str,
        tables: set[str],
    ) -> tuple[set[str], set[int]]:
        """The reasons the object at *path* of *owner* may have a row at *line* -- empty where it has none yet."""
        why: set[str] = set()
        lines: set[int] = set()
        if not offrow._made_here(row, path, cls):
            why.add(AFTER_MADE_ROW_OBJECT)
        own = self.follow(owner, path)
        keyed = [
            site
            for key in self.keys.get(cls, set()) | {_EVERY}
            for site in own.assigned.get(key, [])
            if site.how != _CONSTRUCT
        ]
        if keyed:
            why.add(AFTER_MADE_KEY_SET)
            lines.update(site.line for site in keyed if site.owner == owner)
        making: set[int] = set()
        for write in row.writes:
            if self._of_class(write, tables) and (
                (write.whole and objects._under(path, write.whole) is not None)
                or any(
                    value.path and objects._under(path, value.path) is not None
                    for value in write.columns
                )
            ):
                making.update(write.lines[:1])
        for call in row.call_bindings:
            if (
                not call.path
                or objects._under(path, call.path) is None
                or call.to not in self.rows
            ):
                continue
            reach = self.follow(call.to, call.parameter)
            if (
                reach.holes
                or reach.cut
                or reach.capped
                or self._reaches_write(reach, tables)
            ):
                making.add(call.line)
        making.update(site.line for site in own.holes if site.owner == owner)
        if making:
            order = objects._Order(self.rows, [owner])
            held = {owner: list(own.held.get(owner, [path]))}
            met = [
                one
                for one in sorted(making)
                if order.meets(
                    objects.Site(owner, one, "make"), owner, [line], held, attr, False
                )
            ]
            if met:
                why.add(AFTER_MADE_MADE_BEFORE)
                lines.update(met)
        return why, lines

    def _reaches_write(self, reach: objects.ObjectReach, tables: set[str]) -> bool:
        for at, held in reach.held.items():
            held_set = set(held)
            for write in self.rows[at].writes:
                if not self._of_class(write, tables):
                    continue
                if write.unread or (
                    write.whole and objects._holds(held_set, write.whole)
                ):
                    return True
                if any(
                    value.path
                    and any(
                        objects._under(one, value.path) is not None for one in held_set
                    )
                    for value in write.columns
                ):
                    return True
        return False

    # -- the class: K1 and K2 ------------------------------------------------------------------------------------ #

    def broken(self, cls: str) -> list[str]:
        """Where K1 or K2 breaks for *cls*, as ``owner:line``, sorted; empty where both hold."""
        if cls not in self._broken:
            self._broken[cls] = sorted(set(self._k1(cls)) | set(self._k2(cls)))
        return self._broken[cls]

    def _k1(self, cls: str) -> list[str]:
        found = []
        keys = self.keys.get(cls, set())
        for junction in self.junctions:
            j_cls, _sep, attr = junction.subject.rpartition(".")
            if j_cls != cls or attr not in keys:
                continue
            for branch in junction.branches:
                if branch.off_row is not None or branch.outcome in ("None", "none"):
                    continue
                sites = mustprints._sites(junction, branch, self.base.trees)
                if isinstance(sites, str):
                    continue
                for site in sites:
                    owner = self.base.names.of(site.func)
                    row = self.rows.get(owner) if owner is not None else None
                    if owner is None or row is None:
                        continue
                    span = range(
                        site.statement.lineno,
                        (site.statement.end_lineno or site.statement.lineno) + 1,
                    )
                    for path in sorted(
                        {
                            one.path
                            for one in row.assigns
                            if one.line in span and attr in one.attrs
                        }
                    ):
                        if offrow._made_here(row, path, cls) or self._new_key(
                            owner, row, path, site, cls
                        ):
                            continue
                        found.append(f"{owner}:{site.statement.lineno}")
        return found

    def _new_key(self, owner: str, row, path: str, site, cls: str) -> bool:
        """Whether the key the statement of *site* sets on *path* is a new one."""
        file, _sep, qual = owner.partition("::")
        root = objects._root(path).partition("@")[0]
        tables = self.tables_of(cls)
        for one in self.key_froms.get((file, qual.rpartition(".")[2]), []):
            if (
                one.new_row
                and one.table.rpartition(".")[2].lower() in tables
                and objects._root(one.object or "").partition("@")[0] == root
            ):
                return True
        statement = site.statement
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
            return False
        if self._sequence(file, site.func, statement.value, statement):
            return True
        name = _held_name(statement.value)
        bound = [binding for binding in row.bindings if binding.name == name]
        if name is None or not bound:
            return False
        return all(
            binding.how == BIND_RESULT
            and binding.to
            and all(self._returns_sequence(to) for to in binding.to)
            for binding in bound
        )

    def _returns_sequence(self, owner: str) -> bool:
        """Whether every ``return`` of *owner*'s function giving something gives a sequence's values, and one does."""
        if owner not in self.base.funcs:
            return False
        file, func = self.base.funcs[owner]
        found = False
        for node in C.own_nodes(func):
            if not isinstance(node, ast.Return) or node.value is None:
                continue
            value = node.value
            if isinstance(
                value, (ast.List, ast.Tuple, ast.Set, ast.Dict)
            ) and not getattr(value, "elts", getattr(value, "keys", None)):
                continue
            if isinstance(value, ast.Constant) and value.value is None:
                continue
            held = _held_node(value)
            if not self._sequence(file, func, value, node) and (
                held is None
                or held is value
                or not self._sequence(file, func, held, node)
            ):
                return False
            found = True
        return found

    def _sequence(self, file: str, func, value: ast.AST, at: ast.AST) -> bool:
        """Whether *value* at *at* in *func* holds a sequence's next values its function selects."""
        if self.record is None:
            return False
        from bamboo.codemap.panda.recognizers import sqlwrite

        if self._reading is None:
            self._reading = sqlwrite._Reading(
                self.calls, {junction.owner for junction in self.junctions}, self.record
            )
        filled = self._filled.get(id(func))
        if filled is None:
            filled = self._filled[id(func)] = sqlwrite._Filled(
                self._reading, file, func
            )
        return any(source.kind == KEY_SEQUENCE for source in filled.sources(value, at))

    def _k2(self, cls: str) -> list[str]:
        found = []
        tables = self.tables_of(cls)
        for owner, row in self.rows.items():
            file, _sep, qual = owner.partition("::")
            froms = self.key_froms.get((file, qual.rpartition(".")[2]), [])
            for write in row.writes:
                if (
                    write.kind != "update"
                    or not (write.whole or write.changed)
                    or not self._of_class(write, tables)
                ):
                    continue
                root = (
                    objects._root(write.whole).partition("@")[0] if write.whole else ""
                )
                named = {
                    objects._root(one.object).partition("@")[0]
                    for one in froms
                    if one.object and one.statement in write.lines
                }
                if not root or root not in named:
                    found.append(f"{owner}:{write.lines[0] if write.lines else 0}")
        return found


def _same_value(branch: Branch, attr: str) -> bool:
    """Whether the branch's row precondition asks the column for the value the arm sets."""
    outcome = str(branch.outcome or "")
    if not outcome:
        return False
    pattern = re.compile(
        rf"\s*{re.escape(attr)}\s*=\s*'?{re.escape(outcome)}'?\s*", re.IGNORECASE
    )
    return any(pattern.fullmatch(one) for one in branch.row_precondition)


def _held_name(value: ast.AST) -> Optional[str]:
    """The name *value* takes an element or a conversion of -- ``ids[i]``, ``int(val)``, ``sorted(ids)`` -- or is;
    ``None`` for any other value."""
    held = _held_node(value)
    return held.id if held is not None else None


def _held_node(value: ast.AST) -> Optional[ast.Name]:
    while True:
        if isinstance(value, ast.Name):
            return value
        if isinstance(value, ast.Subscript):
            value = value.value
        elif (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and len(value.args) == 1
        ):
            value = value.args[0]
        else:
            return None
