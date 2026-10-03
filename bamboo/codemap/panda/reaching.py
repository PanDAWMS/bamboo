"""Which writes can be what a statement reads: reaching definitions, one variable at a time.

The question every reader of a local asks -- the walk explaining a name, the
SQL slice asking which bind a run takes, the crossing asking whether the rows
that arrived are still in the name a call takes -- is the same one: of these
writes, which can still be what this statement reads?  It is answered here by
following the function's structure, the way it runs.

**Why structure and not a comparison of two writes.**  The rule this replaces
cancelled a write when a later one sat between it and the read, under no loop
the read was outside of and under no condition the read was not under.  Each
clause was a correct observation and the sum was wrong in both directions.
Over every read of a name with two or more writes in the corpus:

* It kept writes that cannot be read.  The loop's way round was invisible to
  it -- a write at the foot of a body reaching the head with the overwrite at
  the head in between -- and so were branches that cover the read between
  them (``if``/``else`` both assigning, an ``elif`` chain ending in
  ``return``).  24,898 and 344 (read, write) pairs, and noise in one walk in
  five.
* It dropped writes that can be.  ``retVal = False; try: ...; retVal = True;
  except: return retVal`` answered ``True`` on the path where the ``try``
  failed; ``match`` cases cancelled one another; a test spelt the same as an
  earlier one, on a name rebound in between, was read as the same condition
  (``JobSplitter.doSplit``'s ``maxSize``, ``getOriginalConsumers`` deciding
  which table to read).  50 pairs, wrong answers in 183 walks.

**What it gives up.**  The flow is path-insensitive: every branch both ways,
every loop any number of times, including none.  The old rule knew that two
``if taskSpec.useHS06():`` on an unchanged ``taskSpec`` take the same arm, and
this does not -- 31 pairs where a value is offered that only an impossible
path delivers.  That is the safe side: a value offered in excess is one a
later step can rule out, and a value dropped is one nothing brings back.  The
loop that runs no times is real Python (``i = -1; for i in items: ...; use(i)``)
and stays.  :func:`pathcond.can_reach` still filters the answer by the
conditions on the write and the read, as before.

**One variable at a time, and the caller says what it is.**  The writes are
handed in, not found: a name's assignments, a map's key, the names a row
arrived in.  So a statement either is one of them or is not, and the state is
which of them can be live -- a set of bits.  Two conventions carry over from
the old rule because they are about the reader, not the flow:

* ``x += y`` keeps what was there (it gens and kills nothing).
  :func:`pathcond.assigned_expressions` returns the right-hand side alone, so
  the walk can only reconstruct ``x`` as "the earlier writes, and this"; read
  as an overwrite, ``copyArchive.py``'s holding limit came out as sixty
  minutes and two SQL strings built with ``+=`` lost their ``SELECT``.
  ``x = f(x)`` does overwrite: once it has run, it is the whole answer.
* *rereads* are writes the caller knows keep what was there and a statement
  cannot show -- ``keyList.add(tmpKey)``, ``varMap[":diag"] = trunc(varMap[":diag"])``.

**Exceptions.**  The state at the entry of every statement that can raise
flows to the handlers of the ``try`` around it, and on outwards unless one
catches everything.  ``try: nRanges = int(nRanges) / except: nRanges = 8``
is two answers, not "always eight".  ``finally`` runs on every way out --
normal, raised, returned, broken -- and each continues where it was going.
``except ... as e`` unbinds ``e`` at the handler's end, as Python does.

Not modelled: a ``with`` that swallows an exception, a call that never
returns, a generator's suspension, ``global``/``nonlocal``.
"""

from __future__ import annotations

import ast
from typing import Optional, Sequence

#: The nodes whose body is a scope of its own.
_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)

#: Expression nodes that cannot raise on their own.
_SAFE = (
    ast.Constant,
    ast.Name,
    ast.Load,
    ast.Store,
    ast.Del,
    ast.List,
    ast.Tuple,
    ast.Set,
    ast.Dict,
    ast.expr_context,
    ast.Starred,
)

#: Statements that can raise whatever they hold.
_RAISING = (ast.Raise, ast.Assert, ast.Delete, ast.Import, ast.ImportFrom)

_CATCH_ALL = {"Exception", "BaseException"}

#: How many site sets a scope keeps flows for.
_KEPT = 32


def reaching(
    target: ast.AST,
    sites: Sequence[ast.AST],
    killers: Sequence[ast.AST] = (),
    rereads: Sequence[ast.AST] = (),
) -> list[ast.AST]:
    """Of *sites*, the writes that can be what *target* reads, in the order given.

    *sites* and *killers* are writes to one variable: statements, or the
    ``ExceptHandler`` for ``except ... as``.  A killer overwrites like any
    other write and is never an answer -- the walk passes the writes outside
    the stretch it owns that way, since they still ran.  *rereads* add to the
    variable and kill nothing, as ``x += y`` does.

    Where a site writes is decided by what it is: a ``for`` at the top of
    each pass, a ``with`` or a handler on entering its body, a ``while`` or
    ``if`` holding ``:=`` at its test, a ``match`` on entering a case, any
    other statement once it has run.

    The scope is the one *target* is in.  A write inside a nested function is
    to another variable and reaches nothing here.  If no write is in this
    scope, the name is one an enclosing scope binds, read whenever this one
    runs, and every enclosing write comes back.
    """
    scope = _scope_of(target)
    if scope is None:
        return list(sites)
    statement = (
        target
        if isinstance(target, (ast.stmt, ast.ExceptHandler))
        else _statement_of(target)
    )
    if statement is None:
        return list(sites)
    own = [node for node in (*sites, *killers) if _scope_of(node) is scope]
    if not own:
        return [site for site in sites if _encloses(_scope_of(site), scope)]
    ids = sorted(
        {id(node) for node in own}
        | {id(node) for node in rereads if _scope_of(node) is scope}
    )
    bits = {key: 1 << index for index, key in enumerate(ids)}
    keeps = 0
    for node in (*own, *rereads):
        if isinstance(node, ast.AugAssign) or any(node is one for one in rereads):
            keeps |= bits.get(id(node), 0)
    state = _flow(scope, bits, keeps).entry.get(id(statement))
    if not state:
        return []
    return [site for site in sites if state & bits.get(id(site), 0)]


def _scope_of(node: ast.AST) -> Optional[ast.AST]:
    current = getattr(node, "parent", None)
    while current is not None and not isinstance(current, _SCOPES):
        current = getattr(current, "parent", None)
    return current


def _encloses(outer: Optional[ast.AST], scope: ast.AST) -> bool:
    current: Optional[ast.AST] = scope
    while current is not None:
        if current is outer:
            return True
        current = getattr(current, "parent", None)
    return False


def _statement_of(node: ast.AST) -> Optional[ast.AST]:
    current: Optional[ast.AST] = node
    while current is not None and not isinstance(current, ast.stmt):
        current = getattr(current, "parent", None)
    return current


def _flow(scope: ast.AST, bits: dict[int, int], keeps: int) -> _Flow:
    """The flow for this set of writes, kept on the scope.

    The walk asks about one name from every statement that reads it, the SQL
    slice about one key from every run, so the same writes come back with a
    new target many times.
    """
    held: Optional[dict[tuple[tuple[int, ...], int], _Flow]] = getattr(
        scope, "_reaching_flows", None
    )
    if held is None:
        held = {}
        scope._reaching_flows = held  # type: ignore[attr-defined]
    key = (tuple(bits), keeps)
    flow = held.get(key)
    if flow is None:
        if len(held) >= _KEPT:
            del held[next(iter(held))]
        flow = held[key] = _Flow(scope, bits, keeps)
    return flow


def _join(one: Optional[int], other: Optional[int]) -> Optional[int]:
    if one is None:
        return other
    if other is None:
        return one
    return one | other


def _own_expressions(statement: ast.AST) -> list[ast.AST]:
    """The expression nodes *statement* evaluates itself -- not its blocks, not a nested scope."""
    found: list[ast.AST] = []
    stack = [
        child
        for child in ast.iter_child_nodes(statement)
        if not isinstance(child, (ast.stmt, ast.ExceptHandler, ast.match_case))
    ]
    while stack:
        node = stack.pop()
        if isinstance(node, ast.Lambda):
            continue
        found.append(node)
        stack.extend(ast.iter_child_nodes(node))
    return found


def _unpacks(statement: ast.AST) -> bool:
    """Whether *statement* binds by unpacking, which raises on a wrong length or type."""
    if isinstance(statement, ast.Assign):
        targets = statement.targets
    elif isinstance(statement, (ast.For, ast.AsyncFor)):
        targets = [statement.target]
    elif isinstance(statement, (ast.With, ast.AsyncWith)):
        targets = [
            item.optional_vars
            for item in statement.items
            if item.optional_vars is not None
        ]
    else:
        return False
    return any(isinstance(target, (ast.Tuple, ast.List)) for target in targets)


def _raises(statement: ast.AST) -> bool:
    """Whether *statement* can raise before it is done -- its own expressions, not its blocks."""
    held = getattr(statement, "_reaching_raises", None)
    if held is None:
        held = (
            isinstance(statement, _RAISING)
            or _unpacks(statement)
            or any(not isinstance(node, _SAFE) for node in _own_expressions(statement))
        )
        statement._reaching_raises = held  # type: ignore[attr-defined]
    return held


def _catches_all(handler: ast.ExceptHandler) -> bool:
    kind = handler.type
    if kind is None:
        return True
    names = kind.elts if isinstance(kind, ast.Tuple) else [kind]
    return any(isinstance(name, ast.Name) and name.id in _CATCH_ALL for name in names)


def _irrefutable(case: ast.match_case) -> bool:
    pattern = case.pattern
    return (
        case.guard is None
        and isinstance(pattern, ast.MatchAs)
        and pattern.pattern is None
    )


class _Handlers:
    """A ``try``'s handlers, collecting the state at every point that can raise into them."""

    __slots__ = ("catches_all", "state")

    def __init__(self, catches_all: bool) -> None:
        self.catches_all = catches_all
        self.state: Optional[int] = None


class _Finally:
    """A ``finally``, collecting each way out that passes through it."""

    __slots__ = ("raised", "returned", "jumps")

    def __init__(self) -> None:
        self.raised: Optional[int] = None
        self.returned: Optional[int] = None
        # id(loop) -> (loop, [breaks, continues])
        self.jumps: dict[int, tuple[_Loop, list[Optional[int]]]] = {}


class _Loop:
    __slots__ = ("depth", "breaks", "continues")

    def __init__(self, depth: int) -> None:
        # How many catchers were open when the loop started: a ``break``
        # leaves the ``finally`` blocks above that and no others.
        self.depth = depth
        self.breaks: Optional[int] = None
        self.continues: Optional[int] = None


class _Flow:
    """The state at the entry of every statement in one scope, for one set of writes.

    A state is the set of writes that can be live, as bits; ``None`` is a
    point nothing runs to.  Loops are iterated to a fixpoint, so the entry of
    a statement in a loop holds what any pass brings to it.
    """

    def __init__(self, scope: ast.AST, bits: dict[int, int], keeps: int) -> None:
        self.bits = bits
        self.keeps = keeps
        self.entry: dict[int, int] = {}
        self._catchers: list[_Handlers | _Finally] = []
        self._loops: list[_Loop] = []
        self._block(scope.body, 0)  # type: ignore[attr-defined]

    # -- plumbing -------------------------------------------------------------
    def _gen(self, state: Optional[int], node: ast.AST) -> Optional[int]:
        bit = self.bits.get(id(node))
        if bit is None or state is None:
            return state
        return state | bit if bit & self.keeps else bit

    def _record(self, node: ast.AST, state: Optional[int]) -> None:
        if state is None:
            return
        held = self.entry.get(id(node))
        self.entry[id(node)] = state if held is None else held | state

    def _raise(self, state: Optional[int]) -> None:
        if state is None:
            return
        for catcher in reversed(self._catchers):
            if isinstance(catcher, _Finally):
                catcher.raised = _join(catcher.raised, state)
                return
            catcher.state = _join(catcher.state, state)
            if catcher.catches_all:
                return

    def _return(self, state: Optional[int]) -> None:
        if state is None:
            return
        for catcher in reversed(self._catchers):
            if isinstance(catcher, _Finally):
                catcher.returned = _join(catcher.returned, state)
                return

    def _jump(self, state: Optional[int], loop: _Loop, which: int) -> None:
        """A ``break`` (0) or ``continue`` (1): through any ``finally`` inside the loop first."""
        if state is None:
            return
        for catcher in reversed(self._catchers[loop.depth :]):
            if isinstance(catcher, _Finally):
                _held, jumps = catcher.jumps.setdefault(id(loop), (loop, [None, None]))
                jumps[which] = _join(jumps[which], state)
                return
        if which == 0:
            loop.breaks = _join(loop.breaks, state)
        else:
            loop.continues = _join(loop.continues, state)

    def _block(self, statements: list[ast.stmt], state: Optional[int]) -> Optional[int]:
        for statement in statements:
            if state is None:
                break
            state = self._statement(statement, state)
        return state

    def _statement(self, node: ast.stmt, state: Optional[int]) -> Optional[int]:
        self._record(node, state)
        if _raises(node):
            self._raise(state)
        method = getattr(self, "_" + type(node).__name__, None)
        if method is None:
            return self._gen(state, node)
        return method(node, state)

    # -- leaving a block ------------------------------------------------------
    def _Return(self, node: ast.Return, state: Optional[int]) -> Optional[int]:
        self._return(state)
        return None

    def _Raise(self, node: ast.Raise, state: Optional[int]) -> Optional[int]:
        return None

    def _Break(self, node: ast.Break, state: Optional[int]) -> Optional[int]:
        if self._loops:
            self._jump(state, self._loops[-1], 0)
        return None

    def _Continue(self, node: ast.Continue, state: Optional[int]) -> Optional[int]:
        if self._loops:
            self._jump(state, self._loops[-1], 1)
        return None

    # -- compound statements --------------------------------------------------
    def _If(self, node: ast.If, state: Optional[int]) -> Optional[int]:
        state = self._gen(state, node)
        return _join(self._block(node.body, state), self._block(node.orelse, state))

    def _With(
        self, node: ast.With | ast.AsyncWith, state: Optional[int]
    ) -> Optional[int]:
        return self._block(node.body, self._gen(state, node))

    _AsyncWith = _With

    def _loop(
        self, node: ast.For | ast.AsyncFor | ast.While, before: int
    ) -> tuple[int, Optional[int]]:
        """Iterate the body to a fixpoint: (the state at the head, the ``break`` paths)."""
        head = before
        while True:
            loop = _Loop(len(self._catchers))
            self._loops.append(loop)
            end = self._block(node.body, self._gen(head, node))
            self._loops.pop()
            again = before | (end or 0) | (loop.continues or 0)
            if again == head:
                return head, loop.breaks
            head = again

    def _For(self, node: ast.For | ast.AsyncFor, state: Optional[int]) -> Optional[int]:
        if state is None:
            return None
        head, breaks = self._loop(node, state)
        self._raise(head)  # ``next()`` on the iterator, every time round
        return _join(self._block(node.orelse, head), breaks)

    _AsyncFor = _For

    def _While(self, node: ast.While, state: Optional[int]) -> Optional[int]:
        if state is None:
            return None
        head, breaks = self._loop(node, state)
        self._record(node, head)  # the test is read at the head, every time round
        if _raises(node):
            self._raise(head)
        if isinstance(node.test, ast.Constant) and node.test.value:
            return breaks
        return _join(self._block(node.orelse, self._gen(head, node)), breaks)

    def _Try(self, node: ast.Try, state: Optional[int]) -> Optional[int]:
        guard = _Finally() if node.finalbody else None
        if guard is not None:
            self._catchers.append(guard)
        caught = (
            _Handlers(any(_catches_all(h) for h in node.handlers))
            if node.handlers
            else None
        )
        if caught is not None:
            self._catchers.append(caught)
        body = self._block(node.body, state)
        if caught is not None:
            self._catchers.pop()
        normal = self._block(node.orelse, body)
        if caught is not None and caught.state is not None:
            for handler in node.handlers:
                self._record(handler, caught.state)
                out = self._block(handler.body, self._gen(caught.state, handler))
                if out is not None and handler.name and id(handler) in self.bits:
                    out = 0  # Python deletes the name at the handler's end
                normal = _join(normal, out)
        if guard is None:
            return normal
        self._catchers.pop()
        # Each way out runs the ``finally`` and goes on where it was going.
        if guard.raised is not None:
            self._raise(self._block(node.finalbody, guard.raised))
        if guard.returned is not None:
            self._return(self._block(node.finalbody, guard.returned))
        for loop, (breaks, continues) in guard.jumps.values():
            if breaks is not None:
                self._jump(self._block(node.finalbody, breaks), loop, 0)
            if continues is not None:
                self._jump(self._block(node.finalbody, continues), loop, 1)
        if normal is not None:
            normal = self._block(node.finalbody, normal)
        return normal

    _TryStar = _Try

    def _Match(self, node: ast.Match, state: Optional[int]) -> Optional[int]:
        entered = self._gen(state, node)
        out: Optional[int] = None
        for case in node.cases:
            out = _join(out, self._block(case.body, entered))
        if not any(_irrefutable(case) for case in node.cases):
            out = _join(out, state)  # no case matched
        return out
