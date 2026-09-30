"""What values an expression can take.

Resolution rather than recognition, and outside the recognizers for the same
reason attribution is: two slices need the same answer.  The attribute slice
asks it of ``taskSpec.status = <expression>`` and the SQL slice of
``varMap[":status"] = <expression>``, and the value is settled by the same rules
either way -- a map whose two halves resolved the same expression differently
would be describing PanDA twice.

**A declared mapping is the strongest value evidence in the corpus after a
declared vocabulary, and unlike a vocabulary it is complete.**
``JediTaskSpec.commandStatusMap()`` is not a sample of the statuses a command
leads to, it *is* the command-to-status relation::

    def commandStatusMap(cls):
        return {
            "kill":   {"doing": "aborting",  "done": "toabort"},
            "finish": {"doing": "finishing", "done": "passed"},
            ...

So ``commandStatusMap()[commandStr]["done"]`` has a resolvable value set even
though ``commandStr`` is only known at run time: fix the keys the code fixes,
enumerate the ones it does not.  Two of the statuses PanDA declares and the map
could not account for -- ``aborting`` and ``passed`` -- exist nowhere else.

Enumeration is only sound when the whole mapping is literal, so a dict with one
computed entry resolves to nothing rather than to a set missing that entry.  The
difference matters downstream: pruning eliminates, so a value set with a member
missing produces a confident wrong answer where no value set at all produces an
honest tier-2 branch.

The result shape is left to the caller.  The two slices anchor differently -- an
attribute assignment *is* the write, while a bind is written inside a SQL string
and decided at a Python assignment somewhere else -- so they carry different
things alongside the value, and forcing one tuple on both would give each a
field it ignores.
"""

from __future__ import annotations

import ast
import json
import re
from typing import TYPE_CHECKING, NamedTuple, Optional, Union

from bamboo.codemap.models import SourceModule
from bamboo.codemap.panda import pathcond
from bamboo.codemap.panda.pathcond import functions_with_owner, single_definition

if TYPE_CHECKING:
    from bamboo.codemap.calls import Calls

# A literal mapping, nested as deeply as the source nests it.
Mapping = dict[str, Union[str, "Mapping"]]

# ``{0}`` / ``{}`` / ``{schema}`` in a ``str.format`` template.
_FIELD = re.compile(r"\{[^{}]*\}")

# ``action=set_exhausted`` / ``reason=low_efficiency`` -- PanDA's own name for
# the decision a message is about, the same machine-readable shape the
# brokerage slice reads as ``criteria=-<tag>``.  Whole tokens: the token is what
# an observed message is matched on, and it survives interpolation because it is
# the literal part of the frame.
_TAG = re.compile(r"\b((?:action|reason)=[A-Za-z_][A-Za-z0-9_]*)")


def decision_tags(text: str) -> set[str]:
    """The tags *text* names for the decision it reports.

    Here rather than in a recognizer because two of them read it for opposite
    ends: the progress slice asks which branch a tag belongs to, and the emit
    pass asks which line belongs to a tagged branch.  Two copies of this
    pattern would be two things to keep in step.
    """
    return set(_TAG.findall(text))


def rendered_text(node: ast.expr) -> Optional[str]:
    """Render a string expression, marking the parts only run time knows as ``{}``.

    The shape of the text rather than the text: what the code will produce with
    the holes left open.  Two callers want the same answer for opposite reasons
    -- reading a SQL statement, where the interpolation is almost always the
    schema name and blanking it keeps the statement readable; and indexing a
    diagnostic, where the literal frame is precisely the part a message observed
    in production can be matched against.

    ``None`` when nothing literal can be recovered, which is different from an
    empty string: it means the expression says nothing about its own text.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        return "".join(
            value.value if isinstance(value, ast.Constant) and isinstance(value.value, str) else "{}"
            for value in node.values
        )
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, right = rendered_text(node.left), rendered_text(node.right)
        if left is None and right is None:
            return None
        return (left or "") + (right or "")
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "format"
    ):
        # ``"FROM {0}.JEDI_Tasks tabT,{0}.JEDI_AUX_Status_MinTaskID tabA
        # ".format(panda_config.schemaJEDI)`` -- the older spelling of the same
        # f-string, and 62 statements still use it.  Missing it did not merely
        # lose coverage: those statements are where JEDI selects tasks by
        # status, so the map read a dozen task states as ones nothing ever
        # selects on.
        text = rendered_text(node.func.value)
        return None if text is None else _FIELD.sub("{}", text)
    return None


def has_literal_text(template: str) -> bool:
    """Whether anything but the holes is left of *template*.

    The line between a frame and a hole with nothing around it, and both slices
    that read rendered text need it drawn the same way.  ``"{}"`` on its own is
    a name the code did not write -- it says the caller supplies one -- and
    treating it as text gives every caller the same name.
    """
    return bool(_FIELD.sub("", template).strip())


def template_matches(template: str, text: str) -> bool:
    """Whether *text* is something *template* could have rendered.

    Anchored, and the holes are the only wildcards: a step named ``endpoint
    check with DISK_THRESHOLD={} TB`` is the same step whether production ran
    it at 10 TB or 1000 TB, and nothing else in the corpus should answer to
    that name.  What keeps the wildcards from swallowing the corpus is
    :func:`has_literal_text`, which is what admits a template in the first
    place.
    """
    pattern = ".*".join(re.escape(part) for part in _FIELD.split(template))
    return re.fullmatch(pattern, text) is not None


def diagnostic_template(node: ast.expr) -> Optional[str]:
    """The template *node* assembles, if it is one worth indexing.

    Two conditions, both properties of the expression rather than claims about
    the field it is written to -- which is what lets the index avoid deciding
    what counts as a message.

    **Assembled, not written out.**  A template is a frame with holes; a bare
    literal is not one, and an exact message can be found by searching the
    source for itself.  That case the index does not have to earn.

    **Some literal text.**  ``setErrDiag`` appends with ``f"{self.errorDialog}
    {diag}"``, whose frame is two holes and a space: true, and matching nothing.
    A row that cannot serve as a search key is noise in an index that exists to
    be searched.
    """
    assembled = isinstance(node, (ast.JoinedStr, ast.BinOp)) or (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "format"
    )
    if not assembled:
        return None
    text = rendered_text(node)
    return text if text and has_literal_text(text) else None


def _literal_mapping(node: ast.expr) -> Optional[Mapping]:
    """Return a dict literal as a plain mapping, or ``None`` if not all literal.

    All-or-nothing on purpose: a mapping with one computed entry cannot be
    enumerated, and a set of values that quietly omits that entry is worse than
    no set at all, because elimination treats a short candidate list as complete.
    """
    if not isinstance(node, ast.Dict):
        return None
    mapping: Mapping = {}
    for key, value in zip(node.keys, node.values, strict=False):
        if not (isinstance(key, ast.Constant) and isinstance(key.value, str)):
            return None
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            mapping[key.value] = value.value
            continue
        nested = _literal_mapping(value)
        if nested is None:
            return None
        mapping[key.value] = nested
    return mapping or None


def declared_mappings(modules: list[SourceModule]) -> dict[str, Mapping]:
    """Return ``{method name: the literal mapping it returns}``.

    Keyed by method name for the reason the aliases and producers are: a call
    site offers the name, and the class is what is often being resolved.  Where
    two classes return different mappings from the same name the entry is
    dropped -- the call site cannot be told which one it reached, and a wrong
    value set is worse than none.

    Restricted to methods taking nothing but ``cls``/``self``.  A parameter means
    the mapping is built for a caller rather than declared, and the returned dict
    would then be one of several the method can produce.
    """
    found: dict[str, list[Mapping]] = {}
    for module in modules:
        for func, owner in functions_with_owner(module.tree):
            if owner is None:
                continue
            arguments = [*func.args.posonlyargs, *func.args.args, *func.args.kwonlyargs]
            if len(arguments) > 1 or func.args.vararg or func.args.kwarg:
                continue
            for node in ast.walk(func):
                if not isinstance(node, ast.Return) or node.value is None:
                    continue
                mapping = _literal_mapping(node.value)
                if mapping is not None:
                    found.setdefault(func.name, []).append(mapping)
    return {
        name: mappings[0]
        for name, mappings in found.items()
        # ``json`` rather than a set: a nested dict is not hashable, and the
        # comparison wanted here is on content.
        if len({json.dumps(m, sort_keys=True) for m in mappings}) == 1
    }


def _mapping_behind(
    node: ast.expr,
    func: Optional[ast.FunctionDef | ast.AsyncFunctionDef],
    mappings: dict[str, Mapping],
) -> Optional[Mapping]:
    """Return the declared mapping *node* evaluates to.

    Two shapes, both real in the corpus: the call itself, and a local holding
    it -- ``commandStatusMap = JediTaskSpec.commandStatusMap()`` sits at the top
    of the method that subscripts it a hundred lines further down.
    """
    if isinstance(node, ast.Call):
        callee = node.func
        name = (
            callee.id
            if isinstance(callee, ast.Name)
            else callee.attr
            if isinstance(callee, ast.Attribute)
            else None
        )
        return mappings.get(name) if name else None
    if isinstance(node, ast.Name) and func is not None:
        definition = single_definition(func, node.id)
        if definition is not None and definition is not node:
            return _mapping_behind(definition, func, mappings)
    return None


def _mapping_iterated(
    name: str,
    func: Optional[ast.FunctionDef | ast.AsyncFunctionDef],
    mappings: dict[str, Mapping],
) -> Optional[Mapping]:
    """The declared mapping *name* takes an entry of, when a loop binds it.

    ``for commandStr, taskStatusMap in commandStatusMap.items()`` binds
    ``taskStatusMap`` to one value of the mapping per turn, so
    ``taskStatusMap["doing"]`` reaches the same six statuses that
    ``commandStatusMap[commandStr]["doing"]`` does.  The corpus writes it the
    first way in one place and the second way in another, a hundred lines apart
    in the same file, and reading only the subscript is why the map could say
    which statuses a command *leaves behind* but not which ones it *looks for*
    -- so every "command in progress" status looked like a value nothing
    selects on, which is the shape of a stall the map is supposed to explain.
    """
    if func is None:
        return None
    for node in ast.walk(func):
        if not isinstance(node, ast.For) or not isinstance(node.target, ast.Tuple):
            continue
        bound = [
            index
            for index, element in enumerate(node.target.elts)
            if isinstance(element, ast.Name) and element.id == name
        ]
        # The value half only.  Binding the key half would enumerate command
        # names as if they were statuses.
        if bound != [1] or len(node.target.elts) != 2:
            continue
        if not (
            isinstance(node.iter, ast.Call)
            and isinstance(node.iter.func, ast.Attribute)
            and node.iter.func.attr == "items"
        ):
            continue
        mapping = _mapping_behind(node.iter.func.value, func, mappings)
        if mapping is not None:
            return mapping
    return None


def mapping_values(
    expression: ast.expr,
    *,
    func: Optional[ast.FunctionDef | ast.AsyncFunctionDef],
    mappings: dict[str, Mapping],
) -> list[str]:
    """The values a subscript of a declared mapping can produce, if it is one.

    A key the code states narrows to one entry; a key settled at run time
    enumerates the level.  So ``[commandStr]["done"]`` gives the six statuses a
    completed command leaves behind, and ``["incexec"]["done"]`` gives exactly
    ``rerefine``.

    A name a loop bound to one entry of the mapping enters one level down, and
    is otherwise the same walk -- see :func:`_mapping_iterated`.

    Returns ``[]`` when the expression is not this shape, when the mapping is
    not declared, or when a stated key is not in it -- that last one being a
    disagreement between two parts of the source, which is not this function's
    to settle.
    """
    keys: list[ast.expr] = []
    node: ast.expr = expression
    while isinstance(node, ast.Subscript):
        keys.append(node.slice)
        node = node.value
    if not keys:
        return []
    level: list[Union[str, Mapping]]
    mapping = _mapping_behind(node, func, mappings)
    if mapping is not None:
        level = [mapping]
    else:
        iterated = (
            _mapping_iterated(node.id, func, mappings)
            if isinstance(node, ast.Name)
            else None
        )
        if iterated is None:
            return []
        level = list(iterated.values())

    for key in reversed(keys):
        stepped: list[Union[str, Mapping]] = []
        for entry in level:
            if not isinstance(entry, dict):
                return []
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                if key.value not in entry:
                    return []
                stepped.append(entry[key.value])
            else:
                stepped.extend(entry.values())
        level = stepped
    if not level or any(not isinstance(entry, str) for entry in level):
        return []
    return sorted({entry for entry in level if isinstance(entry, str)})


def resolver(mappings: dict[str, Mapping]):
    """Return a ``(expression, func) -> [value, ...]`` callable for one build.

    Handed to :func:`bamboo.codemap.panda.pathcond.literal_values` so that a
    local assigned from a declared mapping resolves the same way one assigned a
    literal does.  That is where the two statuses this closes actually live:
    ``newTaskStatus = commandStatusMap[commandStr]["doing"]`` reaches the
    database through a bind, not through the subscript directly.

    **Any constant is settled, not only a string.**  ``varMap[":frozenTime"] =
    None`` is as decided as ``= "ready"``; recording it as ``runtime(None)``
    would claim the value waits for run time when the source has already
    written it down.  Non-strings are spelled as Python spells them, which is
    the spelling the attribute slice already gives a constant right-hand side,
    so one subject does not end up carrying the same value under two names.
    """

    def resolve(
        expression: ast.expr,
        func: Optional[ast.FunctionDef | ast.AsyncFunctionDef],
    ) -> list[str]:
        if isinstance(expression, ast.Constant):
            if isinstance(expression.value, str):
                return [expression.value]
            return [ast.unparse(expression)]
        return mapping_values(expression, func=func, mappings=mappings)

    return resolve


#: How many calls one parameter is followed into.  Past it the parameter is
#: left unsettled rather than settled by a sample of its callers.
HANDED_CALLERS = 16

#: How many functions out a value is followed.  ``lockJobsForReassign``'s
#: status list is two: the ``DBProxy`` method's parameter is ``TaskBuffer``'s,
#: which is the list ``copyArchive.main`` writes down.
HANDED_DEPTH = 4


class Handed(NamedTuple):
    """One value an expression can take, and the calls it came in through."""

    value: str
    #: ``file:line`` of each call the value was handed through, the one into
    #: the function asked about first.  Empty where the function says it.
    via: tuple[str, ...] = ()


def handed(resolve, calls: Optional["Calls"]):
    """Widen *resolve* to the values a name is handed, in the function and by its callers.

    *resolve* settles what one expression states -- a constant, a declared
    mapping.  Around it, three ways a statement's bind is given its value
    without stating it:

    * a list, tuple or set literal, element by element -- what an ``IN`` list
      is bound from;
    * a local name, through each ``=`` or ``+=`` in the function that writes
      it -- and only where every one of them settles (see ``bindings``);
    * a parameter, through what each call :meth:`Calls.callers` places at the
      function hands it (``calls.bind``), read in the caller the same way.

    Returns ``(expression, func, file) -> [Handed, ...]``.  Without *calls*
    only the function's own writes are read.

    A union, not a choice: a statement that three callers hand three lists
    asks, over its runs, for every value any of them names.  Which caller
    asked for which is kept in ``via``.
    """
    from bamboo.codemap.calls import Target, bind
    from bamboo.codemap.calls import returns as calls_returns

    def parameters(func: ast.AST) -> set[str]:
        args = getattr(func, "args", None)
        if args is None:
            return set()
        return {a.arg for a in (*args.posonlyargs, *args.args, *args.kwonlyargs)}

    def in_scope(node: ast.AST, func: ast.AST) -> bool:
        owner = pathcond.enclosing_function(node)
        return owner is func if owner is not None else isinstance(func, ast.Module)

    def listed(expression: ast.expr) -> bool:
        """Whether a ``+`` chain concatenates sequences: one of its operands is written as one."""
        if isinstance(expression, (ast.List, ast.Tuple, ast.Set)):
            return True
        if isinstance(expression, ast.BinOp) and isinstance(expression.op, ast.Add):
            return listed(expression.left) or listed(expression.right)
        return False

    def bindings(func: ast.AST, name: str) -> Optional[list[ast.expr]]:
        """What each binding of *name* in *func* writes; ``None`` if one writes what cannot be read here.

        All or nothing, because the values are claimed as what the name
        holds: ``pandaID = 0`` before a loop that rebinds it from the rows is
        not a value any statement asks by, and settling the one literal of
        two writes said it was.
        """
        found: list[ast.expr] = []
        for node in ast.walk(func):
            if node is func or not in_scope(node, func):
                continue
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == name:
                        found.append(node.value)
                    elif any(isinstance(one, ast.Name) and one.id == name for one in ast.walk(target)):
                        return None
            elif isinstance(node, (ast.AnnAssign, ast.NamedExpr)):
                if isinstance(node.target, ast.Name) and node.target.id == name and node.value is not None:
                    found.append(node.value)
            elif isinstance(node, ast.AugAssign):
                if isinstance(node.target, ast.Name) and node.target.id == name:
                    # ``statList += ["x"]`` adds members; ``text += "x"``
                    # makes a different string.
                    if not isinstance(node.value, (ast.List, ast.Tuple, ast.Set)):
                        return None
                    found.append(node.value)
            elif isinstance(node, (ast.For, ast.AsyncFor, ast.With, ast.AsyncWith, ast.ExceptHandler)):
                targets = (
                    [node.target] if isinstance(node, (ast.For, ast.AsyncFor))
                    else [item.optional_vars for item in node.items if item.optional_vars is not None]
                    if isinstance(node, (ast.With, ast.AsyncWith))
                    else []
                )
                if isinstance(node, ast.ExceptHandler) and node.name == name:
                    return None
                if any(
                    isinstance(one, ast.Name) and one.id == name
                    for target in targets
                    for one in ast.walk(target)
                ):
                    return None
            elif isinstance(node, (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal)):
                names = [alias.asname or alias.name for alias in node.names] if isinstance(
                    node, (ast.Import, ast.ImportFrom)
                ) else node.names
                if name in names:
                    return None
        return found

    def settle(
        expression: ast.expr, func: ast.AST, file: str, depth: int, seen: frozenset
    ) -> list[Handed]:
        stated = resolve(expression, func)
        if stated:
            return [Handed(value) for value in stated]
        if isinstance(expression, (ast.List, ast.Tuple, ast.Set)):
            return [
                one
                for element in expression.elts
                if not isinstance(element, ast.Starred)
                for one in settle(element, func, file, depth, seen)
            ]
        if isinstance(expression, ast.BinOp) and isinstance(expression.op, ast.Add) and listed(expression):
            # ``cls.getInputTypes() + ["pp_input"]``: every member of both.
            left = settle(expression.left, func, file, depth, seen)
            right = settle(expression.right, func, file, depth, seen)
            return left + right if left and right else []
        if isinstance(expression, ast.Call):
            return returned(expression, func, file, depth, seen)
        if not isinstance(expression, ast.Name):
            return []
        key = (file, getattr(func, "lineno", 0), expression.id)
        if key in seen:
            return []
        seen = seen | {key}
        written = bindings(func, expression.id)
        if written is None:
            return []
        found: list[Handed] = []
        for value in written:
            settled = settle(value, func, file, depth, seen)
            if not settled:
                return []
            found.extend(settled)
        if expression.id not in parameters(func):
            return found
        if calls is None or depth >= HANDED_DEPTH:
            return []
        owned = calls.types.method_class(file, func)
        target = Target(file, func, owned[1] if owned else None)
        sites = calls.callers(target)
        if not sites or len(sites) > HANDED_CALLERS:
            return []
        for site in sites:
            passed = bind(site.call, target).get(expression.id)
            if passed is None:
                continue
            at = f"{site.file}:{site.call.lineno}"
            found.extend(
                Handed(one.value, (at, *one.via))
                for one in settle(passed, site.func, site.file, depth + 1, seen)
            )
        return found

    def returned(call: ast.Call, func: ast.AST, file: str, depth: int, seen: frozenset) -> list[Handed]:
        """What the functions *call* lands in return, where each of their returns settles."""
        if calls is None or depth >= HANDED_DEPTH:
            return []
        placed = calls.callees(file, func, call)
        if not placed.targets or placed.unplaced:
            return []
        found: list[Handed] = []
        for target in placed.targets:
            key = (target.file, target.func.lineno, "<return>")  # type: ignore[attr-defined]
            if key in seen:
                return []
            for value in calls_returns(target.func):
                settled = settle(value, target.func, target.file, depth + 1, seen | {key})
                if not settled:
                    return []
                found.extend(settled)
        return found

    def settled(expression: ast.expr, func: ast.AST, file: str) -> list[Handed]:
        return settle(expression, func, file, 0, frozenset())

    return settled
