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
import re
from typing import TYPE_CHECKING, NamedTuple, Optional, Union

from bamboo.codemap.panda import pathcond
from bamboo.codemap.panda.pathcond import single_definition
from bamboo.codemap.panda.reaching import reaching

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


def blank_fields(template: str) -> str:
    """*template* with every ``str.format`` field spelled ``{}``.

    ``{}`` is the one spelling of a hole every reader of rendered text keys on
    -- the table pattern, the schema strip, the boundary a table nobody writes
    becomes.  A ``{0}`` left standing is a hole none of them recognise: read as
    a table, it became a system named ``{0}`` that some other system maintains.
    """
    return _FIELD.sub("{}", template)


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
        return None if text is None else blank_fields(text)
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


#: How far :func:`value_frames` follows a local back.  Two, because the second
#: hop is the shape ``setup_destination`` has: ``dest_error[dest] = tmp_msg``,
#: with ``tmp_msg`` built two lines above.
FRAME_HOPS = 2
#: More renderings than this and the part is left a hole rather than listed.
FRAME_LIMIT = 32
_HOLE = "{}"
#: The methods that read a dict without changing what its values can be.
_READS_A_DICT = frozenset({"get", "keys", "values", "items", "copy", "setdefault"})


def value_frames(
    node: ast.expr, func: Optional[ast.FunctionDef | ast.AsyncFunctionDef]
) -> list[str]:
    """Every text *node* can evaluate to, holes as ``{}`` -- all of them, or none.

    What an arm writes when the value is only settled at run time, read far
    enough to bound it.  ``self.job.ddmErrorDiag = err_msg + extracted_err_str``
    names no text of its own; ``err_msg = "Could not add files to DDM: "`` two
    lines above does, and an observed ``ddmErrorDiag`` that does not begin with
    it was written by some other arm.  Of the nine writers the map offered for
    one such value, five fall to that -- a local's assignment, a ``+``, a dict
    of messages filled across the function.

    **Complete or empty**, because the caller eliminates with it: a list
    missing one text is a confident wrong answer about the arm that writes it.
    So every part the reading cannot bound is a hole, which matches anything,
    and a value with nothing but holes left is no frame at all.  Followed back
    through a local name or a local dict's subscript, :data:`FRAME_HOPS` deep,
    and through **every** binding that can hold the value where it is read:
    a parameter, a ``global``, an unpacking, a loop or ``with`` target, a dict
    handed to another function or changed by a method -- each is a hole.

    Not :func:`progress._reaching`, which a later write in another arm shadows
    on purpose: it answers which text *belongs* to a write, to attach a tag,
    and dropping a possible text there costs a tag, here an arm.  And an
    earlier pass of a loop holds what this one did not assign, so inside a
    loop around both the two arms of one ``if`` are not kept apart.
    """
    try:
        found = _frames(node, func, pathcond.site_of(node), FRAME_HOPS)
    except RecursionError:
        return []
    if any(_HOLE in frame and not has_literal_text(frame) for frame in found):
        return []
    return found


def _frames(node: ast.expr, func, at: ast.AST, hops: int) -> list[str]:
    if isinstance(node, ast.Constant):
        return [_literal(node.value)] if isinstance(node.value, str) else [_HOLE]
    if isinstance(node, ast.JoinedStr):
        parts = []
        for part in node.values:
            if isinstance(part, ast.Constant):
                parts.append([_literal(part.value)])
            elif part.conversion == -1 and part.format_spec is None:
                parts.append(_frames(part.value, func, at, hops))
            else:
                # ``{x!r}`` quotes it and ``{x:>8}`` pads it: not the text it holds.
                parts.append([_HOLE])
        return _joined(parts)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _frames(node.left, func, at, hops)
        right = _frames(node.right, func, at, hops)
        if left == [_HOLE] and right == [_HOLE]:
            return [_HOLE]  # two numbers, as far as anything here can tell
        return _joined([left, right])
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "format"
        and isinstance(node.func.value, ast.Constant)
        and isinstance(node.func.value.value, str)
    ):
        text = node.func.value.value
        if "{{" in text or "}}" in text:
            return [_HOLE]
        return [_squeezed(blank_fields(text))]
    if func is None or hops == 0:
        return [_HOLE]
    if isinstance(node, ast.Name):
        return _held_by_name(node.id, func, at, hops)
    if (
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name)
        and isinstance(node.ctx, ast.Load)
    ):
        return _held_by_dict(node.value.id, func, at, hops)
    return [_HOLE]


def _literal(text: str) -> str:
    # A brace in the text would read as a hole; reading it as one is safe.
    return _HOLE if "{" in text or "}" in text else text


def _squeezed(frame: str) -> str:
    while _HOLE + _HOLE in frame:
        frame = frame.replace(_HOLE + _HOLE, _HOLE)
    return frame


def _distinct(frames: list[str]) -> list[str]:
    return sorted({_squeezed(frame) for frame in frames})


def _joined(parts: list[list[str]]) -> list[str]:
    joined = [""]
    for options in parts:
        joined = [head + tail for head in joined for tail in options]
        if len(joined) > FRAME_LIMIT:
            return [_HOLE]
    return _distinct(joined)


def _can_hold(site: ast.AST, at: ast.AST) -> bool:
    """Whether what *site* binds can be what the read at *at* sees."""
    if site is at:
        return False
    if pathcond.loops_over(site) & pathcond.loops_over(at):
        return True
    return getattr(site, "lineno", 0) <= getattr(at, "lineno", 0) and not pathcond.parted(
        site, at, exits=False
    )


def _local(func, name: str) -> bool:
    arguments = getattr(func, "args", None)  # a module's own body has none
    every = [*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs] if arguments is not None else []
    every += [argument for argument in (arguments.vararg, arguments.kwarg) if argument] if arguments is not None else []
    if any(argument.arg == name for argument in every):
        return False
    return not any(
        isinstance(node, (ast.Global, ast.Nonlocal)) and name in node.names
        for node in ast.walk(func)
    )


def _held_by_name(name: str, func, at: ast.AST, hops: int) -> list[str]:
    if not _local(func, name):
        return [_HOLE]
    every = pathcond.bindings(func, name)
    if any(binding.nested and binding.kind == pathcond.BIND_WALRUS for binding in every):
        return [_HOLE]  # ``[y := f(x) for x in L]`` binds the function's ``y``
    reaching = [
        binding for binding in every if not binding.nested and _can_hold(binding.site, at)
    ]
    if not reaching:
        return [_HOLE]
    found: list[str] = []
    for binding in reaching:
        if binding.unpacked or binding.slot is not None or binding.value is None:
            return [_HOLE]
        if binding.kind in (pathcond.BIND_ASSIGN, pathcond.BIND_ANNOTATED, pathcond.BIND_WALRUS):
            found += _frames(binding.value, func, binding.site, hops - 1)
        elif binding.kind == pathcond.BIND_AUGMENTED and isinstance(binding.site.op, ast.Add):
            # What was there, then this.  The head is a hole rather than the
            # earlier texts: which of them it was is a second question.
            found += [_HOLE + frame for frame in _frames(binding.value, func, binding.site, hops - 1)]
        else:
            return [_HOLE]
    return _distinct(found)


def _dict_kept_local(name: str, func) -> bool:
    """Whether every use of *name* reads the dict or stores into it, here."""
    for node in ast.walk(func):
        if not (isinstance(node, ast.Name) and node.id == name and isinstance(node.ctx, ast.Load)):
            continue
        parent = getattr(node, "parent", None)
        if isinstance(parent, ast.Subscript) and parent.value is node:
            continue
        if (
            isinstance(parent, ast.Compare)
            and any(comparator is node for comparator in parent.comparators)
            and all(isinstance(op, (ast.In, ast.NotIn)) for op in parent.ops)
        ):
            continue
        if isinstance(parent, ast.Attribute) and parent.attr in _READS_A_DICT:
            continue
        if isinstance(parent, ast.Call) and isinstance(parent.func, ast.Name) and parent.func.id == "len":
            continue
        # Handed on, aliased, returned, updated: someone else can fill it.
        # ``setup_source`` passes ``disp_error`` to the method that fills it.
        return False
    return True


def _held_by_dict(name: str, func, at: ast.AST, hops: int) -> list[str]:
    if not _local(func, name) or not _dict_kept_local(name, func):
        return [_HOLE]
    found: list[str] = []
    for binding in pathcond.bindings(func, name):
        if binding.nested:
            continue
        value = binding.value
        if binding.kind not in (pathcond.BIND_ASSIGN, pathcond.BIND_ANNOTATED) or binding.slot is not None or binding.unpacked:
            return [_HOLE]
        if isinstance(value, ast.Dict) and all(key is not None for key in value.keys):
            if _can_hold(binding.site, at):
                for held in value.values:
                    found += _frames(held, func, binding.site, hops - 1)
            continue
        if (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id == "dict"
            and not value.args
            and not value.keywords
        ):
            continue
        return [_HOLE]
    for node in ast.walk(func):
        if not (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id == name):
            continue
        if isinstance(node.ctx, ast.Load):
            if (
                isinstance(node.parent, ast.Attribute)  # type: ignore[attr-defined]
                and node.parent.attr == "setdefault"  # type: ignore[attr-defined]
            ):
                return [_HOLE]  # ``d[k].setdefault(...)``: a value that is a container
            continue
        if isinstance(node.ctx, ast.Del):
            continue
        statement = node.parent  # type: ignore[attr-defined]
        # Only a plain ``d[k] = v`` in this function's own body says what was
        # stored.  An unpacking, a loop target, a closure filling it: a hole.
        direct = (
            isinstance(statement, ast.Assign) and any(target is node for target in statement.targets)
        ) or (isinstance(statement, (ast.AnnAssign, ast.AugAssign)) and statement.target is node)
        if not direct or pathcond.enclosing_function(statement) is not func:
            return [_HOLE]
        if statement.value is None or not _can_hold(statement, at):
            continue
        if isinstance(statement, ast.AugAssign):
            if not isinstance(statement.op, ast.Add):
                return [_HOLE]
            found += [_HOLE + frame for frame in _frames(statement.value, func, statement, hops - 1)]
        else:
            found += _frames(statement.value, func, statement, hops - 1)
    for node in ast.walk(func):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "setdefault"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == name
        ):
            if len(node.args) != 2 or node.keywords or pathcond.enclosing_function(node) is not func:
                return [_HOLE]
            found += _frames(node.args[1], func, pathcond.site_of(node), hops - 1)
    return _distinct(found) if found else [_HOLE]


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


def _mapping_behind(
    node: ast.expr,
    func: Optional[ast.FunctionDef | ast.AsyncFunctionDef],
) -> Optional[list[Mapping]]:
    """Return the literal mappings *node* can evaluate to.

    Two shapes, both real in the corpus: the call itself, and a local holding
    it -- ``commandStatusMap = JediTaskSpec.commandStatusMap()`` sits at the top
    of the method that subscripts it a hundred lines further down.

    The call is followed through the corpus's calls to what it returns
    (:func:`sql.every_return`), so it is the class the receiver *is* that
    answers, not every class declaring a method of that name.  Read by name, two
    classes returning different mappings under one name had to be dropped, and
    a method taking a parameter could not be read at all.  Every return has to
    be a literal mapping, or the expression has no value set: enumeration
    eliminates, and a set short of a member is a confident wrong answer.
    """
    if isinstance(node, ast.Call):
        if func is None:
            return None
        from bamboo.codemap.panda import sql

        returned = sql.every_return(func, node)
        if returned is None:
            return None
        found: list[Mapping] = []
        for callee, value in returned:
            if isinstance(value, ast.Name):
                definition = single_definition(callee, value.id)
                if definition is not None:
                    value = definition
            mapping = _literal_mapping(value)
            if mapping is None:
                return None
            if mapping not in found:
                found.append(mapping)
        return found or None
    if isinstance(node, ast.Name) and func is not None:
        definition = single_definition(func, node.id)
        if definition is not None and definition is not node:
            return _mapping_behind(definition, func)
    return None


def _mapping_iterated(
    name: str,
    func: Optional[ast.FunctionDef | ast.AsyncFunctionDef],
) -> Optional[list[Mapping]]:
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
        mappings = _mapping_behind(node.iter.func.value, func)
        if mappings is not None:
            return mappings
    return None


def mapping_values(
    expression: ast.expr,
    *,
    func: Optional[ast.FunctionDef | ast.AsyncFunctionDef],
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
    mappings = _mapping_behind(node, func)
    if mappings is not None:
        level = list(mappings)
    else:
        iterated = _mapping_iterated(node.id, func) if isinstance(node, ast.Name) else None
        if iterated is None:
            return []
        level = [entry for mapping in iterated for entry in mapping.values()]

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


def resolver():
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
        return mapping_values(expression, func=func)

    return resolve


#: How many functions out a value is followed.  ``lockJobsForReassign``'s
#: status list is two: the ``DBProxy`` method's parameter is ``TaskBuffer``'s,
#: which is the list ``copyArchive.main`` writes down.
HANDED_DEPTH = 4


class Handed(NamedTuple):
    """One value an expression can take, and the calls it came in through -- or a mark that one is not known."""

    value: str
    #: ``file:line`` of each call the value was handed through, the one into
    #: the function asked about first.  Empty where the function says it.
    via: tuple[str, ...] = ()
    #: Not a value: a call into the function hands one this reading cannot
    #: settle -- ``main_key=self.dc_main_key`` -- or passes it through ``*``,
    #: or a call by the function's name is placed nowhere (:meth:`Calls.missed`).
    #: The values beside it are some of the parameter's, not all.  A route
    #: through a call that handed a value reads it; a statement's own fields,
    #: every run any call makes, do not (:func:`known`).
    unknown: bool = False


def known(found: list[Handed]) -> list[Handed]:
    """*found* as a statement's own fields read it: every value, or none where one is not known."""
    return [] if any(one.unknown for one in found) else found


def _any_value(found: list[Handed]) -> bool:
    return any(not one.unknown for one in found)


def handed(resolve, calls: Optional["Calls"]):
    """Widen *resolve* to the values a name is handed, in the function and by its callers.

    *resolve* settles what one expression states -- a constant, a declared
    mapping.  Around it, three ways a statement's bind is given its value
    without stating it:

    * a list, tuple or set literal, element by element -- what an ``IN`` list
      is bound from -- and only where every element settles: ``[a, b]`` with
      ``b`` unknown is not ``{a}``;
    * a local name, through each ``=`` or ``+=`` in the function that writes
      it -- and only where every one of them settles (see ``bindings``);
    * a parameter, through what each call :meth:`Calls.callers` places at the
      function hands it (``calls.bind``), read in the caller the same way --
      every call, however many.

    Returns ``(expression, func, file) -> [Handed, ...]``.  Without *calls*
    only the function's own writes are read.

    A union, not a choice: a statement that three callers hand three lists
    asks, over its runs, for every value any of them names.  Which caller
    asked for which is kept in ``via``.  A call whose value is not settled,
    and a call by the name placed nowhere, are kept too, as marks
    (:attr:`Handed.unknown`): the union of the rest is not every value, and
    ``getCache_JEDI`` read ``main_key=AtlasSites`` alone while
    ``AtlasQueueFillerWatchDog`` hands its own key from an attribute.  A
    route keeps what its call handed; a statement's own fields take the
    union only where no mark is beside it (:func:`known`).
    """
    from bamboo.codemap.calls import Target
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

    #: Each function's bindings, by name, read in one walk: every caller of a
    #: parameter is read, and walking a large caller once per name it binds
    #: was most of the read side's time.
    indexed: dict[int, tuple[ast.AST, dict[str, Optional[list[ast.expr]]]]] = {}

    def bindings(func: ast.AST, name: str) -> Optional[list[ast.expr]]:
        """What each binding of *name* in *func* writes; ``None`` if one writes what cannot be read here.

        All or nothing, because the values are claimed as what the name
        holds: ``pandaID = 0`` before a loop that rebinds it from the rows is
        not a value any statement asks by, and settling the one literal of
        two writes said it was.
        """
        held = indexed.get(id(func))
        if held is None or held[0] is not func:
            held = indexed[id(func)] = (func, written_by(func))
        return held[1].get(name, [])

    def written_by(func: ast.AST) -> dict[str, Optional[list[ast.expr]]]:
        found: dict[str, Optional[list[ast.expr]]] = {}

        def add(name: str, value: ast.expr) -> None:
            values = found.setdefault(name, [])
            if values is not None:
                values.append(value)

        def spoil(names) -> None:
            for name in names:
                found[name] = None

        for node in ast.walk(func):
            if node is func or not in_scope(node, func):
                continue
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        add(target.id, node.value)
                    else:
                        spoil(one.id for one in ast.walk(target) if isinstance(one, ast.Name))
            elif isinstance(node, (ast.AnnAssign, ast.NamedExpr)):
                if isinstance(node.target, ast.Name) and node.value is not None:
                    add(node.target.id, node.value)
            elif isinstance(node, ast.AugAssign):
                if isinstance(node.target, ast.Name):
                    # ``statList += ["x"]`` adds members; ``text += "x"``
                    # makes a different string.
                    if isinstance(node.value, (ast.List, ast.Tuple, ast.Set)):
                        add(node.target.id, node.value)
                    else:
                        spoil([node.target.id])
            elif isinstance(node, (ast.For, ast.AsyncFor, ast.With, ast.AsyncWith, ast.ExceptHandler)):
                targets = (
                    [node.target] if isinstance(node, (ast.For, ast.AsyncFor))
                    else [item.optional_vars for item in node.items if item.optional_vars is not None]
                    if isinstance(node, (ast.With, ast.AsyncWith))
                    else []
                )
                if isinstance(node, ast.ExceptHandler) and node.name:
                    spoil([node.name])
                spoil(one.id for target in targets for one in ast.walk(target) if isinstance(one, ast.Name))
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                spoil([alias.asname or alias.name for alias in node.names])
            elif isinstance(node, (ast.Global, ast.Nonlocal)):
                spoil(node.names)
        return found

    def settle(
        expression: ast.expr, func: ast.AST, file: str, depth: int, seen: frozenset
    ) -> list[Handed]:
        stated = resolve(expression, func)
        if stated:
            return [Handed(value) for value in stated]
        if isinstance(expression, (ast.List, ast.Tuple, ast.Set)):
            # As for a name's writes below: one element unsettled -- or
            # spread in with ``*`` -- and the rest are not every value.
            found = []
            for element in expression.elts:
                settled = [] if isinstance(element, ast.Starred) else settle(element, func, file, depth, seen)
                if not _any_value(settled):
                    return []
                found.extend(settled)
            return found
        if isinstance(expression, ast.BinOp) and isinstance(expression.op, ast.Add) and listed(expression):
            # ``cls.getInputTypes() + ["pp_input"]``: every member of both.
            left = settle(expression.left, func, file, depth, seen)
            right = settle(expression.right, func, file, depth, seen)
            return left + right if _any_value(left) and _any_value(right) else []
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
            if not _any_value(settled):
                return []
            found.extend(settled)
        if expression.id not in parameters(func):
            return found
        if calls is None or depth >= HANDED_DEPTH:
            return []
        owned = calls.types.method_class(file, func)
        target = Target(file, func, owned[1] if owned else None)
        sites = calls.callers(target)
        if not sites:
            return []
        passed_over = [] if written else never_reaching(expression, func, target)
        if calls.missed(target):
            found.append(Handed("", (), unknown=True))
        for site in sites:
            passed = calls.handed(site, target).get(expression.id)
            held = settle(passed, site.func, site.file, depth + 1, seen) if passed is not None else []
            if not _any_value(held):
                held = [Handed("", (), unknown=True)]
            at = f"{site.file}:{site.call.lineno}"
            for one in held:
                via = (at, *one.via)
                if not any(via[: len(route)] == route for route in passed_over):
                    found.append(Handed(one.value, via, one.unknown))
        return found

    def never_reaching(expression: ast.Name, func: ast.AST, target: "Target") -> list[tuple[str, ...]]:
        """The calls whose constant for *expression* rules out the code that reads it.

        ``getEventRanges`` binds ``:datasetID`` from ``segment_id`` under ``if
        segment_id is not None:``, and both its callers hand ``None``: neither
        runs the bind, so ``None`` is not a value the statement is asked with.
        A parameter's value at one place in the function is what the calls
        reaching that place hand it, not what every call hands.  Only a test
        of the parameter itself is ruled on -- :func:`pathcond.decided` --
        so a call ruled out through another argument still hands its value.
        """
        tested = {
            name.id
            for test in pathcond.tests_on(expression, func)
            for name in ast.walk(test)
            if isinstance(name, ast.Name)
        }
        if expression.id not in tested or calls is None:
            return []
        return [
            via
            for via, constants in calls.handed_constants(target, {expression.id})
            if pathcond.ruled_out_by(expression, func, constants)
        ]

    def returned(call: ast.Call, func: ast.AST, file: str, depth: int, seen: frozenset) -> list[Handed]:
        """What the functions *call* lands in return, where each of their returns settles."""
        if calls is None or depth >= HANDED_DEPTH:
            return []
        placed = calls.callees(file, func, call)
        if not placed.targets or placed.unplaced or placed.thread:
            return []
        found: list[Handed] = []
        for target in placed.targets:
            key = (target.file, target.func.lineno, "<return>")  # type: ignore[attr-defined]
            if key in seen:
                return []
            for value in calls_returns(target.func):
                settled = settle(value, target.func, target.file, depth + 1, seen | {key})
                if not _any_value(settled):
                    return []
                found.extend(settled)
        return found

    def settled(expression: ast.expr, func: ast.AST, file: str) -> list[Handed]:
        return settle(expression, func, file, 0, frozenset())

    return settled


# -- what a value is, read as an amount ------------------------------------

#: The clocks a call can read, by what it resolves to through the module's
#: imports: ``datetime.now()`` under ``from datetime import datetime`` and
#: ``datetime.datetime.now()`` under ``import datetime`` are one clock.
#: ``naive_utcnow`` is PanDA's own (``pandacommon``).  ``datetime.now`` is the
#: server's local time; ``datetime.now(utc)`` is UTC.
CLOCKS = {
    "datetime.datetime.utcnow": "datetime.utcnow",
    "datetime.datetime.now": "datetime.now",
}
_UTC = "datetime.timezone.utc"
#: Seconds in each unit ``datetime.timedelta`` takes, by keyword.
TIMEDELTA_UNITS = {
    "weeks": 604800.0,
    "days": 86400.0,
    "hours": 3600.0,
    "minutes": 60.0,
    "seconds": 1.0,
    "milliseconds": 0.001,
    "microseconds": 0.000001,
}
#: ``timedelta``'s positional order.
_TIMEDELTA_POSITIONS = ("days", "seconds", "microseconds", "milliseconds", "minutes", "hours", "weeks")


def _module_of(node: ast.AST) -> Optional[ast.Module]:
    while node is not None and not isinstance(node, ast.Module):
        node = getattr(node, "parent", None)
    return node


def _resolved(expression: ast.expr) -> str:
    """*expression*, a dotted name, as what the module's imports make of it: ``datetime.datetime.now``."""
    dotted = ast.unparse(expression)
    head, _, rest = dotted.partition(".")
    module = _module_of(expression)
    for node in ast.walk(module) if module is not None else ():
        if isinstance(node, ast.Import):
            for alias in node.names:
                if (alias.asname or alias.name.split(".")[0]) == head:
                    origin = alias.name if alias.asname else head
                    return f"{origin}.{rest}" if rest else origin
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            for alias in node.names:
                if (alias.asname or alias.name) == head:
                    return f"{node.module}.{alias.name}" + (f".{rest}" if rest else "")
    return dotted


def clock_of(call: ast.Call) -> str:
    """The clock *call* reads -- ``naive_utcnow``, ``datetime.utcnow``, ``datetime.now``, ``datetime.now(utc)`` -- or ``""``."""
    if call.keywords or not isinstance(call.func, (ast.Name, ast.Attribute)):
        return ""
    resolved = _resolved(call.func)
    if resolved == "naive_utcnow" or resolved.endswith(".naive_utcnow"):
        return "naive_utcnow" if not call.args else ""
    name = CLOCKS.get(resolved, "")
    if not name or not call.args:
        return name
    if name == "datetime.now" and len(call.args) == 1 and _resolved(call.args[0]) == _UTC:
        return "datetime.now(utc)"
    return ""


class Stated(NamedTuple):
    """A value as an amount the source states: a base, plus an offset.

    ``naive_utcnow() - datetime.timedelta(days=7)`` is the clock less 604800
    seconds; ``800`` is a number and no base; ``priority``, a parameter, is a
    base the call fills.  What the source does not settle is *said* -- the
    expression and its line -- or, where only a parameter is missing, the
    parameters it *needs*: ``timedelta(minutes=time_limit)`` is an offset the
    call settles and this function cannot.
    """

    #: The clock (:func:`clock_of`: ``naive_utcnow``, ``datetime.now``, ...), ``<function>.<parameter>``, or empty.
    base: str = ""
    #: Added to *base*: seconds for a time or a duration, the number itself
    #: where there is no base.  ``None`` where it is not settled here.
    offset: Optional[float] = 0.0
    #: A value that is no number, spelled as ``selects`` spells values: ``pending``, ``None``.
    constant: str = ""
    #: The parameters an unsettled offset waits for.
    needs: tuple[str, ...] = ()
    #: ``<expression>@<line>`` where the value is not settled.
    said: str = ""
    #: A ``timedelta`` -- an amount of time with no clock under it.
    duration: bool = False
    #: The value is a number (or a duration): an offset alone, no base and no constant.
    numeric: bool = False


def _said(node: ast.AST, why: str = "") -> Stated:
    text = ast.unparse(node)
    if len(text) > 120:
        text = text[:117] + "..."
    return Stated(offset=None, said=f"{text}@{getattr(node, 'lineno', 0)}" + (f" ({why})" if why else ""))


#: Every name :func:`clock_of` gives.
_CLOCK_NAMES = frozenset({"naive_utcnow", "datetime.utcnow", "datetime.now", "datetime.now(utc)"})


def is_clock(base: str) -> bool:
    """Whether *base* is a clock rather than a parameter."""
    return base in _CLOCK_NAMES


_clock = is_clock


def _needed(value: Stated) -> tuple[str, ...]:
    """The parameters *value* waits for: its own, and its base where that is a parameter."""
    parameter = (value.base,) if value.base and not _clock(value.base) else ()
    return tuple(dict.fromkeys(value.needs + parameter))


def _sum(left: Stated, right: Stated, sign: float, node: ast.AST) -> Stated:
    """``left + right`` or ``left - right``."""
    if left.said or right.said or left.constant or right.constant:
        return _said(node)
    if (left.base and right.base) or (right.base and sign < 0):
        # ``now - then`` is a duration the source does not state.
        return _said(node)
    base = left.base or right.base
    other = right if left.base else left
    if base and _clock(base) and not other.duration:
        return _said(node)  # a clock plus a plain number is not Python
    if not base and left.duration != right.duration:
        return _said(node)
    offset = None if left.offset is None or right.offset is None else left.offset + sign * right.offset
    return Stated(
        base=base,
        offset=offset,
        needs=tuple(dict.fromkeys(left.needs + right.needs)),
        duration=not base and left.duration,
        numeric=not base,
    )


def _scaled(left: Stated, right: Stated, op: ast.operator, node: ast.AST) -> Stated:
    """``left * right`` or ``left / right``: numbers, or a duration and a number."""
    if left.said or right.said or left.constant or right.constant:
        return _said(node)
    if (left.base and _clock(left.base)) or (right.base and _clock(right.base)):
        return _said(node)
    if left.duration and right.duration:
        return _said(node)
    if isinstance(op, (ast.Div, ast.FloorDiv)) and right.duration:
        return _said(node)
    duration = left.duration or right.duration
    if left.base or right.base or left.offset is None or right.offset is None:
        # ``60 * timeout``: an amount waiting for its parameter.
        return Stated(offset=None, needs=tuple(dict.fromkeys(_needed(left) + _needed(right))), duration=duration, numeric=True)
    if isinstance(op, ast.Mult):
        return Stated(offset=left.offset * right.offset, duration=duration, numeric=True)
    if not right.offset:
        return _said(node)
    quotient = left.offset / right.offset
    return Stated(offset=quotient if isinstance(op, ast.Div) else float(quotient // 1), duration=duration, numeric=True)


def _timedelta(call: ast.Call, func, at: ast.AST, depth: int, handed) -> Stated:
    """``datetime.timedelta(...)`` as seconds, or the parameters its amounts wait for."""
    amounts = [*zip(_TIMEDELTA_POSITIONS, call.args, strict=False), *((one.arg, one.value) for one in call.keywords)]
    total, needs = 0.0, ()
    for unit, amount in amounts:
        if unit not in TIMEDELTA_UNITS:
            return _said(call)
        value = stated(amount, func, at, depth + 1, handed)
        if value.said or value.constant or value.duration or (value.base and _clock(value.base)):
            return _said(call)
        if value.base or value.offset is None:
            needs += _needed(value)
        else:
            total += value.offset * TIMEDELTA_UNITS[unit]
    if needs:
        return Stated(offset=None, needs=tuple(dict.fromkeys(needs)), duration=True, numeric=True)
    return Stated(offset=total, duration=True, numeric=True)


def _parameters(func: ast.AST) -> set[str]:
    args = getattr(func, "args", None)
    if args is None:
        return set()
    return {one.arg for one in (*args.posonlyargs, *args.args, *args.kwonlyargs)}


def _named(name: ast.Name, func, at: ast.AST, depth: int, handed) -> Stated:
    """What *name* holds where *at* reads it: the writes reaching it, the call's value among them."""
    own = [one for one in pathcond.bindings(func, name.id) if not one.nested] if func is not None else []
    sites = [one.site for one in own]
    parameter = func is not None and name.id in _parameters(func)
    target = pathcond.site_of(at)
    entry = [func.args] if parameter else []
    reached = reaching(target, [*entry, *sites]) if (sites or entry) else []
    alive = {id(site) for site in reached}
    found: list[Stated] = []
    if parameter and id(func.args) in alive:
        found.append((handed or {}).get(name.id) or Stated(base=f"{func.name}.{name.id}"))
    for one in own:
        if id(one.site) not in alive or not pathcond.can_reach(one.site, target):
            continue
        if one.kind not in (pathcond.BIND_ASSIGN, pathcond.BIND_ANNOTATED) or one.unpacked or one.value is None:
            return _said(name, f"{name.id} is bound by {one.kind} at {getattr(one.site, 'lineno', 0)}")
        found.append(stated(one.value, func, one.site, depth + 1, handed))
    if not found:
        return _said(name, f"no write of {name.id} reaches it")
    if len(set(found)) > 1:
        lines = sorted({getattr(one.site, "lineno", 0) for one in own if id(one.site) in alive})
        lead = "the call's value and " if parameter and id(func.args) in alive else ""
        return _said(name, f"{lead}writes at {', '.join(map(str, lines))} reach it")
    return found[0]


def stated(
    expression: ast.expr,
    func,
    at: Optional[ast.AST] = None,
    depth: int = 0,
    handed: Optional[dict[str, Stated]] = None,
) -> Stated:
    """*expression*, read where *at* evaluates it in *func*, as a :class:`Stated` amount.

    The clock, a ``timedelta`` of constants, a number, and ``+``/``-``/``*``/
    ``/`` of those settle; a plain name is followed through the writes that
    reach the read (:func:`~bamboo.codemap.panda.reaching.reaching`, the call's
    own value counting as one for a parameter), so ``copyArchive``'s
    twenty-six ``timeLimit`` writes give the one line 375 reads.  Anything
    else -- an attribute, a subscript, a call that is not the clock -- is said.

    *handed* is what one call hands *func*'s parameters, read in the caller:
    a parameter whose call value reaches the read is that value instead of
    a base of its own, so ``timedelta(minutes=time_limit)`` under a call
    handing ``30`` is ``-1800`` seconds.
    """
    at = expression if at is None else at
    if depth > 8:
        return _said(expression, "too deep")
    if isinstance(expression, ast.Constant):
        value = expression.value
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return Stated(offset=float(value), numeric=True)
        return Stated(offset=None, constant=value if isinstance(value, str) else repr(value))
    if isinstance(expression, ast.UnaryOp) and isinstance(expression.op, (ast.USub, ast.UAdd)):
        inner = stated(expression.operand, func, at, depth + 1, handed)
        if not inner.numeric or inner.base:
            return _said(expression)
        if isinstance(expression.op, ast.UAdd) or inner.offset is None:
            return inner
        return inner._replace(offset=-inner.offset)
    if isinstance(expression, ast.Call):
        clock = clock_of(expression)
        if clock:
            return Stated(base=clock)
        if ast.unparse(expression.func).split(".")[-1] == "timedelta":
            return _timedelta(expression, func, at, depth, handed)
        return _said(expression)
    if isinstance(expression, ast.BinOp):
        left = stated(expression.left, func, at, depth + 1, handed)
        right = stated(expression.right, func, at, depth + 1, handed)
        if isinstance(expression.op, (ast.Add, ast.Sub)):
            return _sum(left, right, 1.0 if isinstance(expression.op, ast.Add) else -1.0, expression)
        if isinstance(expression.op, (ast.Mult, ast.Div, ast.FloorDiv)):
            return _scaled(left, right, expression.op, expression)
        return _said(expression)
    if isinstance(expression, ast.Name):
        return _named(expression, func, at, depth, handed)
    return _said(expression)
