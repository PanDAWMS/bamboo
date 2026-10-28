"""Where a value came from, when the answer is outside this map.

The terminal classification the backward walk needs and the map's own edges
cannot give: an expression is a call on an interface, and which interface it
is decides whether the answer is "DDM said so", "a configuration row said
so", or "another junction decided it and the map names which".

**Named interfaces are what makes this static.**  PanDA reaches everything
outside itself through a handful of attributes held on ``self`` --
``ddmIF``, ``siteMapper``, ``taskBufferIF`` -- so the receiver, once resolved,
is the classification.  This is the plan's provenance table, and the reason it
lives in the plugin rather than beside the walk: the walk is about Python, and
these names are about PanDA.

**Resolved, not matched.**  The obvious implementation is a substring test for
``ddmIF`` in the source text, and a prototype that did that put the config
terminal on 545 arms and the external ones on 34.  Those counts are a lower
bound and nothing more: ``ddmIF = self.ddmIF.getInterface(vo, cloud)`` binds a
bare local that no ``self.`` test would match, and the corpus has punished
name matching twice already.  So the receiver chain is walked to its base and,
when the base is a local, one reaching definition is followed -- which is what
turns ``ddmIF.getDatasetMetaData(...)`` into DDM.

**Declared first, named second.**  A receiver whose class the code declares --
annotated, constructed, returned by a method with a return annotation -- is
classified by that class (:data:`INTERFACE_CLASSES`), and the attribute names
below answer only for what nothing declares yet.  The names are the older rule
and the weaker one: ``self.site_mapper`` is the same ``SiteMapper`` as
``self.siteMapper`` and was classified as nothing.
"""

from __future__ import annotations

import ast
import builtins
from typing import TYPE_CHECKING, Callable, NamedTuple, Optional

from bamboo.codemap.astwalk import walk
from bamboo.codemap.models import STOP_CONFIG, STOP_EXTERNAL, STOP_UPSTREAM
from bamboo.codemap.panda.pathcond import names_bound_by, targets_of, written_value

if TYPE_CHECKING:
    from bamboo.codemap.calls import Types

#: The class a receiver is declared to be -> the system on the other side of it.
#:
#: **What the object is, not what someone called it.**  ``self.site_mapper`` and
#: ``self.siteMapper`` are the same ``SiteMapper`` and the spelling table below
#: knew one of them: the setupper walked straight through ``getSite`` where the
#: adder stopped on it, and the two functions deciding ``src == dst`` looked as
#: though they read different things.  The class is read the way a type checker
#: reads it -- an annotation, a constructor, a method's return annotation -- by
#: the walk, which hands it here as ``typed``; the key is the file that declares
#: the class, so a class of the same name elsewhere is not this one.
INTERFACE_CLASSES = {
    "pandaserver/brokerage/SiteMapper.py::SiteMapper": "CRIC site declaration",
    # ``rucioAPI = RucioAPI()`` at the foot of ``ddm.py``, imported by name
    # everywhere the server registers, attaches or erases a dataset.  The
    # attribute table's ``ddmIF`` is JEDI's; the server's never had a name.
    "pandaserver/dataservice/ddm.py::RucioAPI": "DDM / Rucio",
}

#: The system -> the package PanDA reaches it through, where PanDA calls it.
#: Declared beside the interface, because "outside the corpus" alone takes in
#: pandacommon's logger as readily as the Rucio client.  ``SiteMapper`` reads
#: CRIC's declarations already loaded and calls nothing out, so it has none.
OUTSIDE_PACKAGES = {
    "DDM / Rucio": ("rucio",),
}

#: ``self.<attribute>`` -> the system on the other side of it.
#:
#: The spelling, kept where the class cannot be read yet.  JEDI builds its
#: ``self.siteMapper`` from ``self.taskBufferIF.get_site_mapper()``, and that
#: interface forwards every call through ``__getattr__ -> Any``, so no
#: declaration says what comes back; ``siteMapper`` stays here until PanDA
#: states it, and ``ddmIF`` until DDM's classes are read the same way.
EXTERNAL_INTERFACES = {
    "ddmIF": "DDM / Rucio",
    "siteMapper": "CRIC site declaration",
}

#: Interfaces that land back in PanDA's own database.  A terminal for the
#: trace and a continuation for the map, which is the point of naming it
#: separately: the walk stops here and ``producers_of`` takes over.
#:
#: ``cur`` is the Oracle cursor the proxy modules read rows through, and it is
#: the single commonest place a walk ends -- 120 steps of 4852.  Without it
#: those come back as "chosen by whoever called or built this", which is true
#: of the cursor object and false of the row: the row came from the statement
#: executed a few lines above, and the map has read that statement.
DATABASE_INTERFACES = {"taskBufferIF", "taskBuffer", "cur"}

#: The task buffer's facades -> the attribute each borrows a proxy from, and
#: the proxy class whose methods run the statements.
#:
#: Each method of ``TaskBuffer`` and ``JediTaskBuffer`` borrows a connection
#: from ``self.proxyPool`` and calls methods on it, so what a method asks the
#: database is what it calls on the proxy (:func:`facade_calls`).  Mostly that
#: is the proxy's method of the same name, handed the same arguments -- 224 of
#: 265 ``TaskBuffer`` methods and 145 of 152 ``JediTaskBuffer`` ones -- but not
#: always, so it is read from the method rather than assumed from its name:
#: ``peekJobs`` calls ``peekJob`` once per job, ``querySQL`` calls
#: ``querySQLS``, and ``getDatasetsWithJediTaskID_JEDI`` calls two.  The JEDI
#: facade's pool hands out ``JediDBProxy``, a subclass that adds nothing a
#: facade calls.
FACADES = {
    "pandaserver/taskbuffer/TaskBuffer.py::TaskBuffer": ("proxyPool", "pandaserver/taskbuffer/OraDBProxy.py::DBProxy"),
    "pandajedi/jedicore/JediTaskBuffer.py::JediTaskBuffer": ("proxyPool", "pandajedi/jedicore/JediDBProxy.py::DBProxy"),
}
_POOLS = frozenset(pool for pool, _proxy in FACADES.values())

_CONFIG_CALL = "getConfigValue"
_BUILTINS = frozenset(dir(builtins))
_CONFIG_MODULE = "jedi_config"

Resolver = Callable[[str], Optional[ast.expr]]
#: ``self.<field>`` or a local -> ``"<file>::<Class>"`` it is declared to be, or ``None``.
Typer = Callable[[ast.expr], Optional[str]]


def _text(node: ast.AST) -> str:
    try:
        return ast.unparse(node)
    except Exception:  # noqa: BLE001 -- unparse fails on synthesised nodes
        return "..."


def _called(func: ast.expr) -> str:
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return ""


def _config_key(call: ast.Call) -> str:
    """The arguments of a config lookup, which are its key, each as the code spells it.

    A template key comes back as written -- ``PENDING_TIMEOUT_{prodSourceLabel}``
    -- and a name as ``{vo}``: rendering them would need their values, which
    is another walk, and the key as written is what a person changes.  They
    were dropped before, so ``getConfigValue("watchdog",
    f"PENDING_TIMEOUT_{prodSourceLabel}", "jedi", vo)`` said ``watchdog.jedi``,
    a key that is not in the table.
    """
    parts: list[str] = []
    for argument in [*call.args, *(keyword.value for keyword in call.keywords)]:
        if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
            parts.append(argument.value)
        elif isinstance(argument, ast.JoinedStr):
            parts.append(
                "".join(
                    piece.value if isinstance(piece, ast.Constant) else "{" + _text(piece.value) + "}"
                    for piece in argument.values
                )
            )
        else:
            parts.append("{" + _text(argument) + "}")
    return ".".join(parts) if parts else _text(call)


def _base(attribute: ast.Attribute) -> tuple[Optional[ast.expr], str]:
    """Descend a dotted chain to its base, with the attribute nearest to it."""
    node: ast.expr = attribute
    nearest = ""
    while isinstance(node, ast.Attribute):
        nearest = node.attr
        node = node.value
    return node, nearest


def _self_field(attribute: ast.Attribute) -> Optional[ast.Attribute]:
    """The ``self.<field>`` link a chain rooted at ``self`` starts with."""
    node: ast.expr = attribute
    while isinstance(node, ast.Attribute):
        if isinstance(node.value, ast.Name) and node.value.id == "self":
            return node
        node = node.value
    return None


def _rooted_in(
    attribute: ast.Attribute,
    resolve: Optional[Resolver],
    hops: int = 1,
    typed: Optional[Typer] = None,
) -> Optional[tuple[str, str, str]]:
    """``(self.<field>, the call on it, the declared class)`` a dotted chain is rooted in.

    The declared class is the root's -- ``self.<field>``'s, or a local's where
    the chain starts at one -- and empty where nothing declares it.  A local
    whose class is not an interface is still followed one hop: ``src_site_spec``
    is a ``SiteSpec``, and it came out of ``self.siteMapper.getSite(...)``.

    Both halves, because a hop through a local moves the interesting name.
    ``taskSpec.getNumFilesPerJob()`` where ``taskSpec`` came out of
    ``self.taskBufferIF.getTaskWithID_JEDI(...)`` is the database's, and the
    call to name is the one that went to the database -- reporting
    ``getNumFilesPerJob`` instead would be a sentence that is false about
    where the value came from.
    """
    base, nearest = _base(attribute)
    if not isinstance(base, ast.Name):
        return None
    if base.id == "self":
        field = _self_field(attribute)
        declared = (typed(field) if typed is not None and field is not None else None) or ""
        return nearest, attribute.attr, declared
    declared = (typed(base) if typed is not None else None) or ""
    if declared in INTERFACE_CLASSES:
        return "", attribute.attr, declared
    if hops <= 0 or resolve is None:
        return None
    bound = resolve(base.id)
    if bound is None:
        return None
    target = bound.func if isinstance(bound, ast.Call) else bound
    if isinstance(target, ast.Attribute):
        return _rooted_in(target, resolve, hops - 1, typed)
    return None


def _terminal_of(
    node: ast.AST, resolve: Optional[Resolver], typed: Optional[Typer] = None
) -> Optional[tuple[str, str]]:
    if isinstance(node, ast.Call) and _called(node.func) == _CONFIG_CALL:
        return STOP_CONFIG, _config_key(node)
    if isinstance(node, ast.Attribute):
        base, _ = _base(node)
        if isinstance(base, ast.Name) and base.id == _CONFIG_MODULE:
            return STOP_CONFIG, _text(node)
        rooted = _rooted_in(node, resolve, typed=typed)
        if rooted is None:
            return None
        field, call, declared = rooted
        if declared in INTERFACE_CLASSES:
            return STOP_EXTERNAL, INTERFACE_CLASSES[declared]
        if field in EXTERNAL_INTERFACES:
            return STOP_EXTERNAL, EXTERNAL_INTERFACES[field]
        if field in DATABASE_INTERFACES:
            return STOP_UPSTREAM, call if call != field else "the task buffer"
    return None


def database_call(
    expression: ast.expr,
    resolve: Optional[Resolver] = None,
    typed: Optional[Typer] = None,
) -> Optional[ast.Call]:
    """The call :func:`classify` names as the one that went to the database, as a node; ``None`` if it names none.

    Read in :func:`classify`'s order, so the call is the one its answer
    names: ``self.taskBufferIF.getPandaIDsWithTask_JEDI(jediTaskID, True)``
    itself, or the call a name was bound from -- ``taskSpec`` in
    ``taskSpec.getNumFilesPerJob()`` came out of ``getTaskWithID_JEDI(...)``,
    one hop, as :func:`_rooted_in` follows it.  What it hands the method is
    what the method's statements are read with.
    """
    queue: list[tuple[ast.AST, Optional[ast.Call]]] = [(expression, None)]
    while queue:
        node, around = queue.pop(0)
        found = _terminal_of(node, resolve, typed)
        if found is not None:
            if found[0] != STOP_UPSTREAM:
                return None
            method = found[1]
            if around is not None and isinstance(around.func, ast.Attribute) and around.func.attr == method:
                return around
            if isinstance(node, ast.Attribute) and resolve is not None:
                base, _ = _base(node)
                bound = resolve(base.id) if isinstance(base, ast.Name) and base.id != "self" else None
                if isinstance(bound, ast.Call) and isinstance(bound.func, ast.Attribute) and bound.func.attr == method:
                    return bound
            return None
        if isinstance(node, ast.Call) and not _hands_its_arguments_on(node):
            queue.append((node.func, node))
            continue
        queue.extend(
            (child, node if isinstance(node, ast.Call) and child is node.func else None)
            for child in ast.iter_child_nodes(node)
        )
    return None


def _hands_its_arguments_on(call: ast.Call) -> bool:
    """Whether a call's value is made of its arguments, so they say where it came from.

    ``int(getConfigValue(...))`` is the configuration's value, converted.
    ``self.adder_plugin_class(self.job, taskBuffer=self.taskBuffer)`` is a
    plugin built *with* the task buffer, and nothing the database returned: a
    function of PanDA's own makes its value in its body, and the walk goes
    there.  So the arguments are read for a builtin, and for a call on a value
    -- ``x.split(":")``, ``d.get(k)`` -- and not for a call on ``self`` or on a
    name the code defines.
    """
    called = call.func
    if isinstance(called, ast.Name):
        return called.id in _BUILTINS
    if isinstance(called, ast.Attribute):
        root = called.value
        while isinstance(root, ast.Attribute):
            root = root.value
        return not (isinstance(root, ast.Name) and root.id == "self")
    return True


def classify(
    expression: ast.expr,
    resolve: Optional[Resolver] = None,
    typed: Optional[Typer] = None,
) -> tuple[str, str]:
    """``(terminal, detail)`` for *expression*, or ``("", "")`` to keep walking.

    Breadth-first, so the outermost construct wins: in
    ``res = self.taskBufferIF.insertFiles(getConfigValue("jedi", "x"))`` the
    value came from the database call and the config key is an argument to it.
    Returning one rather than all of them keeps the step's answer a single
    category, which is what "where did this come from" means.

    A call's arguments are read only where they make its value -- see
    :func:`_hands_its_arguments_on`.  Read everywhere, one argument that was
    the task buffer made a plugin's construction "a value the database
    supplies", and the walk stopped at the object whose ``result`` it was
    asking about.
    """
    queue: list[ast.AST] = [expression]
    while queue:
        node = queue.pop(0)
        found = _terminal_of(node, resolve, typed)
        if found is not None:
            return found
        if isinstance(node, ast.Call) and not _hands_its_arguments_on(node):
            queue.append(node.func)
            continue
        queue.extend(ast.iter_child_nodes(node))
    return "", ""


class FacadeCall(NamedTuple):
    """One call a facade method makes on the proxy it borrowed, and the proxy's method that runs it."""

    file: str
    method: ast.FunctionDef | ast.AsyncFunctionDef
    call: ast.Call
    proxy_file: str
    proxy_method: ast.FunctionDef | ast.AsyncFunctionDef


def rooted_at_pool(node: ast.expr) -> bool:
    """True for ``self.proxyPool.get()`` and friends, however deep: a pool :data:`FACADES` names."""
    while isinstance(node, (ast.Call, ast.Attribute)):
        if isinstance(node, ast.Call):
            node = node.func
        else:
            if node.attr in _POOLS:
                return True
            node = node.value
    return False


def borrowed_proxies(func: ast.FunctionDef | ast.AsyncFunctionDef) -> frozenset[str]:
    """Names *func* binds to a borrowed proxy.

    Both spellings are in the corpus: ``with self.proxyPool.get() as proxy``
    (JediTaskBuffer) and ``proxy = self.proxyPool.getProxy()`` (TaskBuffer).
    """
    bound: set[str] = set()
    for node in walk(func):
        if isinstance(node, ast.With):
            for item in node.items:
                if item.optional_vars is not None and rooted_at_pool(item.context_expr):
                    bound.update(name for name, _slot, unpacked in names_bound_by(item.optional_vars) if not unpacked)
        elif (borrowed := written_value(node)) is not None and rooted_at_pool(borrowed):
            bound.update(
                name
                for target in targets_of(node)
                for name, _slot, unpacked in names_bound_by(target)
                if not unpacked
            )
    return frozenset(bound)


def facade_calls(types: Types, method: str) -> list[FacadeCall]:
    """What the facades' *method* calls on the proxy, in the order it is written; ``[]`` where no facade has one.

    Every facade's *method*, its own or inherited -- ``JediTaskBuffer``
    inherits most of ``TaskBuffer`` -- because a stop is found by the name of
    the receiver, which does not say which facade it is.  A call the proxy has
    no method for is left out, and a method two facades share is listed once.
    """
    found: list[FacadeCall] = []
    seen: set[int] = set()
    for facade, (_pool, proxy) in FACADES.items():
        file, name = facade.split("::")
        cls = types.class_def(file, name)
        held = types.method(file, cls, method) if cls is not None else None
        if held is None:
            continue
        where, func = held
        proxy_file, proxy_name = proxy.split("::")
        proxy_cls = types.class_def(proxy_file, proxy_name)
        if proxy_cls is None:
            continue
        borrowed = borrowed_proxies(func)
        made = sorted(
            (
                node
                for node in walk(func)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in borrowed
            ),
            key=lambda node: (node.lineno, node.col_offset),
        )
        for call in made:
            runs = types.method(proxy_file, proxy_cls, call.func.attr)  # type: ignore[union-attr]
            if runs is None or id(call) in seen:
                continue
            seen.add(id(call))
            found.append(FacadeCall(where, func, call, runs[0], runs[1]))
    return found
