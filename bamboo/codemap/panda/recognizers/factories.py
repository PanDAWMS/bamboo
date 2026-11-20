"""The plugins JEDI's factories hand out, and the processes its master starts, as an instance's record configures them.

``WatchDog.__init__`` hands ``FactoryBase.__init__`` the string ``jedi_config.watchdog.modConfig``, and
``FactoryBase.initializeMods`` loads one plugin per vo, label and sub type the string names.  Which plugin a call
of ``getImpl`` returns is in that string alone: the code declares the family (``FactoryBase[WatchDogBase]``), and
every subclass of the base is a candidate until a deployment's configuration is read.  ``JediMaster.start`` reads
``jedi_config.<section>.procConfig`` the same way to start its processes, each handed the fields of one item.

Both strings are the instance's.  The section objects are made at import time from ``panda_jedi.cfg``, so the
values come from the instance record (:func:`~bamboo.codemap.calls.read_recorded`), under the module names the
code reads them by -- ``{"file": "pandajedi/jediconfig/jedi_config.py", "name": "watchdog", "value":
{"modConfig": ..., "procConfig": ...}}``.  Without a record nothing is read.

What the code says is read from the code: which classes hand which recorded string to the factory base
(:meth:`~bamboo.codemap.calls.Calls.callers` of its ``__init__``), which module a process loop launches and the
names it binds each field to.  The two formats are the parsers':
``FactoryBase.initializeMods`` splits items at ``,`` and fields at ``:``, lists at ``|``; ``JediMaster.convParams``
splits fields at ``:`` (the loop itself names the item separator).
"""

from __future__ import annotations

import ast
from typing import Optional

from bamboo.codemap.astwalk import walk
from bamboo.codemap.calls import Calls, CallSite, Target
from bamboo.codemap.models import (
    PluginChoice,
    PluginFactory,
    ProcessField,
    ProcessItem,
    SourceModule,
)

#: The factory base: its ``__init__`` takes the string as :data:`MOD_CONFIG`, and ``initializeMods`` loads it --
#: ``<vos>:<labels>:<module>:<class>[:<sub types>]`` items joined by ``,``, each list ``|``-separated.
FACTORY = ("pandajedi/jedicore/FactoryBase.py", "FactoryBase")
#: The parameter of its ``__init__`` taking the string, and the recorded attribute it is read off.  Which factory
#: a call of its ``getImpl`` is made on is not listed here: ``TaskBrokerThread`` declares its ``implFactory``
#: ``FactoryBase[TaskBrokerBase]``, and read by its class every factory would claim the call.
MOD_CONFIG = "modConfig"
#: The recorded attribute a master's loop reads its processes from.
PROC_CONFIG = "procConfig"
#: The separator ``JediMaster.convParams`` splits an item's fields at.
FIELD_SEPARATOR = ":"
#: The sub type an item naming none is loaded under, and the key a lookup falls back to.
ANY = "any"


def recorded_string(calls: Calls, file: str, expression: Optional[ast.expr], attribute: str) -> Optional[tuple[str, str]]:
    """``(<file>::<name>, string)`` where *expression*, in *file*, is ``<module>.<name>.<attribute>`` and the
    instance's record holds *attribute* of that module's *name* as a string; ``None`` otherwise."""
    if not (
        isinstance(expression, ast.Attribute)
        and expression.attr == attribute
        and isinstance(expression.value, ast.Attribute)
    ):
        return None
    section = expression.value
    module = calls.types.module_named(file, section.value)
    if module is None:
        return None
    held = calls.types.recorded.get((module, section.attr))
    if not isinstance(held, dict) or not isinstance(held.get(attribute), str):
        return None
    return f"{module}::{section.attr}", held[attribute]


def plugin_factories(calls: Calls) -> tuple[list[PluginFactory], list[PluginChoice]]:
    """The factories handed a recorded string, and the plugins each string names (:class:`PluginFactory`,
    :class:`PluginChoice`); both empty where the corpus has no :data:`FACTORY` or the record holds no string."""
    init = _method(calls, "__init__")
    if init is None:
        return [], []
    sites: dict[tuple[str, str], set[str]] = {}
    strings: dict[str, str] = {}
    for site in calls.callers(init):
        found = recorded_string(calls, site.file, calls.handed(site, init).get(MOD_CONFIG), MOD_CONFIG)
        if found is None:
            continue
        config, string = found
        strings[config] = string
        sites.setdefault((_factory(site), config), set()).add(f"{site.file}:{site.call.lineno}")
    factories = [
        PluginFactory(factory=factory, config=config, sites=sorted(held, key=_by_line))
        for (factory, config), held in sorted(sites.items())
    ]
    choices = [choice for config, string in sorted(strings.items()) for choice in _choices(calls, config, string)]
    return factories, choices


def process_items(calls: Calls, modules: list[SourceModule]) -> list[ProcessItem]:
    """One row per item of each recorded :data:`PROC_CONFIG` a loop ``for <v> in <module>.<name>.procConfig.split(<sep>)``
    walks, per loop (:class:`ProcessItem`): the fields named by the loop's ``<name> = <items>[i]`` (or one call of
    it), where ``<items>`` is bound to a call handed ``<v>``."""
    found: list[ProcessItem] = []
    for module in modules:
        for loop in walk(module.tree):
            if not isinstance(loop, ast.For) or not isinstance(loop.target, ast.Name):
                continue
            split = loop.iter
            if not (
                isinstance(split, ast.Call)
                and isinstance(split.func, ast.Attribute)
                and split.func.attr == "split"
                and len(split.args) == 1
                and isinstance(split.args[0], ast.Constant)
                and isinstance(split.args[0].value, str)
            ):
                continue
            held = recorded_string(calls, module.rel_path, split.func.value, PROC_CONFIG)
            if held is None:
                continue
            config, string = held
            names = _field_names(loop, loop.target.id)
            launches = _launched(calls, loop)
            for index, item in enumerate(string.split(split.args[0].value)):
                found.append(
                    ProcessItem(
                        config=config,
                        item=index,
                        loop=f"{module.rel_path}:{loop.lineno}",
                        launches=launches,
                        fields=[
                            ProcessField(index=position, name=names.get(position, ""), text=text)
                            for position, text in enumerate(item.split(FIELD_SEPARATOR))
                        ],
                    )
                )
    return sorted(found, key=lambda one: (one.config, one.loop, one.item))


def _method(calls: Calls, name: str) -> Optional[Target]:
    """:data:`FACTORY`'s own method *name*, as a :class:`Target`."""
    file, class_name = FACTORY
    cls = calls.types.class_def(file, class_name)
    if cls is None:
        return None
    for node in cls.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return Target(file, node, cls)
    return None


def _factory(site: CallSite) -> str:
    """The class the factory object is at a call handing the base's ``__init__`` its string: the class whose method
    calls ``<base>.__init__`` / ``super().__init__`` on its object, else the base, built there."""
    called = site.call.func
    if isinstance(called, ast.Attribute) and called.attr == "__init__":
        owner = _enclosing_class(site.func)
        if owner is not None:
            return f"{site.file}::{owner.name}"
    return "::".join(FACTORY)


def _choices(calls: Calls, config: str, string: str) -> list[PluginChoice]:
    """``FactoryBase.initializeMods``'s parse of *string*: an item of fewer than four fields is skipped, as it logs
    ``wrong config definition`` and goes on."""
    found: list[PluginChoice] = []
    for item in string.split(","):
        fields = item.strip().split(":")
        if len(fields) < 4:
            continue
        plugin = _class_named(calls, fields[2], fields[3])
        sub_types = fields[4].split("|") if len(fields) > 4 else [ANY]
        for vo in fields[0].split("|"):
            for label in fields[1].split("|"):
                for sub_type in sub_types:
                    found.append(PluginChoice(config=config, vo=vo, label=label, sub_type=sub_type, plugin=plugin))
    return found


def _class_named(calls: Calls, module: str, name: str) -> str:
    """``<file>::<name>`` where the corpus defines *name* in *module*, else ``<module>.<name>``."""
    file = calls.types.module_file(module)
    if file is not None and calls.types.class_def(file, name) is not None:
        return f"{file}::{name}"
    return f"{module}.{name}"


def _field_names(loop: ast.For, variable: str) -> dict[int, str]:
    """``{i: name}`` for the loop's ``name = items[i]`` (or ``items[i] if ... else ...``, or a call handed
    ``items[i]`` alone -- ``vos = parse_init_params(items[0])``), first by line, where ``items`` is bound to a call
    handed *variable*."""
    statements = sorted(
        (node for node in walk(loop) if isinstance(node, ast.Assign) and len(node.targets) == 1),
        key=lambda node: (node.lineno, node.col_offset),
    )
    items = {
        node.targets[0].id
        for node in statements
        if isinstance(node.targets[0], ast.Name)
        and isinstance(node.value, ast.Call)
        and any(isinstance(one, ast.Name) and one.id == variable for one in node.value.args)
    }
    names: dict[int, str] = {}
    for node in statements:
        if not isinstance(node.targets[0], ast.Name):
            continue
        value = node.value.body if isinstance(node.value, ast.IfExp) else node.value
        if isinstance(value, ast.Call) and len(value.args) == 1 and not value.keywords:
            value = value.args[0]
        if (
            isinstance(value, ast.Subscript)
            and isinstance(value.value, ast.Name)
            and value.value.id in items
            and isinstance(value.slice, ast.Constant)
            and isinstance(value.slice.value, int)
        ):
            names.setdefault(value.slice.value, node.targets[0].id)
    return names


def _launched(calls: Calls, loop: ast.For) -> str:
    """The first string in the loop, by line, naming a module of the corpus: what it hands the launcher."""
    strings = sorted(
        (node.lineno, node.col_offset, node.value)
        for node in walk(loop)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    )
    return next((text for _line, _column, text in strings if calls.types.module_file(text) is not None), "")


def _enclosing_class(node: ast.AST) -> Optional[ast.ClassDef]:
    """The class whose body holds *node*'s function, directly."""
    parent = getattr(node, "parent", None)
    return parent if isinstance(parent, ast.ClassDef) else None


def _by_line(place: str) -> tuple[str, int]:
    file, _, line = place.rpartition(":")
    return file, int(line)
