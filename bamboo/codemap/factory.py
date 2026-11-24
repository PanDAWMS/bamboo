"""Code Map plugin registry.

Mirrors :mod:`bamboo.agents.extractors.factory` so the two plugin systems read
the same way.  Registration happens at import time; the built-in PanDA plugin
is registered lazily on first lookup so importing this module does not pull in
the AST machinery.

A plugin shipped with bamboo declares itself: a package under
``bamboo.codemap`` whose ``__init__`` holds

    CODE_MAP_PLUGIN = ("<map id>", "<module>:<class>")

is found on first lookup and its class imported only when that map is asked
for.  Adding a map is then adding its package, and two maps added apart touch
no file in common.
"""

from __future__ import annotations

import importlib
import logging
import pkgutil

from bamboo.codemap.base import CodeMapPlugin

logger = logging.getLogger(__name__)

# Registry: map_id → plugin class
_plugins: dict[str, type[CodeMapPlugin]] = {}
# map_id → "<module>:<class>" of a declared plugin not imported yet
_declared: dict[str, str] = {}
_discovered = False


def register_code_map_plugin(map_id: str, plugin_class: type[CodeMapPlugin]) -> None:
    """Register *plugin_class* under *map_id*.

    Re-registering the same id replaces the previous entry, which lets tests
    inject a stub.
    """
    _plugins[map_id.lower()] = plugin_class
    logger.debug("Registered Code Map plugin: %s", map_id)


def _discover() -> None:
    """Read ``CODE_MAP_PLUGIN`` off each package under ``bamboo.codemap``; a package's ``__init__`` is all
    that is imported."""
    global _discovered
    if _discovered:
        return
    _discovered = True
    import bamboo.codemap as codemap

    for info in pkgutil.iter_modules(codemap.__path__):
        if not info.ispkg:
            continue
        package = importlib.import_module(f"{codemap.__name__}.{info.name}")
        declared = getattr(package, "CODE_MAP_PLUGIN", None)
        if declared is not None:
            map_id, where = declared
            _declared.setdefault(map_id.lower(), where)


def _register_builtins() -> None:
    if "panda" not in _plugins:
        from bamboo.codemap.panda.plugin import PandaCodeMapPlugin

        register_code_map_plugin("panda", PandaCodeMapPlugin)
    _discover()


def get_code_map_plugin(map_id: str = "panda") -> CodeMapPlugin:
    """Return an instantiated plugin for *map_id*.

    Raises:
        ValueError: If no plugin is registered under that id.
    """
    _register_builtins()
    key = map_id.lower()
    if key not in _plugins and key in _declared:
        module, _, name = _declared[key].partition(":")
        register_code_map_plugin(key, getattr(importlib.import_module(module), name))
    plugin_class = _plugins.get(key)
    if plugin_class is None:
        raise ValueError(
            f"No Code Map plugin registered for {map_id!r}. "
            f"Available: {available_map_ids()}"
        )
    return plugin_class()


def available_map_ids() -> list[str]:
    """Return the registered map ids."""
    _register_builtins()
    return sorted({*_plugins, *_declared})
