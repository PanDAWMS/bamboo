"""Checks that every backend shipped here still satisfies the backend contract.

The contract is an ABC, so Python already enforces it -- but only at the moment
something is instantiated, and nothing instantiated the example backend.  Adding
``merge_map_node`` and ``clear_map`` (since removed) to ``GraphDatabaseBackend`` for the Code Map
therefore broke ``InMemoryGraphBackend``, the reference implementation the
database-plugin pages tell a contributor to copy, and 818 tests stayed green.

These instantiate every backend in the package so that the ABC gets to do its job
in CI rather than in a plugin author's editor.  Instantiation is deliberately the
whole test: a hand-maintained list of required method names would be a third
expression of the contract, free to drift from the other two.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from typing import Iterator

import pytest

import bamboo.database.backends as backends_pkg
from bamboo.database.base import GraphDatabaseBackend, VectorDatabaseBackend


def _shipped_backends() -> Iterator[type]:
    """Every concrete backend under ``bamboo.database.backends``, examples included.

    Discovered rather than listed: a backend added to the package is covered
    without anyone remembering to add it here, which is the failure this file
    exists to stop repeating.
    """
    for info in pkgutil.walk_packages(
        backends_pkg.__path__, prefix=f"{backends_pkg.__name__}."
    ):
        module = importlib.import_module(info.name)
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if obj.__module__ != info.name:
                continue
            if not issubclass(obj, (GraphDatabaseBackend, VectorDatabaseBackend)):
                continue
            if inspect.isabstract(obj):
                continue
            yield obj


SHIPPED = sorted(_shipped_backends(), key=lambda c: c.__name__)


def test_the_package_ships_backends_to_check() -> None:
    """Guards the discovery itself: a walk that silently finds nothing would
    make every check below pass without checking anything."""
    names = [c.__name__ for c in SHIPPED]
    assert "Neo4jBackend" in names
    assert "InMemoryGraphBackend" in names


@pytest.mark.parametrize("backend_class", SHIPPED, ids=lambda c: c.__name__)
def test_a_shipped_backend_implements_its_whole_contract(backend_class: type) -> None:
    """Constructing is the check -- an ABC refuses a subclass with a method missing.

    None of these connect in ``__init__`` (the graph client connects lazily on
    first query), so this costs nothing and needs no services.
    """
    backend_class()
