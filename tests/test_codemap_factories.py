"""The plugins JEDI's factories hand out and the processes its master starts, read from an instance's record."""

from __future__ import annotations

import json
from pathlib import Path

from bamboo.codemap import calls as calls_mod
from bamboo.codemap.calls import parsed
from bamboo.codemap.models import SourceModule
from bamboo.codemap.panda.recognizers import factories

_FILES = {
    "jediconfig/jedi_config.py": (
        "class _SectionClass:\n"
        "    pass\n"
        "\n"
        "watchdog: _SectionClass\n"
        "jobgen: _SectionClass\n"
    ),
    "jedicore/FactoryBase.py": (
        "class FactoryBase:\n"
        "    def __init__(self, vos, sourceLabels, logger, modConfig):\n"
        "        self.modConfig = modConfig\n"
        "\n"
        "    def getImpl(self, vo, sourceLabel, subType='any'):\n"
        "        return None\n"
    ),
    "jedidog/ProdDog.py": "class ProdDog:\n    pass\n",
    "jedidog/UserDog.py": "class UserDog:\n    pass\n",
    "jediorder/WatchDog.py": (
        "from pandajedi.jediconfig import jedi_config\n"
        "from pandajedi.jedicore.FactoryBase import FactoryBase\n"
        "\n"
        "class WatchDog(FactoryBase):\n"
        "    def __init__(self, vos, labels):\n"
        "        FactoryBase.__init__(self, vos, labels, None, jedi_config.watchdog.modConfig)\n"
        "\n"
        "class Literal(FactoryBase):\n"
        "    def __init__(self):\n"
        "        FactoryBase.__init__(self, None, None, None, 'atlas:any:pandajedi.jedidog.ProdDog:ProdDog')\n"
        "\n"
        "def make():\n"
        "    return FactoryBase(vos=None, sourceLabels=None, logger=None, modConfig=jedi_config.watchdog.modConfig)\n"
    ),
    "jediorder/JediMaster.py": (
        "from pandajedi.jediconfig import jedi_config\n"
        "\n"
        "class JediMaster:\n"
        "    def convParams(self, itemStr):\n"
        "        return itemStr.split(':')\n"
        "\n"
        "    def start(self, launch):\n"
        "        for itemStr in jedi_config.jobgen.procConfig.split(';'):\n"
        "            items = self.convParams(itemStr)\n"
        "            vo = items[0]\n"
        "            plabel = items[1]\n"
        "            nProc = items[2]\n"
        "            cloud = items[3]\n"
        "            try:\n"
        "                loop_cycle = items[4]\n"
        "            except IndexError:\n"
        "                loop_cycle = None\n"
        "            launch('pandajedi.jediorder.WatchDog', vo, plabel, nProc, cloud, loop_cycle)\n"
        "        for configStr in jedi_config.watchdog.modConfig.split(','):\n"
        "            launch(configStr)\n"
    ),
    "jedimsgprocessor/msg.py": (
        "from pandajedi.jediconfig import jedi_config\n"
        "\n"
        "def convert(text):\n"
        "    return text.split(':')\n"
        "\n"
        "def parse(text):\n"
        "    return text.split('|')\n"
        "\n"
        "def initialize(found):\n"
        "    for itemStr in jedi_config.watchdog.procConfig.split(';'):\n"
        "        items = convert(itemStr)\n"
        "        vos = parse(items[0])\n"
        "        labels = parse(items[1])\n"
        "        found.append((vos, labels))\n"
    ),
}

_RECORD = {
    "instance": "one",
    "values": [
        {
            "file": "pandajedi/jediconfig/jedi_config.py",
            "name": "watchdog",
            "value": {
                "modConfig": (
                    "atlas:managed|test:pandajedi.jedidog.ProdDog:ProdDog,"
                    " atlas:user:pandajedi.jedidog.UserDog:UserDog,"
                    "any:any:elsewhere.Dog:Dog:async|sync,"
                    "wrong:item"
                ),
                "procConfig": "atlas:user:1:UserDog:60",
            },
        },
        {
            "file": "pandajedi/jediconfig/jedi_config.py",
            "name": "jobgen",
            "value": {"procConfig": "atlas:managed|test:2:WORLD:180;atlas:user:1::180"},
        },
    ],
}


def _tree(tmp_path: Path) -> dict[str, Path]:
    root = tmp_path / "pandajedi"
    for rel, text in {"__init__.py": "", **_FILES}.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        (path.parent / "__init__.py").touch()
        path.write_text(text)
    return {"pandajedi": root}


def _modules(roots: dict[str, Path]) -> list[SourceModule]:
    root = roots["pandajedi"]
    found = []
    for path in sorted(root.rglob("*.py")):
        source, tree = parsed(path)  # type: ignore[misc]
        found.append(
            SourceModule(
                package="pandajedi",
                rel_path=f"pandajedi/{path.relative_to(root)}",
                tree=tree,
                source=source,
                blob_sha="",
            )
        )
    return found


def test_a_factory_s_plugins_and_a_process_list_s_items_are_read_from_the_instance_s_record(tmp_path):
    """``WatchDog`` hands ``FactoryBase.__init__`` ``jedi_config.watchdog.modConfig``, and ``make`` builds a
    ``FactoryBase`` with it: two factories of one recorded string, named by the class the object is.  The string's
    items are the plugins by vo, label and sub type -- ``any`` where an item names none, a dotted path where the
    corpus has no such class, an item of fewer than four fields skipped as ``initializeMods`` skips it; a literal
    string is no record's.  ``JediMaster.start``'s loop over ``jedi_config.jobgen.procConfig`` reads one row per
    item, its fields named by the loop's ``name = items[i]`` (one in a ``try``), an empty field kept as text, and
    the module it launches; ``msg.initialize`` reads ``watchdog``'s list too, naming fields through one call
    (``vos = parse(items[0])``) and launching nothing; a loop over a ``modConfig`` is no process list.  Without a
    record, nothing."""
    roots = _tree(tmp_path)
    record = tmp_path / "record.json"
    record.write_text(json.dumps(_RECORD))
    recorded, _stamp = calls_mod.read_recorded(record)
    calls = calls_mod.Calls(calls_mod.Types(roots, {}, recorded))

    found, choices = factories.plugin_factories(calls)
    config = "pandajedi/jediconfig/jedi_config.py::watchdog"
    assert [(one.factory, one.config, one.sites) for one in found] == [
        ("pandajedi/jedicore/FactoryBase.py::FactoryBase", config, ["pandajedi/jediorder/WatchDog.py:13"]),
        ("pandajedi/jediorder/WatchDog.py::WatchDog", config, ["pandajedi/jediorder/WatchDog.py:6"]),
    ]
    assert [(one.vo, one.label, one.sub_type, one.plugin) for one in choices] == [
        ("atlas", "managed", "any", "pandajedi/jedidog/ProdDog.py::ProdDog"),
        ("atlas", "test", "any", "pandajedi/jedidog/ProdDog.py::ProdDog"),
        ("atlas", "user", "any", "pandajedi/jedidog/UserDog.py::UserDog"),
        ("any", "any", "async", "elsewhere.Dog.Dog"),
        ("any", "any", "sync", "elsewhere.Dog.Dog"),
    ]
    assert {one.config for one in choices} == {config}

    items = factories.process_items(calls, _modules(roots))
    assert [(one.config, one.item, one.loop, one.launches) for one in items] == [
        ("pandajedi/jediconfig/jedi_config.py::jobgen", 0, "pandajedi/jediorder/JediMaster.py:8", "pandajedi.jediorder.WatchDog"),
        ("pandajedi/jediconfig/jedi_config.py::jobgen", 1, "pandajedi/jediorder/JediMaster.py:8", "pandajedi.jediorder.WatchDog"),
        (config, 0, "pandajedi/jedimsgprocessor/msg.py:10", ""),
    ]
    assert [[(field.index, field.name, field.text) for field in one.fields] for one in items] == [
        [(0, "vo", "atlas"), (1, "plabel", "managed|test"), (2, "nProc", "2"), (3, "cloud", "WORLD"), (4, "loop_cycle", "180")],
        [(0, "vo", "atlas"), (1, "plabel", "user"), (2, "nProc", "1"), (3, "cloud", ""), (4, "loop_cycle", "180")],
        [(0, "vos", "atlas"), (1, "labels", "user"), (2, "", "1"), (3, "", "UserDog"), (4, "", "60")],
    ]

    bare = calls_mod.Calls(calls_mod.Types(roots, {}))
    assert factories.plugin_factories(bare) == ([], [])
    assert factories.process_items(bare, _modules(roots)) == []
