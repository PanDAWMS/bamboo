"""A value read as an amount: the clock, a ``timedelta``, a number, a parameter (``values.stated``).

The querier read ``modificationTime<:timeLimit@375`` by regex and then the
Python itself, one plain-name hop, to learn the bind was the clock less seven
days.  These pin the reading the map does instead, on the shapes the corpus
census found: a bind written from the clock directly, through a local written
many times, from a parameter, and from what the source does not settle.
"""

from __future__ import annotations

import ast

from bamboo.codemap.panda import pathcond, values

_SOURCE = '''
import datetime
from datetime import datetime as moment

def f(self, timeLimit, time_limit, days=None):
    varMap = {}
    varMap[":direct"] = naive_utcnow() - datetime.timedelta(days=7)
    timeLimit2 = naive_utcnow() - datetime.timedelta(hours=1)
    timeLimit2 = naive_utcnow() - datetime.timedelta(hours=3)
    varMap[":local"] = timeLimit2
    varMap[":needs"] = naive_utcnow() - datetime.timedelta(minutes=time_limit)
    varMap[":parameter"] = timeLimit
    varMap[":handed"] = timeLimit - datetime.timedelta(hours=1)
    if days is None:
        days = 3
    varMap[":either"] = days
    varMap[":number"] = 6 * 60 * 60
    varMap[":config"] = self.config.limit
    varMap[":rounded"] = (naive_utcnow() - datetime.timedelta(hours=1)).replace(second=0)
    varMap[":word"] = "pending"
    varMap[":local_now"] = moment.now()
    varMap[":utc_now"] = datetime.datetime.now(datetime.timezone.utc)
    varMap[":today"] = datetime.date.today()
'''


def _stated() -> dict[str, values.Stated]:
    tree = ast.parse(_SOURCE)
    pathcond.attach_parents(tree)
    func = tree.body[2]
    found = {}
    for node in ast.walk(func):
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Subscript):
            found[node.targets[0].slice.value] = values.stated(node.value, func, node)
    return found


def test_the_clock_less_a_constant_is_the_clock_and_an_offset_in_seconds():
    said = _stated()
    assert (said[":direct"].base, said[":direct"].offset) == ("naive_utcnow", -604800.0)
    # The write that reaches the bind, not every write of the name.
    assert (said[":local"].base, said[":local"].offset) == ("naive_utcnow", -10800.0)


def test_a_clock_is_named_by_what_the_call_resolves_to():
    """``moment.now()`` under ``from datetime import datetime as moment`` is ``datetime.now``, the server's local time."""
    said = _stated()
    assert said[":local_now"].base == "datetime.now"
    assert said[":utc_now"].base == "datetime.now(utc)"
    assert said[":today"].said.startswith("datetime.date.today()")


def test_a_parameter_is_a_base_or_what_an_offset_needs():
    said = _stated()
    assert said[":parameter"].base == "f.timeLimit"
    assert (said[":handed"].base, said[":handed"].offset) == ("f.timeLimit", -3600.0)
    assert (said[":needs"].base, said[":needs"].offset, said[":needs"].needs) == ("naive_utcnow", None, ("f.time_limit",))


def test_what_the_source_does_not_settle_is_said_not_guessed():
    said = _stated()
    # The call's value and the default both reach the bind.
    assert said[":either"].said.startswith("days@")
    assert said[":config"].said == "self.config.limit@18"
    assert said[":rounded"].said.startswith("(naive_utcnow()")
    assert (said[":number"].offset, said[":number"].numeric) == (21600.0, True)
    assert said[":word"].constant == "pending"
