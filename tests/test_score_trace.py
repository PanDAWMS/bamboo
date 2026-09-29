"""The audit's own reading of the source, where the walk and the check must agree.

``score_trace`` has no other tests: it is a scorer, and what it scores is read
back by eye from the worksheet.  This one is here because the check and the
walk hold the same rule twice -- P1-57 taught the walk that a container is
filled by ``.add`` and the check went on counting only assignments, which made
twelve honest steps read as fabrications.
"""

import ast

from bamboo.scripts.panda.score_trace import _binding_lines

SWEEPER = """\
def run(self):
    skipped = set()
    rows = []
    for report in self.reports():
        skipped.add(int(report.id))
        rows.append(report)
    rows = self.sort(rows)
"""


def _func() -> ast.FunctionDef:
    return ast.parse(SWEEPER).body[0]


def test_a_container_fill_is_a_binding_the_source_has():
    """``skipped.add(...)`` puts a value there, so a step naming line 5 is sound."""
    assert _binding_lines(_func(), "skipped") == {2: "=", 5: ".add"}


def test_a_container_that_is_also_assigned_keeps_both_shapes():
    """The check names every binding; which of them the walk offers is the walk's rule."""
    assert _binding_lines(_func(), "rows") == {3: "=", 6: ".append", 7: "="}


def test_a_method_that_takes_a_value_out_binds_nothing():
    """``discard``/``remove`` answer no question about where a value came from."""
    func = ast.parse("def run(self):\n    seen = set()\n    seen.discard(1)\n").body[0]

    assert _binding_lines(func, "seen") == {2: "="}
