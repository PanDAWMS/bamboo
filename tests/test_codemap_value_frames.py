"""What a run-time value's text can be, read back through the locals that build it.

``values.value_frames`` is what an observed value rules an arm out by, so the
contract is one-sided: every text the value can take renders from one of the
frames, or there are none.  Each form below gives frames or gives nothing, and
the ones that must give nothing are the ways a reading could miss a text --
which is what turns an elimination into a confident wrong answer.
"""

import ast
import textwrap

import pytest

from bamboo.codemap.gitsource import blob_sha
from bamboo.codemap.models import (
    Branch,
    SourceModule,
    branch_excludes,
    pooled_cuts,
    pooled_frames,
)
from bamboo.codemap.panda import pathcond, values
from bamboo.codemap.panda.recognizers import progress


def _frames(body: str) -> list[str]:
    """``value_frames`` of the one ``job.diag = ...`` write in *body*."""
    return _reading(body).frames


def _reading(body: str) -> values.ValueReading:
    """``value_reading`` of the one ``job.diag = ...`` write in *body*."""
    source = "def f(job, p, g, L):\n" + textwrap.indent(textwrap.dedent(body), "    ") + "\n"
    tree = ast.parse(source)
    pathcond.attach_parents(tree)
    func = tree.body[0]
    (write,) = [
        node
        for node in ast.walk(func)
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Attribute)
        and node.targets[0].attr == "diag"
    ]
    return values.value_reading(write.value, func)  # type: ignore[arg-type]


BOUNDED = {
    "f-string": ("job.diag = f'merge_{p}'", ["merge_{}"]),
    "plus": ("job.diag = 'failed: ' + p", ["failed: {}"]),
    "format": ("job.diag = 'a {} b {0}'.format(p, p)", ["a {} b {}"]),
    "a local": ("msg = f'GUID for {p} not found'\njob.diag = msg", ["GUID for {} not found"]),
    "a local holding the head": ("err = 'Could not add: '\njob.diag = err + p", ["Could not add: {}"]),
    "two arms": (
        "if p:\n    msg = f'a {p}'\nelse:\n    msg = 'b'\njob.diag = msg",
        ["a {}", "b"],
    ),
    "the arm the write is not in": (
        "if p:\n    msg = 'a'\n    job.diag = msg\nelse:\n    msg = 'b'",
        ["a"],
    ),
    "appended to": ("msg = 'a'\nif p:\n    msg += ' tail'\njob.diag = msg", ["a", "{} tail"]),
    "a dict of messages": (
        "errs = {}\nif p:\n    errs[p] = ''\nelse:\n    errs[p] = f'failed {p}'\njob.diag = errs[p]",
        ["", "failed {}"],
    ),
    "a dict a value is popped from": (
        "errs = {}\nerrs[p] = 'a'\nerrs.pop(g, None)\njob.diag = errs[p]",
        ["a"],
    ),
    "a dict, two hops": (
        "errs = {}\ntmp = f'could not {p}'\nerrs[p] = tmp\njob.diag = errs[p]",
        ["could not {}"],
    ),
    # Without the loop the two arms of one ``if`` never meet; with it, an
    # earlier pass's ``a`` is what the ``else`` reads.
    "a value an earlier pass left": (
        "msg = 'start'\nfor x in L:\n    if x:\n        msg = 'a'\n    else:\n        job.diag = msg",
        ["a", "start"],
    ),
    # A join holds what its own block put into the list before it, wherever it sits among the others.
    "a join of what was appended": (
        "items = g()\nitems.append(f'scout={p}')\njob.diag = ' '.join(items)",
        ["{}scout={}"],
    ),
    "a join of what was inserted": ("items = []\nitems.insert(0, 'x')\njob.diag = ','.join(items)", ["{}x{}"]),
    # updateJobMetrics_JEDI: the tag alone, or the old items with the tag last, cut to 500.
    "a join cut to a length": (
        "tag = 'scout=' + '|'.join(L)\nif p is None:\n    msg = tag\nelse:\n    items = p.split(' ')\n"
        "    items = [i for i in items if not i.startswith('scout=')]\n    items.append(tag)\n"
        "    msg = ' '.join(items)\nmsg = msg[:500]\njob.diag = msg",
        ["scout={}", "{}scout={}"],
    ),
    # A part cut is not a head of the whole: ``msg[:2]`` is ``ab``, not ``abc{}``.
    "a part cut": ("msg = 'abc' + p\njob.diag = f'x {msg[:2]}'", ["x {}"]),
}


@pytest.mark.parametrize("form", sorted(BOUNDED))
def test_a_value_built_from_text_gives_every_text_it_can_be(form):
    body, expected = BOUNDED[form]
    assert _frames(body) == expected


UNBOUNDED = {
    "a parameter": "job.diag = p",
    "a call": "msg = g()\njob.diag = msg",
    "one arm a call": "if p:\n    msg = 'a'\nelse:\n    msg = g()\njob.diag = msg",
    "unpacked": "msg, other = g()\njob.diag = msg",
    "a loop target": "for msg in L:\n    job.diag = msg",
    "a with target": "with g() as msg:\n    job.diag = msg",
    "caught": "try:\n    pass\nexcept Exception as msg:\n    job.diag = msg",
    "global": "global msg\nmsg = 'a'\njob.diag = msg",
    "a comprehension's walrus": "msg = 'a'\nseen = [msg := g() for _x in L]\njob.diag = msg",
    "a dict handed on": "errs = {}\nerrs[p] = 'a'\ng(errs)\njob.diag = errs[p]",
    "a dict a closure fills": (
        "errs = {}\nerrs[p] = 'a'\ndef fill():\n    errs[p] = g()\njob.diag = errs[p]"
    ),
    "a dict updated": "errs = {}\nerrs[p] = 'a'\nerrs.update(g())\njob.diag = errs[p]",
    "a dict filled by unpacking": "errs = {}\nerrs[p], x = g()\njob.diag = errs[p]",
    "a dict filled as a loop target": "errs = {}\nfor errs[p] in L:\n    pass\njob.diag = errs[p]",
    "a dict handed in": "job.diag = L[p]",
    "a repr": "job.diag = f'{p!r}'",
    "a brace in the text": "job.diag = 'a {b}' + p",
    "holes only": "job.diag = f'{p}{p}'",
    "appended with nothing to read": "msg = g()\nmsg += f' {p}'\njob.diag = msg",
    "a join of an append under an if": "items = g()\nif p:\n    items.append('x')\njob.diag = ' '.join(items)",
    "a join after a pop": "items = g()\nitems.append('x')\nitems.pop()\njob.diag = ' '.join(items)",
    "a join of a list handed on": "items = g()\nitems.append('x')\ng(items)\njob.diag = ' '.join(items)",
    "a tail cut off": "msg = 'a' + p\njob.diag = msg[:-1]",
}


@pytest.mark.parametrize("form", sorted(UNBOUNDED))
def test_a_value_any_part_of_which_is_unbounded_gives_no_frames(form):
    assert _frames(UNBOUNDED[form]) == []


def test_a_branch_is_ruled_out_only_by_a_value_none_of_its_frames_fits():
    branch = Branch(outcome="runtime(msg)", tier=2, value_frames=["a {} b", "c"])

    assert not branch_excludes(branch, "a x b")
    assert not branch_excludes(branch, "a \nmulti\nline b")
    assert not branch_excludes(branch, "c")
    assert branch_excludes(branch, "a x")
    assert branch_excludes(branch, "cc")
    assert not branch_excludes(Branch(outcome="runtime(msg)", tier=2), "anything")
    assert not branch_excludes(Branch(outcome="x", tier=1, value_frames=["y"]), "z")


def test_chunks_are_found_in_order_and_the_tail_is_not_reused():
    branch = Branch(outcome="runtime(msg)", tier=2, value_frames=["ab{}b"])

    assert branch_excludes(branch, "ab")  # the head and the tail cannot share the ``b``
    assert not branch_excludes(branch, "abb")
    ordered = Branch(outcome="runtime(msg)", tier=2, value_frames=["{}x{}y{}"])
    assert branch_excludes(ordered, "yx")
    assert not branch_excludes(ordered, "xy")


def test_two_branches_kept_as_one_keep_both_frames_or_none():
    assert pooled_frames(["a{}"], ["b"]) == ["a{}", "b"]
    assert pooled_frames(["a{}"], []) == []
    assert pooled_cuts([500], []) == [500]


def test_a_cut_of_the_whole_value_is_said_and_a_value_of_its_length_is_read_as_a_head():
    """updateJobMetrics_JEDI cuts its value to 500 characters, which JobSpec does not: a 500-character value without
    the tag may have lost it, a shorter one has not."""
    body = BOUNDED["a join cut to a length"][0]
    assert _reading(body).cuts == [500]
    assert _reading(BOUNDED["a part cut"][0]).cuts == []
    assert _reading(UNBOUNDED["a tail cut off"]) == values.ValueReading([], [])
    branch = Branch(outcome="runtime(msg)", tier=2, value_frames=["scout={}", "{}scout={}"], value_cuts=[12])
    assert branch_excludes(branch, "nEvents=1 workDirSize=8080247")
    assert not branch_excludes(branch, "nEvents=1 ws")  # 12 characters: the head of a longer text
    assert not branch_excludes(branch, "nEvents=1 scout=a|b")
    assert branch_excludes(Branch(outcome="runtime(msg)", tier=2, value_frames=["{}scout={}"]), "nEvents=1 ws")


def _module(source: str, rel: str) -> SourceModule:
    return SourceModule(
        package=rel.split("/")[0],
        rel_path=rel,
        tree=ast.parse(source),
        source=source,
        blob_sha=blob_sha(source),
    )


def test_a_length_limit_counts_where_the_spec_cuts_by_it():
    cuts = (
        "class JobSpec(object):\n"
        "    _limitLength = {'ddmErrorDiag': 500, 'grid': 50}\n"
        "    def valuesMap(self):\n"
        "        val = self.ddmErrorDiag\n"
        "        return val[: self._limitLength['ddmErrorDiag']]\n"
    )
    states = "class OtherSpec(object):\n    _limitLength = {'diag': 9}\n"
    modules = [
        _module(cuts, "pandaserver/taskbuffer/JobSpec.py"),
        _module(states, "pandaserver/taskbuffer/OtherSpec.py"),
    ]

    assert progress.spec_length_limits(modules) == {
        ("JobSpec", "ddmErrorDiag"): 500,
        ("JobSpec", "grid"): 50,
    }


def test_the_attribute_slice_records_the_frames_of_a_run_time_write():
    specs = "class JediTaskSpec(object):\n    _attributes = ('jediTaskID', 'status')\n"
    writer = (
        "class W:\n"
        "    def f(self, taskSpec, s):\n"
        "        msg = f'merge_{s}'\n"
        "        taskSpec.status = msg\n"
    )
    modules = [
        _module(specs, "pandaserver/taskbuffer/JediTaskSpec.py"),
        _module(writer, "pandajedi/jediorder/W.py"),
    ]
    _subjects, junctions, _coverage, _diag, _enum = progress.extract(modules, "panda", "v")
    (junction,) = [j for j in junctions if j.owner.startswith("pandajedi/jediorder/W.py")]

    assert [(b.outcome, b.value_frames) for b in junction.branches] == [
        ("runtime(msg)", ["merge_{}"])
    ]
