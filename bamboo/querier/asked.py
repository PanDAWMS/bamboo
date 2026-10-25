"""A walk's steps, asked one question at a time (a library for the querier's readers).

The map's :class:`trace.Asker` answers the arms of a function (``start()``) and
then one hop per question, handing back as its *frontier* every follow it did
not run.  :func:`walk` starts one and returns its steps as an :class:`Asked`:
the arms' steps, and every answer a reader asked for since.

A reader asks before it reads.  It picks its questions from the frontier by
what it reads -- a name's bindings in a function, what a call of a method
returns, a function's parameters -- asks every one that matches, and again
until no new one matches, since an answer's frontier can hold the same name
at another place or a hop further.  Then it reads the list as it read
:func:`trace.walk`'s: the readers' answers are those of the steps they would
have found there (out/tq-predictions.txt, 2026-10-24: K, A, Fm, Fo and G the
same), and the walk goes only where they read.

A list that was not made here cannot ask, and a reader handed one would read
what it holds and say less without a word; so the asking functions refuse it.
"""

from __future__ import annotations

import ast
import functools

from bamboo.codemap import trace  # noqa: E402
from bamboo.codemap.calls import Types  # noqa: E402
from bamboo.codemap.panda.provenance import classify  # noqa: E402
from bamboo.codemap.reading import containing_function  # noqa: E402
from bamboo.querier import rows as R  # noqa: E402

#: every walk this process asked, in order, for the run's last lines (:func:`summary`)
made: list[Asked] = []


def _key(step) -> tuple:
    """A step as the readers read it: two answers can hand back one step, at another depth or role."""
    return tuple(
        (k, tuple(v) if isinstance(v, list) else v)
        for k, v in step.model_dump(exclude={"depth", "role"}).items()
    )


class Asked(list):
    """The steps of one walk the readers have asked for so far."""

    def __init__(
        self,
        steps,
        asker: trace.Asker | None = None,
        frontier=(),
        note: str = "",
        where: str = "",
    ) -> None:
        super().__init__(steps)
        self.asker = asker
        #: ``function:line`` the walk starts from, as the run prints it
        self.where = where
        #: start()'s note: why there is no walk, when there is none
        self.note = note
        self.frontier: list[trace.Question] = list(frontier)
        self._known = set(self.frontier)
        self.asked: list[trace.Question] = []
        self._asked: set[trace.Question] = set()
        #: an asked answer's note -- a budget it ran out of -- with the question
        self.notes: list[str] = []
        self._have = {_key(s) for s in self} if asker is not None else set()

    def ask(self, pick) -> None:
        """Ask every frontier question *pick* takes, until no new one does."""
        while True:
            new = [q for q in self.frontier if q not in self._asked and pick(q)]
            if not new:
                return
            for q in new:
                self._asked.add(q)
                self.asked.append(q)
                answer = self.asker.ask(q)
                for s in answer.steps:
                    k = _key(s)
                    if k not in self._have:
                        self._have.add(k)
                        self.append(s)
                if answer.note:
                    self.notes.append(
                        f"{q.name} in {q.owner.split('::')[-1]}: {answer.note}"
                    )
                for f in answer.frontier:
                    if f not in self._known:
                        self._known.add(f)
                        self.frontier.append(f)


@functools.cache
def _types() -> Types:
    # one reading of the corpus for every question of the process: the code alone, as trace.walk reads it
    return Types(R.ROOTS, {})


def walk(fn: R.Function, line: int) -> Asked:
    """The arms at *line* of *fn*, asked as the readers read them."""
    asker = trace.Asker(
        R.ROOTS,
        file=fn.rel,
        owner=fn.owner,
        lines=[line],
        classify=classify,
        reaching=False,
        types=_types(),
    )
    start = asker.start()
    out = Asked(start.steps, asker, start.frontier, start.note, f"{fn.name}:{line}")
    made.append(out)
    return out


def given(steps) -> Asked:
    """Steps handed in whole -- a test's: nothing is left to ask, and the readers read them as they are."""
    return Asked(steps)


def _asked(steps) -> Asked:
    if not isinstance(steps, Asked):
        raise TypeError(
            f"the steps were not asked (asked.walk), so a reader would read only what they hold: {type(steps).__name__}"
        )
    return steps


def _called(text: str) -> str:
    """The name a returns question's call calls: ``altStgOutFileList`` for ``self.job.altStgOutFileList()``."""
    try:
        node = ast.parse(text, mode="eval").body
    except SyntaxError:
        return ""
    if isinstance(node, ast.Call):
        node = node.func
    return (
        node.attr
        if isinstance(node, ast.Attribute)
        else node.id
        if isinstance(node, ast.Name)
        else ""
    )


def bindings(steps, owner: str, name: str) -> None:
    """Ask the bindings of *name* in *owner*: where it is bound, and, a parameter, what its callers hand it."""
    _asked(steps).ask(
        lambda q: q.kind == trace.ASK_BINDINGS and q.owner == owner and q.name == name
    )


def returns(steps, method: str) -> None:
    """Ask what every call of *method* the walk came to returns."""
    _asked(steps).ask(
        lambda q: q.kind == trace.ASK_RETURNS and _called(q.name) == method
    )


def parameters(steps, owner: str, line: int) -> None:
    """Ask the bindings of the parameters of *owner*'s function at *line*: its callers' handovers to it."""
    rel = owner.split("::")[0]
    func = containing_function(R.tree(rel), line, owner)
    if func is None:
        return
    a = func.args
    names = {
        x.arg
        for x in [
            *a.posonlyargs,
            *a.args,
            *a.kwonlyargs,
            *(x for x in (a.vararg, a.kwarg) if x),
        ]
    }
    _asked(steps).ask(
        lambda q: q.kind == trace.ASK_BINDINGS and q.owner == owner and q.name in names
    )


def summary() -> list[str]:
    """One line a walk: what its readers asked, and an asked answer a budget cut."""
    out = []
    for w in made:
        line = (
            f"walk from {w.where}: asked {len(w.asked)} question(s), {len(w)} step(s)"
        )
        if w.notes:
            line += "; cut: " + "; ".join(w.notes)
        out.append(line)
    return out
