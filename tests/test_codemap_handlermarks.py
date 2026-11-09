"""What each except handler prints when it is entered -- ``codemap.panda.handlermarks``, the marks a reader of a
run knows the handlers it entered by -- and the handlers a run entered by the marks it shows."""

from __future__ import annotations

import ast

import pytest

from bamboo.codemap.calls import Calls
from bamboo.codemap.gitsource import blob_sha
from bamboo.codemap.models import (
    HANDLER_CONDITIONAL,
    HANDLER_MARKED,
    HANDLER_SILENT,
    HANDLER_UNPLACED,
    SourceModule,
)
from bamboo.codemap.panda import handlermarks, pathcond
from bamboo.codemap.panda.recognizers import sqlwrite

PACKAGE = "pkg"
FILE = f"{PACKAGE}/m.py"

SOURCE = """\
import logging

log = logging.getLogger(__name__)


class Failure(Exception):
    pass


class Base:
    flag: bool

    def dump_error_message(self, tmp_log):
        tmp_log.error("failed")  # P1

    def cleanup(self):
        if self.flag:
            log.debug("cleaned")  # P2


class Proxy(Base):
    def own(self, x):
        try:
            x.go()
        except Exception:  # H1
            if x:
                log.error("own")  # P3
                return None
            log.error("other")  # P4

    def callee(self, x):
        try:
            x.go()
        except Exception:  # H2
            self.dump_error_message(log)  # C1
            return None

    def nowait(self, x):
        try:
            try:
                x.lock()
            except Exception as e:  # H3
                if x.busy(e):
                    log.debug("skip")  # P5
                else:
                    raise
            x.go()
        except Exception:  # H4
            self.dump_error_message(log)  # C2

    def quiet(self, x):
        try:
            x.go()
        except (ValueError, KeyError):  # H5
            pass
        try:
            x.go()
        except Exception:  # H6
            self.cleanup()
        try:
            x.go()
        except Exception:  # H7
            x.undo()
        try:
            x.go()
        except Exception as e:  # H8
            self.error = Failure(str(type(e)))
        try:
            x.go()
        except Exception:  # H9
            print("failed")

    def mixed(self, x):
        try:
            x.go()
        except Exception as e:  # H10
            if x.busy(e):
                log.debug("skip")  # P6
            else:
                self.dump_error_message(log)  # C3
            log.debug("time")  # P7
        for attempt in range(3):
            try:
                x.go()
                return True
            except Exception:  # H11
                if attempt + 1 < 3:
                    log.debug("retry")  # P8
                    continue
                self.dump_error_message(log)  # C4
                return False
"""


def _line(mark: str) -> int:
    return next(
        i for i, text in enumerate(SOURCE.splitlines(), 1) if text.endswith(f"# {mark}")
    )


#: Each function's print statements, as :class:`FunctionPrintsNode` holds them.
PRINTS = {
    f"{FILE}::Base.dump_error_message": {_line("P1")},
    f"{FILE}::Base.cleanup": {_line("P2")},
    f"{FILE}::Proxy.own": {_line("P3"), _line("P4")},
    f"{FILE}::Proxy.nowait": {_line("P5")},
    f"{FILE}::Proxy.mixed": {_line("P6"), _line("P7"), _line("P8")},
}


@pytest.fixture(scope="module")
def corpus(tmp_path_factory) -> tuple[SourceModule, Calls]:
    """The module as a corpus of one file, and the map's calls of it: a call is placed as the map places it."""
    root = tmp_path_factory.mktemp("handlermarks") / PACKAGE
    root.mkdir()
    (root / "__init__.py").write_text("")
    (root / "m.py").write_text(SOURCE)
    tree = ast.parse(SOURCE)
    pathcond.attach_parents(tree)
    module = SourceModule(
        package=PACKAGE,
        rel_path=FILE,
        tree=tree,
        source=SOURCE,
        blob_sha=blob_sha(SOURCE),
    )
    calls = sqlwrite.corpus_calls([module], {PACKAGE: root})
    assert calls is not None
    return module, calls


@pytest.fixture(scope="module")
def marks(corpus):
    module, calls = corpus
    return {
        one.line: one for one in handlermarks.handler_marks([module], calls, PRINTS)
    }


def test_every_handler_has_a_record(marks):
    assert sorted(marks) == sorted(_line(f"H{i}") for i in range(1, 12))
    assert {one.owner for one in marks.values()} == {
        f"{FILE}::Proxy.own",
        f"{FILE}::Proxy.callee",
        f"{FILE}::Proxy.nowait",
        f"{FILE}::Proxy.quiet",
        f"{FILE}::Proxy.mixed",
    }
    h5 = marks[_line("H5")]
    assert (h5.catches, h5.catch_all) == ("(ValueError, KeyError)", False)
    h1 = marks[_line("H1")]
    assert (h1.catches, h1.catch_all, h1.try_first, h1.try_last) == (
        "Exception",
        True,
        h1.line - 1,
        h1.line - 1,
    )


def test_a_handler_is_marked_where_every_way_passes_a_print_of_its_own(marks):
    """H1 prints on both ways out, one by a return and one off its end."""
    h1 = marks[_line("H1")]
    assert (h1.kind, h1.prints, h1.calls, h1.reraises) == (
        HANDLER_MARKED,
        [_line("P3"), _line("P4")],
        [],
        False,
    )


def test_a_call_whose_callee_prints_on_every_way_is_a_mark(marks):
    """``dump_error_message`` prints on every way, so calling it marks H2; ``cleanup`` prints under a test only,
    so H6 has no mark."""
    h2 = marks[_line("H2")]
    assert h2.kind == HANDLER_MARKED
    assert [
        (c.line, [(one.owner, one.prints) for one in c.callees]) for c in h2.calls
    ] == [(_line("C1"), [(f"{FILE}::Base.dump_error_message", [_line("P1")])])]
    h6 = marks[_line("H6")]
    assert (h6.kind, h6.prints, h6.calls) == (HANDLER_SILENT, [], [])


def test_a_handler_that_raises_again_without_a_mark_says_so(marks):
    """H3 prints on one way and raises again on the other: a mark seen says it was entered, none does not say it
    was not, and the raise goes on to H4 without a word."""
    h3 = marks[_line("H3")]
    assert (h3.kind, h3.prints, h3.reraises) == (
        HANDLER_CONDITIONAL,
        [_line("P5")],
        True,
    )
    h4 = marks[_line("H4")]
    assert (h4.kind, h4.reraises) == (HANDLER_MARKED, False)


def test_a_handler_without_a_mark_is_silent_or_holds_a_call_that_may_print(marks):
    """H5 does nothing; H8 calls builtins that write nothing and builds an exception whose ``__init__`` is a
    library's.  H7 calls a method of a parameter nothing declares, which might print; H9 calls ``print``."""
    assert marks[_line("H5")].kind == HANDLER_SILENT
    assert marks[_line("H8")].kind == HANDLER_SILENT
    assert marks[_line("H7")].kind == HANDLER_UNPLACED
    assert marks[_line("H9")].kind == HANDLER_UNPLACED


def test_a_handler_s_own_prints_alone_cover_every_way_or_not(marks):
    """A reader of one key's run can read only the owner's own prints absent.  H1 prints one of its own on
    every way out, and so does H10, after calling ``dump_error_message`` on one way (killJob's ``time=``
    line).  H11, a retry, prints its own only on the way that goes round again and leaves its last try by the
    call alone; H2 and H4 are marked by calls only; H3 is conditional."""
    h10, h11 = marks[_line("H10")], marks[_line("H11")]
    assert (h10.kind, h10.prints, [c.line for c in h10.calls]) == (
        HANDLER_MARKED,
        [_line("P6"), _line("P7")],
        [_line("C3")],
    )
    assert (h11.kind, h11.prints, [c.line for c in h11.calls]) == (
        HANDLER_MARKED,
        [_line("P8")],
        [_line("C4")],
    )
    covered = sorted(line for line, one in marks.items() if one.prints_on_every_way)
    assert covered == [_line("H1"), _line("H10")]


def test_a_handler_passing_a_raise_on_is_entered_with_the_handler_it_reaches(
    corpus, marks
):
    """The run shows H4's mark: H3 may have been entered too, its raise going on to H4.  The run shows H3's
    mark alone: H4 is not added.  No mark: none."""
    module, _ = corpus
    func = next(
        n
        for n in ast.walk(module.tree)
        if isinstance(n, ast.FunctionDef) and n.name == "nowait"
    )
    records = [one for one in marks.values() if one.owner.endswith("::Proxy.nowait")]

    def lines(seen):
        return [one.lineno for one in handlermarks.entered(func, records, seen)]

    h3, h4 = _line("H3"), _line("H4")
    assert lines({h4}) == [h3, h4]
    assert lines({h3}) == [h3]
    assert lines(set()) == []


class _NoCopy:
    """A value no copy may reach: the build's trees hold such (sqleval's interned texts)."""

    def __deepcopy__(self, memo):
        raise AssertionError("a copy followed a link out of the function")


def test_a_function_is_copied_without_the_links_its_shared_nodes_hold(corpus, marks):
    """The context objects (``Load`` and the others) are one of each, shared by every tree parsed, and a tree
    given parents leaves one of its nodes on them: a copy of a function follows none of their links."""
    module, calls = corpus
    load = next(n.ctx for n in ast.walk(ast.parse("x")) if isinstance(n, ast.Name))
    load.held = _NoCopy()
    try:
        again = handlermarks.handler_marks([module], calls, PRINTS)
    finally:
        del load.held
    assert again == sorted(marks.values(), key=lambda one: (one.owner, one.line))
