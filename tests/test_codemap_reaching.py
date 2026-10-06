"""The writes that can be what a statement reads, followed the way the function runs.

Each shape below marks its writes ``# w:<tag>`` and its read ``# r:<name>
<tags>``, the tags being the writes some path brings to the read with nothing
overwriting them on the way.  ``pathcond.surviving_writes`` is asked with the
marked writes, as the walk asks with a name's bindings.
"""

from __future__ import annotations

import ast
import re

import pytest

from bamboo.codemap.panda import pathcond
from bamboo.codemap.panda.reaching import reaching

SHAPES = '''
def s1_if_else_cover(c):
    x = 1  # w:a
    if c:
        x = 2  # w:b
    else:
        x = 3  # w:c
    use(x)  # r:x b c


def s2_elif_else_return(c, d):
    u = 0  # w:a
    if c:
        u = 1  # w:b
    elif d:
        u = 2  # w:c
    else:
        return
    use(u)  # r:u b c


def s3_loop_top_rebind(items):
    for i in items:
        y = 0  # w:a
        use(y)  # r:y a
        y = 5  # w:b


def s4_test_name_rebound(c):
    x = 1  # w:a
    if c:
        x = 2  # w:b
    c = g()
    if c:
        use(x)  # r:x a b


def s6_try_handler_writes():
    z = 0  # w:a
    try:
        z = g()  # w:b
    except Exception:
        z = 9  # w:c
    use(z)  # r:z b c


def s7_nested_loops_wrap(items, more):
    w = 0  # w:a
    for i in items:
        for j in more:
            use(w)  # r:w a b c
            w = 1  # w:b
        w = 2  # w:c


def s8_nested_loops_head(items, more):
    v = 0  # w:a
    for i in items:
        v = 7  # w:b
        for j in more:
            use(v)  # r:v b c
            v = 8  # w:c


def s9_for_continue(items, d):
    y = 1  # w:a
    for i in items:
        if d:
            continue
        y = 2  # w:b
    use(y)  # r:y a b


def s10_while_break(d):
    t = 1  # w:a
    while d:
        t = 2  # w:b
        break
    use(t)  # r:t a b


def s11_with_body():
    s = 1  # w:a
    with ctx():
        s = 2  # w:b
    use(s)  # r:s b


def s12_finally_after():
    a = 1  # w:a
    try:
        a = f()  # w:b
    finally:
        cleanup()
    use(a)  # r:a b


def s12b_read_in_finally():
    a = 1  # w:a
    try:
        a = f()  # w:b
    finally:
        use(a)  # r:a a b


def s13_for_else(items, c):
    x = 1  # w:a
    for i in items:
        if c(i):
            x = 2  # w:b
            break
    else:
        x = 3  # w:c
    use(x)  # r:x b c


def s14_while_else(d, c):
    x = 1  # w:a
    while d:
        if c:
            x = 2  # w:b
            break
    else:
        x = 3  # w:c
    use(x)  # r:x b c


def s15_triple_loops(items, more, most):
    a = 0  # w:a
    for i in items:
        a = 1  # w:b
        for j in more:
            for k in most:
                use(a)  # r:a b c
                a = 2  # w:c
        a = 3  # w:d


def s16_except_as_after():
    e = 0  # w:a
    try:
        f()
    except Exception as e:  # w:b
        pass
    use(e)  # r:e a


def s17_augassign_chain(c):
    n = 1  # w:a
    if c:
        n = 2  # w:b
    n += 3  # w:c
    use(n)  # r:n a b c


def s18_walrus_in_test(c):
    m = None  # w:a
    if (m := f()) and c:  # w:b
        use(m)  # r:m b


def s19_with_as(p):
    fh = None  # w:a
    with open(p) as fh:  # w:b
        pass
    use(fh)  # r:fh b


def s20_try_in_loop_continue(items):
    r = 0  # w:a
    for i in items:
        try:
            r = g(i)  # w:b
        except Exception:
            continue
        use(r)  # r:r b


def s21_continue_then_write(items, c):
    x = 0  # w:a
    for i in items:
        if c:
            continue
        x = i  # w:b
    use(x)  # r:x a b


def s23_two_writes_read_in_handler():
    z = 0  # w:a
    try:
        z = g()  # w:b
        z = h()  # w:c
    except Exception:
        use(z)  # r:z a b


def s24_two_writes_read_after():
    z = 0  # w:a
    try:
        z = g()  # w:b
        z = h()  # w:c
    except Exception:
        pass
    use(z)  # r:z a b c


def s25_handler_returns():
    x = 1  # w:a
    try:
        x = f()  # w:b
    except Exception:
        return
    use(x)  # r:x b


def s26_else_raise(c):
    x = 1  # w:a
    if c:
        x = 2  # w:b
    else:
        raise ValueError()
    use(x)  # r:x b


def s27_while_true(c):
    x = 1  # w:a
    while True:
        x = f()  # w:b
        if x:
            break
    use(x)  # r:x b


def s28_match_no_wildcard(v):
    x = 0  # w:a
    match v:
        case 1:
            x = 1  # w:b
        case 2:
            x = 2  # w:c
    use(x)  # r:x a b c


def s29_match_wildcard(v):
    x = 0  # w:a
    match v:
        case 1:
            x = 1  # w:b
        case 2:
            x = 2  # w:c
        case _:
            x = 3  # w:d
    use(x)  # r:x b c d


def s30_safe_statement_in_try():
    x = 0  # w:a
    try:
        x = 1  # w:b
        y = f()
    except Exception:
        use(x)  # r:x b


def s31_nested_try_specific_inner():
    x = 0  # w:a
    try:
        try:
            x = f()  # w:b
        except KeyError:
            pass
    except Exception:
        use(x)  # r:x a


def s32_return_through_finally():
    a = 0  # w:a
    try:
        a = f()  # w:b
        return a
    finally:
        use(a)  # r:a a b


def s33_while_walrus(read):
    line = None  # w:a
    while (line := read()):  # w:b
        use(line)  # r:line b


def s34_loop_target_after(items):
    i = -1  # w:a
    for i in items:  # w:b
        pass
    use(i)  # r:i a b


def s35_break_through_finally(items):
    x = 0  # w:a
    for i in items:
        try:
            x = 1  # w:b
            break
        finally:
            x = 2  # w:c
    use(x)  # r:x a c


def s36_raise_through_finally():
    x = 0  # w:a
    try:
        try:
            f()
        finally:
            x = 1  # w:b
    except Exception:
        use(x)  # r:x b


def s37_tuple_unpacking_can_raise(results):
    a = None  # w:a
    try:
        a, b = results  # w:b
    except ValueError:
        pass
    use(a)  # r:a a b


def s38_assignment_reading_itself_in_a_loop(items):
    total = 0  # w:a
    for i in items:
        total = total + i  # r:total a b  w:b
    use(total)  # r:total a b
'''


def _parsed() -> ast.Module:
    tree = ast.parse(SHAPES)
    pathcond.attach_parents(tree)
    return tree


def _starting_at(func: ast.AST, line: int) -> ast.AST:
    """The outermost statement or handler that starts on *line*."""
    for node in ast.walk(func):
        if isinstance(node, (ast.stmt, ast.ExceptHandler)) and node.lineno == line:
            return node
    raise AssertionError(f"nothing starts on line {line}")


def _comment(line: str) -> str:
    return line.split("#", 1)[1] if "#" in line else ""


def _cases() -> list[tuple[str, int]]:
    lines = SHAPES.splitlines()
    return [
        (func.name, line)
        for func in _parsed().body
        for line in range(func.lineno, func.end_lineno + 1)
        if re.search(r"\br:\w+", _comment(lines[line - 1]))
    ]


def _ask(func_name: str, line: int) -> tuple[list[str], list[str]]:
    """(the tags :func:`pathcond.surviving_writes` keeps, the tags the shape says reach)."""
    lines = SHAPES.splitlines()
    func = next(node for node in _parsed().body if node.name == func_name)
    tags: dict[int, str] = {}
    sites = []
    for at in range(func.lineno, func.end_lineno + 1):
        marked = re.search(r"\bw:(\w+)", _comment(lines[at - 1]))
        if marked:
            site = _starting_at(func, at)
            tags[id(site)] = marked.group(1)
            sites.append(site)
    read = re.search(r"\br:\w+((?: \w+)+)", _comment(lines[line - 1]))
    assert read is not None
    kept = pathcond.surviving_writes(sites, _starting_at(func, line), as_run=True)
    return sorted(tags[id(site)] for site in kept), sorted(read.group(1).split())


@pytest.mark.parametrize(("func_name", "line"), _cases(), ids=[name for name, _ in _cases()])
def test_the_writes_that_reach_a_read_are_the_ones_some_path_brings_to_it(func_name, line):
    kept, truth = _ask(func_name, line)
    assert kept == truth


def test_the_same_test_asked_twice_is_not_remembered():
    """Given up on purpose: two ``if c:`` on an unchanged ``c`` take the same arm.

    The rule this replaced knew that and dropped ``x = 1`` -- and with the same
    comparison it also dropped ``x = 1`` where ``c`` was rebound between the
    two tests (``s4`` above), a wrong answer.  The flow takes every branch both
    ways, so ``x = 1`` is offered here through a path that cannot run: an
    excess a later step can rule out, where a drop is one nothing brings back.
    """
    source = (
        "def f(c):\n"
        "    x = 1\n"
        "    if c:\n"
        "        x = 2\n"
        "    if c:\n"
        "        use(x)\n"
    )
    tree = ast.parse(source)
    pathcond.attach_parents(tree)
    func = tree.body[0]
    one, two = func.body[0], func.body[1].body[0]
    target = func.body[2].body[0]

    assert pathcond.surviving_writes([one, two], target, as_run=True) == [one, two]


def test_a_killer_overwrites_and_is_never_an_answer():
    """The walk hands the writes outside its stretch over as killers: they still ran.

    ``del x`` is not a write anyone hands in, and handed in as a killer it says
    what Python does -- after it, nothing from above is what ``x`` holds.
    """
    source = (
        "def f(c):\n"
        "    x = 1\n"
        "    if c:\n"
        "        x = 2\n"
        "    else:\n"
        "        del x\n"
        "    use(x)\n"
    )
    tree = ast.parse(source)
    pathcond.attach_parents(tree)
    func = tree.body[0]
    one, two, gone = func.body[0], func.body[1].body[0], func.body[1].orelse[0]
    target = func.body[2]

    assert pathcond.surviving_writes([one, two], target, as_run=True) == [one, two]
    assert pathcond.surviving_writes([one, two], target, as_run=True, killers=[gone]) == [two]


def test_a_reread_adds_to_what_was_there():
    """``seen.add(b)`` keeps ``a`` in the set, as ``x += y`` keeps what ``x`` held."""
    source = (
        "def f(a, b):\n"
        "    seen = set()\n"
        "    seen.add(a)\n"
        "    seen.add(b)\n"
        "    seen = set()\n"
        "    seen.add(b)\n"
        "    use(seen)\n"
    )
    tree = ast.parse(source)
    pathcond.attach_parents(tree)
    func = tree.body[0]
    made, first, second, again, last, target = func.body

    kept = pathcond.surviving_writes(
        [made, first, second, again, last], target, [first, second, last], as_run=True
    )

    assert kept == [again, last]


def test_a_write_in_a_nested_function_is_another_variable():
    """``add_main.main``'s ``taskBuffer = tbuf`` is not replaced by a nested class's ``run()``.

    ``pathcond.assigned_expressions`` counts a nested function's writes as the
    build wants them, and a reader asking what *this* function's read sees
    must not take one for an overwrite.
    """
    source = (
        "def main(tbuf):\n"
        "    taskBuffer = tbuf\n"
        "    class Thread:\n"
        "        def run(self):\n"
        "            taskBuffer = self.taskBuffer\n"
        "    use(taskBuffer)\n"
    )
    tree = ast.parse(source)
    pathcond.attach_parents(tree)
    func = tree.body[0]
    outer = func.body[0]
    inner = func.body[1].body[0].body[0]
    target = func.body[2]

    assert pathcond.surviving_writes([outer, inner], target, as_run=True) == [outer]


def test_a_name_a_nested_function_only_reads_is_the_enclosing_one():
    """A closure reads the enclosing function's name whenever it runs, so every write to it is one."""
    source = (
        "def main(rows):\n"
        "    limit = 1\n"
        "    def check(row):\n"
        "        return row < limit\n"
        "    limit = 2\n"
        "    return [r for r in rows if check(r)]\n"
    )
    tree = ast.parse(source)
    pathcond.attach_parents(tree)
    func = tree.body[0]
    first, second = func.body[0], func.body[2]
    target = func.body[1].body[0]

    assert reaching(target, [first, second]) == [first, second]


def test_the_same_writes_asked_from_two_reads_answer_each():
    source = (
        "def f():\n"
        "    t = first()\n"
        "    y = t + 1\n"
        "    t = second()\n"
        "    use(y, t)\n"
    )
    tree = ast.parse(source)
    pathcond.attach_parents(tree)
    func = tree.body[0]
    one, read, two, last = func.body

    assert pathcond.surviving_writes([one, two], read, as_run=True) == [one]
    assert pathcond.surviving_writes([one, two], last, as_run=True) == [two]


def test_a_parameter_list_is_written_on_entry():
    """``if days is None: days = 3`` leaves the call's value beside the default the body writes.

    The parameter list is handed in as a write that holds when the function
    starts; with no other write on some path, the call's value is what the
    read sees there, and an unconditional rebinding ends it.
    """
    source = (
        "def f(days, limit):\n"
        "    if days is None:\n"
        "        days = 3\n"
        "    use(days)\n"
        "    limit = 5\n"
        "    use(limit)\n"
    )
    tree = ast.parse(source)
    pathcond.attach_parents(tree)
    func = tree.body[0]
    default, reread = func.body[0].body[0], func.body[1]
    rebound, read = func.body[2], func.body[3]

    assert reaching(reread, [func.args, default]) == [func.args, default]
    assert reaching(read, [func.args, rebound]) == [rebound]
