"""A value's way from its write to where it is read (``pathcond.replacements``).

A branch's condition is the condition its value reaches its write under: the
guards on the write, and an entry for every other write of the place that can
replace the value before it is read.  A write whose value no read reaches is
no branch.  Three routes, one rule: a local to the statement putting it into a
write, a statement map's key to the call running the statement, an object's
field to where the object leaves the function.
"""

from __future__ import annotations

import ast

from bamboo.codemap.panda import pathcond, sql


def _func(source: str, name: str = "f"):
    tree = ast.parse(source)
    pathcond.attach_parents(tree)
    return next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == name
    )


def _at(func, line: int) -> ast.stmt:
    """The statement of *func* starting on *line* (1-based in the source)."""
    return next(
        node
        for node in ast.walk(func)
        if isinstance(node, ast.stmt) and node.lineno == line
    )


def _field(func, line: int, callee=None):
    """``field_replacements`` for the attribute write on *line*."""
    write = _at(func, line)
    target = write.targets[0]
    return pathcond.field_replacements(
        write, target.value, target.attr, write.value, callee
    )


def _kinds(replaced) -> list[tuple[int, str, str | None]]:
    return [(one.site.lineno, one.why, one.entry) for one in replaced]


# --------------------------------------------------------------------------- #
# a local
# --------------------------------------------------------------------------- #


def test_a_local_value_no_read_reaches_is_no_value():
    """``retryTask_JEDI`` sets ``newStatus = None`` where no bind reads it."""
    func = _func("def f(x):\n    status = None\n    status = 'a'\n    return status\n")

    values = pathcond.literal_values(
        func, "status", lambda e, _f: [ast.unparse(e)], at=_at(func, 4)
    )

    assert [(value, line) for value, _conditions, line in values] == [("'a'", 3)]


def test_a_computed_replacement_counts_and_a_loop_says_some_pass_ran_it():
    """``reset_resource_type_task``: a default, then a computed name in a loop
    that may find nothing -- the default lands where no pass matched."""
    func = _func(
        "def f(resource_map, c):\n"
        "    resource_name = 'Undefined'\n"
        "    for resource_spec in resource_map:\n"
        "        if resource_spec.match(c):\n"
        "            resource_name = resource_spec.resource_name\n"
        "            break\n"
        "    return resource_name\n"
    )

    values = pathcond.literal_values(func, "resource_name", at=_at(func, 7))

    assert values == [
        (
            "Undefined",
            ["not any(resource_spec.match(c) for resource_spec in resource_map)"],
            2,
        )
    ]


def test_a_replacement_two_tests_deeper_is_negated_as_both():
    """``ppEventServiceJob`` rewrites under ``if doMerging:`` and an inner test:
    "not (inner)" alone is false where the outer test kept the rewrite from running."""
    func = _func(
        "def f(m, doMerging):\n"
        "    p = 'a'\n"
        "    if doMerging:\n"
        "        if m is not None:\n"
        "            p = 'b'\n"
        "    return p\n"
    )

    values = {
        value: conditions
        for value, conditions, _line in pathcond.literal_values(
            func, "p", at=_at(func, 6)
        )
    }

    assert values["a"] == ["not (doMerging and m is not None)"]
    assert values["b"] == ["doMerging", "m is not None"]


def test_without_a_read_every_settled_assignment_comes_under_its_own_guards():
    func = _func("def f(c):\n    s = 'a'\n    if c:\n        s = 'b'\n    return s\n")

    assert pathcond.literal_values(func, "s") == [("a", [], 2), ("b", ["c"], 4)]


# --------------------------------------------------------------------------- #
# a field
# --------------------------------------------------------------------------- #


def test_a_field_rewritten_before_the_object_leaves_is_no_branch():
    """``remAttribute`` and ``extractCommon``: rewritten the next line."""
    func = _func("def f(self):\n    self.attributes = 'x'\n    self.attributes = 'y'\n")

    assert _field(func, 2) is None
    assert _kinds(_field(func, 3)) == [(2, pathcond.NOT_AFTER, None)]


def test_a_replacement_whose_test_the_value_settles_against_it_never_runs():
    """``killJob``: ``if job.jobStatus != "closed"`` after ``job.jobStatus = "closed"``."""
    func = _func(
        "def f(job, code):\n"
        "    if code == '2':\n"
        "        job.jobStatus = 'closed'\n"
        "    if job.jobStatus != 'closed':\n"
        "        job.jobStatus = 'cancelled'\n"
        "    send(job)\n"
    )

    assert _kinds(_field(func, 3)) == [(5, pathcond.NEVER_AFTER, None)]


def test_a_test_the_value_makes_true_drops_out():
    func = _func(
        "def f(ret, c):\n"
        "    ret.maxwdir = 16336\n"
        "    if c:\n"
        "        if ret.maxwdir:\n"
        "            ret.maxwdir = 1\n"
        "    return ret\n"
    )

    assert _kinds(_field(func, 2)) == [(5, pathcond.REPLACES, "not (c)")]


def test_a_value_read_before_the_replacement_gets_no_entry():
    func = _func(
        "def f(job, c):\n    job.s = 'a'\n    send(job)\n    if c:\n        job.s = 'b'\n"
    )

    assert _kinds(_field(func, 2)) == [(5, pathcond.LEFT_BEFORE, None)]


def test_a_field_read_on_the_object_is_not_the_object_leaving():
    func = _func(
        "def f(job, c):\n    job.s = 'a'\n    log(job.PandaID)\n    if c:\n        job.s = 'b'\n    return job\n"
    )

    assert _kinds(_field(func, 2)) == [(5, pathcond.REPLACES, "not (c)")]


def test_no_entry_where_something_can_keep_the_replacement_from_running():
    """A handler, a loop's ``else``, a ``while``, a jump on the way: the
    replacement's tests are not enough for it to have run."""
    handler = _func(
        "def f(job, c):\n"
        "    job.s = 'a'\n"
        "    try:\n"
        "        x()\n"
        "    except Exception:\n"
        "        if c:\n"
        "            job.s = 'b'\n"
        "    return job\n"
    )
    jump = _func(
        "def f(job, c, d):\n"
        "    job.s = 'a'\n"
        "    if d:\n"
        "        return None\n"
        "    if c:\n"
        "        job.s = 'b'\n"
        "    save(job)\n"
    )
    loop = _func(
        "def f(job, items):\n"
        "    job.s = 'a'\n"
        "    while items:\n"
        "        if items.pop():\n"
        "            job.s = 'b'\n"
        "    return job\n"
    )

    assert _kinds(_field(handler, 2)) == [(7, pathcond.NOT_ON_ITS_TESTS, None)]
    assert _kinds(_field(jump, 2)) == [(6, pathcond.SKIPPED, None)]
    assert _kinds(_field(loop, 2)) == [(5, pathcond.NOT_ON_ITS_TESTS, None)]


def test_a_holder_renewed_on_each_pass_is_another_object():
    """``parse_job_output_report``: ``file.status = ...`` on one pass is not
    replaced by the next pass's, and the old object is still in the list."""
    func = _func(
        "def f(files):\n"
        "    for file in files:\n"
        "        if file.bad:\n"
        "            file.status = 'failed'\n"
        "            continue\n"
        "        file.status = 'ok'\n"
    )

    assert _kinds(_field(func, 4)) == [(6, pathcond.NOT_REACHED, None)]
    assert _kinds(_field(func, 6)) == [(4, pathcond.NOT_AFTER, None)]


def test_a_holder_given_another_object_keeps_the_value_in_the_old_one():
    func = _func(
        "def f(job, other):\n    job.s = 'a'\n    job = other\n    job.s = 'b'\n"
    )

    assert _kinds(_field(func, 2)) == [(4, pathcond.NOT_REACHED, None)]


def _on_hold(call: ast.Call):
    if call.func.attr == "setOnHold":
        return pathcond.Callee("setOnHold", (("self.status in ['ready', 'running']",),))
    return None


def test_a_setter_call_replaces_under_its_own_tests_marked_as_the_methods():
    """``TaskRefiner.runImpl``: ``impl.taskSpec.status = taskStatus``, then
    ``impl.taskSpec.setOnHold()``, which writes the status only under its own
    test on ``self.status`` -- a test the caller's names cannot read."""
    func = _func(
        "def f(impl, taskStatus):\n"
        "    impl.taskSpec.status = taskStatus\n"
        "    impl.taskSpec.setOnHold()\n"
        "    save(impl)\n"
    )

    assert _kinds(_field(func, 2, _on_hold)) == [
        (
            3,
            pathcond.REPLACES,
            "not (self.status in ['ready', 'running'])  [in setOnHold()]",
        )
    ]


def test_a_setter_test_the_value_settles_is_read_with_it():
    ready = _func(
        "def f(impl, c):\n    impl.taskSpec.status = 'ready'\n    if c:\n        impl.taskSpec.setOnHold()\n    save(impl)\n"
    )
    done = _func(
        "def f(impl, c):\n    impl.taskSpec.status = 'done'\n    if c:\n        impl.taskSpec.setOnHold()\n    save(impl)\n"
    )

    assert _kinds(_field(ready, 2, _on_hold)) == [(4, pathcond.REPLACES, "not (c)")]
    assert _kinds(_field(done, 2, _on_hold)) == [(4, pathcond.NEVER_AFTER, None)]


def test_a_replacement_in_a_loop_says_some_pass_ran_it():
    """``peekJob``: ``job.jobParameters = None``, then the clob of whichever row the cursor has."""
    func = _func(
        "def f(job, cur):\n"
        "    job.p = None\n"
        "    for (clob,) in cur:\n"
        "        if clob is not None:\n"
        "            job.p = clob.read()\n"
        "        break\n"
        "    return job\n"
    )

    assert _kinds(_field(func, 2)) == [
        (5, pathcond.REPLACES, "not any(clob is not None for (clob,) in cur)")
    ]


# --------------------------------------------------------------------------- #
# a statement map's key
# --------------------------------------------------------------------------- #


def test_a_bind_filled_again_before_the_run_carries_the_later_ones_test():
    """``updateTaskStatusByContFeeder_JEDI`` fills ``:errorDialog`` and fills it
    again under a test before the run."""
    func = _func(
        "def f(self, task, c):\n"
        "    sql = 'UPDATE ATLAS_PANDA.JEDI_Tasks SET errorDialog=:errorDialog'\n"
        "    varMap = {}\n"
        "    varMap[':errorDialog'] = None\n"
        "    if c:\n"
        "        varMap[':errorDialog'] = 'x'\n"
        "    self.cur.execute(sql, varMap)\n"
    )
    run = sql.Execution(
        sql="", varmap="varMap", variable="sql", call=_at(func, 7).value
    )
    binds = sql.bound_values(func, "varMap", ":errorDialog", at=run)

    replaced = sql.bind_replacements(func, run, ":errorDialog", binds)

    assert _kinds(replaced[id(_at(func, 4))]) == [(6, pathcond.REPLACES, "not (c)")]
    assert _kinds(replaced[id(_at(func, 6))]) == [(4, pathcond.NOT_AFTER, None)]


# --------------------------------------------------------------------------- #
# a test reading what changed after the write
# --------------------------------------------------------------------------- #


def test_a_test_reading_the_place_after_the_write_is_marked_with_its_line():
    """``remAttribute``: ``if self.attributes == "":`` reads what the rewrite made
    of it, not what a reader saw before -- the entry is read at the test."""
    func = _func(
        "def f(self, name):\n"
        "    if self.attributes is not None:\n"
        "        self.attributes = re.sub(name, '', self.attributes)\n"
        "        if self.attributes == '':\n"
        "            self.attributes = None\n"
    )

    assert _kinds(_field(func, 3)) == [
        (5, pathcond.REPLACES, "not (self.attributes == '')  [at 4]")
    ]


def test_a_test_reading_a_name_bound_after_the_write_is_marked_with_its_line():
    """``ppEventServiceJob``: ``tmpMatch = re.search(...)`` between the write and the rewrite's test."""
    func = _func(
        "def f(jobSpec, doMerging):\n"
        "    jobSpec.jobParameters = 'x'\n"
        "    if doMerging:\n"
        "        tmpMatch = re.search('<M>(.*)</M>', jobSpec.jobParameters)\n"
        "        if tmpMatch is not None:\n"
        "            jobSpec.jobParameters = tmpMatch.group(1)\n"
        "    return jobSpec\n"
    )

    assert _kinds(_field(func, 2)) == [
        (6, pathcond.REPLACES, "not (doMerging and tmpMatch is not None)  [at 5]")
    ]


def test_a_value_that_settles_its_replacement_to_run_is_no_value():
    """``register_token_key``: ``max_id = None`` meets ``if max_id is None:
    max_id = 0`` wherever it gets to it, so None is never what the bind reads."""
    func = _func(
        "def f(res):\n"
        "    max_id = None\n"
        "    if res is not None:\n"
        "        (max_id,) = res\n"
        "    if max_id is None:\n"
        "        max_id = 0\n"
        "    return max_id\n"
    )

    values = pathcond.literal_values(
        func, "max_id", lambda e, _f: [ast.unparse(e)], at=_at(func, 7)
    )

    assert values == [("0", ["max_id is None"], 6)]


def test_a_return_that_no_read_follows_skips_nothing_the_value_needs():
    """``storeJobs`` returns early for a banned user, between ``userCountry =
    None`` and the loop that may replace it: where the value lands, it did not return."""
    func = _func(
        "def f(banned, fqans):\n"
        "    country = None\n"
        "    if banned:\n"
        "        return []\n"
        "    for fqan in fqans:\n"
        "        if fqan.startswith('/atlas/us'):\n"
        "            country = 'us'\n"
        "            break\n"
        "    return country\n"
    )

    values = {
        value: conditions
        for value, conditions, _line in pathcond.literal_values(
            func, "country", at=_at(func, 9)
        )
    }

    assert pathcond.literal_values(
        func, "country", lambda e, _f: [ast.unparse(e)], at=_at(func, 9)
    )[0] == (
        "None",
        ["not any(fqan.startswith('/atlas/us') for fqan in fqans)"],
        2,
    )
    assert values["us"] == ["fqan.startswith('/atlas/us')"]


# --------------------------------------------------------------------------- #
# a reread
# --------------------------------------------------------------------------- #


def test_an_augmented_binding_replaces_a_locals_literal():
    """``register_token_key`` adds 1 to ``max_id`` before the bind; and
    ``updateUnmergedDatasets`` adds to ``tmpNumFiles = 0`` on each pass."""
    surely = _func(
        "def f(res):\n"
        "    max_id = 0\n"
        "    if res:\n"
        "        (max_id,) = res\n"
        "    max_id += 1\n"
        "    return max_id\n"
    )
    looped = _func(
        "def f(files):\n"
        "    n = 0\n"
        "    for one in files:\n"
        "        n += one.n\n"
        "    return n\n"
    )

    assert pathcond.literal_values(surely, "max_id", at=_at(surely, 6)) == []
    assert pathcond.literal_values(
        looped, "n", lambda e, _f: [ast.unparse(e)], at=_at(looped, 5)
    ) == [("0", ["not any(True for one in files)"], 2)]


def test_a_test_after_a_reread_is_not_read_with_the_value():
    """``job.n += c`` between ``job.n = 0`` and ``if job.n == 0``: the test reads
    what the reread made of it, which the literal does not settle."""
    func = _func(
        "def f(job, c):\n"
        "    job.n = 0\n"
        "    job.n += c\n"
        "    if job.n == 0:\n"
        "        job.n = 5\n"
        "    save(job)\n"
    )

    assert _kinds(_field(func, 2)) == [
        (5, pathcond.REPLACES, "not (job.n == 0)  [at 4]")
    ]


def test_the_writes_own_next_pass_is_not_a_reread_between():
    """``doBasicRefine``: one ``setDatasetAttribute`` call after another in a
    loop; the first one running again on the next pass is the value again,
    not something made of it, so the second's test is still read with it."""
    func = _func(
        "def f(spec, items):\n"
        "    for item in items:\n"
        "        if 'a' in item:\n"
        "            spec.setX('a')\n"
        "        if 'b' in item:\n"
        "            spec.setX('b')\n"
        "    save(spec)\n"
    )
    write = _at(func, 4)

    def set_x(call: ast.Call):
        if call.func.attr == "setX":
            return pathcond.Callee("setX", (("self.x is None",),))
        return None

    replaced = pathcond.field_replacements(
        write, write.value.func.value, "x", ast.Constant(value=""), set_x
    )

    assert _kinds(replaced) == [(6, pathcond.NEVER_AFTER, None)]
