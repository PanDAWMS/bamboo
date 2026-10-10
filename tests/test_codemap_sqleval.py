"""Reading the SQL a function hands its cursor by running it (``sqleval``).

The shapes are the ones the corpus census found the map's reassembly misread
(H1-H6) and the control flow a statement is built through; each pins what
running the Python on every path says, line numbers counted from the source
string's first (empty) line.
"""

from __future__ import annotations

import ast
import re
import textwrap

import pytest

from bamboo.codemap.panda import pathcond, sql, sqleval


@pytest.fixture(autouse=True)
def _fresh():
    # Interned values and callee readings are kept between functions; a test
    # must not read another's.
    sqleval.clear_memo()
    yield
    sqleval.clear_memo()


def _function(src: str, name: str | None = None) -> ast.FunctionDef:
    tree = ast.parse(textwrap.dedent(src))
    return next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and (name is None or n.name == name)
    )


def _reading(src, constants=None, name=None, resolver=None) -> sqleval.Reading:
    reading = sqleval.evaluate(_function(src, name), constants, resolver)
    assert reading.overflow is None, reading.overflow
    return reading


def _texts(src, constants=None, name=None):
    reading = _reading(src, constants, name)
    return {line: sorted(t) for line, t in reading.texts.items()}, reading


def _resolver(defs: dict):
    def resolve(func, call):
        if isinstance(call.func, ast.Name) and call.func.id in defs:
            return [defs[call.func.id]]
        return None

    return resolve


def _lines(spans):
    return [(where.lineno, start, end) for where, start, end in spans]


# ---------------------------------------------------------------- the census shapes

H1 = """
def peek(self, fromDefined, fromActive, forAnal=False):
    sql1_0 = "SELECT %s FROM %s "
    sql1_1 = "WHERE PandaID=:PandaID"
    tables = []
    if fromDefined:
        tables.append("ATLAS_PANDA.jobsDefined4")
    if fromActive:
        tables.append("ATLAS_PANDA.jobsActive4")
    for table in tables:
        sql = sql1_0 % (JobSpec.columnNames(), table) + sql1_1
        self.cur.execute(sql + comment, varMap)
        if table == "ATLAS_PANDA.jobsArchived4" or forAnal:
            self.cur.execute("SELECT metaData FROM ATLAS_PANDA.metaTable", varMap)
"""


def test_a_table_list_built_by_append_is_its_tables():
    """``peekJob``: a list appended under two tests, a loop over it, ``%`` into a template a name holds."""
    got, reading = _texts(H1)
    assert got.get(12) == [
        "SELECT {} FROM ATLAS_PANDA.jobsActive4 WHERE PandaID=:PandaID{}",
        "SELECT {} FROM ATLAS_PANDA.jobsDefined4 WHERE PandaID=:PandaID{}",
    ]
    assert 14 in reading.reached


def test_a_routes_constants_decide_the_list_and_the_loop_variable():
    """Called with ``fromActive`` only, one table; the loop variable then rules out the metadata read."""
    got, reading = _texts(
        H1, {"fromDefined": False, "fromActive": True, "forAnal": False}
    )
    assert got.get(12) == [
        "SELECT {} FROM ATLAS_PANDA.jobsActive4 WHERE PandaID=:PandaID{}"
    ]
    assert 14 not in reading.reached


def test_a_loop_over_tuple_pairs_formats_each():
    src = """
def meta(self, ids):
    for a, m in [("ATLAS_PANDA.jobsArchived4", "ATLAS_PANDA.metaTable"), ("ATLAS_PANDAARCH.jobsArchived", "ATLAS_PANDAARCH.metaTable_ARCH")]:
        sql = "SELECT j.PandaID FROM {0} j,{1} m WHERE m.PandaID=j.PandaID".format(a, m)
        self.cur.execute(sql, varMap)
"""
    got, _ = _texts(src)
    assert got.get(5) == [
        "SELECT j.PandaID FROM ATLAS_PANDA.jobsArchived4 j,ATLAS_PANDA.metaTable m WHERE m.PandaID=j.PandaID",
        "SELECT j.PandaID FROM ATLAS_PANDAARCH.jobsArchived j,ATLAS_PANDAARCH.metaTable_ARCH m WHERE m.PandaID=j.PandaID",
    ]


H3 = """
def inc(self, x):
    sqlAB = "UPDATE {0}.JEDI_Dataset_Contents SET attemptNr=attemptNr+1 ".format(panda_config.schemaJEDI)
    sqlAB += "WHERE jediTaskID=:jediTaskID "
    sqlA = sqlAB + "AND fileID=:fileID "
    self.cur.execute(sqlA + comment, varMap)
"""


def test_a_statement_built_from_another_statements_name():
    got, _ = _texts(H3)
    assert got.get(6) == [
        "UPDATE {}.JEDI_Dataset_Contents SET attemptNr=attemptNr+1 WHERE jediTaskID=:jediTaskID AND fileID=:fileID {}"
    ]


def test_an_fstring_over_a_parenthesised_assignment():
    src = """
def wn(self, site):
    sql = (
        "SELECT a FROM ATLAS_PANDA.worker_node_metrics "
        "WHERE site=:site "
    )
    if site:
        sql = f"{sql} AND x=:x"
    self.cur.execute(sql, varMap)
"""
    got, _ = _texts(src)
    assert got.get(9) == [
        "SELECT a FROM ATLAS_PANDA.worker_node_metrics WHERE site=:site ",
        "SELECT a FROM ATLAS_PANDA.worker_node_metrics WHERE site=:site  AND x=:x",
    ]


def test_percent_filled_from_an_if_else():
    src = """
def num(self, use_archive):
    sql_jt = "SELECT COUNT(*) FROM %s WHERE status=:status "
    if use_archive:
        table = f"{panda_config.schemaPANDA}.jobsArchived4"
    else:
        table = f"{panda_config.schemaPANDA}.jobsActive4"
    self.cur.execute((sql_jt + comment) % table, varMap)
"""
    got, _ = _texts(src)
    assert got.get(8) == [
        "SELECT COUNT(*) FROM {}.jobsActive4 WHERE status=:status {}",
        "SELECT COUNT(*) FROM {}.jobsArchived4 WHERE status=:status {}",
    ]


def test_a_value_known_only_at_run_time_stays_a_hole():
    src = """
def f(self, x):
    archView = self.getView(x)
    sql = f"SELECT a FROM {archView} WHERE b=:b"
    self.cur.execute(sql, varMap)
"""
    got, _ = _texts(src)
    assert got.get(5) == ["SELECT a FROM {} WHERE b=:b"]


# ---------------------------------------------------------------- control flow


def test_an_in_list_over_an_unknown_sequence_is_one_part():
    """No pass at all would give ``IN )``, which no run sends: an unknown loop runs at least once.  However many
    passes run, what they append is one part, and ``sql[:-1]`` cuts the comma every pass ends with."""
    src = """
def f(self, ids):
    sql = "SELECT a FROM T WHERE id IN ("
    for i, x in enumerate(ids):
        sql += f":id{i},"
    sql = sql[:-1] + ")"
    self.cur.execute(sql, varMap)
"""
    got, reading = _texts(src)
    assert got.get(7) == ["SELECT a FROM T WHERE id IN ({})"]
    ((_at, hole),) = reading.cover(7)[0][0].holes
    assert (hole.reason, hole.repeats) == (sqleval.REPEATED, (":id{},",))


def test_the_same_flag_tested_twice_is_taken_the_same_way():
    src = """
def f(self, useJEDI):
    if useJEDI:
        sql = "SELECT a FROM JEDI_Tasks "
    else:
        sql = "SELECT a FROM jobsActive4 "
    if useJEDI:
        sql += "WHERE jediTaskID=:t"
    else:
        sql += "WHERE PandaID=:p"
    self.cur.execute(sql, varMap)
"""
    got, _ = _texts(src)
    assert got.get(11) == [
        "SELECT a FROM JEDI_Tasks WHERE jediTaskID=:t",
        "SELECT a FROM jobsActive4 WHERE PandaID=:p",
    ]


BACKEND = """
def f(self):
    sql = "INSERT INTO T (a,b) VALUES "
    if self.backend == "oracle":
        sql += "(SEQ.nextval,"
    else:
        sql += "(:nextval,"
    BETWEEN
    if self.backend == "oracle":
        sql += "SEQ.currval)"
    else:
        sql += ":currval)"
    self.cur.execute(sql, varMap)
"""


@pytest.mark.parametrize(
    "between, combinations",
    [
        ("pass", 2),
        ("self.cur.arraysize = 100; self.cur.execute(other, {})", 2),
        ("self.reconnect()", 4),
        ("log(self)", 4),
        ("self.backend = pick()", 4),
        ("self = other", 4),
    ],
)
def test_a_test_of_an_attribute_is_taken_the_same_way_until_the_object_is_handed_on(
    between, combinations
):
    """``insertTaskParamsPanda`` tests ``self.backend`` twice.  A call on another object
    (the cursor) is taken not to change it; one handed ``self``, an assignment to the
    attribute and a rebinding of ``self`` forget it."""
    reading = _reading(BACKEND.replace("BETWEEN", between))
    texts = reading.texts[max(reading.texts)]
    assert len(texts) == combinations
    if combinations == 2:
        assert sorted(texts) == [
            "INSERT INTO T (a,b) VALUES (:nextval,:currval)",
            "INSERT INTO T (a,b) VALUES (SEQ.nextval,SEQ.currval)",
        ]


def test_a_test_of_a_loop_variables_attribute_is_taken_again_on_each_pass():
    """The same within a pass; the next pass's ``fileSpec`` is another object."""
    src = """
def f(self, files):
    sql = "S "
    for fileSpec in files:
        if fileSpec.type == "input":
            sql += "A"
        else:
            sql += "B"
        if fileSpec.type == "input":
            sql += "A "
        else:
            sql += "B "
    self.cur.execute(sql, varMap)
"""
    reading = _reading(src)
    line = max(reading.texts)
    assert sorted(reading.texts[line]) == ["S {} "]
    # What one pass appends: never ``AB`` -- one object within a pass.
    ((_at, hole),) = reading.cover(line)[0][0].holes
    assert hole.repeats == ("AA ", "BB ")


def test_a_handler_starts_from_every_state_of_the_try_body():
    src = """
def f(self):
    sql = "A"
    try:
        sql = "B"
        self.cur.execute(sql, varMap)
        sql = "C"
        self.cur.execute(sql, varMap)
    except Exception:
        self.cur.execute(sql + " retry", varMap)
"""
    got, _ = _texts(src)
    assert got.get(10) == ["A retry", "B retry", "C retry"]


def test_while_true_with_break():
    src = """
def f(self):
    n = 0
    sql = "X"
    while True:
        sql = "Y"
        if n:
            break
        self.cur.execute(sql, varMap)
        break
    self.cur.execute(sql + "Z", varMap)
"""
    got, _ = _texts(src)
    assert (got.get(9), got.get(11)) == (["Y"], ["YZ"])


def test_percent_with_a_dict():
    src = """
def f(self):
    sql = "SELECT %(c)s FROM %(t)s" % {"c": "a", "t": "T"}
    self.cur.execute(sql, varMap)
"""
    got, _ = _texts(src)
    assert got.get(4) == ["SELECT a FROM T"]


def test_a_str_default_parameter_reads_as_its_default_or_as_what_a_caller_passes():
    src = """
def f(self, x, tablename="ATLAS_PANDA.Datasets"):
    sql = "INSERT INTO %s (a) VALUES (:a)" % tablename
    self.cur.execute(sql, varMap)
"""
    got, _ = _texts(src)
    assert got.get(4) == [
        "INSERT INTO ATLAS_PANDA.Datasets (a) VALUES (:a)",
        "INSERT INTO {} (a) VALUES (:a)",
    ]


def test_a_str_default_does_not_decide_the_tests_on_its_parameter():
    """``get_shares``: ``parents=""`` runs no ``WHERE``; every corpus caller
    passes None or a share's name.  What a caller passes is not known, not a
    str: ``parents is None`` is undecided on it."""
    src = """
def f(self, parents=""):
    sql = "SELECT NAME FROM T "
    if parents == "":
        pass
    elif parents is None:
        sql += "WHERE parent IS NULL"
    elif isinstance(parents, str):
        sql += "WHERE parent = :parent"
    self.cur.execute(sql, var_map)
"""
    got, reading = _texts(src)
    assert got.get(10) == [
        "SELECT NAME FROM T ",
        "SELECT NAME FROM T WHERE parent = :parent",
        "SELECT NAME FROM T WHERE parent IS NULL",
    ]
    assert sorted(one.text for one in reading.cover(10)[0]) == got[10]


def test_a_piece_keeps_its_way_through_a_return_the_call_and_the_names():
    """Where a piece was written, then each statement that put it into a name,
    the callee's ``return`` and the call it came back from: the way
    ``getJobs``'s terms take out of ``construct_where_clause``."""
    src = """
def build(self, mem):
    where = ""
    if mem:
        where += "AND ram<=:mem "
    return where
def f(self, mem):
    sql = "SELECT a FROM T WHERE x=:x "
    w = build(self, mem)
    sql += w
    self.cur.execute(sql, varMap)
"""
    build = _function(src, "build")
    reading = _reading(src, name="f", resolver=_resolver({"build": build}))
    (one,) = [one for one in reading.cover(11)[0] if "ram" in one.text]
    ways = {one.text[start:end]: way for start, end, way in one.pieces}
    assert [(type(node).__name__, node.lineno) for node in ways["AND ram<=:mem "]] == [
        ("AugAssign", 5),
        ("Return", 6),
        ("Call", 9),
        ("Assign", 9),
        ("AugAssign", 10),
    ]
    assert [
        (type(node).__name__, node.lineno)
        for node in ways["SELECT a FROM T WHERE x=:x "]
    ] == [("Assign", 8)]


def test_join_over_a_known_list_and_a_comprehension():
    src = """
def f(self):
    cols = ["a", "b"]
    sql = "SELECT " + ",".join(cols) + " FROM T WHERE x IN (" + ",".join([f":x{i}" for i in range(3)]) + ")"
    self.cur.execute(sql, varMap)
"""
    got, _ = _texts(src)
    assert got.get(5) == ["SELECT a,b FROM T WHERE x IN (:x0,:x1,:x2)"]


def test_a_value_one_pass_leaves_is_read_by_the_next():
    """``AC`` would need ``xs`` empty, and an unknown loop runs at least once."""
    src = """
def f(self, xs):
    sql = "A"
    for x in xs:
        self.cur.execute(sql, varMap)
        sql = "B"
    self.cur.execute(sql + "C", varMap)
"""
    got, _ = _texts(src)
    assert (got.get(5), got.get(7)) == (["A", "B"], ["BC"])


def test_statements_after_the_call_do_not_change_what_it_ran():
    src = """
def f(self, flag):
    sql = "X"
    if flag:
        sql = "Y"
    self.cur.execute(sql, varMap)
    sql = "Z"
    other = sql + "W"
"""
    got, _ = _texts(src)
    assert got.get(6) == ["X", "Y"]


def test_re_sub_on_a_known_pattern():
    """``peekJobLog``'s archive table names."""
    src = """
def f(self, table):
    table = f"{panda_config.schemaPANDAARCH}.jobsArchived"
    files = re.sub("jobsArchived", "filesTable_ARCH", table)
    self.cur.execute(f"SELECT a FROM {files}", varMap)
"""
    got, _ = _texts(src)
    assert got.get(5) == ["SELECT a FROM {}.filesTable_ARCH"]


def test_a_compound_test_twice_is_taken_the_same_way():
    """``archiveJob``'s ``sql0``."""
    src = """
def f(self, a, b):
    if a or b:
        sql0 = "SELECT x FROM jobsDefined4 "
    else:
        sql0 = "SELECT x FROM jobsActive4 FOR UPDATE "
    if a or b:
        self.cur.execute(sql0, varMap)
    else:
        self.cur.execute(sql0 + "-- lock", varMap)
"""
    got, _ = _texts(src)
    assert (got.get(8), got.get(10)) == (
        ["SELECT x FROM jobsDefined4 "],
        ["SELECT x FROM jobsActive4 FOR UPDATE -- lock"],
    )


def test_a_container_changed_through_a_chain_of_calls_is_not_empty():
    """``_read_unprocessed_inputs``: ``m.setdefault(k, set()).add(v)``."""
    src = """
def f(self):
    m = {}
    for k, v in self.cur.fetchall():
        m.setdefault(k, set()).add(v)
    if m:
        self.cur.execute("SELECT a FROM ddm_endpoint", varMap)
"""
    _, reading = _texts(src)
    assert 7 in reading.reached


def test_dead_code_stays_unreached():
    src = """
def f(self):
    dry_run = False
    if dry_run:
        self.cur.execute("SELECT a FROM T", varMap)
"""
    _, reading = _texts(src)
    assert 5 not in reading.reached


def test_a_retry_sees_what_the_try_body_rebound():
    """``retryJob``'s ``sql1``: the rebound text runs again only on a path through the handler, and is read
    as any other path's text is."""
    src = """
def f(self, x):
    sql1 = "SELECT a FROM T"
    for i in range(3):
        try:
            self.cur.execute(sql1, varMap)
            if x:
                sql1 = "INSERT INTO T"
                self.cur.execute(sql1, varMap)
            return
        except Exception:
            continue
"""
    got, reading = _texts(src)
    assert got.get(6) == ["INSERT INTO T", "SELECT a FROM T"]


def test_a_count_an_unknown_loop_carries_is_unknown():
    """Not the 1..2 a fixed number of passes would give it: no pass may append, or any number of them."""
    src = """
def f(self, files):
    sql = "SELECT a FROM T"
    n = 0
    for f in files:
        n += 1
        if n > 5:
            sql += " WHERE b=:b"
    self.cur.execute(sql, varMap)
"""
    got, _ = _texts(src)
    assert got.get(9) == ["SELECT a FROM T", "SELECT a FROM T{} WHERE b=:b"]


def test_a_text_a_pass_rebinds_is_read_beside_the_one_before():
    src = """
def f(self, xs):
    sql = "SELECT a FROM T"
    for x in xs:
        if x.k:
            sql = "SELECT a FROM U"
    self.cur.execute(sql, varMap)
"""
    got, _ = _texts(src)
    assert got.get(7) == ["SELECT a FROM T", "SELECT a FROM U"]


def test_bind_names_built_pass_by_pass():
    src = """
def f(self, vals):
    params = []
    for i, v in enumerate(vals):
        params.append(f":v{i}")
    sql = f"SELECT a FROM T WHERE b IN ({','.join(params)})"
    self.cur.execute(sql, varMap)
"""
    got, _ = _texts(src)
    assert got.get(7) == ["SELECT a FROM T WHERE b IN ({})"]


def test_a_flag_a_loop_may_set_leaves_both_arms():
    src = """
def f(self, xs):
    found = False
    for x in xs:
        if x.ok:
            found = True
    if found:
        self.cur.execute("SELECT a FROM T", varMap)
    else:
        self.cur.execute("SELECT a FROM U", varMap)
"""
    _, reading = _texts(src)
    assert (8 in reading.reached, 10 in reading.reached) == (True, True)


def test_the_tail_of_a_block_still_binds_what_a_later_call_reads():
    """Run for every call at once, ``rows`` is bound by the block after the first call."""
    src = """
def f(self, a):
    sql = "SELECT a FROM T"
    rows = []
    if a:
        self.cur.execute(sql, varMap)
        rows = self.cur.fetchall()
    for row in rows:
        self.cur.execute("SELECT b FROM U", varMap)
"""
    _, reading = _texts(src)
    assert (reading.mode, 9 in reading.reached) == ("function", True)


USEAND = """
def f(self, crit):
    sql = "UPDATE T SET a=1 WHERE"
    useAND = False
    for k, v in crit.items():
        if useAND:
            sql += " AND"
        else:
            useAND = True
        sql += f" {k}=:{k}"
    self.cur.execute(sql, varMap)
"""


def test_a_flag_a_pass_sets_keeps_what_the_passes_before_left():
    """The first pass takes ``WHERE``, a later one ``AND``: the first pass's text is kept as it is, and the later
    passes' are one part."""
    got, reading = _texts(USEAND)
    assert got.get(11) == [
        "UPDATE T SET a=1 WHERE {}=:{}",
        "UPDATE T SET a=1 WHERE {}=:{}{}",
    ]
    later = [
        hole for one in reading.cover(11)[0] for _at, hole in one.holes if hole.repeats
    ]
    assert [hole.repeats for hole in later] == [(" AND {}=:{}",)]


LENMAP = """
def f(self, map):
    sql1 = "SELECT a FROM T"
    varMap = {}
    for key in map:
        if len(varMap) == 0:
            sql1 += f" WHERE {key}=:{key}"
        else:
            sql1 += f" AND {key}=:{key}"
        varMap[f":{key}"] = map[key]
    self.cur.execute(sql1, varMap)
"""


def test_what_a_loop_of_unknown_length_appends_is_one_part_whatever_it_held():
    """``JobSpec.columnNames``: one ``SELECT {} FROM``, not one per number of passes.  The hole keeps what a pass
    appends, and the comma every pass ends with is outside it, so ``ret[:-1]`` cuts the comma, not the part."""
    src = """
def f(self, cls, prefix=None):
    ret = ""
    for attr in cls.attributes:
        if prefix is not None:
            ret += f"{prefix}."
        ret += f"{attr},"
    ret = ret[:-1]
    self.cur.execute("SELECT " + ret + " FROM t", varMap)
"""
    got, reading = _texts(src)
    assert got.get(9) == ["SELECT {} FROM t"]
    ((_at, hole),) = reading.cover(9)[0][0].holes
    assert (hole.stmt.lineno, hole.source, hole.reason) == (4, "ret", sqleval.REPEATED)
    assert hole.repeats == ("{}.{},", "{},")


def test_a_loop_whose_passes_may_append_nothing_appends_on_the_paths_one_did():
    """``bindUpdateChangesExpression``: ``SET {}`` where some attribute changed, ``SET`` with nothing where none did."""
    src = """
def f(self, spec):
    ret = ""
    for attr in spec.attributes:
        if attr in spec._changedAttrs:
            ret += f"{attr}=:{attr},"
    ret = ret[:-1]
    ret += " "
    self.cur.execute("UPDATE t SET " + ret + "WHERE x=:x", varMap)
"""
    got, _ = _texts(src)
    assert got.get(9) == ["UPDATE t SET  WHERE x=:x", "UPDATE t SET {} WHERE x=:x"]


def test_a_list_a_loop_appends_to_is_one_item_a_join_renders_as_the_part():
    """``getCriteriaForGlobalShares``: the ranking is ``(CASE {} ELSE {} END)`` however many shares there are, and
    the hole keeps both members a pass appends.  ``len`` of the list is a lower bound: it has the part, at least."""
    src = """
def f(self, leaves):
    i = 0
    tmp_list = []
    for leave in leaves:
        if leave.name == "Test":
            tmp_list.append(f"WHEN gshare=:leave{i} THEN 0")
        else:
            tmp_list.append("WHEN gshare=:leave{0} THEN {0}".format(i))
        i += 1
    bindings = " ".join(tmp_list)
    self.cur.execute(f"SELECT x FROM t ORDER BY (CASE {bindings} ELSE {len(leaves)} END)", varMap)
    if len(tmp_list) > 1:
        self.cur.execute("SELECT many FROM t", varMap)
"""
    got, reading = _texts(src)
    assert got.get(12) == ["SELECT x FROM t ORDER BY (CASE {} ELSE {} END)"]
    repeated = [hole for _at, hole in reading.cover(12)[0][0].holes if hole.repeats]
    assert [hole.repeats for hole in repeated] == [
        ("WHEN gshare=:leave{} THEN 0", "WHEN gshare=:leave{} THEN {}")
    ]
    assert 14 in reading.reached


def test_a_list_of_objects_a_loop_appends_to_is_not_one_part():
    """``_read_unprocessed_inputs``' ``file_spec_list``: no text, so a pass starts without it, as it did -- kept, its
    length decided every test on it and split the paths of every pass (the function's run went 3 s to 35 s)."""
    src = """
def f(self, rows):
    specs = []
    for row in rows:
        specs.append(make(row))
    if len(specs) > 1:
        self.cur.execute("SELECT a FROM T", varMap)
    else:
        self.cur.execute("SELECT b FROM T", varMap)
"""
    _, reading = _texts(src)
    assert (7 in reading.reached, 9 in reading.reached) == (True, True)


def test_a_name_a_loop_reads_is_not_one_part():
    """What a pass appends depends on what the name held: the passes are read one by one, as before."""
    src = """
def f(self, xs):
    sql = "SELECT a FROM t WHERE "
    for x in xs:
        if sql.endswith("WHERE "):
            sql += f"{x}=1"
        else:
            sql += f" AND {x}=1"
    self.cur.execute(sql, varMap)
"""
    got, reading = _texts(src)
    assert "SELECT a FROM t WHERE {}=1 AND {}=1" in got.get(9)
    assert not any(
        hole.repeats for one in reading.cover(9)[0] for _at, hole in one.holes
    )


def test_len_of_a_dict_written_at_an_unknown_key():
    """``queryDatasetWithMap``: after the first write ``len(varMap) == 0`` is False, so exactly ``WHERE`` then ``AND``
    -- the later passes' ``AND`` as one part."""
    got, _ = _texts(LENMAP)
    assert got.get(11) == [
        "SELECT a FROM T WHERE {}=:{}",
        "SELECT a FROM T WHERE {}=:{}{}",
    ]


def test_a_dict_read_by_one_key_is_tracked_for_that_key_only():
    """``getScoutJobData_JEDI``: a write to another key brings in nothing."""
    src = """
def f(self, xs, y):
    returnMap = {}
    if y > 1:
        returnMap["walltime"] = y
    for x in xs:
        if x.a:
            returnMap["outDiskCount"] = x.b
    if "outDiskCount" in returnMap:
        self.cur.execute("SELECT b FROM U", varMap)
"""
    _, reading = _texts(src)
    assert reading.reached == {10}
    assert "y" not in sqleval.Ref(_function(src), {}, None).relevant


def test_a_guard_set_under_a_test_on_a_constant_is_read_open():
    """The cost of not tracking the tests that decide a guard: a superset, never less."""
    src = """
def f(self, x):
    mode = "fast"
    flag = False
    if mode == "slow":
        flag = True
    if flag:
        self.cur.execute("SELECT a FROM T", varMap)
"""
    _, reading = _texts(src)
    assert 8 in reading.reached


# ---------------------------------------------------------------- joined paths and expansion

PAIRS = """
def f(self, a, b):
    cols = "x"
    vals = ":x"
    if a:
        cols += ",y"
        vals += ",:y"
    if b:
        cols += ",z"
        vals += ",:z"
    sql = "INSERT INTO T (" + cols + ") VALUES (" + vals + ")"
    self.cur.execute(sql, varMap)
"""


def test_an_inserts_columns_and_values_stay_paired():
    """Joined at the same points, they share their dimensions: no ``(x,y) VALUES (:x,:z)``."""
    reading = _reading(PAIRS)
    assert sorted(reading.texts[12]) == sorted(
        [
            "INSERT INTO T (x) VALUES (:x)",
            "INSERT INTO T (x,y) VALUES (:x,:y)",
            "INSERT INTO T (x,z) VALUES (:x,:z)",
            "INSERT INTO T (x,y,z) VALUES (:x,:y,:z)",
        ]
    )
    assert len(reading.values[12]) == 1


MANY = (
    "def f(self, "
    + ", ".join(f"c{i}" for i in range(12))
    + "):\n    sql = 'SELECT a FROM T WHERE 1=1 '\n"
    + "".join(f"    if c{i}:\n        sql += 'AND k{i}=:k{i} '\n" for i in range(12))
    + "    self.cur.execute(sql, varMap)\n"
)


def test_independent_conditions_are_one_path():
    """Twelve of them: 4,096 statements, one value, at most two paths (``getTasksWithCriteria_JEDI``'s shape)."""
    reading = _reading(MANY)
    line = max(reading.values)
    assert len(reading.texts[line]) == 4096
    assert len(reading.values[line]) == 1
    assert reading.stats["max_paths"] <= 2


def test_a_cover_shows_every_piece_in_few_statements():
    reading = _reading(MANY)
    line = max(reading.values)
    shown, complete = reading.cover(line)
    pieces = {f"AND k{i}=:k{i} " for i in range(12)}
    assert complete
    assert len(shown) == 2
    assert all(any(piece in one.text for one in shown) for piece in pieces)


CALLEE = """
def helper(prefix, n):
    return prefix + "_" + n

def f(self):
    name = helper("ATLAS_PANDA.jobs", "Active4")
    self.cur.execute("SELECT a FROM " + name, varMap)
"""


def test_a_callee_runs_with_the_calls_arguments():
    tree = ast.parse(textwrap.dedent(CALLEE))
    defs = {n.name: n for n in tree.body}
    reading = sqleval.evaluate(defs["f"], None, _resolver(defs))
    assert sorted(reading.texts[7]) == ["SELECT a FROM ATLAS_PANDA.jobs_Active4"]
    shown, _ = reading.cover(7)
    assert _lines(shown[0].spans) == [(7, 0, 14), (6, 14, 30), (3, 30, 31), (6, 31, 38)]


def test_a_callee_is_not_followed_without_a_resolver():
    tree = ast.parse(textwrap.dedent(CALLEE))
    defs = {n.name: n for n in tree.body}
    reading = sqleval.evaluate(defs["f"], None, None)
    assert sorted(reading.texts[7]) == ["SELECT a FROM {}"]


def test_a_statement_only_an_empty_loop_leads_to_is_not_read():
    """The cost of running an unknown loop at least once, pinned so a change shows: rows may be none."""
    src = """
def f(self):
    self.cur.execute("SELECT a FROM T", varMap)
    res = self.cur.fetchall()
    found = False
    for row in res:
        found = True
    if not found:
        self.cur.execute("INSERT INTO T (a) VALUES (1)", varMap)
"""
    reading = _reading(src)
    assert sorted(reading.reached) == [3]


def test_a_flag_a_loop_carries_decides_whether_the_call_runs():
    """``propagateResultToJEDI``'s ``toUpdateFlag``: no ``SET  WHERE`` from a run that added nothing."""
    src = """
def f(self, stats):
    flag = False
    sql = "UPDATE T SET "
    for k in stats:
        if stats[k] == 0:
            continue
        sql += f"{k}={k}+1,"
        flag = True
    sql = sql[:-1]
    sql += " WHERE id=:id"
    if flag:
        self.cur.execute(sql, varMap)
"""
    reading = _reading(src)
    # One part, on the paths some pass appended to: those the flag is set on.
    assert sorted(reading.texts[13]) == ["UPDATE T SET {} WHERE id=:id"]


def test_a_test_on_a_joined_name_splits_every_name_the_same_way():
    src = """
def f(self, a):
    sql = "SELECT a FROM T"
    tab = "T"
    if a:
        sql = "SELECT a FROM U"
        tab = "U"
    if sql.endswith("U"):
        self.cur.execute(sql + " WHERE u=1", varMap)
    else:
        self.cur.execute(sql, varMap)
"""
    reading = _reading(src)
    assert sorted(reading.texts.get(9, [])) == ["SELECT a FROM U WHERE u=1"]
    assert sorted(reading.texts.get(11, [])) == ["SELECT a FROM T"]


# ---------------------------------------------------------------- where a piece came from


def test_pieces_keep_the_statement_that_wrote_them_and_holes_say_why():
    reading = _reading(H3)
    shown, _ = reading.cover(6)
    assert _lines(shown[0].spans) == [
        (3, 0, 58),
        (4, 58, 87),
        (5, 87, 106),
        (6, 106, 108),
    ]
    assert [(at, h.stmt.lineno, h.source, h.reason) for at, h in shown[0].holes] == [
        (7, 3, "panda_config.schemaJEDI", "an attribute"),
        (106, 6, "comment", "a name the function does not bind"),
    ]


def test_and_the_statement_that_put_them_into_the_calls_name():
    """``sqlA = sqlAB + ...`` at line 5 put everything but the comment into ``sqlA``: ``Span.node``'s meaning."""
    reading = _reading(H3)
    shown, _ = reading.cover(6)
    assert _lines(shown[0].carried) == [(5, 0, 106), (6, 106, 108)]


COPY = """
def f(self, res):
    sqlIn = "INSERT INTO T (a) VALUES (:a)"
    sqlUp = "UPDATE T SET a=:a "
    if res is None:
        sql = sqlIn
    else:
        sql = sqlUp
        sql += "WHERE b=:b "
        sql = sql[:-1]
    self.cur.execute(sql, varMap)
"""


def test_a_copy_under_a_test_is_the_statement_whose_condition_counts():
    """``updateSiteData``'s ``sql = sqlIn`` under ``if res is None``, not the literal outside it.

    A name's own pieces keep the statement that first put them there: ``sql = sql[:-1]`` moves nothing.
    """
    reading = _reading(COPY)
    shown, _ = reading.cover(11)
    assert sorted(
        (one.text, _lines(one.carried), _lines(one.spans)) for one in shown
    ) == [
        ("INSERT INTO T (a) VALUES (:a)", [(6, 0, 29)], [(3, 0, 29)]),
        (
            "UPDATE T SET a=:a WHERE b=:b",
            [(8, 0, 18), (9, 18, 28)],
            [(4, 0, 18), (9, 18, 28)],
        ),
    ]


# ---------------------------------------------------------------- widening and work

ALIKE = (
    "def f(self, mode):\n"
    + "".join(
        f"    x{i} = self.get({i})\n    if x{i} is None:\n        x{i} = 0\n"
        for i in range(10)
    )
    + "    if mode:\n        sql = 'SELECT a FROM A WHERE 1=1 '\n"
    + "    else:\n        sql = 'SELECT a FROM B WHERE 1=1 '\n"
    + "".join(f"    if x{i}:\n        sql += 'AND k{i}=1 '\n" for i in range(10))
    + "    self.cur.execute(sql, varMap)\n"
)


def test_widening_keeps_a_name_every_path_holds_alike():
    """``updateJob``'s route constant: past the path cap, ``mode`` stays and only the differing names go."""
    reading = _reading(ALIKE, {"mode": True})
    line = max(reading.values)
    assert line in reading.widened_at
    assert len(reading.texts[line]) == 1024
    assert all(t.startswith("SELECT a FROM A ") for t in reading.texts[line])


def test_a_widened_run_over_every_call_stands():
    """It is not run again call by call; each widened line says so."""
    reading = _reading(
        ALIKE + "    self.cur.execute(sql + 'LIMIT 1', varMap)\n", {"mode": True}
    )
    first, last = sorted(reading.values)
    assert reading.mode == "function"
    assert first in reading.widened_at and last in reading.widened_at
    assert all(
        t.startswith("SELECT a FROM A ")
        for t in [*reading.texts[first], *reading.texts[last]]
    )


def test_every_name_joined_gives_a_dicts_keys_as_their_union():
    """``setScoutJobData_JEDI``'s ``SET`` from the keys of the dict ``getScoutJobData_JEDI`` returns.

    A key only some paths add is counted by a loop over the dict (the union of the columns); ``in`` does not
    decide it.
    """
    src = (
        "def f(self):\n    d = {}\n"
        + "".join(
            f"    x{i} = self.get({i})\n    if x{i} is None:\n        x{i} = 'p{i}'\n"
            for i in range(10)
        )
        + "".join(f"    if self.c{i}():\n        d['k{i}'] = x{i}\n" for i in range(10))
        + "    sql = 'UPDATE T SET '\n    for k, v in d.items():\n        sql += f'{k}=:{k},'\n"
        + "    sql = sql[:-1]\n    sql += ' WHERE id=:id'\n    if 'k0' in d:\n"
        + "        sql += ' AND a=1'\n    self.cur.execute(sql, varMap)\n"
    )
    reading = _reading(src)
    line = max(reading.values)
    columns = ",".join(f"k{i}=:k{i}" for i in range(10))
    assert line in reading.widened_at
    assert reading.stats["joined_all"] >= 1
    assert sorted(reading.texts[line]) == sorted(
        [
            f"UPDATE T SET {columns} WHERE id=:id",
            f"UPDATE T SET {columns} WHERE id=:id AND a=1",
        ]
    )


def test_a_widened_callee_widens_its_caller():
    src = (
        "def helper(n):\n    t = 'T'\n"
        + "".join(
            f"    x{i} = g({i})\n    if x{i} is None:\n        x{i} = 0\n"
            for i in range(10)
        )
        + "".join(f"    if x{i}:\n        t += '{i}'\n" for i in range(10))
        + "    return t\n\n"
        + "def f(self):\n    name = helper(1)\n    self.cur.execute('SELECT a FROM ' + name, varMap)\n"
    )
    tree = ast.parse(src)
    defs = {n.name: n for n in tree.body}
    reading = sqleval.evaluate(defs["f"], None, _resolver(defs))
    line = max(reading.values)
    assert line in reading.widened_at
    assert len(reading.texts[line]) == 1024


def test_a_run_is_capped_by_its_work_not_by_seconds(monkeypatch):
    """The same run twice stops at the same step with the same count, on any machine at any load."""
    monkeypatch.setattr(sqleval, "WORK_BUDGET", 10)
    func = ast.parse(MANY).body[0]
    one, two = sqleval.evaluate(func), sqleval.evaluate(func)
    assert one.overflow == two.overflow == "more than 10 steps"
    assert one.stats["work"] == two.stats["work"]


# ---------------------------------------------------------------- what the readers ask besides a call's text


def test_the_name_a_call_runs_drawn_as_each_covering_statement_was():
    """``Execution.head``: the comment the call adds is not part of the name's text."""
    src = """
def f(self, a):
    sql = "SELECT a FROM T"
    if a:
        sql += " WHERE b=:b"
    self.cur.execute(sql + comment, varMap)
"""
    reading = _reading(src)
    assert sorted(
        (one.text, [h.text for h in held]) for one, held in reading.heads(6, "sql")
    ) == [
        ("SELECT a FROM T WHERE b=:b{}", ["SELECT a FROM T WHERE b=:b"]),
        ("SELECT a FROM T{}", ["SELECT a FROM T"]),
    ]


def test_what_a_copied_name_held_where_it_was_copied():
    """sqlwrite cuts ``sql = sqlIn`` in ``sqlIn``'s own terms: its text there and who put each piece into it."""
    reading = _reading(COPY)
    assert [(one.text, _lines(one.carried)) for one in reading.copied(6, "sqlIn")] == [
        ("INSERT INTO T (a) VALUES (:a)", [(3, 0, 29)])
    ]
    found = reading.copied(6, "sqlIn", "INSERT INTO T (a) VALUES (:a)")
    assert [one.text for one in found] == ["INSERT INTO T (a) VALUES (:a)"]


FLAGGED = """
def f(self, flag, res):
    sql = "SELECT a FROM T"
    if flag:
        sql += " WHERE b=:b"
    if res is None:
        sql += " ORDER BY a"
    self.cur.execute(sql, varMap)
"""


def test_a_constant_decides_a_test_a_value_from_elsewhere_does_not():
    """``decided_here``: the way every path took an ``if``'s test; both ways is undecided."""
    func = _function(FLAGGED)
    tests = [n.test for n in func.body if isinstance(n, ast.If)]
    assert [sqleval.evaluate(func, {"flag": True}).decided(t) for t in tests] == [
        True,
        None,
    ]
    assert [sqleval.evaluate(func).decided(t) for t in tests] == [None, None]


# ---------------------------------------------------------------- what sql.py reads with the evaluator on
#
# Tests in test_codemap.py and test_codemap_calls.py marked ``reassembly(kind)``
# pin how sql.py's reassembly reads a statement the evaluator reads otherwise
# by design.  Under BAMBOO_SQL_EVAL=1 they are skipped (tests/conftest.py says
# each kind), and these pin what the evaluator reads instead, one test per
# kind.  The marked tests go when the reassembly does.


def _run_sql(src: str, name: str = "f") -> list[str]:
    """The statements ``sql.executions`` reads of *name* with the evaluator on."""
    tree = ast.parse(textwrap.dedent(src))
    pathcond.attach_parents(tree)
    func = next(
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name
    )
    with sql.evaluated():
        return [run.sql for run in sql.executions(func)]


def _read_side(tmp_path, text: str) -> dict:
    """``sqlwrite.read_side`` over one module, ``pkg/db.py``, as a build reads it."""
    from bamboo.codemap.gitsource import blob_sha
    from bamboo.codemap.models import SourceModule
    from bamboo.codemap.panda import attribution
    from bamboo.codemap.panda.recognizers import progress, sqlwrite

    root = tmp_path / "pkg"
    root.mkdir(parents=True, exist_ok=True)
    (root / "__init__.py").write_text("")
    (root / "db.py").write_text(text)
    modules = [
        SourceModule(
            package="pkg",
            rel_path="pkg/db.py",
            tree=ast.parse(text),
            source=text,
            blob_sha=blob_sha(text),
        )
    ]
    attributor = attribution.SpecAttributor(
        progress.spec_attributes(modules), attribution.class_bases(modules)
    )
    calls = sqlwrite.corpus_calls(modules, {"pkg": root})
    with sql.across_calls(calls):
        return sqlwrite.read_side(modules, attributor, calls).statements


_FAILED_ONLY = (
    "def get(cur, failed_only):\n"
    "    failed = \"AND jobStatus IN ('failed','closed') \"\n"
    "    sql = 'SELECT PandaID FROM jobsActive4 WHERE jediTaskID=:t '\n"
    "    if failed_only:\n"
    "        sql += failed\n"
    "    cur.execute(sql, {':t': 1})\n"
)
_WRAPPED = (
    "def get(cur, limit):\n"
    "    sql = 'SELECT PandaID FROM jobsActive4 WHERE a=:a '\n"
    "    if limit:\n"
    "        sql = 'SELECT * FROM (' + sql\n"
    "        sql += ') WHERE ROWNUM <= :n'\n"
    "    cur.execute(sql, {':a': 1})\n"
)


@pytest.mark.parametrize(
    "src, texts",
    [
        pytest.param(
            _FAILED_ONLY,
            [
                "SELECT PandaID FROM jobsActive4 WHERE jediTaskID=:t ",
                "SELECT PandaID FROM jobsActive4 WHERE jediTaskID=:t "
                "AND jobStatus IN ('failed','closed') ",
            ],
            id="a fragment held under a local name",
        ),
        pytest.param(
            _WRAPPED,
            [
                "SELECT * FROM (SELECT PandaID FROM jobsActive4 WHERE a=:a ) "
                "WHERE ROWNUM <= :n",
                "SELECT PandaID FROM jobsActive4 WHERE a=:a ",
            ],
            id="a statement wrapped around itself",
        ),
    ],
)
def test_a_fragment_an_if_adds_is_one_statement_with_it_and_one_without(
    tmp_path, src, texts
):
    """``reassembly("representation")``: one statement per way the ``if`` goes.

    The reassembly reads one statement holding the fragment as optional.  What
    ``sqlwrite`` reads of it -- the fragment added under the ``if`` -- is the
    same either way.
    """
    assert sorted(_run_sql(src, "get")) == sorted(texts)
    off = _read_side(tmp_path / "off", src)
    sqleval.clear_memo()
    with sql.evaluated():
        on = _read_side(tmp_path / "on", src)
    assert off and on == off


def test_a_table_an_if_adds_to_the_list_a_loop_reads_is_another_statement():
    """``reassembly("representation")``: ``getPandaIDsWithTask_JEDI``'s table list.

    The reassembly reads one statement over all three tables; the evaluator one
    per way ``onlyActive`` goes, the archive only in the second.
    """
    texts = _run_sql(
        "def f(self, jediTaskID, onlyActive):\n"
        "    tables = [f'{schemaPANDA}.jobsDefined4', f'{schemaPANDA}.jobsActive4']\n"
        "    if not onlyActive:\n"
        "        tables += [f'{schemaARCH}.jobsArchived']\n"
        "    sqlP = ''\n"
        "    for tableName in tables:\n"
        "        sqlP += f'SELECT PandaID FROM {tableName} WHERE jediTaskID=:t '\n"
        "    self.cur.execute(sqlP + comment, varMap)\n"
    )
    assert [re.findall(r"\.(\w+) WHERE", text) for text in texts] == [
        ["jobsDefined4", "jobsActive4"],
        ["jobsDefined4", "jobsActive4", "jobsArchived"],
    ]


_FORTY = ", ".join(f"'ATLAS_PANDA.t{index}'" for index in range(40))


@pytest.mark.parametrize(
    "src, texts",
    [
        pytest.param(
            "def f(self):\n"
            "    sqlJ = 'SELECT PandaID ' + "
            "'FROM {0}.{1} {{}} '.format(schemaPANDA, 'jobsActive4')\n"
            "    self.cur.execute(sqlJ + comment, varMap)\n",
            ["SELECT PandaID FROM {}.jobsActive4 {} "],
            id="an escaped brace is a brace",
        ),
        pytest.param(
            "def f(self):\n"
            "    sqlJ = 'SELECT PandaID ' + "
            "f'FROM {schemaPANDA}.{{1}} '.format('x', 'jobsActive4')\n"
            "    self.cur.execute(sqlJ + comment, varMap)\n",
            ["SELECT PandaID FROM {}.jobsActive4 "],
            id="a format over an f-string with a hole",
        ),
        pytest.param(
            "def f(self, fromDefined, archive):\n"
            "    tables = []\n"
            "    if fromDefined:\n"
            "        tables.append('ATLAS_PANDA.jobsDefined4')\n"
            "    tables.append(archive)\n"
            "    for table in tables:\n"
            "        sql = 'SELECT %s FROM %s ' % (columns, table) + 'WHERE PandaID=:PandaID'\n"
            "        self.cur.execute(sql + comment, varMap)\n",
            [
                "SELECT {} FROM ATLAS_PANDA.jobsDefined4 WHERE PandaID=:PandaID",
                "SELECT {} FROM {} WHERE PandaID=:PandaID",
            ],
            id="a list one unknown element is appended to keeps the known ones",
        ),
        pytest.param(
            "def f(self):\n"
            "    tables = [f'{schemaPANDA}.jobsActive4']\n"
            "    tables.append(pick_one())\n"
            "    sqlP = ''\n"
            "    for tableName in tables:\n"
            "        sqlP += f'SELECT PandaID FROM {tableName} WHERE x=:x '\n"
            "    self.cur.execute(sqlP + comment, varMap)\n",
            [
                "SELECT PandaID FROM {}.jobsActive4 WHERE x=:x SELECT PandaID FROM {} WHERE x=:x "
            ],
            id="a loop over a list of two runs twice",
        ),
        pytest.param(
            "def f(self):\n"
            "    schema = 'ATLAS_PANDA'\n"
            "    sqlT = f'SELECT jediTaskID FROM {schema}.JEDI_Tasks WHERE status=:status '\n"
            "    self.cur.execute(sqlT + comment, varMap)\n",
            ["SELECT jediTaskID FROM ATLAS_PANDA.JEDI_Tasks WHERE status=:status "],
            id="a schema the function names",
        ),
        pytest.param(
            "def f(self):\n"
            "    tableStatMap = {'jobsDefined4': ['defined']}\n"
            "    tableStatMap.clear()\n"
            "    for tableName in tableStatMap:\n"
            "        sqlJ = 'SELECT prodUserName FROM {0} j '.format(tableName)\n"
            "        self.cur.execute(sqlJ + comment, varMap)\n",
            [],
            id="a loop over a cleared dict runs nothing",
        ),
        pytest.param(
            "def f(self):\n"
            f"    for tableName in ({_FORTY}):\n"
            "        sqlP = f'SELECT PandaID FROM {tableName} WHERE x=:x '\n"
            "        self.cur.execute(sqlP + comment, varMap)\n",
            [
                f"SELECT PandaID FROM ATLAS_PANDA.t{index} WHERE x=:x "
                for index in range(40)
            ],
            id="forty tables",
        ),
        pytest.param(
            "def f(self):\n"
            "    sql = 'SELECT a FROM ATLAS_PANDA.t1 WHERE x IN ('\n"
            "    sql = f'{sql}:v) '\n"
            "    self.cur.execute(sql, varMap)\n",
            ["SELECT a FROM ATLAS_PANDA.t1 WHERE x IN (:v) "],
            id="a statement formatted into itself",
        ),
    ],
)
def test_the_statement_is_what_the_python_builds(src, texts):
    """``reassembly("python")``: where the reassembly leaves a hole or misreads.

    The reassembly leaves the brace, the list, the schema and the forty tables
    as holes, splits ``sql = f'{sql}:v) '`` in two, and reads a loop over a
    cleared dict as running.
    """
    assert _run_sql(src) == texts


SCOUT = """
def get_scout_data(task_id):
    returnMap = {}
    returnMap["ramCount"] = ram
    returnMap["walltime"] = wall
    returnMap["newNG"] = ng
    return True, returnMap

def set_scout_data(self, task_id, excluded):
    ok, scoutData = get_scout_data(task_id)
    sqlTSD = "UPDATE ATLAS_PANDA.JEDI_Tasks SET "
    for scoutKey, scoutVal in scoutData.items():
        if SKIP:
            continue
        sqlTSD += f"{scoutKey}=:{scoutKey},"
    sqlTSD = sqlTSD[:-1]
    sqlTSD += " WHERE jediTaskID=:jediTaskID "
    self.cur.execute(sqlTSD, varMap)
"""


@pytest.mark.parametrize(
    "skip, columns",
    [
        ('scoutKey in ["newNG"]', ["ramCount", "walltime"]),
        ("scoutKey in excluded", ["newNG", "ramCount", "walltime"]),
    ],
)
def test_the_columns_a_loop_over_a_dicts_keys_writes_are_their_union(skip, columns):
    """``reassembly("union")``: ``setScoutJobData_JEDI`` with a skip read and unread.

    A skip the evaluator decides leaves its key out; one it cannot leaves every
    key the callee may put in.  The reassembly names no column when it cannot
    read the skip.
    """
    tree = ast.parse(textwrap.dedent(SCOUT.replace("SKIP", skip)))
    defs = {n.name: n for n in tree.body}
    reading = sqleval.evaluate(defs["set_scout_data"], None, _resolver(defs))
    (texts,) = reading.texts.values()
    assert sorted(
        {c for text in texts for c in re.findall(r"(\w+)=:\1", text)}
    ) == sorted([*columns, "jediTaskID"])


@pytest.mark.parametrize(
    "src, name, texts",
    [
        pytest.param(
            "def f(self):\n"
            "    self.cur.execute("
            "'SELECT PandaID FROM ATLAS_PANDA.jobsActive4 WHERE jobStatus=:jobStatus', varMap)\n",
            "f",
            ["SELECT PandaID FROM ATLAS_PANDA.jobsActive4 WHERE jobStatus=:jobStatus"],
            id="with a varmap",
        ),
        pytest.param(
            "class TaskModule:\n"
            "    def inline(self):\n"
            '        self.cur.execute("SELECT LAST_INSERT_ID()", {})\n',
            "inline",
            ["SELECT LAST_INSERT_ID()"],
            id="with an empty one",
        ),
    ],
)
def test_a_statement_written_into_the_call_is_read(src, name, texts):
    """``reassembly("inline")``: the reassembly reads a statement only through a name."""
    assert _run_sql(src, name) == texts


def test_a_name_given_a_new_value_between_two_tests_is_read_as_every_piece_not_every_combination():
    """``reassembly("cover")``, a known cost: ``getTasksWithCriteria_JEDI``.

    Every way each ``if`` goes is in some statement, and a list among the task
    criteria stands beside a scalar among the dataset criteria in one -- but
    not every combination is a statement of its own: a single list beside a
    single scalar is not.  The reassembly lists all four.  The function's own
    tests sit in loops over the criteria, whose passes are one part each, both
    ways a pass goes among what it appends.
    """
    head = "SELECT jediTaskID FROM ATLAS_PANDA.JEDI_Tasks WHERE status=:status "
    texts = _run_sql(
        "def f(self, taskCriteria, datasetCriteria):\n"
        f"    sql = '{head}'\n"
        "    if isinstance(taskCriteria, list):\n"
        "        sql += 'AND a IN (:a) '\n"
        "    else:\n"
        "        sql += 'AND a=:a '\n"
        "    if isinstance(datasetCriteria, list):\n"
        "        sql += 'AND b IN (:b) '\n"
        "    else:\n"
        "        sql += 'AND b=:b '\n"
        "    self.cur.execute(sql, varMap)\n"
    )
    assert sorted(text.removeprefix(head) for text in texts) == [
        "AND a IN (:a) AND b IN (:b) ",
        "AND a=:a AND b=:b ",
    ]
    tree = ast.parse(
        "def f(self, taskCriteria, datasetCriteria):\n"
        f"    sql = '{head}'\n"
        "    for tmpKey, tmpVal in taskCriteria.items():\n"
        "        if isinstance(tmpVal, list):\n"
        "            sql += 'AND a IN (:a) '\n"
        "        else:\n"
        "            sql += 'AND a=:a '\n"
        "    for tmpKey, tmpVal in datasetCriteria.items():\n"
        "        if isinstance(tmpVal, list):\n"
        "            sql += 'AND b IN (:b) '\n"
        "        else:\n"
        "            sql += 'AND b=:b '\n"
        "    self.cur.execute(sql, varMap)\n"
    )
    pathcond.attach_parents(tree)
    with sql.evaluated():
        (run,) = sql.executions(tree.body[0])
    assert run.sql.removeprefix(head) == "{} {} "
    assert [hole.repeats for _at, hole in run.holes] == [
        ("AND a IN (:a) ", "AND a=:a "),
        ("AND b IN (:b) ", "AND b=:b "),
    ]


def test_a_loop_over_an_input_of_unknown_length_runs_at_least_once():
    """``reassembly("one_pass")``, a known cost: a loop over an input runs at least once.

    ``sql = sql[:-4]`` cuts the ``AND`` every pass ends with, and no
    statement is read for an empty input -- nor is the one a rebinding in the
    loop replaces.  The reassembly reads one pass, and the statement before
    the loop beside the one in it.
    """
    assert _run_sql(
        "def f(self, criteria, sites):\n"
        "    sql = 'SELECT status FROM t WHERE '\n"
        "    for key in criteria:\n"
        "        sql += f'{key}=:{key} AND '\n"
        "    sql += 'status IN (:a,:b) AND '\n"
        "    sql = sql[:-4]\n"
        "    self.cur.execute(sql, varMap)\n"
        "    sqlI = 'SELECT PandaID FROM j WHERE site IN ('\n"
        "    for site in sites:\n"
        "        sqlI += ':s,'\n"
        "    sqlI = sqlI[:-1] + ') '\n"
        "    self.cur.execute(sqlI, varMap)\n"
    ) == [
        "SELECT status FROM t WHERE {} AND status IN (:a,:b) ",
        "SELECT PandaID FROM j WHERE site IN ({}) ",
    ]
    assert _run_sql(
        "def f(self):\n"
        "    sql = 'SELECT a FROM ATLAS_PANDA.t1 '\n"
        "    for row in rows:\n"
        "        sql = 'SELECT b FROM ATLAS_PANDA.t2 '\n"
        "    self.cur.execute(sql, varMap)\n"
    ) == ["SELECT b FROM ATLAS_PANDA.t2 "]


def test_the_spans_of_a_name_are_the_ones_the_call_runs():
    """``reassembly("per_call")``: an UPSERT's two runs of ``sql``.

    The reassembly gives each call the spans of every statement the name held,
    so the ``UPDATE`` was asked with the ``INSERT``'s reading.
    """
    tree = ast.parse(
        "def claim(self, row):\n"
        "    if row is None:\n"
        '        sql = "INSERT INTO ATLAS_PANDA.jobsActive4 (PandaID,jobStatus) "\n'
        "        self.cur.execute(sql + comment, varMap)\n"
        "    else:\n"
        "        sql = \"UPDATE ATLAS_PANDA.jobsActive4 SET jobStatus='running' \"\n"
        "        self.cur.execute(sql + comment, varMap)\n"
    )
    pathcond.attach_parents(tree)
    func = tree.body[0]
    calls = [
        n
        for n in ast.walk(func)
        if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "execute"
    ]
    with sql.evaluated():
        spans = {
            call.lineno: [
                (text.split()[0], sorted({span.node.lineno for span in held}))
                for text, held in sql.variant_spans(func, "sql", at=call)
            ]
            for call in calls
        }
    assert spans == {4: [("INSERT", [3])], 7: [("UPDATE", [6])]}


def test_a_run_whose_where_names_nothing_still_reads_its_table_whichever_run_comes_first(
    tmp_path,
):
    """``get_job_statistics_per_site_label_resource``: four statements run from one call by a loop.

    The first has no ``WHERE``, so it files no pair; its table was filed only
    onto a record an earlier run of the line had made, and there was none.
    Read by the evaluator, the route through the call read ``jobsDefined4``
    and the statement did not, and the build stopped.
    """
    src = (
        "def stats(self):\n"
        "    sql_defined = 'SELECT site, COUNT(*) FROM ATLAS_PANDA.jobsDefined4 GROUP BY site '\n"
        "    sql_failed = 'SELECT site, COUNT(*) FROM ATLAS_PANDA.jobsActive4 "
        "WHERE jobStatus = :jobStatus GROUP BY site '\n"
        "    for sql_tmp, var_map in [(sql_defined, {}), (sql_failed, {':jobStatus': 'failed'})]:\n"
        "        self.cur.execute(sql_tmp + comment, var_map)\n"
    )
    with sql.evaluated():
        (statement,) = _read_side(tmp_path, src).values()
    assert sorted(statement.tables) == ["jobsActive4", "jobsDefined4"]


def _corpus_executions(tmp_path, text: str, name: str) -> list:
    """``sql.executions`` of *name* in ``pkg/db.py`` with the evaluator on, through the module's own calls."""
    from bamboo.codemap.gitsource import blob_sha
    from bamboo.codemap.models import SourceModule
    from bamboo.codemap.panda.recognizers import sqlwrite

    root = tmp_path / "pkg"
    root.mkdir(parents=True, exist_ok=True)
    (root / "__init__.py").write_text("")
    (root / "db.py").write_text(text)
    tree = ast.parse(text)
    pathcond.attach_parents(tree)
    module = SourceModule(
        package="pkg",
        rel_path="pkg/db.py",
        tree=tree,
        source=text,
        blob_sha=blob_sha(text),
    )
    calls = sqlwrite.corpus_calls([module], {"pkg": root})
    func = next(
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name
    )
    with sql.across_calls(calls), sql.evaluated():
        return sql.executions(func)


def test_a_fragment_a_callee_appends_stays_a_hole_in_the_statement(tmp_path):
    """``JobSpec.columnNames()``: the callee joins column names it cannot read with ``ret += attr``.

    The ``+=`` a function appends to its own statement with no text of its own
    is left out and said once (``sqlP += sql_where_clause``); one inside the
    callee is a piece of the value the call returns, so it stays ``{}``.  Left
    out, the column list read as ``SELECT  FROM`` and ``SELECT , FROM``.
    """
    src = (
        "class Spec:\n"
        "    _attributes = attribute_names()\n"
        "\n"
        "    @classmethod\n"
        "    def columnNames(cls):\n"
        "        ret = ''\n"
        "        for attr in cls._attributes:\n"
        "            if ret != '':\n"
        "                ret += ','\n"
        "            ret += attr\n"
        "        return ret\n"
        "\n"
        "\n"
        "def get(cur, panda_id):\n"
        "    sql = 'SELECT ' + Spec.columnNames() + ' FROM jobsActive4 WHERE PandaID=:p'\n"
        "    cur.execute(sql, {':p': panda_id})\n"
    )
    runs = _corpus_executions(tmp_path, src, "get")
    assert runs
    for run in runs:
        assert run.sql.startswith("SELECT {}"), run.sql
        assert not run.unresolved, run.unresolved


def test_nothing_is_in_an_empty_dict(tmp_path):
    """``JediFileSpec.columnNames()``: ``defaultVales`` is ``{}`` when the call passes none.

    ``attr in defaultVales`` is False whatever ``attr`` is.  Left open, the
    column list read the default-value arm -- ``SELECT '{}''{}' FROM`` -- which
    PanDA never sends and which does not parse.  ``{}{}`` stays: an attribute
    named ``""`` would leave ``ret`` empty and the comma out.
    """
    src = (
        "class Spec:\n"
        "    _attributes = attribute_names()\n"
        "    _seqAttrMap = sequence_names()\n"
        "\n"
        "    @classmethod\n"
        "    def columnNames(cls, useSeq=False, defaultVales=None, skipDefaultAttr=False):\n"
        "        if defaultVales is None:\n"
        "            defaultVales = {}\n"
        "        ret = ''\n"
        "        for attr in cls._attributes:\n"
        "            if skipDefaultAttr and (attr in cls._seqAttrMap or attr in defaultVales):\n"
        "                continue\n"
        "            if ret != '':\n"
        "                ret += ','\n"
        "            if useSeq and attr in cls._seqAttrMap:\n"
        "                ret += f'{cls._seqAttrMap[attr]}'\n"
        "                continue\n"
        "            if attr in defaultVales:\n"
        "                arg = defaultVales[attr]\n"
        "                if arg is None:\n"
        "                    ret += 'NULL'\n"
        "                elif isinstance(arg, str):\n"
        "                    ret += f\"'{arg}'\"\n"
        "                else:\n"
        "                    ret += f'{arg}'\n"
        "                continue\n"
        "            ret += attr\n"
        "        return ret\n"
        "\n"
        "\n"
        "def get(cur, task_id):\n"
        "    sql = f'SELECT {Spec.columnNames()} FROM JEDI_Dataset_Contents WHERE jediTaskID=:t'\n"
        "    cur.execute(sql, {':t': task_id})\n"
    )
    texts = sorted({run.sql for run in _corpus_executions(tmp_path, src, "get")})
    assert texts == [
        "SELECT {} FROM JEDI_Dataset_Contents WHERE jediTaskID=:t",
        "SELECT {},{} FROM JEDI_Dataset_Contents WHERE jediTaskID=:t",
        "SELECT {}{} FROM JEDI_Dataset_Contents WHERE jediTaskID=:t",
    ]


@pytest.mark.xfail(
    strict=True,
    reason=(
        "the cover counts a branch shown by its dimension, not by its choice: keyed by choice it shows this piece, "
        "and in 61 call lines draws callee loop samples with nothing chosen (SET  WHERE); undecided (X12 S3)"
    ),
)
def test_a_branch_under_each_arm_of_another_test_shows_up_in_the_cover():
    """``getActiveDebugJobs``: ``if prodRole / elif workingGroup is not None / else``, then a loop over two tables.

    The ``elif`` arm's piece sits under one arm of the first test, the
    ``else`` arm's under the other, both chosen by the same second test.  The
    cover counted that test's branch shown once it rendered under either arm,
    and the ``UPPER(workingGroup)`` piece was in no statement.
    """
    src = """
def f(self, dn=None, workingGroup=None, prodRole=False):
    varMap = {}
    sqlX = "SELECT PandaID FROM %s "
    sqlX += "WHERE "
    if prodRole:
        pass
    elif workingGroup is not None:
        sqlX += "UPPER(workingGroup) IN (:wg1,:wg2) AND "
    else:
        sqlX += "prodUserName=:prodUserName AND "
    sqlX += "specialHandling IS NOT NULL "
    for table in ["jobsDefined4", "jobsActive4"]:
        sql = sqlX % table
        self.cur.execute(sql, varMap)
"""
    texts = _run_sql(src)
    for piece in (
        "UPPER(workingGroup) IN (:wg1,:wg2) AND",
        "prodUserName=:prodUserName AND",
    ):
        assert any(piece in text for text in texts), (piece, texts)
    for table in ("jobsDefined4", "jobsActive4"):
        assert any(table in text for text in texts), (table, texts)


def test_a_text_holding_a_written_letter_is_not_empty():
    """``getPandaIDsWithTask_JEDI``: ``if sqlP != "": sqlP += "UNION ALL "`` over a list of tables.

    ``sqlP`` holds the schema as a hole, and written letters around it, so it
    is not ``""`` whatever the hole is.  Left open, the second ``SELECT``
    followed the first with no ``UNION ALL`` between them, a statement PanDA
    never sends.  A text that is only a hole stays open: it can be ``""``.
    """
    src = """
def f(self, task_id):
    tables = [f"{schema}.jobsDefined4", f"{schema}.jobsActive4"]
    sqlP = ""
    for tableName in tables:
        if sqlP != "":
            sqlP += "UNION ALL "
        sqlP += f"SELECT PandaID FROM {tableName} WHERE jediTaskID=:jediTaskID "
    self.cur.execute(sqlP, {":jediTaskID": task_id})
"""
    texts = _run_sql(src)
    assert texts
    for text in texts:
        assert text.count("SELECT") == text.count("UNION ALL") + 1, text
