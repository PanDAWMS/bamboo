"""Reading a statement's terms off its sqlglot tree (``sqltext.terms``).

The text readers filed every compared column against every table a statement
named, under every verb, subqueries included.  These pin what the tree reading
says instead, on the shapes the corpus census found that wrong.
"""

from __future__ import annotations

import ast

from bamboo.codemap.gitsource import blob_sha
from bamboo.codemap.models import SourceModule
from bamboo.codemap.panda import attribution, sql, sqltext
from bamboo.codemap.panda.recognizers import progress, sqlwrite

_SCHEMA = {
    "jedi_tasks": frozenset({"jeditaskid", "status", "vo", "prodsourcelabel"}),
    "jedi_datasets": frozenset({"jeditaskid", "datasetid", "status", "type"}),
    "jedi_aux_status_mintaskid": frozenset({"status", "min_jeditaskid"}),
}


def _said(sql: str, schema=_SCHEMA):
    with sqltext.with_schema(schema):
        reading = sqltext.terms(sql)
    return (
        [
            (t.table, t.column, t.verb, t.scope, t.excluding, t.binds, t.literals, t.undecided)
            for t in reading.terms
        ],
        reading.unresolved,
    )


def test_an_aliased_column_is_the_aliased_tables():
    """``tabD.status<>:st`` excludes dataset statuses and says nothing of a task's.

    ``get_task_ids_with_dataset_attributes`` -- the text reading filed the
    exclusion against ``JEDI_Tasks`` and ``JEDI_AUX_Status_MinTaskID`` as well.
    """
    terms, unresolved = _said(
        "SELECT tabT.jediTaskID FROM {0}.JEDI_Tasks tabT, {0}.JEDI_Datasets tabD, "
        "{0}.JEDI_AUX_Status_MinTaskID tabA WHERE tabT.status=tabA.status "
        "AND tabT.jediTaskID=tabD.jediTaskID AND tabD.status<>:st AND tabT.vo=:vo"
    )

    assert terms == [
        ("JEDI_Datasets", "status", "read", "row", True, (":st",), (), False),
        ("JEDI_Tasks", "vo", "read", "row", False, (":vo",), (), False),
    ]
    assert unresolved == ()


def test_an_unqualified_column_of_a_join_is_the_table_the_schema_gives_it():
    """Valid SQL names a bare column only one of its tables has; the schema says which.

    Without the schema the column is reported rather than filed on both --
    which is what the text reading did, and how ``type IN ('input', ...)``
    became a task's column.
    """
    sql = (
        "SELECT tabT.jediTaskID FROM {0}.JEDI_Tasks tabT, {0}.JEDI_Datasets tabD "
        "WHERE tabT.jediTaskID=tabD.jediTaskID AND vo=:vo AND type IN ('input','pseudo_input')"
    )

    terms, unresolved = _said(sql)
    assert terms == [
        ("JEDI_Tasks", "vo", "read", "row", False, (":vo",), (), False),
        ("JEDI_Datasets", "type", "read", "row", False, (), ("input", "pseudo_input"), False),
    ]
    assert unresolved == ()

    terms, unresolved = _said(sql, schema=None)
    assert terms == []
    assert unresolved == (
        f"type (JEDI_Datasets, JEDI_Tasks) -- {sqltext.NO_SCHEMA}",
        f"vo (JEDI_Datasets, JEDI_Tasks) -- {sqltext.NO_SCHEMA}",
    )

    both = {**_SCHEMA, "jedi_tasks": _SCHEMA["jedi_tasks"] | {"type"}}
    terms, unresolved = _said(sql, schema=both)
    assert unresolved == (f"type (JEDI_Datasets, JEDI_Tasks) -- {sqltext.OWNER_UNDECIDED}",)


def test_a_subquerys_term_is_not_the_statements_own():
    """``NOT EXISTS (... status=:pending)`` is a value the rows must not have company in.

    And ``EXISTS (... jobStatus=:st_merging)`` asks whether merging jobs exist,
    returning none of them; a column ``IN (SELECT ...)`` takes no value from
    the subquery's binds.
    """
    terms, _unresolved = _said(
        "SELECT tabT.jediTaskID FROM {}.JEDI_Tasks tabT WHERE tabT.status=:s "
        "AND NOT EXISTS (SELECT 1 FROM {}.JEDI_Datasets WHERE "
        "{}.JEDI_Datasets.jediTaskID=tabT.jediTaskID AND status=:pending)"
    )
    assert terms == [
        ("JEDI_Datasets", "status", "read", "negated", False, (":pending",), (), False),
        ("JEDI_Tasks", "status", "read", "row", False, (":s",), (), False),
    ]

    terms, _unresolved = _said(
        "SELECT 1 FROM dual WHERE EXISTS (SELECT 1 FROM {}.jobsActive4 WHERE jobStatus=:st_merging)"
    )
    assert terms == [("jobsActive4", "jobStatus", "read", "consult", False, (":st_merging",), (), False)]

    terms, _unresolved = _said(
        "SELECT PandaID FROM jobsActive4 WHERE jediTaskID IN "
        "(SELECT jediTaskID FROM JEDI_Datasets WHERE datasetID=:ds)"
    )
    assert terms == [
        ("JEDI_Datasets", "datasetID", "read", "consult", False, (":ds",), (), False),
        ("jobsActive4", "jediTaskID", "read", "row", False, (), (), True),
    ]


def test_a_subquery_in_a_values_list_is_consulted():
    """``INSERT ... VALUES (:t, (SELECT datasetID FROM ... WHERE type=:type))`` looks a row up.

    sqlglot builds no scope below a ``VALUES`` list, so the subquery read as
    nothing at all (``insertHpoEventAboutIdds_JEDI``).
    """
    terms, _unresolved = _said(
        "INSERT INTO {}.JEDI_Events (jediTaskID,datasetID) VALUES(:jediTaskID,"
        "(SELECT datasetID FROM {}.JEDI_Datasets WHERE jediTaskID=:jediTaskID AND type=:type))"
    )

    assert terms == [
        ("JEDI_Datasets", "jediTaskID", "read", "consult", False, (":jediTaskID",), (), False),
        ("JEDI_Datasets", "type", "read", "consult", False, (":type",), (), False),
    ]


def test_an_update_picks_its_own_rows_and_a_negated_term_excludes():
    """An ``UPDATE``'s ``WHERE`` picks the rows it updates -- sqlglot gives it no scope of its own.

    ``AND NOT jobStatus=:ngStatus`` is ``updateJobStatus``'s guard: rows
    without the value, which the text reading did not read at all.
    """
    terms, _unresolved = _said(
        "UPDATE ATLAS_PANDA.jobsActive4 SET jobStatus=:jobStatus "
        "WHERE PandaID=:PandaID AND NOT jobStatus=:ngStatus"
    )

    assert terms == [
        ("jobsActive4", "PandaID", "updated", "row", False, (":PandaID",), (), False),
        ("jobsActive4", "jobStatus", "updated", "row", True, (":ngStatus",), (), False),
    ]


def test_a_hole_is_a_name_and_a_part_the_grammar_needs_but_never_a_cut():
    """Holes read as names; two read otherwise; a statement cut short is not forgiven.

    ``getJobs`` orders by ``CASE {} ELSE {} END`` -- the ``WHEN`` arms joined
    in Python -- and ``do_preassign`` has three ``.format`` fields where
    predicates go.  A statement ending in ``AND`` does not parse: PanDA cuts
    that ``AND`` off before running it, and reading on would hide that the
    reading did not.
    """
    terms, unresolved = _said(
        "SELECT * FROM (SELECT PandaID FROM ATLAS_PANDA.jobsActive4 tab WHERE jobStatus=:old "
        "ORDER BY (CASE {} ELSE {} END), currentpriority desc) WHERE ROWNUM <= :njobs"
    )
    assert terms == [("jobsActive4", "jobStatus", "read", "row", False, (":old",), (), False)]

    terms, unresolved = _said(
        "SELECT t.jediTaskID FROM {}.JEDI_Tasks t WHERE t.vo=:vo "
        "AND EXISTS (SELECT 1 FROM {}.JEDI_Datasets d WHERE d.jediTaskID=t.jediTaskID) {} {} {} "
        "AND t.status IN ('ready')"
    )
    assert [term[:2] for term in terms] == [("JEDI_Tasks", "vo"), ("JEDI_Tasks", "status")]

    terms, unresolved = _said("SELECT status FROM {}.JEDI_Tasks WHERE jediTaskID=:t AND")
    assert terms == []
    # Said where it stopped, not where it starts: a head clipped short put
    # the cut inside an optimizer hint and made the hint look to blame.
    assert unresolved == (
        f"SELECT status FROM {{}}.JEDI_Tasks WHERE jediTaskID=:t AND -- {sqltext.NOT_PARSED}",
    )
    terms, unresolved = _said(
        "SELECT /*+ index(tab FILESTABLE4_DESTDBLOCK_IDX) */ COUNT(*) FROM ATLAS_PANDA.filesTable4 tab AND {}=:{}"
    )
    assert unresolved == (f"COUNT(*) FROM ATLAS_PANDA.filesTable4 tab AND -- {sqltext.NOT_PARSED}",)


def test_a_json_path_and_a_pseudo_column_are_not_columns():
    """``sc.data.jobseed`` is a key in ``schedconfig_json.data``; ``ROWNUM`` limits the rows."""
    terms, unresolved = _said(
        "SELECT ps.panda_site_name FROM ATLAS_PANDA.panda_site ps, ATLAS_PANDA.schedconfig_json sc "
        "WHERE sc.panda_queue=ps.panda_site_name AND sc.data.jobseed<>'es' "
        "AND ps.site=:site AND rownum<=10"
    )

    assert terms == [("panda_site", "site", "read", "row", False, (":site",), (), False)]
    assert unresolved == (f"sc.data.jobseed -- {sqltext.JSON_PATH}",)


def test_a_name_spelled_through_a_hole_is_not_a_column_or_a_bind():
    """``getNumUnprocessedEvents_JEDI`` adds ``tabT.{0}=:{0}`` per criterion.

    Read as a column called ``__h2``, that came out as a key ``JEDI_Tasks``
    has no column for.  A bind named through a hole has no value to look up,
    so its column is asked on a value the statement does not decide.
    """
    terms, unresolved = _said(
        "SELECT COUNT(1) FROM {}.JEDI_Tasks tabT WHERE tabT.vo=:vo "
        "AND tabT.{}=:{} AND tabT.{}<>:neg_{} AND tabT.status=:{}"
    )

    assert terms == [
        ("JEDI_Tasks", "vo", "read", "row", False, (":vo",), (), False),
        ("JEDI_Tasks", "status", "read", "row", False, (), (), True),
    ]
    assert unresolved == ()


def test_the_schema_is_read_by_paren_depth(tmp_path):
    """A column's type has parentheses of its own, and a body ends at the paren that opened it."""
    (tmp_path / "ATLAS_PANDA.sql").write_text(
        '  CREATE TABLE "ATLAS_PANDA"."JOBSARCHIVED4" \n'
        '   (\t"PANDAID" NUMBER(11,0) DEFAULT \'0\', \n'
        "\t\"CREATIONTIME\" DATE DEFAULT to_date('01-JAN-1970 00:00:00', 'DD-MON-YYYY'), \n"
        '\t"PRODSOURCELABEL" VARCHAR2(20 BYTE) DEFAULT \'managed\'\n'
        '   ) PARTITION BY RANGE ("MODIFICATIONTIME") (PARTITION "P1" VALUES LESS THAN (1));\n'
        '  CREATE TABLE "ATLAS_PANDA"."JEDI_TASKS" ("JEDITASKID" NUMBER(11,0), '
        '"STATUS" VARCHAR2(64), CONSTRAINT "JEDI_TASKS_PK" PRIMARY KEY ("JEDITASKID"));\n'
    )

    assert sqltext.read_schema(tmp_path) == {
        "jobsarchived4": frozenset({"pandaid", "creationtime", "prodsourcelabel"}),
        "jedi_tasks": frozenset({"jeditaskid", "status"}),
    }


_SPECS = """
class JediTaskSpec(object):
    _attributes = ('jediTaskID', 'status', 'vo', 'oldStatus')


class JediDatasetSpec(object):
    _attributes = ('jediTaskID', 'datasetID', 'status', 'type', 'nFiles')
"""


def _module(source: str, rel: str) -> SourceModule:
    return SourceModule(
        package=rel.split("/")[0],
        rel_path=rel,
        tree=ast.parse(source),
        source=source,
        blob_sha=blob_sha(source),
    )


def _read_side(source: str, schema=_SCHEMA):
    modules = [_module(_SPECS, "pandaserver/taskbuffer/Specs.py"), _module(source, "x.py")]
    attributor = attribution.SpecAttributor(
        progress.spec_attributes(modules), attribution.class_bases(modules)
    )
    attributor.learn_table_classes(modules)
    with sqltext.with_schema(schema):
        return sqlwrite.read_side(modules, attributor)


# Writes that teach each table its class -- ``learn_table_classes`` reads writes.
_TEACH = (
    "    def teach(self):\n"
    "        sqlU = 'UPDATE ATLAS_PANDA.JEDI_Tasks SET status=:status,oldStatus=NULL,vo=:vo WHERE jediTaskID=:t'\n"
    "        self.cur.execute(sqlU, {':status': 'ready', ':vo': 'atlas', ':t': 1})\n"
    "        sqlD = 'UPDATE ATLAS_PANDA.JEDI_Datasets SET status=:s,nFiles=:n,type=:ty WHERE datasetID=:d'\n"
    "        self.cur.execute(sqlD, {':s': 'ready', ':n': 0, ':ty': 'input', ':d': 1})\n"
)


def test_a_reader_is_filed_on_the_table_its_column_belongs_to():
    """The dataset status a join excludes is no reader, or excluder, of task statuses."""
    side = _read_side(
        "class TaskModule:\n"
        "    def find(self):\n"
        "        sql = 'SELECT tabT.jediTaskID FROM ATLAS_PANDA.JEDI_Tasks tabT, ATLAS_PANDA.JEDI_Datasets tabD '\n"
        "        sql += \"WHERE tabT.jediTaskID=tabD.jediTaskID AND tabD.status<>'done' AND status=:ts \"\n"
        "        self.cur.execute(sql, {':ts': 'running'})\n" + _TEACH
    )

    (statement,) = [s for (owner, _line), s in side.statements.items() if owner == "x.py::find"]
    assert statement.excludes == {("JediDatasetSpec.status", "done")}
    assert statement.selects == set()
    # ``status`` bare in a join both tables declare it in: not decided, not filed.
    assert statement.unresolved == (
        f"status (JEDI_Datasets, JEDI_Tasks) -- {sqltext.OWNER_UNDECIDED}",
    )


def test_a_subquerys_values_are_consulted_not_selected():
    """``EXISTS (... status=:st)`` checks rows by the value; ``selected_by`` does not get it."""
    side = _read_side(
        "class TaskModule:\n"
        "    def throttle(self):\n"
        "        sql = 'SELECT 1 FROM dual WHERE EXISTS (SELECT 1 FROM ATLAS_PANDA.JEDI_Datasets '\n"
        "        sql += 'WHERE jediTaskID=:t AND status=:st)'\n"
        "        self.cur.execute(sql, {':t': 1, ':st': 'merging'})\n" + _TEACH
    )

    (statement,) = [s for (owner, _line), s in side.statements.items() if owner == "x.py::throttle"]
    assert statement.consults == {
        ("JediDatasetSpec.jediTaskID", "1"),
        ("JediDatasetSpec.status", "merging"),
    }
    assert statement.selects == set()
    assert "merging" not in side.values.get("JediDatasetSpec.status", {})
    (node,) = [
        n
        for n in sqlwrite.read_site_nodes(side.statements, "panda", "v")
        if n.owner == "x.py::throttle"
    ]
    assert node.consults == ["JediDatasetSpec.jediTaskID=1", "JediDatasetSpec.status=merging"]


def test_the_subquery_an_update_picks_its_own_rows_by_is_not_a_consult():
    """``getEventRanges``: the subquery reads the rows the ``UPDATE`` changes -- no reader to send anyone to.

    The statement gets no read site, since nothing it asks lands as a value;
    its ``conditions`` would keep the term whole.
    """
    picking = (
        "UPDATE ATLAS_PANDA.JEDI_Events tab SET PandaID=:pandaID,status=:newEventStatus "
        "WHERE (jediTaskID,PandaID,fileID) IN (SELECT jediTaskID,PandaID,fileID FROM ("
        "SELECT jediTaskID,PandaID,fileID FROM ATLAS_PANDA.JEDI_Events tab "
        "WHERE jediTaskID=:jediTaskID AND status=:eventStatus ORDER BY fileID) WHERE rownum<=3) "
    )
    side = _read_side(
        "class TaskModule:\n"
        "    def reserve(self):\n"
        f"        sql = {picking!r}\n"
        "        self.cur.execute(sql, {':pandaID': 1, ':newEventStatus': 'sent', ':jediTaskID': 2, ':eventStatus': 'ready'})\n"
        + _TEACH
    )

    assert {term.scope for term in sqltext.terms(picking).terms} == {sqltext.PICKS}
    assert [owner for owner, _line in side.statements if owner == "x.py::reserve"] == []
    (condition,) = sqltext.conditions(picking)
    (branch,) = condition.branches
    assert branch.terms[0].startswith("(jediTaskID,PandaID,fileID) IN (SELECT")
    assert branch.binds == (":jediTaskID", ":eventStatus")
    # Another table's rows are a consult, as before.
    other = "UPDATE ATLAS_PANDA.jobsActive4 SET a=1 WHERE PandaID IN (SELECT PandaID FROM ATLAS_PANDA.jobsDefined4 WHERE jobStatus=:s)"
    assert [term.scope for term in sqltext.terms(other).terms if term.column == "jobStatus"] == [sqltext.CONSULT]


def test_a_statement_that_does_not_parse_is_kept_and_says_so():
    """Not read some other way, and not dropped: the tables are known, the terms are not."""
    side = _read_side(
        "class TaskModule:\n"
        "    def broken(self):\n"
        "        sql = 'SELECT status FROM ATLAS_PANDA.JEDI_Tasks WHERE jediTaskID=:t AND ((('\n"
        "        self.cur.execute(sql, {':t': 1})\n" + _TEACH
    )

    (statement,) = [s for (owner, _line), s in side.statements.items() if owner == "x.py::broken"]
    assert statement.selects == set() and statement.keys == set()
    assert [stop.rsplit(" -- ", 1)[1] for stop in statement.unresolved] == [sqltext.NOT_PARSED]


def _required(sql: str) -> list[list[tuple[str, ...]]]:
    with sqltext.with_schema(_SCHEMA):
        return [[branch.terms for branch in condition.branches] for condition in sqltext.conditions(sql)]


def test_a_column_compared_with_one_bind_is_not_a_condition_outside_an_or():
    """``status<>:s`` is ``excludes``, and one fact is one row.

    Inside an ``OR`` the term stays: there it says the statement accepts that
    value *or* something else.  ``IN`` and a quoted value stay as the text
    reading kept them.  ``EXISTS`` is ``consults`` and ``ROWNUM`` is
    ``limits``; ``NOT EXISTS`` asks the subquery turned round, and only the
    term as written says that.
    """
    required = _required(
        "SELECT t.jediTaskID FROM {}.JEDI_Tasks t WHERE t.status<>:s0 AND t.status IN (:s1,:s2) "
        "AND t.vo='atlas' AND (t.status=:s3 OR t.lockedBy IS NULL) "
        "AND NOT EXISTS (SELECT 1 FROM {}.JEDI_Datasets d WHERE d.jediTaskID=t.jediTaskID AND d.status=:ds) "
        "AND EXISTS (SELECT 1 FROM {}.JEDI_Datasets d WHERE d.jediTaskID=t.jediTaskID AND d.type='input') "
        "AND rownum<=:n"
    )

    assert required == [
        [("t.status IN (:s1,:s2)",)],
        [("t.vo='atlas'",)],
        [("t.status=:s3",), ("t.lockedBy IS NULL",)],
        [("NOT EXISTS (SELECT 1 FROM {}.JEDI_Datasets d WHERE d.jediTaskID=t.jediTaskID AND d.status=:ds)",)],
    ]


def test_a_term_the_text_readers_passed_over_is_a_condition():
    """``BETWEEN``, ``LIKE``, arithmetic and functions are requirements too.

    The ``AND`` of a ``BETWEEN`` does not join two terms.  A comparison of two
    columns under one alias is the row's own; under two it is a join.  A
    column named through a hole, and a term naming no column, ask nothing
    the map can say.
    """
    required = _required(
        "SELECT tabD.datasetID FROM {}.JEDI_Datasets tabD, {}.JEDI_Tasks tabT "
        "WHERE tabD.jediTaskID=tabT.jediTaskID AND tabD.nFiles BETWEEN :lo AND :hi "
        "AND tabD.datasetName LIKE :pattern AND (tabD.nFiles-tabD.nFilesUsed)>0 "
        "AND NVL(tabD.nFilesOnHold,0)=0 AND tabD.nFilesUsed<tabD.nFiles AND tabD.{}=:{} AND 1=1"
    )

    assert required == [
        [("tabD.nFiles BETWEEN :lo AND :hi",)],
        [("tabD.datasetName LIKE :pattern",)],
        [("(tabD.nFiles-tabD.nFilesUsed)>0",)],
        [("NVL(tabD.nFilesOnHold,0)=0",)],
        [("tabD.nFilesUsed<tabD.nFiles",)],
    ]


def test_the_rows_a_derived_table_hands_out_met_its_where():
    """``getJobs`` picks its rows inside, and keeps ``ROWNUM <= :njobs`` of them outside.

    A bind named like a keyword is still a bind: ``:limit`` is not ``LIMIT``.
    """
    required = _required(
        "SELECT * FROM (SELECT PandaID FROM ATLAS_PANDA.jobsActive4 tab WHERE jobStatus=:old "
        "AND tab.modificationTime<:limit ORDER BY (CASE {} ELSE {} END), currentpriority desc) "
        "WHERE ROWNUM <= :njobs"
    )

    assert required == [[("tab.modificationTime<:limit",)]]


def test_the_branches_of_a_union_are_alternatives():
    """A row from either table met that branch's ``WHERE``, not both.

    Read as one clause, the first branch's age limit was a requirement on
    every job the statement returns.
    """
    required = _required(
        "SELECT PandaID FROM ATLAS_PANDA.jobsDefined4 WHERE jediTaskID=:t "
        "AND modificationTime>(CURRENT_DATE-30) "
        "UNION ALL SELECT PandaID FROM ATLAS_PANDA.jobsActive4 WHERE jediTaskID=:t "
        "AND jobStatus<>:a AND lockedBy IS NULL"
    )

    assert required == [[("modificationTime>(CURRENT_DATE-30)",), ("lockedBy IS NULL",)]]


def test_only_the_rows_that_come_back_are_ordered_and_limited():
    """A subquery's ``ROWNUM`` and an ``OVER (ORDER BY ...)`` say nothing about the rows returned.

    The text reading took every ``ORDER BY`` and ``ROWNUM`` in the statement.
    """
    orders, limits = sqltext.ordering(
        "SELECT * FROM (SELECT PandaID, ROW_NUMBER() OVER (PARTITION BY computingSite ORDER BY modificationTime) rn "
        "FROM ATLAS_PANDA.jobsActive4 tab WHERE jobStatus=:s AND NOT EXISTS "
        "(SELECT 1 FROM ATLAS_PANDA.jobsDefined4 d WHERE d.PandaID=tab.PandaID AND ROWNUM<=1) "
        "ORDER BY currentPriority DESC) WHERE ROWNUM <= :limit AND rn < {}"
    )

    assert orders == ["currentPriority DESC"]
    assert limits == [(":limit", False)]


def test_rownum_equal_to_one_is_a_limit_of_one():
    """``ppEventServiceJob`` asks ``rownum=1``; read as no limit, it was in the map nowhere."""
    assert sqltext.ordering("SELECT PandaID FROM ATLAS_PANDA.jobsActive4 WHERE jediTaskID=:t AND rownum=1") == (
        [],
        [("1", False)],
    )


def test_an_order_with_no_column_in_it_is_still_read():
    """``ORDER BY DBMS_RANDOM.value`` comes back from sqlglot as a ``RAND()`` with no position."""
    orders, _limits = sqltext.ordering(
        "SELECT {} FROM ( SELECT {} FROM {}.JEDI_Dataset_Contents WHERE jediTaskID=:jediTaskID "
        "ORDER BY DBMS_RANDOM.value) t"
    )

    assert orders == ["DBMS_RANDOM.value"]


def test_an_exclusion_the_map_cannot_name_is_still_filed():
    """``status NOT IN (SELECT ...)`` and ``status<>:s`` with ``:s`` from the caller pass over values.

    Filed as ``{}``: a reader asking whether its value is one of them has to
    be told the statement might pass it over, not that it asks nothing.
    """
    side = _read_side(
        "class TaskModule:\n"
        "    def pick(self, given):\n"
        "        sql = ('SELECT jediTaskID FROM ATLAS_PANDA.JEDI_Tasks WHERE vo=:vo AND status<>:given '\n"
        "               'AND status NOT IN (SELECT status FROM ATLAS_PANDA.JEDI_Datasets WHERE type=:ty)')\n"
        "        self.cur.execute(sql, {':vo': 'atlas', ':given': given, ':ty': 'input'})\n" + _TEACH
    )

    (statement,) = [s for (owner, _line), s in side.statements.items() if owner == "x.py::pick"]
    assert statement.excludes == {("JediTaskSpec.status", "{}")}


def _refused(sql: str) -> list[list[tuple[str, str, tuple[str, ...], tuple[str, ...]]]]:
    with sqltext.with_schema(_SCHEMA):
        return [
            [(t.table, t.column, t.binds, t.literals) for t in alternative]
            for alternative in sqltext.refusals(sql)
        ]


def test_only_an_exclusion_every_row_meets_is_a_refusal():
    """``getFailureCountsForTask_JEDI`` names ``relocationFlag<>:r`` in one branch of an ``OR``.

    Its other branches return those rows, so the term is an exclusion the
    statement names and not one it makes of every row.  Each branch of a
    ``UNION`` is a way a row comes back, and is kept apart for the caller.
    """
    assert _refused(
        "SELECT COUNT(*) FROM JEDI_Tasks WHERE jediTaskID=:t AND "
        "((status=:f AND vo<>'x') OR (status=:c AND prodSourceLabel<>:r) OR status=:fin)"
    ) == [[]]
    assert _refused(
        "SELECT jediTaskID FROM JEDI_Tasks WHERE NOT status IN (:a) AND (vo<>'x') "
        "AND NOT (prodSourceLabel=:p AND jediTaskID=:t)"
    ) == [[("JEDI_Tasks", "status", (":a",), ()), ("JEDI_Tasks", "vo", (), ("x",))]]
    assert _refused(
        "SELECT COUNT(*) FROM (SELECT jediTaskID FROM JEDI_Tasks WHERE status NOT IN (:a,:b) "
        "UNION ALL SELECT jediTaskID FROM JEDI_Datasets WHERE status<>:a) t"
    ) == [
        [("JEDI_Tasks", "status", (":a", ":b"), ())],
        [("JEDI_Datasets", "status", (":a",), ())],
    ]


def _decided(sql: str) -> set[tuple[str, bool, tuple[str, ...], bool]]:
    with sqltext.with_schema(_SCHEMA):
        return {(t.column, t.excluding, t.binds + t.literals, t.decided) for t in sqltext.terms(sql).terms}


def test_a_term_decides_its_column_where_every_branch_of_its_or_rules_the_row_out():
    """``copyArchive``'s ``main`` asks ``jobStatus`` in both branches of an ``OR``; ``getJobs`` asks ``processingType`` in one.

    The first rules out every row with another status; the second rules out
    no row at all, since one passes on the other branch's column.  Testing
    the column in every branch is not enough: ``status<>:a OR status<>:b``
    returns the rows holding ``a`` on its second branch, and ``status='a' OR
    status IS NOT NULL`` every row with a status.
    """
    assert _decided(
        "SELECT jediTaskID FROM JEDI_Tasks WHERE (vo=:v AND status=:a) OR (prodSourceLabel=:p AND status=:b)"
    ) == {
        ("vo", False, (":v",), False),
        ("prodSourceLabel", False, (":p",), False),
        ("status", False, (":a",), True),
        ("status", False, (":b",), True),
    }
    assert _decided("SELECT jediTaskID FROM JEDI_Tasks WHERE status=:s AND (vo=:v OR prodSourceLabel=:p)") == {
        ("status", False, (":s",), True),
        ("vo", False, (":v",), False),
        ("prodSourceLabel", False, (":p",), False),
    }
    assert _decided("SELECT jediTaskID FROM JEDI_Tasks WHERE status<>:a OR status<>:b") == {
        ("status", True, (":a",), False),
        ("status", True, (":b",), False),
    }
    assert _decided("SELECT jediTaskID FROM JEDI_Tasks WHERE status='a' OR status IS NOT NULL") == {
        ("status", False, ("a",), False),
    }


def test_an_or_every_branch_of_which_leaves_the_value_out_is_a_refusal():
    """``relocationFlag IS NULL OR relocationFlag<>:r`` was the one ``OR`` read as refusing; it is one of a kind.

    A null holds no value, and a branch asking only for other quoted values
    holds none of this one, so either branch rules the row out as the
    ``<>`` does -- with or without other terms beside it.
    """
    assert _refused(
        "SELECT jediTaskID FROM JEDI_Tasks WHERE (status IS NULL OR (status<>:a AND vo=:v)) "
        "AND (prodSourceLabel IN ('x','y') OR prodSourceLabel<>'c')"
    ) == [[("JEDI_Tasks", "status", (":a",), ()), ("JEDI_Tasks", "prodSourceLabel", (), ("c",))]]


def test_a_subject_every_row_holds_one_value_of_is_restricted():
    """``selects`` names ``vo`` here; ``restricts`` does not, as a row with another ``vo`` returns on ``oldStatus``.

    Each branch of the ``UNION`` asks ``status`` outright, for a value each,
    so every row holds one of the two.  The second statement asks it for a
    value the caller hands, and a row holding that one is no row the map
    could name.
    """
    side = _read_side(
        "class TaskModule:\n"
        "    def pick(self, given):\n"
        "        sql = ('SELECT jediTaskID FROM ATLAS_PANDA.JEDI_Tasks WHERE status=:st AND (vo=:vo OR oldStatus=:o) '\n"
        "               'UNION SELECT jediTaskID FROM ATLAS_PANDA.JEDI_Tasks WHERE status IN (:a) AND vo=:vo')\n"
        "        self.cur.execute(sql, {':st': 'ready', ':vo': 'atlas', ':o': 'done', ':a': 'running'})\n"
        "        sqlG = 'SELECT jediTaskID FROM ATLAS_PANDA.JEDI_Tasks WHERE status IN (:a,:b) AND vo=:vo'\n"
        "        self.cur.execute(sqlG, {':a': 'running', ':b': given, ':vo': 'atlas'})\n" + _TEACH
    )

    first, second = (s for (owner, _line), s in sorted(side.statements.items()) if owner == "x.py::pick")
    assert ("JediTaskSpec.vo", "atlas") in first.selects
    assert first.restricts == {("JediTaskSpec.status", "ready"), ("JediTaskSpec.status", "running")}
    assert second.restricts == {("JediTaskSpec.vo", "atlas")}
    assert "JediTaskSpec.status" in second.keys
    assert _refused(
        "SELECT jediTaskID FROM JEDI_Tasks WHERE vo=:vo AND NOT EXISTS "
        "(SELECT 1 FROM JEDI_Datasets d WHERE d.jediTaskID=JEDI_Tasks.jediTaskID AND d.status<>:s)"
    ) == [[]]
    assert _refused("SELECT jediTaskID FROM JEDI_Tasks WHERE status NOT IN (") == []


def test_a_value_left_out_of_rows_that_may_be_null_is_still_a_refusal():
    """``(relocationFlag IS NULL OR relocationFlag<>:f)`` -- a row holding ``f`` fails both branches."""
    assert _refused("SELECT jediTaskID FROM JEDI_Tasks t WHERE (t.status IS NULL OR t.status NOT IN (:a,:b))") == [
        [("JEDI_Tasks", "status", (":a", ":b"), ())]
    ]
    # Another column, a test that rows are *not* null, or a second exclusion:
    # each lets a row holding the value through one branch.
    assert _refused("SELECT jediTaskID FROM JEDI_Tasks WHERE (status IS NULL OR vo<>:x)") == [[]]
    assert _refused("SELECT jediTaskID FROM JEDI_Tasks WHERE (status IS NOT NULL OR status<>:x)") == [[]]
    assert _refused("SELECT jediTaskID FROM JEDI_Tasks WHERE (status IS NULL OR status<>:x OR status<>:y)") == [[]]


def test_a_statement_refuses_a_value_only_where_every_run_passes_it_over():
    """Built two ways, bound two ways: what is left out of every row on every run.

    ``done`` is appended under an ``if`` with no ``else``, ``broken`` sits
    inside an ``OR``, and ``:st`` is ``running`` on one path and ``dummy`` on
    the other -- all named, none passed over every time.  Where both branches
    of an ``if``/``else`` leave ``done`` out, every run does.
    """
    side = _read_side(
        "class TaskModule:\n"
        "    def pick(self, flag):\n"
        "        sql = 'SELECT jediTaskID FROM ATLAS_PANDA.JEDI_Tasks WHERE vo=:vo '\n"
        "        if flag:\n"
        "            sql += \"AND status<>'done' \"\n"
        "        sql += \"AND status<>'failed' AND (status<>:a OR prodSourceLabel=:p)\"\n"
        "        self.cur.execute(sql, {':vo': 'atlas', ':a': 'broken', ':p': 'user'})\n"
        "    def kill(self, hard):\n"
        "        varMap = {':vo': 'atlas'}\n"
        "        if hard:\n"
        "            varMap[':st'] = 'running'\n"
        "        else:\n"
        "            varMap[':st'] = 'dummy'\n"
        "        sql = 'SELECT jediTaskID FROM ATLAS_PANDA.JEDI_Tasks WHERE vo=:vo AND status<>:st'\n"
        "        self.cur.execute(sql, varMap)\n"
        "    def choose(self, user):\n"
        "        sql = 'SELECT jediTaskID FROM ATLAS_PANDA.JEDI_Tasks WHERE vo=:vo '\n"
        "        if user:\n"
        "            sql += \"AND status<>'done' AND prodSourceLabel='user' \"\n"
        "        else:\n"
        "            sql += \"AND status<>'done' \"\n"
        "        self.cur.execute(sql, {':vo': 'atlas'})\n" + _TEACH
    )

    (picked,) = [s for (owner, _line), s in side.statements.items() if owner == "x.py::pick"]
    (killed,) = [s for (owner, _line), s in side.statements.items() if owner == "x.py::kill"]
    status = "JediTaskSpec.status"
    assert picked.excludes == {(status, "done"), (status, "failed"), (status, "broken")}
    assert picked.refuses == {(status, "failed")}
    assert killed.excludes == {(status, "running"), (status, "dummy")}
    assert killed.refuses == frozenset()
    (chosen,) = [s for (owner, _line), s in side.statements.items() if owner == "x.py::choose"]
    assert chosen.refuses == {(status, "done")}


_FEED_AND_SITES = (
    "class TaskModule:\n"
    "    def feed(self, force_read):\n"
    "        sql = 'SELECT jediTaskID FROM ATLAS_PANDA.JEDI_Tasks WHERE vo=:vo AND oldStatus IS NULL '\n"
    "        if not force_read:\n"
    "            sql += 'AND lockedBy IS NULL '\n"
    "        self.cur.execute(sql, {':vo': 'atlas'})\n"
    "    def sites(self, sites):\n"
    "        sql = 'SELECT jediTaskID FROM ATLAS_PANDA.JEDI_Tasks WHERE vo=:vo AND oldStatus IS NULL AND site IN ('\n"
    "        var_map = {':vo': 'atlas'}\n"
    "        for idx, site in enumerate(sites):\n"
    "            var_map[f':s_{idx}'] = site\n"
    "            sql += f':s_{idx},'\n"
    "        sql = sql[:-1] + ') '\n"
    "        self.cur.execute(sql, var_map)\n" + _TEACH
)


def test_conditions_drop_what_some_paths_append():
    """``if not force_read:`` appends a test some runs never ask."""
    side = _read_side(_FEED_AND_SITES)

    (fed,) = [s for (owner, _line), s in side.statements.items() if owner == "x.py::feed"]
    assert fed.conditions == ["oldStatus IS NULL"]


def test_conditions_read_a_list_a_loop_fills_as_one_requirement():
    """Run, what the loop appends is one part however many sites there are -- ``IN ({})`` -- and every run asks it:
    ``site`` is one of the list's.  Not a disjunction of lengths, and no ``IN ( )``."""
    side = _read_side(_FEED_AND_SITES)

    (sited,) = [s for (owner, _line), s in side.statements.items() if owner == "x.py::sites"]
    assert sited.conditions == ["oldStatus IS NULL", "site IN ({})"]


def test_a_stated_in_list_refuses_every_member_and_a_handed_one_none():
    """``NOT IN ({names})`` of a list the function states passes over all of it on every run.

    ``get_task_ids_with_dataset_attributes``' ten task statuses are one
    placeholder.  A list a caller hands is no run's in particular.
    """
    side = _read_side(
        "class TaskModule:\n"
        "    def active(self):\n"
        "        var_map = {':vo': 'atlas'}\n"
        "        names, listed = get_sql_IN_bind_variables(['done', 'failed'], prefix=':st_')\n"
        "        var_map.update(listed)\n"
        "        sql = f'SELECT jediTaskID FROM ATLAS_PANDA.JEDI_Tasks WHERE vo=:vo AND status NOT IN ({names})'\n"
        "        self.cur.execute(sql, var_map)\n"
        "    def handed(self, statuses):\n"
        "        var_map = {':vo': 'atlas'}\n"
        "        names, listed = get_sql_IN_bind_variables(statuses, prefix=':st_')\n"
        "        var_map.update(listed)\n"
        "        sql = f'SELECT jediTaskID FROM ATLAS_PANDA.JEDI_Tasks WHERE vo=:vo AND status NOT IN ({names})'\n"
        "        self.cur.execute(sql, var_map)\n"
        "    def run(self):\n"
        "        self.handed(['done'])\n"
        "        self.handed(['broken'])\n" + _TEACH
    )

    (active,) = [s for (owner, _line), s in side.statements.items() if owner == "x.py::active"]
    status = "JediTaskSpec.status"
    assert active.refuses == {(status, "done"), (status, "failed")}
    handed = [s for (owner, _line), s in side.statements.items() if owner == "x.py::handed"]
    assert all(statement.refuses == frozenset() for statement in handed)


def test_a_table_and_its_columns_are_read_off_the_tree():
    """Each shape the text reading got wrong on the corpus, read as the statement says.

    A hint hid the column after it and the ``INTO`` after ``INSERT``; ``,FROM``
    hid the table; an outer projection over a subquery was filed under the
    subquery's table; a function's argument was taken for a column the row
    carries.
    """
    assert sqltext.reads("SELECT /*+ INDEX(tab IDX) */ PandaID,status FROM ATLAS_PANDA.filesTable4 tab") == [
        ("filesTable4", ["PandaID", "status"])
    ]
    assert sqltext.reads("SELECT t.jediTaskID, COALESCE(t.currentPriority, t.taskPriority) FROM {}.JEDI_Tasks t") == [
        ("JEDI_Tasks", ["jediTaskID"])
    ]
    assert sqltext.reads(
        "SELECT jediTaskID,t_status FROM (SELECT tabT.jediTaskID,tabT.status AS t_status FROM {}.JEDI_Tasks tabT)"
    ) == [("JEDI_Tasks", ["jediTaskID"])]
    assert sqltext.reads("SELECT tabT.jediTaskID,tabD.type,FROM {}.JEDI_Tasks tabT,{}.JEDI_Datasets tabD") == [
        ("JEDI_Tasks", ["jediTaskID", "type"])
    ]
    assert sqltext.creates(
        "INSERT /*+ ignore_row_on_dupkey_index (e(region)) */ INTO ATLAS_PANDA.CARBON_REGION_EMISSIONS e "
        "(REGION, VALUE) VALUES(:region, :value)"
    ) == ["CARBON_REGION_EMISSIONS"]
    assert sqltext.updates("SELECT status FROM {}.JEDI_Tasks WHERE jediTaskID=:t FOR UPDATE NOWAIT") == []


def test_a_join_partner_is_read_through_the_alias_it_is_joined_by():
    statement = (
        "DELETE FROM ATLAS_PANDA.Task_Evaluation te WHERE te.jediTaskID IN (SELECT jt.jediTaskID "
        "FROM ATLAS_PANDA.JEDI_Tasks jt JOIN ATLAS_PANDA.Task_Evaluation tez ON tez.jediTaskID=jt.jediTaskID "
        "WHERE te.metric=:m AND jt.status IN ('done'))"
    )
    assert sqltext.joins(statement) == ["Task_Evaluation"]
    assert sqltext.joined_columns(statement, "Task_Evaluation") == ["jediTaskID"]
    assert sqltext.deletes(statement) == ["Task_Evaluation"]


def test_a_statement_that_does_not_parse_still_names_its_tables():
    """``getJobs``'s query and ``updateDataset_JEDI``'s update do not parse; their tables are still theirs."""
    assert sqltext.reads("SELECT {} FROM ATLAS_PANDA.Datasets AND {}=:{}") == [("Datasets", [])]
    assert sqltext.updates("UPDATE {}.JEDI_Datasets SET {} WHERE AND {}={}") == ["JEDI_Datasets"]


def test_every_table_a_statement_names_is_listed_in_the_order_it_names_them():
    """A proxy step said ``reads jobsDefined4, NOWAIT``, and named one table of a join."""
    assert sqltext.tables_named(
        "WITH w AS (SELECT PandaID FROM {}.filesTable4 WHERE d=:d) "
        "SELECT j.PandaID FROM {}.jobsActive4 j, w, dual WHERE j.PandaID=w.PandaID"
    ) == ["filesTable4", "jobsActive4"]
    assert sqltext.tables_named("SELECT PandaID FROM ATLAS_PANDA.jobsDefined4 WHERE PandaID=:p FOR UPDATE NOWAIT") == [
        "jobsDefined4"
    ]
    assert sqltext.tables_named("SELECT a FROM {}.JEDI_Tasks t, {}.JEDI_Datasets d WHERE t.x=d.x") == [
        "JEDI_Tasks", "JEDI_Datasets"
    ]
    # A fragment of an f-string, read off its text.
    assert sqltext.tables_named("INSERT INTO ATLAS_PANDA.filesTable4 ({}) VALUES ") == ["filesTable4"]


def _written(sql: str) -> list[tuple[str, str, dict, dict, dict]]:
    return [
        (w.kind, w.table, {c: (v.kind, v.text) for c, v in w.columns.items()}, w.preconditions, w.keys)
        for w in sqltext.writes(sql)
    ]


def test_what_a_statement_writes_is_read_off_the_tree_in_its_own_spelling():
    """A comma inside a function, nested ``VALUES``, ``INSERT ... SELECT`` and a hinted ``INSERT``."""
    assert _written("UPDATE {}.JEDI_Tasks SET nFiles=NVL(nFiles, 0)+1, status=:status WHERE jediTaskID=:t") == [
        (
            "update",
            "JEDI_Tasks",
            {"nFiles": ("expression", "NVL(nFiles, 0)+1"), "status": ("bind", ":status")},
            {},
            {"jediTaskID": ":t"},
        )
    ]
    assert _written("INSERT INTO {}.JEDI_Events (jediTaskID,fileID,status) VALUES (:t,(SELECT MAX(f) FROM x),'ready')") == [
        (
            "insert",
            "JEDI_Events",
            {"jediTaskID": ("bind", ":t"), "fileID": ("expression", "(SELECT MAX(f) FROM x)"), "status": ("literal", "ready")},
            {},
            {},
        )
    ]
    assert _written(
        "INSERT INTO {}.JEDI_Events (jediTaskID,status,attemptNr) SELECT jediTaskID,:newStatus,attemptNr-1 FROM {}.JEDI_Events"
    ) == [
        (
            "insert",
            "JEDI_Events",
            {
                "jediTaskID": ("expression", "jediTaskID"),
                "status": ("bind", ":newStatus"),
                "attemptNr": ("expression", "attemptNr-1"),
            },
            {},
            {},
        )
    ]
    assert _written("INSERT /*+ ignore_row_on_dupkey_index (e(r)) */ INTO ATLAS_PANDA.CARBON_REGION_EMISSIONS e (REGION) VALUES(:r)") == [
        ("insert", "CARBON_REGION_EMISSIONS", {"REGION": ("bind", ":r")}, {}, {})
    ]
    # A column Python names at run time is no column; the row its WHERE names is still read.
    assert _written("UPDATE {}.JEDI_Tasks SET merge{}={} WHERE jediTaskID=:jediTaskID") == [
        ("update", "JEDI_Tasks", {}, {}, {"jediTaskID": ":jediTaskID"})
    ]
    # A spec's bindUpdateChangesExpression(): the columns are Python's, the row is the WHERE's.
    assert _written(
        "UPDATE ATLAS_PANDA.jobsDefined4 SET {} WHERE PandaID=:PandaID AND (jobStatus=:oldJobStatus1 OR jobStatus=:oldJobStatus2)"
    ) == [("update", "jobsDefined4", {}, {}, {"PandaID": ":PandaID"})]
    # With no SET item at all (a build in which no attribute changed) the statement still writes no column, said.
    assert sqltext.unread_columns("UPDATE ATLAS_PANDA.filesTable4 SET WHERE row_ID=:row_ID") == [
        ("filesTable4", sqltext.COLUMNS_NOT_READ)
    ]


def test_a_write_is_guarded_only_by_what_every_row_it_changes_meets():
    """The lock is the whole ``OR``; a subquery's and an ``OR``'s terms name no row."""
    ((_kind, _table, _columns, preconditions, keys),) = _written(
        "UPDATE {}.JEDI_Tasks SET lockedBy=:lockedBy,lockedTime=CURRENT_DATE,"
        "workqueue_id=(SELECT queue_id FROM {}.JEDI_Work_Queue WHERE queue_name=:queueName) "
        "WHERE jediTaskID=:jediTaskID AND (lockedBy IS NULL OR lockedTime<:timeLimit) AND NOT status=:ng "
        "AND (PandaID=:p OR jobsetID=:p)"
    )
    assert preconditions == {
        "lockedBy": "(lockedBy IS NULL OR lockedTime<:timeLimit)",
        "lockedTime": "(lockedBy IS NULL OR lockedTime<:timeLimit)",
    }
    assert keys == {"jediTaskID": ":jediTaskID"}


def test_where_a_value_is_written_is_an_offset_into_the_text_as_assembled():
    """The variant keeps its holes, and the fragment a value came from is found by the offset."""
    statement = "UPDATE {}.JEDI_Tasks SET status=:status, oldStatus=NULL WHERE jediTaskID=:t"
    end = sqltext.supplied_at(statement, "update", "oldStatus", sqltext.ColumnValue(kind="literal", text="None"))
    assert statement[:end].endswith("oldStatus=NULL")
    insert = "INSERT INTO {}.JEDI_Tasks (jediTaskID,status) VALUES (:t,'ready')"
    end = sqltext.supplied_at(insert, "insert", "status", sqltext.ColumnValue(kind="literal", text="ready"))
    assert insert[:end].endswith("'ready'")
    assert sqltext.supplied_at(insert, "update", "status", sqltext.ColumnValue(kind="literal", text="ready")) is None


def test_every_test_of_a_written_column_is_its_precondition():
    """``(ramCount IS NOT NULL AND ramCount>:ramCount)`` is two conjuncts on the column the update sets."""
    ((_kind, _table, _columns, preconditions, _keys),) = _written(
        "UPDATE {}.JEDI_Dataset_Contents SET ramCount=:ramCount WHERE jediTaskID=:jediTaskID "
        "AND (ramCount IS NOT NULL AND ramCount>:ramCount)"
    )
    assert preconditions == {"ramCount": "ramCount IS NOT NULL AND ramCount>:ramCount"}


def test_a_precondition_says_where_each_of_its_conjuncts_is():
    """Offsets into the text read, holes and all, so a caller holding the pieces it was built from can say which wrote a guard."""
    text = (
        "UPDATE {}.JEDI_Dataset_Contents SET ramCount=:ramCount,status=:s WHERE jediTaskID=:jediTaskID "
        "AND ramCount IS NOT NULL AND status=:old AND ramCount>:ramCount"
    )
    (statement,) = sqltext.writes(text)

    assert {
        column: [text[start:end] for start, end in spans] for column, spans in statement.precondition_spans.items()
    } == {"ramCount": ["ramCount IS NOT NULL", "ramCount>:ramCount"], "status": ["status=:old"]}


_O, _C = sqltext.REPEAT_OPEN, sqltext.REPEAT_CLOSE


def test_a_repeated_part_is_read_as_its_passes_and_said_of_the_statement():
    """``updateJobStatus`` appends ``,{key}=:{key}`` per parameter, and ``,memory_leak=:memory_leak`` on the pass for
    the job's metrics.  Read as the statement holds it -- ``jobStatus=:jobStatus{}`` -- the hole glued itself to
    the bind, and the column the pass writes was not there.  Read with the passes in its place, each column takes
    the value its pass wrote, and every offset is one into the statement's own text: a value the passes wrote is
    at the ``{}``."""
    marked = (
        f"UPDATE ATLAS_PANDA.jobsActive4 SET jobStatus=:jobStatus{_O},{{}}=:{{}},memory_leak=:memory_leak{_C}"
        ",startTime=CURRENT_DATE WHERE PandaID=:PandaID AND NOT jobStatus=:ngStatus"
    )
    text = sqltext.plain(marked)
    assert text == (
        "UPDATE ATLAS_PANDA.jobsActive4 SET jobStatus=:jobStatus{},startTime=CURRENT_DATE "
        "WHERE PandaID=:PandaID AND NOT jobStatus=:ngStatus"
    )
    assert [write.columns["jobStatus"].text for write in sqltext.writes(text)] == [":jobStatus{}"]

    (statement,) = sqltext.writes(marked)

    assert {column: value.text for column, value in statement.columns.items()} == {
        "jobStatus": ":jobStatus",
        "memory_leak": ":memory_leak",
        "startTime": "CURRENT_DATE",
    }
    assert {
        column: [text[start:end] for start, end in spans] for column, spans in statement.precondition_spans.items()
    } == {"jobStatus": ["NOT jobStatus=:ngStatus"]}
    leak = sqltext.supplied_at(marked, "update", "memory_leak", statement.columns["memory_leak"])
    assert text[:leak].endswith(":jobStatus{}")
    started = sqltext.supplied_at(marked, "update", "startTime", statement.columns["startTime"])
    assert text[:started].endswith("startTime=CURRENT_DATE")


def test_a_term_only_a_loops_passes_append_is_required_of_no_row():
    """A loop may give no pass that appends a term -- no key, or none its test lets through -- so a term only the
    passes append holds on no row the loop left alone: not a condition, not a guard, and it decides nothing.  It is
    still a column the statement asks by, on the passes that append it."""
    selected = f"SELECT a FROM T WHERE id=:id {_O}AND status='done'{_C}"
    updated = f"UPDATE T SET status=:new WHERE id=:id {_O}AND status=:old{_C}"

    assert sqltext.conditions(selected) == ()
    assert [branch.terms for one in sqltext.conjoined(selected) for branch in one.branches] == [("id=:id",)]
    assert [(term.column, term.literals, term.decided) for term in sqltext.terms(selected).terms] == [
        ("id", (), True),
        ("status", ("done",), False),
    ]
    assert [[term.column for term in way] for way in sqltext.outright(selected)] == [["id"]]
    (statement,) = sqltext.writes(updated)
    assert statement.preconditions == {}
    # The same statement, every row of it asked for the status, guards the write.
    (unrepeated,) = sqltext.writes(sqltext.unmarked(updated))
    assert unrepeated.preconditions == {"status": "status=:old"}


def test_where_each_term_is_in_the_statement_is_read_off_the_statement_the_evaluator_hands():
    """A term placed by where it is, not by which fragment's text names it: ``jobStatus IN ('activated', 'sent',
    'starting')`` is a template's words and a list a branch writes.  The stretch is the statement's own -- the text
    the fragments index -- past holes the parse names or reads as empty, and a repeated part's ``{}``."""
    statement = (
        f"SELECT {{}} FROM {{}}.jobsActive4 WHERE jobStatus IN ('activated', 'sent', 'starting') AND x=:x {{}} {{}} "
        f"AND (y=:y OR z IN ({{}})) AND NOT EXISTS (SELECT 1 FROM {{}}.u WHERE u.id=jobsActive4.id AND u.k IN (:k1,:k2)) "
        f"ORDER BY (CASE {_O}WHEN gshare=:leave{{}} THEN 0 {_C} ELSE {{}} END), b"
    )
    plain = sqltext.plain(statement)

    def said(spans) -> list:
        return [[plain[a:b] for a, b in place] for place in spans]

    assert [said(at) for at in sqltext.condition_spans(statement)] == [
        [["jobStatus IN ('activated', 'sent', 'starting')"]],
        [["y=:y", "z IN ({})"]],
        [["NOT EXISTS (SELECT 1 FROM {}.u WHERE u.id=jobsActive4.id AND u.k IN (:k1,:k2))"]],
    ]
    assert len(sqltext.condition_spans(statement)) == len(sqltext.conditions(statement))
    assert [said(at) for at in sqltext.ordering_spans(statement)] == [[["(CASE {} ELSE {} END), b"]]]
    assert {
        f"{term.table}.{term.column}": plain[a:b]
        for term, (a, b) in zip(sqltext.terms(statement).terms, sqltext.term_spans(statement), strict=True)
    } == {
        "jobsActive4.jobStatus": "jobStatus IN ('activated', 'sent', 'starting')",
        "jobsActive4.x": "x=:x",
        "jobsActive4.y": "y=:y",
        "jobsActive4.z": "z IN ({})",
        "u.k": "u.k IN (:k1,:k2)",
    }


def test_a_repeated_part_is_spelled_as_the_statement_holds_it():
    """``undo_preassign``'s ``IN ({})`` and ``getJobs``' ``(CASE {} ELSE {} END)``: read with a pass in place, the
    test knows its column and the ranking its share column, and both are spelled as the statement holds them."""
    listed = f"SELECT t.jediTaskID FROM {{}}.JEDI_Tasks t WHERE t.jediTaskID IN ({_O}:pt_{{}}{_C}) AND t.site IS NOT NULL"
    ranked = (
        f"SELECT a FROM jobsActive4 WHERE x=:x ORDER BY (CASE {_O}WHEN gshare=:leave{{}} THEN 0 "
        f"WHEN gshare=:leave{{}} THEN {{}}{_C} ELSE {{}} END), b"
    )

    assert [branch.terms for one in sqltext.conditions(listed) for branch in one.branches] == [
        ("t.jediTaskID IN ({})",),
        ("t.site IS NOT NULL",),
    ]
    tests = [
        (test.test.kind, test.test.column, [side.text for side in test.test.sides])
        for twin in sqltext.condition_twins(listed)
        for branch in twin.items
        for test in branch.items
    ]
    assert tests == [("equal", "jediTaskID", ["{}"]), ("not_null", "site", [])]
    # Read as the statement holds it, the hole is a column the list is compared with.
    assert [
        (test.test.kind, test.test.column)
        for twin in sqltext.condition_twins(sqltext.plain(listed))
        for branch in twin.items
        for test in branch.items
    ][0] == ("other", "")
    assert sqltext.ordering(ranked) == (["(CASE {} ELSE {} END), b"], [])
    assert ("jobsActive4", "gshare") in sqltext.order_owners(ranked)
    assert sqltext.where_clauses(listed) == [" t.jediTaskID IN ({}) AND t.site IS NOT NULL"]


_PRINTED_GETJOBS = (
    "SELECT * FROM (SELECT /*+ INDEX_RS_ASC(tab (PRODSOURCELABEL COMPUTINGSITE JOBSTATUS) ) */ PandaID,currentPriority "
    "FROM ATLAS_PANDA.jobsActive4 tab WHERE computingSite=:computingSite AND jobStatus=:oldJobStatus "
    "AND (processingType=:processingType1 OR prodSourceLabel IN (:p1,:p2)) "
    "AND NOT EXISTS (SELECT 1 FROM ATLAS_PANDA.JEDI_Tasks t WHERE t.jediTaskID=tab.jediTaskID AND t.status=:ts) "
    "\n          ORDER BY (CASE WHEN gshare=:leave0 THEN 0 WHEN gshare=:leave1 THEN 1 ELSE 2 END), currentpriority desc, pandaid asc)"
    "\n          WHERE ROWNUM <= :njobs\n /* DBProxy.getJobs */"
)


def test_a_printed_statement_is_read_query_by_query():
    """``getJobs`` prints the statement it runs: a share ranking built at run time, around the query that filters.

    The outer query reads no table and keeps the first rows; the inner one
    reads ``jobsActive4`` and is where the row is asked for, term by term as
    the map cuts a ``WHERE``.  A subquery's columns are its own, its binds
    the printed map's all the same.  The comment the call adds is read past.
    """
    read = sqltext.printed(_PRINTED_GETJOBS)

    assert read.failure == ""
    assert read.limits == ((":njobs", False),)
    outer, inner = read.queries
    assert outer.tables == () and [term.text for term in outer.where] == ["ROWNUM <= :njobs"]
    assert inner.tables == ("jobsActive4",)
    assert inner.spelled == ("ATLAS_PANDA.jobsActive4",)
    assert inner.where == (
        sqltext.PrintedTerm("computingSite=:computingSite", ("computingSite",), (":computingSite",)),
        sqltext.PrintedTerm("jobStatus=:oldJobStatus", ("jobStatus",), (":oldJobStatus",)),
        sqltext.PrintedTerm(
            "(processingType=:processingType1 OR prodSourceLabel IN (:p1,:p2))",
            ("processingType", "prodSourceLabel"),
            (":processingType1", ":p1", ":p2"),
        ),
        sqltext.PrintedTerm(
            "NOT EXISTS (SELECT 1 FROM ATLAS_PANDA.JEDI_Tasks t WHERE t.jediTaskID=tab.jediTaskID AND t.status=:ts)",
            (),
            (":ts",),
        ),
    )
    assert [(item.columns, item.descending) for item in inner.order] == [
        (("gshare",), False),
        (("currentpriority",), True),
        (("pandaid",), False),
    ]


def test_an_order_list_is_read_item_by_item_with_what_the_map_blanks_said():
    """The map spells ``getJobs``' ranking ``(CASE {} ELSE {} END)``: built at run time, it is not in the source.

    Read the same way, the map's list and the printed one are three items
    each, and only the first is blanked -- whatever a run prints there is it.
    """
    (printed,) = [query.order for query in sqltext.printed(_PRINTED_GETJOBS).queries if query.order]
    mapped = sqltext.order_items("(CASE {} ELSE {} END), currentpriority desc, pandaid asc")

    assert [item.blanked for item in mapped] == [True, False, False]
    assert mapped[0].columns == ()
    assert [(one.expression, one.descending) for one in mapped[1:]] == [
        (one.expression, one.descending) for one in printed[1:]
    ]
    assert not printed[0].blanked


def test_a_where_whose_or_binds_looser_than_its_and_is_one_conjunct():
    """``get_workers_to_synchronize`` writes ``(A) OR (B) AND lastupdate > ... AND (...)``.

    ``AND`` binds tighter, so that is one term -- an ``OR`` whose second
    branch carries the time limits -- and cutting the text at its ``AND``\\ s
    gave four, which the tree does not have.
    """
    read = sqltext.printed(
        "SELECT harvesterID FROM ATLAS_PANDA.harvester_workers "
        "WHERE (status IN ('submitted') AND pilotStatus='running') "
        "OR (status IN ('idle') AND pilotEndTime < :d) AND lastupdate > :x AND (sync > :r OR sync IS NULL)"
    )

    assert read.failure == ""
    ((only,),) = [query.where for query in read.queries]
    assert only.text.startswith("(status IN ('submitted')") and only.text.endswith("sync IS NULL)")
    assert only.binds == (":d", ":x", ":r")


def test_a_printed_term_is_filled_with_its_values_and_nothing_inside_quotes():
    """A query drawn from a printed record is run by hand, with the run's values in it.

    A bind looked for inside a quoted value is not one: ``HH24:MI:SS`` holds
    no ``:MI``.  A value with no one SQL spelling stays a bind.
    """
    text = "x=:x AND d > TO_DATE(:t,'YYYY-MM-DD HH24:MI:SS') AND n IN (:a,:b) AND s=:s AND f=:f"
    values = {":x": 1, ":a": "it's", ":b": None, "s": "done", ":f": True}

    assert sqltext.filled(text, values) == (
        "x=1 AND d > TO_DATE(:t,'YYYY-MM-DD HH24:MI:SS') AND n IN ('it''s',NULL) AND s='done' AND f=:f"
    )


def _shape(twin) -> object:
    """A twin as nested tuples: ``(node, items...)`` or ``(kind, table.column, op, ((side kind, side text), ...))``."""
    if twin.node != "test":
        return (twin.node, *(_shape(one) for one in twin.items))
    test = twin.test
    return (test.kind, f"{test.table}.{test.column}", test.op, tuple((side.kind, side.text) for side in test.sides))


def test_a_condition_is_a_tree_of_branches_and_terms_read_off_its_own_clause():
    """``copyArchive``'s ``(modificationTime<:t AND jobStatus=:a) OR (...)`` and a null-safe time limit.

    An entry is an ``any`` of its branches and each an ``all`` of its terms,
    with the column the term tests named by table, never by alias, and put on
    the left where the statement writes ``:p < currentPriority``.
    """
    sql = (
        "SELECT PandaID FROM ATLAS_PANDA.jobsActive4 tab WHERE "
        "(tab.modificationTime<:t AND tab.jobStatus=:a) OR (tab.stateChangeTime<:t AND tab.jobStatus=:b) "
        "AND (tab.lockedTime IS NULL OR tab.lockedTime<:lock) AND :p < tab.currentPriority "
        "AND tab.creationTime>CURRENT_DATE-3 AND tab.cloud NOT IN (:c1,:c2)"
    )
    entries = sqltext.conditions(sql)
    twins = sqltext.condition_twins(sql)
    assert len(twins) == len(entries)
    shapes = [_shape(twin) for twin in twins]
    assert ("any", ("all", ("compare", "jobsActive4.modificationTime", "<", (("bind", ":t"),)), ("equal", "jobsActive4.jobStatus", "=", (("bind", ":a"),))),
            ("all", ("compare", "jobsActive4.stateChangeTime", "<", (("bind", ":t"),)), ("equal", "jobsActive4.jobStatus", "=", (("bind", ":b"),)),
             # An ``OR`` inside a branch nests, its terms as they are.
             ("any", ("null", "jobsActive4.lockedTime", "", ()), ("compare", "jobsActive4.lockedTime", "<", (("bind", ":lock"),))),
             ("compare", "jobsActive4.currentPriority", ">", (("bind", ":p"),)),
             ("compare", "jobsActive4.creationTime", ">", (("clock", "CURRENT_DATE - 3"),)),
             ("unequal", "jobsActive4.cloud", "NOT IN", (("bind", ":c1"), ("bind", ":c2"))))) in shapes
    clock = next(
        side for twin in twins for one in [twin] for side in _sides(one) if side.kind == "clock"
    )
    assert (clock.clock, clock.offset_seconds) == ("CURRENT_DATE", -259200.0)


def _sides(twin):
    if twin.node == "test":
        yield from twin.test.sides
    for one in twin.items:
        yield from _sides(one)


def test_what_two_branches_of_a_union_require_between_them_folds_into_one_tree():
    """``A AND B UNION A AND C`` requires ``A``, and ``B`` or ``C``: the second entry is an ``any`` of each branch's rest."""
    sql = (
        "SELECT PandaID FROM ATLAS_PANDA.jobsActive4 WHERE modificationTime<:t AND creationTime<:c "
        "UNION SELECT PandaID FROM ATLAS_PANDA.jobsActive4 WHERE modificationTime<:t AND stateChangeTime<:s"
    )
    twins = sqltext.condition_twins(sql)
    assert len(twins) == len(sqltext.conditions(sql)) == 2
    folded = _shape(twins[1])
    assert folded[0] == "any" and len(folded) == 3
    assert [branch[0] for branch in folded[1:]] == ["all", "all"]


def test_a_logged_statement_and_its_map_are_cut_apart_without_losing_a_time():
    """``str(varMap)`` prints a time as ``datetime.datetime(...)``: read as that time, and said to be left unfilled."""
    import datetime

    message = (
        "SELECT PandaID FROM ATLAS_PANDA.jobsActive4 WHERE modificationTime<:timeLimit AND jobStatus=:s "
        "/* DBProxy.x */{':timeLimit': datetime.datetime(2026, 10, 4, 18, 38, 7), ':s': 'activated', "
        "':d': Decimal('1.5')}"
    )
    text, binds = sqltext.cut(message)
    assert text.endswith("/* DBProxy.x */")
    assert binds[":timeLimit"] == datetime.datetime(2026, 10, 4, 18, 38, 7)
    assert binds[":d"] == "Decimal('1.5')"
    assert "'activated'" in sqltext.filled(text, binds)
    assert sqltext.unfilled(text, binds) == (":timeLimit",)
    assert sqltext.cut("Trying 1/5 sql:SELECT 1 FROM dual WHERE a=:a var:{':a': 1}", "sql:", " var:") == (
        "SELECT 1 FROM dual WHERE a=:a",
        {":a": 1},
    )
    assert sqltext.cut("nothing printed after it") is None


def test_a_value_printed_as_its_angle_repr_is_kept_as_text_and_voids_nothing():
    """``<...>`` is how ``repr`` spells a value with no literal: that value's text, not the end of the map."""
    message = (
        "SELECT 1 FROM dual WHERE a=:a AND b=:b"
        "{':a': <cx_Oracle.Var of type DB_TYPE_NUMBER with value None>, ':b': 'x<y>z', "
        "':f': <function f.<locals>.g at 0x10>, ':q': 'it\\'s <'}"
    )
    text, binds = sqltext.cut(message)
    assert text == "SELECT 1 FROM dual WHERE a=:a AND b=:b"
    assert binds == {
        ":a": "<cx_Oracle.Var of type DB_TYPE_NUMBER with value None>",
        ":b": "x<y>z",
        ":f": "<function f.<locals>.g at 0x10>",
        ":q": "it's <",
    }
    assert sqltext.cut("SELECT 1 FROM dual{':a': <unclosed}") is None


def test_every_where_of_a_statement_is_read_each_to_the_end_of_its_own_clause():
    """``checkJobStatus`` asks ``jobsArchived4`` by ``:timeLimit`` after its ``UNION``; the first clause alone missed it."""
    union = (
        "SELECT jobStatus FROM ATLAS_PANDA.jobsActive4 WHERE PandaID=:pandaID "
        "UNION SELECT jobStatus FROM ATLAS_PANDA.jobsArchived4 WHERE PandaID=:pandaID AND modificationTime>:timeLimit "
    )
    derived = (
        "SELECT * FROM (SELECT PandaID FROM t WHERE a=:a UNION SELECT PandaID FROM u WHERE b=:b) "
        "WHERE rownum<:nRows ORDER BY PandaID"
    )
    minus = "SELECT x FROM t WHERE f=:f MINUS SELECT x FROM e WHERE status IN (:esSent,:esRunning)"
    nested = "SELECT x FROM t WHERE a IN (SELECT y FROM u WHERE b=:b) AND c=:c GROUP BY x"

    assert sqltext.where_clauses(union) == [" PandaID=:pandaID ", " PandaID=:pandaID AND modificationTime>:timeLimit "]
    assert sqltext.where_clauses(derived) == [" a=:a ", " b=:b", " rownum<:nRows "]
    assert sqltext.where_clauses(minus) == [" f=:f ", " status IN (:esSent,:esRunning)"]
    assert sqltext.where_clauses(nested) == [" a IN (SELECT y FROM u WHERE b=:b) AND c=:c ", " b=:b"]
    assert sqltext.where_clauses("SELECT 1 FROM dual") == []


def test_a_term_with_a_hole_ends_with_its_own_clause():
    """``getEventRanges``: the derived table's ``rownum<={}`` is one hole, not also ``attemptNr``'s."""
    ranges = (
        "UPDATE {}.JEDI_Events tab SET PandaID=:pandaID WHERE (jediTaskID,PandaID) IN ("
        "SELECT jediTaskID,PandaID FROM (SELECT jediTaskID,PandaID FROM {}.JEDI_Events tab "
        "WHERE jediTaskID=:jediTaskID AND attemptNr>:minAttemptNr ORDER BY jediTaskID,PandaID "
        ") WHERE rownum<={}) "
    )
    grouped = "SELECT x FROM t WHERE a=:a AND (cloud IS NULL OR cloud IN ({}))"

    assert [stop.source for stop in sql._hole_stops(ranges, 7)] == ["rownum<={}"]
    assert [stop.source for stop in sql._hole_stops(grouped, 7)] == ["cloud IN ({})"]
    assert sqltext.own_clause_terms(" attemptNr>:minAttemptNr ORDER BY x ) WHERE rownum<={}") == [" attemptNr>:minAttemptNr "]
    assert sqltext.own_clause_terms(" b=:b ) WHERE rownum<={}") == [" b=:b "]
    assert sqltext.own_clause_terms(" x IN (SELECT y FROM u WHERE z IN ({}))") == [" x IN (SELECT y FROM u WHERE z IN ({}))"]
    assert sqltext.own_clause_terms(" cloud IN ({}))") == [" cloud IN ({})"]


def test_a_term_ends_at_the_subquery_it_closes_and_a_condition_interpolated_whole_is_its_own_hole():
    """``do_preassign``: ``dl.rse IN ({}) ) {} {} {}`` is the subquery's term, then three outer conditions.

    ``updateDataset``'s ``AND {withCriteria}`` names no column: the whole condition is the hole.
    """
    preassign = (
        "SELECT t.jediTaskID FROM {}.JEDI_Tasks t WHERE t.status IN ('ready','running') "
        "AND EXISTS ( SELECT * FROM {}.JEDI_Dataset_Locality dl WHERE dl.jediTaskID=t.jediTaskID "
        "AND dl.rse IN ({}) ) {} {} {} AND EXISTS ( SELECT d.datasetID FROM {}.JEDI_Datasets d "
        "WHERE t.jediTaskID=d.jediTaskID ) ORDER BY t.currentPriority DESC "
    )
    criteria = "UPDATE ATLAS_PANDA.Datasets SET status=:status WHERE vuid=:vuid AND {}"

    assert [(stop.source, stop.reason) for stop in sql._hole_stops(preassign, 7)] == [
        ("dl.rse IN ({})", sql.BINDS_ARE_A_HOLE),
        ("{} {} {}", sql.CONDITION_IS_A_HOLE),
    ]
    assert [(stop.source, stop.reason) for stop in sql._hole_stops(criteria, 7)] == [("{}", sql.CONDITION_IS_A_HOLE)]
    # A column named through a hole is still the column's.
    assert [stop.reason for stop in sql._hole_stops("SELECT x FROM t WHERE {}=:{}", 7)] == [sql.COLUMN_IS_A_HOLE]


def test_a_column_a_write_names_at_run_time_is_said_not_passed_over():
    """``updateJobStatus`` appends ``,{key}=:{key}`` for each key of the pilot's ``param``: the pairing read the
    columns it could and passed over the rest with ``continue``, so the pilot's update wrote ``jobStatus`` alone in
    the map, and nothing said a column was missing.  Each item passed over is said, as the statement spells it."""
    marked = (
        f"UPDATE ATLAS_PANDA.jobsActive4 SET jobStatus=:jobStatus{_O},{{}}=:{{}},memory_leak=:memory_leak{_C}"
        ",startTime=CURRENT_DATE WHERE PandaID=:PandaID"
    )
    (statement,) = sqltext.writes(marked)
    assert set(statement.columns) == {"jobStatus", "memory_leak", "startTime"}
    assert sqltext.unread_columns(marked) == [("SET {}=:{}", sqltext.SET_COLUMN_IS_A_HOLE)]
    # Read whole, nothing is said.
    assert sqltext.unread_columns("UPDATE T SET a=:a, b=CURRENT_DATE WHERE id=:id") == []


def test_assignments_python_writes_whole_are_said_and_so_is_a_set_with_none():
    """``SET {}`` is a spec's ``bindUpdateChangesExpression()`` -- the object's changed attributes; a ``SET`` item
    that is a row of columns from a subquery is no ``column = value``; and the variant where no attribute changed
    writes the table with no column read at all, which no item can say."""
    assert sqltext.unread_columns("UPDATE ATLAS_PANDA.filesTable4 SET {} WHERE row_ID=:row_ID") == [
        ("SET {}", sqltext.SET_ITEM_IS_A_HOLE)
    ]
    attempts = (
        "UPDATE ATLAS_PANDA.TASK_ATTEMPTS SET (endtime, endstatus) = "
        "(SELECT CURRENT_DATE, status FROM ATLAS_PANDA.JEDI_Tasks WHERE jediTaskID=:jediTaskID) WHERE jediTaskID=:jediTaskID"
    )
    assert [why for _what, why in sqltext.unread_columns(attempts)] == [sqltext.SET_ITEM_NOT_READ]
    assert sqltext.unread_columns("UPDATE ATLAS_PANDA.filesTable4 SET  WHERE row_ID=:row_ID") == [
        ("filesTable4", sqltext.COLUMNS_NOT_READ)
    ]


def test_an_insert_whose_columns_python_names_is_said_once_with_its_column_list():
    """``INSERT INTO jobsArchived4 ({}) VALUES({})`` is ``JobSpec.columnNames()``: every column of the class, none
    of them in the text.  Said once, with the list as the statement holds it; a name the values have no item for is
    said too, and so is an ``INSERT`` that names no columns."""
    assert sqltext.unread_columns("INSERT INTO ATLAS_PANDA.jobsArchived4 ({}) VALUES({})") == [
        ("({})", sqltext.INSERT_COLUMN_IS_A_HOLE)
    ]
    # The reading unrolls columnNames()' loop once or twice: one list, said once.
    for unrolled in ("({},{})", "({}{})"):
        assert sqltext.unread_columns(f"INSERT INTO ATLAS_PANDA.jobsArchived4 {unrolled} VALUES({{}})") == [
            ("({})", sqltext.INSERT_COLUMN_IS_A_HOLE)
        ]
    # Values from Python too: the list is said, and the table is not said again.
    assert sqltext.unread_columns("INSERT INTO ATLAS_PANDA.workflows ({}) {}") == [
        ("({})", sqltext.INSERT_COLUMN_IS_A_HOLE)
    ]
    template = "INSERT INTO ATLAS_PANDA.JEDI_Output_Template (outTempID,datasetID,{}) VALUES(:o,:d,{})"
    assert sqltext.unread_columns(template) == [("(outTempID,datasetID,{})", sqltext.INSERT_COLUMN_IS_A_HOLE)]
    assert sqltext.unread_columns("INSERT INTO T (a,b,c) VALUES(:a,:b)") == [("c", sqltext.INSERT_VALUE_NOT_PAIRED)]
    assert sqltext.unread_columns("INSERT INTO ATLAS_PANDA.Metrics VALUES (:site, :metric)") == [
        ("Metrics", sqltext.COLUMNS_NOT_READ)
    ]
    assert sqltext.unread_columns("INSERT INTO T (a,b) VALUES(:a,:b)") == []


def test_unread_columns_are_stops_on_the_statement_and_not_on_the_rows_it_picks():
    """A run keeps them apart from :attr:`sql.Execution.unresolved`, which a read site carries and a gate reads as
    a statement read only in part; both are stops on the statement, and a plain write reads no table.  Where a
    fragment has no text of its own that stop says the statement is not all here, and no table is said again."""
    source = (
        "def f(self, param):\n"
        "    sql1 = 'UPDATE ATLAS_PANDA.jobsActive4 SET jobStatus=:jobStatus'\n"
        "    for key in list(param):\n"
        "        sql1 += ',{0}=:{0}'.format(key)\n"
        "    sql1 += ' WHERE PandaID=:PandaID'\n"
        "    self.cur.execute(sql1, varMap)\n"
        "    sqlA = 'INSERT INTO ATLAS_PANDA.jobsArchived4 ({0}) '.format(JobSpec.columnNames())\n"
        "    sqlA += JobSpec.bindValuesExpression()\n"
        "    self.cur.execute(sqlA, job.valuesMap())\n"
    )
    tree = ast.parse(source)
    from bamboo.codemap.panda.pathcond import attach_parents

    attach_parents(tree)
    func = tree.body[0]
    runs = sql.executions(func)
    by_line = {run.call.lineno: run for run in runs}
    assert {(stop.source, stop.reason) for stop in by_line[6].unread_columns} == {
        ("SET {}=:{}", sqltext.SET_COLUMN_IS_A_HOLE)
    }
    assert by_line[6].unresolved == ()
    assert {stop.reason for stop in by_line[9].unresolved} == {sql.FRAGMENT_NOT_RENDERED}
    assert sqltext.COLUMNS_NOT_READ not in {stop.reason for stop in by_line[9].unread_columns}
    stops = sqlwrite.statement_stops(func, "taskbuffer/db_proxy_mods/job_complex_module.py")
    said = [stop for stop in stops if stop.reason == sqltext.SET_COLUMN_IS_A_HOLE]
    assert [(stop.statement, stop.detail, stop.reads) for stop in said] == [(6, "SET {}=:{}", [])]


def test_a_term_naming_its_column_and_its_bind_by_holes_is_kept_apart_from_the_keys():
    """``{k}=:cr_{k}`` over a mapping's keys names no column a reader of *keys* can take, and was dropped -- in a
    loop's passes or not.  It is kept apart (:attr:`sqltext.SqlWrite.key_holes`): the holes' numbers, for a reader
    holding what the holes are, which says whether it names the row."""
    marked = f"UPDATE T SET a=:a WHERE {sqltext.REPEAT_OPEN}{{}}=:cr_{{}}{sqltext.REPEAT_CLOSE} AND b=:b"
    (write,) = sqltext.writes(marked)
    assert (write.keys, write.key_holes) == ({"b": ":b"}, [(1, ":cr_{}", (2,))])
    (unrolled,) = sqltext.writes("UPDATE T SET a=:a WHERE {}=:cr_{}")
    assert (unrolled.keys, unrolled.key_holes) == ({}, [(1, ":cr_{}", (2,))])
