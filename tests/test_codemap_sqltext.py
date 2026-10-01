"""Reading a statement's terms off its sqlglot tree (``sqltext.terms``).

The text readers filed every compared column against every table a statement
named, under every verb, subqueries included.  These pin what the tree reading
says instead, on the shapes the corpus census found that wrong.
"""

from __future__ import annotations

import ast

from bamboo.codemap.gitsource import blob_sha
from bamboo.codemap.models import SourceModule
from bamboo.codemap.panda import attribution, sqltext
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
    reassembly did not.
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
