"""Code Map extraction, gates, and namespace-scoped storage.

The value-enumeration slice is the map's simplest: the code declares a name, a
value and usually a comment, all in a fixed shape, so nothing has to be
inferred.  These tests pin the parts that were *not* obvious and that real
PanDA source forced corrections to:

* what counts as one enumeration (a module can declare two, with overlapping
  values, and the shared name prefix is the only thing that separates them);
* that a constant nothing reads is not an index entry;
* that rebuilding a Code Map cannot take the incident graph with it.
"""

from __future__ import annotations

import ast
import contextlib
from collections import Counter

import click.testing
import pytest

from bamboo.codemap import diff, evidence, gates
from bamboo.codemap.gitsource import blob_sha
from bamboo.codemap.models import (
    KEY_BOUND,
    KEY_HANDED,
    KEY_KEPT,
    KEY_NO_USE,
    KEY_RETURNED,
    REPORTS_ROWS_CHANGED,
    RESULT_AGGREGATE,
    RESULT_ROWS,
    RESULT_VARIES,
    Anchor,
    AnnotationAudit,
    BoundaryNode,
    Branch,
    EntityNode,
    EntryPoint,
    EnumerationWrite,
    FilterStageNode,
    JunctionNode,
    KeyFrom,
    LogSiteNode,
    LoopCutNode,
    MapFragment,
    ReadingStop,
    ReadSiteNode,
    RowKeySource,
    SourceModule,
    SubjectNode,
    ValueEnumNode,
    required_values,
)
from bamboo.codemap.panda import attribution, pathcond, promotion, sql, sqltext, values
from bamboo.codemap.panda.recognizers import (
    alias,
    boundary,
    emit,
    errorcode,
    flush,
    logfile,
    loopcut,
    progress,
    selection,
    sqlwrite,
    tables,
    trigger,
)
from bamboo.models.graph_element import NodeType
from bamboo.scripts import check_map, diff_map

MAP_ID = "panda"
VERSION = "panda-server-source 1.0.2"


def _module(source: str, rel: str = "pandaserver/taskbuffer/ErrorCode.py") -> SourceModule:
    """Build the parsed module a recognizer takes."""
    return SourceModule(
        package=rel.split("/")[0],
        rel_path=rel,
        tree=ast.parse(source),
        source=source,
        blob_sha=blob_sha(source),
    )


def _extract(source: str, rel: str = "pandaserver/taskbuffer/ErrorCode.py"):
    return errorcode.extract([_module(source, rel)], MAP_ID, VERSION)


# --------------------------------------------------------------------------- #
# extraction
# --------------------------------------------------------------------------- #


def test_extracts_name_value_and_comment():
    """The comment above a constant is its only human-readable meaning.

    ``ast`` throws comments away, so they are recovered from the source text;
    without them an index entry is a bare number with no explanation.
    """
    enums, _ = _extract("# error code\n\n# killed\nEC_Kill = 100\n")

    assert len(enums) == 1
    entry = enums[0]
    assert entry.constant == "EC_Kill"
    assert entry.value == 100
    assert entry.comment == "killed"
    assert entry.node_type is NodeType.VALUE_ENUM
    assert entry.map_id == MAP_ID
    assert entry.derived_from == VERSION


def test_comment_does_not_leak_across_a_blank_line():
    """A blank line ends one entry's comment; otherwise it absorbs its neighbour's."""
    enums, _ = _extract("# killed\nEC_Kill = 100\n\nEC_Transfer = 101\n")

    by_name = {e.constant: e for e in enums}
    assert by_name["EC_Kill"].comment == "killed"
    assert by_name["EC_Transfer"].comment is None


def test_prefix_separates_two_enumerations_in_one_module():
    """One module can declare two enumerations whose values overlap.

    ``EventServiceUtils`` defines event states *and* task types, so ``0`` means
    two different things in one file.  Python has no enum block here, so the
    shared prefix is how the code states the grouping -- keying on the module
    alone would make ``(namespace, value)`` ambiguous.
    """
    source = (
        "ST_ready = 0\nST_sent = 1\n"
        "TASK_NORMAL = 0\nTASK_EVENT_SERVICE = 1\n"
    )
    enums, _ = _extract(source, "pandaserver/taskbuffer/EventServiceUtils.py")

    namespaces = {e.constant: e.namespace for e in enums}
    assert namespaces["ST_ready"] != namespaces["TASK_NORMAL"]
    assert namespaces["ST_ready"] == namespaces["ST_sent"]
    assert namespaces["ST_ready"].endswith(".ST")
    # The display name stays module-scoped so it does not read as "...EC.EC_Kill".
    assert {e.name for e in enums} == {
        "taskbuffer.EventServiceUtils.ST_ready",
        "taskbuffer.EventServiceUtils.ST_sent",
        "taskbuffer.EventServiceUtils.TASK_NORMAL",
        "taskbuffer.EventServiceUtils.TASK_EVENT_SERVICE",
    }


def test_lone_prefixed_constant_is_not_given_its_own_namespace():
    """A prefix only groups when several constants share it."""
    enums, _ = _extract("MESSAGE_JSON = 'bad json'\nLATENCY = 'latency'\n",
                        "pandaserver/api/v1/common.py")

    assert {e.namespace for e in enums} == {"api.common"}


def test_declared_value_set_is_not_an_index_entry():
    """A collection literal is a vocabulary oracle, not a name<->value pair.

    Counting the two together makes an error-code module and a config module
    look like the same kind of file, which is what hid the distinction in the
    first place.
    """
    source = "EC_Kill = 100\nFINAL_TASK_STATUSES = ['done', 'failed']\n"
    enums, coverage = _extract(source)

    assert [e.constant for e in enums] == ["EC_Kill"]
    # Both are declarations, so both are candidates -- the value set shows up
    # as an unexplained candidate rather than vanishing from the denominator.
    assert coverage[0].candidates == 2
    assert coverage[0].explained == 1


def test_computed_constants_stay_out_of_the_denominator():
    """A computed value is not a failed extraction, it is a different statement."""
    source = "EC_Kill = 100\nTIMEOUT = compute_timeout()\nLIMIT = 3 * 60\n"
    enums, coverage = _extract(source)

    assert [e.constant for e in enums] == ["EC_Kill"]
    assert coverage[0].candidates == 1
    assert coverage[0].ratio == 1.0


def test_references_are_counted_across_the_whole_corpus():
    """A constant belongs in a decoding index only if something reads it."""
    definition = _module("EC_Kill = 100\nEC_Unused = 999\n")
    user = _module(
        "from x import ErrorCode\nif code == ErrorCode.EC_Kill:\n    pass\n",
        "pandaserver/dataservice/user.py",
    )
    enums, _ = errorcode.extract([definition, user], MAP_ID, VERSION)

    by_name = {e.constant: e for e in enums}
    assert by_name["EC_Kill"].references >= 1
    assert by_name["EC_Unused"].references == 0


# --------------------------------------------------------------------------- #
# gates
# --------------------------------------------------------------------------- #


def _enum(constant: str, value, namespace: str = "taskbuffer.ErrorCode.EC", references: int = 1):
    return ValueEnumNode(
        map_id=MAP_ID,
        derived_from=VERSION,
        name=f"{namespace}.{constant}",
        namespace=namespace,
        constant=constant,
        value=value,
        references=references,
        anchor=Anchor(package="pandaserver", file="x.py", line_start=1),
    )


def test_unreferenced_constants_are_flagged():
    fragment = MapFragment(
        map_id=MAP_ID,
        derived_from=VERSION,
        value_enums=[_enum("EC_Kill", 100), _enum("EC_Dead", 206, references=0)],
    )
    result = gates.value_enum_referenced(fragment)

    assert not result.passed
    assert result.checked == 2
    assert any("EC_Dead" in f for f in result.failures)


def test_reuse_across_namespaces_is_allowed():
    """Reused numbers are the reason the key carries a namespace at all."""
    fragment = MapFragment(
        map_id=MAP_ID,
        derived_from=VERSION,
        value_enums=[
            _enum("EC_Kill", 100, "taskbuffer.ErrorCode.EC"),
            _enum("EC_Watcher", 100, "jobdispatcher.ErrorCode.EC"),
        ],
    )
    assert gates.namespace_disambiguates(fragment).passed


def test_collision_inside_one_namespace_is_rejected():
    """Within one namespace a repeated value makes the key identify nothing."""
    fragment = MapFragment(
        map_id=MAP_ID,
        derived_from=VERSION,
        value_enums=[
            _enum("ST_ready", 0, "taskbuffer.EventServiceUtils"),
            _enum("TASK_NORMAL", 0, "taskbuffer.EventServiceUtils"),
        ],
    )
    result = gates.namespace_disambiguates(fragment)

    assert not result.passed
    assert "ST_ready" in result.failures[0]


# --------------------------------------------------------------------------- #
# identity
# --------------------------------------------------------------------------- #


def test_content_hash_ignores_the_anchor():
    """A gloss keyed on position would be thrown away by unrelated edits.

    Between two PanDA releases the pilot boundary moved file entirely while
    remaining the same boundary, so anything derived from a junction has to be
    keyed on what it means, not where it was found.
    """
    branches = [Branch(outcome="assigning", path_condition=["cloud is None"])]
    here = JunctionNode(
        map_id=MAP_ID, derived_from=VERSION, name="n", subject="JediTaskSpec.status",
        owner="m::f", branches=branches,
        anchor=Anchor(package="p", file="a.py", line_start=10),
    )
    moved = JunctionNode(
        map_id=MAP_ID, derived_from=VERSION, name="n", subject="JediTaskSpec.status",
        owner="m::f", branches=branches,
        anchor=Anchor(package="p", file="b.py", line_start=999),
    )
    assert here.content_hash() == moved.content_hash()


def test_content_hash_changes_when_the_condition_changes():
    common = {
        "map_id": MAP_ID,
        "derived_from": VERSION,
        "name": "n",
        "subject": "JediTaskSpec.status",
        "owner": "m::f",
    }
    before = JunctionNode(branches=[Branch(outcome="ready", path_condition=["a > 1"])], **common)
    after = JunctionNode(branches=[Branch(outcome="ready", path_condition=["a > 2"])], **common)

    assert before.content_hash() != after.content_hash()


# --------------------------------------------------------------------------- #
# version stamping
# --------------------------------------------------------------------------- #


def test_blob_sha_matches_git_for_lf_source():
    """The hash equals ``git rev-parse HEAD:<path>`` for the same bytes.

    That equality is what lets a map built from a checkout be compared file by
    file with one built from the installed package.
    """
    # `git hash-object` on b"hello\n" -- the canonical example.
    assert blob_sha("hello\n") == "ce013625030ba8dba906f756967f9e9ca394464a"


def test_blob_sha_tracks_content_not_position():
    assert blob_sha("EC_Kill = 100\n") != blob_sha("EC_Kill = 101\n")
    assert blob_sha("EC_Kill = 100\n") == blob_sha("EC_Kill = 100\n")


def test_anchors_carry_the_content_hash():
    """Without it a rebuild cannot tell whether the enclosing file moved on."""
    enums, _ = _extract("# killed\nEC_Kill = 100\n")

    assert enums[0].anchor is not None
    assert enums[0].anchor.blob_sha == blob_sha("# killed\nEC_Kill = 100\n")


def test_describe_returns_none_outside_a_git_checkout(tmp_path):
    """Not being a checkout is a source form, not a failure.

    An installed distribution has no repository; the caller falls back to the
    identity that form does offer rather than the build refusing to run.
    """
    from bamboo.codemap.gitsource import describe

    assert describe(tmp_path) is None


def test_code_map_labels_are_distinct_from_incident_labels():
    """Label separation is what makes a namespace-scoped delete possible."""
    code_map = {NodeType.JUNCTION_POINT, NodeType.BOUNDARY, NodeType.SUBJECT, NodeType.VALUE_ENUM}
    incident = {NodeType.SYMPTOM, NodeType.CAUSE, NodeType.RESOLUTION, NodeType.PROCEDURE}

    assert not {n.value for n in code_map} & {n.value for n in incident}


# --------------------------------------------------------------------------- #
# boundaries
# --------------------------------------------------------------------------- #


_ENDPOINT = '''
@request_validation(_logger, secure=True, production=True, request_method="POST")
def update_job(req: PandaRequest, job_id: int, job_status: str, pilot_id: str = None):
    _logger.debug(f"update_job({job_id}, {job_status})")
    return None
'''


def _boundaries(source: str, rel: str = "pandaserver/api/v1/pilot_api.py"):
    return boundary.extract([_module(source, rel)], MAP_ID, VERSION)[0]


def test_endpoint_is_recognised_by_its_declaration():
    """The decorator marks the entry point; a ``req`` parameter only hints at it."""
    found = _boundaries(_ENDPOINT)

    assert len(found) == 1
    assert found[0].interface == "pilot_api::update_job"
    assert found[0].system == "pilot"
    assert found[0].node_type is NodeType.BOUNDARY


def test_request_object_is_not_a_carried_value():
    """``req`` is the transport, not something the far side supplies."""
    assert _boundaries(_ENDPOINT)[0].carried_values == ["job_id", "job_status", "pilot_id"]


def test_access_conditions_come_from_the_declaration():
    """These are the first place a request can be rejected.

    A command that appears to have vanished may have been turned away here,
    before any junction saw it -- so the boundary has to carry them.
    """
    assert _boundaries(_ENDPOINT)[0].access_conditions == {
        "secure": True,
        "production": True,
        "request_method": "POST",
    }


def test_only_logged_values_are_observable():
    """What crossed and was never written down cannot be recovered afterwards."""
    found = _boundaries(_ENDPOINT)[0]

    assert found.observable_values == ["job_id", "job_status"]
    assert "pilot_id" not in found.observable_values


def test_version_binding_is_picked_out_of_the_payload():
    """A system whose version moves independently cannot share this map's stamp."""
    assert _boundaries(_ENDPOINT)[0].version_binding == ["pilot_id"]


def test_endpoint_carrying_nothing_is_not_a_boundary():
    """An introspection endpoint is an entry point but nothing crosses it."""
    source = (
        '@request_validation(_logger, secure=True)\n'
        'def is_alive(req: PandaRequest):\n'
        '    return {"alive": True}\n'
    )
    assert _boundaries(source, "pandaserver/api/v1/system_api.py") == []


def test_arbitrary_kwargs_are_recorded_rather_than_ignored():
    """Listing no carried values here would claim nothing crosses -- the opposite."""
    source = (
        '@request_validation(_logger, secure=True)\n'
        'def get_attributes(req: PandaRequest, **kwargs):\n'
        '    return {}\n'
    )
    found = _boundaries(source, "pandaserver/api/v1/system_api.py")

    assert len(found) == 1
    assert found[0].accepts_arbitrary is True
    assert found[0].carried_values == []


def test_callers_without_their_own_release_cycle_are_not_separate_systems():
    """Crossing a trust boundary is not the same as crossing a version boundary."""
    found = _boundaries(_ENDPOINT, "pandaserver/api/v1/task_api.py")

    assert found[0].system == "client"


def test_ownership_gate_catches_a_parameter_the_signature_lacks():
    """The decorator and the signature must agree on which parameter carries the task."""
    source = (
        '@request_validation(_logger, task_owner=True, task_id_param="task_id")\n'
        'def kill(req: PandaRequest, jedi_task_id: int):\n'
        '    return None\n'
    )
    fragment = MapFragment(
        map_id=MAP_ID,
        derived_from=VERSION,
        boundaries=_boundaries(source, "pandaserver/api/v1/task_api.py"),
    )
    result = gates.boundary_ownership_param_declared(fragment)

    assert not result.passed
    assert "task_id" in result.failures[0]


# --------------------------------------------------------------------------- #
# progress: attribution and path conditions
# --------------------------------------------------------------------------- #

# Two classes declaring ``status`` is the situation the whole attribution layer
# exists for; ``jobStatus`` is the control, declared once and so never in doubt.
_SPECS = """
class JediTaskSpec(object):
    _attributes = ("jediTaskID", "status", "oldStatus")

    def statusToUpdateContents(cls):
        return ["defined", "ready"]

class JediFileSpec(object):
    _attributes = ("fileID", "status", "proc_status")

class FileSpec(object):
    _attributes = ("lfn", "status")

class JobSpec(object):
    _attributes = ("PandaID", "jobStatus")
"""


def _progress(source: str, rel: str):
    """Extract from *source* alongside the spec declarations it writes to."""
    modules = [
        _module(_SPECS, "pandaserver/taskbuffer/Specs.py"),
        _module(source, rel),
    ]
    subjects, junctions, coverage, _diag, _enum = progress.extract(modules, MAP_ID, VERSION)
    return subjects, [j for j in junctions if j.owner.startswith(rel)], coverage


def test_a_non_spec_receiver_wins_over_the_single_declaring_class():
    """The shortcut that skips the receiver must not skip a receiver that answers.

    ``atom = NucleusSpec(...)`` then ``atom.state = ...`` in SiteMapper was
    filed as ``JediDatasetSpec.state`` with basis ``certain``, because ``state``
    is declared by exactly one class and the shortcut never looks at the
    object.  ``NucleusSpec`` declares no columns, so it is not a spec and there
    is nothing to resolve -- the guard for that sat one rung too low to be
    reached.  ``structural-attribution-agrees`` cannot catch it: both readings
    come from the same declaration table and so agree on the wrong answer.
    """
    source = (
        "class NucleusSpec(object):\n"
        "    def __init__(self, name):\n"
        "        self.name = name\n"
        "\n"
        "class M:\n"
        "    def f(self, site):\n"
        "        atom = NucleusSpec(site.name)\n"
        "        atom.proc_status = site.proc_status\n"
    )
    _subjects, junctions, _coverage = _progress(source, "pandajedi/jedibrokerage/B.py")

    assert junctions == []


def test_a_receiver_built_from_an_import_alias_of_a_spec_is_still_a_spec_write():
    """``from ...FileSpec import FileSpec as JobFileSpec`` is a spec by another name.

    The guard above reads the constructor's name, and the name in the source
    is the alias.  Without following it, moving the guard up would drop three
    correct writes in ``JediFileSpec.py`` along with the one it is for.
    """
    source = (
        "from pandaserver.taskbuffer.FileSpec import FileSpec as JobFileSpec\n"
        "\n"
        "class M:\n"
        "    def f(self, token):\n"
        "        jobFileSpec = JobFileSpec()\n"
        "        jobFileSpec.lfn = token\n"
    )
    _subjects, junctions, _coverage = _progress(source, "pandaserver/taskbuffer/JediFileSpec.py")

    assert [j.subject for j in junctions] == ["FileSpec.lfn"]


def test_single_declaring_class_needs_no_object_type():
    """One declaring class settles the subject whatever the object expression is.

    ``self.job.jobStatus`` has an attribute, not a name, on the left -- there is
    no variable to resolve.  It does not matter: only ``JobSpec`` declares
    ``jobStatus``, so the write can only be to a ``JobSpec``.
    """
    source = "def f(self):\n    self.job.jobStatus = 'failed'\n"
    _subjects, junctions, _cov = _progress(source, "pandaserver/dataservice/adder_gen.py")

    assert [(j.subject, j.attribution) for j in junctions] == [("JobSpec.jobStatus", "certain")]


def test_constructor_call_attributes_the_write():
    """The code states the type outright, so nothing is guessed."""
    source = (
        "def f():\n"
        "    spec = JediFileSpec()\n"
        "    spec.status = 'ready'\n"
    )
    _subjects, junctions, _cov = _progress(source, "pandajedi/jedirefine/TaskRefinerBase.py")

    assert junctions[0].subject == "JediFileSpec.status"
    assert junctions[0].attribution == "certain"


def test_a_name_and_an_import_are_not_evidence():
    """Only the import points at ``JediFileSpec`` here, and that is not enough.

    Attributing from the variable's name -- narrowed by what the module
    imports -- used to settle writes like this one.  It was removed: across the
    whole of PanDA it decided two writes out of 278, and it was the only basis
    whose answers had to be marked as untrusted, since importing a class is not
    evidence that this particular variable holds one.

    The remedy for a shape like this is a type annotation upstream, not another
    rule here.
    """
    source = (
        "from pandaserver.taskbuffer.JediFileSpec import JediFileSpec\n"
        "def f(files):\n"
        "    for file in files:\n"
        "        file.status = 'ready'\n"
    )
    _subjects, junctions, _cov = _progress(source, "pandajedi/jediorder/JobGenerator.py")

    assert junctions[0].attribution == "unresolved"


def test_an_annotation_settles_what_the_name_could_not():
    """The intended remedy, and it lands in the strongest basis.

    One line of standard Python -- useful to mypy and a reader either way --
    replaces the inference rule that was removed.
    """
    source = (
        "from pandaserver.taskbuffer.JediFileSpec import JediFileSpec\n"
        "def f(file: JediFileSpec):\n"
        "    file.status = 'ready'\n"
    )
    _subjects, junctions, _cov = _progress(source, "pandajedi/jediorder/JobGenerator.py")

    assert junctions[0].subject == "JediFileSpec.status"
    assert junctions[0].attribution == "certain"


def test_importing_both_leaves_the_write_unresolved():
    """Where the evidence does not decide, the map says so rather than picking.

    The junction is still emitted: the writer is known even when the subject is
    not, which is what localize and prune work from.
    """
    source = (
        "from pandaserver.taskbuffer.FileSpec import FileSpec\n"
        "from pandaserver.taskbuffer.JediFileSpec import JediFileSpec\n"
        "def f(files):\n"
        "    for file in files:\n"
        "        file.status = 'ready'\n"
    )
    _subjects, junctions, _cov = _progress(source, "pandaserver/dataservice/adder_gen.py")

    assert junctions[0].attribution == "unresolved"
    assert junctions[0].subject == "?.status"
    assert junctions[0].branches[0].outcome == "ready"


def test_writes_to_attributes_no_spec_declares_are_not_junctions():
    """The subject universe is the spec declarations, not every attribute write.

    Without this bound the map fills with bookkeeping that can never be
    attributed and never belongs to a subject.
    """
    source = "def f(self):\n    self.plugin_flavor = 'atlas'\n"
    _subjects, junctions, coverage = _progress(source, "pandajedi/jedicore/Plugin.py")

    assert junctions == []
    assert coverage == []


def test_path_condition_records_the_negated_branch():
    """An ``else`` is a reason for the outcome, so it contributes ``not (...)``."""
    source = (
        "def f(self, taskSpec, taskBroken):\n"
        "    if taskBroken:\n"
        "        taskSpec.status = 'tobroken'\n"
        "    else:\n"
        "        taskSpec.status = 'finishing'\n"
    )
    _subjects, junctions, _cov = _progress(source, "pandajedi/jediorder/ContentsFeeder.py")

    by_outcome = {b.outcome: b.path_condition for b in junctions[0].branches}
    assert by_outcome["tobroken"] == ["taskBroken"]
    assert by_outcome["finishing"] == ["not (taskBroken)"]


def test_bare_name_condition_carries_the_expression_behind_it():
    """``if not allowed:`` names no predicate until the assignment is substituted.

    Real PanDA moved a command's acceptance test into a helper between
    releases, which is exactly this shape.
    """
    source = (
        "def f(self, taskSpec, comStr):\n"
        "    allowed = self._check_command_allowed(comStr)\n"
        "    if not allowed:\n"
        "        taskSpec.status = 'tobroken'\n"
    )
    _subjects, junctions, _cov = _progress(source, "pandaserver/taskbuffer/task_event_module.py")

    condition = junctions[0].branches[0].path_condition[0]
    assert "_check_command_allowed" in condition


def test_declared_subsets_do_not_gate_the_outcomes():
    """A purpose-built status list is a lower bound, not a vocabulary.

    ``statusToUpdateContents`` returns the statuses eligible for a content
    update, not every status a task can hold, so an outcome outside it is
    normal.  Reported for a human, never failed.
    """
    source = "def f(self, taskSpec):\n    taskSpec.status = 'finishing'\n"
    _subjects, junctions, _cov = _progress(source, "pandajedi/jediorder/ContentsFeeder.py")
    fragment = MapFragment(map_id=MAP_ID, derived_from=VERSION, junctions=junctions)

    assert all(r.gate != "outcome-in-vocabulary" for r in gates.run_all(fragment))


def test_unresolved_attributes_are_reported_where_they_concentrate():
    """A total would hide which attributes the map is thin on.

    Each row is a candidate for one line of annotation upstream, so naming the
    attribute is the whole point of the report.
    """
    source = (
        "def f(self, files):\n"
        "    for file in files:\n"
        "        file.status = 'ready'\n"
        "    self.job.jobStatus = 'failed'\n"
    )
    _subjects, junctions, _cov = _progress(source, "pandaserver/dataservice/adder_gen.py")
    fragment = MapFragment(map_id=MAP_ID, derived_from=VERSION, junctions=junctions)

    # ``jobStatus`` has a single declaring class, so only ``status`` is thin.
    assert [row[0] for row in gates.unresolved_attributes(fragment)] == ["status"]


def test_self_write_in_a_non_spec_class_is_not_a_spec_write():
    """``self.attr`` writes the enclosing class's own field, whatever it is named.

    A WatchDog's ``self.vo = "atlas"`` shares a name with a spec attribute and
    nothing else.  Recording it as an unresolved junction would put the
    WatchDog's bookkeeping in the map and leave it in the coverage denominator
    as a gap that can never close.
    """
    source = (
        "class AtlasTaskWithholderWatchDog(WatchDogBase):\n"
        "    def __init__(self):\n"
        "        self.vo = 'atlas'\n"
    )
    _subjects, junctions, coverage = _progress(
        source, "pandajedi/jedidog/AtlasTaskWithholderWatchDog.py"
    )

    assert junctions == []
    assert coverage == []


def test_subclass_of_a_spec_resolves_to_the_spec():
    """``PickleFileSpec(FileSpec)`` is real, and its ``self.status`` is a FileSpec's.

    The same hierarchy walk that drops the WatchDog write has to keep this one,
    which is why the rule reads the base classes instead of asking whether the
    enclosing class itself declares the attribute.
    """
    source = (
        "class PickleFileSpec(FileSpec):\n"
        "    def load(self):\n"
        "        self.status = 'ready'\n"
    )
    _subjects, junctions, _cov = _progress(source, "pandaserver/taskbuffer/PickleFileSpec.py")

    assert junctions[0].subject == "FileSpec.status"
    assert junctions[0].attribution == "certain"


def test_attribute_chain_resolves_from_its_own_usage():
    """``impl.taskSpec.status`` has no plain name on the left, and does not need one.

    Structural inference keys on the whole object expression, so a chain is
    read exactly like a variable.  What it never does is fall back to the
    chain's last *name* -- that was the naming heuristic, and it is gone.
    """
    source = (
        "def f(impl):\n"
        "    impl.taskSpec.status = 'staging'\n"
        "    use(impl.taskSpec.oldStatus)\n"
    )
    _subjects, junctions, _cov = _progress(source, "pandajedi/jediorder/TaskRefiner.py")

    assert junctions[0].subject == "JediTaskSpec.status"
    assert junctions[0].attribution == "structural"


def test_subscripted_object_stays_unresolved():
    """``self.dataset_map[name].status`` has no name to read; each container differs."""
    source = (
        "from pandaserver.taskbuffer.JediFileSpec import JediFileSpec\n"
        "class Adder:\n"
        "    def run(self):\n"
        "        self.dataset_map['x'].status = 'ready'\n"
    )
    _subjects, junctions, _cov = _progress(
        source, "pandaserver/dataservice/adder_atlas_plugin.py"
    )

    assert junctions[0].attribution == "unresolved"


def test_usage_settles_what_the_name_cannot():
    """The attributes touched on an object say more than what it was called.

    ``file`` fits ``FileSpec`` and ``JediFileSpec`` and the module imports
    neither, so naming is exhausted.  But only ``FileSpec`` declares every
    attribute the loop touches, and that is the answer without guessing.
    """
    source = (
        "class AdderGen:\n"
        "    def finalize(self):\n"
        "        for file in self.job.Files:\n"
        "            if file.lfn in self.merging:\n"
        "                file.status = 'merging'\n"
        "            self.log(file.GUID, file.fsize)\n"
    )
    _subjects, junctions, _cov = _progress(source, "pandaserver/dataservice/adder_gen.py")

    assert junctions[0].subject == "FileSpec.status"
    assert junctions[0].attribution == "structural"


def test_usage_contradicts_the_name_and_usage_is_taken():
    """A variable called ``fileSpec`` in a module importing ``JediFileSpec``…

    …that touches ``lfn``, which only ``FileSpec`` declares.  Naming would have
    answered ``JediFileSpec`` here and been wrong, which is the concrete reason
    that basis is gone rather than merely unprofitable.
    """
    source = (
        "from pandaserver.taskbuffer.JediFileSpec import JediFileSpec\n"
        "def f(files):\n"
        "    for fileSpec in files:\n"
        "        fileSpec.status = 'ready'\n"
        "        use(fileSpec.lfn)\n"
    )
    _subjects, junctions, _cov = _progress(source, "pandajedi/jediorder/JobGenerator.py")

    assert junctions[0].subject == "FileSpec.status"
    assert junctions[0].attribution == "structural"


def test_too_few_attributes_imply_nothing():
    """Touching only the attribute being written is not evidence of a class."""
    source = (
        "class Adder:\n"
        "    def run(self):\n"
        "        self.dataset_map['x'].status = 'running'\n"
    )
    _subjects, junctions, _cov = _progress(
        source, "pandaserver/dataservice/adder_atlas_plugin.py"
    )

    assert junctions[0].attribution == "unresolved"
    assert junctions[0].structural_subject is None


def test_gate_catches_declaration_and_usage_disagreeing():
    """Two independent readings of one fact; a split means one of them is wrong."""
    junction = JunctionNode(
        map_id=MAP_ID,
        derived_from=VERSION,
        name="j",
        subject="JediFileSpec.status",
        owner="pandaserver/dataservice/adder_gen.py::finalize",
        attribution="certain",
        structural_subject="FileSpec.status",
        branches=[Branch(outcome="ready")],
    )
    fragment = MapFragment(map_id=MAP_ID, derived_from=VERSION, junctions=[junction])
    result = gates.structural_attribution_agrees(fragment)

    assert not result.passed
    assert result.checked == 1
    assert "FileSpec.status (usage)" in result.failures[0]


def test_gate_skips_junctions_usage_cannot_speak_for():
    """No structural answer is silence, not disagreement."""
    junction = JunctionNode(
        map_id=MAP_ID,
        derived_from=VERSION,
        name="j",
        subject="JobSpec.jobStatus",
        owner="x.py::f",
        attribution="certain",
        structural_subject=None,
    )
    fragment = MapFragment(map_id=MAP_ID, derived_from=VERSION, junctions=[junction])
    result = gates.structural_attribution_agrees(fragment)

    assert result.passed
    assert result.checked == 0


# The adder idiom, as PanDA writes it: a method that appends its parameter, and
# a caller that hands it a freshly constructed spec.  Both containers are named
# ``Files`` and hold different classes, which is the whole point.
_ADDERS = """
class JobSpec(object):
    _attributes = ("PandaID", "jobStatus")
    __slots__ = _attributes + ("Files",)

    def addFile(self, file):
        self.Files.append(file)

    def load(self, states):
        for state in states:
            file_spec = FileSpec()
            self.addFile(file_spec)

class JediDatasetSpec(object):
    _attributes = ("datasetID", "status")
    __slots__ = _attributes + ("Files",)

    def addFile(self, fileSpec):
        self.Files.append(fileSpec)
"""

_JEDI_ADDER_CALLER = """
def refine(datasetSpec):
    use(datasetSpec.datasetID)
    fileSpec = JediFileSpec()
    datasetSpec.addFile(fileSpec)
"""


def _progress_multi(*sources: tuple[str, str]):
    """Extract with the spec fixtures plus several caller modules."""
    modules = [_module(_SPECS, "pandaserver/taskbuffer/Specs.py")]
    modules += [_module(text, rel) for text, rel in sources]
    subjects, junctions, coverage, _diag, _enum = progress.extract(modules, MAP_ID, VERSION)
    return subjects, junctions, coverage


def test_element_type_comes_from_what_was_put_in_the_container():
    """``for file in job.Files`` is settled by the ``FileSpec()`` that went in.

    PanDA never annotates ``addFile``, but it does construct the spec two lines
    before handing it over, and that is the same fact stated where the code
    happens to state it.
    """
    consumer = (
        "def finalize(job):\n"
        "    for file in job.Files:\n"
        "        file.status = 'merging'\n"
        "    job.jobStatus = 'merging'\n"      # pins ``job`` the way adder_gen does
    )
    _subjects, junctions, _cov = _progress_multi(
        (_ADDERS, "pandaserver/taskbuffer/JobSpec.py"),
        (consumer, "pandaserver/dataservice/adder_gen.py"),
    )
    written = [
        j for j in junctions if j.owner.endswith("::finalize") and j.subject.endswith(".status")
    ]

    assert written[0].subject == "FileSpec.status"
    assert written[0].attribution == "container"



def test_a_name_only_loops_bind_is_what_its_loops_containers_hold():
    """``propagateResultToJEDI``'s ``for fileSpec in jobSpec.Files`` -- and ``jobSpec.Files + pseudoFiles`` -- holds
    ``FileSpec`` rows for ``class_of`` too, where the fields the loop uses (a JEDI file's ``status`` as well) could not
    tell it; a name an assignment binds as well is not read by its loop."""
    consumer = (
        "def propagate(jobSpec: JobSpec, pseudoFiles: list[FileSpec]):\n"
        "    for fileSpec in jobSpec.Files:\n"
        "        use(fileSpec.status)\n"
        "    for other in jobSpec.Files + pseudoFiles:\n"
        "        use(other.status)\n"
        "    for mixed in jobSpec.Files:\n"
        "        use(mixed.status)\n"
        "    mixed = make()\n"
        "    use(mixed.status)\n"
    )
    modules = [
        _module(_SPECS, "pandaserver/taskbuffer/Specs.py"),
        _module(_ADDERS, "pandaserver/taskbuffer/JobSpec.py"),
        _module(consumer, "pandaserver/taskbuffer/db_proxy_mods/job_module.py"),
    ]
    attributor = attribution.SpecAttributor(progress.spec_attributes(modules), attribution.class_bases(modules))
    attributor.learn_element_types(modules)
    func = next(node for node in ast.walk(modules[-1].tree) if isinstance(node, ast.FunctionDef))

    def read(name: str):
        return attributor.class_of(ast.Name(id=name, ctx=ast.Load()), func, None)

    assert {name: read(name) for name in ("fileSpec", "other", "mixed")} == {
        "fileSpec": "FileSpec",
        "other": "FileSpec",
        "mixed": None,
    }

def test_two_containers_named_files_hold_different_classes():
    """``JobSpec.Files`` and ``JediDatasetSpec.Files`` are the pair naming cannot split.

    Resolving them apart is what makes the container basis worth the hop: the
    variable is called ``file`` in both cases.
    """
    consumer = (
        "def walk(job, datasetSpec):\n"
        "    for file in job.Files:\n"
        "        file.status = 'merging'\n"
        "    for other in datasetSpec.Files:\n"
        "        other.status = 'ready'\n"
        "    use(job.jobStatus, datasetSpec.datasetID)\n"   # pins both holders
    )
    _subjects, junctions, _cov = _progress_multi(
        (_ADDERS, "pandaserver/taskbuffer/JobSpec.py"),
        (_JEDI_ADDER_CALLER, "pandajedi/jedirefine/TaskRefinerBase.py"),
        (consumer, "pandaserver/dataservice/adder_gen.py"),
    )
    subjects = {j.subject for j in junctions if j.owner.endswith("::walk")}

    assert subjects == {"FileSpec.status", "JediFileSpec.status"}


def test_container_attribute_need_not_be_a_declared_column():
    """``Files`` is an object-graph edge, not a DB column.

    ``__slots__ = _attributes + ("Files", ...)`` says so outright.  Requiring
    the container to be a declared attribute made the whole pass learn nothing.
    """
    modules = [
        _module(_SPECS, "pandaserver/taskbuffer/Specs.py"),
        _module(_ADDERS, "pandaserver/taskbuffer/JobSpec.py"),
    ]
    declarations = progress.spec_attributes(modules)
    attributor = attribution.SpecAttributor(declarations, attribution.class_bases(modules))
    attributor.learn_element_types(modules)
    attributor.learn_self_attributes(modules)

    assert "Files" not in declarations["JobSpec"]
    assert attributor._element_types[("JobSpec", "Files")] == {"FileSpec"}


def test_module_qualified_constructor_is_still_a_constructor():
    """``SiteSpec.SiteSpec()`` states the type as plainly as ``SiteSpec()``.

    PanDA imports the module and calls through it, so the callee is an
    attribute rather than a name.  Reading only names sent
    ``entity_module.getSiteInfo`` to structural inference for a type the code
    was stating outright, one attribute along.
    """
    source = (
        "from pandaserver.taskbuffer import FileSpec\n"
        "def get_file():\n"
        "    ret = FileSpec.FileSpec()\n"
        "    ret.status = 'ready'\n"
        "    return ret\n"
    )
    _subjects, junctions, _cov = _progress(
        source, "pandaserver/taskbuffer/db_proxy_mods/entity_module.py"
    )

    assert junctions[0].subject == "FileSpec.status"
    assert junctions[0].attribution == "certain"


def test_what_a_class_does_with_its_own_field_is_pooled_across_its_methods():
    """``self.taskSpec`` is one field of one class, so every method that touches
    it describes the same object.

    ``TaskRefinerBase`` is why: the method writing ``self.taskSpec.status``
    touches only ``jediTaskID`` beside it -- two attributes half the specs
    declare -- while the class touches twelve, which only ``JediTaskSpec`` has.
    That write is the sole producer of ``topreprocess``, so reading the method
    alone left a declared status looking unreachable.
    """
    source = (
        "class TaskRefinerBase(object):\n"
        "    def remember(self, taskSpec):\n"
        "        self.taskSpec.oldStatus = self.taskSpec.status\n"
        "    def refine(self):\n"
        "        self.taskSpec.status = 'topreprocess'\n"
    )
    _subjects, junctions, _cov = _progress(source, "pandajedi/jedirefine/TaskRefinerBase.py")
    written = [j for j in junctions if j.owner.endswith("::refine")]

    assert [(j.subject, j.attribution) for j in written] == [
        ("JediTaskSpec.status", "structural")
    ]


def test_a_local_name_is_not_pooled_across_methods():
    """A local is one thing only for as long as its function lasts -- two
    methods using ``spec`` need not mean the same kind of object, so the
    evidence for widening the scope is missing.  It stays unresolved, which is
    reported rather than guessed at.
    """
    source = (
        "class Refiner(object):\n"
        "    def remember(self, spec):\n"
        "        spec.oldStatus = 'ready'\n"
        "    def refine(self, spec):\n"
        "        spec.status = 'topreprocess'\n"
    )
    _subjects, junctions, _cov = _progress(source, "pandajedi/jedirefine/Other.py")
    written = [j for j in junctions if j.owner.endswith("::refine")]

    assert [j.attribution for j in written] == ["unresolved"]


def test_a_copy_holds_what_the_original_held():
    """``copy.copy(spec)`` preserves the type; that is semantics, not a guess."""
    source = (
        "import copy\n"
        "def clone():\n"
        "    lib_file_spec = FileSpec()\n"
        "    runFileSpec = copy.copy(lib_file_spec)\n"
        "    runFileSpec.status = 'ready'\n"
    )
    _subjects, junctions, _cov = _progress(source, "pandajedi/jediorder/JobGenerator.py")

    assert junctions[0].subject == "FileSpec.status"
    assert junctions[0].attribution == "certain"


def test_a_copy_of_a_container_element_carries_the_element_type():
    """``copy.copy(d[key])`` -- the original is an element, not a name.

    ``create_pseudo_files_for_dyn_num_events`` builds its pseudo files this way,
    and the copy chain stopped at the subscript, so two writes stayed
    unattributed *with an annotation on the mapping sitting right above them*.
    The annotation could not help because nothing looked past the ``copy``.
    """
    source = (
        "import copy\n"
        "class Mod:\n"
        "    def make(self, rows):\n"
        "        row_id_spec_map: Dict[int, FileSpec] = {}\n"
        "        for row in rows:\n"
        "            tmp_file_spec = copy.copy(row_id_spec_map[row])\n"
        "            tmp_file_spec.status = 'cached'\n"
    )
    _subjects, junctions, _cov = _progress(
        source, "pandaserver/taskbuffer/db_proxy_mods/task_event_module.py"
    )

    assert junctions[0].subject == "FileSpec.status"
    assert junctions[0].attribution == "container"


def test_a_container_takes_its_element_type_from_what_is_put_in_it():
    """No annotation, and none needed: the code says it one line above.

    A local's evidence is always in the same function, so asking the target
    system to annotate one is asking it to restate what is already there --
    which is how a wrong element type got in.  The corpus agrees: of eighteen
    container element annotations, seventeen are on parameters or fields, where
    the evidence really is in another scope, and one was on a local.

    Measured on 36 sites before being trusted: the reading agreed with the
    existing answer 36 times and disagreed none.
    """
    source = (
        "import copy\n"
        "class Mod:\n"
        "    def make(self, rows):\n"
        "        row_id_spec_map = {}\n"
        "        for spec in rows:\n"
        "            use(spec.lfn)\n"
        "            row_id_spec_map[1] = spec\n"
        "        tmp_file_spec = copy.copy(row_id_spec_map[1])\n"
        "        tmp_file_spec.status = 'cached'\n"
    )
    _subjects, junctions, _cov = _progress(
        source, "pandaserver/taskbuffer/db_proxy_mods/task_event_module.py"
    )
    written = [j for j in junctions if j.subject.endswith(".status")]

    assert [(j.subject, j.attribution) for j in written] == [
        ("FileSpec.status", "container")
    ]


def test_an_annotation_outranks_what_is_put_in_the_container():
    """The annotation wins, which is what keeps the gate from being circular.

    ``container_annotations_agree`` compares the stated element type against
    the stored one.  If the stored reading were preferred there would be
    nothing left to check -- the gate would be comparing the attribution
    against itself, the tautology that sank attribution by declared vocabulary.
    So this reading ranks *below* the annotation and only fills its silence.
    """
    source = (
        "class Mod:\n"
        "    def make(self):\n"
        "        m: Dict[int, JediFileSpec] = {}\n"
        "        spec = FileSpec()\n"
        "        m[1] = spec\n"
        "        m[1].status = 'ready'\n"
    )
    _subjects, junctions, _cov = _progress(source, "pandajedi/jedirefine/TaskRefinerBase.py")
    written = [j for j in junctions if j.owner.endswith("::make")]

    assert [(j.subject, j.attribution) for j in written] == [
        ("JediFileSpec.status", "certain")
    ]


def test_a_container_holding_two_classes_settles_nothing():
    """The reading assumes the container is homogeneous, so it has to check.

    Same assumption the adder idiom carries, which is why both report
    ``container`` rather than ``certain``.  Where the assumption is visibly
    false the honest answer is the unresolved write.
    """
    source = (
        "class Mod:\n"
        "    def make(self):\n"
        "        m = {}\n"
        "        m[1] = FileSpec()\n"
        "        m[2] = JediFileSpec()\n"
        "        m[1].status = 'ready'\n"
    )
    _subjects, junctions, _cov = _progress(source, "pandajedi/jedirefine/TaskRefinerBase.py")
    written = [j for j in junctions if j.owner.endswith("::make")]

    assert [j.attribution for j in written] == ["unresolved"]


def test_a_mapping_is_read_through_get_as_well_as_subscript():
    """``d.get(key)`` and ``d[key]`` are the same read, and PanDA writes both.

    Reading only the subscript left an annotation that was already correct doing
    nothing: ``workflow_core`` states ``data_spec_map: Dict[str, WFDataSpec]``
    and then binds the receiver with ``data_spec_map.get(name)``, so three
    writes stayed unattributed with the answer written above them.
    """
    source = (
        "class Core:\n"
        "    def apply(self, name):\n"
        "        data_spec_map: Dict[str, JediFileSpec] = self.load()\n"
        "        data_spec = data_spec_map.get(name)\n"
        "        data_spec.status = 'done'\n"
    )
    _subjects, junctions, _cov = _progress(source, "pandaserver/workflow/workflow_core.py")

    assert junctions[0].subject == "JediFileSpec.status"
    assert junctions[0].attribution == "certain"


def test_a_bare_get_on_a_queue_is_not_a_container_read():
    """``.get()`` with no argument is not an element read, so nothing is claimed.

    The method name is common on objects that are not mappings.  The annotation
    is what carries the risk, not the name, so the guard is cheap -- but a queue
    handed the wrong element type would be a confident wrong answer.
    """
    source = (
        "class Core:\n"
        "    def apply(self):\n"
        "        queue: Dict[str, JediFileSpec] = self.load()\n"
        "        item = queue.get()\n"
        "        item.status = 'done'\n"
    )
    _subjects, junctions, _cov = _progress(source, "pandaserver/workflow/workflow_core.py")

    assert junctions[0].attribution == "unresolved"


# --------------------------------------------------------------------------- #
# progress: what the right-hand side settles
# --------------------------------------------------------------------------- #
#
# The slice used to read literal right-hand sides only, which left the map with
# two states for a write -- resolved or invisible -- when the model has a third
# the SQL slice was already using.  These pin the ladder that fixes it, and in
# particular the one rung that must not slip: ``passthrough`` names a place a
# value lives, so a local variable can never be its target.


def _outcomes(junctions):
    """``(outcome, tier)`` for every branch, in order."""
    return [(b.outcome, b.tier) for j in junctions for b in j.branches]


def test_both_arms_of_an_if_else_are_on_the_map():
    """The shape that showed the slice was wrong -- ``AtlasProdWatchDog.py:350``.

    Reading literals only, this junction said it always produces ``ready``.  The
    other arm is how a reassigned task returns to the status it held, so the
    reasoning could not offer "the restore did not fire" as a candidate at all.
    """
    source = (
        "def doActionForReassign(self, taskSpec):\n"
        "    if taskSpec.oldStatus in ['assigning', 'exhausted', None]:\n"
        "        taskSpec.status = 'ready'\n"
        "    else:\n"
        "        taskSpec.status = taskSpec.oldStatus\n"
    )
    _subjects, junctions, _cov = _progress(source, "pandajedi/jedidog/AtlasProdWatchDog.py")

    assert _outcomes(junctions) == [
        ("ready", 1),
        ("passthrough(JediTaskSpec.oldStatus)", 2),
    ]
    assert junctions[0].branches[1].path_condition == [
        "not (taskSpec.oldStatus in ['assigning', 'exhausted', None])"
    ]


def test_clearing_a_field_is_an_outcome():
    """``oldStatus = None`` decides the field as much as any word does.

    Spelled as the source spells it, which is also how a log line interpolating
    the field reads -- the map is compared against production text.
    """
    source = "def release(self, taskSpec):\n    taskSpec.oldStatus = None\n"
    _subjects, junctions, _cov = _progress(source, "pandajedi/jedidog/AtlasProdWatchDog.py")

    assert _outcomes(junctions) == [("None", 1)]


def test_a_local_is_resolved_by_reaching_definitions():
    """A guarded chain assigning the local carries its conditions to the write.

    The write's own guards come first because reaching it is necessary for any
    outcome; the assignment's guards say which one.
    """
    source = (
        "def refine(self, taskSpec, ok):\n"
        "    if taskSpec.nucleus:\n"
        "        newStatus = 'ready'\n"
        "    else:\n"
        "        newStatus = 'pending'\n"
        "    if ok:\n"
        "        taskSpec.status = newStatus\n"
    )
    _subjects, junctions, _cov = _progress(source, "pandajedi/jedirefine/TaskRefiner.py")

    assert _outcomes(junctions) == [("ready", 1), ("pending", 1)]
    assert [b.path_condition for b in junctions[0].branches] == [
        ["ok", "taskSpec.nucleus"],
        ["ok", "not (taskSpec.nucleus)"],
    ]


def test_an_unresolvable_local_is_a_run_time_outcome_not_a_passthrough():
    """``passthrough(X)`` claims the value lives in X and the walk continues
    there.  A local is a step in a computation, not a place a value lives, so a
    passthrough onto one is an edge whose far end cannot exist -- and the
    reference gate would not catch it, because it excuses a passthrough landing
    off the promoted set as a provenance terminal.
    """
    source = (
        "def refine(self, taskSpec, incoming):\n"
        "    newStatus = compute(incoming)\n"
        "    taskSpec.status = newStatus\n"
    )
    _subjects, junctions, _cov = _progress(source, "pandajedi/jedirefine/TaskRefiner.py")

    assert _outcomes(junctions) == [("runtime(newStatus)", 2)]


def test_a_qualified_field_name_is_a_passthrough():
    """``self.oldStatus = self.status`` is how ``oldStatus`` gets recorded -- the
    edge prune works from, and the first hop out of the flagship symptom."""
    source = (
        "class JediTaskSpec(object):\n"
        "    _attributes = ('jediTaskID', 'status', 'oldStatus')\n"
        "    def setOnHold(self):\n"
        "        self.oldStatus = self.status\n"
    )
    _subjects, junctions, _cov = _progress(source, "pandaserver/taskbuffer/JediTaskSpec.py")

    assert _outcomes(junctions) == [("passthrough(JediTaskSpec.status)", 2)]


def test_an_attribute_of_something_that_is_not_a_spec_is_a_run_time_outcome():
    """``self.status`` in a WatchDog is the WatchDog's own field: a name
    collision, not a place on the map.  Naming it as a passthrough source would
    point the walk at a node no map contains."""
    source = (
        "class AtlasProdWatchDog(object):\n"
        "    def doAction(self, taskSpec):\n"
        "        taskSpec.status = self.status\n"
    )
    _subjects, junctions, _cov = _progress(source, "pandajedi/jedidog/AtlasProdWatchDog.py")

    assert _outcomes(junctions) == [("runtime(self.status)", 2)]


def test_assembled_text_carries_its_own_search_key():
    """Free text is not a state, so there is no outcome to enumerate -- but the
    rendered expression keeps the literal frame, which is what a diagnostic seen
    in production is matched against.

    It is deliberately not copied into ``emits``: that field means the branch
    logs this line, which is true of a diagnostic and false of a computed
    ``lfn``, and no structural reading separates the two.
    """
    source = (
        "class JobSpec(object):\n"
        "    _attributes = ('PandaID', 'jobStatus', 'ddmErrorDiag')\n"
        "    def fail(self, n):\n"
        "        self.ddmErrorDiag = f'failed to get {n} files'\n"
    )
    _subjects, junctions, _cov = _progress(source, "pandaserver/taskbuffer/JobSpec.py")

    assert _outcomes(junctions) == [("runtime(f'failed to get {n} files')", 2)]
    assert junctions[0].branches[0].emits == []


def test_the_return_alias_shape_is_left_to_its_own_slice():
    """``taskSpec.status = self.getFinalTaskStatus(...)`` is resolved one hop up
    into a branch per returned value.  A second reading here would put "decided
    at run time" beside branches that say what the value is -- a weaker claim
    contradicting a stronger one at the same junction.
    """
    source = (
        "class PostProcessorBase(object):\n"
        "    def doPostProcess(self, taskSpec):\n"
        "        taskSpec.status = self.getFinalTaskStatus(taskSpec)\n"
    )
    _subjects, junctions, _cov = _progress(source, "pandajedi/jedipprocess/PostProcessorBase.py")

    assert junctions == []


# --------------------------------------------------------------------------- #
# sql-write: statements, table classes, bind writes
# --------------------------------------------------------------------------- #

# The flagship shape, cut down: one statement writing and another reading
# through the identically named bind, in one function.
_SQL_SOURCE = '''
class TaskModule:
    def updateTaskStatus(self, jediTaskID, taskStatus, broken):
        sqlU = f"UPDATE {panda_config.schemaJEDI}.JEDI_Tasks "
        sqlU += "SET status=:status,oldStatus=:oldStatus "
        sqlU += "WHERE jediTaskID=:jediTaskID "
        sqlL = f"UPDATE {panda_config.schemaJEDI}.JEDI_Tasks "
        sqlL += "SET lockedBy=NULL WHERE status=:status "
        varMap = {}
        varMap[":jediTaskID"] = jediTaskID
        if broken:
            varMap[":status"] = "tobroken"
        else:
            varMap[":status"] = "finishing"
        self.cur.execute(sqlU + comment, varMap)
'''


def _sql_extract(source: str, rel: str = "pandaserver/taskbuffer/db_proxy_mods/task_module.py"):
    modules = [_module(_SPECS, "pandaserver/taskbuffer/Specs.py"), _module(source, rel)]
    attributor = attribution.SpecAttributor(
        progress.spec_attributes(modules), attribution.class_bases(modules)
    )
    conflicts = attributor.learn_table_classes(modules)
    subjects, junctions, coverage, uncovered, _diag, _why, _stops = sqlwrite.extract(
        modules, MAP_ID, VERSION, attributor
    )
    return subjects, junctions, coverage, uncovered, conflicts, attributor


def _sql_unexplained(
    source: str, rel: str = "pandaserver/taskbuffer/db_proxy_mods/task_module.py"
):
    """The columns the slice counted and could not account for, and its coverage.

    Beside :func:`_sql_extract` rather than widening it: nineteen tests unpack
    that one, and none of them are about this.
    """
    modules = [_module(_SPECS, "pandaserver/taskbuffer/Specs.py"), _module(source, rel)]
    attributor = attribution.SpecAttributor(
        progress.spec_attributes(modules), attribution.class_bases(modules)
    )
    attributor.learn_table_classes(modules)
    _s, _j, coverage, _u, _d, unexplained, _stops = sqlwrite.extract(
        modules, MAP_ID, VERSION, attributor
    )
    return unexplained, coverage


_EXECUTEMANY = '''
class TaskModule:
    def bulkUpdate(self, ids):
        sqlU = f"UPDATE {panda_config.schemaJEDI}.JEDI_Tasks "
        sqlU += "SET status=:status "
        sqlU += "WHERE jediTaskID=:jediTaskID "
        varMaps = []
        for jediTaskID in ids:
            varMap = {}
            varMap[":jediTaskID"] = jediTaskID
            varMap[":status"] = "finished"
            varMaps.append(varMap)
        self.cur.executemany(sqlU + comment, varMaps)

    def stagedUpdate(self, ids):
        sqlU = f"UPDATE {panda_config.schemaJEDI}.JEDI_Tasks "
        sqlU += "SET status=:new_status "
        sqlU += "WHERE jediTaskID=:jediTaskID AND status=:old_status "
        varMap = {}
        varMap[":old_status"] = "defined"
        varMap[":new_status"] = "ready"
        varMaps = []
        for jediTaskID in ids:
            tmp_varMap = varMap.copy()
            tmp_varMap[":jediTaskID"] = jediTaskID
            varMaps.append(tmp_varMap)
        self.cur.executemany(sqlU + comment, varMaps)

    def literalUpdate(self, ids):
        sqlU = f"UPDATE {panda_config.schemaJEDI}.JEDI_Tasks "
        sqlU += "SET status=:status "
        sqlU += "WHERE jediTaskID=:jediTaskID "
        varMaps = []
        for jediTaskID in ids:
            varMaps.append({":jediTaskID": jediTaskID, ":status": "running"})
        self.cur.executemany(sqlU + comment, varMaps)
'''


def test_a_varmap_handed_to_executemany_is_read_through_the_maps_appended_to_it():
    # 104 of the corpus's 140 unaccounted-for written columns were this one
    # shape once, and on pin k updateInputFilesStaged_JEDI's staging -> pending
    # had no junction for it.  executemany takes a *sequence* of maps, so the
    # name in the call is never subscripted with a placeholder: each map
    # appended to it holds what it held when it was appended -- one filled in
    # the loop, a copy of one filled before it, or a dict literal.
    _subjects, junctions, coverage, _u, _c, _a = _sql_extract(_EXECUTEMANY)

    outcomes = {
        (junction.owner.rsplit("::", 1)[-1], branch.outcome, tuple(branch.row_precondition))
        for junction in junctions
        for branch in junction.branches
    }
    assert outcomes == {
        ("bulkUpdate", "finished", ()),
        ("stagedUpdate", "ready", ("status=defined",)),
        ("literalUpdate", "running", ()),
    }
    written = [c for c in coverage if c.slice_name == sqlwrite.SLICE_NAME]
    assert [(c.explained, c.candidates) for c in written] == [(3, 3)]


def test_a_placeholder_bound_without_its_colon_binds_it_as_the_cursor_does():
    # hs_scrapers writes r["source"] = url against VALUES (:source), and
    # insertFilesForDataset_JEDI varMap["status"] = "lost" against
    # SET status=:status.  WrappedCursor.execute binds a key without its colon
    # to the placeholder with it, so the value is decided right here; the
    # other column is still one nothing binds.
    unexplained, _coverage = _sql_unexplained('''
class TaskModule:
    def insertRows(self, rows, url):
        sqlI = f"INSERT INTO {panda_config.schemaJEDI}.JEDI_Tasks "
        sqlI += "(status,oldStatus) VALUES (:status,:oldStatus) "
        for r in rows:
            r["status"] = url
            self.cur.execute(sqlI + comment, r)
''')

    assert {row.column: row.reason for row in unexplained} == {
        "oldStatus": sqlwrite.NOTHING_BINDS_IT,
    }

def test_a_junction_spans_the_arms_it_was_built_from_whatever_order_they_were_found():
    """The anchor's span has to hold the outcomes, not the ends of a scan.

    ``scan_binds`` walks the function with ``ast.walk``, which is breadth-first:
    a bind assigned inside an ``if`` is found *after* one assigned below the
    block at the function's own level.  Taking the span from the first and last
    of that order gave seventeen junctions a ``line_end`` above their
    ``line_start`` -- ``resetFileStatusInJEDI`` said 2249..2083 -- and another
    nine a span that simply did not reach their topmost arm.

    Nothing caught it because nothing in production reads ``line_end``:
    ``reading.region_in`` derives the region from the containing function, and
    ``diff.py`` leaves the anchor out of its comparison on purpose.  An
    analyser reading the span is the first use that would believe it.
    """
    _subjects, junctions, *_rest = _sql_extract('''
class TaskModule:
    def resetStatus(self, jediTaskID, lost):
        sqlU = f"UPDATE {panda_config.schemaJEDI}.JEDI_Tasks "
        sqlU += "SET status=:status "
        sqlU += "WHERE jediTaskID=:jediTaskID "
        varMap = {}
        varMap[":jediTaskID"] = jediTaskID
        if lost:
            varMap[":status"] = "lost"
            self.cur.execute(sqlU + comment, varMap)
        varMap[":status"] = "finished"
        self.cur.execute(sqlU + comment, varMap)
''')

    # The fixture declares no spec for the table, so the subject is the table's
    # own -- which is beside the point here and keeps the source to one method.
    (junction,) = junctions
    assert junction.subject == "JEDI_Tasks.status"

    # Discovery order is not line order: the run inside the ``if`` comes
    # second.  Each run reads its own bind -- ``lost`` is overwritten before
    # the second one, and ``finished`` is below the first.
    assert [(b.outcome, b.line) for b in junction.branches] == [
        ("finished", 12),
        ("lost", 10),
    ]
    # The span covers both of them anyway.
    assert (junction.anchor.line_start, junction.anchor.line_end) == (10, 12)


def test_a_junction_spans_the_arms_of_every_text_a_call_runs():
    """One call runs two texts of a statement, each with its own branch's
    status (``sql.bound_values`` pairs a text with its own branch's binds), so
    the first text read is not every arm: the span is over all of them."""
    _subjects, junctions, *_rest = _sql_extract('''
class TaskModule:
    def setStatus(self, jediTaskID, taskSpec):
        sqlU = f"UPDATE {panda_config.schemaJEDI}.JEDI_Tasks SET status=:status"
        if taskSpec is not None:
            sqlU += ",splitRule=:splitRule"
        sqlU += " WHERE jediTaskID=:jediTaskID "
        varMap = {}
        varMap[":jediTaskID"] = jediTaskID
        if taskSpec is not None:
            varMap[":status"] = "pending"
            varMap[":splitRule"] = taskSpec.splitRule
        else:
            varMap[":status"] = "ready"
        self.cur.execute(sqlU + comment, varMap)
''')

    (junction,) = [one for one in junctions if one.subject == "JEDI_Tasks.status"]
    assert sorted((b.outcome, b.line) for b in junction.branches) == [("pending", 11), ("ready", 14)]
    assert (junction.anchor.line_start, junction.anchor.line_end) == (11, 14)


def test_statement_is_read_from_its_concatenation():
    """A statement is built by ``=`` then a run of ``+=``, interleaved with others."""
    module = _module(_SQL_SOURCE, "x.py")
    func = next(
        n for n in ast.walk(module.tree)
        if isinstance(n, ast.FunctionDef) and n.name == "updateTaskStatus"
    )
    (text,) = _held(func, "sqlU")

    assert "UPDATE {}.JEDI_Tasks" in text
    assert "SET status=:status" in text
    # The other statement being built in the same function must not bleed in.
    assert "lockedBy" not in text


def test_a_bind_in_a_where_clause_is_not_a_write():
    """``:status`` is a write in one statement and a predicate in another.

    Both are in the same function, so the bind key alone cannot tell them
    apart; only the ``SET`` clause can.
    """
    writes = sqltext.writes("UPDATE {}.JEDI_Tasks SET lockedBy=NULL WHERE status=:status ")

    assert set(writes[0].columns) == {"lockedBy"}


def test_inline_values_are_kept_without_a_bind():
    """``stateChangeTime=CURRENT_DATE`` is a write whose value is in the statement."""
    writes = sqltext.writes("UPDATE {}.JEDI_Tasks SET status=:status,stateChangeTime=CURRENT_DATE ")

    assert writes[0].columns["status"] == sqltext.ColumnValue(kind="bind", text=":status")
    assert writes[0].columns["stateChangeTime"] == sqltext.ColumnValue(
        kind="expression", text="CURRENT_DATE"
    )


def test_a_column_copied_from_another_column_is_told_from_a_literal():
    """``SET status=oldStatus`` is a passthrough; ``SET status='ready'`` is not.

    The distinction has to survive the fact that both are bare-looking text on
    the right of an ``=``.  It is what turns "a task left pending" into an edge
    the backward walk can follow.
    """
    writes = sqltext.writes("UPDATE {}.JEDI_Tasks SET status=oldStatus,oldStatus=NULL ")

    assert writes[0].columns["status"] == sqltext.ColumnValue(kind="column", text="oldStatus")
    # ``NULL`` is bare-looking too, and it is neither a column nor unsettled:
    # the statement says exactly what the field will hold afterwards.
    assert writes[0].columns["oldStatus"] == sqltext.ColumnValue(kind="literal", text="None")

    literal = sqltext.writes("UPDATE ATLAS_PANDA.filesTable4 SET status='ready' ")
    assert literal[0].columns["status"] == sqltext.ColumnValue(kind="literal", text="ready")


def test_an_update_is_read_through_a_hint_and_a_table_alias():
    """``UPDATE /*+ index(tab ...) */ ... filesTable4 tab SET ...`` is one statement.

    Both forms occur on ``filesTable4`` and neither is exotic; without the
    allowance the statement reads as no statement at all, which is silent.
    """
    writes = sqltext.writes(
        "UPDATE /*+ index(tab FILESTABLE4_IDX) */ ATLAS_PANDA.filesTable4 tab "
        "SET status='ready' WHERE PandaID=:PandaID "
    )

    assert len(writes) == 1
    assert writes[0].table == "filesTable4"
    assert writes[0].columns["status"] == sqltext.ColumnValue(kind="literal", text="ready")


def test_a_statement_chosen_through_another_variable_is_still_read():
    """``sql = sqlTU`` then ``execute(sql, ...)`` -- and the alias carries the reason.

    ``reactivatePendingTasks_JEDI`` builds a timeout statement and a release
    statement unconditionally and picks between them in an ``if``/``else``, so
    reading only the statement loses which one ran and why.
    """
    source = (
        "def reactivate(self):\n"
        "    sqlTO = f'UPDATE {schema}.JEDI_Tasks '\n"
        "    sqlTO += 'SET status=:newStatus '\n"
        "    sqlTU = f'UPDATE {schema}.JEDI_Tasks '\n"
        "    sqlTU += 'SET status=oldStatus '\n"
        "    if timeout:\n"
        "        sql = sqlTO\n"
        "    else:\n"
        "        sql = sqlTU\n"
        "    self.cur.execute(sql + comment, varMap)\n"
    )
    func = ast.parse(source).body[0]

    statements = [w for text in _held(func, "sql") for w in sqltext.writes(text)]

    assert [w.columns["status"].text for w in statements] == [":newStatus", "oldStatus"]


def test_a_statement_assembled_across_an_if_else_is_two_statements():
    """Folding the arms together builds a statement that cannot exist::

        UPDATE ... SET status=:status,SET status=oldStatus,...

    and the damage is not that it reads oddly: the second ``SET`` overwrote the
    first in the column map, so the bind arm of the write vanished.  That is the
    same shape as the attribute-slice if/else whose second arm was missing --
    one arm on the map, one lost.
    """
    source = (
        "def f(self, newTaskStatus):\n"
        "    sqlTU = 'UPDATE ATLAS_PANDA.JEDI_Tasks '\n"
        "    if newTaskStatus != 'dummy':\n"
        "        sqlTU += 'SET status=:status,'\n"
        "    else:\n"
        "        sqlTU += 'SET status=oldStatus,'\n"
        "    sqlTU += 'modificationTime=CURRENT_DATE WHERE jediTaskID=:jediTaskID '\n"
    )
    func = _func(source)

    statements = [w for text in _held(func, "sqlTU") for w in sqltext.writes(text)]

    assert [w.columns["status"].kind for w in statements] == ["bind", "column"]
    assert [w.columns["status"].text for w in statements] == [":status", "oldStatus"]
    # The head is never dropped -- it is what the statement is.
    assert all(w.table == "JEDI_Tasks" for w in statements)


def test_two_independent_if_elses_give_every_combination():
    """``getTasksToExecCommand_JEDI`` picks the status arm and the oldStatus arm
    on unrelated tests, so all four statements are reachable: every statement
    the call runs (:func:`sql.every_statement`), not only a covering set."""
    source = (
        "def f(self, newTaskStatus, taskStatus):\n"
        "    sqlTU = 'UPDATE ATLAS_PANDA.JEDI_Tasks '\n"
        "    if newTaskStatus != 'dummy':\n"
        "        sqlTU += 'SET status=:status,'\n"
        "    else:\n"
        "        sqlTU += 'SET status=oldStatus,'\n"
        "    if taskStatus in ['paused']:\n"
        "        sqlTU += 'oldStatus=NULL,'\n"
        "    else:\n"
        "        sqlTU += 'oldStatus=status,'\n"
        "    sqlTU += 'modificationTime=CURRENT_DATE WHERE jediTaskID=:jediTaskID '\n"
        "    self.cur.execute(sqlTU + comment, varMap)\n"
    )
    func = _func(source)

    statements = [w for text in sql.every_statement(func, 12) for w in sqltext.writes(text)]

    assert sorted((w.columns["status"].text, w.columns["oldStatus"].text) for w in statements) == [
        (":status", "None"),
        (":status", "status"),
        ("oldStatus", "None"),
        ("oldStatus", "status"),
    ]


def test_a_loop_that_runs_nothing_builds_one_statement_over_its_tables():
    """Nothing in the loop runs a statement, so the loop is writing one, pass by pass.

    ``getQueuedPandaIDsWithTask_JEDI``.  Split per table, every pass kept the
    ``UNION ALL`` -- ``if sqlP != ""`` reads what earlier passes appended --
    and each read as a statement no parser takes.  A loop that runs the
    statement on every pass is still one statement per table.
    """
    func = _func(
        "def f(self):\n"
        "    tables = ['ATLAS_PANDA.jobsDefined4', 'ATLAS_PANDA.jobsActive4']\n"
        "    sqlP = ''\n"
        "    for tableName in tables:\n"
        "        if sqlP != '':\n"
        "            sqlP += 'UNION ALL '\n"
        "        sqlP += f'SELECT PandaID FROM {tableName} WHERE jediTaskID=:t '\n"
        "        if tableName == 'ATLAS_PANDA.jobsActive4':\n"
        "            sqlP += 'AND jobStatus<>:st '\n"
        "    self.cur.execute(sqlP, varMap)\n"
        "    for table in tables:\n"
        "        sqlC = f'SELECT COUNT(*) FROM {table} WHERE x=:x '\n"
        "        self.cur.execute(sqlC, varMap)\n"
    )

    assert [run.sql for run in sql.executions(func)] == [
        "SELECT PandaID FROM ATLAS_PANDA.jobsDefined4 WHERE jediTaskID=:t "
        "UNION ALL SELECT PandaID FROM ATLAS_PANDA.jobsActive4 WHERE jediTaskID=:t AND jobStatus<>:st ",
        "SELECT COUNT(*) FROM ATLAS_PANDA.jobsDefined4 WHERE x=:x ",
        "SELECT COUNT(*) FROM ATLAS_PANDA.jobsActive4 WHERE x=:x ",
    ]


def test_a_statement_executed_in_three_pieces_is_read_whole():
    """``execute(sqlU + sql + comment, varMap)`` -- the ``WHERE`` is a name too.

    ``updateTask_JEDI`` and ``updateJobStatus`` both build the ``SET`` half and
    the ``WHERE`` half under separate names and join them at the call.  Reading
    only the leftmost operand dropped the statement entirely, taking the
    ``jobStatus`` write and every predicate with it.
    """
    source = (
        "def f(self):\n"
        "    sqlU = 'UPDATE ATLAS_PANDA.JEDI_Tasks SET status=:status '\n"
        "    sqlW = 'WHERE jediTaskID=:jediTaskID AND status IN (:old_1) '\n"
        "    self.cur.execute(sqlU + sqlW + comment, varMap)\n"
    )
    func = _func(source)

    texts = [run.sql for run in sql.executions(func)]

    assert len(texts) == 1
    assert "SET status=:status" in texts[0]
    assert "status IN (:old_1)" in texts[0]


def test_a_fragment_with_no_text_of_its_own_is_named_rather_than_dropped():
    """``sqlP += sql_where_clause`` -- ``getJobs``, and the whole pilot request.

    The reassembly kept only what rendered, so the fragment holding every
    condition the job dispatcher selects on vanished and the statement read as
    a bare ``SELECT ... FROM jobsActive4``.  Nothing was wrong with the
    statement that was left, which is why it went unnoticed for the map's
    whole life: a truncated ``WHERE`` looks exactly like no ``WHERE``.
    """
    source = (
        "def f(self):\n"
        "    sqlP = 'SELECT PandaID FROM ATLAS_PANDA.jobsActive4 tab '\n"
        "    sqlP += sql_where_clause\n"
        "    self.cur.execute(sqlP + comment, getValMap)\n"
    )
    func = _func(source)

    stops = {stop for run in sql.executions(func) for stop in run.unresolved}

    assert {stop.source for stop in stops} == {"sql_where_clause"}
    assert {stop.reason for stop in stops} == {sql.FRAGMENT_NOT_RENDERED}
    assert {stop.line for stop in stops} == {3}


def test_a_where_whose_binds_are_interpolated_names_the_term_it_cannot_fill():
    """``AND jobStatus IN ({})`` -- ``lockJobsForReassign``.

    The column is readable and the placeholders are not, because the code
    spells them with a helper over a list it was handed.  The term is what the
    consumer needs: told only "a hole", it knows something is wrong; told
    ``jobStatus IN ({})`` it knows which argument at the call site to read.
    """
    source = (
        "def f(self):\n"
        "    sql = 'SELECT PandaID FROM ATLAS_PANDA.jobsActive4 '\n"
        "    sql += 'WHERE modificationTime<:modificationTime '\n"
        "    sql += f'AND jobStatus IN ({names}) '\n"
        "    self.cur.execute(sql + comment, varMap)\n"
    )
    func = _func(source)

    stops = {stop for run in sql.executions(func) for stop in run.unresolved}

    assert [stop.source for stop in stops] == ["jobStatus IN ({})"]
    assert {stop.reason for stop in stops} == {sql.BINDS_ARE_A_HOLE}


def test_a_schema_interpolated_into_the_where_is_not_something_it_stopped_on():
    """``datasetID IN (SELECT datasetID FROM {}.JEDI_Datasets ...)``.

    The reading blanks the schema everywhere on purpose, and a subquery puts
    one inside the ``WHERE``.  Counting it would report 17 statements as
    unreadable that are read in full, and a census with false entries is worse
    than none: it is what the next round would be planned from.
    """
    source = (
        "def f(self):\n"
        "    sql = f'SELECT lfn FROM {schema}.JEDI_Dataset_Contents '\n"
        "    sql += f'WHERE jediTaskID=:jediTaskID AND datasetID IN "
        "(SELECT datasetID FROM {schema}.JEDI_Datasets WHERE jediTaskID=:jediTaskID) '\n"
        "    self.cur.execute(sql + comment, varMap)\n"
    )
    func = _func(source)

    assert [run.unresolved for run in sql.executions(func)] == [()]


def _the_execute(func):
    return next(
        node
        for node in ast.walk(func)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "execute"
    )


def test_a_statement_returning_into_a_bind_writes_that_key_back():
    """``RETURNING row_ID INTO :newRowID`` -- the one key the cursor writes.

    A reader that cannot see inside ``execute`` takes the map as rewritten
    whole; the statement says which key is, spelt as the map spells it.
    """
    func = _func(
        "def f(self):\n"
        "    sql = 'INSERT INTO ATLAS_PANDA.jobParamsTable (PandaID) VALUES (:PandaID) '\n"
        "    sql += 'RETURNING row_ID INTO :newRowID '\n"
        "    varMap[':newRowID'] = self.cur.var(varNUMBER)\n"
        "    self.cur.execute(sql + comment, varMap)\n"
    )

    assert sql.written_back(func, _the_execute(func)) == (":newRowID",)


def test_a_statement_with_no_returning_writes_nothing_back_past_its_schema_hole():
    """``kickExhaustedTasks_JEDI``'s ``UPDATE``: the arm's ``:newStatus`` is what was sent.

    The schema is a hole in every statement read, and a hole there cannot
    hold a clause; read as "cannot say", it would leave nearly
    every map in the corpus rewritten.
    """
    func = _func(
        "def f(self, flag):\n"
        "    sql = f'UPDATE {schema}.JEDI_Tasks SET status=:newStatus WHERE jediTaskID=:jediTaskID '\n"
        "    if flag:\n"
        "        varMap[':newStatus'] = 'exhausted'\n"
        "    else:\n"
        "        varMap[':newStatus'] = 'finishing'\n"
        "    self.cur.execute(sql + comment, varMap)\n"
    )

    assert sql.written_back(func, _the_execute(func)) == ()


def test_a_statement_missing_a_piece_cannot_say_what_it_writes_back():
    """``sqlP += sql_where_clause``: the piece nobody can read could hold the clause."""
    func = _func(
        "def f(self):\n"
        "    sqlP = 'SELECT PandaID FROM ATLAS_PANDA.jobsActive4 tab '\n"
        "    sqlP += sql_where_clause\n"
        "    self.cur.execute(sqlP + comment, getValMap)\n"
    )

    assert sql.written_back(func, _the_execute(func)) is None


def test_a_hole_at_the_end_of_an_update_cannot_say_what_it_writes_back():
    """``updateDataset``'s ``AND {withCriteria}``: the caller's condition ends the statement.

    ``RETURNING ... INTO`` is a statement's last clause, and nothing stands
    after this hole.  Read past it, the answer said the map was not rewritten
    on no evidence at all.
    """
    func = _func(
        "def f(self, datasets, withCriteria):\n"
        "    sql1 = 'UPDATE ATLAS_PANDA.Datasets SET status=:status WHERE vuid=:vuid'\n"
        "    if withCriteria != '':\n"
        "        sql1 += f' AND {withCriteria}'\n"
        "    self.cur.execute(sql1 + comment, varMap)\n"
    )

    assert sql.written_back(func, _the_execute(func)) is None


@pytest.mark.parametrize(
    "source",
    [
        # updateTask_JEDI: a bind named after each key it is handed; the end is a name's piece
        "def f(self, criteria):\n"
        "    sql = f'UPDATE {schema}.JEDI_Tasks SET status=:status WHERE '\n"
        "    for tmpKey in criteria:\n"
        "        sql += f'{tmpKey}=:cr_{tmpKey} AND '\n"
        "    sql = sql[:-4]\n"
        "    self.cur.execute(sql + comment, varMap)\n",
        # an IN list: the hole is inside the parentheses
        "def f(self, ids):\n"
        "    sql = f'UPDATE ATLAS_PANDA.jedi_tasks SET gshare=:gshare WHERE jeditaskid IN ({ids})'\n"
        "    self.cur.execute(sql + comment, varMap)\n",
        # the SET list: a WHERE comes after it
        "def f(self, changes):\n"
        "    sql = f'UPDATE ATLAS_PANDA.filesTable4 SET {changes} WHERE row_ID=:row_ID'\n"
        "    self.cur.execute(sql + comment, varMap)\n",
        # a SELECT returns into nothing, whatever ends it
        "def f(self, extra):\n"
        "    sql = f'SELECT PandaID FROM ATLAS_PANDA.jobsActive4 WHERE {extra}'\n"
        "    self.cur.execute(sql + comment, varMap)\n",
    ],
    ids=["bind-name", "in-list", "set-list", "select"],
)
def test_a_hole_that_cannot_end_the_statement_is_read_past(source):
    """The holes that cannot hold the clause: inside, before more text, a bind's name, a ``SELECT``."""
    func = _func(source)

    assert sql.written_back(func, _the_execute(func)) == ()


def test_a_block_with_a_hole_anywhere_cannot_say_what_it_writes_back():
    """A PL/SQL block takes whole statements: any of its holes could return into a bind."""
    func = _func(
        "def f(self, body):\n"
        "    sql = f'BEGIN {body}; COMMIT; END;'\n"
        "    self.cur.execute(sql, varMap)\n"
    )

    assert sql.written_back(func, _the_execute(func)) is None


def test_a_call_running_either_of_two_statements_writes_back_what_either_returns():
    """One arm inserts and returns the new id, the other updates in place: the key may be written."""
    func = _func(
        "def f(self, new):\n"
        "    if new:\n"
        "        sql = 'INSERT INTO ATLAS_PANDA.T (a) VALUES (:a) RETURNING PandaID INTO :newPandaID '\n"
        "    else:\n"
        "        sql = 'UPDATE ATLAS_PANDA.T SET a=:a WHERE PandaID=:PandaID '\n"
        "    self.cur.execute(sql + comment, varMap)\n"
    )

    assert sql.written_back(func, _the_execute(func)) == (":newPandaID",)


def test_written_back_is_found_from_a_file_and_the_line_of_the_call(tmp_path):
    """The reader holds a file and a line, not a tree."""
    module = tmp_path / "m.py"
    module.write_text(
        "def f(self):\n"
        "    sql = 'INSERT INTO ATLAS_PANDA.T (a) VALUES (:a) RETURNING row_ID INTO :newRowID '\n"
        "    self.cur.execute(sql + comment, varMap)\n"
        "    return varMap[':newRowID']\n"
    )

    assert sql.written_back_at(module, 3) == (":newRowID",)
    assert sql.written_back_at(module, 2) is None


def test_written_back_from_a_file_reads_through_the_calls_the_build_reads_through(tmp_path):
    """``sql += JobSpec.bindValuesExpression()`` is a ``VALUES`` list only in the callee.

    Read without the corpus's calls it is a piece with no text of its own,
    which could hold the clause: 23 ``INSERT``s the build reads as writing
    nothing back answered "cannot say" to a reader holding a file and a line.
    """
    proxy = tmp_path / "pandaserver" / "taskbuffer" / "db_proxy_mods" / "job_module.py"
    proxy.parent.mkdir(parents=True)
    (tmp_path / "pandaserver" / "taskbuffer" / "JobSpec.py").write_text(
        "class JobSpec:\n"
        "    attributes = ('PandaID', 'jobStatus')\n"
        "\n"
        "    @classmethod\n"
        "    def bindValuesExpression(cls):\n"
        "        values_str = ','.join([f':{attr}' for attr in cls.attributes])\n"
        "        return f'VALUES({values_str})'\n"
    )
    proxy.write_text(
        "from pandaserver.taskbuffer.JobSpec import JobSpec\n"
        "\n"
        "\n"
        "class JobModule:\n"
        "    def insert(self, job):\n"
        "        sql = 'INSERT INTO ATLAS_PANDA.jobsDefined4 (PandaID,jobStatus) '\n"
        "        sql += JobSpec.bindValuesExpression()\n"
        "        self.cur.execute(sql + comment, job.valuesMap())\n"
    )
    func = _func(proxy.read_text(), "insert")

    assert sql.written_back(func, _the_execute(func)) is None
    assert sql.written_back_at(proxy, 8) == ()
    # the file changes under the reader: the calls are read again, not left on the old tree
    proxy.write_text("\n" + proxy.read_text())
    assert sql.written_back_at(proxy, 9) == ()


def test_a_function_is_read_once_for_all_the_calls_asked_about(monkeypatch):
    """``archiveJob``'s 27 calls read its statements 27 times: 5.0 seconds, where once is enough."""
    func = _func(
        "def f(self):\n"
        "    sql = 'UPDATE ATLAS_PANDA.T SET a=:a WHERE b=:b '\n"
        "    self.cur.execute(sql + comment, varMap)\n"
        "    sqlI = 'INSERT INTO ATLAS_PANDA.T (a) VALUES (:a) RETURNING row_ID INTO :newRowID '\n"
        "    self.cur.execute(sqlI + comment, varMap)\n"
    )
    first, second = [
        node
        for node in ast.walk(func)
        if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "execute"
    ]
    reads = []
    real = sql._executions_read
    monkeypatch.setattr(sql, "_executions_read", lambda one: reads.append(one) or real(one))

    assert sql.written_back(func, first) == ()
    assert sql.written_back(func, second) == (":newRowID",)
    assert reads == [func]


def test_a_column_the_statement_interpolates_is_its_own_reason():
    """``WHERE {}=:{}`` -- ``queryDatasetWithMap`` builds the column from a key.

    A different answer from a hole where the values go, and the request asked
    for it by name: the ten fragments whose column is left to the caller are
    counted from the other end by ``key_column_census``, and this is where the
    same statements say so at the point they are read.
    """
    source = (
        "def f(self):\n"
        "    sql = 'SELECT vuid FROM ATLAS_PANDA.Datasets '\n"
        "    sql += f'WHERE {key}=:{key} '\n"
        "    self.cur.execute(sql + comment, varMap)\n"
    )
    func = _func(source)

    stops = {stop for run in sql.executions(func) for stop in run.unresolved}

    assert [stop.source for stop in stops] == ["{}=:{}"]
    assert {stop.reason for stop in stops} == {sql.COLUMN_IS_A_HOLE}


def test_a_column_a_loops_passes_interpolate_is_named_as_a_pass_wrote_it():
    """``getDatasetAttributesWithMap_JEDI`` appends ``AND {key}=:{key}`` per key after a term of its own.  Run, the
    statement holds a ``{}`` where the passes are, glued to that term, and the stop named the term as unread; the
    stop names what a pass wrote, as the reassembly did."""
    source = (
        "def f(self, criteria):\n"
        "    sql = 'SELECT vuid FROM ATLAS_PANDA.Datasets WHERE vuid=:vuid '\n"
        "    for key in criteria:\n"
        "        sql += f'AND {key}=:{key} '\n"
        "    self.cur.execute(sql + comment, varMap)\n"
    )
    func = _func(source)

    stops = {stop for run in sql.executions(func) for stop in run.unresolved}

    assert [(stop.source, stop.reason) for stop in stops] == [("{}=:{}", sql.COLUMN_IS_A_HOLE)]


def test_a_row_precondition_is_stored_with_the_value_its_bind_holds():
    """``jobStatus=:oldJobStatus`` is what the statement says; ``activated`` is
    what it means, and only the second can be compared with what a reader
    selects on.

    ``getJobs`` is the case: the one query that moves a job from ``activated``
    to ``sent``, holding the guard and unable to say which value it guards.
    """
    source = (
        "def f(self):\n"
        "    sqlJ = 'UPDATE ATLAS_PANDA.jobsActive4 SET jobStatus=:newJobStatus'\n"
        "    sqlJ += ' WHERE PandaID=:PandaID AND jobStatus=:oldJobStatus'\n"
        "    varMap = {}\n"
        "    varMap[':newJobStatus'] = 'sent'\n"
        "    varMap[':oldJobStatus'] = 'activated'\n"
        "    self.cur.execute(sqlJ + comment, varMap)\n"
    )
    func = _func(source)
    (run,) = sql.executions(func)
    (statement,) = sqltext.writes(run.sql)

    settled, done = sql.settled_precondition(
        statement.preconditions["jobStatus"], func, run.varmap, run.window, lambda v, f: []
    )

    assert done is False, "nothing settles when the resolver answers nothing"
    settled, done = sql.settled_precondition(
        statement.preconditions["jobStatus"],
        func,
        run.varmap,
        run.window,
        lambda v, f: [v.value] if isinstance(v, ast.Constant) else [],
    )
    assert (settled, done) == ("jobStatus=activated", True)


def test_a_row_precondition_whose_bind_arrives_from_outside_keeps_its_bind():
    """Returned unchanged, and that is the honest answer rather than a failure.

    ``jobStatus=:oldStatus`` still says truly that this column is tested; what
    it cannot say is which value. The caller is told which of the two it got
    so it can report the second as somewhere the reading stopped.
    """
    source = (
        "def f(self, oldStatus):\n"
        "    sqlJ = 'UPDATE ATLAS_PANDA.jobsActive4 SET jobStatus=:newJobStatus'\n"
        "    sqlJ += ' WHERE jobStatus=:oldStatus'\n"
        "    varMap = {}\n"
        "    varMap[':newJobStatus'] = 'sent'\n"
        "    varMap[':oldStatus'] = oldStatus\n"
        "    self.cur.execute(sqlJ + comment, varMap)\n"
    )
    func = _func(source)
    (run,) = sql.executions(func)
    (statement,) = sqltext.writes(run.sql)

    assert sql.settled_precondition(
        statement.preconditions["jobStatus"], func, run.varmap, run.window, lambda v, f: []
    ) == ("jobStatus=:oldStatus", False)


@pytest.mark.parametrize(
    "where, binds, expected",
    [
        # A list of binds, each a literal (insertNewJob 3302, resetDefinedJob 412).
        ("jobStatus IN (:old1,:old2)", ["varMap[':old1'] = 'activated'", "varMap[':old2'] = 'throttled'"],
         "jobStatus IN (activated,throttled)"),
        # A test negated outright (updateJobStatus 242, updateUnmergedDatasets 1834).
        ("NOT jobStatus=:old1", ["varMap[':old1'] = 'holding'"], "jobStatus<>holding"),
        ("NOT jobStatus IN (:old1,:old2)", ["varMap[':old1'] = 'activated'", "varMap[':old2'] = 'throttled'"],
         "jobStatus NOT IN (activated,throttled)"),
        # One bind of the list nothing here fills: the others are not all the row may hold.
        ("jobStatus IN (:old1,:old2)", ["varMap[':old1'] = 'activated'", "varMap[':old2'] = oldStatus"],
         "jobStatus IN (:old1,:old2)"),
        # One assignment of the bind nothing settles beside one that does.
        ("jobStatus=:old1",
         ["if first:", "    varMap[':old1'] = 'activated'", "else:", "    varMap[':old1'] = oldStatus"],
         "jobStatus=:old1"),
        # The status read off a row, bound under a test of it (getTasksToBeFinished_JEDI 1705).
        ("jobStatus=:old1", ["if oldStatus == 'scouted':", "    varMap[':old1'] = oldStatus"], "jobStatus=scouted"),
        ("jobStatus=:old1", ["if oldStatus in ('scouted', 'running') and first:", "    varMap[':old1'] = oldStatus"],
         "jobStatus IN (running,scouted)"),
        # Bound again under the test: the test says nothing of it.
        ("jobStatus=:old1",
         ["if oldStatus == 'scouted':", "    oldStatus = self.get()", "    varMap[':old1'] = oldStatus"],
         "jobStatus=:old1"),
    ],
)
def test_a_row_precondition_settles_each_shape_a_bind_takes(where, binds, expected):
    """The guards whose values are in Python, whatever shape the statement spells them in -- a list of binds, a
    negation -- and a bind holding what a test above it says; and none settled from part of what it may hold."""
    lines = "".join(f"    {line}\n" for line in binds)
    source = (
        "def f(self, first, oldStatus):\n"
        "    sqlJ = 'UPDATE ATLAS_PANDA.jobsActive4 SET jobStatus=:newJobStatus'\n"
        f"    sqlJ += ' WHERE PandaID=:PandaID AND {where}'\n"
        "    varMap = {}\n"
        "    varMap[':newJobStatus'] = 'sent'\n"
        f"{lines}"
        "    self.cur.execute(sqlJ + comment, varMap)\n"
    )
    func = _func(source)
    (run,) = sql.executions(func)
    (statement,) = sqltext.writes(run.sql)

    settled, done = sql.settled_precondition(
        statement.preconditions["jobStatus"], func, run.varmap, run.window, values.resolver(), at=run
    )

    assert (settled, done) == (expected, ":" not in expected)


_HANDED_GUARDS = {
    "pandaserver/taskbuffer/EventServiceUtils.py": "ST_ready = 0\nST_done = 3\n",
    "pandaserver/taskbuffer/JediTaskSpec.py": (
        "class JediTaskSpec(object):\n"
        "    @classmethod\n"
        "    def statusForJobGenerator(cls):\n"
        "        return ['ready', 'running']\n"
    ),
    "pandaserver/taskbuffer/db_proxy_mods/event_module.py": (
        "from pandaserver.taskbuffer import EventServiceUtils\n"
        "from pandaserver.taskbuffer.JediTaskSpec import JediTaskSpec\n"
        "\n"
        "\n"
        "class EventModule:\n"
        "    def finishEvents(self, pandaID):\n"
        "        sql = 'UPDATE ATLAS_PANDA.JEDI_Events SET status=:newStatus "
        "WHERE PandaID=:PandaID AND status=:oldStatus '\n"
        "        varMap = {}\n"
        "        varMap[':PandaID'] = pandaID\n"
        "        varMap[':newStatus'] = EventServiceUtils.ST_done\n"
        "        varMap[':oldStatus'] = EventServiceUtils.ST_ready\n"
        "        self.cur.execute(sql + comment, varMap)\n"
        "\n"
        "    def setStatus(self, jediTaskID, oldStatus):\n"
        "        sql = 'UPDATE ATLAS_PANDA.JEDI_Tasks SET status=:newStatus "
        "WHERE jediTaskID=:jediTaskID AND status=:oldStatus '\n"
        "        varMap = {':jediTaskID': jediTaskID, ':newStatus': 'running', ':oldStatus': oldStatus}\n"
        "        self.cur.execute(sql + comment, varMap)\n"
        "\n"
        "    def kick(self, jediTaskID):\n"
        "        self.setStatus(jediTaskID, 'ready')\n"
        "\n"
        "    def setRead(self, jediTaskID, oldStatus):\n"
        "        sql = 'UPDATE ATLAS_PANDA.JEDI_Tasks SET status=:newStatus "
        "WHERE jediTaskID=:jediTaskID AND status=:oldStatus '\n"
        "        varMap = {':jediTaskID': jediTaskID, ':newStatus': 'running', ':oldStatus': oldStatus}\n"
        "        self.cur.execute(sql + comment, varMap)\n"
        "\n"
        "    def reread(self, jediTaskID):\n"
        "        self.cur.execute('SELECT status FROM ATLAS_PANDA.JEDI_Tasks WHERE jediTaskID=:jediTaskID', "
        "{':jediTaskID': jediTaskID})\n"
        "        (status,) = self.cur.fetchone()\n"
        "        self.setRead(jediTaskID, status)\n"
        "\n"
        "    def generate(self, jediTaskID):\n"
        "        names, old_map = get_sql_IN_bind_variables("
        "JediTaskSpec.statusForJobGenerator() + ['pending'], prefix=':old_')\n"
        "        sql = f'UPDATE ATLAS_PANDA.JEDI_Tasks SET status=:newStatus "
        "WHERE jediTaskID=:jediTaskID AND status IN ({names}) '\n"
        "        varMap = {':jediTaskID': jediTaskID, ':newStatus': 'scouting'}\n"
        "        varMap.update(old_map)\n"
        "        self.cur.execute(sql + comment, varMap)\n"
    ),
}


@pytest.mark.parametrize(
    "function, expected",
    [
        # A module's constant, by the name the event service binds every status with.
        ("finishEvents", "status=0"),
        # A parameter its one caller hands a literal.
        ("setStatus", "status=ready"),
        # A parameter its caller hands what it read off the row: a compare-and-set, no value to say.
        ("setRead", "status=:oldStatus"),
        # A helper's IN list of a method's literal list and one more (JobGenerator.runImpl's oldStatus).
        ("generate", "status IN (pending,ready,running)"),
    ],
)
def test_a_row_precondition_settles_its_binds_as_the_read_side_does(tmp_path, function, expected):
    """The write side's guards read their binds the way the read side reads its own (values.bind_settler): a
    constant of a module, what a parameter's callers hand it -- and nothing where one of those is read off a row."""
    modules, calls, _record = _keyed_corpus(tmp_path, _HANDED_GUARDS)
    func = next(
        node
        for module in modules
        for node in ast.walk(module.tree)
        if isinstance(node, ast.FunctionDef) and node.name == function
    )
    with sql.across_calls(calls):
        (run,) = sql.executions(func)
        (statement,) = sqltext.writes(run.reading)
        settled, done = sql.settled_precondition(
            statement.preconditions["status"], func, run.varmap, run.window, values.bind_settler(calls), at=run
        )
        assert (settled, done) == (expected, ":" not in expected)
        if function == "finishEvents":
            # The value the write gives the row reads in the same spelling, by the resolver both sides share.
            bound = sql.bound_values(func, run.varmap, ":newStatus", run.window, at=run)
            assert [values.resolver()(bind.value, func) for bind in bound] == [["3"]]


@pytest.mark.parametrize("guarded_first", [False, True])
def test_a_row_precondition_is_what_every_statement_writing_the_value_requires(guarded_first):
    """``killJob`` binds ``tobekilled`` once and runs one of two statements:
    a force kill guarding nothing, or one that skips a job already being killed.

    The value lands under either, so the branch holds the terms both give --
    none.  Kept from the first statement read, it said the guard when the
    guarded statement came first, and nothing when the other did.
    """
    plain = "'UPDATE ATLAS_PANDA.ddm_endpoint SET blacklisted=:blacklisted WHERE rse=:rse '"
    guarded = (
        "'UPDATE ATLAS_PANDA.ddm_endpoint SET blacklisted=:blacklisted WHERE rse=:rse "
        "AND blacklisted=:old '"
    )
    first, second = (guarded, plain) if guarded_first else (plain, guarded)
    source = (
        "class M:\n"
        "    def f(self, rse, force):\n"
        f"        sqlA = {first}\n"
        f"        sqlB = {second}\n"
        "        varMap = {}\n"
        "        varMap[':rse'] = rse\n"
        "        varMap[':blacklisted'] = 'Y'\n"
        "        varMap[':old'] = 'N'\n"
        "        if force:\n"
        "            self.cur.execute(sqlA + comment, varMap)\n"
        "        else:\n"
        "            self.cur.execute(sqlB + comment, varMap)\n"
    )
    _s, junctions, _c, _u, _conf, _a = _sql_extract(source)
    (branch,) = junctions[0].branches

    assert branch.outcome == "Y"
    assert branch.row_precondition == []


def test_a_guard_testing_two_written_columns_is_one_term():
    """``request_lock`` takes the lock where nobody holds it or the holder's time is up -- ``(locked_by IS NULL OR
    locked_by=:locked_by OR lock_time < :min_lock_time)`` -- and the statement files that one test under each
    column it names.  Read column by column, a branch held it twice."""
    source = (
        "class M:\n"
        "    def f(self, rse, me, now, limit):\n"
        "        sql = 'UPDATE ATLAS_PANDA.ddm_endpoint SET locked_by=:locked_by, lock_time=:lock_time '\n"
        "        sql += 'WHERE rse=:rse AND (locked_by IS NULL OR locked_by=:locked_by OR lock_time < :min_lock_time) '\n"
        "        varMap = {}\n"
        "        varMap[':rse'] = rse\n"
        "        varMap[':locked_by'] = me\n"
        "        varMap[':lock_time'] = now\n"
        "        varMap[':min_lock_time'] = limit\n"
        "        self.cur.execute(sql + comment, varMap)\n"
    )
    _s, junctions, _c, _u, _conf, _a = _sql_extract(source)
    held = [branch.row_precondition for junction in junctions for branch in junction.branches]

    assert held and all(len(terms) == len(set(terms)) == 1 for terms in held)


def test_a_column_one_pass_of_a_loop_writes_is_said_where_its_bind_is_filled():
    """``updateJobStatus`` appends ``,{key}=:{key}`` per parameter, and ``,memory_leak=:memory_leak`` on the pass
    for the job's metrics.  Run, the statement holds a ``{}`` where the passes are, glued to the value before it:
    ``status=:status{}`` named no bind, and the column a pass writes was not in the text.  Read with the passes in
    place, each column takes its own bind, and the branch is where the bind is filled, under the pass's test."""
    source = (
        "class M:\n"
        "    def f(self, jediTaskID, status, params):\n"
        "        sql = 'UPDATE ATLAS_PANDA.JEDI_Tasks SET status=:status'\n"
        "        varMap = {}\n"
        "        varMap[':jediTaskID'] = jediTaskID\n"
        "        varMap[':status'] = status\n"
        "        for key in params:\n"
        "            sql += f',{key}=:{key}'\n"
        "            varMap[f':{key}'] = params[key]\n"
        "            if key == 'restart':\n"
        "                sql += ',oldStatus=:oldStatus'\n"
        "                varMap[':oldStatus'] = 'paused'\n"
        "        sql += ' WHERE jediTaskID=:jediTaskID '\n"
        "        self.cur.execute(sql + comment, varMap)\n"
    )
    _s, junctions, _c, _u, _conf, _a = _sql_extract(source)
    by_column = {
        junction.subject.split(".")[-1]: [(branch.outcome, branch.line, branch.path_condition) for branch in junction.branches]
        for junction in junctions
    }

    assert by_column["oldStatus"] == [("paused", 12, ["key == 'restart'"])]
    assert by_column["status"] == [("runtime(status)", 6, [])]


def test_a_value_written_before_a_loops_passes_is_said_where_it_was_written():
    """``harvesterIsAlive`` writes ``lastUpdate=CURRENT_DATE`` and then a column per key.  Run, the statement holds
    ``lastUpdate=CURRENT_DATE{}``: read so, the value was a hole's, placed at the loop.  Read with the passes in
    place, it is ``CURRENT_DATE``, written by the statement that wrote it -- found in the variant read the same way,
    or not found at all."""
    source = (
        "class M:\n"
        "    def f(self, jediTaskID, data):\n"
        "        sql = \"UPDATE ATLAS_PANDA.JEDI_Tasks SET status=:status,oldStatus='paused'\"\n"
        "        for key in data:\n"
        "            sql += ',{0}=:{0}'.format(key)\n"
        "        sql += ' WHERE jediTaskID=:jediTaskID '\n"
        "        self.cur.execute(sql + comment, varMap)\n"
    )
    _s, junctions, _c, _u, _conf, _a = _sql_extract(source)
    (junction,) = [one for one in junctions if one.subject.endswith(".oldStatus")]

    assert [(branch.outcome, branch.line) for branch in junction.branches] == [("paused", 3)]


def test_a_guard_some_writes_carry_is_said_with_the_test_that_adds_it():
    """``updateJobStatus`` appends ``AND NOT jobStatus=:ngStatus`` only when the
    new status is ``transferring``: a ``holding`` job may not move there.

    Not every write of the value carries it, so it is not ``row_precondition``;
    the pieces the run built the statement from say which test added it.
    """
    source = (
        "class M:\n"
        "    def f(self, rse, status):\n"
        "        sqlU = 'UPDATE ATLAS_PANDA.ddm_endpoint SET blacklisted=:blacklisted'\n"
        "        sqlW = ' WHERE rse=:rse '\n"
        "        if status == 'Y':\n"
        "            sqlW += 'AND NOT blacklisted=:ng '\n"
        "        varMap = {}\n"
        "        varMap[':rse'] = rse\n"
        "        varMap[':blacklisted'] = status\n"
        "        varMap[':ng'] = 'X'\n"
        "        self.cur.execute(sqlU + sqlW + comment, varMap)\n"
    )
    _s, junctions, _c, _u, _conf, _a = _sql_extract(source)
    (branch,) = junctions[0].branches

    assert branch.row_precondition == []
    # Settled as a test negated outright is (the bind holds "X").
    assert [(guard.term, guard.when) for guard in branch.row_precondition_when] == [
        ("blacklisted<>X", ["status == 'Y'"])
    ]


@pytest.mark.parametrize("apart", [True, False])
def test_a_guard_one_execute_carries_is_said_only_where_no_other_can_run(apart):
    """``killJob`` runs its guarded statement in the arm that is not a force
    kill, so the guard holds when that arm's test does.

    ``ppEventServiceJob`` writes ``ready`` by a guarded ``UPDATE`` and then by
    an ``INSERT ... SELECT`` guarding nothing, one after the other: which test
    built the ``UPDATE``'s text says nothing about the other write, so no
    test is said at all.
    """
    plain = "'UPDATE ATLAS_PANDA.ddm_endpoint SET blacklisted=:blacklisted WHERE rse=:rse '"
    runs = (
        (
            "        if force:\n"
            "            self.cur.execute(sqlF + comment, varMap)\n"
            "        else:\n"
            "            self.cur.execute(sqlG + comment, varMap)\n"
        )
        if apart
        else (
            "        self.cur.execute(sqlG + comment, varMap)\n"
            "        self.cur.execute(sqlF + comment, varMap)\n"
        )
    )
    source = (
        "class M:\n"
        "    def f(self, rse, force, strict):\n"
        f"        sqlF = {plain}\n"
        f"        sqlG = {plain}\n"
        "        if strict:\n"
        "            sqlG += 'AND blacklisted=:old '\n"
        "        varMap = {}\n"
        "        varMap[':rse'] = rse\n"
        "        varMap[':blacklisted'] = 'Y'\n"
        "        varMap[':old'] = 'N'\n"
        f"{runs}"
    )
    _s, junctions, _c, _u, _conf, _a = _sql_extract(source)
    (branch,) = junctions[0].branches

    assert branch.row_precondition == []
    said = [(guard.term, guard.when) for guard in branch.row_precondition_when]
    assert said == ([("blacklisted=N", ["not (force)", "strict"])] if apart else [])


def test_a_guard_something_appended_inside_made_another_term_is_not_said_for_it():
    """``getEventRanges`` narrows the subquery of its guard when not ``scattered``.

    The shorter guard's text is written under ``strict``, so its own pieces
    say "when ``strict``" -- yet under ``strict`` and not ``scattered`` the
    statement holds the longer one instead.  The statement holding the longer
    one passed every test the shorter one names, so the shorter one is not
    said; the longer one is, with the test that narrows it.
    """
    source = (
        "class M:\n"
        "    def f(self, rse, strict, scattered):\n"
        "        sqlU = 'UPDATE ATLAS_PANDA.ddm_endpoint SET blacklisted=:blacklisted WHERE rse=:rse '\n"
        "        if strict:\n"
        "            sqlU += 'AND blacklisted IN (SELECT b FROM ATLAS_PANDA.T WHERE x=:x'\n"
        "            if not scattered:\n"
        "                sqlU += ' AND y=:y'\n"
        "            sqlU += ') '\n"
        "        varMap = {}\n"
        "        varMap[':rse'] = rse\n"
        "        varMap[':blacklisted'] = 'Y'\n"
        "        self.cur.execute(sqlU + comment, varMap)\n"
    )
    _s, junctions, _c, _u, _conf, _a = _sql_extract(source)
    (branch,) = junctions[0].branches

    assert branch.row_precondition == []
    assert [(guard.term, guard.when) for guard in branch.row_precondition_when] == [
        (
            "blacklisted IN (SELECT b FROM ATLAS_PANDA.T WHERE x=:x AND y=:y)",
            ["strict", "not scattered"],
        )
    ]


def test_a_guard_is_asked_of_every_statement_the_call_runs_not_of_a_covering_set():
    """``getEventRanges`` spells its jumbo subquery with one limit or another, narrowed or not.

    The covering set has three of the five statements -- each arm once, not
    each combination -- and the narrowed one with the limit it does not hold
    passes every test the narrowed one it holds names.  Asked of the covering
    set, that guard was said "when strict and not scattered"; asked of every
    statement, no guard here holds whenever its tests do.
    """
    source = (
        "class M:\n"
        "    def f(self, rse, strict, scattered, limit=None):\n"
        "        n = 9 if limit is None else limit\n"
        "        sqlU = 'UPDATE ATLAS_PANDA.ddm_endpoint SET blacklisted=:blacklisted WHERE rse=:rse '\n"
        "        if strict:\n"
        "            sqlU += f'AND blacklisted IN (SELECT b FROM ATLAS_PANDA.T WHERE rownum<={n}'\n"
        "            if not scattered:\n"
        "                sqlU += ' AND y=:y'\n"
        "            sqlU += ') '\n"
        "        varMap = {}\n"
        "        varMap[':rse'] = rse\n"
        "        varMap[':blacklisted'] = 'Y'\n"
        "        self.cur.execute(sqlU + comment, varMap)\n"
    )
    _s, junctions, _c, _u, _conf, _a = _sql_extract(source)
    (branch,) = junctions[0].branches

    assert branch.row_precondition == []
    assert branch.row_precondition_when == []


def test_a_table_the_call_substitutes_is_read_from_the_loop_that_supplies_it():
    """``execute((sql + comment) % table)`` over a literal tuple of table names.

    ``killJob`` and six others run one statement against ``jobsDefined4`` and
    ``jobsActive4`` in turn.  The names are written out in the loop, so the
    statement is two statements on tables the map already resolves to
    ``JobSpec`` -- and reading it as ``UPDATE %s`` resolves to nothing at all,
    since the table pattern does not even accept a ``%``.
    """
    source = (
        "def f(self):\n"
        "    sqlU = 'UPDATE %s SET commandToPilot=:commandToPilot WHERE PandaID=:PandaID '\n"
        "    for table in ('ATLAS_PANDA.jobsDefined4', 'ATLAS_PANDA.jobsActive4'):\n"
        "        self.cur.execute((sqlU + comment) % table, varMap)\n"
    )
    func = _func(source)

    tables = [w.table for run in sql.executions(func) for w in sqltext.writes(run.sql)]

    assert tables == ["jobsDefined4", "jobsActive4"]


def test_an_indexed_placeholder_is_filled_from_the_matching_argument():
    """``sqlJ.format(tableName)`` -- an f-string never renders ``{0}``."""
    source = (
        "def f(self):\n"
        "    sqlJ = 'UPDATE ATLAS_PANDA.{0} SET currentPriority=:newPriority WHERE x=:x '\n"
        "    tableName = 'jobsActive4'\n"
        "    self.cur.execute(sqlJ.format(tableName) + comment, varMap)\n"
    )
    func = _func(source)

    writes = [w for run in sql.executions(func) for w in sqltext.writes(run.sql)]

    assert [(w.table, sorted(w.columns)) for w in writes] == [
        ("jobsActive4", ["currentPriority"])
    ]


def test_a_bare_placeholder_is_filled_only_where_no_f_string_built_the_text():
    """``{}`` is ambiguous, and the fragments settle it.

    The map renders an f-string interpolation as ``{}`` too, so a bare brace in
    the statement's text is either a hole the schema went into or a placeholder the
    call fills.  Which one is not a guess: if no fragment building the name
    interpolates, every brace in it is a literal the source wrote.
    """
    plain = (
        "def f(self):\n"
        "    sqlU = 'UPDATE ATLAS_PANDA.{} SET commandToPilot=:c WHERE PandaID=:P '\n"
        "    self.cur.execute(sqlU.format('jobsActive4') + comment, varMap)\n"
    )
    interpolated = (
        "def f(self):\n"
        "    sqlU = f'UPDATE {schema}.JEDI_Tasks SET status=:status WHERE x=:x '\n"
        "    self.cur.execute(sqlU.format('jobsActive4') + comment, varMap)\n"
    )

    assert [w.table for run in sql.executions(_func(plain)) for w in sqltext.writes(run.sql)] == [
        "jobsActive4"
    ]
    assert [
        w.table for run in sql.executions(_func(interpolated)) for w in sqltext.writes(run.sql)
    ] == ["JEDI_Tasks"]


def test_a_format_call_at_the_assignment_names_one_table_per_pass():
    """``(...).format(schemaPANDA, table)`` -- the older spelling, and the one
    that still hid three statements.

    ``rendered_text`` flattens ``{0}`` and ``{1}`` to ``{}`` so the statement
    reads at all, and the arguments behind them were then out of reach: the map
    knew ``getUsageBreakdown_JEDI`` queried a table and not which one.  The
    indices are read back from the receiver instead, because a ``{0}`` left in
    the rendered text is a ``{}`` that stopped working -- every reader
    downstream keys on the two literal characters.
    """
    source = (
        "def f(self):\n"
        "    for table in ['jobsActive4', 'jobsArchived4']:\n"
        "        sqlJ = (\n"
        "            'SELECT COUNT(*),prodUserName '\n"
        "            'FROM {0}.{1} '\n"
        "            'WHERE prodSourceLabel=:label '\n"
        "        ).format(panda_config.schemaPANDA, table)\n"
        "        self.cur.execute(sqlJ + comment, varMap)\n"
    )

    assert [run.sql for run in sql.executions(_func(source))] == [
        "SELECT COUNT(*),prodUserName FROM {}.jobsActive4 WHERE prodSourceLabel=:label ",
        "SELECT COUNT(*),prodUserName FROM {}.jobsArchived4 WHERE prodSourceLabel=:label ",
    ]


def test_an_index_used_twice_fills_both_of_its_holes():
    """``"FROM {0}.{1} j, {0}.Datasets d"`` has three markers and two arguments.

    One entry per *marker* is the only shape ``_table_choices`` accepts -- it
    measures the list against the number of ``{}`` the text carries -- so a
    list of distinct arguments is one short and the fragment is dropped.
    Dropped, not truncated, and without a word about it.
    """
    source = (
        "def f(self, vo):\n"
        "    tableName = 'jobsDefined4'\n"
        "    sqlJ = 'SELECT distinct prodUserName '\n"
        "    sqlJ += 'FROM {0}.{1} j, {0}.Datasets d '.format(schemaPANDA, tableName)\n"
        "    sqlJ += 'WHERE vo=:vo '\n"
        "    self.cur.execute(sqlJ + comment, varMap)\n"
    )

    assert [run.sql for run in sql.executions(_func(source))] == [
        "SELECT distinct prodUserName FROM {}.jobsDefined4 j, {}.Datasets d WHERE vo=:vo "
    ]


def test_a_derived_field_holds_its_position_without_naming_anything():
    """``{0[1]}`` and ``{2:.3%}`` take a marker but name a value computed from
    an argument rather than the argument.

    Leaving them out would move every later hole onto the wrong argument, and
    the count guard that catches that drops the whole fragment -- so they hold
    a position that resolves to nothing, and the hole beside them still fills.
    """
    source = (
        "def f(self):\n"
        "    sqlJ = 'SELECT PandaID '\n"
        "    sqlJ += 'FROM {0[1]}.{1} WHERE pct>{2:.3%} '.format(names, 'jobsActive4', ratio)\n"
        "    self.cur.execute(sqlJ + comment, varMap)\n"
    )

    assert [run.sql for run in sql.executions(_func(source))] == [
        "SELECT PandaID FROM {}.jobsActive4 WHERE pct>{} "
    ]


def test_a_schema_qualified_hole_is_still_the_table_position():
    """Where a new PanDA job is born, and the map could not name the table.

    ``INSERT INTO {schema}.{table}`` puts the schema in one hole and the table
    in the next, so the table hole is preceded by ``{}.`` rather than by the
    keyword.  Reading only the keyword's immediate neighbour dropped it, and
    with it the two statements that create a job row -- the one place the
    system's most-read row comes into existence.
    """
    source = (
        "def f(self, jobSpec):\n"
        "    if jobSpec.jobStatus in ['defined', 'assigned']:\n"
        "        table_name = 'jobsDefined4'\n"
        "    else:\n"
        "        table_name = 'jobsActive4'\n"
        "    sql1 = f'INSERT INTO {schemaPANDA}.{table_name} (PandaID) '\n"
        "    sql1 += 'VALUES (:PandaID)'\n"
        "    self.cur.execute(sql1 + comment, varMap)\n"
    )

    created = [table for run in sql.executions(_func(source)) for table in sqltext.creates(run.sql)]

    assert sorted(created) == ["jobsActive4", "jobsDefined4"]


def test_a_hole_in_the_column_list_is_not_a_table():
    """The widened reading stops at the parenthesis.

    ``INSERT INTO {schema}.{table} ({columns})`` has three holes and only the
    middle one names rows.  Filling the column list would put one spec class's
    hundred-odd attributes into a statement as though the code had written
    them, and a reader has no way to tell that apart from a column list the
    source really does spell out.
    """
    source = (
        "def f(self):\n"
        "    table_name = 'jobsDefined4'\n"
        "    sql1 = f'INSERT INTO {schemaPANDA}.{table_name} ({JobSpec.columnNames()}) '\n"
        "    sql1 += 'VALUES (:PandaID)'\n"
        "    self.cur.execute(sql1 + comment, varMap)\n"
    )

    assert [run.sql for run in sql.executions(_func(source))] == [
        "INSERT INTO {}.jobsDefined4 ({}) VALUES (:PandaID)"
    ]


def test_a_bind_variable_after_into_is_not_a_table():
    """``RETURNING PandaID INTO :newPandaID`` names where the sequence's value
    lands, not a table, and the keyword before it is the same one an insert
    uses.  The bind marker settles it."""
    source = (
        "def f(self):\n"
        "    table_name = 'jobsDefined4'\n"
        "    sql1 = f'INSERT INTO {schemaPANDA}.{table_name} (PandaID) '\n"
        "    sql1 += 'VALUES (:PandaID) RETURNING PandaID INTO :newPandaID'\n"
        "    self.cur.execute(sql1 + comment, varMap)\n"
    )

    created = [table for run in sql.executions(_func(source)) for table in sqltext.creates(run.sql)]

    assert created == ["jobsDefined4"]


def test_a_name_the_statement_defines_for_itself_is_not_a_table_it_reads():
    """``joins`` already knew this; ``reads`` did not.

    ``copyArchive`` opens ``WITH p AS (SELECT ... FROM jobsActive4 ...)`` and
    then selects ``FROM p``.  Reading ``p`` as a table put a table nothing
    writes into the map, and the next step calls that a boundary -- a claim
    that PanDA depends on state some other system maintains, about a name that
    exists only inside one statement.
    """
    statement = (
        "WITH p AS ("
        "SELECT MIN(PandaID) PandaID,computingSite FROM ATLAS_PANDA.jobsActive4 "
        "WHERE jobStatus=:jobStatus) "
        "SELECT p.computingSite,s.modificationTime "
        "FROM p, ATLAS_PANDA.jobs_statuslog s WHERE s.PandaID=p.PandaID"
    )

    # The real tables are still both there: the one inside the common table
    # expression is its own ``SELECT``, and the outer query's partner is a
    # join.  Only the self-defined name goes.
    assert [table for table, _columns in sqltext.reads(statement)] == ["jobsActive4"]
    assert sqltext.joins(statement) == ["jobs_statuslog"]


def test_oracle_s_one_row_pseudo_table_is_not_a_table():
    """``SELECT <seq>.nextval FROM dual`` asks the sequence for a number.

    ``dual`` sits where a table goes and holds none of PanDA's rows --
    ``WrappedCursor`` deletes ``FROM dual`` outright when the backend is not
    Oracle.  Counted as a table it is read eleven times and written never,
    which is the exact shape the next step turns into "maintained outside
    PanDA".
    """
    statement = "SELECT ATLAS_PANDA.JOBSDEFINED4_PANDAID_SEQ.nextval FROM dual"

    assert sqltext.reads(statement) == []


def test_a_literal_written_where_a_column_goes_is_not_a_column():
    """``reads`` said the archive table carries a column called ``NULL``.

    ``recoverLostFiles`` pads a ``UNION`` so both branches project four
    positions: the live branch selects ``fileID,datasetID,lfn,NULL`` and the
    branch it lines up with selects ``outPandaID`` in that fourth position.  So
    the literal is not merely a value the row does not carry -- it stands where
    a differently named column stands, and reading it as one put ``NULL`` on the
    boundary ``filesTable_ARCH`` becomes.  The docstring already said
    expressions are dropped because a boundary is about what the row carries; a
    bare literal is the same claim written without parentheses.
    """
    statement = (
        "SELECT fileID,datasetID,lfn,NULL FROM ATLAS_PANDA.filesTable4 "
        "WHERE PandaID=:PandaID AND type IN (:type1,:type2) "
        "UNION "
        "SELECT fileID,datasetID,lfn,NULL FROM ATLAS_PANDAARCH.filesTable_ARCH "
        "WHERE PandaID=:PandaID AND type IN (:type1,:type2)"
    )

    assert sqltext.reads(statement) == [
        ("filesTable4", ["fileID", "datasetID", "lfn"]),
        ("filesTable_ARCH", ["fileID", "datasetID", "lfn"]),
    ]


def test_the_database_s_own_clock_is_not_a_column_the_row_carries():
    """``SELECT CURRENT_DATE,status`` asks the database what time it is.

    Two spellings of it in this corpus, both in a projection beside real
    columns: ``closeTaskAttempt`` copies ``CURRENT_DATE`` into an attempt's end
    time, and ``base_module`` stamps ``CURRENT_TIMESTAMP`` into a status log.
    Neither names anything ``JEDI_Tasks`` stores.
    """
    assert sqltext.reads(
        "SELECT CURRENT_DATE,status FROM ATLAS_PANDA.JEDI_Tasks "
        "WHERE jediTaskID=:jediTaskID"
    ) == [("JEDI_Tasks", ["status"])]

    assert sqltext.reads(
        "SELECT jediTaskID,CURRENT_TIMESTAMP,status,attemptNr "
        "FROM ATLAS_PANDA.JEDI_Tasks WHERE jediTaskID=:jediTaskID"
    ) == [("JEDI_Tasks", ["jediTaskID", "status", "attemptNr"])]


def test_an_upper_case_column_name_is_still_a_column():
    """The guard on the rule above: case cannot be what drops a literal.

    ``get_shares`` selects twelve columns from ``GLOBAL_SHARES`` and every one
    of them is spelled in upper case, exactly like the literals are.  Excluding
    SQL's reserved words as a class would be worse still -- ``NAME``, ``VALUE``
    and ``PARENT`` are three of these twelve.  Only the words this corpus writes
    where a column goes and that name no column are dropped.
    """
    statement = (
        "SELECT NAME, VALUE, PARENT, PRODSOURCELABEL, WORKINGGROUP, CAMPAIGN, "
        "PROCESSINGTYPE, TRANSPATH, RTYPE, VO, QUEUE_ID, THROTTLED "
        "FROM ATLAS_PANDA.GLOBAL_SHARES"
    )

    assert sqltext.reads(statement) == [
        (
            "GLOBAL_SHARES",
            [
                "NAME", "VALUE", "PARENT", "PRODSOURCELABEL", "WORKINGGROUP",
                "CAMPAIGN", "PROCESSINGTYPE", "TRANSPATH", "RTYPE", "VO",
                "QUEUE_ID", "THROTTLED",
            ],
        )
    ]


def test_a_projection_the_caller_decides_leaves_no_column_behind():
    """An empty string was entering the column list, and it is not a name.

    ``getTaskAttributesPanda`` builds its projection from a list its caller
    passes::

        sqlRR = "SELECT "
        for attr in attrs:
            sqlRR += f"{attr},"
        sqlRR = sqlRR[:-1]
        sqlRR += f" FROM {panda_config.schemaJEDI}.JEDI_Tasks "

    The assembler folds the loop body to ``{}`` and cannot model the slice that
    removes the trailing comma, so it sees ``SELECT {}, FROM ...``.  ``{}`` was
    dropped, as an interpolated name should be, and then the empty part after
    the comma was *kept* -- the guard tested ``part or "_"`` and the fallback
    matched.  The truthful answer is that this projection names nothing the
    statement decides.
    """
    assert sqltext.reads("SELECT {}, FROM {}.JEDI_Tasks WHERE jediTaskID=:jediTaskID") == [
        ("JEDI_Tasks", [])
    ]


def test_a_join_written_with_the_keyword_names_the_same_partner():
    """Two spellings of one statement, and the map read only one of them.

    A join partner bounds which rows the query can see, which is why it is a
    gate.  That is true of ``FROM a, b`` and of ``FROM a JOIN b ON ...``
    alike, and reading only the comma form left ``resource_types`` -- read by
    the corpus, written by none of it -- with no node at all.
    """
    comma = (
        "SELECT jss.resource_type FROM ATLAS_PANDA.JOBS_SHARE_STATS jss,"
        "ATLAS_PANDA.resource_types rt WHERE jss.resource_type=rt.resource_name"
    )
    keyword = (
        "SELECT jss.resource_type FROM ATLAS_PANDA.JOBS_SHARE_STATS jss "
        "JOIN ATLAS_PANDA.resource_types rt ON jss.resource_type=rt.resource_name"
    )

    assert sqltext.joins(comma) == sqltext.joins(keyword) == ["resource_types"]


def test_a_join_key_is_not_a_condition_on_a_value():
    """``ON`` carries the equality that makes the join, not a test on a value.

    The terms after ``WHERE``/``AND``/``OR`` are what ``row_precondition``
    reads, and every one of the corpus's twenty-nine ``ON`` clauses is a plain
    key equality -- none compares a column to a bind variable or a literal.
    Reading them would add the join's own plumbing to the conditions a row had
    to satisfy.
    """
    joined = (
        "SELECT j.PandaID FROM ATLAS_PANDA.jobsActive4 j "
        "JOIN ATLAS_PANDA.schedconfig_json s ON j.computingSite=s.panda_queue "
        "WHERE j.jobStatus=:jobStatus"
    )

    assert sqltext.joins(joined) == ["schedconfig_json"]
    assert [(term.column, term.binds) for term in sqltext.terms(joined).terms] == [
        ("jobStatus", (":jobStatus",))
    ]
    assert _conditions(joined) == []


def _conditions(statement: str) -> list[str]:
    """``sqltext.conditions`` rendered the way a reader reads it."""
    return [
        " OR ".join(" AND ".join(branch.terms) for branch in condition.branches)
        for condition in sqltext.conditions(statement)
    ]


def test_a_condition_is_spelled_as_the_statement_spells_it():
    """``endTime IS NOT NULL`` comes back from sqlglot as ``NOT endTime IS NULL``.

    So each term is cut from the statement's own text, where a reader can find
    it in the source, and only what it is comes off the tree:
    ``jobStatus=:jobStatus`` is the ``selects`` list, and the row also had to
    be older than a time limit, which nothing else says.
    """
    statement = (
        "SELECT PandaID FROM ATLAS_PANDA.jobsActive4 WHERE jobStatus=:jobStatus "
        "AND (modificationTime<:modificationTime OR (endTime IS NOT NULL "
        "AND endTime<:endTime))"
    )

    assert [(term.column, term.binds) for term in sqltext.terms(statement).terms] == [
        ("jobStatus", (":jobStatus",))
    ]
    assert _conditions(statement) == [
        "modificationTime<:modificationTime OR endTime IS NOT NULL AND endTime<:endTime"
    ]


def test_a_term_selects_already_carries_is_not_repeated_as_a_condition():
    """One fact, one row.  ``jobStatus=:jobStatus`` is the ``selects`` list."""
    statement = (
        "SELECT PandaID FROM ATLAS_PANDA.jobsActive4 WHERE jobStatus=:jobStatus "
        "AND stateChangeTime != modificationTime"
    )

    assert _conditions(statement) == ["stateChangeTime != modificationTime"]


def test_a_join_term_is_not_a_condition_but_a_column_pair_is():
    """``c1.jediTaskID=d.jediTaskID`` says how two tables meet.

    A row is not passed over for failing a join key; it is passed over for
    failing ``stateChangeTime != modificationTime``, which compares two of its
    own columns and looks identical until the alias is read.  182 of the
    corpus's terms are joins and would bury the 473 that are answers.
    """
    statement = (
        "SELECT tabT.jediTaskID FROM {}.JEDI_Tasks tabT,{}.JEDI_Datasets tabD "
        "WHERE tabT.jediTaskID=tabD.jediTaskID AND tabT.status=tabA.status "
        "AND stateChangeTime != modificationTime"
    )

    assert _conditions(statement) == ["stateChangeTime != modificationTime"]


def test_an_or_group_stays_one_condition():
    """Split into two entries the list would ask for a row that cannot exist.

    The conditions list reads as a conjunction -- "the row had to satisfy all
    of these" -- so ``(status=:running OR status=:scouting)`` has to arrive as
    one entry.  Flattened it says the status must be both at once, and a
    reader ruling out the first term rules out a statement that never required
    it.  124 of the corpus's 473 terms sit in such a group.
    """
    statement = (
        "SELECT jediTaskID FROM {}.JEDI_Tasks tabT WHERE tabT.vo=:vo "
        "AND (tabT.status=:running OR tabT.status=:scouting)"
    )

    assert _conditions(statement) == ["tabT.status=:running OR tabT.status=:scouting"]


def test_a_branch_of_an_or_group_keeps_the_terms_that_go_together():
    """``(A AND B) OR (C AND D)`` is two branches of two, not four branches.

    Flattening the ``AND``s would report a weaker requirement than the
    statement makes: a row satisfying ``B`` alone has not satisfied the group.
    ``getFailureCountsForTask_JEDI`` is this shape, and the innermost paren
    around each of its three ``jobStatus`` terms holds no ``OR`` at all --
    which is why the group is found by looking outward.
    """
    statement = (
        "SELECT COUNT(*) FROM {}.jobsArchived4 WHERE jediTaskID=:jediTaskID "
        "AND ((jobStatus=:jobFailed AND pilotErrorCode IS NOT NULL) "
        "OR (jobStatus=:jobClosed AND jobSubStatus=:toReassign))"
    )

    assert [
        [branch.terms for branch in condition.branches]
        for condition in sqltext.conditions(statement)
    ] == [
        [
            ("jobStatus=:jobFailed", "pilotErrorCode IS NOT NULL"),
            ("jobStatus=:jobClosed", "jobSubStatus=:toReassign"),
        ]
    ]


def test_a_top_level_or_is_one_condition_even_with_no_parens_of_its_own():
    """``(A) OR (B) AND C`` is two alternatives, not a requirement plus two.

    ``get_workers_to_synchronize`` is this shape.  Looking outward for an
    enclosing paren -- which is what this did before -- finds the group around
    ``A``, sees no ``OR`` inside it, and reports every one of its terms as
    required on its own.  ``AND`` binds tighter than ``OR``, so ``C`` belongs
    to the second alternative.
    """
    statement = (
        "SELECT workerID FROM ATLAS_PANDA.harvester_workers "
        "WHERE (status in ('submitted') AND pilotStatus='running') "
        "OR (status in ('ready') AND pilotStatus='finished') "
        "AND lastupdate > sysdate FOR UPDATE"
    )

    assert [
        [branch.terms for branch in condition.branches]
        for condition in sqltext.conditions(statement)
    ] == [
        [
            ("status in ('submitted')", "pilotStatus='running'"),
            ("status in ('ready')", "pilotStatus='finished'", "lastupdate > sysdate"),
        ]
    ]


def test_a_disjunction_inside_a_branch_keeps_its_parentheses():
    """``X AND a OR b`` reads as ``(X AND a) OR b``, which is looser.

    A row that fails ``X`` would be reported as still able to come back, which
    is the direction that lets a wrong explanation survive.  The branch's
    terms are joined by ``AND``, so a nested disjunction has to carry its own
    brackets into the text.
    """
    statement = (
        "SELECT x FROM T WHERE (a='1' AND b<:p) "
        "OR (c='2' AND d<:q AND (e>:r OR e IS NULL))"
    )

    assert [
        [branch.terms for branch in condition.branches]
        for condition in sqltext.conditions(statement)
    ] == [
        [
            ("a='1'", "b<:p"),
            ("c='2'", "d<:q", "(e>:r OR e IS NULL)"),
        ]
    ]


def test_what_follows_the_where_clause_is_not_a_condition():
    """``ORDER BY`` and ``FOR UPDATE`` are about the rows that did come back."""
    statement = (
        "SELECT PandaID FROM ATLAS_PANDA.jobsActive4 WHERE lockedBy IS NULL "
        "ORDER BY PandaID"
    )

    assert _conditions(statement) == ["lockedBy IS NULL"]


def test_a_disjunction_that_lost_a_branch_to_a_join_is_not_recorded():
    """One alternative gone, and the row satisfies the rest whatever it holds.

    Recording what is left would state a requirement the statement does not
    make -- the direction that gets a row ruled out for failing a test it
    never faced.
    """
    statement = (
        "SELECT a.x FROM A a,B b WHERE a.vo=:vo AND (a.id=b.id OR a.late IS NULL)"
    )

    assert _conditions(statement) == []


def test_a_condition_that_names_no_bind_is_still_returned():
    """An ``IS NULL`` names no value to go looking for, and is still a reason.

    The map has nowhere to send a walk for it, and saying so is the point:
    ``Branch.binds`` is empty, and the renderer leaves the term plain rather
    than dropping it.
    """
    statement = "SELECT PandaID FROM ATLAS_PANDA.jobsActive4 WHERE lockedBy IS NULL"

    assert [condition.binds for condition in sqltext.conditions(statement)] == [()]
    assert _conditions(statement) == ["lockedBy IS NULL"]


def test_an_interpolated_table_is_still_reported_as_a_read():
    """The hole is kept on purpose, and the difference from ``joins`` is real.

    A statement whose table nobody could name is a gap, and the build report
    says how many there are and in which functions.  ``joins`` drops the hole
    because an unnameable partner cannot bound anything; ``reads`` keeps it
    because dropping it would turn a counted gap into silence.
    """
    statement = "SELECT PandaID FROM {} WHERE jediTaskID=:jediTaskID"

    assert [table for table, _columns in sqltext.reads(statement)] == ["{}"]
    assert sqltext.joins(statement) == []


def test_an_f_string_that_only_glues_reads_as_the_concatenation_it_is():
    """``execute(f"{sql} {comment}")`` is ``execute(sql + comment)`` respelled.

    Read as one opaque expression it has no head that names a statement, so
    the execution was dropped whole -- not truncated, dropped.  Five statements
    went that way, and one of them is the only place ``JOBSDEFINED_SHARE_STATS``
    is named, so the map did not know the table exists.
    """
    glued = (
        "def f(self):\n"
        "    comment = ' /* tag */'\n"
        "    sql_get = 'SELECT data FROM ATLAS_PANDA.Job_Metrics WHERE jediTaskID=:jediTaskID'\n"
        "    self.cur.execute(f'{sql_get} {comment}', var_map)\n"
    )
    added = glued.replace("f'{sql_get} {comment}'", "sql_get + comment")

    spelled = [run.sql.strip() for run in sql.executions(_func(glued))]

    assert spelled == [run.sql.strip() for run in sql.executions(_func(added))]
    assert spelled == ["SELECT data FROM ATLAS_PANDA.Job_Metrics WHERE jediTaskID=:jediTaskID"]


def test_an_f_string_carrying_its_own_text_is_not_split():
    """A wrapper's literal parts are whitespace; a statement's are the statement.

    Splitting one that carries text would keep only the piece before the first
    hole, which is a truncation the reader cannot tell from a short statement.
    """
    source = (
        "def f(self, taskBuffer):\n"
        "    taskBuffer.querySQLS(f'UPDATE {schemaPANDA}.jobsActive4 SET jobStatus=:new', varMap)\n"
    )

    assert [run.sql for run in sql.executions(_func(source))] == [
        "UPDATE {}.jobsActive4 SET jobStatus=:new"
    ]


def test_one_f_string_elsewhere_does_not_disqualify_the_other_statements():
    """Whether a ``{}`` is a placeholder is a question about *this* statement.

    It was asked about every assignment to the name anywhere in the function.
    ``copyArchive.main`` assigns ``sql`` forty-four times and one of them is an
    f-string, so the other forty-three were treated as possibly carrying an
    interpolation's residue -- and the query whose table the loop supplies was
    left reading ``FROM ATLAS_PANDA.{}``, naming no table at all.
    """
    source = (
        "def main(self, taskBuffer):\n"
        "    sql = f'SELECT PandaID FROM {schemaPANDA}.jobsActive4 WHERE x=:x'\n"
        "    taskBuffer.querySQLS(sql, {})\n"
        "    sql = 'SELECT COMPUTINGSITE FROM ATLAS_PANDA.{} WHERE vo=:vo '\n"
        "    for table in ['JOBS_SHARE_STATS', 'JOBSDEFINED_SHARE_STATS']:\n"
        "        taskBuffer.querySQLS(sql.format(table), {})\n"
    )

    tables = {
        table
        for run in sql.executions(_func(source, name="main"))
        for table, _columns in sqltext.reads(run.sql)
    }

    assert {"JOBS_SHARE_STATS", "JOBSDEFINED_SHARE_STATS"} <= tables


def test_every_clause_of_a_multi_part_with_defines_a_name():
    """The second ``WITH`` clause follows a parenthesis, not a word.

    ``get_active_gshare_rtype_mapping`` opens two: ``WITH gshare_results AS
    (...), wq_results AS (...)``.  A word boundary before the comma demands a
    word character to its left, and the character there is ``)`` -- so the
    second name was read as a table, read by the corpus and written by none of
    it, which is the shape the next step calls a boundary.
    """
    statement = (
        "WITH gshare_results AS ("
        "SELECT gshare AS name FROM ATLAS_PANDA.JOBS_SHARE_STATS WHERE vo=:vo"
        "), wq_results AS ("
        "SELECT QUEUE_NAME AS name FROM ATLAS_PANDA.JEDI_WORK_QUEUE WHERE vo=:vo"
        ") SELECT name FROM gshare_results UNION SELECT name FROM wq_results"
    )

    assert [table for table, _columns in sqltext.reads(statement)] == [
        "JOBS_SHARE_STATS",
        "JEDI_WORK_QUEUE",
    ]


def test_a_table_name_derived_by_substitution_is_read_from_the_literals():
    """``peekJobLog`` builds three archive table names out of one.

    ``re.sub("jobsArchived", "metaTable_ARCH", table)`` states both halves and
    the value being rewritten is already known, so the result is computed, not
    guessed.  Without it ``metaTable_ARCH`` appeared nowhere in the map at all,
    while its two siblings from the same loop -- ``filesTable_ARCH`` and
    ``jobParamsTable_ARCH`` -- were both present, which is the worst shape for
    a reader: two of three named, and nothing saying the third was missed.
    """
    source = (
        "def f(self):\n"
        "    tables = [f'{schemaPANDAARCH}.jobsArchived']\n"
        "    for table in tables:\n"
        "        meta = re.sub('jobsArchived', 'metaTable_ARCH', table)\n"
        "        sql_get_meta = f'SELECT metaData FROM {meta} WHERE PandaID=:PandaID'\n"
        "        self.cur.execute(sql_get_meta + comment, var_map)\n"
    )

    assert [table for run in sql.executions(_func(source)) for table, _c in sqltext.reads(run.sql)] == [
        "metaTable_ARCH"
    ]


def test_a_substitution_the_source_does_not_spell_out_leaves_the_hole():
    """All three parts have to be written down.

    A pattern or a replacement the map cannot read makes the result a guess,
    and a guessed table name is attributed with exactly the confidence of one
    the code states.
    """
    source = (
        "def f(self, wanted):\n"
        "    tables = [f'{schemaPANDAARCH}.jobsArchived']\n"
        "    for table in tables:\n"
        "        meta = re.sub('jobsArchived', wanted, table)\n"
        "        sql_get_meta = f'SELECT metaData FROM {meta} WHERE PandaID=:PandaID'\n"
        "        self.cur.execute(sql_get_meta + comment, var_map)\n"
    )

    assert [run.sql for run in sql.executions(_func(source))] == [
        "SELECT metaData FROM {} WHERE PandaID=:PandaID"
    ]


def test_a_default_argument_names_the_table_the_function_writes():
    """``insertDataset(self, dataset, tablename="ATLAS_PANDA.Datasets")``.

    The caller passes nothing, so the name the statement interpolates is
    written in the signature -- by the same function, not by a caller that
    would have to be traced.  Until it was read, the one statement that creates
    a ``Datasets`` row named no table, and the map reported that kind of row as
    changed here but never created here.  The evaluator also reads the
    statement of a caller that passes a table, with the table a hole.
    """
    source = (
        "def f(self, dataset, tablename='ATLAS_PANDA.Datasets'):\n"
        "    sql1 = f'INSERT INTO {tablename} (vuid) '\n"
        "    sql1 += 'VALUES (:vuid)'\n"
        "    self.cur.execute(sql1 + comment, dataset.valuesMap())\n"
    )

    created = [table for run in sql.executions(_func(source)) for table in sqltext.creates(run.sql)]

    assert "Datasets" in created


def test_an_argument_with_no_default_leaves_the_hole():
    """``lockJobsForReassign(self, tableName, ...)`` takes the table from its
    caller, and four call sites pass four literals.  Reading them needs the
    call graph, which this does not have, so the hole stays and the build
    report counts the statement."""
    source = (
        "def f(self, tableName):\n"
        "    sql = f'SELECT PandaID FROM {tableName} WHERE x=:x'\n"
        "    self.cur.execute(sql + comment, varMap)\n"
    )

    assert [run.sql for run in sql.executions(_func(source))] == [
        "SELECT PandaID FROM {} WHERE x=:x"
    ]


def test_a_dict_keyed_by_table_name_is_a_list_of_tables():
    """``getDispatchDatasetsPerUser`` writes its list of tables as a dict, so
    each one can carry the statuses to ask it about, and adds two of the three
    under a flag.

    Reading only the literal answers ``jobsDefined4`` -- one table of three,
    stated with the confidence of the whole list -- so the entries added by
    subscript are read too, or none of it is.
    """
    source = (
        "def f(self, onlyActive):\n"
        "    tableStatMap = {'jobsDefined4': ['defined', 'assigned']}\n"
        "    if not onlyActive:\n"
        "        tableStatMap['jobsActive4'] = None\n"
        "        tableStatMap['jobsArchived4'] = None\n"
        "    for tableName in tableStatMap:\n"
        "        sqlJ = 'SELECT prodUserName '\n"
        "        sqlJ += 'FROM {0}.{1} j '.format(schemaPANDA, tableName)\n"
        "        self.cur.execute(sqlJ + comment, varMap)\n"
    )

    assert sorted(run.sql for run in sql.executions(_func(source))) == [
        "SELECT prodUserName FROM {}.jobsActive4 j ",
        "SELECT prodUserName FROM {}.jobsArchived4 j ",
        "SELECT prodUserName FROM {}.jobsDefined4 j ",
    ]


def test_a_dict_whose_keys_are_not_written_out_leaves_the_hole_alone():
    """A key computed at run time names no table, and ``{**other}`` is a whole
    key set from somewhere this cannot see."""
    for literal in ("{picked: ['defined']}", "{'jobsDefined4': ['defined'], **other}"):
        source = (
            "def f(self):\n"
            f"    tableStatMap = {literal}\n"
            "    for tableName in tableStatMap:\n"
            "        sqlJ = 'SELECT prodUserName '\n"
            "        sqlJ += 'FROM {0}.{1} j '.format(schemaPANDA, tableName)\n"
            "        self.cur.execute(sqlJ + comment, varMap)\n"
        )

        assert [run.sql for run in sql.executions(_func(source))] == [
            "SELECT prodUserName FROM {}.{} j "
        ], literal


def test_one_loop_variable_in_two_fragments_takes_one_value_at_a_time():
    """The holes are keyed on the expression, not counted separately.

    Treating them independently crosses them, and a union of two tables read as
    one statement is a query the code never runs.
    """
    source = (
        "def f(self):\n"
        "    for tableName in ('ATLAS_PANDA.jobsActive4', 'ATLAS_PANDA.jobsDefined4'):\n"
        "        sqlP = f'SELECT PandaID FROM {tableName} '\n"
        "        sqlP += f'WHERE x IN (SELECT x FROM {tableName}) '\n"
        "        self.cur.execute(sqlP + comment, varMap)\n"
    )

    assert [run.sql for run in sql.executions(_func(source))] == [
        "SELECT PandaID FROM ATLAS_PANDA.jobsActive4 "
        "WHERE x IN (SELECT x FROM ATLAS_PANDA.jobsActive4) ",
        "SELECT PandaID FROM ATLAS_PANDA.jobsDefined4 "
        "WHERE x IN (SELECT x FROM ATLAS_PANDA.jobsDefined4) ",
    ]


def test_a_substitution_that_resolves_to_nothing_leaves_the_hole():
    """A run-time filler is not guessed at; the field stays a hole, spelled as every hole is."""
    source = (
        "def f(self, order_by_policy):\n"
        "    sqlR = 'UPDATE ATLAS_PANDA.{0} SET status=:status WHERE x=:x '\n"
        "    self.cur.execute(sqlR.format(order_by_policy) + comment, varMap)\n"
    )
    func = _func(source)

    assert [run.sql for run in sql.executions(func)] == [
        "UPDATE ATLAS_PANDA.{} SET status=:status WHERE x=:x "
    ]


def test_a_named_field_no_argument_fills_is_a_hole_the_reading_stops_on():
    """``.format(q=...)`` fills nothing here; the field is a hole, not text the WHERE drops."""
    source = (
        "def f(self, statuses):\n"
        "    tmpl = 'SELECT a FROM ATLAS_PANDA.jobsActive4 WHERE site=:site AND jobStatus IN {q} '\n"
        "    sql = tmpl.format(q=statuses)\n"
        "    self.cur.execute(sql + comment, varMap)\n"
        "    lit = \"SELECT a FROM ATLAS_PANDA.jobsActive4 WHERE doc='{k}' \"\n"
        "    self.cur.execute(lit + comment, varMap)\n"
    )
    func = _func(source)

    formatted, literal = sql.executions(func)

    assert formatted.sql == "SELECT a FROM ATLAS_PANDA.jobsActive4 WHERE site=:site AND jobStatus IN {} "
    assert [(stop.source, stop.reason) for stop in formatted.unresolved] == [
        ("jobStatus IN {}", sql.BINDS_ARE_A_HOLE)
    ]
    # Not a template: its brace is the statement's own text.
    assert literal.sql == "SELECT a FROM ATLAS_PANDA.jobsActive4 WHERE doc='{k}' "


def test_a_query_whose_table_the_call_leaves_open_counts_as_read():
    """The run spells the unfilled field ``{}``; the statement assembled still spells it ``{0}``."""
    source = (
        "def f(self, pairs):\n"
        "    sqlC = 'SELECT j.PandaID FROM {0} j WHERE j.jediTaskID=:t '\n"
        "    for table in pairs:\n"
        "        self.cur.execute(sqlC.format(table) + comment, varMap)\n"
    )

    (stat,), unexplained = sqlwrite.read_coverage(
        [_module(source, "pandaserver/taskbuffer/db_proxy_mods/job_standalone_module.py")]
    )

    assert (stat.candidates, stat.explained) == (1, 1)
    assert unexplained == []


def test_a_query_the_call_fills_in_counts_as_read_by_its_own_runs_only():
    """A template is read when a run of the same name fills it; another name's run does not count."""
    source = (
        "def f(self, other):\n"
        "    sqlR = 'SELECT commandToPilot FROM ATLAS_PANDA.{0} WHERE PandaID=:PandaID '\n"
        "    for table in ('jobsDefined4', 'jobsActive4'):\n"
        "        self.cur.execute(sqlR.format(table) + comment, varMap)\n"
        "    sqlC2 = 'SELECT computingSite FROM ATLAS_PANDA.jobsArchived4 WHERE PandaID=:PandaID '\n"
        "    self.cur.execute(sqlC2 + comment, varMap)\n"
        "    sqlJ = 'SELECT {0} FROM ATLAS_PANDA.{1} WHERE PandaID=:PandaID '\n"
        "    other(sqlJ)\n"
    )

    (stat,), unexplained = sqlwrite.read_coverage(
        [_module(source, "pandaserver/taskbuffer/db_proxy_mods/job_standalone_module.py")]
    )

    # sqlR by its two fills, sqlC2 as written; sqlJ's holes fit sqlC2's run,
    # which is not one of its own.
    assert (stat.candidates, stat.explained) == (3, 2)
    assert [(one.table, one.reason) for one in unexplained] == [
        ("sqlJ", f"{sqlwrite.RUN_THROUGH_OTHER_FORM}: other")
    ]


def test_a_query_run_under_another_name_counts_as_read_and_one_nothing_here_runs_does_not():
    """Run, the denominator is what each name holds where the code uses it or the function ends.

    ``sql_exe = (sql_jt + comment) % column`` runs ``sql_jt``'s two statements:
    read, though no run is of ``sql_jt`` itself -- counted under both names, as
    written and as run, the way the reassembly counted them.  A statement handed to a call
    the reader does not accept, and one written and never used, are assembled
    here and not read here -- which is what the denominator is for.  The half
    a ``try`` had built when it raised is no statement, and a log line saying
    "select" is none either.
    """
    source = (
        "def f(self, column, flag):\n"
        "    sql_jt = 'SELECT a FROM ATLAS_PANDA.jobsActive4 WHERE %s=:b '\n"
        "    if flag:\n"
        "        sql_jt += 'AND c=:c '\n"
        "    sql_exe = (sql_jt + comment) % column\n"
        "    self.cur.execute(sql_exe, varMap)\n"
        "    sqlC = 'SELECT data FROM ATLAS_PANDA.Secrets WHERE owner=:owner '\n"
        "    self.getClobObj(sqlC, varMap)\n"
        "    try:\n"
        "        sqlA = 'SELECT PandaID FROM ATLAS_PANDA.jobsDefined4 '\n"
        "        sqlA += 'WHERE jobDefinitionID=:j '\n"
        "    except Exception:\n"
        "        pass\n"
        "    tmpLog = 'failed to select from ATLAS_PANDA.jobsDefined4'\n"
    )
    module = _module(source, "pandaserver/taskbuffer/db_proxy_mods/job_standalone_module.py")

    (stat,), unexplained = sqlwrite.read_coverage([module])
    assembled = sqlwrite._assembled_queries(module.tree.body[0])

    assert sorted(assembled.values()) == ["sqlA", "sqlC", "sql_exe", "sql_exe", "sql_jt", "sql_jt"]
    assert "SELECT PandaID FROM ATLAS_PANDA.jobsDefined4 " not in assembled
    assert (stat.candidates, stat.explained) == (6, 4)
    assert sorted((one.table, one.reason) for one in unexplained) == [
        ("sqlA", sqlwrite.NOT_RUN_HERE),
        ("sqlC", f"{sqlwrite.RUN_THROUGH_OTHER_FORM}: getClobObj"),
    ]


def test_a_run_statement_is_held_by_the_name_the_call_runs():
    """Run, ``sql_exe = (sql_jt + comment) % table`` is read where ``sql_exe`` runs, under that name.

    The reassembly that came before could not read the ``%`` and looked past ``sql_exe`` to
    ``sql_jt``; the run reads it.  A call with no bind map is a run with none,
    and the schema an f-string interpolates is a hole of the run.
    """
    source = (
        "def f(self, table):\n"
        "    sql_jt = f'SELECT a FROM {panda_config.schemaPANDA}.%s WHERE b=:b '\n"
        "    sql_exe = (sql_jt + comment) % table\n"
        "    self.cur.execute(sql_exe, varMap)\n"
        "    sqlN = 'SELECT c FROM ATLAS_PANDA.jobsActive4'\n"
        "    self.cur.execute(sqlN + comment)\n"
    )
    func = _func(source)

    run, plain = sql.executions(func, skipped := [])
    interpolated = sql.interpolations(func, "sql_exe")

    assert (run.variable, run.head) == ("sql_exe", run.sql)
    assert (plain.variable, plain.varmap, plain.sql) == ("sqlN", None, "SELECT c FROM ATLAS_PANDA.jobsActive4")
    assert skipped == []
    assert "panda_config.schemaPANDA" in interpolated


def test_an_unreadable_piece_leaves_a_hole_rather_than_dropping_the_statement():
    """A run-time operand is a hole, the same as one inside an f-string."""
    source = (
        "def f(self):\n"
        "    sqlU = 'UPDATE ATLAS_PANDA.JEDI_Tasks SET status=:status '\n"
        "    self.cur.execute(sqlU + where_clause(criteria) + comment, varMap)\n"
    )
    func = _func(source)

    texts = [run.sql for run in sql.executions(func)]

    assert texts == ["UPDATE ATLAS_PANDA.JEDI_Tasks SET status=:status {}"]


def test_a_write_tested_against_the_column_it_writes_records_the_guard():
    """``SET status=:status WHERE ... AND status IN (...)`` is a compare-and-set.

    Whoever else moved the row first wins, and this write changes no rows and
    says nothing -- so seeing the value decided in a log does not prove the row
    took it.  Predicates on columns the statement does *not* write are row
    identity, not a race, and are left out.
    """
    text = (
        "UPDATE ATLAS_PANDA.JEDI_Tasks SET status=:status,frozenTime=NULL "
        "WHERE jediTaskID=:jediTaskID AND status IN (:old_1,:old_2) "
    )

    statement = sqltext.writes(text)[0]

    assert statement.preconditions == {"status": "status IN (:old_1,:old_2)"}


def test_a_negated_guard_on_a_written_column_counts():
    """``NOT jobStatus=:ngStatus`` in ``updateJobStatus`` is the same race."""
    text = (
        "UPDATE ATLAS_PANDA.jobsActive4 SET jobStatus=:jobStatus "
        "WHERE PandaID=:PandaID AND NOT jobStatus=:ngStatus "
    )

    statement = sqltext.writes(text)[0]

    assert statement.preconditions == {"jobStatus": "NOT jobStatus=:ngStatus"}


def test_each_branch_carries_the_prior_value_its_own_arm_binds():
    """A resume and a pause through one statement require opposite rows."""
    source = '''
class TaskModule:
    def command(self, jediTaskID, resume):
        sqlU = f"UPDATE {panda_config.schemaJEDI}.JEDI_Tasks "
        # Two columns: ``status`` alone is declared by several specs, and an
        # ambiguous table is left unattributed.
        sqlU += "SET status=:newStatus,oldStatus=NULL "
        sqlU += "WHERE jediTaskID=:jediTaskID AND status=:oldStatus "
        varMap = {}
        if resume:
            varMap[":newStatus"] = "running"
            varMap[":oldStatus"] = "paused"
        else:
            varMap[":newStatus"] = "paused"
            varMap[":oldStatus"] = "running"
        self.cur.execute(sqlU + comment, varMap)
'''
    _s, junctions, _c, _u, _conf, _a = _sql_extract(source)
    (status,) = [j for j in junctions if j.subject == "JediTaskSpec.status"]

    assert {b.outcome: b.row_precondition for b in status.branches} == {
        "running": ["status=paused"],
        "paused": ["status=running"],
    }


def test_a_status_bound_for_another_tables_statement_is_no_arm_of_the_task():
    """The querier's round-4 note 9.  ``prepareTasksToBeFinished_JEDI`` binds
    ``:status`` to ``prepared`` for the output datasets and to the task's new
    status for the task; each statement's junction gets its own."""
    source = '''
class TaskModule:
    def prepareTask(self, jediTaskID, fileID, newTaskStatus):
        sqlF = f"UPDATE {panda_config.schemaJEDI}.JEDI_Dataset_Contents "
        sqlF += "SET status=:status WHERE fileID=:fileID "
        sqlT = f"UPDATE {panda_config.schemaJEDI}.JEDI_Tasks "
        sqlT += "SET status=:status WHERE jediTaskID=:jediTaskID "
        varMap = {}
        varMap[":fileID"] = fileID
        varMap[":status"] = "ready"
        self.cur.execute(sqlF + comment, varMap)
        varMap = {}
        varMap[":jediTaskID"] = jediTaskID
        varMap[":status"] = "prepared"
        self.cur.execute(sqlT + comment, varMap)
'''
    _s, junctions, _c, _u, _conf, _a = _sql_extract(source)
    outcomes = {j.subject: sorted(b.outcome for b in j.branches) for j in junctions}

    assert outcomes["JEDI_Tasks.status"] == ["prepared"]
    assert outcomes["JEDI_Dataset_Contents.status"] == ["ready"]


def test_a_write_nothing_races_for_records_no_guard():
    """An unguarded write lands whatever the row currently says."""
    text = "UPDATE ATLAS_PANDA.JEDI_Tasks SET status=:status WHERE jediTaskID=:jediTaskID "

    assert sqltext.writes(text)[0].preconditions == {}


def test_the_branch_carries_the_row_the_write_needed():
    """The guard reaches the branch, where prune reads it."""
    source = '''
class TaskModule:
    def finishTask(self, jediTaskID):
        sqlU = f"UPDATE {panda_config.schemaJEDI}.JEDI_Tasks "
        sqlU += "SET status=:status,oldStatus=:oldStatus "
        sqlU += "WHERE jediTaskID=:jediTaskID AND status=:priorStatus "
        varMap = {}
        varMap[":status"] = "finished"
        self.cur.execute(sqlU + comment, varMap)
'''
    _s, junctions, _c, _u, _conf, _a = _sql_extract(source)
    status = [j for j in junctions if j.subject == "JediTaskSpec.status"]

    assert len(status) == 1
    assert [b.outcome for b in status[0].branches] == ["finished"]
    assert status[0].branches[0].row_precondition == ["status=:priorStatus"]


# ---------------------------------------------------------------------------
# The line a junction leaves behind
# ---------------------------------------------------------------------------


_PROXY_CHAIN = [
    (
        '_logger = PandaLogger().getLogger("DBProxy")\n'
        "class DBProxy(task_module.TaskModule):\n    pass\n",
        "pandaserver/taskbuffer/OraDBProxy.py",
    ),
    (
        'logger = PandaLogger().getLogger(__name__.split(".")[-1])\n'
        "class DBProxy(OraDBProxy.DBProxy):\n    pass\n",
        "pandajedi/jedicore/JediDBProxy.py",
    ),
]


def _emits(
    source: str,
    rel: str = "pandajedi/jediorder/Knight.py",
    with_proxy_chain: bool = False,
):
    """Extract from *source*, resolve log files, then attach the emits."""
    modules = [
        _module(_SPECS, "pandaserver/taskbuffer/Specs.py"),
        _module(source, rel),
    ]
    if with_proxy_chain:
        modules += [_module(text, where) for text, where in _PROXY_CHAIN]
    subjects, junctions, _c, _d, _e = progress.extract(modules, MAP_ID, VERSION)
    fragment = MapFragment(map_id=MAP_ID, derived_from=VERSION)
    fragment.subjects.extend(subjects)
    fragment.junctions.extend(junctions)
    logfile.attach(fragment, modules)
    attributor = attribution.SpecAttributor(
        progress.spec_attributes(modules), attribution.class_bases(modules)
    )
    attributor.learn_element_types(modules)
    attributor.learn_self_attributes(modules)
    emit.attach(fragment, modules, attributor)
    return [j for j in fragment.junctions if j.owner.startswith(rel)]


def test_the_line_a_junction_writes_is_recorded_with_its_level():
    """``observe`` needs the sentence, not only the file it lands in."""
    source = '''
class Knight:
    def runImpl(self, taskSpec: JediTaskSpec):
        taskSpec.status = "finishing"
        tmpLog.info(f"set task_status={taskSpec.status}")
'''
    status = [j for j in _emits(source) if j.subject == "JediTaskSpec.status"]

    assert len(status) == 1
    assert [(e.template, e.log_level) for e in status[0].branches[0].emits] == [
        ("set task_status={}", "info")
    ]


def test_a_message_built_into_a_local_first_is_still_the_line():
    """``tmpMsg = f"..."`` then ``tmpLog.info(tmpMsg)`` -- 37% of the corpus.

    Without this hop all four knights that write ``set task_status=`` are
    missed, which is every one the transition gate reads.  The same shape gate
    1 found when ``criteria = "-link_unusable"`` was assigned before being
    interpolated.
    """
    source = '''
class Knight:
    def runImpl(self, taskSpec: JediTaskSpec):
        taskSpec.status = "finishing"
        tmpMsg = f"set task_status={taskSpec.status}"
        tmpLog.info(tmpMsg)
'''
    status = [j for j in _emits(source) if j.subject == "JediTaskSpec.status"]

    assert [e.template for e in status[0].branches[0].emits] == ["set task_status={}"]


def test_a_line_naming_the_value_through_a_local_counts():
    """``f"set task_status={newTaskStatus}"`` -- the value, not the attribute.

    ``TaskRefiner`` logs the local it just assigned rather than reading the
    spec back, so an attribute test alone does not see it.  What ties the line
    to the junction is that the local holds a value this junction writes.
    """
    source = '''
class Knight:
    def runImpl(self, taskSpec: JediTaskSpec):
        newTaskStatus = "tobroken"
        taskSpec.status = newTaskStatus
        tmpLog.info(f"set task_status={newTaskStatus}")
'''
    status = [j for j in _emits(source) if j.subject == "JediTaskSpec.status"]

    assert [e.template for e in status[0].branches[0].emits] == ["set task_status={}"]


def test_a_line_naming_the_expression_a_tier_two_outcome_is_named_for():
    """``runtime(newTaskStatus)`` and ``f"...={newTaskStatus}"`` are one fact.

    ``TaskRefiner`` assigns a local the reaching definitions cannot settle, so
    the outcome is recorded as the expression itself -- and the line it logs
    interpolates that same expression.  Without this the map has a branch it
    admits it cannot value and a line that would have shown the value, and
    fails to connect them.
    """
    source = '''
class Knight:
    def runImpl(self, taskSpec: JediTaskSpec, newTaskStatus):
        taskSpec.status = newTaskStatus
        tmpLog.info(f"set task_status={newTaskStatus} sourceLabel={self.label}")
'''
    status = [j for j in _emits(source) if j.subject == "JediTaskSpec.status"]

    assert [b.outcome for b in status[0].branches] == ["runtime(newTaskStatus)"]
    assert [e.template for e in status[0].branches[0].emits] == [
        "set task_status={} sourceLabel={}"
    ]


def test_the_line_reporting_how_many_rows_changed_is_kept_apart():
    """The only trace a compare-and-set leaves when it loses.

    ``updateTask_JEDI`` ends with ``updated {nRows} rows``, tagged with the
    task id and written at DEBUG, which production runs -- so whether a write
    landed is already observable and the map simply did not know the line.
    It carries no value, so it is marked as reporting the row count: read as a
    decision line it would build a probe for a value that never appears in it.
    """
    source = '''
class TaskModule:
    def updateTask_JEDI(self, taskSpec: JediTaskSpec):
        taskSpec.status = "finishing"
        self.cur.execute(sqlU + comment, varMap)
        nRows = self.cur.rowcount
        tmpLog.debug(f"updated {nRows} rows")
'''
    status = [j for j in _emits(source) if j.subject == "JediTaskSpec.status"]
    rows = [
        e
        for b in status[0].branches
        for e in b.emits
        if e.reports == REPORTS_ROWS_CHANGED
    ]

    assert [(e.template, e.log_level) for e in rows] == [("updated {} rows", "debug")]


def test_a_line_about_something_else_is_not_this_junctions():
    """Every function logs; only some of it is about the value being settled."""
    source = '''
class Knight:
    def runImpl(self, taskSpec: JediTaskSpec):
        taskSpec.status = "finishing"
        tmpLog.info(f"start for {self.pid}")
'''
    status = [j for j in _emits(source) if j.subject == "JediTaskSpec.status"]

    assert status[0].branches[0].emits == []


def test_the_line_lands_in_the_log_of_the_module_that_writes_it():
    """One junction can leave two lines in two files.

    ``updateTask_JEDI`` writes ``updated N rows`` into the proxy's own
    inherited logger, while the ``set task_status=`` line about the same
    junction is written by the knight that called it.  ``owns_logger`` answers
    that per junction; the truth is per line -- so the file belongs to the emit,
    resolved from the module the line is written in.
    """
    source = '''
class TaskModule:
    def updateTask_JEDI(self, taskSpec: JediTaskSpec):
        taskSpec.status = "finishing"
        tmpLog.debug(f"set task_status={taskSpec.status}")
'''
    proxy = _emits(
        source,
        "pandaserver/taskbuffer/db_proxy_mods/task_module.py",
        with_proxy_chain=True,
    )
    status = [j for j in proxy if j.subject == "JediTaskSpec.status"]

    assert status[0].branches[0].emits[0].log_files == [
        "panda-DBProxy.log",
        "panda-JediDBProxy.log",
    ]


def test_a_line_from_a_module_with_no_logger_of_its_own_names_no_file():
    """``PostProcessorBase`` logs through the wrapper its caller passed in.

    Nothing in the map says where that lands from here, and naming a file for
    it would be worse than naming none -- a probe against an invented file
    comes back empty, and an empty answer is what the eliminator reads.
    """
    source = '''
class PostProcessorBase:
    def doBasicPostProcess(self, taskSpec: JediTaskSpec, tmpLog):
        taskSpec.status = "finished"
        tmpLog.info(f"set task_status={taskSpec.status}")
'''
    base = _emits(source, "pandajedi/jedipprocess/PostProcessorBase.py")
    status = [j for j in base if j.subject == "JediTaskSpec.status"]

    assert status[0].branches[0].emits[0].template == "set task_status={}"
    assert status[0].branches[0].emits[0].log_files == []


def test_an_emit_under_an_exclusive_branch_stays_on_that_branch():
    """Two arms, two lines, and neither is evidence for the other."""
    source = '''
class Knight:
    def runImpl(self, taskSpec: JediTaskSpec, broken):
        if broken:
            taskSpec.status = "tobroken"
            tmpLog.info(f"set task_status={taskSpec.status} broken")
        else:
            taskSpec.status = "finishing"
            tmpLog.info(f"set task_status={taskSpec.status} ok")
'''
    status = [j for j in _emits(source) if j.subject == "JediTaskSpec.status"]
    by_outcome = {b.outcome: [e.template for e in b.emits] for b in status[0].branches}

    assert by_outcome["tobroken"] == ["set task_status={} broken"]
    assert by_outcome["finishing"] == ["set task_status={} ok"]


def test_a_tagged_line_goes_to_the_branch_that_names_those_tags():
    """Six arms of ``setScoutJobData_JEDI`` write ``exhausted`` and none of
    their lines interpolates the value -- they name the reason instead.  Without
    reading the tag the deciding junction has no line to ask production for, and
    reading it by anything looser gives all six arms the same line.
    """
    source = '''
class Knight:
    def scout(self, taskSpec: JediTaskSpec, scoutData, thr):
        if scoutData["cpuTime"] > thr:
            errMsg = f"#KV #ATM action=set_exhausted reason=scout_cpuTime {thr}"
            tmpLog.info(errMsg)
            taskSpec.status = "exhausted"
        if taskSpec.status != "exhausted":
            if scoutData["ramCount"] > thr:
                errMsg = f"#KV #ATM action=set_exhausted reason=scout_ramCount {thr}"
                tmpLog.info(errMsg)
                taskSpec.status = "exhausted"
'''
    status = [j for j in _emits(source) if j.subject == "JediTaskSpec.status"]
    lines = [[e.template for e in b.emits] for b in status[0].branches]

    assert lines == [
        ["#KV #ATM action=set_exhausted reason=scout_cpuTime {}"],
        ["#KV #ATM action=set_exhausted reason=scout_ramCount {}"],
    ]


def test_an_untagged_line_does_not_displace_a_tag():
    """A branch that names its own decision is not also described by a line that
    merely mentions the value.

    The line here is one ``_names_the_value`` accepts -- it interpolates the
    status -- and it is in the same block, so nothing else would keep it out.
    Letting it in would put arms that differ only by reason back on one shared
    sentence, which is the distinction this reading exists to make.
    """
    source = '''
class Knight:
    def scout(self, taskSpec: JediTaskSpec, thr):
        if thr:
            errMsg = f"#ATM action=set_exhausted reason=low_efficiency {thr}"
            tmpLog.info(errMsg)
            taskSpec.status = "exhausted"
            tmpLog.info(f"set task_status={taskSpec.status}")
'''
    status = [j for j in _emits(source) if j.subject == "JediTaskSpec.status"]

    assert [e.template for e in status[0].branches[0].emits] == [
        "#ATM action=set_exhausted reason=low_efficiency {}"
    ]


# ---------------------------------------------------------------------------
# The write that happens somewhere other than where the value was decided
# ---------------------------------------------------------------------------

_FLUSH_PROXY = '''
_logger = PandaLogger().getLogger("DBProxy")


class TaskModule:
    def updateTask_JEDI(self, taskSpec: JediTaskSpec, criteria, oldStatus=None):
        taskSpec.resetChangedAttr("jediTaskID")
        sqlU = f"UPDATE {panda_config.schemaJEDI}.JEDI_Tasks SET {taskSpec.bindUpdateChangesExpression()} "
        sql = "WHERE "
        varMap = {}
        for tmpKey, tmpVal in criteria.items():
            crKey = f":cr_{tmpKey}"
            sql += f"{tmpKey}={crKey} AND "
            varMap[crKey] = tmpVal
        if oldStatus is not None:
            sql += "status IN (:old_1) AND "
            varMap[":old_1"] = oldStatus[0]
        sql = sql[:-4]
        self.cur.execute(sqlU + sql + comment, varMap)
        nRows = self.cur.rowcount
        tmpLog.debug(f"updated {nRows} rows")
'''


def _flush(caller: str, rel: str = "pandajedi/jediorder/Knight.py"):
    """Run the attribute slice over *caller*, then annotate it from the proxy."""
    modules = [
        _module(_SPECS, "pandaserver/taskbuffer/Specs.py"),
        _module(_FLUSH_PROXY, "pandaserver/taskbuffer/db_proxy_mods/task_module.py"),
        _module(caller, rel),
    ]
    _subjects, junctions, _coverage, _diag, _enum = progress.extract(modules, MAP_ID, VERSION)
    attributor = attribution.SpecAttributor(
        progress.spec_attributes(modules), attribution.class_bases(modules)
    )
    attributor.learn_element_types(modules)
    attributor.learn_self_attributes(modules)
    attributor.learn_table_classes(modules)
    flush.attach(junctions, modules, attributor)
    return [j for j in junctions if j.owner.startswith(rel)]


def test_the_proxy_flush_is_recognised_by_the_expression_it_builds():
    """``SET {spec.bindUpdateChangesExpression()}`` is the method saying it flushes."""
    modules = [
        _module(_SPECS, "pandaserver/taskbuffer/Specs.py"),
        _module(_FLUSH_PROXY, "pandaserver/taskbuffer/db_proxy_mods/task_module.py"),
    ]
    attributor = attribution.SpecAttributor(
        progress.spec_attributes(modules), attribution.class_bases(modules)
    )
    attributor.learn_self_attributes(modules)

    found = flush.flush_methods(modules, attributor)

    assert set(found) == {"updateTask_JEDI"}
    assert found["updateTask_JEDI"].spec_classes == frozenset({"JediTaskSpec"})
    assert found["updateTask_JEDI"].excluded == frozenset({"jediTaskID"})
    assert found["updateTask_JEDI"].guards["status"] == (
        "status IN (:old_1)",
        frozenset({"oldStatus"}),
    )


def test_the_caller_is_told_where_its_flush_reports_the_row_count():
    """The decision is here; the line saying whether the row took it is there.

    ``updateTask_JEDI`` ends with ``updated {nRows} rows`` in the proxy's own
    log, and the knight that chose the value never sees it.  Carrying it to the
    caller's branches is what makes "did the write land" a question the map can
    point at a file for -- and it needs no change to PanDA, because the line is
    already written, already tagged with the task id, and already at a level
    this deployment runs.
    """
    caller = '''
class PostProcessor:
    def runImpl(self, taskSpec: JediTaskSpec):
        taskSpec.status = "finished"
        self.taskBufferIF.updateTask_JEDI(taskSpec, {"jediTaskID": 1})
'''
    status = [j for j in _flush(caller) if j.subject == "JediTaskSpec.status"]
    rows = [e for b in status[0].branches for e in b.emits if e.reports == REPORTS_ROWS_CHANGED]

    assert [(e.template, e.log_level) for e in rows] == [("updated {} rows", "debug")]
    assert rows[0].log_files == ["panda-DBProxy.log"]


def test_a_caller_that_asks_for_the_guard_is_told_its_write_races():
    """``oldStatus=`` is the caller opting into the compare-and-set."""
    caller = '''
class TaskRefiner:
    def runImpl(self, taskSpec: JediTaskSpec, taskStatus):
        taskSpec.status = "tobroken"
        self.taskBufferIF.updateTask_JEDI(taskSpec, {"jediTaskID": 1}, oldStatus=[taskStatus])
'''
    status = [j for j in _flush(caller) if j.subject == "JediTaskSpec.status"]

    assert len(status) == 1
    assert status[0].branches[0].row_precondition == ["status IN (:old_1)"]


def test_a_caller_that_does_not_is_not():
    """The fragment is appended under ``if oldStatus is not None``.

    Reading the flush without that condition would tell ten of
    ``updateTask_JEDI``'s callers their write races when it does not -- and an
    invented caveat is the damaging direction, because it weakens exactly the
    candidates production confirmed.
    """
    caller = '''
class PostProcessor:
    def runImpl(self, taskSpec: JediTaskSpec):
        taskSpec.status = "finished"
        self.taskBufferIF.updateTask_JEDI(taskSpec, {"jediTaskID": 1})
'''
    status = [j for j in _flush(caller) if j.subject == "JediTaskSpec.status"]

    assert len(status) == 1
    assert status[0].branches[0].row_precondition == []


def test_a_column_the_flush_takes_back_out_is_not_one_it_writes():
    """``resetChangedAttr("jediTaskID")`` says the ``WHERE`` is row identity.

    Without it the guard identifying the row would read as a race, because
    ``jediTaskID`` is as much a declared column as ``status`` is.
    """
    caller = '''
class Knight:
    def runImpl(self, taskSpec: JediTaskSpec):
        taskSpec.jediTaskID = 1
        taskSpec.status = "finished"
        self.taskBufferIF.updateTask_JEDI(taskSpec, {"jediTaskID": 1}, oldStatus=["running"])
'''
    guards = {
        j.subject: sorted({g for b in j.branches for g in b.row_precondition})
        for j in _flush(caller)
    }

    assert guards["JediTaskSpec.status"] == ["status IN (:old_1)"]
    assert guards.get("JediTaskSpec.jediTaskID", []) == []


def _flush_methods(proxy: str) -> dict:
    modules = [
        _module(_SPECS, "pandaserver/taskbuffer/Specs.py"),
        _module(proxy, "pandaserver/taskbuffer/db_proxy_mods/task_module.py"),
    ]
    attributor = attribution.SpecAttributor(
        progress.spec_attributes(modules), attribution.class_bases(modules)
    )
    attributor.learn_self_attributes(modules)
    return flush.flush_methods(modules, attributor)


def test_a_guard_the_flush_formats_is_still_tied_to_its_parameter():
    """``updateTask_JEDI`` formats its ``status IN`` term from the names the in-list helper returns.

    The term's text, ``status IN (:old___in)``, is in none of the method's
    literals, so looking for it there tied it to no parameter, and every caller
    -- the ones that pass no ``oldStatus`` too -- was told its write races.
    """
    proxy = '''
class TaskModule:
    def updateTask_JEDI(self, taskSpec: JediTaskSpec, criteria, oldStatus=None):
        sqlU = f"UPDATE {panda_config.schemaJEDI}.JEDI_Tasks SET {taskSpec.bindUpdateChangesExpression()} "
        sql = "WHERE jediTaskID=:cr_jediTaskID AND "
        if oldStatus is not None:
            names, old_map = get_sql_IN_bind_variables(oldStatus, prefix=":old_", value_as_suffix=True)
            sql += f"status IN ({names}) AND "
        sql = sql[:-4]
        self.cur.execute(sqlU + sql + comment, varMap)
'''
    term, gating = _flush_methods(proxy)["updateTask_JEDI"].guards["status"]

    assert "status IN (" in term
    assert gating == frozenset({"oldStatus"})


def test_a_guard_is_not_tied_to_a_test_above_the_same_text_elsewhere():
    """``registerTaskInOneShot_JEDI`` flushes under no test; another statement with the same term is under one."""
    proxy = '''
class TaskModule:
    def registerTask(self, taskSpec: JediTaskSpec, taskParams):
        sql = f"UPDATE {panda_config.schemaJEDI}.JEDI_Tasks SET {taskSpec.bindUpdateChangesExpression()} WHERE "
        sql += "jediTaskID=:jediTaskID AND status=:preStatus "
        self.cur.execute(sql + comment, varMap)
        if taskParams is not None:
            sql = f"UPDATE {panda_config.schemaJEDI}.JEDI_TaskParams SET taskParams=:taskParams "
            sql += "WHERE jediTaskID=:jediTaskID AND status=:preStatus "
            self.cur.execute(sql + comment, varMap)
'''
    guards = _flush_methods(proxy)["registerTask"].guards

    assert guards["status"] == ("status=:preStatus", frozenset())


def test_a_set_the_method_left_empty_is_not_a_flush():
    """``updateJob`` appends a ``SET`` under ``if`` / ``elif`` with no ``else``: on the third path it is empty.

    An empty clause looked like the flush's hole, and that statement's
    ``WHERE`` -- another table's row -- was read as the flush's guards.
    """
    proxy = '''
class TaskModule:
    def updateTask(self, taskSpec: JediTaskSpec, cType):
        sqlU = f"UPDATE {panda_config.schemaJEDI}.JEDI_Tasks SET {taskSpec.bindUpdateChangesExpression()} "
        sqlU += "WHERE jediTaskID=:jediTaskID "
        self.cur.execute(sqlU + comment, varMap)
        sqlD = f"UPDATE {panda_config.schemaJEDI}.JEDI_Datasets SET "
        if cType == "hold":
            sqlD += "nFilesOnHold=nFilesOnHold+:diffNum "
        elif cType == "touse":
            sqlD += "nFilesTobeUsed=nFilesTobeUsed+:diffNum "
        sqlD += "WHERE jediTaskID=:jediTaskID AND datasetID=:datasetID "
        self.cur.execute(sqlD + comment, varMap)
'''
    guards = _flush_methods(proxy)["updateTask"].guards

    assert set(guards) == {"jediTaskID"}


def test_table_class_is_inferred_from_the_column_names():
    """Nothing declares which spec a table holds; the columns give it away."""
    _s, _j, _c, _u, conflicts, attributor = _sql_extract(_SQL_SOURCE)

    assert conflicts == {}
    assert attributor.class_for_table("JEDI_Tasks") == "JediTaskSpec"


def test_a_table_named_in_another_case_is_the_same_table():
    """SQL identifiers are case-insensitive, and the corpus uses both spellings.

    ``reassignShare`` runs its update over ``["jobsactive4", "jobsdefined4"]``
    while every other statement writes ``jobsActive4``.  Matching the spelling
    exactly made those two look like tables holding no spec, which put the
    ``gshare`` write on a table-qualified subject instead of ``JobSpec``.
    """
    _s, _j, _c, _u, _conf, attributor = _sql_extract(_SQL_SOURCE)

    assert attributor.class_for_table("jedi_tasks") == "JediTaskSpec"
    assert attributor.class_for_table("JEDI_TASKS") == "JediTaskSpec"
    assert attributor.class_for_table("no_such_table") is None


def test_the_statement_can_name_the_class_outright():
    """``INSERT INTO filesTable4 ({FileSpec.columnNames()})`` says it in the f-string.

    The stronger source, and the only one that settles ``filesTable4``: every
    ``UPDATE`` on it sets columns ``FileSpec`` and ``JediFileSpec`` both
    declare, so no column set ever separates the two.
    """
    source = (
        "def insert_file(self):\n"
        "    sqlF = f'INSERT INTO ATLAS_PANDA.filesTable4 ({FileSpec.columnNames()}) '\n"
        "    self.cur.execute(sqlF + comment, varMap)\n"
    )
    _s, _j, _c, _u, conflicts, attributor = _sql_extract(source)

    assert conflicts == {}
    assert attributor.class_for_table("filesTable4") == "FileSpec"


def test_bind_writes_become_branches_with_their_conditions():
    """The junction is anchored at the bind, where the value and its ``if`` are."""
    _s, junctions, _c, _u, _conf, _a = _sql_extract(_SQL_SOURCE)
    status = [j for j in junctions if j.subject == "JediTaskSpec.status"]

    assert len(status) == 1
    by_outcome = {b.outcome: b.path_condition for b in status[0].branches}
    assert by_outcome["tobroken"] == ["broken"]
    assert by_outcome["finishing"] == ["not (broken)"]
    assert status[0].attribution == "certain"


def test_a_value_decided_at_run_time_is_recorded_not_dropped():
    """The writer is known even when the value is not."""
    source = (
        "class M:\n"
        "    def f(self, newStatus):\n"
        "        sqlU = 'UPDATE ATLAS_PANDA.JEDI_Tasks SET status=:status,oldStatus=:oldStatus '\n"
        "        varMap = {}\n"
        "        varMap[':status'] = newStatus\n"
        "        self.cur.execute(sqlU + comment, varMap)\n"
    )
    _s, junctions, _c, _u, _conf, _a = _sql_extract(source)

    branch = junctions[0].branches[0]
    assert branch.tier == 2
    assert branch.outcome == "runtime(newStatus)"


def test_clearing_a_column_is_spelled_the_way_the_attribute_slice_spells_it():
    """``SET oldStatus=NULL`` and ``taskSpec.oldStatus = None`` are one fact.

    They reach the same subject, so two spellings would make the branch table
    offer two outcomes where the source has one -- and everything downstream
    compares an outcome to a value observed through Python.
    """
    sql_source = (
        "class M:\n"
        "    def f(self):\n"
        "        sqlU = 'UPDATE ATLAS_PANDA.JEDI_Tasks SET oldStatus=NULL '\n"
        "        self.cur.execute(sqlU + comment, {})\n"
    )
    _s, junctions, _c, _u, _conf, _a = _sql_extract(sql_source)
    from_sql = [(b.outcome, b.tier) for b in junctions[0].branches]

    attribute_source = (
        "from pandaserver.taskbuffer.Specs import JediTaskSpec\n"
        "class M:\n"
        "    def f(self):\n"
        "        taskSpec = JediTaskSpec()\n"
        "        taskSpec.oldStatus = None\n"
    )
    _s, attribute_junctions, _c = _progress(attribute_source, "pandajedi/jediorder/W.py")
    from_attribute = [(b.outcome, b.tier) for b in attribute_junctions[0].branches]

    assert from_sql == from_attribute == [("None", 1)]


def test_a_number_settles_a_column_as_firmly_as_a_quoted_string():
    """Quoting is a property of the type, not of how decided the value is."""
    source = (
        "class M:\n"
        "    def f(self):\n"
        "        sqlU = 'UPDATE ATLAS_PANDA.JEDI_Tasks SET coreCount=0 '\n"
        "        self.cur.execute(sqlU + comment, {})\n"
    )
    _s, junctions, _c, _u, _conf, _a = _sql_extract(source)

    assert [(b.outcome, b.tier) for b in junctions[0].branches] == [("0", 1)]


def test_what_the_database_itself_decides_is_a_branch_not_a_dropped_write():
    """The clock and the row's own contents are tier 2, on the same grounds a
    bind filled at run time is: the writer is known, and that is what prune
    reads.

    Dropping them was not neutral.  Promotion weighs a subject's literal writes
    against all of them, so a counter the source only ever increments looked,
    from its lone ``= 0``, like a field with a closed vocabulary.
    """
    source = (
        "class M:\n"
        "    def f(self):\n"
        "        sqlU = 'UPDATE ATLAS_PANDA.JEDI_Datasets SET nFiles=nFiles+:iFiles,'\n"
        "        sqlU += 'modificationTime=CURRENT_DATE '\n"
        "        self.cur.execute(sqlU + comment, {})\n"
    )
    _s, junctions, _c, _u, _conf, _a = _sql_extract(source)
    outcomes = {j.subject: [(b.outcome, b.tier) for b in j.branches] for j in junctions}

    # Table-qualified: neither column name belongs to one declaring class, so
    # the statement cannot say which spec owns the row and the table does.
    assert outcomes["JEDI_Datasets.nFiles"] == [("runtime(nFiles+:iFiles)", 2)]
    assert outcomes["JEDI_Datasets.modificationTime"] == [("runtime(CURRENT_DATE)", 2)]


def test_a_bind_filled_with_a_non_string_constant_is_settled():
    """``varMap[':frozenTime'] = None`` is decided, and ``runtime(None)`` --
    which is what the slice used to record -- says it is not."""
    source = (
        "class M:\n"
        "    def f(self):\n"
        "        sqlU = 'UPDATE ATLAS_PANDA.JEDI_Tasks SET frozenTime=:frozenTime '\n"
        "        varMap = {}\n"
        "        varMap[':frozenTime'] = None\n"
        "        self.cur.execute(sqlU + comment, varMap)\n"
    )
    _s, junctions, _c, _u, _conf, _a = _sql_extract(source)

    assert [(b.outcome, b.tier) for b in junctions[0].branches] == [("None", 1)]


def test_a_bind_filled_from_a_local_is_resolved_like_an_attribute_write_is():
    """The same reading the attribute slice makes, on the slice where most of
    PanDA's writes actually are: a knight decides and a proxy method binds.

    Both sets of guards are carried.  The ones reaching ``varMap[...] = local``
    say the write happened; the ones on the assignment that gave the local its
    value say which value -- and neither is derivable from the other's node.
    """
    source = (
        "class M:\n"
        "    def f(self, toSkip):\n"
        "        sqlU = 'UPDATE ATLAS_PANDA.JEDI_Tasks SET status=:status,oldStatus=:oldStatus '\n"
        "        if toSkip:\n"
        "            newStatus = 'scouted'\n"
        "        else:\n"
        "            newStatus = 'running'\n"
        "        if self.ready:\n"
        "            varMap = {}\n"
        "            varMap[':status'] = newStatus\n"
        "            self.cur.execute(sqlU + comment, varMap)\n"
    )
    _s, junctions, _c, _u, _conf, _a = _sql_extract(source)
    status = [j for j in junctions if j.subject == "JediTaskSpec.status"]

    assert [(b.outcome, b.tier) for b in status[0].branches] == [
        ("scouted", 1),
        ("running", 1),
    ]
    assert status[0].branches[0].path_condition == ["self.ready", "toSkip"]
    assert status[0].branches[1].path_condition == ["self.ready", "not (toSkip)"]


def test_a_bind_filled_from_a_declared_mapping_resolves_to_its_values():
    """``newTaskStatus = commandStatusMap[commandStr]["doing"]`` reaches the
    database through a bind, which is where the statuses only this mapping
    produces actually live."""
    source = _IMPORTS_TASK_SPEC + (
        "class M:\n"
        "    def f(self, commandStr):\n"
        "        sqlU = 'UPDATE ATLAS_PANDA.JEDI_Tasks SET status=:status,oldStatus=:oldStatus '\n"
        "        varMap = {}\n"
        "        varMap[':status'] = JediTaskSpec.commandStatusMap()[commandStr]['doing']\n"
        "        self.cur.execute(sqlU + comment, varMap)\n"
    )
    modules = [
        _module(_SPECS, "pandaserver/taskbuffer/Specs.py"),
        _module(_COMMAND_MAP, "pandaserver/taskbuffer/JediTaskSpec.py"),
        _module(source, "pandaserver/taskbuffer/db_proxy_mods/task_module.py"),
    ]
    attributor = attribution.SpecAttributor(
        progress.spec_attributes(modules), attribution.class_bases(modules)
    )
    attributor.learn_table_classes(modules)
    with _with_calls(modules):
        _s, junctions, _c, _u, _d, _x, _stops = sqlwrite.extract(
            modules, MAP_ID, VERSION, attributor
        )
    status = [j for j in junctions if j.subject == "JediTaskSpec.status"]

    assert {(b.outcome, b.tier) for b in status[0].branches} == {
        ("aborting", 1),
        ("finishing", 1),
        ("paused", 1),
    }


def test_a_table_with_no_spec_is_qualified_by_the_table():
    """A subject's key needs a qualifier that disambiguates, not a Python class.

    Requiring a spec class was doing filtering work it should not: it dropped
    ``ddm_endpoint.blacklisted``, the writer behind a blacklisted RSE, along
    with the bookkeeping.  What keeps bookkeeping out is promotion.
    """
    source = (
        "class M:\n"
        "    def f(self):\n"
        "        sqlU = 'UPDATE ATLAS_PANDA.ddm_endpoint SET blacklisted=:blacklisted '\n"
        "        varMap = {}\n"
        "        varMap[':blacklisted'] = 'Y'\n"
        "        self.cur.execute(sqlU + comment, varMap)\n"
    )
    subjects, junctions, _c, uncovered, _conf, _a = _sql_extract(source)

    assert junctions[0].subject == "ddm_endpoint.blacklisted"
    assert subjects[0].qualifier_kind == "table"
    assert uncovered == {"ddm_endpoint"}


def test_a_bind_filled_by_a_dict_literal_is_read():
    """``var_map = {":blacklisted": "Y"}`` binds exactly as the subscript form does.

    Reading only ``varMap[":x"] = v`` left 176 of the corpus's 1261 written
    columns with no outcome -- 56% of everything this slice could not explain,
    and the whole of the harvester, worker and data-carousel side of the
    system, which writes its binds this way and was therefore invisible.
    """
    source = (
        "class M:\n"
        "    def f(self):\n"
        "        sqlU = 'UPDATE ATLAS_PANDA.ddm_endpoint SET blacklisted=:blacklisted '\n"
        "        varMap = {':blacklisted': 'Y'}\n"
        "        self.cur.execute(sqlU + comment, varMap)\n"
    )
    _subjects, junctions, _c, _uncovered, _conf, _a = _sql_extract(source)

    assert junctions[0].subject == "ddm_endpoint.blacklisted"
    assert [(b.outcome, b.tier) for b in junctions[0].branches] == [("Y", 1)]


def test_a_dict_literal_bind_carries_the_guard_that_chose_the_dict():
    """The condition on the write is the one on the statement holding the dict."""
    source = (
        "class M:\n"
        "    def f(self, broken):\n"
        "        sqlU = 'UPDATE ATLAS_PANDA.ddm_endpoint SET blacklisted=:blacklisted '\n"
        "        if broken:\n"
        "            varMap = {':blacklisted': 'Y'}\n"
        "        else:\n"
        "            varMap = {':blacklisted': 'N'}\n"
        "        self.cur.execute(sqlU + comment, varMap)\n"
    )
    _subjects, junctions, _c, _uncovered, _conf, _a = _sql_extract(source)

    assert sorted(
        (b.outcome, tuple(b.path_condition)) for b in junctions[0].branches
    ) == [("N", ("not (broken)",)), ("Y", ("broken",))]


def test_a_dict_literal_bind_whose_value_is_a_name_still_names_the_writer():
    """126 of the 176 hold a name, so tier 2 is the common answer, not the rare one."""
    source = (
        "class M:\n"
        "    def f(self, status):\n"
        "        sqlU = 'UPDATE ATLAS_PANDA.ddm_endpoint SET blacklisted=:blacklisted '\n"
        "        varMap = {':blacklisted': status}\n"
        "        self.cur.execute(sqlU + comment, varMap)\n"
    )
    _subjects, junctions, _c, _uncovered, _conf, _a = _sql_extract(source)

    assert [(b.outcome, b.tier) for b in junctions[0].branches] == [("runtime(status)", 2)]


def test_a_spec_backed_table_keeps_the_class_as_its_qualifier():
    """``jobsActive4`` and ``jobsArchived4`` are one JobSpec.jobStatus, not two.

    Keying on the table would split a subject across a job's lifetime, and
    would file an attribute write and a SQL write to the same field under
    different names.
    """
    source = (
        "class M:\n"
        "    def f(self):\n"
        "        sqlA = 'UPDATE ATLAS_PANDA.jobsActive4 SET jobStatus=:jobStatus,PandaID=:p '\n"
        "        varMap = {}\n"
        "        varMap[':jobStatus'] = 'running'\n"
        "        self.cur.execute(sqlA + comment, varMap)\n"
    )
    subjects, junctions, _c, _u, _conf, _a = _sql_extract(source)

    assert junctions[0].subject == "JobSpec.jobStatus"
    assert [s.qualifier_kind for s in subjects if s.attribute == "jobStatus"] == ["spec"]


# --------------------------------------------------------------------------- #
# promotion
# --------------------------------------------------------------------------- #


def _fragment_with(*subject_specs):
    """Build a fragment of subjects and their junctions for promotion tests."""
    subjects, junctions = [], []
    for qualifier, attribute, outcomes in subject_specs:
        name = f"{qualifier}.{attribute}"
        subjects.append(
            SubjectNode(
                map_id=MAP_ID,
                derived_from=VERSION,
                name=name,
                spec_class=qualifier,
                attribute=attribute,
            )
        )
        junctions.append(
            JunctionNode(
                map_id=MAP_ID,
                derived_from=VERSION,
                name=f"j:{name}",
                subject=name,
                owner="x.py::f",
                branches=[Branch(outcome=o, tier=t) for o, t in outcomes],
            )
        )
    return MapFragment(map_id=MAP_ID, derived_from=VERSION, subjects=subjects, junctions=junctions)


def test_a_predicate_against_literals_is_a_state_gate():
    """``WHERE t.status IN ('ready','running')`` gates another component."""
    modules = [_module("sql = \"SELECT x FROM t WHERE t.status IN ('ready','running') \"\n", "x.py")]

    assert promotion.gated_fields(modules)["status"] == 1


def test_a_predicate_against_bind_variables_is_a_lookup():
    """``WHERE PandaID=:PandaID`` selects a row, not a state.

    Without this ``lfn`` and ``jediTaskID`` rank first among subjects, which is
    how the narrowing was found.
    """
    modules = [
        _module('sql = "SELECT x FROM t WHERE PandaID=:PandaID "\n', "x.py"),
        _module('sql2 = "SELECT x FROM t WHERE fileID IN (:a,:b) "\n', "y.py"),
    ]
    gated = promotion.gated_fields(modules)

    assert gated["PandaID"] == 0
    assert gated["fileID"] == 0


def test_a_quote_inside_a_subquery_does_not_promote_the_outer_field():
    """``IN (SELECT ... WHERE x='y')`` -- the literal belongs to the subquery."""
    source = "sql = \"SELECT a FROM t WHERE jediTaskID IN (SELECT id FROM u WHERE type='x') \"\n"

    assert promotion.gated_fields([_module(source, "x.py")])["jediTaskID"] == 0


def test_one_literal_does_not_close_a_set():
    """An identifier is written from a row or a sequence, so it offers no set.

    This is what keeps ``jediTaskID`` out, measured: of the corpus's dropped
    subjects, 91 have an attribute ending in ``ID`` and not one of them reaches
    two distinct literal outcomes.
    """
    fragment = _fragment_with(
        ("JediTaskSpec", "jediTaskID", [("0", 1)] + [("x", 2)] * 8),
        ("JediTaskSpec", "status", [("ready", 1), ("running", 1), ("done", 1)]),
    )
    criteria = promotion.criteria_for(fragment, Counter(), {})

    assert "JediTaskSpec.jediTaskID" not in criteria
    assert criteria["JediTaskSpec.status"] == ["3:closed-literal-set"]


def test_runtime_writes_do_not_outvote_the_literals():
    """The literals close the set however many other writes the map can read.

    Weighing them against the junction count weighs them against how much of
    the corpus this build managed to read: ``JediTaskSpec.ramCount`` left the
    map when ``setScoutJobData_JEDI`` became readable, although PanDA had not
    changed.  Two distinct literals and eight runtime writes is the shape that
    used to fall out at exactly the old majority.
    """
    fragment = _fragment_with(
        ("JediTaskSpec", "errorDialog", [("a", 1), ("b", 1)] + [("x", 2)] * 8),
    )
    criteria = promotion.criteria_for(fragment, Counter(), {})

    assert criteria["JediTaskSpec.errorDialog"] == ["3:closed-literal-set"]


def test_a_subject_no_criterion_marks_is_kept_with_none():
    """The criteria say which subjects decide something; a reader asking about any written field still finds it.

    Kept as a filter, they hid two map defects -- crossings resolved by a bare name, no edge into
    ``JobSpec.__setattr__`` -- and dropped what they could not see decide (``JediTaskSpec.ramCount`` under
    criterion 3).  The count of the unmarked is what the build reports.
    """
    fragment = _fragment_with(
        ("JediTaskSpec", "status", [("ready", 1), ("done", 1)]),
        ("JobSpec", "modificationTime", [("now", 2)]),
    )
    unmarked = promotion.apply(fragment, promotion.criteria_for(fragment, Counter(), {}))

    assert unmarked == (1, 1)
    assert [(s.name, s.criteria) for s in fragment.subjects] == [
        ("JediTaskSpec.status", ["3:closed-literal-set"]),
        ("JobSpec.modificationTime", []),
    ]
    assert [j.subject for j in fragment.junctions] == ["JediTaskSpec.status", "JobSpec.modificationTime"]


def test_an_unresolved_junction_survives_promotion():
    """It has no subject to judge: kept as every junction is, and not counted among the unmarked."""
    fragment = _fragment_with(("JobSpec", "modificationTime", [("now", 2)]))
    fragment.junctions.append(
        JunctionNode(
            map_id=MAP_ID,
            derived_from=VERSION,
            name="j:?",
            subject="?.status",
            owner="x.py::f",
            attribution="unresolved",
            branches=[Branch(outcome="failed")],
        )
    )
    unmarked = promotion.apply(fragment, promotion.criteria_for(fragment, Counter(), {}))

    assert [j.subject for j in fragment.junctions] == ["JobSpec.modificationTime", "?.status"]
    assert unmarked == (1, 1)


def test_a_column_copied_from_another_becomes_a_passthrough_branch():
    """``SET status=oldStatus`` is how a task leaves ``pending``.

    The outcome names the source as a *subject*, not as a column, so the
    backward walk has somewhere to go: ``passthrough(oldStatus)`` would be a
    string, ``passthrough(JediTaskSpec.oldStatus)`` is an edge.
    """
    source = (
        "class TaskModule:\n"
        "    def release(self, jediTaskID):\n"
        "        sqlTU = f'UPDATE {schema}.JEDI_Tasks '\n"
        "        sqlTU += 'SET status=oldStatus,oldStatus=NULL '\n"
        "        sqlTU += 'WHERE jediTaskID=:jediTaskID '\n"
        "        varMap = {}\n"
        "        self.cur.execute(sqlTU + comment, varMap)\n"
    )
    _s, junctions, _c, _u, _conf, _a = _sql_extract(source)

    branches = [b for j in junctions if j.subject == "JediTaskSpec.status" for b in j.branches]
    assert [(b.outcome, b.tier) for b in branches] == [
        ("passthrough(JediTaskSpec.oldStatus)", 2)
    ]


def test_a_conditionally_appended_fragment_carries_its_own_condition():
    """A statement's text flattens the ``if``; the fragment is what still holds it.

    ``getTasksToExecCommand_JEDI`` appends ``SET status=:status`` or ``SET
    status=oldStatus`` in the two arms of one test, so anchoring an inline value
    at the ``execute`` would report no reason at all.
    """
    source = (
        "class TaskModule:\n"
        "    def exec_command(self, newTaskStatus):\n"
        "        sqlTU = f'UPDATE {schema}.JEDI_Tasks '\n"
        "        if newTaskStatus != 'dummy':\n"
        "            sqlTU += 'SET status=:status,'\n"
        "        else:\n"
        "            sqlTU += 'SET status=oldStatus,'\n"
        "        sqlTU += 'oldStatus=NULL '\n"
        "        varMap = {}\n"
        "        varMap[':status'] = 'running'\n"
        "        self.cur.execute(sqlTU + comment, varMap)\n"
    )
    _s, junctions, _c, _u, _conf, _a = _sql_extract(source)

    branches = {
        b.outcome: b.path_condition
        for j in junctions
        if j.subject == "JediTaskSpec.status"
        for b in j.branches
    }
    assert branches["passthrough(JediTaskSpec.oldStatus)"] == ["not (newTaskStatus != 'dummy')"]


def test_promotion_follows_a_passthrough_into_its_source():
    """A subject a promoted one copies from is worth asking about too.

    Otherwise the first hop of the backward walk out of the flagship symptom
    points at a subject the map does not contain.
    """
    fragment = _fragment_with(
        ("JediTaskSpec", "status", [("ready", 1), ("done", 1)]),
        ("JediTaskSpec", "oldStatus", [("x", 2)]),
    )
    fragment.junctions[0].branches.append(
        Branch(outcome="passthrough(JediTaskSpec.oldStatus)", tier=2)
    )
    criteria = promotion.criteria_for(fragment, Counter(), {})
    assert "JediTaskSpec.oldStatus" not in criteria

    closed = promotion.close_over_passthrough(fragment, criteria)
    promotion.apply(fragment, closed)

    assert closed["JediTaskSpec.oldStatus"] == ["4:carried-into-a-promoted-subject"]
    assert {s.name for s in fragment.subjects} == {
        "JediTaskSpec.status",
        "JediTaskSpec.oldStatus",
    }


# --------------------------------------------------------------------------- #
# reaching definitions for one local
# --------------------------------------------------------------------------- #


def _func(source: str, name: str = "f"):
    tree = ast.parse(source)
    from bamboo.codemap.panda.pathcond import attach_parents

    attach_parents(tree)
    return next(
        n
        for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name
    )


def _returned(func):
    """The ``return`` of *func*: the statement a returned local is read at."""
    return next(node for node in ast.walk(func) if isinstance(node, ast.Return))


def _held(func, name: str) -> list[str]:
    """Every statement the local *name* holds where the code uses it (:func:`sql.assembled_texts`)."""
    return sql.assembled_texts(func).get(name, [])


def test_a_guarded_chain_gives_each_value_its_own_condition():
    func = _func(
        "def f(self, spec):\n"
        "    if spec.status == 'tobroken':\n"
        "        status = 'broken'\n"
        "    elif spec.status == 'toabort':\n"
        "        status = 'aborted'\n"
        "    return status\n"
    )

    values = pathcond.literal_values(func, "status", at=_returned(func))

    assert [(literal, conditions) for literal, conditions, _line in values] == [
        ("broken", ["spec.status == 'tobroken'"]),
        (
            "aborted",
            ["not (spec.status == 'tobroken')", "spec.status == 'toabort'"],
        ),
    ]


def test_a_later_write_that_is_not_exclusive_becomes_a_condition():
    """The part dominating-guard analysis cannot see.  ``getFinalTaskStatus``
    decides a status in an if/elif chain and then rechecks twice at the end of
    the function, and either recheck can replace whatever the chain decided --
    so the chain's own guards are necessary and not sufficient.

    Comparing nesting depth or prefixes misses this: the rechecks sit at the
    top level, *shallower* than the branch they overwrite.
    """
    func = _func(
        "def f(self, spec):\n"
        "    if spec.status == 'toabort':\n"
        "        status = 'aborted'\n"
        "    else:\n"
        "        status = 'done'\n"
        "    if spec.is_hpo():\n"
        "        status = 'finished'\n"
        "    return status\n"
    )

    values = {
        literal: conditions
        for literal, conditions, _line in pathcond.literal_values(func, "status", at=_returned(func))
    }

    assert values["aborted"] == ["spec.status == 'toabort'", "not (spec.is_hpo())"]
    assert values["done"] == ["not (spec.status == 'toabort')", "not (spec.is_hpo())"]
    # The overwriting write itself has nothing after it.
    assert values["finished"] == ["spec.is_hpo()"]


def test_exclusive_siblings_do_not_negate_each_other():
    """Otherwise every branch of a chain carries the negation of every other:
    ``-dest_blacklisted`` came out requiring ``not (totalQueued >= limit)``, a
    condition with nothing to do with it."""
    func = _func(
        "def f(self, spec):\n"
        "    criteria = '-link_unusable'\n"
        "    if spec.blacklisted:\n"
        "        criteria = '-dest_blacklisted'\n"
        "    elif spec.queued >= spec.limit:\n"
        "        criteria = '-links_full'\n"
        "    return criteria\n"
    )

    values = {
        literal: conditions
        for literal, conditions, _line in pathcond.literal_values(func, "criteria", at=_returned(func))
    }

    assert values["-dest_blacklisted"] == ["spec.blacklisted"]
    assert values["-links_full"] == [
        "not (spec.blacklisted)",
        "spec.queued >= spec.limit",
    ]
    # The unconditional default carries both, because either replaces it --
    # each as every test it is under: the ``elif`` runs where the ``if`` did not.
    assert values["-link_unusable"] == [
        "not (spec.blacklisted)",
        "not (not (spec.blacklisted) and spec.queued >= spec.limit)",
    ]


# --------------------------------------------------------------------------- #
# declared mappings -- the value set behind a subscript
# --------------------------------------------------------------------------- #
#
# ``commandStatusMap()`` is the one declaration in the corpus that is complete
# rather than a sample: it *is* the command-to-status relation.  Two statuses
# PanDA declares exist nowhere else, so nothing else can account for them.
#
# Read through the corpus's calls, as a build reads it: the call lands in the
# method the receiver's class declares, and what that method returns is the
# mapping.

_COMMAND_MAP = """
class JediTaskSpec(object):
    _attributes = ("jediTaskID", "status", "oldStatus")

    def commandStatusMap(cls):
        return {
            "kill": {"doing": "aborting", "done": "toabort"},
            "finish": {"doing": "finishing", "done": "passed"},
            "pause": {"doing": "paused", "done": "dummy"},
        }

    commandStatusMap = classmethod(commandStatusMap)
"""

_IMPORTS_TASK_SPEC = "from pandaserver.taskbuffer.JediTaskSpec import JediTaskSpec\n"


def _with_calls(modules):
    """The corpus's calls over in-memory *modules*, for the length of a block."""
    from pathlib import Path

    roots = {package: Path("/nonexistent") / package for package in ("pandaserver", "pandajedi")}
    return sql.across_calls(sqlwrite.corpus_calls(modules, roots))


def _settled(caller: str, *others: tuple[str, str]) -> list[list[str]]:
    """What each assignment in ``f`` settles to, with the command map in the corpus."""
    modules = [
        _module(_COMMAND_MAP, "pandaserver/taskbuffer/JediTaskSpec.py"),
        *(_module(text, rel) for text, rel in others),
        _module(_IMPORTS_TASK_SPEC + caller, "pandaserver/taskbuffer/caller.py"),
    ]
    func = next(
        node
        for node in ast.walk(modules[-1].tree)
        if isinstance(node, ast.FunctionDef) and node.name == "f"
    )
    settle = values.resolver()
    with _with_calls(modules):
        return [settle(node.value, func) for node in ast.walk(func) if isinstance(node, ast.Assign)]


def test_a_run_time_key_enumerates_the_level_a_stated_one_narrows_it():
    """``[commandStr]["done"]`` cannot say which command ran, but it can say the
    six statuses a completed command leaves behind -- and ``["kill"]["doing"]``
    resolves to exactly one."""
    done, doing = _settled(
        "def f(self, commandStr):\n"
        "    a = JediTaskSpec.commandStatusMap()[commandStr]['done']\n"
        "    b = JediTaskSpec.commandStatusMap()['kill']['doing']\n"
    )

    assert done == ["dummy", "passed", "toabort"]
    assert doing == ["aborting"]


def test_a_local_holding_the_mapping_resolves_too():
    """``commandStatusMap = JediTaskSpec.commandStatusMap()`` sits at the top of
    the method that subscripts it a hundred lines further down."""
    _held, written = _settled(
        "def f(self, commandStr):\n"
        "    commandStatusMap = JediTaskSpec.commandStatusMap()\n"
        "    newTaskStatus = commandStatusMap[commandStr]['doing']\n"
    )

    assert written == ["aborting", "finishing", "paused"]


def test_a_loop_over_the_mapping_reaches_the_same_values():
    """The read side spells it the other way, and only the read side matters
    for "will anything pick this task up".

    ``for commandStr, taskStatusMap in commandStatusMap.items()`` binds one
    entry per turn, so ``taskStatusMap["doing"]`` is the same set of statuses
    ``commandStatusMap[commandStr]["doing"]`` gives.  Reading only the
    subscript is why every command-in-progress status looked like a value no
    query asks for -- a state a task enters and never leaves.
    """
    *_held, bound = _settled(
        "def f(self):\n"
        "    commandStatusMap = JediTaskSpec.commandStatusMap()\n"
        "    for commandStr, taskStatusMap in commandStatusMap.items():\n"
        "        varMap[':status'] = taskStatusMap['doing']\n"
    )

    assert bound == ["aborting", "finishing", "paused"]


def test_the_key_half_of_the_loop_is_not_a_value():
    """Binding it would enumerate command names as if they were statuses."""
    *_held, bound = _settled(
        "def f(self):\n"
        "    commandStatusMap = JediTaskSpec.commandStatusMap()\n"
        "    for commandStr, taskStatusMap in commandStatusMap.items():\n"
        "        varMap[':status'] = commandStr['doing']\n"
    )

    assert bound == []


def _settled_by_name(caller: str) -> dict[str, list[str]]:
    """:func:`_settled`, by the name each assignment in ``f`` binds, on a tree with its parents, as a build reads it."""
    modules = [
        _module(_COMMAND_MAP, "pandaserver/taskbuffer/JediTaskSpec.py"),
        _module(_IMPORTS_TASK_SPEC + caller, "pandaserver/taskbuffer/caller.py"),
    ]
    for module in modules:
        pathcond.attach_parents(module.tree)
    func = next(
        node
        for node in ast.walk(modules[-1].tree)
        if isinstance(node, ast.FunctionDef) and node.name == "f"
    )
    settle = values.resolver()
    with _with_calls(modules):
        return {
            node.targets[0].id: settle(node.value, func)
            for node in ast.walk(func)
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Subscript)
        }


def test_a_run_time_key_a_test_on_the_way_decides_reads_only_the_entries_it_leaves():
    """``retryTask_JEDI`` reads ``JediTaskSpec.commandStatusMap()[commStr]["done"]`` under ``if commStr ==
    "retry":`` and in its ``else``, past ``if commStr not in ["retry", "incexec"]: return``: the first can only be
    "ready", the second only "rerefine" -- not the six statuses of every command."""
    settled = _settled_by_name(
        "def f(self, commandStr):\n"
        "    if commandStr not in ['kill', 'finish']:\n"
        "        return\n"
        "    if commandStr == 'kill':\n"
        "        a = JediTaskSpec.commandStatusMap()[commandStr]['done']\n"
        "    else:\n"
        "        b = JediTaskSpec.commandStatusMap()[commandStr]['done']\n"
        "    c = JediTaskSpec.commandStatusMap()[commandStr]['doing']\n"
    )

    assert settled == {"a": ["toabort"], "b": ["passed"], "c": ["aborting", "finishing"]}


def test_a_key_bound_on_each_pass_before_its_test_is_read_as_the_test_reads_it():
    """``TaskCommando.runImpl`` takes ``commandStr`` from each command's row and reads ``[commandStr]["done"]``
    under ``if commandStr in ["kill", "finish", "reassign"]`` on the same pass: the next pass binds it again, and
    tests it again before the read."""
    settled = _settled_by_name(
        "def f(self, commands):\n"
        "    for command in commands:\n"
        "        commandStr = command['command']\n"
        "        if commandStr in ['kill', 'pause']:\n"
        "            a = JediTaskSpec.commandStatusMap()[commandStr]['done']\n"
    )

    assert settled["a"] == ["dummy", "toabort"]


def test_a_key_bound_again_between_its_test_and_the_read_is_not_read_as_tested():
    """What the test saw is not what the read reads: every entry stays."""
    inside = _settled_by_name(
        "def f(self, commandStr, other):\n"
        "    if commandStr == 'kill':\n"
        "        commandStr = other\n"
        "        a = JediTaskSpec.commandStatusMap()[commandStr]['done']\n"
    )
    past_an_exit = _settled_by_name(
        "def f(self, commandStr, other):\n"
        "    if commandStr != 'kill':\n"
        "        return\n"
        "    commandStr = other\n"
        "    a = JediTaskSpec.commandStatusMap()[commandStr]['done']\n"
    )

    assert inside["a"] == past_an_exit["a"] == ["dummy", "passed", "toabort"]


def test_a_key_the_tests_leave_no_entry_for_resolves_to_nothing():
    """The read runs only with a key the mapping lacks: as a stated key it lacks, two parts of the source disagree."""
    settled = _settled_by_name(
        "def f(self, commandStr):\n"
        "    if commandStr == 'resume':\n"
        "        a = JediTaskSpec.commandStatusMap()[commandStr]['done']\n"
    )

    assert settled["a"] == []


def test_a_mapping_with_one_computed_entry_resolves_to_nothing():
    """Elimination treats a short candidate list as complete, so a value set
    missing a member is worse than no value set: the caller then records an
    honest run-time branch instead of a closed set that is not closed."""
    (written,) = _settled(
        "from pandaserver.taskbuffer.spec import Spec\n"
        "def f(self):\n"
        "    a = Spec.statusMap()['kill']\n",
        (
            "class Spec(object):\n"
            "    def statusMap(cls):\n"
            "        return {'kill': 'aborting', 'finish': compute()}\n"
            "    statusMap = classmethod(statusMap)\n",
            "pandaserver/taskbuffer/spec.py",
        ),
    )

    assert written == []


def test_two_classes_declaring_one_name_each_answer_for_their_own_receiver():
    """Read by method name, ``A.statusMap()`` and ``B.statusMap()`` were one
    entry holding two different mappings, and both were dropped because a call
    site could not be told which it reached.  The receiver's class tells it."""
    first, second = _settled(
        "from pandaserver.taskbuffer.ab import A, B\n"
        "def f(self):\n"
        "    a = A.statusMap()['kill']\n"
        "    b = B.statusMap()['kill']\n",
        (
            "class A(object):\n"
            "    def statusMap(cls):\n"
            "        return {'kill': 'aborting'}\n"
            "class B(object):\n"
            "    def statusMap(cls):\n"
            "        return {'kill': 'toabort'}\n",
            "pandaserver/taskbuffer/ab.py",
        ),
    )

    assert (first, second) == (["aborting"], ["toabort"])


def test_a_method_taking_a_parameter_is_read_when_every_return_is_a_mapping():
    """Read by name, a parameter meant the mapping was built for a caller, and
    the method was passed over.  Followed through the call, each return is
    read, and a subscript reaches into all of them."""
    (written,) = _settled(
        "from pandaserver.taskbuffer.spec import Spec\n"
        "def f(self, flag):\n"
        "    a = Spec.statusMap(flag)['kill']\n",
        (
            "class Spec(object):\n"
            "    def statusMap(cls, flag):\n"
            "        if flag:\n"
            "            return {'kill': 'aborting'}\n"
            "        return {'kill': 'toabort'}\n",
            "pandaserver/taskbuffer/spec.py",
        ),
    )

    assert written == ["aborting", "toabort"]


def test_a_return_the_calls_cannot_follow_leaves_no_value_set():
    """``return other.statusMap()`` on one branch, with nothing saying what
    ``other`` is, would read as the literal branch alone."""
    (written,) = _settled(
        "from pandaserver.taskbuffer.spec import Spec\n"
        "def f(self, flag):\n"
        "    a = Spec.statusMap(flag)['kill']\n",
        (
            "class Spec(object):\n"
            "    def statusMap(cls, flag):\n"
            "        if flag:\n"
            "            return {'kill': 'aborting'}\n"
            "        return other.statusMap()\n",
            "pandaserver/taskbuffer/spec.py",
        ),
    )

    assert written == []


def test_a_stated_key_the_mapping_lacks_resolves_to_nothing():
    """Two parts of the source disagreeing is not this resolver's to settle."""
    (written,) = _settled("def f(self):\n    a = JediTaskSpec.commandStatusMap()['resume']['done']\n")

    assert written == []


def test_outside_a_build_a_mapping_is_not_read():
    """No calls, no claim: which method a call lands in is the calls' to say."""
    func = _func("def f(self):\n    a = JediTaskSpec.commandStatusMap()['kill']['doing']\n")
    written = next(node for node in ast.walk(func) if isinstance(node, ast.Assign))

    assert values.resolver()(written.value, func) == []


def test_assembled_text_is_indexed_and_a_bare_literal_is_not():
    """The index answers "this message was seen, who wrote it" -- which needs a
    frame with holes.  An exact message can be found by searching the source for
    itself, so a bare literal is not a template and does not earn a row.
    """
    source = (
        "class JobSpec(object):\n"
        "    _attributes = ('PandaID', 'jobStatus', 'ddmErrorDiag')\n"
        "    def fail(self, n):\n"
        "        self.ddmErrorDiag = f'failed to get {n} files'\n"
        "        self.jobStatus = 'failed'\n"
    )
    modules = [
        _module(_SPECS, "pandaserver/taskbuffer/Specs.py"),
        _module(source, "pandaserver/taskbuffer/JobSpec.py"),
    ]
    _s, _j, _c, diagnostics, _enum = progress.extract(modules, MAP_ID, VERSION)

    assert [(d.template, d.field, d.form) for d in diagnostics] == [
        ("failed to get {} files", "JobSpec.ddmErrorDiag", "attribute")
    ]


def test_a_frame_of_nothing_but_holes_is_not_a_search_key():
    """``setErrDiag`` appends with ``f"{self.errorDialog} {diag}"``, whose frame
    is two holes and a space: true, and matching nothing.  A row that cannot be
    searched for is noise in an index that exists to be searched."""
    source = (
        "class JediTaskSpec(object):\n"
        "    _attributes = ('jediTaskID', 'status', 'errorDialog')\n"
        "    def setErrDiag(self, diag):\n"
        "        self.errorDialog = f'{self.errorDialog} {diag}'\n"
    )
    modules = [
        _module(_SPECS, "pandaserver/taskbuffer/Specs.py"),
        _module(source, "pandaserver/taskbuffer/JediTaskSpec.py"),
    ]
    _s, _j, _c, diagnostics, _enum = progress.extract(modules, MAP_ID, VERSION)

    assert diagnostics == []


def test_a_bound_template_is_filed_against_the_column_not_the_bind():
    """On its own the bind is called ``:errDiag`` and could belong to any
    statement in the method; which column it lands in is the join the SQL slice
    has already made, which is why the index is collected there.

    Qualified by the table here because a one-column statement cannot say which
    spec the row holds -- the same fallback a write gets, so the two agree.
    """
    source = (
        "class M:\n"
        "    def f(self, site):\n"
        "        sqlU = 'UPDATE ATLAS_PANDA.JEDI_Tasks SET errorDialog=:errDiag '\n"
        "        varMap = {}\n"
        "        varMap[':errDiag'] = f'site {site} is unknown'\n"
        "        self.cur.execute(sqlU + comment, varMap)\n"
    )
    modules = [
        _module(_SPECS, "pandaserver/taskbuffer/Specs.py"),
        _module(source, "pandaserver/taskbuffer/db_proxy_mods/task_module.py"),
    ]
    attributor = attribution.SpecAttributor(
        progress.spec_attributes(modules), attribution.class_bases(modules)
    )
    attributor.learn_table_classes(modules)
    _s, _j, _c, _u, diagnostics, _x, _stops = sqlwrite.extract(
        modules, MAP_ID, VERSION, attributor
    )

    assert [(d.template, d.field, d.form) for d in diagnostics] == [
        ("site {} is unknown", "JEDI_Tasks.errorDialog", "bind")
    ]


def test_the_index_is_kept_beside_a_field_no_criterion_marks():
    """The whole point.  ``ddmErrorDiag`` satisfies no promotion criterion --
    correctly, since it has no value set to enumerate -- so its subject carries
    none.  The writes that assemble its text are still the answer to "who
    wrote this line", and an index makes no claim about the field that the
    criteria could contradict.
    """
    source = (
        "class JobSpec(object):\n"
        "    _attributes = ('PandaID', 'jobStatus', 'ddmErrorDiag')\n"
        "    def fail(self, n, why):\n"
        "        self.ddmErrorDiag = f'failed to get {n} files'\n"
        "        self.ddmErrorDiag = why\n"
    )
    modules = [
        _module(_SPECS, "pandaserver/taskbuffer/Specs.py"),
        _module(source, "pandaserver/taskbuffer/JobSpec.py"),
    ]
    subjects, junctions, _c, diagnostics, _enum = progress.extract(modules, MAP_ID, VERSION)
    fragment = MapFragment(map_id=MAP_ID, derived_from=VERSION)
    fragment.subjects.extend(subjects)
    fragment.junctions.extend(junctions)
    fragment.diagnostics.extend(diagnostics)

    promotion.apply(fragment, promotion.criteria_for(fragment, Counter(), {}))

    assert {s.name: s.criteria for s in fragment.subjects}["JobSpec.ddmErrorDiag"] == []
    assert [d.template for d in fragment.diagnostics] == ["failed to get {} files"]


# --------------------------------------------------------------------------- #
# the tag a branch names for itself
# --------------------------------------------------------------------------- #

_TAGGED = (
    "class M:\n"
    "    def setScoutJobData(self, taskSpec, scoutData, thr):\n"
    "        tmpLog = self.create_logger()\n"
    "        if scoutData['cpuTime'] > thr:\n"
    "            errMsg = f\"#KV #ATM action=set_exhausted reason=scout_cpuTime\"\n"
    "            errMsg += f\" cpuTime ({scoutData['cpuTime']}) is larger than {thr}\"\n"
    "            tmpLog.info(errMsg)\n"
    "            taskSpec.setErrDiag(errMsg)\n"
    "            taskSpec.status = 'exhausted'\n"
    "        if taskSpec.status != 'exhausted':\n"
    "            if scoutData['ramCount'] > thr:\n"
    "                other = f'#ATM action=set_exhausted reason=scout_ramCount {thr}'\n"
    "                tmpLog.info(other)\n"
    "                taskSpec.setErrDiag(other)\n"
    "                taskSpec.status = 'exhausted'\n"
)

_SETTER = (
    "class JediTaskSpec(object):\n"
    "    _attributes = ('jediTaskID', 'status', 'oldStatus', 'errorDialog')\n"
    "    def setErrDiag(self, diag, append=False):\n"
    "        self.errorDialog = diag\n"
)


def _tagged_modules(source: str = _TAGGED):
    return [
        _module(_SETTER, "pandaserver/taskbuffer/JediTaskSpec.py"),
        _module(source, "pandaserver/taskbuffer/db_proxy_mods/task_utils_module.py"),
    ]


def test_a_branch_carries_the_tag_the_block_records():
    """Which of six arms fired is the whole question for ``exhausted``, and the
    arms are not if/elif siblings -- each is its own ``if taskSpec.status !=
    'exhausted':`` -- so the path condition cannot separate them.  The block
    that records the line can: it holds exactly one write and names the reason.
    """
    _s, junctions, _c, _d, _e = progress.extract(_tagged_modules(), MAP_ID, VERSION)

    junction = next(j for j in junctions if j.subject == "JediTaskSpec.status")
    tags = [b.tags for b in junction.branches if b.outcome == "exhausted"]
    assert tags == [
        ["action=set_exhausted", "reason=scout_cpuTime"],
        ["action=set_exhausted", "reason=scout_ramCount"],
    ]


def test_a_branch_knows_the_line_of_its_own_write():
    """The junction's anchor is the first write; six branches sharing it answer
    "which branch" with a line belonging to another."""
    _s, junctions, _c, _d, _e = progress.extract(_tagged_modules(), MAP_ID, VERSION)

    junction = next(j for j in junctions if j.subject == "JediTaskSpec.status")
    lines = [b.line for b in junction.branches if b.outcome == "exhausted"]
    assert lines == [9, 15]
    assert junction.anchor.line_start == 9


def test_a_tag_assigned_in_a_sibling_arm_still_reaches_the_branch():
    """``reason=low_efficiency`` is assigned in the ``else`` of an unrelated
    check and only the local reaches the block that writes.  One hop is the
    same reading the emit pass needs, and without it the reason is lost."""
    source = (
        "class M:\n"
        "    def scout(self, taskSpec, io, limit):\n"
        "        tmpLog = self.create_logger()\n"
        "        if io > limit:\n"
        "            errMsg = 'not to set exhausted since high IO intensity'\n"
        "        else:\n"
        "            errMsg = f'#ATM action=set_exhausted reason=low_efficiency {io}'\n"
        "        if taskSpec.useExhausted():\n"
        "            tmpLog.info(errMsg)\n"
        "            taskSpec.setErrDiag(errMsg)\n"
        "            taskSpec.status = 'exhausted'\n"
    )
    _s, junctions, _c, _d, _e = progress.extract(_tagged_modules(source), MAP_ID, VERSION)

    junction = next(j for j in junctions if j.subject == "JediTaskSpec.status")
    assert [b.tags for b in junction.branches] == [
        ["action=set_exhausted", "reason=low_efficiency"]
    ]


def test_a_block_settling_two_attributes_names_both():
    """The message names the *block*, and the block ran, so both writes are in
    it.  ``doActionForReassign`` is this shape -- it logs
    ``action=trigger_new_brokerage by setting task_status={}`` beside a write to
    ``status`` and one to ``oldStatus`` -- and refusing there was the reading
    being over-cautious about a question that does not arise.
    """
    source = (
        "class M:\n"
        "    def reassign(self, taskSpec, n):\n"
        "        tmpLog = self.create_logger()\n"
        "        if n > 1:\n"
        "            tmpLog.info('#ATM action=trigger_new_brokerage by setting it')\n"
        "            taskSpec.oldStatus = 'scouting'\n"
        "            taskSpec.status = 'exhausted'\n"
    )
    _s, junctions, _c, _d, _e = progress.extract(_tagged_modules(source), MAP_ID, VERSION)

    settled = [j for j in junctions if j.owner.endswith("::reassign")]
    assert sorted(j.subject for j in settled) == [
        "JediTaskSpec.oldStatus",
        "JediTaskSpec.status",
    ]
    assert [b.tags for j in settled for b in j.branches] == [
        ["action=trigger_new_brokerage"],
        ["action=trigger_new_brokerage"],
    ]


def test_a_block_settling_the_same_attribute_twice_names_neither():
    """Here the question does arise: two writes of one attribute in one block,
    and nothing says which of them the message is about."""
    source = (
        "class M:\n"
        "    def scout(self, taskSpec, n):\n"
        "        tmpLog = self.create_logger()\n"
        "        if n > 1:\n"
        "            tmpLog.info('#ATM action=set_exhausted reason=low_efficiency')\n"
        "            taskSpec.status = 'scouting'\n"
        "            self.taskBufferIF.updateTask_JEDI(taskSpec)\n"
        "            taskSpec.status = 'exhausted'\n"
    )
    _s, junctions, _c, _d, _e = progress.extract(_tagged_modules(source), MAP_ID, VERSION)

    # Both land -- the task is saved between -- and the tag names neither.
    settled = [j for j in junctions if j.owner.endswith("::scout")]
    assert [b.tags for j in settled for b in j.branches] == [[], []]


def test_a_branch_keeps_the_untagged_frame_as_well():
    """Of thirty tasks found in ``exhausted`` with a message on the record, none
    carried a tag: they were retry refusals and the goal check, which write
    prose.  The frame is weaker evidence and the only evidence there is."""
    source = (
        "class M:\n"
        "    def check(self, taskSpec, rate, limit):\n"
        "        tmpLog = self.create_logger()\n"
        "        if rate > limit:\n"
        "            msg = f'exhausted upon retry since failed rate ({rate}) exceeds {limit}'\n"
        "            tmpLog.debug(msg)\n"
        "            taskSpec.status = 'exhausted'\n"
    )
    _s, junctions, _c, _d, _e = progress.extract(_tagged_modules(source), MAP_ID, VERSION)

    junction = next(j for j in junctions if j.subject == "JediTaskSpec.status")
    assert [b.tags for b in junction.branches] == [[]]
    assert [b.messages for b in junction.branches] == [
        ["exhausted upon retry since failed rate ({}) exceeds {}"]
    ]


def test_a_message_a_setter_persists_is_indexed_against_the_field_it_lands_in():
    """``setErrDiag`` is a method call, so the attribute slice never saw it --
    and it is how ``errorDialog`` is written 127 times in the corpus, which is
    the field an investigation actually starts from.

    One row per assembled piece, not one for the concatenation: ``errMsg +=``
    under a condition may not run, so composing them would claim an order the
    reading has not established.
    """
    _s, _j, _c, diagnostics, _e = progress.extract(_tagged_modules(), MAP_ID, VERSION)

    assert [(d.template, d.field, d.form) for d in diagnostics] == [
        ("#KV #ATM action=set_exhausted reason=scout_cpuTime", "JediTaskSpec.errorDialog", "alias"),
        (" cpuTime ({}) is larger than {}", "JediTaskSpec.errorDialog", "alias"),
        ("#ATM action=set_exhausted reason=scout_ramCount {}", "JediTaskSpec.errorDialog", "alias"),
    ]


def test_a_setter_two_classes_define_differently_indexes_neither():
    """``setDdmBackEnd`` writes ``JediTaskSpec.splitRule`` and
    ``JobSpec.specialHandling``; the name alone cannot say which field the text
    landed in.  The same restriction the trigger slice and ``callers_of`` put on
    every name-based hop, and the corpus has this collision today.
    """
    both = (
        "class JediTaskSpec(object):\n"
        "    _attributes = ('jediTaskID', 'status', 'splitRule')\n"
        "    def setDdmBackEnd(self, backEnd):\n"
        "        self.splitRule = backEnd\n"
        "class JobSpec(object):\n"
        "    _attributes = ('PandaID', 'specialHandling')\n"
        "    def setDdmBackEnd(self, backEnd):\n"
        "        self.specialHandling = backEnd\n"
    )
    caller = (
        "class M:\n"
        "    def f(self, taskSpec, n):\n"
        "        taskSpec.setDdmBackEnd(f'ddm {n}')\n"
    )
    modules = [
        _module(both, "pandaserver/taskbuffer/Specs.py"),
        _module(caller, "pandaserver/jediorder/TaskRefiner.py"),
    ]
    _s, _j, _c, diagnostics, _e = progress.extract(modules, MAP_ID, VERSION)

    assert diagnostics == []


_ENUM_WRITER = (
    "class JobSpec(object):\n"
    "    _attributes = ('PandaID', 'jobStatus', 'taskBufferErrorCode', 'pilotErrorCode')\n"
    "    def kill(self):\n"
    "        self.taskBufferErrorCode = ErrorCode.EC_Kill\n"
    "        self.pilotErrorCode = 0\n"
)


def test_a_field_is_bound_to_the_enumeration_that_decodes_it():
    """``jobSpec.taskBufferErrorCode = ErrorCode.EC_Kill`` states both halves.

    ``errorcode`` says this binding is "not recoverable from the constant's
    location", and it is right about the location -- the write is where it is
    recoverable from, and the corpus has 79 of them.
    """
    modules = [
        _module(_SPECS, "pandaserver/taskbuffer/Specs.py"),
        _module(_ENUM_WRITER, "pandaserver/taskbuffer/JobSpec.py"),
    ]
    _s, _j, _c, _d, bindings = progress.extract(
        modules, MAP_ID, VERSION, {"EC_Kill": "taskbuffer.ErrorCode.EC"}
    )

    assert [(b.field, b.constant, b.namespace) for b in bindings] == [
        ("JobSpec.taskBufferErrorCode", "EC_Kill", "taskbuffer.ErrorCode.EC")
    ]


def test_the_binding_is_kept_beside_a_field_no_criterion_marks():
    """No criterion fires on an error-code field -- every write is tier 2, the
    right-hand side being a constant -- so its subject carries none, and rightly.

    "Why is this field 100?" is not the question; "what does 100 mean here?"
    is, and an index makes no claim the criteria could contradict.
    """
    modules = [
        _module(_SPECS, "pandaserver/taskbuffer/Specs.py"),
        _module(_ENUM_WRITER, "pandaserver/taskbuffer/JobSpec.py"),
    ]
    subjects, junctions, _c, _d, bindings = progress.extract(
        modules, MAP_ID, VERSION, {"EC_Kill": "taskbuffer.ErrorCode.EC"}
    )
    fragment = MapFragment(map_id=MAP_ID, derived_from=VERSION)
    fragment.subjects.extend(subjects)
    fragment.junctions.extend(junctions)
    fragment.enumeration_writes.extend(bindings)

    promotion.apply(fragment, promotion.criteria_for(fragment, Counter(), {}))

    assert {s.name: s.criteria for s in fragment.subjects}["JobSpec.taskBufferErrorCode"] == []
    assert [b.constant for b in fragment.enumeration_writes] == ["EC_Kill"]


def test_a_constant_two_enumerations_share_is_not_a_binding():
    """A name that decodes to two namespaces decodes nothing, so it is dropped
    rather than guessed at -- the plugin only hands over the unambiguous ones."""
    modules = [
        _module(_SPECS, "pandaserver/taskbuffer/Specs.py"),
        _module(_ENUM_WRITER, "pandaserver/taskbuffer/JobSpec.py"),
    ]
    _s, _j, _c, _d, bindings = progress.extract(modules, MAP_ID, VERSION, {})

    assert bindings == []


def test_the_attribute_slice_reads_a_subscript_of_a_declared_mapping():
    """``TaskCommando.py:178`` -- the write that accounts for ``passed``; under its test, of the kill and finish
    commands alone (pause's ``dummy`` is no value it writes)."""
    caller = _IMPORTS_TASK_SPEC + (
        "def runImpl(self, tmpTaskSpec, commandStr):\n"
        "    if commandStr in ['kill', 'finish']:\n"
        "        tmpTaskSpec.status = JediTaskSpec.commandStatusMap()[commandStr]['done']\n"
    )
    modules = [
        _module(_SPECS, "pandaserver/taskbuffer/Specs.py"),
        _module(_COMMAND_MAP, "pandaserver/taskbuffer/JediTaskSpec.py"),
        _module(caller, "pandajedi/jediorder/TaskCommando.py"),
    ]
    with _with_calls(modules):
        _subjects, junctions, _cov, _diag, _enum = progress.extract(modules, MAP_ID, VERSION)
    written = [j for j in junctions if j.owner.endswith("::runImpl")]

    assert _outcomes(written) == [("passed", 1), ("toabort", 1)]


# --------------------------------------------------------------------------- #
# write alias
# --------------------------------------------------------------------------- #

_ON_HOLD = """
class JediTaskSpec(object):
    _attributes = ("jediTaskID", "status", "oldStatus")

    def setOnHold(self):
        if self.status in ["ready", "running"]:
            self.oldStatus = self.status
            self.status = "pending"
"""


def _alias_extract(*sources: tuple[str, str]):
    modules = [_module(_ON_HOLD, "pandaserver/taskbuffer/JediTaskSpec.py")]
    modules += [_module(text, rel) for text, rel in sources]
    attributor = attribution.SpecAttributor(
        progress.spec_attributes(modules), attribution.class_bases(modules)
    )
    return alias.extract(
        modules, MAP_ID, VERSION, progress.spec_attributes(modules), attributor
    )


def test_a_call_site_of_a_write_alias_is_a_junction():
    """The map must answer *why* ``pending``, not merely that ``setOnHold`` ran.

    Recorded only at the definition, the most contended value in the system has
    exactly one writer and no reason attached to it.
    """
    caller = (
        "def generate(self, taskSpec, inputChunk):\n"
        "    if not inputChunk.hasCandidates():\n"
        "        taskSpec.setOnHold()\n"
    )
    _subjects, junctions, _cov = _alias_extract((caller, "pandajedi/jediorder/JobGenerator.py"))

    assert [j.owner for j in junctions] == ["pandajedi/jediorder/JobGenerator.py::generate"]
    branch = junctions[0].branches[0]
    assert (junctions[0].subject, branch.outcome) == ("JediTaskSpec.status", "pending")
    assert branch.path_condition == [
        "not inputChunk.hasCandidates()",
        "self.status in ['ready', 'running']  [in setOnHold()]",
    ]


def test_the_aliases_own_guard_is_marked_as_its_own():
    """Reaching ``pending`` needs both conditions and they fail differently.

    A caller that never ran and a guard that refused the status are different
    diagnoses, and a flat conjunction cannot tell them apart.
    """
    caller = "def refine(self, taskSpec):\n    taskSpec.setOnHold()\n"
    _subjects, junctions, _cov = _alias_extract((caller, "pandajedi/jediorder/TaskRefiner.py"))

    assert junctions[0].branches[0].path_condition == [
        "self.status in ['ready', 'running']  [in setOnHold()]"
    ]


def test_only_literal_writes_make_a_method_an_alias():
    """Widened to any declared-attribute write, the rule matches serialization.

    On real PanDA that is 36 methods and 600 call sites led by ``__init__``,
    ``pack`` and ``setErrDiag`` -- none of which settle anything.
    """
    spec = (
        "class JediTaskSpec(object):\n"
        "    _attributes = ('jediTaskID', 'status', 'errorDialog')\n"
        "\n"
        "    def setErrDiag(self, diag):\n"
        "        self.errorDialog = diag\n"
    )
    modules = [_module(spec, "pandaserver/taskbuffer/JediTaskSpec.py")]

    assert alias.find_aliases(modules, progress.spec_attributes(modules)) == {}


def test_a_literal_the_alias_adds_to_is_written_at_its_calls_as_what_it_computes():
    """``setDatasetAttribute`` sets ``self.attributes = ""`` and then adds the label to it.

    Its callers' objects never hold ``""``, and each call still writes the field:
    what the method computes, ``runtime(setDatasetAttribute())``.  A literal a
    ``+=`` only may add to leaves where the ``+=`` does not run, and the ``+=``
    is a write of its own under its test.
    """
    spec = (
        "class JediDatasetSpec(object):\n"
        "    _attributes = ('datasetID', 'attributes')\n"
        "\n"
        "    def setDatasetAttribute(self, attr):\n"
        "        if self.attributes is None:\n"
        "            self.attributes = ''\n"
        "        else:\n"
        "            self.attributes += ','\n"
        "        self.attributes += attr\n"
        "\n"
        "    def setLabel(self, label, more):\n"
        "        self.attributes = label\n"
        "        self.attributes = 'plain'\n"
        "        if more:\n"
        "            self.attributes += ',more'\n"
    )
    caller = (
        "def refine(self, datasetSpec):\n"
        "    datasetSpec.setDatasetAttribute('nosplit')\n"
        "    save(datasetSpec)\n"
        "\n"
        "def label(self, datasetSpec):\n"
        "    datasetSpec.setLabel('x', True)\n"
        "    save(datasetSpec)\n"
    )
    _subjects, junctions, _cov = _alias_extract(
        (spec, "pandaserver/taskbuffer/JediDatasetSpec.py"),
        (caller, "pandajedi/jedirefine/TaskRefinerBase.py"),
    )

    branches = {
        j.owner.rsplit("::", 1)[-1]: [(b.outcome, b.tier, b.path_condition) for b in j.branches]
        for j in junctions
        if j.subject == "JediDatasetSpec.attributes"
    }
    assert branches == {
        "refine": [("runtime(setDatasetAttribute())", 2, [])],
        "label": [
            ("plain", 1, ["not (more)  [in setLabel()]"]),
            ("runtime(setLabel())", 2, ["more  [in setLabel()]"]),
        ],
    }


def test_a_plus_equals_that_adds_to_a_value_is_a_write_of_what_it_makes():
    """The attribute slice's side of the setter: ``self.attributes = ''`` never
    leaves ``setDatasetAttribute`` as written, and the ``+=`` that adds to it is
    the method's write -- what it makes, said ``runtime``.  A ``+=`` that may
    run leaves the value where it does not; a count from ``0`` is ``0`` only
    where no pass added to it, and the count otherwise.
    """
    spec = (
        "class JediDatasetSpec(object):\n"
        "    _attributes = ('datasetID', 'attributes', 'nFiles')\n"
        "\n"
        "    def setDatasetAttribute(self, attr):\n"
        "        if self.attributes is None:\n"
        "            self.attributes = ''\n"
        "        else:\n"
        "            self.attributes += ','\n"
        "        self.attributes += attr\n"
        "\n"
        "    def setLabel(self, more):\n"
        "        self.attributes = 'plain'\n"
        "        if more:\n"
        "            self.attributes += ',more'\n"
        "\n"
        "    def count(self, files):\n"
        "        self.nFiles = 0\n"
        "        for f in files:\n"
        "            if f.type == 'output':\n"
        "                self.nFiles += 1\n"
    )
    modules = [_module(spec, "pandaserver/taskbuffer/JediDatasetSpec.py")]
    _subjects, junctions, _c, _d, _e = progress.extract(modules, MAP_ID, VERSION)

    branches = {
        j.owner.rsplit("::", 1)[-1]: [(b.outcome, b.tier, b.path_condition, b.line) for b in j.branches]
        for j in junctions
    }
    assert branches == {
        "setDatasetAttribute": [("runtime(self.attributes + attr)", 2, [], 9)],
        "setLabel": [
            ("plain", 1, ["not (more)"], 12),
            ("runtime(self.attributes + ',more')", 2, ["more"], 14),
        ],
        "count": [
            ("0", 1, ["not any(f.type == 'output' for f in files)"], 17),
            ("runtime(self.nFiles + 1)", 2, ["f.type == 'output'"], 20),
        ],
    }


# --------------------------------------------------------------------------- #
# return alias -- a helper that returns the value the caller writes
# --------------------------------------------------------------------------- #

_FINAL_STATUS = """
class JediTaskSpec(object):
    _attributes = ("jediTaskID", "status", "oldStatus")
"""

_POST_PROCESSOR = """
class PostProcessorBase(object):
    def doBasicPostProcess(self, taskSpec):
        taskSpec.status = self.getFinalTaskStatus(taskSpec)

    def getFinalTaskStatus(self, taskSpec, checkGoal=False):
        if taskSpec.status == 'tobroken':
            status = 'broken'
        elif taskSpec.status == 'toabort':
            status = 'aborted'
        else:
            status = 'done'
        if taskSpec.is_hpo_workflow():
            status = 'finished'
        if checkGoal:
            return True
        return status
"""


def _producer_extract(*sources: tuple[str, str]):
    modules = [_module(_FINAL_STATUS, "pandaserver/taskbuffer/JediTaskSpec.py")]
    modules += [_module(text, rel) for text, rel in sources]
    declarations = progress.spec_attributes(modules)
    attributor = attribution.SpecAttributor(declarations, attribution.class_bases(modules))
    with _with_calls(modules):
        return alias.extract_producers(modules, MAP_ID, VERSION, declarations, attributor)


def test_a_status_decided_inside_a_helper_reaches_the_map():
    """The gap gate nine found.  The attribute slice takes only literal
    right-hand sides, so ``taskSpec.status = self.getFinalTaskStatus(...)`` was
    not on the map at all -- and production put 104 tasks into ``aborted``,
    which is decided only inside that helper.
    """
    _subjects, junctions, _cov = _producer_extract(
        (_POST_PROCESSOR, "pandajedi/jedipprocess/PostProcessorBase.py")
    )

    junction = next(j for j in junctions if j.subject == "JediTaskSpec.status")
    outcomes = {b.outcome for b in junction.branches if b.tier == 1}

    assert outcomes == {"broken", "aborted", "done", "finished"}
    aborted = next(b for b in junction.branches if b.outcome == "aborted")
    # The helper's own guards, marked with where they came from, and the recheck
    # that can replace whatever the chain decided.
    assert aborted.path_condition == [
        "not (taskSpec.status == 'tobroken')  [in getFinalTaskStatus()]",
        "taskSpec.status == 'toabort'  [in getFinalTaskStatus()]",
        "not (taskSpec.is_hpo_workflow())  [in getFinalTaskStatus()]",
    ]


def test_an_unreadable_return_keeps_the_list_open():
    """``return True`` on the goal-checking path is a value this slice cannot
    follow, so the junction says so rather than letting the outcomes read as a
    closed set."""
    _subjects, junctions, _cov = _producer_extract(
        (_POST_PROCESSOR, "pandajedi/jedipprocess/PostProcessorBase.py")
    )

    junction = next(j for j in junctions if j.subject == "JediTaskSpec.status")

    assert [b.outcome for b in junction.branches if b.tier == 2] == [
        "runtime(getFinalTaskStatus())"
    ]


def test_the_callers_conditions_come_first_and_are_not_marked():
    """A reader who cannot tell the caller's guard from the helper's cannot tell
    a caller that never ran from a helper that decided otherwise."""
    caller = (
        "from .PostProcessorBase import PostProcessorBase\n"
        "class AtlasProdPostProcessor(PostProcessorBase):\n"
        "    def doPostProcess(self, taskSpec):\n"
        "        if taskSpec.gshare != 'Test':\n"
        "            taskSpec.status = self.getFinalTaskStatus(taskSpec)\n"
    )
    _subjects, junctions, _cov = _producer_extract(
        (_POST_PROCESSOR, "pandajedi/jedipprocess/PostProcessorBase.py"),
        (caller, "pandajedi/jedipprocess/AtlasProdPostProcessor.py"),
    )

    junction = next(j for j in junctions if "AtlasProdPostProcessor" in j.owner)
    aborted = next(b for b in junction.branches if b.outcome == "aborted")

    assert aborted.path_condition[0] == "taskSpec.gshare != 'Test'"
    assert all("[in getFinalTaskStatus()]" in c for c in aborted.path_condition[1:])


def test_a_helper_that_computes_its_answer_is_not_a_producer():
    """``makeBuildJobParameters`` and ``getLargestAttemptNr`` settle nothing from
    a closed set, so the coverage row says these helpers compute rather than
    decide -- 2 of 7 sites in the corpus, and that is the honest number.

    The writer is still recorded, at tier 2.  This slice owns the whole
    ``self.helper()`` shape, so the attribute slice stays out of it entirely;
    dropping the sites this one cannot follow would put them on no slice at all.
    """
    source = (
        "class JobGenerator(object):\n"
        "    def generate(self, taskSpec):\n"
        "        taskSpec.status = self.computeStatus()\n"
        "    def computeStatus(self):\n"
        "        return compute(self.state)\n"
    )

    _subjects, junctions, coverage = _producer_extract(
        (source, "pandajedi/jediorder/JobGenerator.py")
    )

    assert [(b.outcome, b.tier) for j in junctions for b in j.branches] == [
        ("runtime(computeStatus())", 2)
    ]
    assert [(c.candidates, c.explained) for c in coverage] == [(1, 0)]


def test_only_a_bare_self_receiver_is_followed():
    """Widened to any call the shape matches forty-two sites of which two touch
    a subject; the rest are plugin lookups and config reads."""
    source = (
        "class Refiner(object):\n"
        "    def refine(self, taskSpec):\n"
        "        taskSpec.status = self.helper.getFinalTaskStatus(taskSpec)\n"
    )

    _subjects, junctions, coverage = _producer_extract(
        (_POST_PROCESSOR, "pandajedi/jedipprocess/PostProcessorBase.py"),
        (source, "pandajedi/jedirefine/TaskRefinerBase.py"),
    )

    assert not any("TaskRefinerBase" in j.owner for j in junctions)
    assert not any(c.file.endswith("TaskRefinerBase.py") for c in coverage)


def test_a_helper_returning_another_methods_call_is_followed_into_it():
    """``return self.decide(taskSpec)`` hands back what ``decide`` returns.
    Read one hop deep, the helper was a call it could not follow and the write
    had no outcome at all.  Each guard is marked with the method it is in."""
    source = (
        "class JobGenerator(object):\n"
        "    def generate(self, taskSpec):\n"
        "        taskSpec.status = self.finalStatus(taskSpec)\n"
        "    def finalStatus(self, taskSpec):\n"
        "        if taskSpec.ready:\n"
        "            return self.decide(taskSpec)\n"
        "        return 'pending'\n"
        "    def decide(self, taskSpec):\n"
        "        if taskSpec.broken:\n"
        "            return 'broken'\n"
        "        return 'running'\n"
    )

    _subjects, junctions, coverage = _producer_extract((source, "pandajedi/jediorder/JobGenerator.py"))

    branches = {b.outcome: b.path_condition for j in junctions for b in j.branches}
    assert set(branches) == {"broken", "running", "pending"}
    assert branches["broken"] == [
        "taskSpec.ready  [in finalStatus()]",
        "taskSpec.broken  [in decide()]",
    ]
    assert [(c.candidates, c.explained) for c in coverage] == [(1, 1)]


def test_a_helper_another_class_declares_does_not_answer_for_this_self():
    """Found by name, a method declared once in the corpus answered for every
    ``self.<name>()``.  ``Refiner`` neither declares nor inherits
    ``getFinalTaskStatus``, so whatever its ``self`` runs, it is not that."""
    source = (
        "class Refiner(object):\n"
        "    def refine(self, taskSpec):\n"
        "        taskSpec.status = self.getFinalTaskStatus(taskSpec)\n"
    )

    _subjects, junctions, coverage = _producer_extract(
        (_POST_PROCESSOR, "pandajedi/jedipprocess/PostProcessorBase.py"),
        (source, "pandajedi/jedirefine/TaskRefinerBase.py"),
    )

    refined = [j for j in junctions if "TaskRefinerBase" in j.owner]
    assert [(b.outcome, b.tier) for j in refined for b in j.branches] == [
        ("runtime(getFinalTaskStatus())", 2)
    ]


# --------------------------------------------------------------------------- #
# shared-table boundaries
# --------------------------------------------------------------------------- #

_SCHEMAS = """
if "schemaPANDA" not in tmpSelf.__dict__:
    tmpSelf.__dict__["schemaPANDA"] = "ATLAS_PANDA"
if "schemaDEFT" not in tmpSelf.__dict__:
    tmpSelf.__dict__["schemaDEFT"] = "ATLAS_DEFT"
"""


def _channels(source: str, rel: str = "pandaserver/taskbuffer/db_proxy_mods/task_module.py"):
    modules = [
        _module(_SCHEMAS, "pandaserver/config/panda_config.py"),
        _module(source, rel),
    ]
    return boundary.extract_shared_tables(modules, MAP_ID, VERSION)


def test_the_schema_qualifier_says_whose_table_it_is():
    """``panda_config`` states which Oracle schema each name means.

    Read rather than assumed, so the boundary's interface is the schema an
    operator would type into a query.
    """
    modules = [_module(_SCHEMAS, "pandaserver/config/panda_config.py")]

    assert boundary.schema_names(modules) == {
        "PANDA": "ATLAS_PANDA",
        "DEFT": "ATLAS_DEFT",
    }


def test_a_foreign_schema_table_is_a_boundary_in_both_directions():
    """A shared table is a channel, and which way is broken is the first question.

    An endpoint is inbound only; ``PRODSYS_COMM`` carries commands out to DEFT
    and back, so recording one direction would describe half the channel.
    """
    source = (
        "class TaskModule:\n"
        "    def exec_command(self, jediTaskID):\n"
        "        sqlR = f'SELECT comm_task,comm_cmd FROM {panda_config.schemaDEFT}.PRODSYS_COMM '\n"
        "        sqlR += 'WHERE comm_owner=:comm_owner '\n"
        "        varMap = {}\n"
        "        self.cur.execute(sqlR + comment, varMap)\n"
        "    def send_command(self, jediTaskID):\n"
        "        sqlD = f'DELETE FROM {panda_config.schemaDEFT}.PRODSYS_COMM WHERE COMM_TASK=:t '\n"
        "        self.cur.execute(sqlD + comment, varMap)\n"
        "        sqlI = f'INSERT INTO {panda_config.schemaDEFT}.PRODSYS_COMM "
        "(COMM_TASK,COMM_CMD) VALUES (:t,:c) '\n"
        "        self.cur.execute(sqlI + comment, varMap)\n"
    )
    channels, _coverage = _channels(source)

    assert len(channels) == 1
    channel = channels[0]
    assert (channel.system, channel.interface) == ("deft", "ATLAS_DEFT.PRODSYS_COMM")
    assert channel.transport == "shared_table"
    assert channel.carried_values == ["comm_cmd", "comm_task"]
    assert channel.handed_over == ["COMM_CMD", "COMM_TASK"]
    # A DELETE next to an INSERT on a command table is the finding: a second
    # command silently replaces one that was never picked up.
    assert channel.operations == ["DELETE", "INSERT", "SELECT"]


def test_one_column_is_not_two_because_the_case_differs():
    """SQL identifiers are case-insensitive and PanDA writes both spellings."""
    source = (
        "class TaskModule:\n"
        "    def a(self):\n"
        "        s1 = f'UPDATE {panda_config.schemaDEFT}.T_TASK SET status=:s '\n"
        "        self.cur.execute(s1 + comment, varMap)\n"
        "    def b(self):\n"
        "        s2 = f'UPDATE {panda_config.schemaDEFT}.T_TASK SET STATUS=:s '\n"
        "        self.cur.execute(s2 + comment, varMap)\n"
    )
    channels, _coverage = _channels(source)

    assert channels[0].handed_over == ["status"]


def test_a_statement_spanning_two_schemas_is_left_alone():
    """A join no longer says which side of the boundary a column sits on."""
    source = (
        "class TaskModule:\n"
        "    def joined(self):\n"
        "        s = f'SELECT t.status FROM {panda_config.schemaDEFT}.T_TASK t, "
        "{panda_config.schemaPANDA}.JEDI_Tasks j '\n"
        "        self.cur.execute(s + comment, varMap)\n"
    )
    channels, coverage = _channels(source)

    assert channels == []
    assert coverage == []


def test_a_shared_table_is_not_reported_as_unlogged():
    """Asking whether a table was also logged inverts the question.

    Its values are queryable afterwards precisely because nobody had to write
    them down a second time.
    """
    fragment = MapFragment(map_id=MAP_ID, derived_from=VERSION)
    fragment.boundaries.append(
        BoundaryNode(
            map_id=MAP_ID,
            derived_from=VERSION,
            name="b:deft",
            system="deft",
            transport="shared_table",
            interface="ATLAS_DEFT.PRODSYS_COMM",
            carried_values=["comm_task", "comm_cmd"],
        )
    )

    assert gates.unobservable_boundaries(fragment) == []


# --------------------------------------------------------------------------- #
# what bounds a query's reach
# --------------------------------------------------------------------------- #
#
# A row can be missed for a reason no predicate mentions.  One was: a task sat
# in ``finishing`` while the query that rescues orphaned commands ran every
# cycle, because the auxiliary table it joins had stopped being updated and its
# watermark had risen above the task's id.

_ORPHAN_RESCUE = """
class TaskModule:
    def getTasksToExecCommand_JEDI(self, vo):
        sqlOrpS = "SELECT jediTaskID,errorDialog "
        sqlOrpS += "FROM {0}.JEDI_Tasks tabT,{0}.JEDI_AUX_Status_MinTaskID tabA "
        sqlOrpS += "WHERE tabT.status=tabA.status AND tabT.jediTaskID>=tabA.min_jediTaskID "
        sqlOrpS += "AND tabT.status=:status "
        self.cur.execute(sqlOrpS + comment, varMap)
        sqlOrpU = "UPDATE {0}.JEDI_Tasks SET oldStatus=:oldStatus "
        sqlOrpU += "WHERE jediTaskID=:jediTaskID "
        self.cur.execute(sqlOrpU + comment, varMap)
"""


def test_the_second_table_in_a_from_list_is_read():
    """Stopping at the first hid a table joined by thirty-one functions.

    ``reads`` answers where the row came from and rightly names the leading
    table; what bounds which rows can be seen is a different question, and the
    join partner is where the answer is written.
    """
    statement = (
        "SELECT jediTaskID FROM {0}.JEDI_Tasks tabT,{0}.JEDI_AUX_Status_MinTaskID tabA "
        "WHERE tabT.status=tabA.status AND tabT.jediTaskID>=tabA.min_jediTaskID"
    )

    assert [table for table, _c in sqltext.reads(statement)] == ["JEDI_Tasks"]
    assert sqltext.joins(statement) == ["JEDI_AUX_Status_MinTaskID"]
    assert sqltext.joined_columns(statement, "JEDI_AUX_Status_MinTaskID") == [
        "min_jediTaskID",
        "status",
    ]


def test_a_name_the_statement_defines_for_itself_is_not_a_join_partner():
    """``checkDuplication_JEDI`` joins its own ``WITH`` block to itself, which
    is a step in the query rather than a dependency on anything outside it."""
    statement = (
        "WITH tmpTab AS (SELECT PandaID FROM {}.filesTable4) "
        "SELECT a.PandaID FROM tmpTab a,tmpTab b WHERE a.PandaID=b.PandaID"
    )

    assert sqltext.joins(statement) == []


_READ_ONLY_SOURCE = """
class TaskModule:
    def pick(self):
        sqlR = "SELECT retry_action FROM ATLAS_PANDA.RETRYACTIONS WHERE errorCode=:errorCode"
        self.cur.execute(sqlR + comment, varMap)

    def touch(self):
        sqlU = "UPDATE ATLAS_PANDA.JEDI_Tasks SET status=:status WHERE jediTaskID=:jediTaskID"
        self.cur.execute(sqlU + comment, varMap)

    def count(self):
        sqlC = "SELECT nJobs FROM ATLAS_PANDA.JEDI_Tasks WHERE jediTaskID=:jediTaskID"
        self.cur.execute(sqlC + comment, varMap)
"""


def test_a_table_read_without_being_joined_is_still_a_boundary():
    """A table need not bound anything to be one.

    This walked join partners only, because the name it went by was
    ``selection_gates`` and a gate is what a partner is.  Whatever keeps the
    table current is outside the source either way, so a value read straight
    out of one is as unaccountable as a row hidden by one -- and twenty of the
    corpus's twenty-two read-only tables were reached no other way, among them
    ``RETRYACTIONS``, which decides whether a job is retried at all.
    """
    modules = [_module(_READ_ONLY_SOURCE, "pandaserver/taskbuffer/db_proxy_mods/task_module.py")]
    never_written = boundary.tables_never_written(modules)

    found = boundary.extract_read_only_tables(modules, MAP_ID, VERSION, never_written, set())

    # ``JEDI_Tasks`` is read here too, and written here, so it is a step in
    # this map rather than a dependency on anything outside it.
    assert [b.interface for b in found] == ["RETRYACTIONS"]
    assert found[0].carried_values == ["retry_action"]
    assert found[0].transport == "shared_table"
    assert found[0].handed_over == []


def test_a_table_reached_both_ways_is_one_boundary():
    """The two readings answer for the same table, so they fold into one node.

    Storing a second would be the same fact under two names, which is what
    folding the table's spellings exists to prevent.
    """
    source = (
        "class TaskModule:\n"
        "    def joined(self):\n"
        "        sqlJ = ('SELECT t.jediTaskID FROM ATLAS_PANDA.JEDI_Tasks t,"
        "ATLAS_PANDA.RETRYACTIONS r WHERE t.errorCode=r.errorCode')\n"
        "        self.cur.execute(sqlJ + comment, varMap)\n"
        "    def read(self):\n"
        "        sqlR = 'SELECT retry_action FROM ATLAS_PANDA.RETRYACTIONS WHERE errorCode=:e'\n"
        "        self.cur.execute(sqlR + comment, varMap)\n"
        "    def touch(self):\n"
        "        sqlU = 'UPDATE ATLAS_PANDA.JEDI_Tasks SET status=:s WHERE jediTaskID=:j'\n"
        "        self.cur.execute(sqlU + comment, varMap)\n"
    )
    modules = [_module(source, "pandaserver/taskbuffer/db_proxy_mods/task_module.py")]

    found = boundary.extract_read_only_tables(
        modules, MAP_ID, VERSION, boundary.tables_never_written(modules), set()
    )

    assert [b.interface for b in found] == ["RETRYACTIONS"]
    # Both readings' columns reach the one node: the join predicate's, found
    # through the alias, and the projection's.
    assert found[0].carried_values == ["errorCode", "retry_action"]


def test_a_table_does_not_bound_itself():
    """The gates of a statement were offered to every table in it.

    A subject sitting on the gating table therefore got its own table back as
    the thing limiting which of its rows can be seen, and the verdict that
    produces tells a reader to go and find out who maintains a table they are
    already looking at.  Seventeen of the fifty-eight gated subjects were that.
    """
    modules = [_module(_ORPHAN_RESCUE, "pandaserver/taskbuffer/db_proxy_mods/task_module.py")]
    attributor = attribution.SpecAttributor(
        progress.spec_attributes(modules), attribution.class_bases(modules)
    )
    attributor.learn_table_classes(modules)

    gates = sqlwrite.selection_gates(
        modules, attributor, boundary.tables_never_written(modules)
    )

    assert not [
        subject
        for subject, tables in gates.items()
        if subject.rpartition(".")[0].lower() in {t.lower() for t in tables}
    ]


def test_only_a_table_nothing_here_writes_is_a_gate():
    """A join to a table the corpus maintains is a step in a query; a join to
    one it only ever reads is a dependency that can go stale unaccounted for."""
    modules = [_module(_ORPHAN_RESCUE, "pandaserver/taskbuffer/db_proxy_mods/task_module.py")]

    never_written = boundary.tables_never_written(modules)

    # Keyed by the folded name: SQL identifiers are case-insensitive and this
    # corpus spells one table several ways.
    assert never_written["jedi_aux_status_mintaskid"] == "JEDI_AUX_Status_MinTaskID"
    assert "jedi_tasks" not in never_written

    found = boundary.extract_read_only_tables(modules, MAP_ID, VERSION, never_written, set())

    assert [b.interface for b in found] == ["JEDI_AUX_Status_MinTaskID"]


_MIXED_CASE_SPELLING = """
class TaskModule:
    def readOne(self, vo):
        sqlR = "SELECT status FROM {0}.JEDI_Tasks tabT,{0}.datasets tabD "
        sqlR += "WHERE tabT.jediTaskID=tabD.jediTaskID AND tabT.status=:status "
        self.cur.execute(sqlR + comment, varMap)

    def readAgain(self, vo):
        sqlA = "SELECT status FROM {0}.JEDI_Tasks tabT,{0}.filesTable_arch tabF "
        sqlA += "WHERE tabT.jediTaskID=tabF.jediTaskID AND tabT.status=:status "
        self.cur.execute(sqlA + comment, varMap)

    def readThird(self, vo):
        sqlT = "SELECT status FROM {0}.JEDI_Tasks tabT,{0}.filesTable_ARCH tabF "
        sqlT += "WHERE tabT.jediTaskID=tabF.jediTaskID "
        self.cur.execute(sqlT + comment, varMap)

    def writeIt(self, vo):
        sqlW = "UPDATE {0}.Datasets SET status=:status "
        sqlW += "WHERE vuid=:vuid "
        self.cur.execute(sqlW + comment, varMap)

    def writeTask(self, vo):
        sqlK = "UPDATE {0}.JEDI_Tasks SET oldStatus=:oldStatus "
        sqlK += "WHERE jediTaskID=:jediTaskID "
        self.cur.execute(sqlK + comment, varMap)
"""


def test_a_table_written_under_another_spelling_is_not_a_gate():
    """SQL identifiers are case-insensitive and this corpus spells one table
    several ways, so comparing the spellings claims PanDA does not maintain a
    table it writes on the next line.

    Measured on the installed corpus before this was folded: ten of the
    thirty-nine were tables PanDA writes -- ``Datasets`` read as ``datasets``,
    ``jobsArchived4`` as ``jobsarchived4`` -- and each became a boundary saying
    something outside keeps it current.
    """
    modules = [_module(_MIXED_CASE_SPELLING, "pandaserver/taskbuffer/db_proxy_mods/task_module.py")]

    never_written = boundary.tables_never_written(modules)

    # Read as ``datasets``, written as ``Datasets``: one table, and PanDA owns it.
    assert "datasets" not in never_written
    # The control: written under the one spelling it is read under.
    assert "jedi_tasks" not in never_written
    # And the table nothing here writes is still found.
    assert "filestable_arch" in never_written


def test_one_table_spelled_three_ways_is_one_boundary():
    """``filesTable_arch`` and ``filesTable_ARCH`` are one dependency."""
    modules = [_module(_MIXED_CASE_SPELLING, "pandaserver/taskbuffer/db_proxy_mods/task_module.py")]

    never_written = boundary.tables_never_written(modules)

    assert "filestable_arch" in never_written

    found = boundary.extract_read_only_tables(modules, MAP_ID, VERSION, never_written, set())

    assert [b.interface for b in found] == ["filesTable_ARCH"]


def test_the_spelling_shown_is_the_one_the_corpus_uses_most():
    """The name in a report has to be what a reader will find when they grep.

    Folding decides *identity*; it does not get to decide how the table is
    spelled back to a person.  Ties go to the first in sorted order so the
    choice is reproducible rather than dependent on walk order.
    """
    source = _MIXED_CASE_SPELLING + """
class OtherModule:
    def readFourth(self, vo):
        sqlF = "SELECT status FROM {0}.JEDI_Tasks tabT,{0}.filesTable_ARCH tabX "
        sqlF += "WHERE tabT.jediTaskID=tabX.jediTaskID AND tabX.type=:type "
        self.cur.execute(sqlF + comment, varMap)
"""
    modules = [_module(source, "pandaserver/taskbuffer/db_proxy_mods/task_module.py")]

    assert boundary.tables_never_written(modules)["filestable_arch"] == "filesTable_ARCH"


def test_a_subject_does_not_carry_a_gate_on_a_table_panda_maintains():
    """The ablation for the fold, at the level the claim is actually shipped.

    ``selection_gates`` is what a report shows as "what limits which rows this
    query can see", and a gate on a table PanDA writes is a false statement
    about the system's boundary -- the staleness it warns about cannot happen.
    Measured on the installed corpus: twenty subjects carried one, among them
    ``JobSpec.jobStatus``, ``DatasetSpec.type`` and ``FileSpec.PandaID``.

    Removing the fold from :func:`boundary.tables_never_written` puts the gate
    straight back, which is what makes this a test of the mechanism rather than
    of the fixture.
    """
    modules = [_module(_MIXED_CASE_SPELLING, "pandaserver/taskbuffer/db_proxy_mods/task_module.py")]
    attributor = attribution.SpecAttributor(
        progress.spec_attributes(modules), attribution.class_bases(modules)
    )
    attributor.learn_table_classes(modules)

    gated = sqlwrite.selection_gates(
        modules, attributor, boundary.tables_never_written(modules)
    )

    # ``datasets`` is read here and written as ``Datasets`` two methods down.
    assert not any("datasets" in {g.lower() for g in gates} for gates in gated.values())
    # The table nothing here writes is still a gate, so the fold did not simply
    # empty the answer.
    assert any("filestable_arch" in {g.lower() for g in gates} for gates in gated.values())


def test_a_table_that_bounds_a_query_and_nothing_writes_is_a_boundary():
    """A complete answer -- the value comes from outside -- and a work item.

    Found from the read side rather than the schema qualifier, which is why it
    reaches a table sitting in JEDI's own schema.
    """
    modules = [_module(_ORPHAN_RESCUE, "pandaserver/taskbuffer/db_proxy_mods/task_module.py")]

    found = boundary.extract_read_only_tables(
        modules, MAP_ID, VERSION, boundary.tables_never_written(modules), set()
    )

    assert [b.interface for b in found] == ["JEDI_AUX_Status_MinTaskID"]
    assert found[0].system == boundary.UNRESOLVED_SYSTEM
    assert found[0].transport == "shared_table"
    # Read-only is the finding, not a gap in the reading.
    assert found[0].operations == ["SELECT"]
    assert found[0].handed_over == []
    assert found[0].carried_values == ["min_jediTaskID", "status"]


def test_a_table_the_schema_recognizer_already_named_is_not_repeated():
    """One crossing, found two ways, must not become two boundaries."""
    modules = [_module(_ORPHAN_RESCUE, "pandaserver/taskbuffer/db_proxy_mods/task_module.py")]

    found = boundary.extract_read_only_tables(
        modules,
        MAP_ID,
        VERSION,
        boundary.tables_never_written(modules),
        {"JEDI_AUX_Status_MinTaskID"},
    )

    assert found == []


# --------------------------------------------------------------------------- #
# entry points and triggers
# --------------------------------------------------------------------------- #

_KNIGHT = """
class TaskCommando:
    def start(self):
        while True:
            time.sleep(jedi_config.taskcommando.loopCycle)
            self.doAction()

    def doAction(self):
        tasks = self.taskBufferIF.getTasksToExecCommand_JEDI(vo, label, pid=self.pid)
"""

_PROXY = """
class TaskModule:
    def getTasksToExecCommand_JEDI(self, vo, prodSourceLabel, pid=None):
        sqlC = f'SELECT comm_task FROM {panda_config.schemaDEFT}.PRODSYS_COMM '
        self.cur.execute(sqlC + comment, varMap)
        self.markTask()

    def markTask(self):
        pass
"""


def _junction(owner: str, subject: str = "JediTaskSpec.status") -> JunctionNode:
    return JunctionNode(
        map_id=MAP_ID,
        derived_from=VERSION,
        name=f"j:{owner}",
        subject=subject,
        owner=owner,
    )


def _attach(*sources: tuple[str, str], junctions, tables=None):
    modules = [_module(text, rel) for text, rel in sources]
    return trigger.attach(junctions, modules, tables or {"PRODSYS_COMM"})


def test_a_sleeping_forever_loop_is_a_polled_entry():
    """A ``while True`` that sleeps starts something; a conditional loop retries.

    Accepting any sleeping loop matched 33 modules including ``Interaction``
    and ``ddm``, neither of which starts anything.
    """
    modules = [
        _module(_KNIGHT, "pandajedi/jediorder/TaskCommando.py"),
        _module(
            "def fetch(self):\n"
            "    while not done:\n"
            "        time.sleep(1)\n",
            "pandaserver/dataservice/ddm.py",
        ),
    ]
    kinds = trigger.classify(modules, set())

    assert kinds["pandajedi/jediorder/TaskCommando.py"] == {"polled"}
    assert "pandaserver/dataservice/ddm.py" not in kinds


def test_polling_for_a_foreign_row_is_also_a_command_entry():
    """It runs again next cycle; the command it failed to act on does not."""
    kinds = trigger.classify(
        [
            _module(_KNIGHT, "pandajedi/jediorder/TaskCommando.py"),
            _module(_PROXY, "pandaserver/taskbuffer/db_proxy_mods/task_module.py"),
        ],
        {"PRODSYS_COMM"},
    )

    assert kinds["pandajedi/jediorder/TaskCommando.py"] == {"polled", "command"}
    # The module that merely *contains* the read starts nothing.
    assert "pandaserver/taskbuffer/db_proxy_mods/task_module.py" not in kinds


def test_reach_follows_self_calls_past_the_door():
    """The door and the write are rarely the same method.

    ``add_main`` starts ``AdderGen.run``; the ``jobStatus`` writes are several
    ``self`` calls further in, and stopping at the door reported them as though
    nothing ran them.
    """
    junction = _junction("pandaserver/taskbuffer/db_proxy_mods/task_module.py::markTask")
    _attach(
        (_KNIGHT, "pandajedi/jediorder/TaskCommando.py"),
        (_PROXY, "pandaserver/taskbuffer/db_proxy_mods/task_module.py"),
        junctions=[junction],
    )

    assert {(e.trigger, e.via) for e in junction.entry_points} == {
        ("polled", "getTasksToExecCommand_JEDI"),
        ("command", "getTasksToExecCommand_JEDI"),
    }


def test_a_name_several_modules_implement_carries_no_edge():
    """``run`` is defined by every daemon, so following it invents callers.

    Before this, one ``datasetManager`` junction collected fourteen entry
    points, thirteen of them from daemons that have never heard of it.
    """
    daemon = "class D:\n    def run(self):\n        self.act()\n    def act(self):\n        pass\n"
    other = "class O:\n    def run(self):\n        pass\n"
    junction = _junction("pandaserver/daemons/scripts/first.py::act")
    _attach(
        (daemon, "pandaserver/daemons/scripts/first.py"),
        (other, "pandaserver/daemons/scripts/second.py"),
        junctions=[junction],
    )

    # Its own module is polled, so it is reached -- but not *by* the other
    # daemon, which shares only the method name.
    assert {e.entry for e in junction.entry_points} == {
        "pandaserver/daemons/scripts/first.py"
    }


def test_an_import_evidences_an_edge_a_shared_name_cannot():
    """``from ...adder_gen import AdderGen`` says which ``run`` is meant."""
    worker = (
        "class AdderGen:\n"
        "    def run(self):\n"
        "        self.finalize()\n"
        "    def finalize(self):\n"
        "        pass\n"
    )
    daemon = (
        "from pandaserver.dataservice.adder_gen import AdderGen\n"
        "def main():\n"
        "    AdderGen().run()\n"
    )
    other = "class O:\n    def run(self):\n        pass\n"
    junction = _junction("pandaserver/dataservice/adder_gen.py::finalize")
    _attach(
        (worker, "pandaserver/dataservice/adder_gen.py"),
        (daemon, "pandaserver/daemons/scripts/add_main.py"),
        (other, "pandaserver/daemons/scripts/other.py"),
        junctions=[junction],
    )

    assert {e.entry for e in junction.entry_points} == {
        "pandaserver/daemons/scripts/add_main.py"
    }


def test_a_facade_that_forwards_is_not_a_second_implementation():
    """``TaskBuffer`` and ``JediTaskBuffer`` are doors, not rival definitions."""
    facade = (
        "class TaskBuffer:\n"
        "    def markTask(self):\n"
        "        ret = proxy.markTask()\n"
        "        return ret\n"
    )
    modules = [
        _module(_PROXY, "pandaserver/taskbuffer/db_proxy_mods/task_module.py"),
        _module(facade, "pandaserver/taskbuffer/TaskBuffer.py"),
    ]

    assert (
        trigger.sole_definitions(modules)["markTask"]
        == "pandaserver/taskbuffer/db_proxy_mods/task_module.py"
    )


_KNIGHT_CONSULTING = """
class TaskCommando:
    def runImpl(self):
        while True:
            tasks = self.taskBufferIF.getTasksToExecCommand_JEDI(vo, label)
            ids = self.taskBufferIF.getPandaIDsWithTask_JEDI(jediTaskID, True)
            self.report(ids)

    def report(self, ids):
        pass
"""

_JOB_PROXY = """
class JobModule:
    def getPandaIDsWithTask_JEDI(self, jediTaskID, onlyActive):
        self.cur.execute(sqlP + comment, varMap)
"""


def test_a_call_into_another_module_is_recorded_as_a_qualified_target():
    """What a junction consults is rarely in its own file.

    ``TaskCommando.runImpl`` decides a task's status and asks
    ``getPandaIDsWithTask_JEDI`` for that task's jobs, and the two sit in
    different packages with a ``self.taskBufferIF`` call between them.  Keeping
    the edge inside one module left every such question looking as though the
    map had nothing to say.
    """
    junction = _junction("pandajedi/jediorder/TaskCommando.py::runImpl")
    modules = [
        _module(_KNIGHT_CONSULTING, "pandajedi/jediorder/TaskCommando.py"),
        _module(_JOB_PROXY, "pandaserver/taskbuffer/db_proxy_mods/job_module.py"),
        _module(_PROXY, "pandaserver/taskbuffer/db_proxy_mods/task_module.py"),
    ]
    trigger.attach_calls([junction], modules)

    assert junction.calls == [
        "pandajedi/jediorder/TaskCommando.py::report",
        "pandaserver/taskbuffer/db_proxy_mods/job_module.py::getPandaIDsWithTask_JEDI",
        "pandaserver/taskbuffer/db_proxy_mods/task_module.py::getTasksToExecCommand_JEDI",
    ]


def test_the_call_belongs_to_the_function_that_makes_it():
    """Keying the outward calls by name alone pooled a file's whole surface.

    ``_outward_calls`` answers "does this module call X", which is the right
    question for the door but the wrong one here: two methods of one knight
    consult different things, and attributing both to both is the shared-owner
    join by another name.
    """
    source = (
        "class K:\n"
        "    def finish(self):\n"
        "        self.taskBufferIF.getPandaIDsWithTask_JEDI(i, True)\n"
        "    def refine(self):\n"
        "        self.taskBufferIF.markTask()\n"
    )
    modules = [
        _module(source, "pandajedi/jediorder/K.py"),
        _module(_JOB_PROXY, "pandaserver/taskbuffer/db_proxy_mods/job_module.py"),
        _module(_PROXY, "pandaserver/taskbuffer/db_proxy_mods/task_module.py"),
    ]
    finish = _junction("pandajedi/jediorder/K.py::finish")
    refine = _junction("pandajedi/jediorder/K.py::refine")
    trigger.attach_calls([finish, refine], modules)

    assert finish.calls == [
        "pandaserver/taskbuffer/db_proxy_mods/job_module.py::getPandaIDsWithTask_JEDI"
    ]
    assert refine.calls == [
        "pandaserver/taskbuffer/db_proxy_mods/task_module.py::markTask"
    ]


def test_a_shared_name_the_caller_does_not_import_carries_no_call():
    """The same restriction the door hop turns on, one direction over.

    ``run`` is defined by every daemon; following it by name gave one junction
    fourteen entry points of which thirteen were wrong.  A call resolves only
    where the name means one thing, or where the caller says which module it
    means.
    """
    knight = "class K:\n    def act(self):\n        self.helper.run()\n"
    modules = [
        _module(knight, "pandajedi/jediorder/K.py"),
        _module("class A:\n    def run(self):\n        pass\n", "pandaserver/a.py"),
        _module("class B:\n    def run(self):\n        pass\n", "pandaserver/b.py"),
    ]
    junction = _junction("pandajedi/jediorder/K.py::act")
    trigger.attach_calls([junction], modules)

    assert junction.calls == []


def test_a_method_on_data_passing_through_is_not_a_consultation():
    """``newScanSiteList.append(x)`` is a list, whatever else defines ``append``.

    Resolving by name alone put ``SQLManager.append`` on 241 junctions, every
    one of them a list.  What a junction consults is what its object holds, and
    the receiver is what says so.
    """
    broker = (
        "class B:\n"
        "    def choose(self):\n"
        "        newScanSiteList = []\n"
        "        newScanSiteList.append(site)\n"
    )
    modules = [
        _module(broker, "pandajedi/jedibrokerage/B.py"),
        _module(
            "class SQLManager:\n    def append(self, sql):\n        pass\n",
            "pandaserver/taskbuffer/SQLManager.py",
        ),
    ]
    junction = _junction("pandajedi/jedibrokerage/B.py::choose")
    trigger.attach_calls([junction], modules)

    assert junction.calls == []


def test_the_accessor_a_mixin_reaches_its_siblings_through_is_a_consultation():
    """``get_task_event_module(self).updateInputStatusJedi(...)``.

    The proxy is assembled from mixins and reaches a sibling through an
    accessor taking ``self``, which is the same claim as
    ``self.taskBufferIF.<method>()`` spelled the way the composite forces.
    Refusing it costs 33 junctions their only route to another entity's rows.
    """
    caller = (
        "from pandaserver.taskbuffer.db_proxy_mods.event_module import get_event_module\n"
        "class JobModule:\n"
        "    def insertNewJob(self):\n"
        "        get_event_module(self).updateInputStatusJedi(i, p, s)\n"
    )
    sibling = (
        "class EventModule:\n"
        "    def updateInputStatusJedi(self, jediTaskID, pandaID, status):\n"
        "        pass\n"
    )
    modules = [
        _module(caller, "pandaserver/taskbuffer/db_proxy_mods/job_module.py"),
        _module(sibling, "pandaserver/taskbuffer/db_proxy_mods/event_module.py"),
    ]
    junction = _junction("pandaserver/taskbuffer/db_proxy_mods/job_module.py::insertNewJob")
    trigger.attach_calls([junction], modules)

    assert junction.calls == [
        "pandaserver/taskbuffer/db_proxy_mods/event_module.py::updateInputStatusJedi"
    ]


def test_the_facade_hop_is_not_a_call_here_either():
    """A method reached through a borrowed proxy is the same handoff
    :func:`_outward_calls` already refuses, and naming ``TaskBuffer`` as what a
    junction consults would point every descent at a door."""
    facade = (
        "class JediTaskBuffer:\n"
        "    def door(self):\n"
        "        with self.proxyPool.get() as proxy:\n"
        "            return proxy.markTask()\n"
    )
    modules = [
        _module(facade, "pandaserver/taskbuffer/JediTaskBuffer.py"),
        _module(_PROXY, "pandaserver/taskbuffer/db_proxy_mods/task_module.py"),
    ]
    junction = _junction("pandaserver/taskbuffer/JediTaskBuffer.py::door")
    trigger.attach_calls([junction], modules)

    assert junction.calls == []


def test_entries_that_hand_over_different_arguments_are_reported():
    """An argument one entry omits is a guard that cannot fire on that path."""
    junction = _junction("x.py::f")
    junction.entry_points = [
        EntryPoint(trigger="polled", entry="JobGenerator.py", via="get", arg_binding={"minPriority": "p"}),
        EntryPoint(trigger="message", entry="msg.py", via="get", arg_binding={"target_tasks": "t"}),
    ]

    assert trigger.differing_arguments([junction]) == [
        (
            "JediTaskSpec.status",
            "x.py::f",
            {"JobGenerator.py": ["minPriority"], "msg.py": ["target_tasks"]},
        )
    ]


_DOOR_FACADE = '''
class TaskBuffer:
    def storeJobs(self, jobs, user, fqans=None):
        with self.proxyPool.get() as proxy:
            return proxy.insertNewJob(jobs, user)
'''

_DOOR_PROXY = '''
class JobComplexModule:
    def insertNewJob(self, job, user):
        self.recordStatusChange(job)
        return 1

    def recordStatusChange(self, job):
        return 2
'''

_DOOR_API = '''
import TaskBuffer

@request_validation(secure=True)
def submit(req, jobs):
    return global_task_buffer.storeJobs(jobs, req.user, fqans=req.fqans)
'''


def test_an_entry_reaches_past_the_facade_it_borrows_a_proxy_through():
    """Where a job row is created reported that nothing starts it.

    ``TaskBuffer.storeJobs`` is the only caller of ``proxy.insertNewJob``, and
    the facade hop is dropped so that its silent log file is not named -- which
    left the API that submits jobs invisible to the trigger question.
    """
    junction = _junction("proxy.py::insertNewJob", "JobSpec.jobStatus")
    _attach(
        (_DOOR_FACADE, "taskbuffer/TaskBuffer.py"),
        (_DOOR_PROXY, "proxy.py"),
        (_DOOR_API, "api/v1/job_api.py"),
        junctions=[junction],
    )

    assert [(e.trigger, e.entry, e.via, e.reached_by) for e in junction.entry_points] == [
        ("request", "api/v1/job_api.py", "storeJobs", "door")
    ]


def test_what_the_door_was_handed_is_not_reported_as_what_this_entry_passed():
    """``storeJobs(jobs, user, fqans=...)`` is not ``insertNewJob(job, user)``.

    The keywords at the door belong to the door.  Carrying them across would
    put a false argument list under a true edge, and comparing them against a
    direct caller's would report a guard that cannot fire.
    """
    junction = _junction("proxy.py::insertNewJob", "JobSpec.jobStatus")
    _attach(
        (_DOOR_FACADE, "taskbuffer/TaskBuffer.py"),
        (_DOOR_PROXY, "proxy.py"),
        (_DOOR_API, "api/v1/job_api.py"),
        junctions=[junction],
    )

    assert junction.entry_points[0].arg_binding == {}
    assert trigger.differing_arguments([junction]) == []


def test_a_door_reaches_what_the_method_behind_it_calls_on_itself():
    """The door and the write are rarely the same method here either."""
    junction = _junction("proxy.py::recordStatusChange", "JobSpec.jobStatus")
    _attach(
        (_DOOR_FACADE, "taskbuffer/TaskBuffer.py"),
        (_DOOR_PROXY, "proxy.py"),
        (_DOOR_API, "api/v1/job_api.py"),
        junctions=[junction],
    )

    assert [e.entry for e in junction.entry_points] == ["api/v1/job_api.py"]


def test_the_log_question_still_stops_at_the_facade():
    """The two consumers of ``reaching_modules`` want opposite things here.

    The facade rule exists because ``JediTaskBuffer`` logs twice in the whole
    file, so naming its log sent 71 junctions to a file that says nothing.
    Reach crosses the door; the log question must not follow.
    """
    modules = [
        _module(_DOOR_FACADE, "taskbuffer/TaskBuffer.py"),
        _module(_DOOR_PROXY, "proxy.py"),
        _module(_DOOR_API, "api/v1/job_api.py"),
    ]

    assert "insertNewJob" not in trigger.reaching_modules(modules).get("proxy.py", {})
    assert "insertNewJob" in trigger.reaching_modules(modules, through_doors=True)["proxy.py"]


def test_a_door_to_a_name_two_modules_define_carries_no_edge():
    """The honesty of the hop is unchanged: one implementation, or an import.

    Measured on the corpus: ``send_command_to_job`` is written in two modules,
    so the door to it is refused even though a trigger calls the door.
    """
    junction = _junction("proxy.py::insertNewJob", "JobSpec.jobStatus")
    rival = "class Other:\n    def insertNewJob(self, job, user):\n        return 3\n"
    _attach(
        (_DOOR_FACADE, "taskbuffer/TaskBuffer.py"),
        (_DOOR_PROXY, "proxy.py"),
        (rival, "other.py"),
        (_DOOR_API, "api/v1/job_api.py"),
        junctions=[junction],
    )

    assert junction.entry_points == []


_INHERITED_BASE = '''
class BaseModule:
    def recordStatusChange(self, job):
        return 2
'''

_INHERITED_SUB = '''
from base_module import BaseModule


class JobStandaloneModule(BaseModule):
    def insertNewJob(self, job, user):
        self.recordStatusChange(job)
        return 1
'''


def test_an_inherited_self_call_reaches_the_module_its_base_lives_in():
    """The junction is in the base's file and the call to it is in the subclass.

    ``JobStandaloneModule.insertNewJob`` calls ``self.recordStatusChange``,
    which ``BaseModule`` defines one file up, so three junctions in
    ``recordStatusChange`` read as though nothing ran them -- while the API that
    submits jobs, ``JobGenerator`` and ``TaskCommando`` all reach it.
    """
    junction = _junction("base_module.py::recordStatusChange", "JobSpec.jobStatus")
    _attach(
        (_INHERITED_BASE, "base_module.py"),
        (_INHERITED_SUB, "proxy.py"),
        (_DOOR_FACADE, "taskbuffer/TaskBuffer.py"),
        (_DOOR_API, "api/v1/job_api.py"),
        junctions=[junction],
    )

    assert [e.entry for e in junction.entry_points] == ["api/v1/job_api.py"]


def test_a_method_two_ancestor_modules_define_carries_no_inherited_edge():
    """Unique or nothing, so the hierarchy resolves rather than the name.

    Python's MRO would pick one of the two, but ``class_bases`` records bases
    unordered, and a confident wrong home is worse than the unresolved method
    it replaces.
    """
    junction = _junction("base_module.py::recordStatusChange", "JobSpec.jobStatus")
    rival = "class Mixin:\n    def recordStatusChange(self, job):\n        return 9\n"
    two_bases = _INHERITED_SUB.replace(
        "class JobStandaloneModule(BaseModule):",
        "class JobStandaloneModule(BaseModule, Mixin):",
    )
    _attach(
        (_INHERITED_BASE, "base_module.py"),
        (rival, "mixin.py"),
        (two_bases, "proxy.py"),
        (_DOOR_FACADE, "taskbuffer/TaskBuffer.py"),
        (_DOOR_API, "api/v1/job_api.py"),
        junctions=[junction],
    )

    assert junction.entry_points == []


def test_the_log_question_does_not_follow_an_inherited_call():
    """The immediate caller answers whose log says it ran.

    Measured on the corpus: following it there would have ``recordStatusChange``
    name ``panda-adder.log``, ``panda-api_event.log`` and
    ``panda-AtlasAnalWatchDog.log`` at once -- the P1-37 failure, in which 71
    junctions were sent to a file that says nothing, in the other direction.
    """
    modules = [
        _module(_INHERITED_BASE, "base_module.py"),
        _module(_INHERITED_SUB, "proxy.py"),
        _module(_DOOR_FACADE, "taskbuffer/TaskBuffer.py"),
        _module(_DOOR_API, "api/v1/job_api.py"),
    ]

    def entries(**flags):
        reached = trigger.reaching_modules(modules, through_doors=True, **flags)
        return [e for e, _door, _call in reached["base_module.py"]["recordStatusChange"]]

    # The edge itself is already there without the hop -- ``recordStatusChange``
    # is written in one module, so the name resolves.  What the hop carries is
    # the *entry*: the subclass's own file starts nothing, and the API does.
    assert entries() == ["proxy.py"]
    assert entries(through_inheritance=True) == ["proxy.py", "api/v1/job_api.py"]


_DISPATCH_BASE = '''
class WatchDogBase:
    def pre_action(self, log, vo, label, pid):
        return None
'''

_DISPATCH_FACTORY = '''
from dog_base import WatchDogBase


class FactoryBase:
    def getImpl(self, vo, label):
        return self.implMap.get((vo, label))


class WatchDog(FactoryBase):
    def getImpl(self, vo, label) -> WatchDogBase | None:
        return FactoryBase.getImpl(self, vo, label)

    def start(self):
        while True:
            time.sleep(jedi_config.watchdog.loopCycle)
            impl = self.getImpl("atlas", "managed")
            if impl is not None:
                impl.doAction()
'''

_DISPATCH_PLUGIN = '''
from dog_base import WatchDogBase


class AtlasProdWatchDog(WatchDogBase):
    def doAction(self):
        self.doForPriorityMassage()
        return 0

    def doForPriorityMassage(self):
        return 1
'''

# Without a rival the name resolves on its own and the test below would pass
# for the wrong reason.  The corpus has eight of these.
_DISPATCH_RIVAL = '''
from dog_base import WatchDogBase


class AtlasAnalWatchDog(WatchDogBase):
    def doAction(self):
        return 0
'''


def _dogs():
    return (
        (_DISPATCH_BASE, "pandajedi/jedidog/dog_base.py"),
        (_DISPATCH_FACTORY, "pandajedi/jediorder/WatchDog.py"),
        (_DISPATCH_PLUGIN, "pandajedi/jedidog/AtlasProdWatchDog.py"),
        (_DISPATCH_RIVAL, "pandajedi/jedidog/AtlasAnalWatchDog.py"),
    )


def test_a_dispatched_call_reaches_the_family_the_factory_declares():
    """The knight never names the plugin; the factory's return type does.

    ``WatchDog.start`` calls ``impl.doAction()`` on whatever
    ``self.getImpl`` handed back, and the plugin is loaded from a config
    string -- so the knight does not import it and eight modules define
    ``doAction``.  The junction is in what ``doAction`` calls, not in
    ``doAction`` itself, which is why the edge has to carry downstream.
    """
    junction = _junction(
        "pandajedi/jedidog/AtlasProdWatchDog.py::doForPriorityMassage",
        "JediTaskSpec.currentPriority",
    )
    _attach(*_dogs(), junctions=[junction])

    assert [(e.trigger, e.entry) for e in junction.entry_points] == [
        ("polled", "pandajedi/jediorder/WatchDog.py")
    ]


def test_a_class_outside_the_declared_family_gains_no_dispatched_edge():
    """The declared type is the whole restriction, so it has to narrow.

    ``doAction`` is a common enough name that accepting every definition of
    it would hand the WatchDog knight edges into modules it dispatches
    nothing to.
    """
    stranger = (
        "class Unrelated:\n"
        "    def doAction(self):\n"
        "        return self.doForPriorityMassage()\n"
        "\n"
        "    def doForPriorityMassage(self):\n"
        "        return 1\n"
    )
    junction = _junction(
        "pandaserver/dataservice/unrelated.py::doForPriorityMassage",
        "JediTaskSpec.currentPriority",
    )
    _attach(
        *_dogs(),
        (stranger, "pandaserver/dataservice/unrelated.py"),
        junctions=[junction],
    )

    assert junction.entry_points == []


_REFINER_BASE = '''
class TaskRefinerBase:
    def doBasicRefine(self, task_params):
        return 1
'''

_DISPATCH_WORKER = '''
from refiner_base import TaskRefinerBase


class TaskRefiner(FactoryBase):
    def instantiateImpl(self, vo, label, *args) -> Optional[TaskRefinerBase]:
        return FactoryBase.instantiateImpl(self, vo, label, *args)

    def start(self):
        while True:
            time.sleep(jedi_config.taskrefine.loopCycle)
            worker = TaskRefinerThread(self)
            worker.start()


class TaskRefinerThread(WorkerThread):
    def __init__(self, implFactory: TaskRefiner):
        self.implFactory = implFactory

    def runImpl(self):
        impl = self.implFactory.instantiateImpl("atlas", "managed")
        if impl is None:
            return
        impl.doRefine(1, {})
'''

_DISPATCH_REFINER = '''
from refiner_base import TaskRefinerBase


class AtlasProdTaskRefiner(TaskRefinerBase):
    def doRefine(self, task_id, task_params):
        self.doBasicRefine(task_params)
        return 0
'''

_DISPATCH_REFINER_RIVAL = '''
from refiner_base import TaskRefinerBase


class AtlasAnalTaskRefiner(TaskRefinerBase):
    def doRefine(self, task_id, task_params):
        return 0
'''


def test_the_factory_a_worker_was_handed_resolves_through_its_annotation():
    """21 of the 24 sites put the factory in a worker thread first.

    ``TaskRefiner.start`` hands ``self`` to ``TaskRefinerThread``, which
    keeps it as ``self.implFactory`` and calls ``instantiateImpl`` on that --
    so the receiver is ``self.<attr>`` and only the parameter annotation says
    what it is.  ``Optional[X]`` here and ``X | None`` above, because the
    corpus would be free to write either and an annotation nothing reads is
    the failure this is guarding.

    Both junctions are asserted: the plugin's own ``doRefine`` and the base's
    ``doBasicRefine``, which needs the inherited hop as well -- that pair is
    14 of the 24.
    """
    own = _junction(
        "pandajedi/jedirefine/AtlasProdTaskRefiner.py::doRefine",
        "JediTaskSpec.status",
    )
    inherited = _junction(
        "pandajedi/jedirefine/refiner_base.py::doBasicRefine",
        "JediTaskSpec.splitRule",
    )
    _attach(
        (_REFINER_BASE, "pandajedi/jedirefine/refiner_base.py"),
        (_DISPATCH_WORKER, "pandajedi/jediorder/TaskRefiner.py"),
        (_DISPATCH_REFINER, "pandajedi/jedirefine/AtlasProdTaskRefiner.py"),
        (_DISPATCH_REFINER_RIVAL, "pandajedi/jedirefine/AtlasAnalTaskRefiner.py"),
        junctions=[own, inherited],
    )

    assert [e.entry for e in own.entry_points] == ["pandajedi/jediorder/TaskRefiner.py"]
    assert [e.entry for e in inherited.entry_points] == [
        "pandajedi/jediorder/TaskRefiner.py"
    ]


def test_the_log_question_does_not_follow_a_dispatched_call():
    """Whose log says it ran is answered by the immediate caller.

    ``AtlasProdWatchDog`` declares its own logger and writes to
    ``panda-AtlasProdWatchDog.log``; the knight that dispatched it writes to
    ``panda-jedi.log``.  Following the dispatch here would name the wrong
    file, which is the P1-37 failure.
    """
    modules = [_module(text, rel) for text, rel in _dogs()]

    def entries(**flags):
        reached = trigger.reaching_modules(modules, through_doors=True, **flags)
        found = reached.get("pandajedi/jedidog/AtlasProdWatchDog.py", {})
        return [e for e, _door, _call in found.get("doForPriorityMassage", ())]

    assert entries() == []
    assert entries(through_dispatch=True) == ["pandajedi/jediorder/WatchDog.py"]


_GENERIC_FACTORY = '''
from typing import Generic, TypeVar

T = TypeVar("T")


class FactoryBase(Generic[T]):
    def getImpl(self, vo, label) -> T | None:
        return self.implMap.get((vo, label))

    def instantiateImpl(self, vo, label, *args) -> T | None:
        return self.classMap.get((vo, label))
'''

_GENERIC_KNIGHT = '''
from dog_base import WatchDogBase
from pandajedi.jedicore.FactoryBase import FactoryBase


class WatchDog(FactoryBase[WatchDogBase]):
    def start(self):
        while True:
            time.sleep(jedi_config.watchdog.loopCycle)
            impl = self.getImpl("atlas", "managed")
            if impl is not None:
                impl.doAction()
'''


def test_a_subscripted_base_class_is_still_a_base_class():
    """``FactoryBase[WatchDogBase]`` states the same descent ``FactoryBase`` does.

    ``ast.Subscript`` carries neither ``id`` nor ``attr``, so the base was read
    as nothing and dropped -- which would take ``FactoryBase`` out of eight
    classes' ancestry the moment the corpus parameterises it, and four readers
    of the hierarchy with it.
    """
    source = (
        "class JediKnight:\n    pass\n"
        "\n"
        "class WatchDog(JediKnight, FactoryBase[WatchDogBase]):\n    pass\n"
        "\n"
        "class Plain(JediKnight, FactoryBase):\n    pass\n"
    )
    module = _module(source, "pandajedi/jediorder/WatchDog.py")
    bases = attribution.class_bases([module])

    assert bases["WatchDog"] == ["JediKnight", "FactoryBase"]
    # The unsubscripted form is what the corpus writes today, and it must read
    # exactly as it did -- this rule is a widening, not a replacement.
    assert bases["Plain"] == ["JediKnight", "FactoryBase"]
    # Every other reader of a base clause reads it the same way: the subclass
    # edges a mixin's log file and a dispatched worker are found by, and the
    # same-module lineage the walk reads a field through.  ``calls.Types.lineage``
    # is held in test_codemap_calls.
    edges = trigger.subclass_edges([module])
    assert sorted(node.name for node, _where in edges["FactoryBase"]) == ["Plain", "WatchDog"]
    factory = ast.parse("class FactoryBase:\n    pass\nclass WatchDog(FactoryBase[WatchDogBase]):\n    pass\n")
    assert [held.name for held in pathcond.bases_in(factory, factory.body[1])] == ["FactoryBase"]


def test_a_type_argument_is_not_mistaken_for_a_base_class():
    """The family a base is parameterised *with* is not something the class derives from.

    ``WatchDog`` is a ``FactoryBase``; it is not a ``WatchDogBase``.  Recording
    the argument as a base would put every knight in its own plugins' family and
    let ``_declared_family`` match the knight's own module.
    """
    source = "class WatchDog(FactoryBase[WatchDogBase]):\n    pass\n"
    bases = attribution.class_bases([_module(source, "pandajedi/jediorder/WatchDog.py")])

    assert bases["WatchDog"] == ["FactoryBase"]


def _generic_dogs():
    return (
        (_GENERIC_FACTORY, "pandajedi/jedicore/FactoryBase.py"),
        (_DISPATCH_BASE, "pandajedi/jedidog/dog_base.py"),
        (_GENERIC_KNIGHT, "pandajedi/jediorder/WatchDog.py"),
        (_DISPATCH_PLUGIN, "pandajedi/jedidog/AtlasProdWatchDog.py"),
        (_DISPATCH_RIVAL, "pandajedi/jedidog/AtlasAnalWatchDog.py"),
    )


def test_a_dispatched_call_resolves_through_the_class_type_argument():
    """The family is stated once, on the knight's own base clause.

    ``FactoryBase.getImpl`` returns ``T | None``; which class ``T`` is comes
    from ``class WatchDog(FactoryBase[WatchDogBase])``.  That is one statement
    per knight instead of an override whose only purpose is to restate a type,
    and it is the form that also carries the worker-thread case below.
    """
    junction = _junction(
        "pandajedi/jedidog/AtlasProdWatchDog.py::doForPriorityMassage",
        "JediTaskSpec.currentPriority",
    )
    _attach(*_generic_dogs(), junctions=[junction])

    assert [(e.trigger, e.entry) for e in junction.entry_points] == [
        ("polled", "pandajedi/jediorder/WatchDog.py")
    ]


_PPROCESS_BASE = '''
class PostProcessorBase:
    def doBasicPostProcess(self, task):
        return 1
'''

_GENERIC_WORKER = '''
from pandajedi.jedicore.FactoryBase import FactoryBase
from pprocess_base import PostProcessorBase


class PostProcessor(FactoryBase[PostProcessorBase]):
    def start(self):
        while True:
            time.sleep(jedi_config.postprocessor.loopCycle)
            worker = PostProcessorThread(self)
            worker.start()


class PostProcessorThread(WorkerThread):
    def __init__(self, implFactory: FactoryBase[PostProcessorBase]):
        self.implFactory = implFactory

    def post_process_tasks(self, task_list):
        impl = self.implFactory.instantiateImpl("atlas", "managed")
        if impl is None:
            return
        impl.doPostProcess(task_list)
'''

_GENERIC_PPROCESS = '''
from pprocess_base import PostProcessorBase


class AtlasProdPostProcessor(PostProcessorBase):
    def doPostProcess(self, task):
        self.doBasicPostProcess(task)
        return 0
'''

_GENERIC_PPROCESS_RIVAL = '''
from pprocess_base import PostProcessorBase


class AtlasAnalPostProcessor(PostProcessorBase):
    def doPostProcess(self, task):
        return 0
'''


def test_a_dispatched_call_resolves_through_a_parameterised_parameter():
    """21 of the 24 sites reach the factory through a worker thread's parameter.

    ``PostProcessorThread`` is constructed twice -- by the knight with ``self``
    and by a message processor with a bare factory -- so no concrete class names
    that parameter and only the parameterised base does.  This is the case a
    concrete override cannot reach, and the reason the generic form is what was
    asked for upstream.
    """
    own = _junction(
        "pandajedi/jedipprocess/AtlasProdPostProcessor.py::doPostProcess",
        "JediTaskSpec.status",
    )
    inherited = _junction(
        "pandajedi/jedipprocess/pprocess_base.py::doBasicPostProcess",
        "JediTaskSpec.splitRule",
    )
    _attach(
        (_GENERIC_FACTORY, "pandajedi/jedicore/FactoryBase.py"),
        (_PPROCESS_BASE, "pandajedi/jedipprocess/pprocess_base.py"),
        (_GENERIC_WORKER, "pandajedi/jediorder/PostProcessor.py"),
        (_GENERIC_PPROCESS, "pandajedi/jedipprocess/AtlasProdPostProcessor.py"),
        (_GENERIC_PPROCESS_RIVAL, "pandajedi/jedipprocess/AtlasAnalPostProcessor.py"),
        junctions=[own, inherited],
    )

    assert [e.entry for e in own.entry_points] == ["pandajedi/jediorder/PostProcessor.py"]
    assert [e.entry for e in inherited.entry_points] == [
        "pandajedi/jediorder/PostProcessor.py"
    ]


def _reasons(*sources: tuple[str, str], junctions, tables=None):
    modules = [_module(text, rel) for text, rel in sources]
    trigger.attach(junctions, modules, tables or {"PRODSYS_COMM"})
    return trigger.unreached_reasons(junctions, modules, tables or {"PRODSYS_COMM"})


def test_a_junction_nothing_calls_is_told_from_one_whose_caller_starts_nothing():
    """The ratio covered both, and only one of them is work this slice could do.

    ``lonely`` is called by nobody; ``driven`` is called by ``helper.py``, which
    carries no trigger of its own.  Answering the second means asking what
    starts ``helper``, which is the hop the module docstring refuses.
    """
    lonely = _junction("proxy.py::lonely")
    driven = _junction("proxy.py::driven")
    reasons = _reasons(
        (
            "class P:\n"
            "    def lonely(self):\n        return 1\n"
            "    def driven(self):\n        return 2\n",
            "proxy.py",
        ),
        ("import proxy\n\ndef helper(p):\n    return p.driven()\n", "helper.py"),
        junctions=[lonely, driven],
    )

    assert reasons[trigger.NO_CALLER_RESOLVED] == [lonely]
    assert reasons[trigger.CALLER_STARTS_NOTHING] == [driven]
    assert trigger.UNCLASSIFIED not in reasons


def test_a_caller_in_the_owners_own_module_still_leaves_it_in_this_bucket():
    """And the bucket is named for that, because it is a narrowing and not an
    absence: ``what starts this`` is not answered by a sibling, only moved to
    what starts the sibling.  4 of the corpus's 32 have a caller like this, and
    a name reading "no caller the map resolves" said the map had found none."""
    inner = _junction("watchdog.py::do_reassign")
    reasons = _reasons(
        (
            "class W:\n"
            "    def doAction(self, log):\n        return self.do_reassign(log)\n"
            "    def do_reassign(self, log):\n        return 1\n",
            "watchdog.py",
        ),
        junctions=[inner],
    )

    assert reasons == {trigger.NO_CALLER_RESOLVED: [inner]}
    assert "in another module" in trigger.NO_CALLER_RESOLVED


def test_module_level_code_is_not_reported_as_uncalled():
    """"Who imports it" is a different question, so it gets its own answer."""
    at_import = _junction("sitemapper.py::<module>")
    reasons = _reasons(("x = 1\n", "sitemapper.py"), junctions=[at_import])

    assert reasons == {trigger.RUNS_AT_IMPORT: [at_import]}


def test_a_reached_junction_is_given_no_reason():
    """The groups have to add up to ``total - reached`` or they mean nothing."""
    reached = _junction("daemons/scripts/d.py::f")
    reasons = _reasons(
        ("def f():\n    return 1\n", "daemons/scripts/d.py"), junctions=[reached]
    )

    assert reached.entry_points
    assert reasons == {}


def test_a_subject_no_loop_reaches_does_not_repair_itself():
    """The question a stalled task actually asks: will waiting help?"""
    polled = _junction("a.py::f", "JediTaskSpec.status")
    polled.entry_points = [EntryPoint(trigger="polled", entry="a.py")]
    once = _junction("b.py::g", "T_TASK.vo")
    once.entry_points = [EntryPoint(trigger="command", entry="b.py")]

    assert trigger.fragile_subjects([polled, once]) == [("T_TASK.vo", ["command"])]


# --------------------------------------------------------------------------- #
# selection
# --------------------------------------------------------------------------- #

_BROKER = '''
class JobBrokerBase:
    def add_summary_message(self, old_list, new_list, message, tmp_log, msg_map):
        for site in msg_map:
            tmp_log.info(msg_map[site])
        tmp_log.info(f"{len(new_list)} candidates passed {message}")

class AtlasProdJobBroker(JobBrokerBase):
    def doBrokerage(self):
        newScanSiteList = []
        msg_map = {}
        for tmpSiteName in scanSiteList:
            if diskio_usage > diskio_limit and diskio_task > diskio_limit:
                msg_map[tmpSiteName] = f"  skip site={tmpSiteName} due to diskIO overload criteria=-diskIO"
            newScanSiteList.append(tmpSiteName)
        self.add_summary_message(oldScanSiteList, scanSiteList, "diskIO check", tmpLog, msg_map)
        newScanSiteList = []
        for tmpSiteName in scanSiteList:
            if taskSpec.ioIntensity > site_max:
                msg_map[tmpSiteName] = f"  skip site={tmpSiteName} since ioIntensity={taskSpec.ioIntensity} criteria=-max_io_intensity"
            newScanSiteList.append(tmpSiteName)
        self.add_summary_message(oldScanSiteList, scanSiteList, "IO intensity check", tmpLog, msg_map)
'''

_UNTAGGED = '''
class GenJobBroker:
    def doBrokerage(self):
        newScanSiteList = []
        for tmpSiteName in scanSiteList:
            if minDiskCount > tmpSiteSpec.maxwdir:
                tmpLog.debug(f"  skip {tmpSiteName} due to small scratch disk")
                continue
            newScanSiteList.append(tmpSiteName)
        scanSiteList = newScanSiteList
        tmpLog.debug(f"{len(scanSiteList)} candidates passed scratch disk check")
'''


def _selection(*sources: tuple[str, str]):
    modules = [_module(text, rel) for text, rel in sources]
    return selection.extract(modules, MAP_ID, VERSION)


def test_a_tagged_rejection_carries_its_condition():
    """The tag and the guard that reaches it are the whole content of a stage.

    ``criteria=-diskIO`` is what production logs carry per rejected site, so a
    stage keyed on it can be counted from the logs and explained from the map
    in one step.
    """
    stages, _cov, _gaps = _selection((_BROKER, "pandajedi/jedibrokerage/AtlasProdJobBroker.py"))

    diskio = next(s for s in stages if s.criteria_tag == "-diskIO")
    assert diskio.funnel_label == "diskIO check"
    assert diskio.conditions == [
        "diskio_usage > diskio_limit and diskio_task > diskio_limit"
    ]
    assert diskio.inputs == ["diskio_usage", "diskio_limit", "diskio_task"]


_VARIABLE_TAG = '''
class JobBrokerBase:
    def add_summary_message(self, old_list, new_list, message, tmp_log, msg_map):
        for site in msg_map:
            tmp_log.info(msg_map[site])

class AtlasProdJobBroker(JobBrokerBase):
    def doBrokerage(self, taskSpec, scanSiteList):
        nucleus = taskSpec.nucleus
        if nucleus:
            msg_map = {}
            for tmpPandaSiteName in scanSiteList:
                criteria = "-link_unusable"
                reason = ""
                if nucleus == tmpAtlasSiteName:
                    pass
                elif nucleus in unwritable_over_wan:
                    reason = "unwritable over WAN"
                    criteria = "-dest_blacklisted"
                elif totalQueued >= self.total_queue_threshold:
                    reason = "too many queued"
                    criteria = "-links_full"
                elif closeness == BLOCKED_LINK:
                    reason = "blocked link"
                tmpStr = f"  skip site={tmpPandaSiteName} due to {reason}"
                tmpStr += f": criteria={criteria}"
                msg_map[tmpPandaSiteName] = tmpStr
            self.add_summary_message(oldScanSiteList, scanSiteList, "link check", tmpLog, msg_map)
'''


def test_a_tag_assigned_to_a_variable_is_still_a_tag():
    """The blind spot ``tags-are-known`` found from the other side.

    ``AtlasProdJobBroker`` decides three of its reasons by assigning them to a
    variable and interpolating it later, so nothing in the message names a tag.
    Production logs ``criteria=-link_unusable`` and the map had no stage for it,
    while the funnel counted a cut at "link check" the slice could not explain --
    the same hole seen twice.
    """
    stages, _cov, gaps = _selection(
        (_VARIABLE_TAG, "pandajedi/jedibrokerage/AtlasProdJobBroker.py")
    )

    by_tag = {s.criteria_tag: s for s in stages}
    assert set(by_tag) == {"-link_unusable", "-dest_blacklisted", "-links_full"}
    assert {s.funnel_label for s in stages} == {"link check"}
    # Anchored where the reason is decided, not where the message is built.
    assert by_tag["-dest_blacklisted"].anchor.line_start < by_tag["-links_full"].anchor.line_start
    assert gaps == []


def test_a_default_tag_carries_what_would_have_replaced_it():
    """Reassignment is the one thing dominating-guard analysis cannot see.

    ``criteria = "-link_unusable"`` sits above the chain that overwrites it, so
    its own guard is necessary and not sufficient; left at that the map would
    claim this cut happens whenever the task has a nucleus.
    """
    stages, _cov, _gaps = _selection(
        (_VARIABLE_TAG, "pandajedi/jedibrokerage/AtlasProdJobBroker.py")
    )

    default = next(s for s in stages if s.criteria_tag == "-link_unusable")

    # Each replacement as every test it runs under: an ``elif`` runs where the
    # arms above it did not.
    assert default.conditions == [
        "nucleus  [nucleus := taskSpec.nucleus]",
        "not (not (nucleus == tmpAtlasSiteName) and nucleus in unwritable_over_wan)",
        "not (not (nucleus == tmpAtlasSiteName) and not (nucleus in unwritable_over_wan)"
        " and totalQueued >= self.total_queue_threshold)",
    ]


def test_a_sibling_branch_is_not_treated_as_overwriting():
    """Comparing nesting depth looked right and was not: Python nests an ``elif``
    inside the previous ``if``'s ``orelse``, so a sibling is always deeper.  That
    put ``not (totalQueued >= limit)`` on the cut for a blacklisted destination,
    a condition with nothing to do with it."""
    stages, _cov, _gaps = _selection(
        (_VARIABLE_TAG, "pandajedi/jedibrokerage/AtlasProdJobBroker.py")
    )

    blacklisted = next(s for s in stages if s.criteria_tag == "-dest_blacklisted")

    assert blacklisted.conditions == [
        "nucleus  [nucleus := taskSpec.nucleus]",
        "not (nucleus == tmpAtlasSiteName)",
        "nucleus in unwritable_over_wan",
    ]


def test_a_variable_tag_promises_no_template():
    """``": criteria={}"`` is the tail of a message assembled across statements
    and is shared by every rejection in the file, so offering it as the line to
    look for would confirm nothing."""
    stages, _cov, _gaps = _selection(
        (_VARIABLE_TAG, "pandajedi/jedibrokerage/AtlasProdJobBroker.py")
    )

    assert all(s.emits == [] for s in stages)
    # The level still comes from the helper that logs the map.
    assert {s.log_level for s in stages} == {"info"}


def test_an_interpolated_criteria_that_is_not_a_bare_name_is_not_a_tag():
    """``f"start with criteria={str(criteria)}"`` is a progress log, not a
    rejection.  Requiring a bare name keeps the two other interpolated
    ``criteria=`` messages in the corpus out without naming them."""
    source = '''
class TaskEventModule:
    def send_command(self, criteria):
        tmpLog.debug(f"start with criteria={str(criteria)}")
'''

    stages, _cov, _gaps = _selection((source, "pandaserver/taskbuffer/task_event_module.py"))

    assert stages == []


def test_stage_order_follows_the_chain():
    """"Which step cut the candidates" is a question about position."""
    stages, _cov, _gaps = _selection((_BROKER, "pandajedi/jedibrokerage/AtlasProdJobBroker.py"))

    assert [(s.criteria_tag, s.order) for s in stages] == [
        ("-diskIO", 0),
        ("-max_io_intensity", 1),
    ]


def test_the_level_comes_from_the_helper_that_logs_it():
    """Most rejections are never logged where they are written.

    The broker fills a ``msg_map`` and hands it to ``add_summary_message``.
    Recording ``None`` would say the level is unknown when one hop settles it,
    and the level is what decides whether the observation exists in production.
    """
    stages, _cov, _gaps = _selection((_BROKER, "pandajedi/jedibrokerage/AtlasProdJobBroker.py"))

    assert {s.log_level for s in stages} == {"info"}


def test_a_named_step_with_no_tag_is_read_from_its_continues():
    """``GenJobBroker`` names eight steps and tags one.

    Without this, seven cuts in a production broker would be invisible -- and
    it is the file the chain matcher handles best, so a recognizer that only
    followed tags would have a hole exactly where the plan expected one.
    """
    stages, _cov, _gaps = _selection((_UNTAGGED, "pandajedi/jedibrokerage/GenJobBroker.py"))

    assert len(stages) == 1
    assert (stages[0].criteria_tag, stages[0].funnel_label) == ("", "scratch disk check")
    assert stages[0].conditions == ["minDiskCount > tmpSiteSpec.maxwdir"]
    assert stages[0].log_level == "debug"


def test_a_templated_step_name_is_not_a_step():
    """``f"{len(new_list)} candidates passed {message}"`` is the helper itself.

    The code templated the step name rather than naming one, so counting it as
    a step invents a cut inside the function that reports every other cut.
    """
    helper = (
        "class JobBrokerBase:\n"
        "    def add_summary_message(self, old_list, new_list, message, tmp_log, msg_map):\n"
        "        tmp_log.info(f'{len(new_list)} candidates passed {message}')\n"
    )
    stages, coverage, gaps = _selection((helper, "pandajedi/jedibrokerage/JobBrokerBase.py"))

    assert (stages, coverage, gaps) == ([], [], [])


def test_a_step_named_after_a_run_time_value_is_still_a_step():
    """A hole with words around it is a name; a hole alone is not.

    ``endpoint check with DISK_THRESHOLD={} TB`` is a step this broker runs
    1176 times in a day's logs, and requiring the name to be fixed left the
    funnel counting a cut the map could not place.
    """
    source = (
        "class B:\n"
        "    def runImpl(self):\n"
        "        for n in nucleusList:\n"
        "            if bad(n):\n"
        "                tmpLog.info(f'  skip nucleus={n} criteria=-space')\n"
        "                continue\n"
        "        tmpLog.info(f'{len(nucleusList)} candidates passed endpoint check "
        "with DISK_THRESHOLD={thr} TB')\n"
    )
    stages, _cov, gaps = _selection((source, "pandajedi/jedibrokerage/AtlasProdTaskBroker.py"))

    assert [s.funnel_label for s in stages] == ["endpoint check with DISK_THRESHOLD={} TB"]
    assert gaps == []


def test_a_step_reported_twice_is_one_step_named_by_its_funnel_line():
    """``AtlasProdTaskBroker`` writes its own funnel line and then files a
    summary entry, and at one step the two disagree on the name.

    Read as two steps the summary entry counts a cut whose every reason
    attached to the line above it.  The funnel line's name wins: it is the one
    production puts on a ``candidates passed`` line.
    """
    source = (
        "class B:\n"
        "    def runImpl(self):\n"
        "        for n in nucleusList:\n"
        "            if bad(n):\n"
        "                tmpLog.info(f'  skip nucleus={n} criteria=-space')\n"
        "                continue\n"
        "        tmpLog.info(f'{len(nucleusList)} candidates passed endpoint check "
        "with DISK_THRESHOLD={thr} TB')\n"
        "        self.add_summary_message(old, new, 'storage endpoint check')\n"
    )
    stages, _cov, gaps = _selection((source, "pandajedi/jedibrokerage/AtlasProdTaskBroker.py"))

    assert [(s.criteria_tag, s.funnel_label) for s in stages] == [
        ("-space", "endpoint check with DISK_THRESHOLD={} TB")
    ]
    assert gaps == []


def test_a_candidate_not_appended_is_cut_as_surely_as_one_skipped():
    """A loop can drop a candidate by skipping ahead or by keeping the
    survivors elsewhere, and the two are the same cut written differently.

    This is ``AtlasProdJobBroker``'s deferred "temporary problem check" -- 1111
    runs of it in a day's logs, and the reason the funnel could count
    candidates disappearing where the map had nothing to say.
    """
    source = (
        "class B:\n"
        "    def doBrokerage(self):\n"
        "        for tmpSiteName in scanSiteList:\n"
        "            if tmpSiteName in siteSkippedTmp:\n"
        "                msg_map[tmpSiteName] = siteSkippedTmp[tmpSiteName]\n"
        "            else:\n"
        "                newScanSiteList.append(tmpSiteName)\n"
        "        self.add_summary_message(old, new, 'temporary problem check', log, msg_map)\n"
    )
    stages, _cov, gaps = _selection((source, "pandajedi/jedibrokerage/AtlasProdJobBroker.py"))

    assert [(s.criteria_tag, s.funnel_label) for s in stages] == [
        ("", "temporary problem check")
    ]
    # The condition is the arm that does not append, positively: no negation to
    # compose, and no passthrough invented for a local that is not a place a
    # value lives.
    assert stages[0].conditions == ["tmpSiteName in siteSkippedTmp"]
    assert gaps == []


def test_one_tag_used_at_two_steps_stays_two_stages():
    """``criteria=-disk`` is emitted by both "disk check" and "Storage check".

    The tag alone is ambiguous within a chain; the funnel counter is what tells
    a reader which of the two a log line came from, so folding them together
    would lose one of the cuts.
    """
    source = (
        "class B:\n"
        "    def doBrokerage(self):\n"
        "        if a > b:\n"
        "            msg = f'skip criteria=-disk'\n"
        "        self.add_summary_message(old, new, 'disk check', log, msg_map)\n"
        "        if c > d:\n"
        "            msg = f'skip criteria=-disk'\n"
        "        self.add_summary_message(old, new, 'Storage check', log, msg_map)\n"
    )
    stages, _cov, _gaps = _selection((source, "pandajedi/jedibrokerage/AtlasProdJobBroker.py"))

    assert [(s.criteria_tag, s.funnel_label) for s in stages] == [
        ("-disk", "disk check"),
        ("-disk", "Storage check"),
    ]


def test_a_step_whose_reason_cannot_be_read_is_named():
    """The funnel will report the cut and the map has nothing to say about it."""
    source = (
        "class B:\n"
        "    def doBrokerage(self):\n"
        "        scanSiteList = [s for s in scanSiteList if keep(s)]\n"
        "        self.add_summary_message(old, new, 'link check', log, msg_map)\n"
    )
    _stages, _cov, gaps = _selection((source, "pandajedi/jedibrokerage/AtlasProdJobBroker.py"))

    assert gaps == [
        "pandajedi/jedibrokerage/AtlasProdJobBroker.py::doBrokerage "
        "counts a cut at 'link check' with no readable reason"
    ]


def test_a_field_a_filter_stage_gates_is_promoted():
    """Criterion 1 reads SQL predicates, and brokerage gates in Python.

    ``nucleus``, ``minRamCount`` and ``ioIntensity`` decide whether a site
    survives the chain without appearing in any ``WHERE``, so the first
    criterion cannot see them and the fifth restates it where it can.
    """
    fragment = _fragment_with(("JediTaskSpec", "ioIntensity", [("x", 2)]))
    assert promotion.criteria_for(fragment, Counter(), {}) == {}

    fragment.filter_stages.append(
        FilterStageNode(
            map_id=MAP_ID,
            derived_from=VERSION,
            name="f:-max_io_intensity",
            owner="AtlasProdJobBroker.py::doBrokerage",
            criteria_tag="-max_io_intensity",
            inputs=["ioIntensity", "max_io_intensity"],
        )
    )

    assert promotion.criteria_for(fragment, Counter(), {}) == {
        "JediTaskSpec.ioIntensity": ["5:gates-a-filter-stage"]
    }


def test_a_rejection_that_logs_none_of_what_it_tested_is_reported():
    """What a rejection did not log cannot be checked afterwards.

    ``criteria=-diskIO`` compares three numbers and logs only the site name;
    the numbers are on a separate line a different path emits.
    """
    fragment = MapFragment(map_id=MAP_ID, derived_from=VERSION)
    fragment.filter_stages.append(
        FilterStageNode(
            map_id=MAP_ID,
            derived_from=VERSION,
            name="f:-diskIO",
            owner="b.py::doBrokerage",
            criteria_tag="-diskIO",
            conditions=["diskio_usage > diskio_limit"],
            inputs=["diskio_usage", "diskio_limit"],
            emits=["  skip site={} due to diskIO overload criteria=-diskIO"],
        )
    )

    assert gates.unexplainable_rejections(fragment) == [
        ("-diskIO", ["diskio_usage", "diskio_limit"], "b.py::doBrokerage")
    ]


# --------------------------------------------------------------------------- #
# graph invariants
# --------------------------------------------------------------------------- #


def _graph(*, subjects, junctions):
    return MapFragment(
        map_id=MAP_ID, derived_from=VERSION, subjects=subjects, junctions=junctions
    )


def test_a_declared_status_nothing_writes_is_unreachable():
    """The sound direction of the comparison that was refuted the other way.

    Checking that every outcome sits inside a declared list fails on correct
    code -- those lists are purpose-built subsets.  Checking that every
    declared value is written somewhere is a genuine lower bound, and it could
    only be run once every write form was in.
    """
    fragment = _fragment_with(("JediTaskSpec", "status", [("defined", 1)]))
    fragment.subjects[0].vocabulary = ["defined", "aborting"]
    result = gates.declared_status_is_written(fragment)

    assert not result.passed
    assert result.checked == 2
    assert result.failures == [
        "JediTaskSpec.status declares 'aborting' and no writer produces it"
    ]


def test_a_run_time_writer_is_named_alongside_the_miss():
    """``commandStatusMap()[cmd]["doing"]`` produces 'aborting' with no literal.

    The writer is in the map and the value is not, which is a different thing
    from nothing writing it at all -- so the count goes in the failure rather
    than being left for the reader to work out.
    """
    fragment = _fragment_with(("JediTaskSpec", "status", [("runtime(newStatus)", 2)]))
    fragment.subjects[0].vocabulary = ["aborting"]

    assert gates.declared_status_is_written(fragment).failures == [
        "JediTaskSpec.status declares 'aborting' and no writer produces it "
        "(1 run-time writer(s) could)"
    ]


def test_every_junction_lands_on_a_subject_the_map_contains():
    """A backward walk that steps onto a missing node cannot report that it did."""
    fragment = _fragment_with(("JediTaskSpec", "status", [("ready", 1)]))
    fragment.junctions.append(
        JunctionNode(
            map_id=MAP_ID,
            derived_from=VERSION,
            name="j:ghost",
            subject="JediTaskSpec.ghost",
            owner="x.py::f",
            branches=[Branch(outcome="ready")],
        )
    )
    result = gates.map_references_resolve(fragment)

    assert not result.passed
    assert result.failures == ["x.py::f writes JediTaskSpec.ghost, which is not a subject"]


def test_a_class_declaring_columns_the_reader_cannot_parse_is_a_finding():
    """A declaration form nobody reads is silent in exactly the wrong way.

    ``WFDataSpec`` states its columns as ``AttributeWithType("status", str)``
    and derives the older name from that, so a reader taking only literal
    tuples matched the assignment and took nothing.  Twenty-four writes in one
    module went unattributed and ``unresolved`` swallowed them, because no gate
    was looking at a class that declares nothing.
    """
    fragment = MapFragment(
        map_id=MAP_ID,
        derived_from=VERSION,
        declaration_yields={"JobSpec (JobSpec.py)": 126, "WFDataSpec (workflow_base.py)": 0},
    )

    result = gates.spec_declarations_are_read(fragment)

    assert not result.passed
    assert result.checked == 2
    assert result.failures == [
        "WFDataSpec (workflow_base.py) declares columns and the extraction read none"
    ]


# --------------------------------------------------------------------------- #
# auditing the element types the map takes on trust
# --------------------------------------------------------------------------- #
#
# ``Dict[str, DatasetSpec]`` is asked of PanDA because a bare ``Dict`` says
# nothing, which makes the element type a fact with no second reading -- and one
# went in naming the wrong class while resolving nothing.  These two gates are
# the second reading.


def _audited(*rows):
    return MapFragment(map_id=MAP_ID, derived_from=VERSION, annotation_readings=list(rows))


def test_an_annotation_the_code_contradicts_is_a_finding():
    """The failure that prompted the gate, reduced to its shape.

    ``row_id_spec_map: Dict[int, JediFileSpec]`` sat one line above
    ``row_id_spec_map[fileSpec.row_ID] = fileSpec``, where ``fileSpec`` comes
    from ``JobSpec.Files`` and only ``FileSpec`` declares ``row_ID``.  No other
    gate could see it: ``structural_attribution_agrees`` compares two readings
    of one expression, and these are an expression apart.
    """
    fragment = _audited(
        AnnotationAudit(
            where="task_event_module.py:2949",
            container="row_id_spec_map",
            stated="JediFileSpec",
            put_in=["FileSpec"],
            read=True,
        )
    )

    result = gates.container_annotations_agree(fragment)

    assert not result.passed
    assert result.failures == [
        "task_event_module.py:2949 says row_id_spec_map holds JediFileSpec, "
        "but the code puts in FileSpec"
    ]


def test_no_independent_reading_is_not_evidence_against_an_annotation():
    """An unresolvable value is the case the annotation was asked for.

    Silence has to stay silence here.  Treating "nothing else could be read" as
    disagreement would fire on every container the annotation exists to
    explain, which is all of the useful ones.
    """
    fragment = _audited(
        AnnotationAudit(
            where="closer.py:49",
            container="dataset_map",
            stated="DatasetSpec",
            put_in=[],
            read=True,
        )
    )

    assert gates.container_annotations_agree(fragment).passed


def test_an_annotation_that_changes_no_write_is_a_finding():
    """The other half of the same failure: correct class, read form missing.

    ``data_spec_map: Dict[str, WFDataSpec]`` named the right class and closed
    nothing, because the receiver is bound by ``data_spec_map.get(name)`` and
    only ``d[key]`` was read.  It was reported as resolving three writes.

    What makes it a finding rather than a redundancy is the write left over: a
    ``WFDataSpec`` attribute is written in that scope and still resolves by
    inference or not at all, which is what a missing read form looks like from
    here.
    """
    fragment = _audited(
        AnnotationAudit(
            where="workflow_core.py:859",
            container="data_spec_map",
            stated="WFDataSpec",
            put_in=[],
            read=False,
            unsettled=["880:status"],
        )
    )

    result = gates.annotations_are_read(fragment)

    assert not result.passed
    assert result.failures == [
        "workflow_core.py:859 states WFDataSpec for data_spec_map "
        "and no write in scope resolves differently without it"
    ]


def test_an_annotation_a_stronger_reading_made_redundant_is_not_a_finding():
    """The population changed under this gate, and the verdict had to follow.

    It was built when the only spec annotations in panda-server were the ones
    this map asked for -- five of them -- so "nothing depends on it" meant
    something was wrong.  An annotation campaign then annotated the tree, and
    inert became the normal case: ``closer.py`` states ``DatasetSpec`` for a
    mapping whose element type the extraction now learns from what the code
    puts in it, so the annotation changes nothing and nothing is wrong.

    The remaining defect is narrower and still fails: an annotation that
    changes nothing *while a write of the class it names is unsettled*.
    """
    fragment = _audited(
        AnnotationAudit(
            where="closer.py:54",
            container="dataset_map",
            stated="DatasetSpec",
            put_in=["DatasetSpec"],
            read=False,
            unsettled=[],
        )
    )

    result = gates.annotations_are_read(fragment)

    assert result.passed
    assert "1 more state a class the map reaches another way" in (result.note or "")


def test_an_annotation_the_extraction_cannot_read_is_a_finding():
    """The half of the declaration lesson that had no gate.

    A declaration drifting out of reach is caught by counting what came back.
    An annotation drifting out of reach was caught by nothing: the reader
    returns nothing, and that is indistinguishable from an annotation about
    ``int``. Four forms of one fact have now drifted, each found by a person
    reading source.
    """
    fragment = MapFragment(
        map_id=MAP_ID,
        derived_from=VERSION,
        spec_annotation_forms={
            "setupper_plugin_base.py:13": "JobSpec",
            "adder_gen.py:42": "",
        },
    )

    result = gates.annotation_forms_are_understood(fragment)

    assert not result.passed
    assert result.checked == 2
    assert result.failures == [
        "adder_gen.py:42 names a declared spec class and the extraction read none"
    ]


def test_an_annotation_with_no_write_in_its_scope_is_not_audited():
    """A comparison that never happens must not be reported as one.

    The ablation removes the annotation and re-resolves the writes around it.
    Where there are none, both runs answer the same thing for the same reason
    -- there was nothing to answer -- and counting that as an annotation the
    map depends on inflates the gate with rows it never examined.  Twenty-one
    of the checkout's fifty-seven audited rows were this.
    """
    source = (
        "class Setupper:\n"
        "    def __init__(self):\n"
        "        self.jobs: List[JobSpec] = []\n"
        "\n"
        "    def count(self):\n"
        "        total = 0\n"
        "        for job in self.jobs:\n"
        "            total += job.PandaID\n"
        "        return total\n"
    )
    modules = [
        _module(_SPECS, "pandaserver/taskbuffer/Specs.py"),
        _module(source, "pandaserver/dataservice/setupper_plugin_base.py"),
    ]
    attributor = attribution.SpecAttributor(
        progress.spec_attributes(modules), attribution.class_bases(modules)
    )
    attributor.learn_element_types(modules)
    attributor.learn_self_attributes(modules)
    scope = [
        (node, "Setupper")
        for node in ast.walk(modules[1].tree)
        if isinstance(node, ast.FunctionDef)
    ]

    readings = attribution.annotation_readings(
        modules, attributor, set().union(*progress.spec_attributes(modules).values())
    )

    # The container is opened here -- the row is marked unaudited for having no
    # write to compare, not skipped for going unnoticed, which is how the same
    # test passed for the wrong reason when the loop was a comprehension.  It
    # stays on the list because its other reading, what the code puts in, is
    # what ``container-annotations-agree`` compares.
    assert attribution._container_is_opened("jobs", scope)
    assert [(r.container, r.audited) for r in readings] == [("jobs", False)]
    assert gates.annotations_are_read(
        MapFragment(
            map_id=MAP_ID,
            derived_from=VERSION,
            annotation_readings=[
                AnnotationAudit(
                    where=r.where,
                    kind=r.kind,
                    container=r.container,
                    stated=r.stated,
                    put_in=sorted(r.put_in),
                    read=r.read,
                    audited=r.audited,
                    unsettled=sorted(r.unsettled),
                )
                for r in readings
            ],
        )
    ).checked == 0


def test_an_annotation_leaving_a_write_of_its_own_class_unsettled_is_a_finding():
    """The narrowed gate still catches what it was built for.

    ``tasks: List[JediTaskSpec]`` changes no write here -- ``oldStatus`` has
    one declaring class and resolves without it -- but a ``status`` write in
    the same scope is unresolved, and ``JediTaskSpec`` declares ``status``.
    That is what a missing read path looks like from the outside, so the row
    keeps its failure while the merely redundant ones stop failing.
    """
    source = (
        "class Refiner:\n"
        "    def refine(self, tasks: List[JediTaskSpec]):\n"
        "        for task in tasks:\n"
        "            task.oldStatus = None\n"
        "        spec = self.fetch()\n"
        "        spec.status = 'ready'\n"
    )
    modules = [
        _module(_SPECS, "pandaserver/taskbuffer/Specs.py"),
        _module(source, "pandajedi/jedirefine/TaskRefinerBase.py"),
    ]
    attributor = attribution.SpecAttributor(
        progress.spec_attributes(modules), attribution.class_bases(modules)
    )
    attributor.learn_element_types(modules)
    attributor.learn_self_attributes(modules)

    readings = attribution.annotation_readings(
        modules, attributor, set().union(*progress.spec_attributes(modules).values())
    )

    assert [(r.container, r.read, sorted(r.unsettled)) for r in readings] == [
        ("tasks", False, ["6:status"])
    ]


def test_an_annotation_naming_two_spec_classes_is_not_a_finding():
    """``fileSpec: FileSpec | JediFileSpec`` -- correct, and deliberately open.

    ``SiteCandidate.getFileLocality`` says so in a comment of its own: the
    callers pass a JEDI file spec from a dataset and a job file spec built from
    one, so both have to be accepted.  Answering either class would be a guess,
    so the reader is right to say nothing -- and a gate that calls a correct
    annotation a defect is the failure this repo has treated as the most
    expensive one there is.
    """
    source = (
        "class SiteCandidate:\n"
        "    def getFileLocality(self, fileSpec: FileSpec | JediFileSpec):\n"
        "        return fileSpec.status\n"
    )
    modules = [
        _module(_SPECS, "pandaserver/taskbuffer/Specs.py"),
        _module(source, "pandajedi/jedicore/SiteCandidate.py"),
    ]
    attributor = attribution.SpecAttributor(
        progress.spec_attributes(modules), attribution.class_bases(modules)
    )

    forms = attribution.spec_annotation_forms(modules, attributor)

    assert forms == {
        "pandajedi/jedicore/SiteCandidate.py:2": "ambiguous: names FileSpec and JediFileSpec"
    }
    assert gates.annotation_forms_are_understood(
        MapFragment(map_id=MAP_ID, derived_from=VERSION, spec_annotation_forms=forms)
    ).passed


def test_an_annotation_whose_spec_sits_under_another_container_is_not_a_finding():
    """``dict[str, dict[str, list[WorkQueue]]]`` -- legible, and correctly silent.

    One read peels one level, so ``d[k]`` here is a mapping rather than a
    ``WorkQueue``, and the reader answering the class at the bottom would name
    something no single read produces.  The corpus performs no read deep enough
    to reach it -- no write in it goes through a chained subscript -- so this is
    a form understood rather than a form missed, and if a deep read ever
    appears it surfaces as an unresolved write rather than a wrong one.
    """
    source = (
        "class WorkQueueMapper:\n"
        "    def __init__(self):\n"
        "        self.queue_map: dict[str, dict[str, list[JobSpec]]] = {}\n"
    )
    modules = [
        _module(_SPECS, "pandaserver/taskbuffer/Specs.py"),
        _module(source, "pandaserver/taskbuffer/WorkQueueMapper.py"),
    ]
    attributor = attribution.SpecAttributor(
        progress.spec_attributes(modules), attribution.class_bases(modules)
    )

    forms = attribution.spec_annotation_forms(modules, attributor)

    assert forms == {
        "pandaserver/taskbuffer/WorkQueueMapper.py:3": "ambiguous: holds containers of JobSpec"
    }
    assert gates.annotation_forms_are_understood(
        MapFragment(map_id=MAP_ID, derived_from=VERSION, spec_annotation_forms=forms)
    ).passed


def test_a_form_the_reader_cannot_read_still_fails_beside_the_ambiguous_ones():
    """The gate keeps the only failure it was built for.

    An explained silence and an unexplained one look the same from the outside
    -- both answer no class -- so widening the vocabulary is only safe while
    the unexplained kind still fails.
    """
    fragment = MapFragment(
        map_id=MAP_ID,
        derived_from=VERSION,
        spec_annotation_forms={
            "setupper_plugin_base.py:13": "JobSpec",
            "SiteCandidate.py:65": "ambiguous: names FileSpec and JediFileSpec",
            "adder_gen.py:42": "",
        },
    )

    result = gates.annotation_forms_are_understood(fragment)

    assert not result.passed
    assert result.checked == 3
    assert result.failures == [
        "adder_gen.py:42 names a declared spec class and the extraction read none"
    ]


def test_the_legibility_census_leaves_out_what_the_reader_never_reads():
    """Two exclusions, both measured, both about not inventing a finding.

    A plain ``->`` return annotation is not read on purpose -- 29 of the
    corpus's 493 name a declared spec, too few for the mechanism -- so listing
    one would report a reading that was never attempted, and the heterogeneous
    ``tuple[Spec | None, str | None]`` returns are exactly the shape that would
    misreport.  Inside a ``Callable`` the spec names describe what the callable
    takes, not the class of the annotated object.
    """
    source = (
        "class Adder:\n"
        "    def find(self, key) -> tuple['JediFileSpec', str]:\n"
        "        ...\n"
        "\n"
        "    def register(self, done: Callable[[JediFileSpec], None]):\n"
        "        ...\n"
        "\n"
        "    def update(self, spec: JediFileSpec):\n"
        "        ...\n"
    )
    modules = [
        _module(_SPECS, "pandaserver/taskbuffer/Specs.py"),
        _module(source, "pandaserver/dataservice/adder_gen.py"),
    ]
    attributor = attribution.SpecAttributor(
        progress.spec_attributes(modules), attribution.class_bases(modules)
    )

    forms = attribution.spec_annotation_forms(modules, attributor)

    assert forms == {"pandaserver/dataservice/adder_gen.py:8": "JediFileSpec"}


def test_a_tuple_is_a_record_and_states_no_element_type():
    """The gate reported one of these and the other three were worse.

    ``failedRet`` is a three-tuple: it is not a ``JediFileSpec`` and it does
    not hold ``JediFileSpec``\\ s, but the last-argument rule said it was one.
    The unread annotation was the honest half; the two the reader *had* read
    were wrong, which no gate would have said.
    """
    modules = [_module(_SPECS, "pandaserver/taskbuffer/Specs.py")]
    attributor = attribution.SpecAttributor(
        progress.spec_attributes(modules), attribution.class_bases(modules)
    )

    def element_of(text: str):
        return attributor.element_annotation(ast.parse(text, mode="eval").body)

    assert element_of("tuple[bool, JediDatasetSpec | None, JediFileSpec | None]") is None
    assert element_of("dict[int, list[tuple[JediFileSpec, str]]]") is None
    # A genuine container still reads, so this refuses a shape rather than a name.
    assert element_of("list[JediFileSpec]") == "JediFileSpec"


def test_a_yield_annotation_is_audited_the_same_way_as_a_container_one():
    """A context manager's yield type is audited the same way, and has to be.

    It is the *only* mechanical check available for it.  Two of the three
    workflow locks have no second reading -- the attributes touched on the
    locked spec are declared by every workflow spec -- so the agreement gate
    corroborates one annotation and is silent about the other two.  What is
    left is the ablation: does removing it change a write.

    ``request_lock`` is the case that decided *not* to ask PanDA for one: the
    seven ``with`` sites write nothing on the locked spec, so the annotation
    would resolve nothing.  That is still worth saying and still reported --
    it is how the harvest of a proposed annotation gets measured -- but it is
    not a defect in the map, and after an annotation campaign filled the tree
    with annotations the map does not need, failing on it would have made the
    gate permanently red for the normal case.
    """
    fragment = _audited(
        AnnotationAudit(
            where="DataCarousel.py:691",
            kind="yield",
            container="request_lock()",
            stated="DataCarouselRequestSpec",
            put_in=[],
            read=False,
            unsettled=[],
        ),
        AnnotationAudit(
            where="workflow_core.py:266",
            kind="yield",
            container="workflow_lock()",
            stated="WorkflowSpec",
            put_in=[],
            read=False,
            unsettled=["445:status"],
        ),
    )

    result = gates.annotations_are_read(fragment)

    assert not result.passed
    assert result.failures == [
        "workflow_core.py:266 states WorkflowSpec for workflow_lock() "
        "and no write in scope resolves differently without it"
    ]
    assert "1 more state a class the map reaches another way" in (result.note or "")


def test_the_agreement_gate_counts_only_what_it_can_compare():
    """A yield type has nothing stored in it, so it is not this gate's business.

    Counting it as ``checked`` would report a comparison that never happened --
    the same thing the production report was rebuilt to stop doing when a
    truncated sample was shown as a completed check.
    """
    fragment = _audited(
        AnnotationAudit(
            where="closer.py:49",
            container="dataset_map",
            stated="DatasetSpec",
            put_in=["DatasetSpec"],
            read=True,
        ),
        AnnotationAudit(
            where="workflow_core.py:266",
            kind="yield",
            container="workflow_lock()",
            stated="WorkflowSpec",
            put_in=[],
            read=True,
        ),
    )

    assert gates.container_annotations_agree(fragment).checked == 1
    assert gates.annotations_are_read(fragment).checked == 2


def test_a_base_class_annotation_is_read_in_the_subclass():
    """PanDA annotates where a field is initialised, not where it is used.

    ``self.jobs: List[JobSpec]`` is stated in ``SetupperPluginBase`` and
    iterated in the dummy plugin that derives from it.  Scoping the lookup to
    the enclosing class left the annotation doing nothing -- and inheritance is
    already how ``self`` writes resolve, so following it reads a relationship
    the language guarantees rather than matching a name.
    """
    base = (
        "class Base:\n"
        "    def __init__(self):\n"
        "        self.jobs: List[JobSpec] = []\n"
    )
    derived = (
        "class Dummy(Base):\n"
        "    def run(self):\n"
        "        for job_spec in self.jobs:\n"
        "            for file_spec in job_spec.Files:\n"
        "                file_spec.status = 'failed'\n"
    )
    _subjects, junctions, _cov = _progress_multi(
        (_ADDERS, "pandaserver/taskbuffer/JobSpec.py"),
        (base, "pandaserver/dataservice/setupper_plugin_base.py"),
        (derived, "pandaserver/dataservice/setupper_dummy_plugin.py"),
    )
    written = [j for j in junctions if "setupper_dummy" in j.owner]

    # ``status`` is declared by three classes here, so nothing but the base
    # class's annotation can start the chain -- which is what makes this the
    # real shape rather than a write the declaring-class rule settles anyway.
    assert [(j.subject, j.attribution) for j in written] == [
        ("FileSpec.status", "container")
    ]


def test_a_container_held_by_a_loop_variable_is_read():
    """Nested loops -- the outer one already answered what the inner one asks.

    ``class_of`` did not consult the loop rule, so ``for job in jobs`` followed
    by ``for file in job.Files`` was unreadable even with ``jobs`` annotated:
    the two entry points into attribution disagreed about which readings exist.
    """
    source = (
        "class Setupper:\n"
        "    def update_failed_jobs(self, jobs: List[JobSpec]):\n"
        "        for job in jobs:\n"
        "            for file in job.Files:\n"
        "                file.status = 'failed'\n"
    )
    _subjects, junctions, _cov = _progress_multi(
        (_ADDERS, "pandaserver/taskbuffer/JobSpec.py"),
        (source, "pandaserver/dataservice/setupper_plugin_base.py"),
    )
    written = [j for j in junctions if "setupper_plugin_base" in j.owner]

    assert [(j.subject, j.attribution) for j in written] == [
        ("FileSpec.status", "container")
    ]


def test_a_context_manager_states_the_class_its_with_target_holds():
    """``with self.workflow_lock(id) as spec`` -- the last unresolved shape.

    The write is four expressions from anything that names a class: the lock
    method yields the spec, the ``with`` binds it, and the attributes touched
    (``status``, ``end_time``) are declared by every workflow spec, so
    structural inference cannot separate them either.  The declared yield type
    is the only reading, and PanDA's docstring already stated it.
    """
    source = (
        "class WorkflowCore:\n"
        "    @contextmanager\n"
        "    def workflow_lock(self, workflow_id: int) -> Iterator[JediTaskSpec | None]:\n"
        "        yield self.tbif.get_workflow(workflow_id)\n"
        "\n"
        "    def cancel_workflow(self, workflow_id: int) -> bool:\n"
        "        with self.workflow_lock(workflow_id) as workflow_spec:\n"
        "            workflow_spec.status = 'cancelled'\n"
    )
    _subjects, junctions, _cov = _progress(source, "pandaserver/workflow/workflow_core.py")

    assert [(j.subject, j.attribution) for j in junctions] == [
        ("JediTaskSpec.status", "certain")
    ]


def test_the_yield_type_is_the_first_argument_of_a_generator():
    """``Generator[Y, S, R]`` puts the yield type first, unlike a mapping.

    The container reading takes the *last* argument so that ``Dict[K, V]``
    gives its values.  Reusing it here would read ``Generator[WorkflowSpec,
    None, None]`` as ``None`` -- a silent no-op on the fuller spelling of the
    same annotation.
    """
    source = (
        "class WorkflowCore:\n"
        "    @contextmanager\n"
        "    def data_lock(self, data_id) -> Generator[JediFileSpec, None, None]:\n"
        "        yield self.tbif.get_data(data_id)\n"
        "\n"
        "    def cancel_data(self, data_id):\n"
        "        with self.data_lock(data_id) as data_spec:\n"
        "            data_spec.status = 'cancelled'\n"
    )
    _subjects, junctions, _cov = _progress(source, "pandaserver/workflow/workflow_core.py")

    assert [(j.subject, j.attribution) for j in junctions] == [
        ("JediFileSpec.status", "certain")
    ]


def test_an_unannotated_context_manager_states_nothing():
    """The shape alone is not the fact -- the annotation is.

    Every one of PanDA's eight context managers looked like this before the
    yield type was asked for, and reading the shape as evidence would have
    invented an answer for all of them.
    """
    source = (
        "class WorkflowCore:\n"
        "    @contextmanager\n"
        "    def workflow_lock(self, workflow_id):\n"
        "        yield self.tbif.get_workflow(workflow_id)\n"
        "\n"
        "    def cancel_workflow(self, workflow_id):\n"
        "        with self.workflow_lock(workflow_id) as workflow_spec:\n"
        "            workflow_spec.status = 'cancelled'\n"
    )
    _subjects, junctions, _cov = _progress(source, "pandaserver/workflow/workflow_core.py")

    assert [(j.subject, j.attribution) for j in junctions] == [("?.status", "unresolved")]


def test_an_iterator_that_is_not_a_context_manager_types_nothing():
    """``as`` binds ``__enter__()``'s result, which a bare iterator has not got.

    So ``-> Iterator[X]`` without ``@contextmanager`` on something used in a
    ``with`` describes code that cannot run, and answering ``X`` there would be
    a confident wrong answer about broken code -- worse than the unresolved
    write it replaces.  The decorator is the declaration that makes the yield
    type what ``as`` receives.
    """
    source = (
        "class WorkflowCore:\n"
        "    def workflow_lock(self, workflow_id) -> Iterator[JediTaskSpec]:\n"
        "        yield self.tbif.get_workflow(workflow_id)\n"
        "\n"
        "    def cancel_workflow(self, workflow_id):\n"
        "        with self.workflow_lock(workflow_id) as workflow_spec:\n"
        "            workflow_spec.status = 'cancelled'\n"
    )
    _subjects, junctions, _cov = _progress(source, "pandaserver/workflow/workflow_core.py")

    assert [(j.subject, j.attribution) for j in junctions] == [("?.status", "unresolved")]


def test_a_receiver_built_from_a_runtime_registry_is_not_a_spec_write():
    """``impl = srcImplMap[subType](*args)`` -- the class is chosen at run time.

    ``FactoryBase`` instantiates the knight plugins named in a config string
    and stamps two fields on each, and those two are ``vo`` and
    ``prodSourceLabel``, which among the declaring classes only ``JediTaskSpec``
    has both of.  So structural inference answered ``JediTaskSpec`` -- for a
    plugin object that is not a spec at all -- and the map ended up naming a
    plugin factory as one of the four places a task's ``vo`` is decided, on a
    promoted subject, which is half of its writers.

    This is the trap the design named at the start, arriving by a different
    road: ``WatchDog.vo = "atlas"`` was refused because ``self`` resolves
    through the class, and here there is no class to resolve through.
    """
    source = (
        "class FactoryBase:\n"
        "    def instantiateImpl(self, vo, sourceLabel, subType, *args):\n"
        "        impl = self.classMap[vo][sourceLabel][subType](*args)\n"
        "        use(impl.proc_status)\n"
        "        impl.status = 'ready'\n"
    )
    _subjects, junctions, _cov = _progress(source, "pandajedi/jedicore/FactoryBase.py")

    assert junctions == []


def test_a_receiver_built_from_a_class_looked_up_by_name_is_not_a_spec_write():
    """The same factory, one line earlier: ``cls = getattr(mod, className)``.

    ``initializeMods`` imports the module named in the config and pulls the
    class out of it by name, so the class is a string until run time.  Two
    spellings of one fact, and reading only the subscript would have left half
    of the writes standing.
    """
    source = (
        "class FactoryBase:\n"
        "    def initializeMods(self, *args):\n"
        "        mod = __import__(moduleName)\n"
        "        cls = getattr(mod, className)\n"
        "        impl = cls(*args)\n"
        "        use(impl.proc_status)\n"
        "        impl.status = 'ready'\n"
    )
    _subjects, junctions, _cov = _progress(source, "pandajedi/jedicore/FactoryBase.py")

    assert junctions == []


def test_a_receiver_from_a_named_method_still_resolves_structurally():
    """The guard is about a class chosen at run time, not about calls.

    Every knight takes its spec from the task buffer, and where the facade has
    no return annotation structural inference is what settles those writes --
    three of them in the corpus.  A rule that refused any call-assigned
    receiver would take those away to remove four wrong ones.
    """
    source = (
        "class ContentsFeeder:\n"
        "    def feed(self, jediTaskID):\n"
        "        spec = self.taskBufferIF.peekFileSpec(jediTaskID)\n"
        "        use(spec.proc_status)\n"
        "        spec.status = 'ready'\n"
    )
    _subjects, junctions, _cov = _progress(source, "pandajedi/jediorder/ContentsFeeder.py")

    assert [(j.subject, j.attribution) for j in junctions] == [
        ("JediFileSpec.status", "structural")
    ]


_FACADE = """
class JediTaskBuffer(object):
    def getTaskWithID_JEDI(self, jediTaskID: int) -> tuple[bool, JediTaskSpec | None]:
        return True, self.proxy.peek(jediTaskID)

    def getTasksToReassign_JEDI(self, vo: str | None = None) -> list[JediTaskSpec]:
        return self.proxy.reassign(vo)

    def queryDatasetWithMap(self, criteria: dict) -> FileSpec | None:
        return self.proxy.query(criteria)
"""


def test_a_stated_return_type_settles_the_local_it_is_assigned_to():
    """The facade states what it hands back, and the caller writes to it.

    ``queryDatasetWithMap`` is the very call the design once cited as needing
    interprocedural analysis, and decided to ask PanDA to annotate instead.
    PanDA annotated it, so the reading is of a statement rather than an
    inference -- ``status`` alone is declared by three classes here, which is
    what structural inference is left with when the return type goes unread.
    """
    source = (
        "def main(taskBuffer):\n"
        "    tmpDS = taskBuffer.queryDatasetWithMap({'name': 'sub'})\n"
        "    tmpDS.status = 'deleting'\n"
    )
    _subjects, junctions, _cov = _progress_multi(
        (_FACADE, "pandajedi/jedicore/JediTaskBuffer.py"),
        (source, "pandaserver/daemons/scripts/datasetManager.py"),
    )
    written = [j for j in junctions if "datasetManager" in j.owner]

    assert [(j.subject, j.attribution) for j in written] == [("FileSpec.status", "certain")]


def test_a_stated_return_type_is_read_at_the_position_the_assignment_names():
    """``tmpStat, taskSpec = f()`` against ``-> tuple[bool, JediTaskSpec | None]``.

    A tuple is a record, so the last-argument rule that reads containers does
    not apply to it.  The position is not tracked through the program either:
    it is read off the assignment target standing right there, which is the
    only place it is stated.
    """
    source = (
        "class ContentsFeeder:\n"
        "    def feed_contents_to_tasks(self, jediTaskID):\n"
        "        tmpStat, taskSpec = self.taskBufferIF.getTaskWithID_JEDI(jediTaskID)\n"
        "        taskSpec.status = 'ready'\n"
    )
    _subjects, junctions, _cov = _progress_multi(
        (_FACADE, "pandajedi/jedicore/JediTaskBuffer.py"),
        (source, "pandajedi/jediorder/ContentsFeeder.py"),
    )
    written = [j for j in junctions if "ContentsFeeder" in j.owner]

    assert [(j.subject, j.attribution) for j in written] == [
        ("JediTaskSpec.status", "certain")
    ]


def test_the_other_position_of_the_same_return_states_nothing():
    """Position 0 of that tuple is a ``bool``, and answering anyway is a guess.

    The point of reading the assignment target is that it says *which* member
    was taken; a reader that scanned the record for a spec would answer the
    same class whichever name the write is on.
    """
    source = (
        "class ContentsFeeder:\n"
        "    def feed_contents_to_tasks(self, jediTaskID):\n"
        "        taskSpec, tmpStat = self.taskBufferIF.getTaskWithID_JEDI(jediTaskID)\n"
        "        taskSpec.status = 'ready'\n"
    )
    _subjects, junctions, _cov = _progress_multi(
        (_FACADE, "pandajedi/jedicore/JediTaskBuffer.py"),
        (source, "pandajedi/jediorder/ContentsFeeder.py"),
    )
    written = [j for j in junctions if "ContentsFeeder" in j.owner]

    assert [(j.subject, j.attribution) for j in written] == [("?.status", "unresolved")]


def test_a_loop_over_what_a_stated_return_hands_back_is_read():
    """``taskList = f()`` then ``for taskSpec in taskList``, with ``-> list[X]``.

    The element type is stated one assignment away from the loop, which is how
    every watchdog in the corpus reads its work list.
    """
    source = (
        "class AtlasProdWatchDog:\n"
        "    def doActionForReassign(self, gTmpLog):\n"
        "        taskList = self.taskBufferIF.getTasksToReassign_JEDI(self.vo)\n"
        "        for taskSpec in taskList:\n"
        "            taskSpec.status = 'assigning'\n"
    )
    _subjects, junctions, _cov = _progress_multi(
        (_FACADE, "pandajedi/jedicore/JediTaskBuffer.py"),
        (source, "pandajedi/jedidog/AtlasProdWatchDog.py"),
    )
    written = [j for j in junctions if "AtlasProdWatchDog" in j.owner]

    assert [(j.subject, j.attribution) for j in written] == [
        ("JediTaskSpec.status", "container")
    ]


def test_return_types_that_disagree_state_nothing():
    """The facade and the proxy declare one method twice, and must agree.

    They do, for all seventy-five names the corpus states a spec return for --
    and the one name whose definitions disagree is real: ``doGenerate`` returns
    a bare status code in two plugins and a tuple in a third.  Reading either
    would be picking one at random.
    """
    other = (
        "class JobGeneratorProxy(object):\n"
        "    def getTaskWithID_JEDI(self, jediTaskID: int) -> FileSpec:\n"
        "        return self.proxy.peek(jediTaskID)\n"
    )
    source = (
        "class ContentsFeeder:\n"
        "    def feed_contents_to_tasks(self, jediTaskID):\n"
        "        tmpStat, taskSpec = self.taskBufferIF.getTaskWithID_JEDI(jediTaskID)\n"
        "        taskSpec.status = 'ready'\n"
    )
    _subjects, junctions, _cov = _progress_multi(
        (_FACADE, "pandajedi/jedicore/JediTaskBuffer.py"),
        (other, "pandaserver/taskbuffer/TaskBuffer.py"),
        (source, "pandajedi/jediorder/ContentsFeeder.py"),
    )
    written = [j for j in junctions if "ContentsFeeder" in j.owner]

    assert [(j.subject, j.attribution) for j in written] == [("?.status", "unresolved")]


def test_an_unannotated_definition_of_the_same_name_states_nothing():
    """One definition annotated is not the corpus stating a return type.

    PanDA defines its JEDI methods twice -- the facade and the proxy module --
    and the campaign annotated both.  Trusting a name where one definition is
    silent would be reading an annotation about the other one.
    """
    other = (
        "class JobGeneratorProxy(object):\n"
        "    def getTaskWithID_JEDI(self, jediTaskID):\n"
        "        return True, self.proxy.peek(jediTaskID)\n"
    )
    source = (
        "class ContentsFeeder:\n"
        "    def feed_contents_to_tasks(self, jediTaskID):\n"
        "        tmpStat, taskSpec = self.taskBufferIF.getTaskWithID_JEDI(jediTaskID)\n"
        "        taskSpec.status = 'ready'\n"
    )
    _subjects, junctions, _cov = _progress_multi(
        (_FACADE, "pandajedi/jedicore/JediTaskBuffer.py"),
        (other, "pandaserver/taskbuffer/TaskBuffer.py"),
        (source, "pandajedi/jediorder/ContentsFeeder.py"),
    )
    written = [j for j in junctions if "ContentsFeeder" in j.owner]

    assert [(j.subject, j.attribution) for j in written] == [("?.status", "unresolved")]


def test_the_typed_declaration_form_yields_the_same_column_names():
    """Both forms say the same thing, so both are read the same way."""
    source = (
        "class WFDataSpec:\n"
        "    attributes_with_types = (\n"
        "        AttributeWithType('data_id', int),\n"
        "        AttributeWithType('status', str),\n"
        "    )\n"
        "    attributes = tuple([a.attribute for a in attributes_with_types])\n"
    )

    declared = progress.spec_attributes([_module(source)])

    assert declared == {"WFDataSpec": {"data_id", "status"}}


def test_a_local_bound_from_an_annotated_field_keeps_its_class():
    """One hop from the field to the local, and the annotation stops being seen.

    ``finisher`` guards ``self.dataset`` for None once at the top and works
    through a local from there, which is the right way to write it -- and it
    put the sole producer of ``DatasetSpec.status = 'cleanup'`` beyond the
    reader, filed under a placeholder subject while eleven other writers of
    that subject were on the map.  The reading is the same shape as the one
    already followed for a local bound from an annotated *container*: a
    reaching definition in the same function, and a declaration one hop away.
    """
    source = (
        "class Finisher:\n"
        "    def __init__(self, dataset):\n"
        "        self.dataset: JediFileSpec | None = dataset\n"
        "\n"
        "    def run(self):\n"
        "        dataset = self.dataset\n"
        "        dataset.status = 'cleanup'\n"
    )
    _subjects, junctions, _cov = _progress(source, "pandaserver/dataservice/finisher.py")

    assert [(j.subject, j.attribution) for j in junctions] == [
        ("JediFileSpec.status", "certain")
    ]


def test_a_local_bound_from_two_different_fields_settles_nothing():
    """The hop is only worth taking when it lands in one place.

    Returning the first assignment found would make the answer depend on walk
    order, which is not a reading of anything.
    """
    source = (
        "class Finisher:\n"
        "    def __init__(self, a, b):\n"
        "        self.a: JediFileSpec | None = a\n"
        "        self.b: FileSpec | None = b\n"
        "\n"
        "    def run(self, flag):\n"
        "        spec = self.a\n"
        "        if flag:\n"
        "            spec = self.b\n"
        "        spec.status = 'cleanup'\n"
    )
    _subjects, junctions, _cov = _progress(source, "pandaserver/dataservice/finisher.py")

    assert [(j.subject, j.attribution) for j in junctions] == [("?.status", "unresolved")]


def test_a_quoted_annotation_states_what_an_unquoted_one_states():
    """Quoting is a runtime concern, not a change of statement.

    panda-server evaluates annotations at import time -- no module uses
    ``from __future__ import annotations`` -- so ``if TYPE_CHECKING`` plus a
    quoted name is the only way to name a type that cannot be imported at
    runtime, and ``base_module`` says why in a comment: importing
    ``WrappedCursor`` there would close an import cycle.  Discarding the quoted
    form makes runtime safety and map visibility exclusive.
    """
    source = (
        "class Setupper:\n"
        "    def update(self, file_spec: 'JediFileSpec'):\n"
        "        file_spec.status = 'ready'\n"
    )
    _subjects, junctions, _cov = _progress(source, "pandaserver/dataservice/setupper.py")

    # ``status`` is declared by three classes here, so nothing but the
    # annotation can settle the write -- which is what makes this the real
    # shape rather than one the declaring-class rule answers anyway.
    assert [(j.subject, j.attribution) for j in junctions] == [
        ("JediFileSpec.status", "certain")
    ]


def test_a_quoted_element_type_states_what_an_unquoted_one_states():
    """The inner quote is reached through the reading that is already there.

    ``JediTaskSpec`` writes ``datasetSpecList: list["JediDatasetSpec"]``, so the
    element reading has to see through a quote one level down rather than only
    at the top of an annotation.
    """
    source = (
        "class Refiner:\n"
        "    def refine(self, specs: list['JediFileSpec']):\n"
        "        for spec in specs:\n"
        "            spec.status = 'ready'\n"
    )
    _subjects, junctions, _cov = _progress(source, "pandajedi/jedirefine/TaskRefinerBase.py")

    assert [(j.subject, j.attribution) for j in junctions] == [
        ("JediFileSpec.status", "container")
    ]


def test_an_annotation_that_is_not_a_type_expression_reads_as_nothing():
    """A string in an annotation need not parse, so reading one cannot raise.

    Prose in an annotation slot is not a type, and the reader has to say
    "nothing" for it the same way it does for ``int`` -- a traceback here would
    take down the whole build over one malformed annotation.
    """
    attributor = attribution.SpecAttributor({"JobSpec": {"jobStatus"}}, {})

    for text in ("'not a type'", "'JobSpec('", "''"):
        node = ast.parse(text, mode="eval").body
        assert attributor.stated_annotation(node) is None
        assert attributor.element_annotation(node) is None


def test_an_annotated_declaration_is_read_like_a_plain_one():
    """A type on the declaration does not change what it declares.

    ``pandacommon`` already writes ``attributes: tuple[str, ...] = ()`` and a
    mypy campaign is annotating panda-server, so this form is arriving.  The
    statement filter accepted only ``ast.Assign``, which meant an annotated
    declaration was not merely misread -- it was never looked at, so the class
    contributed no entry at all and the gate that exists to catch exactly this
    had nothing to count.
    """
    source = "class JediDatasetSpec:\n    _attributes: tuple[str, ...] = ('datasetID', 'status')\n"

    declared = progress.spec_attributes([_module(source)])

    assert declared == {"JediDatasetSpec": {"datasetID", "status"}}


def test_a_column_named_attributes_is_not_a_declaration():
    """The discriminator is whether the statement assigns the list.

    ``JediDatasetSpec`` really does have a column called ``attributes`` -- it
    is in the declared tuple -- and the type block near the top of the class
    states its type.  That statement declares nothing about the class's
    columns, and reading it as a declaration would put a correctly-read class
    on the unread list.
    """
    source = (
        "class JediDatasetSpec:\n"
        "    _attributes = ('datasetID', 'status', 'attributes')\n"
        "    attributes: str | None\n"
    )

    declarations = progress.spec_declarations([_module(source)])

    assert [(cls, sorted(names)) for cls, _file, names in declarations] == [
        ("JediDatasetSpec", ["attributes", "datasetID", "status"])
    ]


def test_an_annotated_declaration_that_yields_nothing_is_counted():
    """The blind spot proper: found and empty has to beat never looked at.

    A derived declaration legitimately reads as empty, and the per-class
    aggregation absorbs that when the class also states its columns outright.
    What must not happen is the class dropping out of the count entirely --
    that is the state in which the reader can lose every column of a class and
    the gate still passes.
    """
    source = (
        "class WFDataSpec:\n"
        "    attributes: tuple[str, ...] = tuple(a.attribute for a in _pairs)\n"
    )

    declarations = progress.spec_declarations([_module(source)])

    assert [(cls, names) for cls, _file, names in declarations] == [("WFDataSpec", set())]


def test_two_nodes_sharing_a_signature_are_reported_rather_than_one_being_lost():
    """The signature is the merge key, so a shared one is a node about to vanish.

    Found by storing the map for the first time: the build said 109 filter
    stages and Neo4j held 108.  ``AtlasProdJobBroker`` runs "temporary problem
    check" at two points guarded by ``hintForTB``, and keyed on the label alone
    they were one node -- so the map ended up claiming that cut is
    unconditional.  Neither side could see it; only the two counts together.
    """
    fragment = _fragment_with(("JediTaskSpec", "status", [("ready", 1)]))
    twin = fragment.junctions[0].model_copy(deep=True)
    twin.branches = [Branch(outcome="broken")]
    fragment.junctions.append(twin)

    result = gates.map_identities_are_distinct(fragment)

    assert not result.passed
    assert result.failures == [
        f"2 junction(s) share the signature {fragment.junctions[0].name}"
    ]


def test_a_map_whose_signatures_are_all_distinct_says_nothing():
    fragment = _fragment_with(("JediTaskSpec", "status", [("ready", 1)]))

    assert gates.map_identities_are_distinct(fragment).passed


def test_an_arm_the_map_offers_with_no_line_is_reported_rather_than_just_counted():
    """A positionless arm is counted, stored, offered and compared regardless.

    Which is why it needed a gate of its own: the branch survives every check
    the map already had, because none of them reads a line.  What does read one
    is the derivation -- ``reading.lines`` is filled from ``branch.line`` and
    nowhere else, and ``attach_traces`` walks nothing for a reading that ends
    up with none.  Measured against the corpus when this was written: 56 arms
    over 15 junctions, and 18 readings that came back with no steps and no
    skeleton, all of them about a task's status.
    """
    fragment = _fragment_with(("JediTaskSpec", "status", [("ready", 1), ("broken", 1)]))
    junction = fragment.junctions[0]
    junction.branches[0].line = 812

    result = gates.branch_positions_are_known(fragment)

    assert not result.passed
    assert result.checked == 2
    assert result.failures == [f"1/2 arm(s) of {junction.name} say no line -- broken"]


def test_a_map_whose_every_arm_carries_a_line_says_nothing():
    fragment = _fragment_with(("JediTaskSpec", "status", [("ready", 1), ("broken", 1)]))
    for offset, branch in enumerate(fragment.junctions[0].branches):
        branch.line = 812 + offset

    assert gates.branch_positions_are_known(fragment).passed


def test_a_subject_nothing_writes_does_not_belong_in_the_map():
    fragment = _graph(
        subjects=[
            SubjectNode(
                map_id=MAP_ID,
                derived_from=VERSION,
                name="JediTaskSpec.status",
                spec_class="JediTaskSpec",
                attribute="status",
            )
        ],
        junctions=[],
    )

    assert gates.map_references_resolve(fragment).failures == [
        "JediTaskSpec.status is a subject nothing writes"
    ]


def test_promotion_does_not_follow_a_passthrough_to_a_node_that_is_absent():
    """``currentPriority`` is carried from ``taskPriority``, which nothing writes.

    Promoting the name anyway put a criterion on a node that did not exist --
    caught by the reference gate, which is what a gate over the assembly is
    for.
    """
    fragment = _fragment_with(("JediTaskSpec", "currentPriority", [("x", 2)]))
    fragment.junctions[0].branches.append(
        Branch(outcome="passthrough(JediTaskSpec.taskPriority)", tier=2)
    )
    closed = promotion.close_over_passthrough(
        fragment, {"JediTaskSpec.currentPriority": ["5:gates-a-filter-stage"]}
    )

    assert "JediTaskSpec.taskPriority" not in closed
    assert gates.carried_from_outside(fragment) == [
        ("JediTaskSpec.currentPriority", "JediTaskSpec.taskPriority")
    ]


def test_a_value_nothing_selects_on_is_reported_not_failed():
    """A terminal status is supposed to be a sink and nothing declares which."""
    fragment = _fragment_with(("JediTaskSpec", "status", [("ready", 1), ("broken", 1)]))
    fragment.subjects[0].selected_values = ["ready"]

    assert gates.unreachable_values(fragment) == [("JediTaskSpec.status", ["broken"])]


def test_a_subject_with_no_read_side_is_not_reported_as_all_sinks():
    """There the map has no question to answer, which is not an answer of no."""
    fragment = _fragment_with(("JediTaskSpec", "status", [("ready", 1)]))

    assert gates.unreachable_values(fragment) == []


def test_two_queries_in_one_function_are_two_read_sites():
    """The reason the statement is the unit and the function is not.

    ``copyArchive.py::main`` asks for ``jobStatus='holding'`` from three
    statements with three different time limits, and named by the function
    those are one answer -- which is what the read side held until the
    statement became a node.  ``read_side`` already told them apart, because
    ``(sql, window)`` is what it deduplicates on; the position was dropped at
    ``record``.
    """
    source = (
        "class TaskModule:\n"
        "    def sweep(self, timeLimit):\n"
        "        varMap = {}\n"
        "        varMap[':status'] = 'pending'\n"
        "        sqlA = f'SELECT jediTaskID FROM {schema}.JEDI_Tasks '\n"
        "        sqlA += 'WHERE status=:status AND modificationtime<:timeLimit '\n"
        "        self.cur.execute(sqlA + comment, varMap)\n"
        "        varMap2 = {}\n"
        "        varMap2[':status'] = 'finishing'\n"
        "        sqlB = f'SELECT jediTaskID FROM {schema}.JEDI_Tasks '\n"
        "        sqlB += 'WHERE status=:status '\n"
        "        self.cur.execute(sqlB + comment, varMap2)\n"
        # Below the selects so their lines stay put.  Here only to teach the
        # table its class: ``learn_table_classes`` reads writes, and a corpus
        # of pure queries leaves every subject spelled by its table.
        "    def touch(self, jediTaskID):\n"
        "        varMap3 = {}\n"
        "        varMap3[':status'] = 'ready'\n"
        # Two columns, because ``status`` alone is ambiguous -- ``FileSpec``,
        # ``DatasetSpec`` and ``JediFileSpec`` declare one too -- and an
        # ambiguous table is left unattributed, which spells the subject by its
        # table instead of its class.
        "        sqlU = f'UPDATE {schema}.JEDI_Tasks SET status=:status,oldStatus=NULL '\n"
        "        sqlU += 'WHERE jediTaskID=:jediTaskID '\n"
        "        self.cur.execute(sqlU + comment, varMap3)\n"
    )
    modules = [_module(_SPECS, "pandaserver/taskbuffer/Specs.py"), _module(source, "x.py")]
    attributor = attribution.SpecAttributor(
        progress.spec_attributes(modules), attribution.class_bases(modules)
    )
    attributor.learn_table_classes(modules)

    side = sqlwrite.read_side(modules, attributor)
    nodes = sqlwrite.read_site_nodes(side.statements, MAP_ID, VERSION)

    # One owner in ``selected_by`` -- unchanged, and deliberately so.
    assert side.values["JediTaskSpec.status"]["pending"].selected_by == {"x.py::sweep"}
    # Two statements, told apart by the line the query runs on.
    assert [(n.anchor.line_start, n.selects) for n in nodes] == [
        (7, ["JediTaskSpec.status=pending"]),
        (12, ["JediTaskSpec.status=finishing"]),
    ]
    # One row per statement, so the identities are distinct.
    assert len({n.name for n in nodes}) == 2
    assert [n.tables for n in nodes] == [["JEDI_Tasks"], ["JEDI_Tasks"]]
    # The ``execute`` form gets a window too, since P1-52.  It resolves its
    # binds by key and needs no scope for *that* -- which is why none was read
    # here before -- but where a walk should start is a different question,
    # and ``varMap2 = {}`` / ``varMap3 = {}`` answer it the same way the
    # forwarded form's do.
    assert [n.bind_window for n in nodes] == [[3, 7], [8, 12]]
    # And the stretch of the function each statement owns: the first runs out
    # where the second starts binding.
    assert [n.region for n in nodes] == [[3, 7], [8, 12]]
    assert [(n.anchor.line_start, n.anchor.line_end) for n in nodes] == [(7, 7), (12, 12)]


def test_a_value_a_statement_passes_over_is_an_exclude_not_a_select():
    """``status<>:s`` asks for every row but the value, so it is no reader of it.

    Filed as a select, the 36 excluding predicates of the corpus each named a
    function as picking up rows in the one state it skips.
    """
    source = (
        "class TaskModule:\n"
        "    def sweep(self):\n"
        "        varMap = {}\n"
        "        varMap[':status'] = 'running'\n"
        "        varMap[':old'] = 'done'\n"
        "        sqlA = f'SELECT jediTaskID FROM {schema}.JEDI_Tasks '\n"
        "        sqlA += \"WHERE status=:status AND oldStatus<>:old AND NOT status IN ('broken','failed') \"\n"
        "        self.cur.execute(sqlA + comment, varMap)\n"
        "    def touch(self, jediTaskID):\n"
        "        varMap3 = {}\n"
        "        varMap3[':status'] = 'ready'\n"
        "        sqlU = f'UPDATE {schema}.JEDI_Tasks SET status=:status,oldStatus=NULL '\n"
        "        sqlU += 'WHERE jediTaskID=:jediTaskID '\n"
        "        self.cur.execute(sqlU + comment, varMap3)\n"
    )
    modules = [_module(_SPECS, "pandaserver/taskbuffer/Specs.py"), _module(source, "x.py")]
    attributor = attribution.SpecAttributor(
        progress.spec_attributes(modules), attribution.class_bases(modules)
    )
    attributor.learn_table_classes(modules)

    side = sqlwrite.read_side(modules, attributor)
    (node,) = sqlwrite.read_site_nodes(side.statements, MAP_ID, VERSION)

    assert node.selects == ["JediTaskSpec.status=running"]
    assert node.excludes == [
        "JediTaskSpec.oldStatus=done",
        "JediTaskSpec.status=broken",
        "JediTaskSpec.status=failed",
    ]
    assert "done" not in side.values.get("JediTaskSpec.oldStatus", {})
    assert set(side.values["JediTaskSpec.status"]) == {"running"}


def test_a_statement_asking_by_a_key_alone_is_a_read_site():
    """``WHERE jediTaskID=:jediTaskID`` reads the task's rows though it names no value."""
    source = (
        "class TaskModule:\n"
        "    def tasks(self, jediTaskID):\n"
        "        varMap = {}\n"
        "        varMap[':jediTaskID'] = jediTaskID\n"
        "        sqlA = f'SELECT status FROM {schema}.JEDI_Tasks '\n"
        "        sqlA += 'WHERE jediTaskID=:jediTaskID '\n"
        "        self.cur.execute(sqlA + comment, varMap)\n"
        "    def touch(self, jediTaskID):\n"
        "        varMap3 = {}\n"
        "        varMap3[':status'] = 'ready'\n"
        "        sqlU = f'UPDATE {schema}.JEDI_Tasks SET status=:status,oldStatus=NULL '\n"
        "        sqlU += 'WHERE jediTaskID=:jediTaskID '\n"
        "        self.cur.execute(sqlU + comment, varMap3)\n"
    )
    modules = [_module(_SPECS, "pandaserver/taskbuffer/Specs.py"), _module(source, "x.py")]
    attributor = attribution.SpecAttributor(
        progress.spec_attributes(modules), attribution.class_bases(modules)
    )
    attributor.learn_table_classes(modules)

    side = sqlwrite.read_side(modules, attributor)
    nodes = sqlwrite.read_site_nodes(side.statements, MAP_ID, VERSION)

    # The read, not the update: an update by key is a write, and the map
    # already holds it as one.
    assert [(n.owner, n.selects, n.keys, n.tables) for n in nodes] == [
        ("x.py::tasks", [], ["JediTaskSpec.jediTaskID"], ["JEDI_Tasks"])
    ]


def test_binds_written_into_the_call_are_read_and_windowed_there():
    """``querySQLS(sql, {":status": "running"})`` fills its binds in the call itself."""
    source = (
        "class TaskModule:\n"
        "    def sweep(self, taskBuffer):\n"
        "        res = taskBuffer.querySQLS(\n"
        "            'SELECT jediTaskID FROM ATLAS_PANDA.JEDI_Tasks WHERE status=:status ',\n"
        "            {':status': 'running'},\n"
        "        )\n"
        "    def touch(self, jediTaskID):\n"
        "        varMap3 = {}\n"
        "        varMap3[':status'] = 'ready'\n"
        "        sqlU = f'UPDATE {schema}.JEDI_Tasks SET status=:status,oldStatus=NULL '\n"
        "        sqlU += 'WHERE jediTaskID=:jediTaskID '\n"
        "        self.cur.execute(sqlU + comment, varMap3)\n"
    )
    modules = [_module(_SPECS, "pandaserver/taskbuffer/Specs.py"), _module(source, "x.py")]
    attributor = attribution.SpecAttributor(
        progress.spec_attributes(modules), attribution.class_bases(modules)
    )
    attributor.learn_table_classes(modules)

    side = sqlwrite.read_side(modules, attributor)
    (node,) = sqlwrite.read_site_nodes(side.statements, MAP_ID, VERSION)

    assert (node.owner, node.selects, node.bind_window) == (
        "x.py::sweep",
        ["JediTaskSpec.status=running"],
        [3, 6],
    )


def test_the_forwarded_form_carries_the_region_its_binds_are_filled_in():
    """``var_map = {}`` twice is the code declaring its own block boundaries.

    Kept because it is the region a walk would have to be scoped to: the
    statement's own value is the SQL text, which reads no names, while the
    threshold that decides whether a row comes back is an assignment inside the
    window -- ``copyArchive.py``'s first ``holding`` query binds
    ``:modificationTime`` to a ``timeLimit`` set three hours back, and the same
    function rebinds ``timeLimit`` five times elsewhere.
    """
    source = (
        "def main():\n"
        "    sqlA = f'SELECT jediTaskID FROM {schema}.JEDI_Tasks WHERE status=:status '\n"
        "    var_map = {}\n"
        "    var_map[':status'] = 'pending'\n"
        "    taskBuffer.querySQLS(sqlA + comment, var_map)\n"
        "    var_map = {}\n"
        "    var_map[':status'] = 'finishing'\n"
        "    taskBuffer.querySQLS(sqlA + comment, var_map)\n"
        "def touch(jediTaskID):\n"
        "    varMap3 = {}\n"
        "    varMap3[':status'] = 'ready'\n"
        "    sqlU = f'UPDATE {schema}.JEDI_Tasks SET status=:status,oldStatus=NULL '\n"
        "    sqlU += 'WHERE jediTaskID=:jediTaskID '\n"
        "    cur.execute(sqlU + comment, varMap3)\n"
    )
    modules = [
        _module(_SPECS, "pandaserver/taskbuffer/Specs.py"),
        _module(source, "pandaserver/daemons/scripts/x.py"),
    ]
    attributor = attribution.SpecAttributor(
        progress.spec_attributes(modules), attribution.class_bases(modules)
    )
    attributor.learn_table_classes(modules)

    nodes = sqlwrite.read_site_nodes(
        sqlwrite.read_side(modules, attributor).statements, MAP_ID, VERSION
    )

    # One statement text, two runs, told apart by the window -- which is the
    # same key ``read_side`` deduplicates on.
    assert [(n.anchor.line_start, n.bind_window, n.selects) for n in nodes] == [
        (5, [3, 5], ["JediTaskSpec.status=pending"]),
        (8, [6, 8], ["JediTaskSpec.status=finishing"]),
    ]


def test_a_stop_in_any_variant_marks_the_statement():
    """A branchy statement is several variants at one call, and exactly one ran.

    ``conditions`` deliberately does *not* union them -- a union reads as a
    conjunction and over-states what the row must be -- but a stop is not a
    requirement. It says the reader could not follow a fragment, and a
    fragment one variant carries is one the statement has.  Taking the first
    variant's left ``lockJobsForReassign`` unmarked, because its two-column
    variant, which has no ``WHERE`` at all, recorded before the one carrying
    ``jobStatus IN ({})``.
    """
    source = (
        "def lockJobs(self, statList):\n"
        "    varMap = {}\n"
        "    varMap[':status'] = 'holding'\n"
        "    if wide:\n"
        "        sql = 'SELECT PandaID,lockedby FROM ATLAS_PANDA.jobsActive4 '\n"
        "    else:\n"
        "        sql = 'SELECT PandaID FROM ATLAS_PANDA.jobsActive4 '\n"
        "    sql += 'WHERE jobStatus=:status '\n"
        "    if wide:\n"
        "        sql += f'AND prodSourceLabel IN ({names}) '\n"
        "    self.cur.execute(sql + comment, varMap)\n"
    )
    modules = [
        _module(_SPECS, "pandaserver/taskbuffer/Specs.py"),
        _module(source, "pandaserver/daemons/scripts/x.py"),
    ]
    attributor = attribution.SpecAttributor(
        progress.spec_attributes(modules), attribution.class_bases(modules)
    )
    attributor.learn_table_classes(modules)

    (node,) = sqlwrite.read_site_nodes(
        sqlwrite.read_side(modules, attributor).statements, MAP_ID, VERSION
    )

    assert node.selects == ["jobsActive4.jobStatus=holding"]
    assert node.unresolved == ["prodSourceLabel IN ({}) -- " + sql.BINDS_ARE_A_HOLE]


def test_two_statements_sharing_one_var_map_keep_their_own_rows():
    """The region ends at the next *call* when the next window opens above it.

    Twenty of the corpus's 264 statements fill their binds into a ``var_map``
    a statement above them opened.  Both windows then start on the same line,
    so ending the first region "just before the next window" put its close
    above its own ``execute`` -- and the ``fetchall`` two lines under the call,
    which is where the rows the statement asked for actually arrive, fell
    outside the stretch the statement owns.
    """
    source = (
        "class TaskModule:\n"
        "    def sweep(self):\n"
        "        varMap = {}\n"
        "        varMap[':status'] = 'pending'\n"
        "        sqlA = f'SELECT jediTaskID FROM {schema}.JEDI_Tasks '\n"
        "        sqlA += 'WHERE status=:status '\n"
        "        self.cur.execute(sqlA + comment, varMap)\n"
        "        resA = self.cur.fetchall()\n"
        "        varMap[':status'] = 'finishing'\n"
        "        sqlB = f'SELECT taskName FROM {schema}.JEDI_Tasks '\n"
        "        sqlB += 'WHERE status=:status '\n"
        "        self.cur.execute(sqlB + comment, varMap)\n"
        "        resB = self.cur.fetchall()\n"
        "    def touch(self, jediTaskID):\n"
        "        varMap3 = {}\n"
        "        varMap3[':status'] = 'ready'\n"
        "        sqlU = f'UPDATE {schema}.JEDI_Tasks SET status=:status,oldStatus=NULL '\n"
        "        sqlU += 'WHERE jediTaskID=:jediTaskID '\n"
        "        self.cur.execute(sqlU + comment, varMap3)\n"
    )
    modules = [_module(_SPECS, "pandaserver/taskbuffer/Specs.py"), _module(source, "x.py")]
    attributor = attribution.SpecAttributor(
        progress.spec_attributes(modules), attribution.class_bases(modules)
    )
    attributor.learn_table_classes(modules)

    nodes = sqlwrite.read_site_nodes(
        sqlwrite.read_side(modules, attributor).statements, MAP_ID, VERSION
    )

    # One ``varMap = {}`` above both, so both windows open on line 3.
    assert [(n.anchor.line_start, n.bind_window) for n in nodes] == [
        (7, [3, 7]),
        (12, [3, 12]),
    ]
    # The first statement still owns its ``fetchall`` on line 8: the second
    # statement's *call* ends the region, not the window they share.
    assert [n.region for n in nodes] == [[3, 11], [3, 13]]


def _read_nodes(source: str, rel: str = "x.py"):
    modules = [_module(_SPECS, "pandaserver/taskbuffer/Specs.py"), _module(source, rel)]
    attributor = attribution.SpecAttributor(
        progress.spec_attributes(modules), attribution.class_bases(modules)
    )
    attributor.learn_table_classes(modules)
    side = sqlwrite.read_side(modules, attributor)
    return modules, sqlwrite.read_site_nodes(side.statements, MAP_ID, VERSION)


_EXIT_TAIL = (
    "    def touch(self, jediTaskID):\n"
    "        varMap9 = {}\n"
    "        varMap9[':status'] = 'ready'\n"
    "        sqlU = f'UPDATE {schema}.JEDI_Tasks SET status=:status,oldStatus=NULL '\n"
    "        sqlU += 'WHERE jediTaskID=:jediTaskID '\n"
    "        self.cur.execute(sqlU + comment, varMap9)\n"
)


def test_a_row_dropped_in_the_loop_over_the_rows_is_an_exit():
    """The request's own example: ``if int(id) in xmlIDs: continue``.

    The answer a reader came for when the map says a value was selected and
    the job never moved -- the row *was* returned and the code passed it over.
    """
    source = (
        "class TaskModule:\n"
        "    def sweep(self):\n"
        "        varMap = {}\n"
        "        varMap[':status'] = 'pending'\n"
        "        sqlA = f'SELECT jediTaskID FROM {schema}.JEDI_Tasks '\n"
        "        sqlA += 'WHERE status=:status '\n"
        "        self.cur.execute(sqlA + comment, varMap)\n"
        "        res = self.cur.fetchall()\n"
        "        for (taskID,) in res:\n"
        "            if taskID in skipList:\n"
        "                continue\n"
        "            self.doIt(taskID)\n"
    ) + _EXIT_TAIL
    _modules, nodes = _read_nodes(source)

    (node,) = [n for n in nodes if n.anchor.line_start == 7]
    assert node.exits == ["continue@11"]
    # And inside the stretch the statement owns, which is the one invariant
    # this field has: a line outside it belongs to another statement.
    first, last = node.region
    assert first <= 11 <= last


def test_a_loop_that_is_not_over_the_rows_contributes_no_exit():
    """Every loop in the region gives 85 exits over the corpus against 48.

    What the extra 37 are is visible in one statement:
    ``insertFilesForDataset_JEDI``'s first query owns 363 lines of its
    function and loops over its own rows in 7 of them, so all 14 of its exits
    would be filed as "this row was skipped" when they are the function
    working through an unrelated list.
    """
    source = (
        "class TaskModule:\n"
        "    def sweep(self):\n"
        "        varMap = {}\n"
        "        varMap[':status'] = 'pending'\n"
        "        sqlA = f'SELECT jediTaskID FROM {schema}.JEDI_Tasks '\n"
        "        sqlA += 'WHERE status=:status '\n"
        "        self.cur.execute(sqlA + comment, varMap)\n"
        "        res = self.cur.fetchall()\n"
        "        for siteName in siteList:\n"
        "            if siteName in blackList:\n"
        "                continue\n"
        "        return res\n"
    ) + _EXIT_TAIL
    _modules, nodes = _read_nodes(source)

    (node,) = [n for n in nodes if n.anchor.line_start == 7]
    # Neither the ``continue`` in the site loop nor the function's own
    # ``return``: one is about a site and the other about the function.
    assert node.exits == []


def test_a_handler_that_swallows_is_not_an_exit():
    """``except Exception: pass`` is written so the row is *not* stopped.

    P1-53 counted every handler in the row loop, and 2 of the 3 statements it
    credited with an ``except`` said the opposite of what their code does:
    ``getDispatchDatasetsPerUser@2745`` catches and passes, and
    ``getCommands@1795`` catches and assigns a fallback.  Both rows go on.
    """
    source = (
        "class TaskModule:\n"
        "    def sweep(self):\n"
        "        varMap = {}\n"
        "        varMap[':status'] = 'pending'\n"
        "        sqlA = f'SELECT jediTaskID FROM {schema}.JEDI_Tasks '\n"
        "        sqlA += 'WHERE status=:status '\n"
        "        self.cur.execute(sqlA + comment, varMap)\n"
        "        res = self.cur.fetchall()\n"
        "        for (taskID,) in res:\n"
        "            try:\n"
        "                self.doIt(taskID)\n"
        "            except Exception:\n"
        "                pass\n"
    ) + _EXIT_TAIL
    _modules, nodes = _read_nodes(source)

    (node,) = [n for n in nodes if n.anchor.line_start == 7]
    assert node.exits == []


def test_a_handler_that_leaves_is_an_exit():
    """The qualification is what the handler *contains*, not whether it is empty.

    ``setScoutJobDataToTasks_JEDI@3605`` is the corpus's one real case: its
    handler logs and then gives up on the row.
    """
    source = (
        "class TaskModule:\n"
        "    def sweep(self):\n"
        "        varMap = {}\n"
        "        varMap[':status'] = 'pending'\n"
        "        sqlA = f'SELECT jediTaskID FROM {schema}.JEDI_Tasks '\n"
        "        sqlA += 'WHERE status=:status '\n"
        "        self.cur.execute(sqlA + comment, varMap)\n"
        "        res = self.cur.fetchall()\n"
        "        for (taskID,) in res:\n"
        "            try:\n"
        "                self.doIt(taskID)\n"
        "            except Exception:\n"
        "                continue\n"
    ) + _EXIT_TAIL
    _modules, nodes = _read_nodes(source)

    (node,) = [n for n in nodes if n.anchor.line_start == 7]
    assert node.exits == ["continue@13", "except@12"]


def test_a_handler_assigning_a_fallback_is_not_an_exit():
    """``getCommands@1795`` catches and assigns, which is not empty and not an exit.

    The rule reads the handler for a way out, not for a body -- an assignment
    is the code saying the row carries on with a substitute.
    """
    source = (
        "class TaskModule:\n"
        "    def sweep(self):\n"
        "        varMap = {}\n"
        "        varMap[':status'] = 'pending'\n"
        "        sqlA = f'SELECT jediTaskID FROM {schema}.JEDI_Tasks '\n"
        "        sqlA += 'WHERE status=:status '\n"
        "        self.cur.execute(sqlA + comment, varMap)\n"
        "        res = self.cur.fetchall()\n"
        "        for (taskID,) in res:\n"
        "            try:\n"
        "                params = self.parse(taskID)\n"
        "            except Exception:\n"
        "                params = {}\n"
        "            self.doIt(params)\n"
    ) + _EXIT_TAIL
    _modules, nodes = _read_nodes(source)

    (node,) = [n for n in nodes if n.anchor.line_start == 7]
    assert node.exits == []


def test_the_stretch_the_rows_are_worked_through_is_recorded():
    """``rows_handled`` is the loop over the rows, and it is not ``region``.

    The region ends where the next statement's binds start being filled; the
    loop runs to its own end.  For 64 of the corpus's 151 looping statements
    the second is past the first, because the next query opens *inside* the
    loop -- which is the shape here.
    """
    source = (
        "class TaskModule:\n"
        "    def sweep(self):\n"
        "        varMap = {}\n"
        "        varMap[':status'] = 'pending'\n"
        "        sqlA = f'SELECT jediTaskID FROM {schema}.JEDI_Tasks '\n"
        "        sqlA += 'WHERE status=:status '\n"
        "        self.cur.execute(sqlA + comment, varMap)\n"
        "        res = self.cur.fetchall()\n"
        "        for (taskID,) in res:\n"
        "            varMap2 = {}\n"
        "            varMap2[':jediTaskID'] = taskID\n"
        "            sqlB = f'SELECT status FROM {schema}.JEDI_Tasks '\n"
        "            sqlB += 'WHERE jediTaskID=:jediTaskID '\n"
        "            self.cur.execute(sqlB + comment, varMap2)\n"
        "            resB = self.cur.fetchall()\n"
        "            if not resB:\n"
        "                continue\n"
        "            self.doIt(taskID)\n"
    ) + _EXIT_TAIL
    _modules, nodes = _read_nodes(source)

    (node,) = [n for n in nodes if n.anchor.line_start == 7]
    # The region stops where the inner query's binds start being filled...
    assert node.region == [3, 9]
    # ...and the rows go on being worked through to the end of the loop.
    assert node.rows_handled == [9, 18]


def test_an_exit_in_the_loop_past_the_region_is_still_this_statements():
    """The one thing P1-53 did not deliver for the analyser.

    ``getTasksToExecCommand_JEDI@4591`` owns 4585-4603 and works its rows
    through 4600-4621, because another query opens on 4606 *inside* the loop.
    Read in the region alone it has no exits at all; read to the loop's end it
    has the ``except`` that gives up the lock and the ``continue`` that skips
    the task, which is the answer to "the map says this task was selected and
    nothing happened to it".
    """
    source = (
        "class TaskModule:\n"
        "    def sweep(self):\n"
        "        varMap = {}\n"
        "        varMap[':status'] = 'pending'\n"
        "        sqlA = f'SELECT jediTaskID FROM {schema}.JEDI_Tasks '\n"
        "        sqlA += 'WHERE status=:status '\n"
        "        self.cur.execute(sqlA + comment, varMap)\n"
        "        res = self.cur.fetchall()\n"
        "        for (taskID,) in res:\n"
        "            varMap2 = {}\n"
        "            varMap2[':jediTaskID'] = taskID\n"
        "            sqlB = f'SELECT status FROM {schema}.JEDI_Tasks '\n"
        "            sqlB += 'WHERE jediTaskID=:jediTaskID '\n"
        "            self.cur.execute(sqlB + comment, varMap2)\n"
        "            resB = self.cur.fetchall()\n"
        "            if resB[0][0] == 'done':\n"
        "                continue\n"
        "            self.doIt(taskID)\n"
    ) + _EXIT_TAIL
    _modules, nodes = _read_nodes(source)

    (node,) = [n for n in nodes if n.anchor.line_start == 7]
    # Line 17 is nine lines past where this statement's region ends...
    assert node.region == [3, 9]
    # ...and it is still where a row this statement asked for stops.
    assert node.exits == ["continue@17"]


def test_an_exit_in_a_loop_nested_in_the_row_loop_is_not_this_statements():
    """The exit belongs to the loop it is *taken in*, not to a line range.

    Followed to the loop's end and judged by range, 19 more exits come in from
    loops nested inside it: ``copyArchive@206`` picks up ``continue@245``,
    which passes over one of a job's *files*, not the job.
    """
    source = (
        "class TaskModule:\n"
        "    def sweep(self):\n"
        "        varMap = {}\n"
        "        varMap[':status'] = 'pending'\n"
        "        sqlA = f'SELECT jediTaskID FROM {schema}.JEDI_Tasks '\n"
        "        sqlA += 'WHERE status=:status '\n"
        "        self.cur.execute(sqlA + comment, varMap)\n"
        "        res = self.cur.fetchall()\n"
        "        for (taskID,) in res:\n"
        "            for fileSpec in self.filesOf(taskID):\n"
        "                if fileSpec.status == 'bad':\n"
        "                    continue\n"
        "            self.doIt(taskID)\n"
    ) + _EXIT_TAIL
    _modules, nodes = _read_nodes(source)

    (node,) = [n for n in nodes if n.anchor.line_start == 7]
    assert node.exits == []
    # The file loop is inside the row loop, so the rows are still worked
    # through across all of it -- the filter is on the exit, not the stretch.
    assert node.rows_handled == [9, 13]


def test_a_loop_opening_outside_the_region_is_not_this_statements_rows():
    """The loop must *open* inside the region; only its end is followed.

    Searching the function for every loop over the same names gives 174 exits
    over 71 statements against 90 over 45. ``res`` is reused:
    ``copyArchive::main``'s first query would pick up the ``continue``
    belonging to its query 285 lines later, and one handler on line 1364 would
    land on 19 statements. The region is what makes ``res`` mean *this*
    ``res``.
    """
    source = (
        "class TaskModule:\n"
        "    def sweep(self):\n"
        "        varMap = {}\n"
        "        varMap[':status'] = 'pending'\n"
        "        sqlA = f'SELECT jediTaskID FROM {schema}.JEDI_Tasks '\n"
        "        sqlA += 'WHERE status=:status '\n"
        "        self.cur.execute(sqlA + comment, varMap)\n"
        "        res = self.cur.fetchall()\n"
        "        varMap2 = {}\n"
        "        varMap2[':status'] = 'running'\n"
        # A different select list, or the two runs share a dedup key: a
        # statement is kept once per ``(text, window)``.
        "        sqlB = f'SELECT taskName FROM {schema}.JEDI_Tasks '\n"
        "        sqlB += 'WHERE status=:status '\n"
        "        self.cur.execute(sqlB + comment, varMap2)\n"
        "        res = self.cur.fetchall()\n"
        "        for (taskID,) in res:\n"
        "            if taskID in skipList:\n"
        "                continue\n"
        "            self.doIt(taskID)\n"
    ) + _EXIT_TAIL
    _modules, nodes = _read_nodes(source)

    first, second = [n for n in nodes if n.anchor.line_start in (7, 13)]
    # The loop names ``res``, which the first statement also fills -- and it
    # opens in the second statement's stretch, so it is the second's rows.
    assert first.exits == []
    assert first.rows_handled == []
    assert second.exits == ["continue@17"]
    assert second.rows_handled == [15, 18]


def test_a_statement_whose_rows_nothing_loops_over_handles_them_nowhere():
    """Empty is the answer, not a gap: 113 of the corpus's 264 have no loop."""
    source = (
        "class TaskModule:\n"
        "    def sweep(self):\n"
        "        varMap = {}\n"
        "        varMap[':status'] = 'pending'\n"
        "        sqlA = f'SELECT jediTaskID FROM {schema}.JEDI_Tasks '\n"
        "        sqlA += 'WHERE status=:status '\n"
        "        self.cur.execute(sqlA + comment, varMap)\n"
        "        res = self.cur.fetchall()\n"
        "        return res\n"
    ) + _EXIT_TAIL
    _modules, nodes = _read_nodes(source)

    (node,) = [n for n in nodes if n.anchor.line_start == 7]
    assert node.rows_handled == []


def test_an_exit_because_nothing_came_back_says_so():
    """A different answer from a row being dropped, so a different prefix.

    "The rows were there and this one was passed over" sends a reader to the
    handling; "nothing came back" sends them to the ``conditions``.
    """
    source = (
        "class TaskModule:\n"
        "    def sweep(self):\n"
        "        varMap = {}\n"
        "        varMap[':status'] = 'pending'\n"
        "        sqlA = f'SELECT jediTaskID FROM {schema}.JEDI_Tasks '\n"
        "        sqlA += 'WHERE status=:status '\n"
        "        self.cur.execute(sqlA + comment, varMap)\n"
        "        res = self.cur.fetchall()\n"
        "        if res is None:\n"
        "            return None\n"
        "        return res\n"
    ) + _EXIT_TAIL
    _modules, nodes = _read_nodes(source)

    (node,) = [n for n in nodes if n.anchor.line_start == 7]
    # The ``return`` on line 11 is the function's, under no test on the rows.
    assert node.exits == ["no-rows:return@10"]


def test_the_branch_where_the_rows_did_come_back_is_not_a_no_rows_exit():
    """``if res is None: ... else: <work> ; break`` -- the ``break`` means the opposite.

    Caught by the gate on the real corpus, not here first:
    ``doForRedoStalledJobs`` logs "no files" when the query comes back empty
    and works through the files when it does not, ending in a ``break`` that
    says "found what I came for".  Filed as ``no-rows:`` it told a reader the
    query returned nothing, which is the one thing that ``break`` rules out.

    The same statement showed the other half of it: that ``break`` is on line
    631 and the statement's region ends at 613, so it also belongs to the next
    query's stretch.
    """
    source = (
        "class TaskModule:\n"
        "    def sweep(self):\n"
        "        varMap = {}\n"
        "        varMap[':status'] = 'pending'\n"
        "        sqlA = f'SELECT jediTaskID FROM {schema}.JEDI_Tasks '\n"
        "        sqlA += 'WHERE status=:status '\n"
        "        self.cur.execute(sqlA + comment, varMap)\n"
        "        resT = self.cur.fetchall()\n"
        "        if resT is None or len(resT) == 0:\n"
        "            tmpLog.debug('no tasks')\n"
        "        else:\n"
        "            for (taskID,) in resT:\n"
        "                self.doIt(taskID)\n"
        "            break\n"
    ) + _EXIT_TAIL
    _modules, nodes = _read_nodes(source)

    (node,) = [n for n in nodes if n.anchor.line_start == 7]
    # Neither exit: the ``break`` is in the branch where the rows arrived, and
    # the loop over them holds nothing that leaves it.
    assert node.exits == []


def test_the_rows_are_followed_to_the_fetch_that_binds_them():
    """``execute`` binds nothing at the call -- 195 of 264 arrive at a fetch.

    Reading the call line alone leaves most statements with no name to test
    an exit against, and the no-rows exits drop from 14 to 3.
    """
    source = (
        "class TaskModule:\n"
        "    def sweep(self):\n"
        "        varMap = {}\n"
        "        varMap[':status'] = 'pending'\n"
        "        sqlA = f'SELECT jediTaskID FROM {schema}.JEDI_Tasks '\n"
        "        sqlA += 'WHERE status=:status '\n"
        "        self.cur.execute(sqlA + comment, varMap)\n"
        "        self.conn.commit()\n"
        "        resTask = self.cur.fetchall()\n"
        "        if not resTask:\n"
        "            return None\n"
        "        for (taskID,) in resTask:\n"
        "            if taskID in skipList:\n"
        "                break\n"
    ) + _EXIT_TAIL
    _modules, nodes = _read_nodes(source)

    (node,) = [n for n in nodes if n.anchor.line_start == 7]
    assert node.exits == ["break@14", "no-rows:return@11"]


@pytest.mark.parametrize(
    ("rows", "exits", "handled"),
    [
        (
            "        has_task = self.cur.fetchone() is not None\n        if not has_task:\n            return None\n",
            ["no-rows:return@10"],
            [],
        ),
        ("        n_rows = self.cur.rowcount\n        if n_rows == 0:\n            return None\n", ["no-rows:return@10"], []),
        (
            "        for (taskID,) in self.cur:\n            if taskID in skipList:\n                continue\n",
            ["continue@10"],
            [8, 10],
        ),
    ],
    ids=["fetch-in-a-value", "rowcount", "loop-over-the-cursor"],
)
def test_rows_land_where_a_fetch_computes_from_them_and_in_a_loop_over_the_cursor(rows, exits, handled):
    """``has_merging = self.cur.fetchone() is not None`` and ``nRow = self.cur.rowcount`` hold the rows, and a loop
    over the cursor that ran the statement is a loop over them -- read as the whole value only, none was."""
    source = (
        "class TaskModule:\n"
        "    def sweep(self):\n"
        "        varMap = {}\n"
        "        varMap[':status'] = 'pending'\n"
        "        sqlA = f'SELECT jediTaskID FROM {schema}.JEDI_Tasks '\n"
        "        sqlA += 'WHERE status=:status '\n"
        "        self.cur.execute(sqlA + comment, varMap)\n"
        f"{rows}"
    ) + _EXIT_TAIL
    _modules, nodes = _read_nodes(source)

    (node,) = [n for n in nodes if n.anchor.line_start == 7]
    assert (node.exits, node.rows_handled) == (exits, handled)


def test_a_statement_says_whether_it_returns_rows_or_values_computed_from_them():
    """Read off every way it is built: an ``if`` choosing a ``COUNT(*)`` or the IDs is the one and the other."""
    source = (
        "class TaskModule:\n"
        "    def sweep(self):\n"
        "        varMap = {}\n"
        "        varMap[':status'] = 'pending'\n"
        "        sqlA = f'SELECT COUNT(*) FROM {schema}.JEDI_Tasks '\n"
        "        sqlA += 'WHERE status=:status '\n"
        "        self.cur.execute(sqlA + comment, varMap)\n"
        "        sqlB = f'SELECT jediTaskID FROM {schema}.JEDI_Tasks '\n"
        "        sqlB += 'WHERE status=:status '\n"
        "        self.cur.execute(sqlB + comment, varMap)\n"
        "        if self.flag:\n"
        "            sqlC = f'SELECT MAX(jediTaskID) FROM {schema}.JEDI_Tasks WHERE status=:status '\n"
        "        else:\n"
        "            sqlC = f'SELECT jediTaskID, vo FROM {schema}.JEDI_Tasks WHERE status=:status '\n"
        "        self.cur.execute(sqlC + comment, varMap)\n"
    ) + _EXIT_TAIL
    _modules, nodes = _read_nodes(source)

    assert {n.anchor.line_start: n.result for n in nodes} == {
        7: RESULT_AGGREGATE,
        10: RESULT_ROWS,
        15: RESULT_VARIES,
    }


def _crossed(tmp_path, source, junctions, *, others=()):
    """Write *source* as ``pandaserver/x.py`` and *others* beside it, build the read sites, and resolve them.

    Through the corpus's calls and its reach record, as the build does: a crossing is placed, never named, so the
    fixtures are a corpus on disk whose receivers say what they are.
    """
    from bamboo.codemap import reach
    from bamboo.codemap.panda import plugin

    files = {
        "pandaserver/taskbuffer/Specs.py": _SPECS,
        "pandaserver/x.py": source,
        **{f"pandaserver/{name}": text for text, name in others},
    }
    modules = []
    for rel, text in files.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        modules.append(_module(text, rel))
    attributor = attribution.SpecAttributor(
        progress.spec_attributes(modules), attribution.class_bases(modules)
    )
    attributor.learn_table_classes(modules)
    nodes = sqlwrite.read_site_nodes(
        sqlwrite.read_side(modules, attributor).statements, MAP_ID, VERSION
    )
    calls = sqlwrite.corpus_calls(modules, {"pandaserver": tmp_path / "pandaserver"})
    with sql.across_calls(calls):
        record = reach.ReachGraph(reach.graph(calls, MAP_ID, VERSION, plugin._DatabaseLeaves()))
        census = sqlwrite.attach_crossings(
            nodes, modules, [_junction(f"pandaserver/{owner}") for owner in junctions], calls, record
        )
    return nodes, census


def _typed(source: str, *fields: tuple[str, str, str]) -> str:
    """*source*'s last class given ``self.<field>`` of each class ``(field, class, module)``, and the imports, at its end.

    At the end, so the lines a test names stay where they are.
    """
    held = "".join(f", {field}: {cls}" for field, cls, _module_name in fields)
    sets = "".join(f"        self.{field} = {field}\n" for field, _cls, _module_name in fields)
    imports = "".join(f"from pandaserver.{module_name} import {cls}\n" for _field, cls, module_name in fields)
    return f"{source}    def __init__(self{held}):\n{sets}\n\n{imports}"


def test_a_statement_says_which_junction_its_rows_are_carried_to(tmp_path):
    """The request's sixth point: a statement is somewhere to lead *from*.

    Two calls of the region, and the thread one is not decoration --
    ``copyArchive.py``'s ``holding`` query hands each job to
    ``Watcher(...).start()``, so what happens to a held job next is
    ``Watcher.run``: ``start`` runs it on another thread.
    """
    source = _typed(
        (
            "class TaskModule:\n"
            "    def sweep(self):\n"
            "        varMap = {}\n"
            "        varMap[':status'] = 'pending'\n"
            "        sqlA = f'SELECT jediTaskID FROM {schema}.JEDI_Tasks '\n"
            "        sqlA += 'WHERE status=:status '\n"
            "        self.cur.execute(sqlA + comment, varMap)\n"
            "        res = self.cur.fetchall()\n"
            "        for (taskID,) in res:\n"
            "            thr = Watcher(taskID)\n"
            "            thr.start()\n"
            "            self.retrier.retryTask(taskID)\n"
        )
        + _EXIT_TAIL,
        ("retrier", "Retry", "r"),
    ) + "from pandaserver.w import Watcher\n"
    watcher = (
        "import threading\n\n\n"
        "class Watcher(threading.Thread):\n"
        "    def __init__(self, task_id):\n"
        "        threading.Thread.__init__(self)\n"
        "        self.task_id = task_id\n\n"
        "    def run(self):\n"
        "        if self.task_id:\n"
        "            pass\n"
    )
    retrier = "class Retry:\n    def retryTask(self, t):\n        if t:\n            pass\n"
    nodes, census = _crossed(
        tmp_path, source, ["w.py::run", "r.py::retryTask"], others=[(watcher, "w.py"), (retrier, "r.py")]
    )

    (node,) = [n for n in nodes if n.anchor.line_start == 7]
    assert node.crosses_to == ["pandaserver/r.py::retryTask", "pandaserver/w.py::run"]
    assert census.stored == 2


@pytest.mark.parametrize("unread_kept", [True, False], ids=["the build's", "strict"])
def test_a_count_of_the_rows_crosses_nowhere_and_another_systems_answer_is_kept(tmp_path, monkeypatch, unread_kept):
    """U1: what a callee is handed made from the rows is none of them -- ``len(res)`` is a number -- while what the map
    cannot read, another system's answer made from them, is kept as a crossing in the build's way (and is none read
    strictly, as the KeyFroms read it)."""
    source = _typed(
        (
            "class TaskModule:\n"
            "    def sweep(self):\n"
            "        varMap = {}\n"
            "        varMap[':status'] = 'pending'\n"
            "        sqlA = f'SELECT jediTaskID FROM {schema}.JEDI_Tasks '\n"
            "        sqlA += 'WHERE status=:status '\n"
            "        self.cur.execute(sqlA + comment, varMap)\n"
            "        res = self.cur.fetchall()\n"
            "        self.retrier.retryTask(len(res))\n"
            "    def sweep_more(self):\n"
            "        varMap = {}\n"
            "        varMap[':status'] = 'ready'\n"
            "        sqlB = f'SELECT jediTaskID FROM {schema}.JEDI_Tasks '\n"
            "        sqlB += 'WHERE status=:status '\n"
            "        self.cur.execute(sqlB + comment, varMap)\n"
            "        res = self.cur.fetchall()\n"
            "        self.retrier.retryTask(self.ddm.lookup(res))\n"
        )
        + _EXIT_TAIL,
        ("retrier", "Retry", "r"),
    )
    retrier = "class Retry:\n    def retryTask(self, t):\n        if t:\n            pass\n"
    real = sqlwrite.attach_crossings
    monkeypatch.setattr(sqlwrite, "attach_crossings", lambda *a: real(*a, unread_kept=unread_kept))
    nodes, _census = _crossed(tmp_path, source, ["r.py::retryTask"], others=[(retrier, "r.py")])

    crossed = {n.anchor.line_start: n.crosses_to for n in nodes if n.anchor.line_start in (7, 15)}
    assert crossed == {7: [], 15: ["pandaserver/r.py::retryTask"] if unread_kept else []}


_NEXT_STATEMENT = (
    "        varMap2 = {}\n"
    "        varMap2[':status'] = 'ready'\n"
    "        sqlB = f'SELECT jediTaskID FROM {schema}.JEDI_Tasks '\n"
    "        sqlB += 'WHERE status=:status '\n"
    "        self.cur.execute(sqlB + comment, varMap2)\n"
)


def test_a_call_crosses_where_the_rows_reach_it_not_where_it_stands(tmp_path):
    """The call running the statement, and each call handed its rows, past its region too.

    A call of the region the rows do not reach is no crossing -- ``copyArchive.py::main@696``'s
    ``taskBuffer.lockJobsForReassign`` -- nor, for the next statement, a call handed only the first one's rows.
    """
    source = _typed(
        (
            "class TaskModule:\n"
            "    def sweep(self):\n"
            "        varMap = {}\n"
            "        varMap[':status'] = 'pending'\n"
            "        sqlA = f'SELECT jediTaskID FROM {schema}.JEDI_Tasks '\n"
            "        sqlA += 'WHERE status=:status '\n"
            "        self.cur.execute(sqlA + comment, varMap)\n"
            "        res = self.cur.fetchall()\n"
            "        self.counter.tick()\n"
            f"{_NEXT_STATEMENT}"
            "        self.retrier.retryTask(res)\n"
        )
        + _EXIT_TAIL,
        ("retrier", "Retry", "r"),
        ("counter", "Counter", "k"),
    )
    retrier = "class Retry:\n    def retryTask(self, t):\n        if t:\n            pass\n"
    counter = "class Counter:\n    def tick(self):\n        pass\n"
    nodes, _census = _crossed(
        tmp_path, source, ["r.py::retryTask", "k.py::tick"], others=[(retrier, "r.py"), (counter, "k.py")]
    )

    (node,) = [n for n in nodes if n.anchor.line_start == 7]
    assert node.crosses_to == ["pandaserver/r.py::retryTask"]
    assert [n.crosses_to for n in nodes if n is not node] == [[]] * (len(nodes) - 1)


_PICK = (
    "class TaskModule:\n"
    "    def pick(self, target_tasks=None):\n"
    "        varMap = {}\n"
    "        if not target_tasks:\n"
    "            varMap[':status'] = 'ready'\n"
    "            sql = f'SELECT jediTaskID FROM {schema}.JEDI_Tasks WHERE status=:status '\n"
    "        else:\n"
    "            varMap[':jediTaskID'] = target_tasks[0]\n"
    "            sql = f'SELECT jediTaskID FROM {schema}.JEDI_Tasks WHERE jediTaskID=:jediTaskID '\n"
    "        self.cur.execute(sql + comment, varMap)\n"
    "        res = self.cur.fetchall()\n"
    "        return res\n"
)


def _picking(cls: str, handed: str, to: str) -> str:
    """A class whose ``run`` calls ``TaskModule.pick`` handing *handed* and gives the rows to the call *to*."""
    return (
        "from pandaserver.x import TaskModule\n"
        "from pandaserver.s import Setter\n"
        "from pandaserver.y import relay\n\n\n"
        f"class {cls}:\n"
        "    def __init__(self, proxy: TaskModule, setter: Setter):\n"
        "        self.proxy = proxy\n"
        "        self.setter = setter\n\n"
        "    def run(self):\n"
        f"        rows = self.proxy.pick({handed})\n"
        f"        {to}(rows)\n"
    )


@pytest.mark.parametrize(
    ("broker", "generator", "selects"),
    [
        ("self.setter.setCloud", "len", {"pandaserver/s.py::setCloud": []}),
        ("relay", "len", {"pandaserver/s.py::setCloud": []}),
        ("relay", "relay", {}),
        ("self.setter.setCloud", "self.setter.setCloud", {}),
    ],
    ids=["its-own-way-in", "through-a-function", "a-function-both-ways-go-through", "both-ways-in"],
)
def test_a_crossing_takes_the_values_its_own_routes_select(tmp_path, broker, generator, selects):
    """1898's rows reach ``setCloudToTasks_JEDI`` only back up TaskBroker 259, which hands ``simTasks=[...]``: the
    status values of the branch it does not take go nowhere there (XB).

    Two callers of one statement: ``Broker`` hands a display, so the statement selects no status on its way, and
    ``Gen`` hands nothing.  Where both ways reach ``Setter.setCloud`` it takes the statement's values -- through
    ``relay`` too, which the walk goes through once: the second way in still counts.
    """
    from bamboo.codemap import reach
    from bamboo.codemap.panda import plugin

    files = {
        "pandaserver/taskbuffer/Specs.py": _SPECS,
        "pandaserver/x.py": _PICK,
        "pandaserver/s.py": "class Setter:\n    def setCloud(self, t):\n        if t:\n            pass\n",
        "pandaserver/y.py": "from pandaserver.s import Setter\n\n\ndef relay(t):\n    Setter().setCloud(t)\n",
        "pandaserver/b.py": _picking("Broker", "target_tasks=[1]", broker),
        "pandaserver/g.py": _picking("Gen", "", generator),
    }
    modules = []
    for rel, text in files.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        modules.append(_module(text, rel))
    attributor = attribution.SpecAttributor(progress.spec_attributes(modules), attribution.class_bases(modules))
    attributor.learn_table_classes(modules)
    calls = sqlwrite.corpus_calls(modules, {"pandaserver": tmp_path / "pandaserver"})
    with sql.across_calls(calls):
        nodes = sqlwrite.read_site_nodes(sqlwrite.read_side(modules, attributor, calls).statements, MAP_ID, VERSION)
        record = reach.ReachGraph(reach.graph(calls, MAP_ID, VERSION, plugin._DatabaseLeaves()))
        sqlwrite.attach_crossings(nodes, modules, [_junction("pandaserver/s.py::setCloud")], calls, record)

    (node,) = [n for n in nodes if n.anchor.line_start == 10]
    assert node.selects == ["JEDI_Tasks.status=ready"]
    assert sorted(route.chain[-1].split(":")[0] for route in node.call_readings) == [
        "pandaserver/b.py",
        "pandaserver/g.py",
    ]
    assert node.crosses_to == ["pandaserver/s.py::setCloud"]
    assert node.crossing_selects == selects


@pytest.mark.parametrize(
    ("fill", "crosses"),
    [
        ("        spec.rows = res\n", True),
        ("        setattr(spec, 'rows', res)\n", True),
        ("        spec.pack(res)\n", True),
        ("        spec.note(res)\n", True),
        ("        spec.info(res)\n", False),
        ("        spec.label = res\n", False),
    ],
    ids=["attribute", "setattr", "a-method-keeping-it", "a-method-handing-it-to-another", "a-log-line", "another-field"],
)
def test_rows_put_into_an_object_are_held_by_it(tmp_path, fill, crosses):
    """``dataset_spec.pack(tmp_res)`` keeps the rows in ``dataset_spec``: ``pack`` sets each value on ``self``.

    In the field it puts them into: ``retryTask`` reads ``t.rows``, so rows put into ``spec.label`` are not what it
    reads.  At any depth, through another method of the receiver too (``note`` hands them to ``self.keep``), but not
    by a log line: ``MsgWrapper.info`` hands its message to ``self.keepMsg``, a log's buffer.
    """
    source = _typed(
        (
            "class TaskModule:\n"
            "    def sweep(self):\n"
            "        varMap = {}\n"
            "        varMap[':status'] = 'pending'\n"
            "        sqlA = f'SELECT jediTaskID FROM {schema}.JEDI_Tasks '\n"
            "        sqlA += 'WHERE status=:status '\n"
            "        self.cur.execute(sqlA + comment, varMap)\n"
            "        res = self.cur.fetchall()\n"
            "        spec = Holder()\n"
            f"{fill}"
            f"{_NEXT_STATEMENT}"
            "        self.retrier.retryTask(spec)\n"
        )
        + _EXIT_TAIL,
        ("retrier", "Retry", "r"),
    ) + "from pandaserver.h import Holder\n"
    holder = (
        "class Holder:\n"
        "    def pack(self, values):\n"
        "        object.__setattr__(self, 'rows', values)\n\n"
        "    def note(self, message):\n"
        "        self.keep(message)\n\n"
        "    def info(self, message):\n"
        "        self.keep(message)\n\n"
        "    def keep(self, message):\n"
        "        self.rows = message\n"
    )
    retrier = "class Retry:\n    def retryTask(self, t):\n        if t.rows:\n            pass\n"
    nodes, _census = _crossed(tmp_path, source, ["r.py::retryTask"], others=[(holder, "h.py"), (retrier, "r.py")])

    (node,) = [n for n in nodes if n.anchor.line_start == 7]
    assert node.crosses_to == (["pandaserver/r.py::retryTask"] if crosses else [])


@pytest.mark.parametrize(
    ("bases", "crosses"),
    [
        ("from pandacommon.pandautils.base import SpecBase\n\n\nclass Holder(SpecBase):\n", True),
        ("class Holder:\n", False),
        ("class Base:\n    pass\n\n\nclass Holder(Base):\n", False),
    ],
    ids=["a-base-outside-the-corpus", "no-base", "a-base-of-the-corpus"],
)
def test_a_method_an_outside_base_supplies_keeps_what_it_is_handed_in_its_object(tmp_path, bases, crosses):
    """``dc_req_spec.pack(row)``: ``DataCarouselRequestSpec`` gets ``pack`` from pandacommon's ``SpecBase``, which the
    corpus does not hold, so nothing says what it does with the row -- it is kept in the spec, which goes on to
    ``retryTask``.  A class with no base outside the corpus has no such method, and keeps nothing."""
    source = _typed(
        (
            "class TaskModule:\n"
            "    def sweep(self):\n"
            "        varMap = {}\n"
            "        varMap[':status'] = 'pending'\n"
            "        sqlA = f'SELECT jediTaskID FROM {schema}.JEDI_Tasks '\n"
            "        sqlA += 'WHERE status=:status '\n"
            "        self.cur.execute(sqlA + comment, varMap)\n"
            "        res = self.cur.fetchall()\n"
            "        spec = Holder()\n"
            "        spec.pack(res)\n"
            f"{_NEXT_STATEMENT}"
            "        self.retrier.retryTask(spec)\n"
        )
        + _EXIT_TAIL,
        ("retrier", "Retry", "r"),
    ) + "from pandaserver.h import Holder\n"
    holder = f"{bases}    _attributes = ('rows',)\n"
    retrier = "class Retry:\n    def retryTask(self, t):\n        if t.rows:\n            pass\n"
    nodes, _census = _crossed(tmp_path, source, ["r.py::retryTask"], others=[(holder, "h.py"), (retrier, "r.py")])

    (node,) = [n for n in nodes if n.anchor.line_start == 7]
    assert node.crosses_to == (["pandaserver/r.py::retryTask"] if crosses else [])


def _sweeping(lines: str, imports: str = "") -> str:
    """The task module of these tests: its statement at line 7, its rows ``res``, then *lines*."""
    return (
        _typed(
            (
                "class TaskModule:\n"
                "    def sweep(self):\n"
                "        varMap = {}\n"
                "        varMap[':status'] = 'pending'\n"
                "        sqlA = f'SELECT jediTaskID FROM {schema}.JEDI_Tasks '\n"
                "        sqlA += 'WHERE status=:status '\n"
                "        self.cur.execute(sqlA + comment, varMap)\n"
                "        res = self.cur.fetchall()\n"
                f"{lines}"
                f"{_NEXT_STATEMENT}"
            )
            + _EXIT_TAIL,
            ("retrier", "Retry", "r"),
        )
        + imports
    )


_RETRY_TESTING = "class Retry:\n    def retryTask(self, t):\n        if t:\n            pass\n"


@pytest.mark.parametrize(
    ("declared", "crosses"),
    [("def declares(lists: WeightedLists) -> None:\n    pass\n", True), ("", False)],
    ids=["a-class-the-corpus-annotates", "a-name-the-corpus-never-annotates"],
)
def test_a_library_object_keeps_what_its_method_is_handed(tmp_path, declared, crosses):
    """add_main's ``jor_lists = WeightedLists(lock)`` then ``jor_lists.add(3, jor_list)``: pandacommon's
    ``WeightedLists.add(weight, list_data)`` is no builtin's, and nothing says what it does with the rows, so the
    object keeps them, whole (:mod:`callfacts`).  It is a class's object because the corpus declares a parameter
    ``WeightedLists`` somewhere; where nothing does, the call reads as ``set.add`` by its name, which puts the 3."""
    source = _sweeping(
        "        lists = WeightedLists(lock)\n"
        "        lists.add(3, res)\n"
        "        self.retrier.retryTask(lists.pop())\n",
        "from pandacommon.pandautils.thread_utils import WeightedLists\n\n\n" + declared,
    )
    nodes, _census = _crossed(tmp_path, source, ["r.py::retryTask"], others=[(_RETRY_TESTING, "r.py")])

    (node,) = [n for n in nodes if n.anchor.line_start == 7]
    assert node.crosses_to == (["pandaserver/r.py::retryTask"] if crosses else [])


@pytest.mark.parametrize(
    ("target", "crosses"),
    [("self.run", True), ("self.done", False)],
    ids=["the-method-handing-the-rows-on", "one-handing-nothing"],
)
def test_a_method_handed_bound_is_entered_with_its_object(tmp_path, target, crosses):
    """AdderThread is built with the job output reports, and ``proc_launch`` hands
    ``multiprocessing.Process(target=self.run)``: ``run`` runs with the object holding them, and hands one on.  The
    walk enters a method handed bound to an object holding a part of the rows, its object that, as the reach
    record's ``handed`` edge does."""
    source = _sweeping("        worker = Worker(res)\n        worker.launch()\n", "from pandaserver.w import Worker\n")
    worker = (
        "from multiprocessing import Process\n\n"
        "from pandaserver.r import Retry\n\n\n"
        "class Worker:\n"
        "    def __init__(self, rows):\n"
        "        self.rows = rows\n\n"
        "    def launch(self):\n"
        f"        Process(target={target}).start()\n\n"
        "    def run(self):\n"
        "        Retry().retryTask(self.rows)\n\n"
        "    def done(self):\n"
        "        pass\n"
    )
    nodes, _census = _crossed(
        tmp_path, source, ["r.py::retryTask"], others=[(worker, "w.py"), (_RETRY_TESTING, "r.py")]
    )

    (node,) = [n for n in nodes if n.anchor.line_start == 7]
    assert node.crosses_to == (["pandaserver/r.py::retryTask"] if crosses else [])


@pytest.mark.parametrize(
    ("read", "crosses"),
    [("status", True), ("label", False)],
    ids=["the-attribute-the-rows-are-in", "another"],
)
def test_the_pickle_round_trip_copies_what_it_is_handed_attribute_for_attribute(tmp_path, read, crosses):
    """``PickleJobSpec.update(spec)`` is ``self.__setstate__(spec.__getstate__())``: both bodies go over
    ``_attributes`` by a name the loop holds, and read so the copy would hold every part of the spec under every
    attribute.  Read as pickle's round trip, it holds what the spec holds where the spec holds it: ``status``
    carries the rows, ``label`` nothing."""
    source = _sweeping(
        "        spec = Spec()\n"
        "        spec.status = res[0]\n"
        "        spec.label = 'x'\n"
        "        copied = Copy()\n"
        "        copied.update(spec)\n"
        "        self.retrier.retryTask(copied)\n",
        "from pandaserver.s import Copy, Spec\n",
    )
    specs = (
        "class Spec:\n"
        "    _attributes = ('status', 'label')\n\n"
        "    def __getstate__(self):\n"
        "        state = []\n"
        "        for attr in self._attributes:\n"
        "            state.append(getattr(self, attr))\n"
        "        return state\n\n"
        "    def __setstate__(self, state):\n"
        "        for i in range(len(self._attributes)):\n"
        "            object.__setattr__(self, self._attributes[i], state[i])\n\n\n"
        "class Copy(Spec):\n"
        "    def update(self, spec):\n"
        "        self.__setstate__(spec.__getstate__())\n"
    )
    retrier = f"class Retry:\n    def retryTask(self, t):\n        if t.{read}:\n            pass\n"
    nodes, _census = _crossed(tmp_path, source, ["r.py::retryTask"], others=[(specs, "s.py"), (retrier, "r.py")])

    (node,) = [n for n in nodes if n.anchor.line_start == 7]
    assert node.crosses_to == (["pandaserver/r.py::retryTask"] if crosses else [])


def test_the_pickle_round_trip_is_no_read_of_what_it_copies(tmp_path):
    """Setupper's ``p_job.update(job_spec)``: ``PickleFileSpec.update`` copies each file, and ``FileSpec.__getstate__``
    tests a field on the way.  Read by the bodies, ``storeJobs`` -- which only hands its jobs to the Setupper --
    read the rows, and the walk stopped there.  The round trip is a copy, either half: what copies the rows and does
    nothing else with them reads none of them."""
    source = _sweeping(
        "        spec = Spec()\n        spec.status = res[0]\n        self.retrier.retryTask(spec)\n",
        "from pandaserver.s import Spec\n",
    )
    specs = (
        "class Spec:\n"
        "    _attributes = ('status', 'label')\n\n"
        "    def __getstate__(self):\n"
        "        state = []\n"
        "        for attr in self._attributes:\n"
        "            value = getattr(self, attr)\n"
        "            if value is None:\n"
        "                value = 'NULL'\n"
        "            state.append(value)\n"
        "        return state\n\n"
        "    def __setstate__(self, state):\n"
        "        for i in range(len(self._attributes)):\n"
        "            object.__setattr__(self, self._attributes[i], state[i])\n\n\n"
        "class Copy(Spec):\n"
        "    def update(self, spec: Spec):\n"
        "        self.__setstate__(spec.__getstate__())\n"
    )
    retrier = (
        "from pandaserver.s import Copy\n\n\n"
        "class Retry:\n"
        "    def retryTask(self, t):\n"
        "        copied = Copy()\n"
        "        copied.update(t)\n"
    )
    nodes, _census = _crossed(tmp_path, source, ["r.py::retryTask"], others=[(specs, "s.py"), (retrier, "r.py")])

    (node,) = [n for n in nodes if n.anchor.line_start == 7]
    assert node.crosses_to == []


def test_a_function_keeps_what_it_is_handed_in_an_object_it_is_handed(tmp_path):
    """getScoutJobData_JEDI's nested ``addTag(jobTagMap, cpuEffDict, ...)`` files each job's PandaID under
    ``jobTagMap``, whose keys go on to ``updateJobMetrics_JEDI``: a function of the corpus, not only a method, is read
    by its body for what it keeps in what it is handed."""
    source = _sweeping(
        "        box = {}\n        fill(box, res)\n        self.retrier.retryTask(box)\n",
        "from pandaserver.h import fill\n",
    )
    filler = "def fill(d, rows):\n    d['rows'] = rows\n"
    retrier = "class Retry:\n    def retryTask(self, t):\n        if t['rows']:\n            pass\n"
    nodes, _census = _crossed(tmp_path, source, ["r.py::retryTask"], others=[(filler, "h.py"), (retrier, "r.py")])

    (node,) = [n for n in nodes if n.anchor.line_start == 7]
    assert node.crosses_to == ["pandaserver/r.py::retryTask"]


def test_a_method_of_the_corpus_named_as_a_container_s_is_read_by_its_body(tmp_path):
    """JEDI's ``ListWithLock.append(item)`` puts the item into ``self.dataList``, and its ``get`` hands back a slice
    of that: read by its body -- not as ``list.append`` by its name, which put the rows where ``get`` does not look
    -- the rows go into the shelf and come out to ``retryTask``."""
    source = _sweeping(
        "        shelf = Shelf()\n        shelf.append(res)\n        self.retrier.retryTask(shelf.get(1))\n",
        "from pandaserver.l import Shelf\n",
    )
    shelf = (
        "class Shelf:\n"
        "    def __init__(self):\n"
        "        self.dataList = []\n\n"
        "    def append(self, item):\n"
        "        if item not in self.dataList:\n"
        "            self.dataList.append(item)\n\n"
        "    def get(self, num):\n"
        "        return self.dataList[:num]\n"
    )
    retrier = (
        "class Retry:\n    def retryTask(self, t):\n        for one in t:\n            if one:\n                pass\n"
    )
    nodes, _census = _crossed(tmp_path, source, ["r.py::retryTask"], others=[(shelf, "l.py"), (retrier, "r.py")])

    (node,) = [n for n in nodes if n.anchor.line_start == 7]
    assert node.crosses_to == ["pandaserver/r.py::retryTask"]


def test_a_function_gone_through_is_read_for_each_way_in(tmp_path):
    """Entered once a statement for what it is handed, not once: ``handle`` given the rows as ``first`` and then as
    ``second`` reaches both.  Keyed by the definition alone, the second way in was not taken, and which came first
    decided the answer."""
    source = _typed(
        (
            "class TaskModule:\n"
            "    def sweep(self):\n"
            "        varMap = {}\n"
            "        varMap[':status'] = 'pending'\n"
            "        sqlA = f'SELECT jediTaskID FROM {schema}.JEDI_Tasks '\n"
            "        sqlA += 'WHERE status=:status '\n"
            "        self.cur.execute(sqlA + comment, varMap)\n"
            "        res = self.cur.fetchall()\n"
            f"{_NEXT_STATEMENT}"
            "        self.helper.handle(res, None)\n"
            "        self.helper.handle(None, res)\n"
        )
        + _EXIT_TAIL,
        ("helper", "Helper", "h"),
    )
    helper = _typed(
        "class Helper:\n"
        "    def handle(self, first, second):\n"
        "        self.retrier.retryTask(first)\n"
        "        self.finisher.finish(second)\n\n",
        ("retrier", "Retry", "r"),
        ("finisher", "Finisher", "f"),
    )
    retrier = "class Retry:\n    def retryTask(self, t):\n        if t:\n            pass\n"
    finisher = "class Finisher:\n    def finish(self, t):\n        if t:\n            pass\n"
    nodes, _census = _crossed(
        tmp_path,
        source,
        ["r.py::retryTask", "f.py::finish"],
        others=[(helper, "h.py"), (retrier, "r.py"), (finisher, "f.py")],
    )

    (node,) = [n for n in nodes if n.anchor.line_start == 7]
    assert node.crosses_to == ["pandaserver/f.py::finish", "pandaserver/r.py::retryTask"]


def test_a_statement_does_not_cross_to_a_name_no_call_lands_at(tmp_path):
    """A call lands where the corpus's calls place it, not wherever a function of its name owns a junction.

    ``self.helper.retryTask`` is ``Helper``'s, and the junction is ``Retry``'s.  Read by name,
    ``datasetManager.py::run``'s ``thr.run()`` crossed to five ``run``s and ``RecoverLostFilesCore.main``'s
    ``main(...)`` to two other files' ``main``.  Nor is the statement's own owner a lead -- rows do not go next to
    where they are -- or a call to something the map has no answer for.
    """
    source = _typed(
        (
            "class TaskModule:\n"
            "    def sweep(self):\n"
            "        varMap = {}\n"
            "        varMap[':status'] = 'pending'\n"
            "        sqlA = f'SELECT jediTaskID FROM {schema}.JEDI_Tasks '\n"
            "        sqlA += 'WHERE status=:status '\n"
            "        self.cur.execute(sqlA + comment, varMap)\n"
            "        res = self.cur.fetchall()\n"
            "        for (taskID,) in res:\n"
            "            tmpLog.debug(taskID)\n"
            "            self.sweep()\n"
            "            self.helper.retryTask(taskID)\n"
        )
        + _EXIT_TAIL,
        ("helper", "Helper", "h"),
    )
    helper = "class Helper:\n    def retryTask(self, t):\n        pass\n"
    retrier = "class Retry:\n    def retryTask(self, t):\n        if t:\n            pass\n"
    nodes, _census = _crossed(
        tmp_path, source, ["x.py::sweep", "r.py::retryTask"], others=[(helper, "h.py"), (retrier, "r.py")]
    )

    (node,) = [n for n in nodes if n.anchor.line_start == 7]
    assert node.crosses_to == []


def test_a_call_of_the_region_is_followed_through_a_function_owning_no_junction(tmp_path):
    """``doForTaskBoost``'s ``self.taskBufferIF.updateTask_JEDI(task_spec, ...)`` lands in ``JediTaskBuffer``, which
    owns none and hands ``taskSpec`` to ``proxy.updateTask_JEDI``, which does.

    What the function owning none does with what it was handed is read, and only that: ``finishTask(self.other)``
    is handed nothing the call gave it.
    """
    source = _typed(
        (
            "class TaskModule:\n"
            "    def sweep(self):\n"
            "        varMap = {}\n"
            "        varMap[':status'] = 'pending'\n"
            "        sqlA = f'SELECT jediTaskID FROM {schema}.JEDI_Tasks '\n"
            "        sqlA += 'WHERE status=:status '\n"
            "        self.cur.execute(sqlA + comment, varMap)\n"
            "        res = self.cur.fetchall()\n"
            "        for (taskID,) in res:\n"
            "            self.helper.retire(taskID)\n"
        )
        + _EXIT_TAIL,
        ("helper", "Helper", "h"),
    )
    helper = _typed(
        "class Helper:\n"
        "    def retire(self, task_id):\n"
        "        self.retrier.retryTask(task_id)\n"
        "        self.retrier.finishTask(self.other)\n",
        ("retrier", "Retry", "r"),
    )
    retrier = (
        "class Retry:\n"
        "    def retryTask(self, t):\n        if t:\n            pass\n\n"
        "    def finishTask(self, t):\n        if t:\n            pass\n"
    )
    nodes, _census = _crossed(
        tmp_path,
        source,
        ["r.py::retryTask", "r.py::finishTask"],
        others=[(helper, "h.py"), (retrier, "r.py")],
    )

    (node,) = [n for n in nodes if n.anchor.line_start == 7]
    assert node.crosses_to == ["pandaserver/r.py::retryTask"]


def _sweep(body: str) -> str:
    """``TaskModule.sweep``: the statement on line 7, its rows fetched into ``res`` on line 8, then *body*."""
    return (
        "class TaskModule:\n"
        "    def sweep(self):\n"
        "        varMap = {}\n"
        "        varMap[':status'] = 'pending'\n"
        "        sqlA = f'SELECT jediTaskID FROM {schema}.JEDI_Tasks '\n"
        "        sqlA += 'WHERE status=:status '\n"
        "        self.cur.execute(sqlA + comment, varMap)\n"
        "        res = self.cur.fetchall()\n"
    ) + body


_EACH_TASK_RETRIED = "        for (taskID,) in res:\n            self.retrier.retryTask(taskID)\n"
_READS_EACH = "class Retry:\n    def retryTask(self, t):\n        for one in t:\n            if one:\n                pass\n"


def test_an_owner_that_does_not_read_what_it_is_handed_is_gone_through(tmp_path):
    """A junction's owner is a crossing where it reads the part of the rows it is handed; one that only logs it and
    hands it on is gone through, as a function owning none is.  ``Retry.retryTask`` logs ``t`` and hands it to
    ``Finisher.finish``, which tests it."""
    source = _typed(_sweep(_EACH_TASK_RETRIED) + _EXIT_TAIL, ("retrier", "Retry", "r"))
    retrier = _typed(
        "class Retry:\n"
        "    def retryTask(self, t):\n"
        "        tmpLog.debug(t)\n"
        "        self.finisher.finish(t)\n\n",
        ("finisher", "Finisher", "f"),
    )
    finisher = "class Finisher:\n    def finish(self, t):\n        if t:\n            pass\n"
    nodes, _census = _crossed(
        tmp_path, source, ["r.py::retryTask", "f.py::finish"], others=[(retrier, "r.py"), (finisher, "f.py")]
    )

    (node,) = [n for n in nodes if n.anchor.line_start == 7]
    assert node.crosses_to == ["pandaserver/f.py::finish"]


@pytest.mark.parametrize(
    ("maker", "crosses"),
    [("MsgWrapper", False), ("make_logger", False), ("Note", True)],
    ids=["a-logger", "pandacommon-s-maker", "a-class"],
)
def test_a_logger_made_from_the_rows_does_not_read_them(tmp_path, maker, crosses):
    """``tmpLog = MsgWrapper(logger, token)``: the token is a log line's tag, though ``MsgWrapper.__init__`` tests it
    (as ``create_tagged_logger`` tests ``if tag is None``).  Read as a use, 7 crossings of pin k came by a tag alone.
    ``make_logger(logger, token)`` makes one too (callfacts.LOGGER_MAKERS), whatever its body is read as here.  The
    same ``__init__`` under another name reads it."""
    source = _typed(_sweep(_EACH_TASK_RETRIED) + _EXIT_TAIL, ("retrier", "Retry", "r"))
    retrier = (
        f"from pandaserver.m import {maker}\n\n\n"
        "class Retry:\n"
        "    def retryTask(self, t):\n"
        f"        tmpLog = {maker}(None, t)\n"
        "        tmpLog.info('retried')\n"
    )
    wrapper = (
        f"class {maker}:\n"
        "    def __init__(self, logger, token=None):\n"
        "        if token is None:\n"
        "            token = ''\n"
        "        self.token = token\n"
    )
    nodes, _census = _crossed(tmp_path, source, ["r.py::retryTask"], others=[(retrier, "r.py"), (wrapper, "m.py")])

    (node,) = [n for n in nodes if n.anchor.line_start == 7]
    assert node.crosses_to == (["pandaserver/r.py::retryTask"] if crosses else [])


@pytest.mark.parametrize(
    ("lookup", "crosses"),
    [("limits.get(taskID)", False), ("limits.get(taskID, taskID)", True)],
    ids=["the-key", "a-default"],
)
def test_a_key_the_rows_hold_chooses_a_value_that_is_not_theirs(tmp_path, lookup, crosses):
    """``d.get(k)`` on a dict holding none of the rows, ``k`` one of them: the value is one ``d`` held already, the
    rows only choose it -- as ``d[k]`` reads.  A default handed beside the key may be the value."""
    source = _typed(
        _sweep(f"        limits = {{}}\n        for (taskID,) in res:\n            self.retrier.retryTask({lookup})\n")
        + _EXIT_TAIL,
        ("retrier", "Retry", "r"),
    )
    retrier = "class Retry:\n    def retryTask(self, t):\n        if t:\n            pass\n"
    nodes, _census = _crossed(tmp_path, source, ["r.py::retryTask"], others=[(retrier, "r.py")])

    (node,) = [n for n in nodes if n.anchor.line_start == 7]
    assert node.crosses_to == (["pandaserver/r.py::retryTask"] if crosses else [])


def test_rows_appended_to_a_dicts_default_are_held_by_the_dict(tmp_path):
    """``d.setdefault(k, []).append(v)``: the list ``setdefault`` hands back is one of ``d``'s values, so ``v`` is in
    ``d`` (``getDatasetsWithJediTaskID_JEDI@4049`` -> ``insertFilesForDataset_JEDI``)."""
    source = _typed(
        _sweep(
            "        byStatus = {}\n"
            "        for (taskID,) in res:\n"
            "            byStatus.setdefault('pending', []).append(taskID)\n"
            "        self.retrier.retryTask(byStatus['pending'])\n"
        )
        + _EXIT_TAIL,
        ("retrier", "Retry", "r"),
    )
    nodes, _census = _crossed(tmp_path, source, ["r.py::retryTask"], others=[(_READS_EACH, "r.py")])

    (node,) = [n for n in nodes if n.anchor.line_start == 7]
    assert node.crosses_to == ["pandaserver/r.py::retryTask"]


@pytest.mark.parametrize(
    ("read", "crosses"), [("t.a.b.c.d.e.f", True), ("t.a.b.c.d.e.g", True), ("t.a.x", False)], ids=["it", "cut", "other"]
)
def test_a_path_past_five_steps_is_anything_below_the_cut(tmp_path, read, crosses):
    """A holding tells apart five steps of an access path: below them, any part may hold the rows, so a read of
    ``.g`` past the cut is a read of what was put at ``.f``; one parting from the path above the cut is not."""
    source = _typed(
        _sweep(
            "        spec = Holder()\n"
            "        for (taskID,) in res:\n"
            "            spec.a.b.c.d.e.f = taskID\n"
            "        self.retrier.retryTask(spec)\n"
        )
        + _EXIT_TAIL,
        ("retrier", "Retry", "r"),
    )
    retrier = f"class Retry:\n    def retryTask(self, t):\n        if {read}:\n            pass\n"
    nodes, _census = _crossed(tmp_path, source, ["r.py::retryTask"], others=[(retrier, "r.py")])

    (node,) = [n for n in nodes if n.anchor.line_start == 7]
    assert node.crosses_to == (["pandaserver/r.py::retryTask"] if crosses else [])


def test_a_fetch_after_the_next_execute_takes_that_statements_rows(tmp_path):
    """In the loop over the rows the next ``execute`` runs another statement, and what is fetched after it is that
    one's, though the first one's rows are still being handled: the seeds end with the region, as the exits do
    (``ppEventServiceJob`` 5654 / 5660; census task 3: 144 seeds of pin k took another statement's rows)."""
    source = _typed(
        _sweep(
            "        for (taskID,) in res:\n"
            "            varMap2 = {}\n"
            "            varMap2[':status'] = 'ready'\n"
            "            sqlB = f'SELECT datasetID FROM {schema}.JEDI_Datasets '\n"
            "            sqlB += 'WHERE status=:status '\n"
            "            self.cur.execute(sqlB + comment, varMap2)\n"
            "            other = self.cur.fetchall()\n"
            "            self.retrier.retryTask(other)\n"
        )
        + _EXIT_TAIL,
        ("retrier", "Retry", "r"),
    )
    nodes, _census = _crossed(tmp_path, source, ["r.py::retryTask"], others=[(_READS_EACH, "r.py")])

    crossed = {n.anchor.line_start: n.crosses_to for n in nodes if n.anchor.line_start in (7, 14)}
    assert crossed == {7: [], 14: ["pandaserver/r.py::retryTask"]}


ROWS_HANDED_BACK = (
    "class TaskModule:\n"
    "    def getTasks(self):\n"
    "        varMap = {}\n"
    "        varMap[':status'] = 'pending'\n"
    "        sqlA = f'SELECT jediTaskID FROM {schema}.JEDI_Tasks '\n"
    "        sqlA += 'WHERE status=:status '\n"
    "        self.cur.execute(sqlA + comment, varMap)\n"
    "        res = self.cur.fetchall()\n"
    "        kept = []\n"
    "        for (taskID,) in res:\n"
    "            kept.append(taskID)\n"
    "        return kept\n"
)


def _caller(cls: str, body: str, *methods: str, buffer: tuple[str, str] = ("TaskModule", "x")) -> str:
    """``class <cls>`` whose ``start`` is *body*, its ``taskBufferIF`` a *buffer* ``(class, module)``, and *methods*
    defined on it, each reading every element of what it is handed."""
    defined = "".join(
        f"    def {name}(self, value=None):\n        for one in value:\n            if one:\n                pass\n\n"
        for name in methods
    )
    return _typed(f"class {cls}:\n    def start(self):\n{body}\n{defined}", ("taskBufferIF", *buffer))


def test_rows_given_back_cross_to_what_the_caller_does_with_them(tmp_path):
    """The handed-back shape, and the one the request's first round asked for.

    ``getTasksToExecCommand_JEDI@4591`` owns 4573-4603 and works its rows
    through 4600-4621, while the function returns on 4647 -- so "a ``return``
    inside the region", which is how this hole used to be sized, does not
    include it.  Followed as the function runs, the rows reach the return and
    the caller is where the investigation goes next.
    """
    caller = _caller(
        "Commando", "        tasks = self.taskBufferIF.getTasks()\n        self.retryTask(tasks)\n", "retryTask"
    )
    nodes, census = _crossed(tmp_path, ROWS_HANDED_BACK, ["c.py::retryTask"], others=[(caller, "c.py")])

    (node,) = [n for n in nodes if n.anchor.line_start == 7]
    assert node.crosses_to == ["pandaserver/c.py::retryTask"]
    assert (census.handing_back, census.resolved) == (1, 1)


def test_a_caller_reading_the_rows_given_back_is_a_crossing(tmp_path):
    """The tasks go back to ``Commando.start``, a junction's owner, which tests each: that is where they are read
    next, though it hands them to nothing (q052: ``getJobStatisticsByGlobalShare`` back into its caller's check)."""
    caller = _caller(
        "Commando", "        tasks = self.taskBufferIF.getTasks()\n        for one in tasks:\n            if one:\n                pass\n"
    )
    nodes, census = _crossed(tmp_path, ROWS_HANDED_BACK, ["c.py::start"], others=[(caller, "c.py")])

    (node,) = [n for n in nodes if n.anchor.line_start == 7]
    assert node.crosses_to == ["pandaserver/c.py::start"]
    assert (census.handing_back, census.resolved) == (1, 1)


def test_rows_landing_in_an_annotated_name_are_given_back_as_well(tmp_path):
    """``res_list: list[tuple[Any, ...]] = self.cur.fetchall()`` lands the rows as ``res = ...`` does
    (``_get_tasks_datasets_with_unprocessed_inputs``)."""
    caller = _caller(
        "Commando", "        tasks = self.taskBufferIF.getTasks()\n        self.retryTask(tasks)\n", "retryTask"
    )
    annotated = ROWS_HANDED_BACK.replace(
        "        res = self.cur.fetchall()\n", "        res: list = self.cur.fetchall()\n"
    )
    assert annotated != ROWS_HANDED_BACK
    nodes, census = _crossed(tmp_path, annotated, ["c.py::retryTask"], others=[(caller, "c.py")])

    (node,) = [n for n in nodes if n.anchor.line_start == 7]
    assert node.crosses_to == ["pandaserver/c.py::retryTask"]
    assert (census.handing_back, census.resolved) == (1, 1)


@pytest.mark.parametrize(
    "layer",
    ["        return self.module.getTasks()\n", "        res = self.module.getTasks()\n        return res\n"],
    ids=["returned", "bound-then-returned"],
)
def test_rows_handed_back_through_a_layer_reach_the_layers_callers(tmp_path, layer):
    """``JediTaskBuffer.getTasksToExecCommand_JEDI`` is ``return proxy.<same>``; ``TaskBuffer.getConfigValue`` binds
    what ``proxy.getConfigValue`` returns and returns that.

    The layer is not where the rows went, and is never named: it hands them back again, and its callers are read.
    The name had found them through the one name the layers share; placed, the proxy's caller is the layer, and
    stopping there lost 87 of pin k's 266 crossings.
    """
    proxy = _typed(f"class Buffer:\n    def getTasks(self):\n{layer}\n", ("module", "TaskModule", "x"))
    caller = _caller(
        "Commando",
        "        tasks = self.taskBufferIF.getTasks()\n        self.retryTask(tasks)\n",
        "retryTask",
        buffer=("Buffer", "p"),
    )
    nodes, census = _crossed(
        tmp_path, ROWS_HANDED_BACK, ["p.py::getTasks", "c.py::retryTask"], others=[(proxy, "p.py"), (caller, "c.py")]
    )

    (node,) = [n for n in nodes if n.anchor.line_start == 7]
    assert node.crosses_to == ["pandaserver/c.py::retryTask"]
    assert (census.handing_back, census.resolved) == (1, 1)


def test_rows_handed_to_a_started_thread_reach_what_its_run_calls_on_it(tmp_path):
    """``TaskCommando`` starts a ``TaskCommandoThread`` with the tasks; its ``run`` is ``WorkerThread``'s, which owns
    no junction and calls ``self.runImpl()`` -- the thread's.

    Placed for the class the thread was built as, not for every ``WorkerThread``: ``Other.runImpl`` owns one too.
    Read by name, a started ``JobGeneratorThread`` crossed to all six of its methods owning one; ``runImpl`` is the
    one ``run`` calls.
    """
    threads = (
        "import threading\n\n\n"
        "class WorkerThread(threading.Thread):\n"
        "    def run(self):\n"
        "        self.runImpl()\n\n\n"
        "class Commando(WorkerThread):\n"
        "    def __init__(self, tasks):\n"
        "        WorkerThread.__init__(self)\n"
        "        self.tasks = tasks\n\n"
        "    def runImpl(self):\n"
        "        for one in self.tasks:\n"
        "            if one:\n"
        "                pass\n\n"
        "    def finish(self):\n"
        "        for one in self.tasks:\n"
        "            if one:\n"
        "                pass\n"
    )
    other = (
        "from pandaserver.t import WorkerThread\n\n\n"
        "class Other(WorkerThread):\n"
        "    def runImpl(self):\n"
        "        for one in self.tasks:\n"
        "            if one:\n"
        "                pass\n"
    )
    caller = _caller(
        "Starter",
        "        tasks = self.taskBufferIF.getTasks()\n        thr = Commando(tasks)\n        thr.start()\n",
    ) + "from pandaserver.t import Commando\n"
    nodes, _census = _crossed(
        tmp_path,
        ROWS_HANDED_BACK,
        ["t.py::runImpl", "t.py::finish", "o.py::runImpl"],
        others=[(threads, "t.py"), (other, "o.py"), (caller, "c.py")],
    )

    (node,) = [n for n in nodes if n.anchor.line_start == 7]
    assert node.crosses_to == ["pandaserver/t.py::runImpl"]


def test_a_call_on_the_way_no_reading_places_is_said_to_be_open(tmp_path):
    """``ContentsFeeder``'s ``taskSpec`` is unpacked from a call's result and declared nowhere, so
    ``taskSpec.setErrDiag(...)`` is a hole of the reach record -- and the rows may go on there.

    Said on the statement, as the record names the hole, so ``crosses_to`` reads as what is known rather than all.
    A hole whose name no junction owner's function has cannot land at one -- ``spec.debug(tasks)`` -- and is not.
    """
    caller = _caller(
        "Commando",
        "        tasks = self.taskBufferIF.getTasks()\n"
        "        spec = self.unknown\n"
        "        spec.retryTask(tasks)\n"
        "        spec.debug(tasks)\n",
        "retryTask",
        "debug",
    )
    nodes, census = _crossed(tmp_path, ROWS_HANDED_BACK, ["c.py::retryTask"], others=[(caller, "c.py")])

    (node,) = [n for n in nodes if n.anchor.line_start == 7]
    assert node.crosses_to == []
    assert [(one.owner, one.line, one.callee) for one in node.crosses_open] == [
        ("pandaserver/c.py::Commando.start", 5, "spec.retryTask")
    ]
    assert census.opened == 1


def test_a_caller_the_rows_go_back_to_that_no_reading_places_is_said_to_be_open(tmp_path):
    """``insert_step_task``'s task id goes back to ``self.tbif.insert_step_task(...)`` in
    ``PandaTaskStepHandler.submit_target``, its receiver declared ``Any``: no caller :meth:`Calls.callers` finds, and
    the statement said nowhere and open nothing.

    A hole spelled with the returning function's name may be the caller, and is open; one spelled otherwise --
    ``self.unknown.other()`` -- cannot be, nor is a plain name called, which could be any function's caller.
    """
    caller = _caller(
        "Commando",
        "        tasks = self.unknown.getTasks()\n"
        "        self.retryTask(tasks)\n"
        "        self.unknown.other(tasks)\n"
        "        made = self.factory\n"
        "        made(tasks)\n",
        "retryTask",
    )
    nodes, census = _crossed(tmp_path, ROWS_HANDED_BACK, ["c.py::retryTask"], others=[(caller, "c.py")])

    (node,) = [n for n in nodes if n.anchor.line_start == 7]
    assert node.crosses_to == []
    assert [(one.owner, one.line, one.callee) for one in node.crosses_open] == [
        ("pandaserver/c.py::Commando.start", 3, "self.unknown.getTasks")
    ]
    assert (census.handing_back, census.opened) == (1, 1)


def test_a_caller_that_does_nothing_the_map_answers_for_leaves_it_empty(tmp_path):
    """The caller is found and does something with the rows, and none of it is
    somewhere the map holds a junction.  Naming the caller's own function
    would satisfy the question and not the purpose.
    """
    caller = _caller("Commando", "        tasks = self.taskBufferIF.getTasks()\n        print(tasks)\n")
    nodes, census = _crossed(tmp_path, ROWS_HANDED_BACK, ["c.py::somethingElse"], others=[(caller, "c.py")])

    (node,) = [n for n in nodes if n.anchor.line_start == 7]
    assert node.crosses_to == []
    assert (census.handing_back, census.resolved) == (1, 0)


def test_rows_that_never_reach_a_return_do_not_cross_to_the_caller(tmp_path):
    """The function returns, and not these rows: a name they never reach is an answer of no."""
    source = (
        "class TaskModule:\n"
        "    def getTasks(self):\n"
        "        varMap = {}\n"
        "        varMap[':status'] = 'pending'\n"
        "        sqlA = f'SELECT jediTaskID FROM {schema}.JEDI_Tasks '\n"
        "        sqlA += 'WHERE status=:status '\n"
        "        self.cur.execute(sqlA + comment, varMap)\n"
        "        res = self.cur.fetchall()\n"
        "        for (taskID,) in res:\n"
        "            tmpLog.debug(taskID)\n"
        "        return self.counter\n"
    )
    caller = _caller(
        "Commando", "        tasks = self.taskBufferIF.getTasks()\n        self.retryTask(tasks)\n", "retryTask"
    )
    nodes, census = _crossed(tmp_path, source, ["c.py::retryTask"], others=[(caller, "c.py")])

    (node,) = [n for n in nodes if n.anchor.line_start == 7]
    assert node.crosses_to == []
    assert (census.handing_back, census.resolved) == (0, 0)


def test_the_caller_reusing_the_name_gets_only_its_own_rows(tmp_path):
    """``datasetManager.py::main`` runs seven queries into the same ``res``.

    Each is handed to a different thread class, so carried by name one
    statement's rows resolve to all seven.  This is the reuse P1-54 met on the
    exits, met again a function further out; carried from the line the rows
    arrived on, the second ``tasks`` overwrites the first.
    """
    caller = _caller(
        "Manager",
        "        tasks = self.taskBufferIF.getTasks()\n"
        "        self.retryTask(tasks)\n"
        "        tasks = self.taskBufferIF.getOtherTasks()\n"
        "        self.finishTask(tasks)\n",
        "retryTask",
        "finishTask",
    )
    nodes, _census = _crossed(
        tmp_path, ROWS_HANDED_BACK, ["c.py::retryTask", "c.py::finishTask"], others=[(caller, "c.py")]
    )

    (node,) = [n for n in nodes if n.anchor.line_start == 7]
    assert node.crosses_to == ["pandaserver/c.py::retryTask"]


def test_a_status_name_the_next_call_reuses_does_not_take_the_rows_with_it(tmp_path):
    """``ret_val, tasks = ...`` then ``ret_val, other = ...``: ``tasks`` still holds the rows.

    ``AtlasAnalJobBroker.doBrokerage`` reuses ``ret_val`` for every call it
    checks.  Asked whether the arriving write still held with every name's
    writes at once, the second write replaced the first, and the rows that
    arrived in ``analy_sites_class_dict`` crossed nowhere.
    """
    caller = _caller(
        "Manager",
        "        ret_val, tasks = self.taskBufferIF.getTasks()\n"
        "        ret_val, other = self.taskBufferIF.getOtherTasks()\n"
        "        self.retryTask(tasks)\n",
        "retryTask",
    )
    nodes, _census = _crossed(tmp_path, ROWS_HANDED_BACK, ["c.py::retryTask"], others=[(caller, "c.py")])

    (node,) = [n for n in nodes if n.anchor.line_start == 7]
    assert node.crosses_to == ["pandaserver/c.py::retryTask"]


def test_a_link_written_before_the_rows_arrived_is_not_a_lead(tmp_path):
    """``w = x`` above ``x = tasks``: ``w`` holds what ``x`` held then, not the rows.

    Carried by name, ``x`` taking the rows on a later line made every name
    ever built from ``x`` hold them too.  ``AtlasAnalJobBroker.doBrokerage``
    was handed ``setErrDiag`` at 496 that way, from ``failureCounts`` read at
    388, by a chain that runs up the function.
    """
    caller = _caller(
        "Manager",
        "        x = self.make()\n"
        "        w = x\n"
        "        tasks = self.taskBufferIF.getTasks()\n"
        "        x = tasks\n"
        "        self.setErrDiag(w)\n"
        "        self.retryTask(x)\n",
        "make",
        "setErrDiag",
        "retryTask",
    )
    nodes, _census = _crossed(
        tmp_path, ROWS_HANDED_BACK, ["c.py::retryTask", "c.py::setErrDiag"], others=[(caller, "c.py")]
    )

    (node,) = [n for n in nodes if n.anchor.line_start == 7]
    assert node.crosses_to == ["pandaserver/c.py::retryTask"]


def test_rows_handed_back_as_a_dicts_keys_cross_to_what_the_caller_does(tmp_path):
    """``retMap[jediTaskID] = set()`` and the caller loops over ``.items()``.

    ``throttleJobsInPausedTasks_JEDI`` hands its tasks back so, and the
    caller kills each task's jobs.  Following only what is filed, never what
    it is filed under, said these rows reach no ``return``.
    """
    source = (
        "class TaskModule:\n"
        "    def getTasks(self):\n"
        "        varMap = {}\n"
        "        varMap[':status'] = 'pending'\n"
        "        sqlA = f'SELECT jediTaskID FROM {schema}.JEDI_Tasks '\n"
        "        sqlA += 'WHERE status=:status '\n"
        "        self.cur.execute(sqlA + comment, varMap)\n"
        "        res = self.cur.fetchall()\n"
        "        retMap = {}\n"
        "        for (taskID,) in res:\n"
        "            retMap[taskID] = set()\n"
        "        return retMap\n"
    )
    caller = _caller(
        "Commando",
        "        tasks = self.taskBufferIF.getTasks()\n"
        "        for taskID, ids in tasks.items():\n"
        "            self.retryTask(taskID)\n",
        "retryTask",
    )
    nodes, census = _crossed(tmp_path, source, ["c.py::retryTask"], others=[(caller, "c.py")])

    (node,) = [n for n in nodes if n.anchor.line_start == 7]
    assert node.crosses_to == ["pandaserver/c.py::retryTask"]
    assert (census.handing_back, census.resolved) == (1, 1)


def test_a_popped_pair_s_first_is_the_key_the_rows_are_filed_under(tmp_path):
    """``retMap[jediTaskID] = set()`` and the caller takes ``tasks.popitem()``: a (key, value) pair
    (callfacts.ITEM), so the task the rows are filed under is the pair's first, not one of the values."""
    source = (
        "class TaskModule:\n"
        "    def getTasks(self):\n"
        "        varMap = {}\n"
        "        varMap[':status'] = 'pending'\n"
        "        sqlA = f'SELECT jediTaskID FROM {schema}.JEDI_Tasks '\n"
        "        sqlA += 'WHERE status=:status '\n"
        "        self.cur.execute(sqlA + comment, varMap)\n"
        "        res = self.cur.fetchall()\n"
        "        retMap = {}\n"
        "        for (taskID,) in res:\n"
        "            retMap[taskID] = set()\n"
        "        return retMap\n"
    )
    caller = _caller(
        "Commando",
        "        tasks = self.taskBufferIF.getTasks()\n"
        "        taskID, ids = tasks.popitem()\n"
        "        self.retryTask(taskID)\n",
        "retryTask",
    )
    nodes, census = _crossed(tmp_path, source, ["c.py::retryTask"], others=[(caller, "c.py")])

    (node,) = [n for n in nodes if n.anchor.line_start == 7]
    assert node.crosses_to == ["pandaserver/c.py::retryTask"]


def test_a_conversion_of_one_value_is_a_plain_copy_of_it_where_the_name_is_the_builtin():
    """A bind's value ``repr(job.PandaID)`` or ``float(job.PandaID)`` is the field again, as another type
    (callfacts.CONVERTING); ``str(...)`` with a ``str`` the function binds, and ``len(...)``, are computed."""
    tree = ast.parse(
        "def f(job, other):\n"
        "    a = repr(job.PandaID)\n"
        "    b = float(job.PandaID)\n"
        "    c = len(job.PandaID)\n\n"
        "def g(job, str):\n"
        "    d = str(job.PandaID)\n"
    )
    pathcond.attach_parents(tree)
    values = [node.value for node in ast.walk(tree) if isinstance(node, ast.Assign)]
    assert [sqlwrite._plain(value) for value in values] == [True, True, False, False]
    assert [getattr(sqlwrite._root(value), "id", None) for value in values] == ["job", "job", None, None]


def test_one_of_two_values_a_builtin_picks_holds_what_each_holds(tmp_path):
    """``max(taskID, 0)`` is ``taskID`` or ``0`` (callfacts.ONE): the task the rows name reaches the call it is
    handed to, as ``taskID`` itself would; ``next(iter(ids), None)`` is one of the ids or None."""
    source = (
        "class TaskModule:\n"
        "    def getTasks(self):\n"
        "        varMap = {}\n"
        "        varMap[':status'] = 'pending'\n"
        "        sqlA = f'SELECT jediTaskID FROM {schema}.JEDI_Tasks '\n"
        "        sqlA += 'WHERE status=:status '\n"
        "        self.cur.execute(sqlA + comment, varMap)\n"
        "        res = self.cur.fetchall()\n"
        "        return res\n"
    )
    caller = _caller(
        "Commando",
        "        tasks = self.taskBufferIF.getTasks()\n"
        "        (taskID,) = tasks[0]\n"
        "        self.retryTask(max(taskID, 0))\n"
        "        ids = [one for (one,) in tasks]\n"
        "        self.killTask(next(iter(ids), None))\n",
        "retryTask",
        "killTask",
    )
    nodes, census = _crossed(tmp_path, source, ["c.py::retryTask", "c.py::killTask"], others=[(caller, "c.py")])

    (node,) = [n for n in nodes if n.anchor.line_start == 7]
    assert node.crosses_to == ["pandaserver/c.py::killTask", "pandaserver/c.py::retryTask"]


def test_rows_used_only_to_choose_are_not_handed_back(tmp_path):
    """The rows pick which candidates are kept; the candidates are what is returned.

    ``insertFilesForDataset_JEDI`` and ``copyArchive.py::main`` read lists
    only to test membership in, and carried by name the rows reached the
    ``return`` through a loop name used again for the candidates.
    """
    source = (
        "class TaskModule:\n"
        "    def getTasks(self):\n"
        "        varMap = {}\n"
        "        varMap[':status'] = 'pending'\n"
        "        sqlA = f'SELECT jediTaskID FROM {schema}.JEDI_Tasks '\n"
        "        sqlA += 'WHERE status=:status '\n"
        "        self.cur.execute(sqlA + comment, varMap)\n"
        "        res = self.cur.fetchall()\n"
        "        seen = set()\n"
        "        for (taskID,) in res:\n"
        "            seen.add(taskID)\n"
        "        kept = []\n"
        "        for taskID in self.candidates:\n"
        "            if taskID in seen:\n"
        "                kept.append(taskID)\n"
        "        return kept\n"
    )
    caller = _caller(
        "Commando", "        tasks = self.taskBufferIF.getTasks()\n        self.retryTask(tasks)\n", "retryTask"
    )
    nodes, census = _crossed(tmp_path, source, ["c.py::retryTask"], others=[(caller, "c.py")])

    (node,) = [n for n in nodes if n.anchor.line_start == 7]
    assert node.crosses_to == []
    assert (census.handing_back, census.resolved) == (0, 0)


def _site_with(**fields):
    base = {
        "map_id": MAP_ID,
        "derived_from": VERSION,
        "name": "r:1",
        "owner": "x.py::sweep",
        "anchor": Anchor(package="server", file="x.py", line_start=7, line_end=7),
        "region": [3, 20],
    }
    base.update(fields)
    return ReadSiteNode(**base)


def _movers(
    guard: str,
    *,
    selects: list[str],
    stops: list[str] = (),
    conditions: list[str] = (),
    tables: list[str] = ("jobsActive4",),
    stop_statement: int = 7,
    stop_reads: list[str] = ("jobsActive4",),
    row_keys: list[RowKeySource] = (),
) -> MapFragment:
    """A junction guarding on *guard* and one statement, in one function.

    The statement runs on line 7 and reads *tables*; each of *stops* is a stop
    in the statement on *stop_statement* reading *stop_reads* -- line 7 by
    default, so the statement's own node carries it.  Two kinds of row are
    named, jobs and tasks, so a statement on the other kind can be asked about.
    """
    fragment = MapFragment(map_id=MAP_ID, derived_from=VERSION)
    fragment.subjects.append(
        SubjectNode(
            map_id=MAP_ID,
            derived_from=VERSION,
            name="JobSpec.jobStatus",
            spec_class="JobSpec",
            attribute="jobStatus",
        )
    )
    fragment.entities.extend(
        EntityNode(
            name=table.lower(),
            map_id=MAP_ID,
            derived_from=VERSION,
            spec_class=spec,
            tables=[table],
        )
        for table, spec in (("jobsActive4", "JobSpec"), ("JEDI_Tasks", "JediTaskSpec"))
    )
    fragment.junctions.append(
        JunctionNode(
            map_id=MAP_ID,
            derived_from=VERSION,
            name="j:getJobs",
            subject="JobSpec.jobStatus",
            owner="x.py::getJobs",
            branches=[Branch(outcome="sent", row_precondition=[guard], order=0)],
        )
    )
    fragment.read_sites.append(
        _site_with(
            owner="x.py::getJobs",
            selects=selects,
            conditions=list(conditions),
            tables=list(tables),
            unresolved=list(stops) if stop_statement == 7 else [],
        )
    )
    fragment.reading_stops.extend(
        ReadingStop(
            reason="a fragment of the statement has no text of its own",
            detail=stop,
            file="x.py",
            owner="x.py::getJobs",
            line=stop_statement - 2,
            statement=stop_statement,
            reads=list(stop_reads),
        )
        for stop in stops
    )
    fragment.row_key_sources.extend(row_keys)
    return fragment


def _row_key(handed_by: str = "", statement: int = 30) -> RowKeySource:
    """The guarded ``UPDATE`` on line *statement*, its key filled by *handed_by*."""
    return RowKeySource(
        owner="x.py::getJobs",
        file="x.py",
        statement=statement,
        columns=["jobStatus"],
        keys=["PandaID=:PandaID"],
        handed_by=handed_by,
    )


def test_gate_catches_a_function_that_guards_on_a_value_and_asks_for_nothing():
    """The map holds a statement for the function, read it in full, and that
    statement asks for nothing at all -- so the junction's guard is the only
    thing in the map that knows which rows this function touches.
    """
    fragment = _movers("jobStatus=activated", selects=[])

    result = gates.movers_of_a_value_agree(fragment)

    assert not result.passed
    assert "x.py::getJobs" in result.failures[0]
    assert "JobSpec.jobStatus=activated" in result.failures[0]
    assert gates.MOVER_ASKS_NOTHING in result.failures[0]


def test_a_function_whose_query_on_these_rows_asks_by_other_terms_is_an_answer():
    """``removeFilesIndexInconsistent_JEDI`` reads its files for ``lost`` and
    requires ``ready`` of the one it writes.

    The status is the compare-and-set, not what the rows were asked for by,
    and every statement on those rows was read -- so the map can say why the
    two lists differ, and it is not a gap.
    """
    fragment = _movers(
        "jobStatus=activated", selects=[], conditions=["jediTaskID=:jediTaskID@9"]
    )

    result = gates.movers_of_a_value_agree(fragment)
    (difference,) = gates.mover_differences(fragment)

    assert result.passed
    assert difference.reason == gates.MOVER_OTHER_TERMS
    assert difference.where == "@7"


def test_a_statement_read_only_in_part_is_a_gap_and_fails_the_gate():
    """The same difference, with the map's own statement read in part.

    This used to pass as "a reason", with a count in the note.  It is a
    reason -- and it is the map's, not PanDA's: what the statement selects on
    is not all of what it selects on, so the value the guard requires may be
    exactly the part nobody read.  A work item, and named as one.
    """
    fragment = _movers(
        "jobStatus=activated",
        selects=["JobSpec.jobStatus=holding"],
        stops=["build_where_clause()"],
    )

    result = gates.movers_of_a_value_agree(fragment)

    assert not result.passed
    assert gates.MOVER_PARTIAL in result.failures[0]
    assert "(@7)" in result.failures[0]


def test_a_stop_in_a_statement_with_no_node_names_that_statement():
    """``getJobs``: the ``SELECT`` that picks the pilot's jobs runs on line 2391,
    its ``WHERE`` is a fragment nobody could render, and so it has no node --
    while the function's other query, on ``sent``, was read in full.

    Judged per function, the full read of the *other* statement excused it.
    Judged per statement, the one that stopped is named with what stopped it.
    """
    fragment = _movers(
        "jobStatus=activated",
        selects=["JobSpec.jobStatus=sent"],
        conditions=["modificationTime>:modificationTime@5"],
        stops=["sql_where_clause", "sorting_sql"],
        stop_statement=2391,
    )

    result = gates.movers_of_a_value_agree(fragment)
    (difference,) = gates.mover_differences(fragment)

    assert not result.passed
    assert difference.reason == gates.MOVER_UNNODED
    assert difference.where == "@2391 sql_where_clause, sorting_sql"


def test_a_stop_in_a_write_says_nothing_about_which_rows_were_picked_up():
    """A precondition whose bind nothing fills, or an ``INSERT``'s
    ``bindValuesExpression()``, is a gap in reading a write.

    Seven of the eight functions the gate used to excuse as "read only in
    part" had stopped only there.  A stop that reads no table neither excuses
    a difference nor accuses one.
    """
    fragment = _movers(
        "jobStatus=activated",
        selects=["JobSpec.jobStatus=sent"],
        stops=["FileSpec.bindValuesExpression(useSeq=True)"],
        stop_statement=40,
        stop_reads=[],
    )

    (difference,) = gates.mover_differences(fragment)

    assert difference.reason == gates.MOVER_OTHER_TERMS


def test_a_statement_on_another_kind_of_row_does_not_decide():
    """``throttleJobsInPausedTasks_JEDI`` asks for paused *tasks* and requires
    ``activated`` of the *jobs* it writes.

    The task query was read in full, and it is not a statement about jobs, so
    it answers nothing about which jobs were picked up.  What the map can say
    is that it holds no query here on jobs at all -- and it says that, rather
    than the key it cannot see.
    """
    fragment = _movers(
        "jobStatus=activated",
        selects=["JediTaskSpec.status=paused"],
        tables=["JEDI_Tasks"],
        row_keys=[_row_key()],
    )

    result = gates.movers_of_a_value_agree(fragment)
    (difference,) = gates.mover_differences(fragment)

    assert result.passed
    assert difference.reason == gates.MOVER_OWN_WHERE
    assert difference.where == "@30"


def test_a_stop_on_a_table_the_map_cannot_place_counts_as_a_gap():
    """A table no entity names could be any kind of row, so a stop in a query
    on it is not allowed to be read as "about something else".  The direction
    is chosen: a stop the map cannot place errs towards being called a gap.
    """
    fragment = _movers(
        "jobStatus=activated",
        selects=[],
        conditions=["jediTaskID=:jediTaskID@9"],
        stops=["build_where_clause()"],
        stop_statement=90,
        stop_reads=["{}.jobs"],
    )

    (difference,) = gates.mover_differences(fragment)

    assert difference.reason == gates.MOVER_UNNODED


def test_a_query_joining_a_table_the_map_cannot_place_is_not_taken_for_one_on_these_rows():
    """``throttleJobsInPausedTasks_JEDI`` asks for paused tasks through a join
    with ``JEDI_AUX_Status_MinTaskID``, which no entity names.

    An unplaced table may be anything, and for a *gap* that is enough to count
    it.  For an answer it is not: "every query here on jobs was read in full
    and asks by other terms" is a claim about a job query, and the function
    runs none.
    """
    fragment = _movers(
        "jobStatus=activated",
        selects=["JediTaskSpec.status=paused"],
        tables=["JEDI_AUX_Status_MinTaskID", "JEDI_Tasks"],
        row_keys=[_row_key()],
    )

    (difference,) = gates.mover_differences(fragment)

    assert difference.reason == gates.MOVER_OWN_WHERE


def test_a_statement_the_caller_hands_in_is_a_gap_and_fails_the_gate():
    """``queryTasksToBePending_JEDI`` runs the ``SELECT`` its caller builds.

    Its tables are not known here, so an empty ``reads`` would have read as a
    write and excused it as "nothing here reads these rows".  A statement
    that may read anything is counted as one that may read these.
    """
    fragment = _movers("jobStatus=activated", selects=[], tables=["JEDI_Tasks"])
    fragment.read_sites.clear()
    fragment.reading_stops.append(
        ReadingStop(
            reason=sql.SKIP_HANDED_IN,
            detail="sql_query",
            file="x.py",
            owner="x.py::getJobs",
            line=12,
            statement=12,
            reads_known=False,
        )
    )

    result = gates.movers_of_a_value_agree(fragment)
    (difference,) = gates.mover_differences(fragment)

    assert not result.passed
    assert difference.reason == gates.MOVER_HANDED_IN
    assert difference.where == "@12 sql_query"


def test_a_statement_read_at_the_callers_sends_the_reader_there_and_passes_the_gate():
    """Read through a corpus, ``queryTasksToBePending_JEDI`` is a runner: the ``SELECT`` it is handed is read where
    its callers run it.  Not a gap, and not "no query here reads these rows" either -- one does, and it is theirs."""
    fragment = _movers("jobStatus=activated", selects=[], tables=["JEDI_Tasks"])
    fragment.read_sites.clear()
    fragment.reading_stops.append(
        ReadingStop(
            reason=sql.SKIP_CALLERS,
            detail="sql_query",
            file="x.py",
            owner="x.py::getJobs",
            line=12,
            statement=12,
            reads_known=False,
        )
    )

    result = gates.movers_of_a_value_agree(fragment)
    (difference,) = gates.mover_differences(fragment)

    assert result.passed
    assert (difference.reason, difference.where) == (gates.MOVER_CALLERS_QUERY, "@12 sql_query")


def test_rows_named_by_what_the_caller_hands_in_send_the_reader_to_the_caller():
    """``setCloudToTasks_JEDI`` binds its row key from the map its caller passed.

    An answer, and a different one from a write that picks its own rows: the
    rows it moves are whichever the caller chose, so the caller is where to go.
    """
    fragment = _movers(
        "jobStatus=activated",
        selects=[],
        tables=["JEDI_Tasks"],
        row_keys=[_row_key("jobs"), _row_key("", statement=40)],
    )

    result = gates.movers_of_a_value_agree(fragment)
    (difference,) = gates.mover_differences(fragment)

    assert result.passed
    assert difference.reason == gates.MOVER_HANDED_ROWS
    assert difference.where == "by jobs"


def test_a_row_key_on_another_column_does_not_decide():
    """A key recorded for a write guarding a different column says nothing here."""
    other = _row_key("jobs")
    other.columns = ["lockedBy"]
    fragment = _movers(
        "jobStatus=activated", selects=[], tables=["JEDI_Tasks"], row_keys=[other]
    )

    (difference,) = gates.mover_differences(fragment)

    assert difference.reason == gates.MOVER_UNDECIDED


@pytest.mark.parametrize(
    "term, expected",
    [
        ("jobStatus=activated", ("jobstatus", ["activated"])),
        ("status='ready'", ("status", ["ready"])),
        ("tab.jobStatus=activated", ("jobstatus", ["activated"])),
        ("keepTrack=1", ("keeptrack", ["1"])),
        ("jobStatus IN (activated,throttled)", ("jobstatus", ["activated", "throttled"])),
        ("status IN ('ready','running','scouting')", ("status", ["ready", "running", "scouting"])),
        ("jobStatus=:oldJobStatus", None),
        ("status IN (:st1,:st2)", None),
        ("status IN ({})", None),
        ("NOT jobStatus=:ngStatus", None),
        ("status<>missing", None),
        ("status NOT IN (a,b)", None),
        ("NOT status IN (a,b)", None),
        ("cloud IS NULL", None),
        ("modificationTime>=x", None),
    ],
)
def test_a_guard_is_read_for_the_values_it_requires(term, expected):
    """A negation lands on a row holding almost anything, so it requires no
    value; a bind, a hole and a comparison have no value to read off."""
    assert required_values(term) == expected


def test_a_guard_spelled_as_a_list_is_compared_too():
    """``getTasksToExecCommand_JEDI`` requires ``jobStatus IN (activated,
    throttled)``, and the gate matched ``col=value`` alone -- so a write that
    moves activated jobs on a command was a mover of ``activated`` nowhere."""
    fragment = _movers("jobStatus IN (activated,throttled)", selects=[])

    pairs = sorted((d.subject, d.value) for d in gates.mover_differences(fragment))

    assert pairs == [("JobSpec.jobStatus", "activated"), ("JobSpec.jobStatus", "throttled")]


def test_gate_catches_a_lead_to_a_junction_the_map_does_not_hold():
    """This gate exists because the resolution ran at the wrong moment.

    ``attach_crossings`` first ran after the junctions were merged but before
    promotion, so every ``self.cur.execute(...)`` inside a region resolved to
    the ``execute`` of an adder plugin that promotion then dropped: 520
    crossings over 239 statements where 71 over 38 are real.  Every unit test
    passed and the diff called it content.
    """
    fragment = MapFragment(
        map_id=MAP_ID,
        derived_from=VERSION,
        read_sites=[_site_with(crosses_to=["pandaserver/dataservice/adder_gen.py::execute"])],
        junctions=[_junction("x.py::retryTask")],
    )

    result = gates.statement_leads_resolve(fragment)

    assert not result.passed
    assert "holds no junction for" in result.failures[0]


def test_gate_catches_an_exit_outside_the_statements_own_region():
    """A line outside the region answers for a different statement.

    ``copyArchive.py::main`` runs three ``holding`` queries in 1353 lines; an
    exit attributed across the boundary tells a reader their row was dropped
    by a query that never saw it.
    """
    fragment = MapFragment(
        map_id=MAP_ID,
        derived_from=VERSION,
        read_sites=[_site_with(exits=["continue@390"])],
        junctions=[_junction("x.py::retryTask")],
    )

    result = gates.statement_leads_resolve(fragment)

    assert not result.passed
    assert "outside both its region" in result.failures[0]


def test_gate_accepts_an_exit_in_the_loop_that_outlives_the_region():
    """The region ends where the next statement's binds start being filled.

    A row loop opened inside it runs to its own end, and for 64 of the
    corpus's 151 looping statements that end is past the region --
    ``getTasksToExecCommand_JEDI@4591`` owns 4585-4603 and works its rows
    through 4600-4621 because another query opens on 4606, inside the loop.
    Judged against the region alone, its only two exits read as foreign.
    """
    fragment = MapFragment(
        map_id=MAP_ID,
        derived_from=VERSION,
        read_sites=[
            _site_with(region=[3, 20], rows_handled=[18, 40], exits=["continue@33"])
        ],
        junctions=[_junction("x.py::retryTask")],
    )

    result = gates.statement_leads_resolve(fragment)

    assert result.passed


def test_gate_passes_a_statement_that_leads_nowhere():
    """Not a coverage gate: most statements have no exit and no crossing.

    "The rows go nowhere in particular" is the answer a reader needs when it
    is true, so an empty field is never a failure here.
    """
    fragment = MapFragment(
        map_id=MAP_ID,
        derived_from=VERSION,
        read_sites=[_site_with(), _site_with(name="r:2", exits=["continue@11"])],
        junctions=[_junction("x.py::retryTask")],
    )

    result = gates.statement_leads_resolve(fragment)

    assert result.passed
    assert result.checked == 2


def test_gate_fails_an_open_crossing_the_reach_record_holds_no_hole_for():
    """``crosses_open`` says "the rows may go on here", and the why is the reach record's: an entry naming no hole
    of it is a claim no reader can look up."""
    from bamboo.codemap.models import OpenCrossing, ReachHole, ReachNode

    hole = ReachHole(line=5, callee="spec.retryTask", why="the receiver's class is not declared")
    row = ReachNode(
        name="reach:c", map_id=MAP_ID, derived_from=VERSION, owner="c.py::Commando.start", function="start", holes=[hole]
    )
    held = OpenCrossing(owner="c.py::Commando.start", line=5, callee="spec.retryTask")
    stray = OpenCrossing(owner="c.py::Commando.start", line=6, callee="spec.retryTask")

    passed, failed = (
        gates.statement_leads_resolve(
            MapFragment(
                map_id=MAP_ID,
                derived_from=VERSION,
                read_sites=[_site_with(crosses_open=[one])],
                reach_graph=[row],
            )
        )
        for one in (held, stray)
    )

    assert passed.passed
    assert not failed.passed and "spec.retryTask at c.py::Commando.start:6" in failed.failures[0]


def test_a_read_site_lists_every_value_one_statement_asks_rows_by():
    """One row per statement, not per pair -- 189 of the corpus's 264 read sites
    ask on more than one, and keying per pair would put several nodes on one
    line for ``map-identities-are-distinct`` to reject."""
    source = (
        "class TaskModule:\n"
        "    def sweep(self):\n"
        "        varMap = {}\n"
        "        varMap[':status'] = 'pending'\n"
        "        sqlA = f'SELECT jediTaskID FROM {schema}.JEDI_Tasks '\n"
        "        sqlA += \"WHERE status=:status AND vo IN ('atlas','test') \"\n"
        "        self.cur.execute(sqlA + comment, varMap)\n"
        "    def touch(self, jediTaskID):\n"
        "        varMap2 = {}\n"
        "        varMap2[':status'] = 'ready'\n"
        # Two columns, because ``status`` alone is ambiguous -- ``FileSpec``,
        # ``DatasetSpec`` and ``JediFileSpec`` declare one too -- and an
        # ambiguous table is left unattributed, which spells the subject by its
        # table instead of its class.
        "        sqlU = f'UPDATE {schema}.JEDI_Tasks SET status=:status,oldStatus=NULL '\n"
        "        sqlU += 'WHERE jediTaskID=:jediTaskID '\n"
        "        self.cur.execute(sqlU + comment, varMap2)\n"
    )
    modules = [_module(_SPECS, "pandaserver/taskbuffer/Specs.py"), _module(source, "x.py")]
    attributor = attribution.SpecAttributor(
        progress.spec_attributes(modules), attribution.class_bases(modules)
    )
    attributor.learn_table_classes(modules)

    (node,) = sqlwrite.read_site_nodes(
        sqlwrite.read_side(modules, attributor).statements, MAP_ID, VERSION
    )

    assert node.selects == [
        "JEDI_Tasks.vo=atlas",
        "JEDI_Tasks.vo=test",
        "JediTaskSpec.status=pending",
    ]


def test_a_reader_named_with_no_statement_to_read_it_at_is_reported():
    """The whole of the read side until the statement became a node: the map
    names the function and a function is not somewhere to be sent."""
    fragment = _fragment_with(("JediTaskSpec", "status", [("ready", 1)]))
    fragment.subjects[0].selected_by = {"finishing": ["x.py::sweep"]}

    result = gates.readers_have_a_statement(fragment)

    assert not result.passed
    assert result.checked == 1
    assert result.failures == [
        "x.py::sweep is named as selecting JediTaskSpec.status=finishing"
        " -- no statement recorded"
    ]


def test_a_reader_whose_statement_the_map_holds_says_nothing():
    fragment = _fragment_with(("JediTaskSpec", "status", [("ready", 1)]))
    fragment.subjects[0].selected_by = {"finishing": ["x.py::sweep"]}
    fragment.read_sites = [
        ReadSiteNode(
            map_id=MAP_ID,
            derived_from=VERSION,
            name=ReadSiteNode.make_name(MAP_ID, "x.py::sweep", 7),
            owner="x.py::sweep",
            selects=["JediTaskSpec.status=finishing"],
            anchor=Anchor(package="pandaserver", file="x.py", line_start=7, line_end=7),
        )
    ]

    assert gates.readers_have_a_statement(fragment).passed


@pytest.mark.asyncio
async def test_the_read_site_kind_is_on_every_list_that_has_to_know_about_it():
    """``log_sites`` was once missing from ``_index`` and all 294 of them
    compared as absent from both builds.  Several lists have to know about a
    kind, and the two that matter most fail silently: a kind the reader does not
    take is read as nothing, and a kind ``_index`` skips compares equal to
    itself however much it changed."""
    from bamboo.codemap import lookup

    assert NodeType.READ_SITE.value in diff.CONTENT_FIELDS
    assert lookup._MODELS[ReadSiteNode] is NodeType.READ_SITE
    assert "read_sites" in MapFragment.model_fields

    site = ReadSiteNode(
        map_id=MAP_ID,
        derived_from=VERSION,
        name=ReadSiteNode.make_name(MAP_ID, "x.py::sweep", 7),
        owner="x.py::sweep",
        selects=["JediTaskSpec.status=finishing"],
        anchor=Anchor(package="pandaserver", file="x.py", line_start=7, line_end=7),
    )

    # The reader takes it.
    assert [node for node, _ in lookup._fragment_nodes(
        MapFragment(map_id=MAP_ID, derived_from=VERSION, read_sites=[site]), NodeType.READ_SITE
    )] == [site]

    # And ``_index`` reaches it, so a change to one is a change the diff reports.
    fragment = MapFragment(map_id=MAP_ID, derived_from=VERSION, read_sites=[site])
    other = fragment.model_copy(deep=True)
    other.read_sites[0].selects = ["JediTaskSpec.status=pending"]

    result = diff.compare(fragment, other)

    assert [(c.field, c.before, c.after) for c in result.changes] == [
        ("selects", "[JediTaskSpec.status=finishing]", "[JediTaskSpec.status=pending]")
    ]


def test_a_where_clause_says_which_values_something_acts_on():
    """The other half of the same statements, and the half that makes it a graph."""
    source = (
        "class TaskModule:\n"
        "    def find(self, vo):\n"
        "        varMap = {}\n"
        "        varMap[':oldStatus'] = 'pending'\n"
        "        sqlU = f'UPDATE {schema}.JEDI_Tasks '\n"
        "        sqlU += 'SET status=:status,oldStatus=NULL '\n"
        "        sqlU += \"WHERE status=:oldStatus AND vo IN ('atlas','test') \"\n"
        "        self.cur.execute(sqlU + comment, varMap)\n"
    )
    modules = [_module(_SPECS, "pandaserver/taskbuffer/Specs.py"), _module(source, "x.py")]
    attributor = attribution.SpecAttributor(
        progress.spec_attributes(modules), attribution.class_bases(modules)
    )
    attributor.learn_table_classes(modules)

    selected = sqlwrite.read_side(modules, attributor).values
    assert set(selected["JediTaskSpec.status"]) == {"pending"}
    assert set(selected["JEDI_Tasks.vo"]) == {"atlas", "test"}


def test_the_query_that_selects_a_value_is_named_and_not_only_the_value():
    """Which function asked is in hand while the value is read, and was dropped.

    "Something selects ``finishing``" is half an answer; the other half is that
    exactly one query does, and where its diagnostics land.  Discarding the
    enclosing function left ``FollowUp`` pooling triggers over the *writers* of
    the subject as a stand-in -- an approximation its own docstring named.
    """
    source = (
        "class TaskModule:\n"
        "    def write(self):\n"
        "        sqlU = f'UPDATE {schema}.JEDI_Tasks SET status=:status,oldStatus=NULL '\n"
        "        self.cur.execute(sqlU + comment, varMap)\n"
        "    def rescue(self, vo):\n"
        "        varMap = {}\n"
        "        varMap[':oldStatus'] = 'pending'\n"
        "        sqlO = f'SELECT jediTaskID FROM {schema}.JEDI_Tasks '\n"
        "        sqlO += 'WHERE status=:oldStatus '\n"
        "        self.cur.execute(sqlO + comment, varMap)\n"
    )
    modules = [_module(_SPECS, "pandaserver/taskbuffer/Specs.py"), _module(source, "x.py")]
    attributor = attribution.SpecAttributor(
        progress.spec_attributes(modules), attribution.class_bases(modules)
    )
    attributor.learn_table_classes(modules)

    selected = sqlwrite.read_side(modules, attributor).values

    assert selected["JediTaskSpec.status"]["pending"].selected_by == {"x.py::rescue"}


def test_a_bind_from_a_declared_mapping_is_a_value_something_selects_on():
    """The read side settles a bind the way the write side already did.

    ``varMap[":status"] = taskStatusMap["doing"]`` is the orphan-rescue query.
    Taking only literals here left every status a command passes through
    looking like a value no query asks for -- a state a task enters and never
    leaves -- while the write side resolved the same mapping through the other
    spelling.  One map, two slices, one fact, two answers.
    """
    source = _IMPORTS_TASK_SPEC + (
        "class TaskModule:\n"
        "    def rescue(self):\n"
        "        sqlT = f'UPDATE {schema}.JEDI_Tasks SET oldStatus=:oldStatus '\n"
        "        self.cur.execute(sqlT + comment, varMap)\n"
        "        commandStatusMap = JediTaskSpec.commandStatusMap()\n"
        "        for commandStr, taskStatusMap in commandStatusMap.items():\n"
        "            varMap = {}\n"
        "            varMap[':status'] = taskStatusMap['doing']\n"
        "            sqlO = f'SELECT jediTaskID FROM {schema}.JEDI_Tasks '\n"
        "            sqlO += 'WHERE status=:status '\n"
        "            self.cur.execute(sqlO + comment, varMap)\n"
    )
    modules = [
        _module(_COMMAND_MAP, "pandaserver/taskbuffer/JediTaskSpec.py"),
        _module(source, "pandaserver/taskbuffer/x.py"),
    ]
    attributor = attribution.SpecAttributor(
        progress.spec_attributes(modules), attribution.class_bases(modules)
    )
    attributor.learn_table_classes(modules)

    with _with_calls(modules):
        selected = sqlwrite.read_side(modules, attributor).values

    assert set(selected["JediTaskSpec.status"]) == {"aborting", "finishing", "paused"}


#: What teaches the attributor that ``jobsActive4`` holds a ``JobSpec``:
#: ``jobStatus`` is declared by that class and no other.
_JOB_TABLE_WRITER = (
    "class JobModule:\n"
    "    def archive(self):\n"
    "        sqlA = f'UPDATE {schema}.jobsActive4 SET jobStatus=:jobStatus '\n"
    "        self.cur.execute(sqlA + comment, varMap)\n"
)

#: The same for ``JEDI_Tasks``.  Needed wherever a test turns on two kinds of
#: row being attributable at once: without it the second table resolves to no
#: class and the assertion passes because nothing was read, not because the
#: reading was withheld.
_TASK_TABLE_WRITER = (
    "class TaskTableModule:\n"
    "    def touch(self):\n"
    "        sqlT = f'UPDATE {schema}.JEDI_Tasks SET oldStatus=:oldStatus '\n"
    "        self.cur.execute(sqlT + comment, varMap)\n"
)


def test_a_query_naming_no_promoted_column_still_says_whose_rows_it_read():
    """The half a pair of class and attribute cannot carry.

    ``SELECT PandaID FROM jobsActive4 WHERE jediTaskID=:jediTaskID`` selects a
    task's jobs on the join key alone.  No promoted attribute appears, so the
    value reading is empty and the map read the helper as observing nothing --
    when it observes jobs, and the class was already in hand here.
    """
    source = (
        "class TaskModule:\n"
        "    def rows(self, jediTaskID):\n"
        "        varMap = {}\n"
        "        varMap[':jediTaskID'] = jediTaskID\n"
        "        sqlP = f'SELECT PandaID FROM {schema}.jobsActive4 '\n"
        "        sqlP += 'WHERE jediTaskID=:jediTaskID '\n"
        "        self.cur.execute(sqlP + comment, varMap)\n"
    )
    modules = [
        _module(_SPECS, "pandaserver/taskbuffer/Specs.py"),
        # A table's class is learned from what is written to it, never from a
        # SELECT -- a query's column list belongs as often to a joined table as
        # to the one after FROM.  So the corpus has to state it somewhere for
        # the read to be attributable at all.
        _module(_JOB_TABLE_WRITER, "teacher.py"),
        _module(source, "x.py"),
    ]
    attributor = attribution.SpecAttributor(
        progress.spec_attributes(modules), attribution.class_bases(modules)
    )
    attributor.learn_table_classes(modules)

    side = sqlwrite.read_side(modules, attributor)

    assert side.entities["jobsactive4"].read_by == {"x.py::rows"}
    assert side.entities["jobsactive4"].tables == {"jobsActive4"}


def test_the_predicate_of_an_update_does_not_make_the_function_a_reader():
    """Selecting a task's jobs and updating them are different claims.

    Pooling the verbs is the conflation that lets an ``UPDATE ... WHERE`` be
    reported as a query that selects on a value -- and a descent follows the
    rows a decision was read *from*, so a writer counted as a reader sends the
    walk forwards while it says it is stepping down.
    """
    source = (
        "class TaskModule:\n"
        "    def kill(self, jediTaskID):\n"
        "        varMap = {}\n"
        "        varMap[':jediTaskID'] = jediTaskID\n"
        "        sqlU = f'UPDATE {schema}.jobsActive4 SET jobStatus=:jobStatus '\n"
        "        sqlU += 'WHERE jediTaskID=:jediTaskID '\n"
        "        self.cur.execute(sqlU + comment, varMap)\n"
    )
    modules = [_module(_SPECS, "pandaserver/taskbuffer/Specs.py"), _module(source, "x.py")]
    attributor = attribution.SpecAttributor(
        progress.spec_attributes(modules), attribution.class_bases(modules)
    )
    attributor.learn_table_classes(modules)

    side = sqlwrite.read_side(modules, attributor)

    assert side.entities["jobsactive4"].updated_by == {"x.py::kill"}
    assert not side.entities["jobsactive4"].read_by


def test_an_update_predicate_does_not_name_a_query_that_selects_the_value():
    """Who *asks for* a value and who *acts on rows already holding it* differ.

    ``UPDATE jobsActive4 SET ... WHERE jobStatus=:jobStatus`` says which rows
    the statement was willing to change; it is not a query anything could have
    failed to pick the row up in.  Reported as a selection it sends a reader to
    a function that never asked, and one hop out of a junction it opens a
    descent on the strength of a write.
    """
    source = (
        "class TaskModule:\n"
        "    def reassign(self):\n"
        "        varMap = {}\n"
        "        varMap[':jobStatus'] = 'activated'\n"
        "        sqlU = f'UPDATE {schema}.jobsActive4 SET PandaID=:PandaID '\n"
        "        sqlU += 'WHERE jobStatus=:jobStatus '\n"
        "        self.cur.execute(sqlU + comment, varMap)\n"
    )
    modules = [_module(_SPECS, "pandaserver/taskbuffer/Specs.py"), _module(source, "x.py")]
    attributor = attribution.SpecAttributor(
        progress.spec_attributes(modules), attribution.class_bases(modules)
    )
    attributor.learn_table_classes(modules)

    use = sqlwrite.read_side(modules, attributor).values["JobSpec.jobStatus"]["activated"]

    assert use.updated_by == {"x.py::reassign"}
    assert not use.selected_by


def test_a_value_only_an_update_acts_on_is_still_not_a_sink():
    """The two readings of a ``WHERE`` part company here, and both are right.

    "Does anything move a task out of this status?" is answered by an
    ``UPDATE ... WHERE status=:old`` as squarely as by a query -- so the value
    stays in ``selected_values`` and the sink report keeps its meaning.  "Who
    has to pick this row up, so who to ask why they did not?" is not, and that
    is the reading that must not be told a writer is a reader.
    """
    source = (
        "class TaskModule:\n"
        "    def release(self):\n"
        "        varMap = {}\n"
        "        varMap[':oldStatus'] = 'pending'\n"
        "        sqlU = f'UPDATE {schema}.JEDI_Tasks '\n"
        "        sqlU += 'SET status=:status,oldStatus=NULL '\n"
        "        sqlU += 'WHERE status=:oldStatus '\n"
        "        self.cur.execute(sqlU + comment, varMap)\n"
    )
    modules = [_module(_SPECS, "pandaserver/taskbuffer/Specs.py"), _module(source, "x.py")]
    attributor = attribution.SpecAttributor(
        progress.spec_attributes(modules), attribution.class_bases(modules)
    )
    attributor.learn_table_classes(modules)

    values = sqlwrite.read_side(modules, attributor).values["JediTaskSpec.status"]

    assert set(values) == {"pending"}
    assert values["pending"].updated_by == {"x.py::release"}
    assert not values["pending"].selected_by


def test_one_query_naming_two_kinds_of_row_records_the_join():
    """A single FROM list is the corpus stating the relation itself.

    Which datasets a task is waiting on cannot be read off two separate facts
    about the same function -- that is the join this corpus has charged for
    twice -- but it can be read off one statement that asks for both.
    """
    source = (
        "class TaskModule:\n"
        "    def finish(self):\n"
        "        sqlF = f'SELECT tabT.jediTaskID FROM {schema}.JEDI_Tasks tabT,'\n"
        "        sqlF += '{schema}.JEDI_Datasets tabD '\n"
        "        sqlF += \"WHERE tabT.status=:status AND tabD.proc_status='ready' \"\n"
        "        self.cur.execute(sqlF + comment, varMap)\n"
    )
    modules = [
        _module(_SPECS, "pandaserver/taskbuffer/Specs.py"),
        # Both tables have to be attributable, and a class is only ever learned
        # from what is written to a table.
        _module(
            "class W:\n"
            "    def w(self):\n"
            "        s1 = f'UPDATE {schema}.JEDI_Tasks SET oldStatus=:oldStatus '\n"
            "        self.cur.execute(s1 + comment, varMap)\n"
            "        s2 = f'UPDATE {schema}.JEDI_Datasets SET proc_status=:proc '\n"
            "        self.cur.execute(s2 + comment, varMap)\n",
            "teacher.py",
        ),
        _module(source, "x.py"),
    ]
    attributor = attribution.SpecAttributor(
        progress.spec_attributes(modules), attribution.class_bases(modules)
    )
    attributor.learn_table_classes(modules)

    joins = sqlwrite.read_side(modules, attributor).joins

    assert joins["x.py::finish"] == {frozenset({"JediTaskSpec", "JediFileSpec"})}


def test_two_kinds_of_row_read_by_two_statements_are_not_joined():
    """Co-residence in a function is not a relation, and saying so is the point.

    The tempting reading -- this function reads tasks and it reads jobs, so the
    task's jobs are where this goes next -- opens 113 junctions against the 31 a
    shared statement opens, and a method dispatching several commands reads one
    kind of row in one arm and decides about another somewhere else entirely.
    """
    source = (
        "class TaskModule:\n"
        "    def dispatch(self):\n"
        "        sqlT = f'SELECT jediTaskID FROM {schema}.JEDI_Tasks '\n"
        "        sqlT += 'WHERE status=:status '\n"
        "        self.cur.execute(sqlT + comment, varMap)\n"
        "        sqlJ = f'SELECT PandaID FROM {schema}.jobsActive4 '\n"
        "        sqlJ += 'WHERE jediTaskID=:jediTaskID '\n"
        "        self.cur.execute(sqlJ + comment, varMap)\n"
    )
    modules = [
        _module(_SPECS, "pandaserver/taskbuffer/Specs.py"),
        _module(_JOB_TABLE_WRITER, "teacher.py"),
        _module(_TASK_TABLE_WRITER, "teacher2.py"),
        _module(source, "x.py"),
    ]
    attributor = attribution.SpecAttributor(
        progress.spec_attributes(modules), attribution.class_bases(modules)
    )
    attributor.learn_table_classes(modules)

    side = sqlwrite.read_side(modules, attributor)

    # Both kinds of row really are attributable here -- what is withheld is the
    # relation between them, not the reading of either.
    assert side.entities["jobsactive4"].read_by == {"x.py::dispatch"}
    assert "x.py::dispatch" in side.entities["jedi_tasks"].read_by
    assert side.joins == {}


def test_a_statement_that_writes_one_kind_of_row_and_reads_another_is_not_a_join():
    """A join is between rows one query *asked for*, and only that.

    ``UPDATE JEDI_Tasks ... WHERE jediTaskID IN (SELECT ... FROM jobsActive4)``
    names two kinds of row, but it reads one and writes the other, and a
    descent follows the rows a decision was taken *from* -- pooling the verbs
    here is the same conflation ``read_by`` and ``written_by`` were split for.
    Held to the tightest claim the statement supports: one query asked for
    both.
    """
    source = (
        "class TaskModule:\n"
        "    def sweep(self):\n"
        "        sqlU = f'UPDATE {schema}.JEDI_Tasks SET oldStatus=:oldStatus '\n"
        "        sqlU += 'WHERE jediTaskID IN '\n"
        "        sqlU += f'(SELECT PandaID FROM {schema}.jobsActive4) '\n"
        "        self.cur.execute(sqlU + comment, varMap)\n"
    )
    modules = [
        _module(_SPECS, "pandaserver/taskbuffer/Specs.py"),
        _module(_JOB_TABLE_WRITER, "teacher.py"),
        _module(source, "x.py"),
    ]
    attributor = attribution.SpecAttributor(
        progress.spec_attributes(modules), attribution.class_bases(modules)
    )
    attributor.learn_table_classes(modules)

    side = sqlwrite.read_side(modules, attributor)

    # Both kinds of row are seen, each under the verb that names it, and each
    # under its own table rather than under a spec class covering three.
    assert side.entities["jobsactive4"].read_by == {"x.py::sweep"}
    assert side.entities["jedi_tasks"].updated_by == {"x.py::sweep"}
    assert side.joins == {}


def test_a_table_holding_no_spec_is_still_a_kind_of_row():
    """Reversed deliberately, and here is the reason.

    This used to require a spec class, on the grounds that keying by table
    would split ``JobSpec`` into three -- one per lifetime table.  That reason
    holds for a *subject*, where ``jobsActive4`` and ``jobsArchived4`` are one
    ``JobSpec.jobStatus``, and it inverts for a row's lifetime: which of those
    tables a row was created in is the fact, not an accident to fold away.
    Requiring the class also answered for seven of the forty-eight tables this
    corpus inserts into, and the forty-one it dropped are the ones where
    arrival means something -- PRODSYS_COMM among them.

    What is excluded now is a table nothing here writes, because that already
    has a node: an unbound boundary saying the value comes from outside.
    """
    source = (
        "class TaskModule:\n"
        "    def rows(self):\n"
        "        sqlP = f'SELECT COMM_CMD FROM {schema}.PRODSYS_COMM '\n"
        "        sqlP += 'WHERE COMM_TASK=:COMM_TASK '\n"
        "        self.cur.execute(sqlP + comment, varMap)\n"
        "    def issue(self):\n"
        "        sqlI = f'INSERT INTO {schema}.PRODSYS_COMM (COMM_TASK) VALUES (:COMM_TASK) '\n"
        "        self.cur.execute(sqlI + comment, varMap)\n"
    )
    modules = [_module(_SPECS, "pandaserver/taskbuffer/Specs.py"), _module(source, "x.py")]
    attributor = attribution.SpecAttributor(
        progress.spec_attributes(modules), attribution.class_bases(modules)
    )
    attributor.learn_table_classes(modules)

    side = sqlwrite.read_side(modules, attributor)
    nodes = {n.name: n for n in sqlwrite.entity_nodes(side.entities, MAP_ID, VERSION, attributor)}

    assert nodes["prodsys_comm"].spec_class is None
    assert nodes["prodsys_comm"].created_by == ["x.py::issue"]
    assert nodes["prodsys_comm"].read_by == ["x.py::rows"]


def test_a_subject_carries_what_bounds_the_queries_that_select_it():
    """A status the map says is selected is half an answer without this.

    The task really was in a status the query asks for; it was outside what the
    query could see, because the table it joins had stopped being updated.
    """
    modules = [
        _module(_SPECS, "pandaserver/taskbuffer/Specs.py"),
        _module(_ORPHAN_RESCUE, "pandaserver/taskbuffer/db_proxy_mods/task_module.py"),
    ]
    attributor = attribution.SpecAttributor(
        progress.spec_attributes(modules), attribution.class_bases(modules)
    )
    attributor.learn_table_classes(modules)

    gated = sqlwrite.selection_gates(
        modules, attributor, boundary.tables_never_written(modules)
    )

    assert gated["JediTaskSpec.status"] == {"JEDI_AUX_Status_MinTaskID"}


def test_a_statement_built_with_str_format_is_read():
    """62 statements still use ``.format`` rather than an f-string.

    Missing them did not merely lose coverage: those are where JEDI selects
    tasks by status, so a dozen task states read as ones nothing selects on.
    """
    source = (
        "class TaskModule:\n"
        "    def find(self):\n"
        "        sqlR = 'SELECT jediTaskID FROM {0}.JEDI_Tasks tabT '.format(schemaJEDI)\n"
        "        sqlR += \"WHERE tabT.status='running' \"\n"
        "        self.cur.execute(sqlR + comment, varMap)\n"
    )
    func = ast.parse(source).body[0].body[0]

    (text,) = _held(func, "sqlR")
    assert text.startswith("SELECT jediTaskID FROM {}.JEDI_Tasks tabT ")
    ((term,),) = [sqltext.terms(text).terms]
    assert (term.table, term.column, term.literals) == ("JEDI_Tasks", "status", ("running",))


# ---------------------------------------------------------------------------
# Which log file holds the evidence
# ---------------------------------------------------------------------------


def test_a_logger_is_named_either_by_a_literal_or_by_the_module():
    """Two forms, and only two, across both packages.

    JEDI takes the module's own name and the server mostly spells one out,
    but both appear in both, so the shape is what is read rather than the
    package.
    """
    jedi = _module(
        'logger = PandaLogger().getLogger(__name__.split(".")[-1])\n',
        "pandajedi/jedibrokerage/AtlasProdJobBroker.py",
    )
    server = _module(
        '_logger = PandaLogger().getLogger("api_async_process")\n',
        "pandaserver/api/v1/async_process_api.py",
    )

    assert logfile.logger_name(jedi) == "AtlasProdJobBroker"
    assert logfile.logger_name(server) == "api_async_process"
    assert logfile.declared_files([jedi, server]) == {
        "pandajedi/jedibrokerage/AtlasProdJobBroker.py": "panda-AtlasProdJobBroker.log",
        "pandaserver/api/v1/async_process_api.py": "panda-api_async_process.log",
    }


def test_a_module_that_declares_no_logger_is_left_alone():
    """A wrong filename is worse than none: the query comes back empty, and
    empty reads as "production never emitted this"."""
    base = _module(
        "class JobBrokerBase:\n    pass\n",
        "pandajedi/jedibrokerage/JobBrokerBase.py",
    )

    assert logfile.logger_name(base) is None
    assert logfile.declared_files([base]) == {}


def test_a_mixin_inherits_the_log_files_of_what_mixes_it_in():
    """The proxy modules are the map's largest group and declare no logger.

    ``OraDBProxy.DBProxy`` mixes them in and ``JediDBProxy.DBProxy``
    subclasses that, so the same junction writes to the server's proxy log or
    JEDI's depending on which process ran it.  Both are named because which
    one is a runtime fact, not an ambiguity to be resolved.
    """
    mixin = _module(
        "class TaskStandaloneModule:\n    pass\n",
        "pandaserver/taskbuffer/db_proxy_mods/task_standalone_module.py",
    )
    ora = _module(
        '_logger = PandaLogger().getLogger("DBProxy")\n'
        "class DBProxy(task_standalone_module.TaskStandaloneModule):\n    pass\n",
        "pandaserver/taskbuffer/OraDBProxy.py",
    )
    jedi = _module(
        'logger = PandaLogger().getLogger(__name__.split(".")[-1])\n'
        "class DBProxy(OraDBProxy.DBProxy):\n    pass\n",
        "pandajedi/jedicore/JediDBProxy.py",
    )
    modules = [mixin, ora, jedi]

    inherited = logfile.inherited_files(modules, logfile.declared_files(modules))

    assert inherited[mixin.rel_path] == ["panda-DBProxy.log", "panda-JediDBProxy.log"]


def test_the_inheritance_walk_does_not_stop_at_one_link():
    """Stopping at the first would name the server's log and silently omit
    JEDI's -- the half most junctions actually run under."""
    mixin = _module("class Mixin:\n    pass\n", "p/mixin.py")
    middle = _module('l = PandaLogger().getLogger("Mid")\nclass Mid(Mixin):\n    pass\n', "p/middle.py")
    leaf = _module('l = PandaLogger().getLogger("Leaf")\nclass Leaf(Mid):\n    pass\n', "p/leaf.py")
    modules = [mixin, middle, leaf]

    inherited = logfile.inherited_files(modules, logfile.declared_files(modules))

    assert inherited["p/mixin.py"] == ["panda-Leaf.log", "panda-Mid.log"]


# ---------------------------------------------------------------------------
# Whose log says it ran -- a different question from what starts it
# ---------------------------------------------------------------------------


def _log_attach(*sources: tuple[str, str], junctions):
    fragment = MapFragment(map_id=MAP_ID, derived_from=VERSION, junctions=list(junctions))
    logfile.attach(fragment, [_module(text, rel) for text, rel in sources])
    return fragment


PROXY_METHOD = """
class TaskModule:
    def makeTaskPending_JEDI(self, task_id):
        self.cur.execute(sql, varMap)
"""

KNIGHT = """
logger = PandaLogger().getLogger(__name__.split(".")[-1])


class Knight:
    def start(self):
        while True:
            self.taskBufferIF.makeTaskPending_JEDI(task_id)
            time.sleep(60)
"""


def test_a_writer_with_no_logger_of_its_own_takes_its_callers():
    """The proxy mixins hold the code; the knights write the line about it.

    ``set task_status=`` is in ContentsFeeder, JobGenerator, PostProcessor and
    TaskRefiner and in neither proxy file, so reporting only the owner's file
    sends the query somewhere that never answers.
    """
    junction = _junction("pandaserver/taskbuffer/db_proxy_mods/task_module.py::makeTaskPending_JEDI")

    fragment = _log_attach(
        (PROXY_METHOD, "pandaserver/taskbuffer/db_proxy_mods/task_module.py"),
        (KNIGHT, "pandajedi/jediorder/Knight.py"),
        junctions=[junction],
    )

    assert fragment.junctions[0].log_files == []
    assert fragment.junctions[0].caller_log_files == ["panda-Knight.log"]


def test_whether_a_junctions_own_files_are_its_own_is_recorded():
    """An inherited file is not a place a line about this junction can appear.

    Both of these are true of a proxy method: the SQL comment trace really does
    land in ``panda-DBProxy.log``, and ``set task_status=`` never does, because
    the caller writes it.  Without the distinction a reader asking the inherited
    file gets an empty answer it cannot interpret -- and reading it as "the
    junction did not fire" rules out every proxy candidate at once.
    """
    proxy = _junction("pandaserver/taskbuffer/db_proxy_mods/task_module.py::makeTaskPending_JEDI")
    knight = _junction("pandajedi/jediorder/Knight.py::start")

    fragment = _log_attach(
        (PROXY_METHOD, "pandaserver/taskbuffer/db_proxy_mods/task_module.py"),
        (KNIGHT, "pandajedi/jediorder/Knight.py"),
        junctions=[proxy, knight],
    )
    owns = {j.owner.split("::")[-1]: j.owns_logger for j in fragment.junctions}

    assert owns == {"makeTaskPending_JEDI": False, "start": True}


def test_a_caller_that_starts_nothing_still_names_a_log():
    """The two questions come apart here, which is the whole point.

    ``event_picker`` is driven by a daemon script rather than a loop of its
    own, so it carries no trigger -- and it is still the only thing that calls
    ``updateTaskModTimeJEDI`` and the only file that would mention it.
    """
    picker = """
logger = PandaLogger().getLogger(__name__.split(".")[-1])


class EventPicker:
    def run(self):
        self.task_buffer.makeTaskPending_JEDI(task_id)
"""
    junction = _junction("pandaserver/taskbuffer/db_proxy_mods/task_module.py::makeTaskPending_JEDI")

    fragment = _log_attach(
        (PROXY_METHOD, "pandaserver/taskbuffer/db_proxy_mods/task_module.py"),
        (picker, "pandaserver/dataservice/event_picker.py"),
        junctions=[junction],
    )

    assert fragment.junctions[0].caller_log_files == ["panda-event_picker.log"]
    # ...and nothing starts it, which stays a separate and still-true answer.
    assert fragment.junctions[0].entry_points == []


def test_the_facade_forwarding_a_call_is_not_a_caller():
    """``JediTaskBuffer`` declares a logger and logs twice, both in __init__.

    Naming its file would send every proxy query to something that never says
    anything, and an empty answer there reads as "this code never ran".  It sat
    on 71 junctions before the borrowed-proxy receiver was excluded.
    """
    facade = """
logger = PandaLogger().getLogger(__name__.split(".")[-1])


class JediTaskBuffer:
    def makeTaskPending_JEDI(self, task_id):
        with self.proxyPool.get() as proxy:
            return proxy.makeTaskPending_JEDI(task_id)

    def checkWaitingTaskPrio_JEDI(self, task_id):
        with self.proxyPool.get() as proxy:
            return proxy.makeTaskPending_JEDI(task_id)
"""
    junction = _junction("pandaserver/taskbuffer/db_proxy_mods/task_module.py::makeTaskPending_JEDI")

    fragment = _log_attach(
        (PROXY_METHOD, "pandaserver/taskbuffer/db_proxy_mods/task_module.py"),
        (facade, "pandajedi/jedicore/JediTaskBuffer.py"),
        junctions=[junction],
    )

    # Both spellings are refused: the one that forwards under the same name and
    # the one that forwards under a different one.
    assert fragment.junctions[0].caller_log_files == []


def test_a_call_sharing_the_enclosing_functions_name_is_still_a_call():
    """``datasetManager.run`` builds a ``Closer`` and calls ``closer.run()``.

    Keying the facade rule on the name rather than the receiver discarded that
    edge and three others, taking ``closer`` and ``finisher`` off the daemon
    cycle entirely.  The receiver is what makes a handoff a handoff.
    """
    closer = """
logger = PandaLogger().getLogger(__name__.split(".")[-1])


class Closer:
    def run(self):
        self.taskBufferIF.makeTaskPending_JEDI(task_id)
"""
    manager = """
logger = PandaLogger().getLogger(__name__.split(".")[-1])


def run():
    closer_process = Closer(taskBuffer, blocks, job)
    closer_process.run()
"""
    junction = _junction("pandaserver/dataservice/closer.py::run")

    fragment = _log_attach(
        (closer, "pandaserver/dataservice/closer.py"),
        (manager, "pandaserver/daemons/scripts/datasetManager.py"),
        junctions=[junction],
    )

    assert "panda-datasetManager.log" in fragment.junctions[0].caller_log_files


def test_a_caller_log_the_owner_already_names_is_not_repeated():
    """The lists answer different questions; saying the same file twice would
    make one of them look like corroboration it is not."""
    owner = """
logger = PandaLogger().getLogger("shared")


class Owner:
    def writes(self):
        pass
"""
    caller = """
logger = PandaLogger().getLogger("shared")


class Caller:
    def go(self):
        self.other.writes()
"""
    junction = _junction("p/owner.py::writes")

    fragment = _log_attach((owner, "p/owner.py"), (caller, "p/caller.py"), junctions=[junction])

    assert fragment.junctions[0].log_files == ["panda-shared.log"]
    assert fragment.junctions[0].caller_log_files == []


# ---------------------------------------------------------------------------
# Production evidence and the (ii) gates
# ---------------------------------------------------------------------------
#
# These pin the distinction the whole production half rests on: an empty grep
# result means three different things, and only one of them is a fact about
# PanDA.  Everything else here follows from getting that wrong being silent.

BROKER_LOG = "panda-AtlasProdJobBroker.log"
PROXY_LOG = "panda-DBProxy.log"


def _log_line(level: str, message: str, name: str = "JobBroker") -> str:
    """One line in PandaLogger's format.

    ``"%(asctime)s %(name)-12s: %(levelname)-8s %(message)s"`` -- the level is
    the token after the first ``": "``, which is what the parser anchors on.
    """
    return f"2026-08-30 12:00:01,123 {name:<12}: {level:<8} {message}"


def _sample(lines: list[str], filename: str = BROKER_LOG, **kwargs) -> evidence.GrepResult:
    """A result shaped the way the fetch path shapes one.

    ``matched`` and ``level_counts`` are filled in here for the same reason the
    fetch fills them: the level query keeps no lines, so a fixture that only
    set ``lines`` would be testing a result that cannot occur.
    """
    pattern = kwargs.pop("pattern", evidence.ANY_LINE_PATTERN)
    return evidence.GrepResult(
        query=evidence.GrepQuery(
            pattern=pattern,
            log_filename=filename,
            service=kwargs.pop("service", evidence.JEDI),
        ),
        machine=kwargs.pop("machine", "m1"),
        lines=lines,
        matched=len(lines),
        level_counts=dict(evidence._levels_in(lines)),
        return_code=kwargs.pop("return_code", 0),
        **kwargs,
    )


def _missing(filename: str = BROKER_LOG, machine: str = "m1") -> evidence.GrepResult:
    return _sample(
        [],
        filename,
        machine=machine,
        return_code=2,
        error=f"rg: /var/log/panda/{filename}: No such file or directory (os error 2)",
    )


def _evidence(*results: evidence.GrepResult) -> evidence.Evidence:
    return evidence.Evidence(fetched_at="2026-08-30T00:00:00+00:00", results=list(results))


def test_the_level_is_measured_per_log_file():
    """PanDA configures a level per logger, so a threshold taken from one file
    and applied to another would be a number established for something else."""
    ev = _evidence(
        _sample([_log_line("INFO", "a")], BROKER_LOG),
        _sample([_log_line("DEBUG", "b"), _log_line("INFO", "c")], PROXY_LOG),
    )

    assert evidence.effective_level(ev, log_filename=BROKER_LOG) == "INFO"
    assert evidence.effective_level(ev, log_filename=PROXY_LOG) == "DEBUG"
    assert evidence.effective_level(ev, log_filename="panda-absent.log") is None


def test_a_level_word_in_a_message_is_not_a_level():
    """The separator is what makes it a level, not the word.

    Counting the bare word would read a line that merely mentions DEBUG as
    proof that DEBUG is enabled, and so keep an observable production never
    emits.
    """
    ev = _evidence(_sample([_log_line("INFO", "restarting in DEBUG mode")]))

    assert evidence.level_histogram(ev) == Counter({"INFO": 1})


def test_a_continuation_line_carries_no_level():
    """Tracebacks span lines, and only the first one is formatted."""
    ev = _evidence(_sample([_log_line("ERROR", "boom"), "  File 'x.py', line 3"]))

    assert evidence.level_histogram(ev) == Counter({"ERROR": 1})


def test_an_empty_result_is_conclusive_only_when_the_tool_read_everything():
    """Three ways to come back empty, one of which is an answer.

    ``rg`` exits 1 for "no match" and 2 for "could not read the file", and a
    result over a megabyte is truncated.  A missing file and a cut sample look
    exactly like "production never emits this", so only exit 1 on a whole
    result licenses reading absence as evidence.
    """
    assert _sample([], return_code=1).conclusive
    assert not _missing().conclusive
    assert not _sample([_log_line("INFO", "a")], truncated=True).conclusive


def test_every_query_carries_its_bounds():
    """Unbounded is not an option: panda-DBProxy.log is six gigabytes and the
    processor buffers a matcher's whole output before storing a slice of it."""
    query = evidence.GrepQuery(pattern="x", log_filename=BROKER_LOG, service=evidence.JEDI)

    assert query.max_matches == evidence.DEFAULT_MAX_MATCHES
    assert query.tail_bytes == evidence.DEFAULT_TAIL_BYTES


def test_hitting_the_match_cap_reaches_the_gates_as_truncated():
    """The server sets truncated when it stops at the cap, and that is the
    whole point of sending one: a capped answer must not read as an absent one."""
    capped = _sample([_log_line("INFO", "a")], truncated=True, return_code=0)

    assert not capped.conclusive
    assert not _evidence(capped).conclusive(evidence.ANY_LINE_PATTERN)


def test_a_file_absent_everywhere_says_the_code_never_ran():
    """The strongest thing production says about the map.

    PandaLogger creates the file on first emit, so no file on any machine
    means that logger has never emitted -- not quiet, unexecuted.
    """
    ev = _evidence(_missing(machine="m1"), _missing(machine="m2"))

    assert ev.file_status(BROKER_LOG) == "absent"


def test_one_machine_having_the_file_makes_it_present():
    """The services run different knights; a union, not an intersection."""
    ev = _evidence(_missing(machine="m1"), _sample([_log_line("INFO", "a")], machine="m2"))

    assert ev.file_status(BROKER_LOG) == "present"
    assert len(ev.lines(evidence.ANY_LINE_PATTERN)) == 1


def test_another_kind_of_error_is_unknown_not_absent():
    """"Not authorized" must never be read as "the code never ran"."""
    ev = _evidence(_sample([], return_code=2, error="'me' is not authorized"))

    assert ev.file_status(BROKER_LOG) == "unknown"


def test_a_file_nobody_asked_about_is_unknown():
    ev = _evidence(_sample([_log_line("INFO", "a")], BROKER_LOG))

    assert ev.file_status(PROXY_LOG) == "unknown"


def test_a_read_error_becomes_an_error_not_an_empty_answer():
    """Exit 2 is usually a wrong filename; stderr turns that into a one-step fix."""
    query = evidence.GrepQuery(pattern="x", log_filename="panda-typo.log", service=evidence.JEDI)
    payload = {
        "expected_machines": ["m1"],
        "results": [
            {
                "machine_name": "m1",
                "result": "",
                "stderr": "panda-typo.log: No such file or directory",
                "return_code": 2,
            }
        ],
    }

    (result,) = evidence._results_from(query, payload)

    assert result.error == "panda-typo.log: No such file or directory"
    assert not result.conclusive


def test_a_machine_that_never_answered_is_recorded_as_silent():
    """Silence and "found nothing" are different facts about a machine."""
    query = evidence.GrepQuery(pattern="x", log_filename=BROKER_LOG, service=evidence.JEDI)
    payload = {
        "expected_machines": ["m1", "m2"],
        "results": [{"machine_name": "m1", "result": "hit\n", "return_code": 0}],
    }

    results = {r.machine: r for r in evidence._results_from(query, payload)}

    assert results["m1"].lines == ["hit"]
    assert results["m2"].error == "no result returned"


def test_evidence_round_trips_through_a_file(tmp_path):
    """Fetching and checking are separate steps, so the record has to survive
    the gap intact -- that is what lets the gates re-run offline."""
    ev = _evidence(_sample([_log_line("DEBUG", "a")], truncated=True))
    path = tmp_path / "nested" / "evidence.json"

    ev.save(path)
    back = evidence.Evidence.load(path)

    assert back.results[0].truncated
    assert evidence.effective_level(back, log_filename=BROKER_LOG) == "DEBUG"


def test_log_format_recognised_fails_when_no_line_carries_a_level():
    """A pattern that matches nothing looks exactly like a gate that passed.

    Every later production gate reads the log by matching against this
    format, so it is checked directly rather than assumed.
    """
    ev = _evidence(_sample(["something in another format"]))

    result = gates.log_format_recognised(ev)

    assert not result.passed
    assert "log format is not" in result.failures[0]


def test_log_format_recognised_names_a_query_that_did_not_run():
    """Not authorized, or a wrong filename, is not production disagreeing."""
    ev = _evidence(_sample([], return_code=2, error="'me' is not authorized"))

    result = gates.log_format_recognised(ev)

    assert not result.passed
    assert result.failures == ["jedi/m1: 'me' is not authorized"]


def test_log_format_recognised_passes_on_a_well_formed_sample():
    assert gates.log_format_recognised(_evidence(_sample([_log_line("INFO", "a")]))).passed


def _stage(tag: str, level: str, owner: str, files: list[str], **kwargs) -> FilterStageNode:
    return FilterStageNode(
        map_id=MAP_ID,
        derived_from=VERSION,
        name=f"f:{tag}",
        owner=owner,
        criteria_tag=tag,
        log_level=level,
        log_files=files,
        **kwargs,
    )


BROKER = "pandajedi/jedibrokerage/AtlasProdJobBroker.py::doBrokerage"


def test_an_observable_below_the_threshold_is_a_promise_the_map_cannot_keep():
    """A DEBUG line does not exist in a file whose logger runs at INFO.

    Worse than having no observable: a strategy would spend a step fetching
    a line that is never written.
    """
    fragment = MapFragment(
        map_id=MAP_ID,
        derived_from=VERSION,
        filter_stages=[
            _stage("-diskIO", "debug", BROKER, [BROKER_LOG]),
            _stage("-status", "info", BROKER, [BROKER_LOG]),
        ],
    )
    ev = _evidence(_sample([_log_line("INFO", "a")], BROKER_LOG))

    result = gates.observables_are_emitted(fragment, ev)

    assert result.checked == 2
    assert not result.passed
    assert result.failures == [
        f"-diskIO at {BROKER} emits at DEBUG but {BROKER_LOG} at INFO"
    ]


def test_emitting_in_either_candidate_file_is_enough():
    """The proxy mixins run under two processes; emitting in one makes the
    observable real, so an all-candidates rule is what the code needs."""
    fragment = MapFragment(
        map_id=MAP_ID,
        derived_from=VERSION,
        filter_stages=[_stage("-x", "debug", BROKER, [BROKER_LOG, PROXY_LOG])],
    )
    ev = _evidence(
        _sample([_log_line("INFO", "a")], BROKER_LOG),
        _sample([_log_line("DEBUG", "b")], PROXY_LOG),
    )

    assert gates.observables_are_emitted(fragment, ev).passed


def test_a_stage_whose_file_never_existed_is_not_a_logging_failure():
    """A path that never ran is a different finding from one that runs
    quietly, and conflating them would blame the map for the deployment."""
    fragment = MapFragment(
        map_id=MAP_ID,
        derived_from=VERSION,
        filter_stages=[_stage("-x", "debug", BROKER, [BROKER_LOG])],
    )
    ev = _evidence(_missing())

    emitted = gates.observables_are_emitted(fragment, ev)
    live = gates.code_paths_are_live(fragment, ev)

    assert emitted.failures == []
    # Says the file is on no machine rather than "no sample": the absence is a
    # finding of its own, and calling it missing evidence made one fact read as
    # two separate problems.
    assert emitted.inconclusive == [
        f"{BROKER_LOG} is on no machine, so -x at {BROKER} never ran here"
    ]
    assert not live.passed
    assert BROKER_LOG in live.failures[0]


def test_a_stage_the_source_gives_no_file_for_is_inconclusive():
    fragment = MapFragment(
        map_id=MAP_ID,
        derived_from=VERSION,
        filter_stages=[_stage("-x", "debug", BROKER, [])],
    )

    result = gates.observables_are_emitted(fragment, _evidence(_sample([_log_line("INFO", "a")])))

    assert result.passed
    assert result.inconclusive == [f"no log file is named in the source for -x at {BROKER}"]


def test_an_inconclusive_row_leads_with_its_reason():
    """The report summarises these rows by their first line, so the reason has
    to come before the anchor -- otherwise the summary keeps the part that says
    where and drops the part that says what."""
    fragment = MapFragment(
        map_id=MAP_ID,
        derived_from=VERSION,
        filter_stages=[_stage("-x", "debug", BROKER, [BROKER_LOG]), _stage("-y", "debug", BROKER, [])],
    )

    rows = gates.observables_are_emitted(fragment, _evidence(_missing())).inconclusive

    assert all(not row.startswith("-") for row in rows)


def test_an_absent_caller_log_does_not_strand_a_junction_another_caller_reaches():
    """The trap in checking caller files, and it is easy to fall into.

    ``kickExhaustedTasks_JEDI`` is defined in ``TypicalWatchDogBase``, so every
    ``Atlas*WatchDog`` inherits it.  ``panda-GenWatchDog.log`` is on no machine
    and ``panda-AtlasProdWatchDog.log`` exists, so the component is not running
    and the junction is reached anyway.  Reading the first as "this junction
    never runs" would report a live code path as dead.
    """
    junction = JunctionNode(
        map_id=MAP_ID,
        derived_from=VERSION,
        name="j",
        subject="JediTaskSpec.status",
        owner="pandaserver/taskbuffer/db_proxy_mods/m.py::kickExhaustedTasks_JEDI",
        caller_log_files=["panda-GenWatchDog.log", BROKER_LOG],
        owns_logger=False,
    )
    ev = _evidence(
        _missing("panda-GenWatchDog.log"),
        _sample([_log_line("INFO", "a")], BROKER_LOG),
    )

    result = gates.code_paths_are_live(
        MapFragment(map_id=MAP_ID, derived_from=VERSION, junctions=[junction]), ev
    )

    assert not result.passed  # the component really is absent
    assert "reached through a log that does exist" in result.failures[0]
    assert "never run here" not in result.failures[0]


def test_a_junction_whose_every_log_is_absent_is_reported_as_dead():
    """The other half: with no surviving file the node really is unreachable."""
    junction = JunctionNode(
        map_id=MAP_ID,
        derived_from=VERSION,
        name="j",
        subject="JediTaskSpec.status",
        owner="pandaserver/taskbuffer/db_proxy_mods/m.py::onlyHere_JEDI",
        caller_log_files=["panda-GenWatchDog.log"],
        owns_logger=False,
    )

    result = gates.code_paths_are_live(
        MapFragment(map_id=MAP_ID, derived_from=VERSION, junctions=[junction]),
        _evidence(_missing("panda-GenWatchDog.log")),
    )

    assert "0 stage(s), 1 junction(s) of the map never run here" in result.failures[0]


def test_code_paths_are_live_passes_when_every_file_is_there():
    fragment = MapFragment(
        map_id=MAP_ID,
        derived_from=VERSION,
        filter_stages=[_stage("-x", "info", BROKER, [BROKER_LOG])],
    )

    assert gates.code_paths_are_live(fragment, _evidence(_sample([_log_line("INFO", "a")]))).passed


def test_the_package_places_a_module_in_a_machine_group():
    """A first approximation only: JEDI opens its own TaskBuffer, so
    ``db_proxy_mods`` called from a knight logs to JEDI's files even though it
    lives in ``pandaserver``.  Sound for choosing which service declares a
    file, which is all it is used for."""
    assert evidence.service_for_module("pandajedi/jedibrokerage/x.py") == evidence.JEDI
    assert evidence.service_for_module("pandaserver/api/v1/pilot_api.py") == evidence.SERVER


def test_a_gate_summary_separates_issues_from_inconclusive():
    """Conflating them is the failure mode this whole half is built to avoid."""
    result = gates.GateResult(
        gate="g", passed=False, checked=3, failures=["a"], inconclusive=["b", "c"]
    )

    assert result.summary() == "[FAIL] g: 3 checked (1 issue(s), 2 inconclusive)"


# ---------------------------------------------------------------------------
# Comparing two builds
# ---------------------------------------------------------------------------
#
# The only check that can see a threshold move.  Comparing the map with its
# own source agrees with whatever the source now says, and a condition is
# never echoed to a log, so a build from the wrong release explains a decision
# with a number that has since changed -- convincingly.


def _stage_for_diff(tag: str, conditions: list[str], line: int, **kwargs) -> FilterStageNode:
    return FilterStageNode(
        map_id=MAP_ID,
        derived_from=kwargs.pop("version", VERSION),
        name=f"f:{tag}",
        owner="pandajedi/jedibrokerage/AtlasProdJobBroker.py::doBrokerage",
        criteria_tag=tag,
        conditions=conditions,
        anchor=Anchor(
            package="pandajedi",
            file="pandajedi/jedibrokerage/AtlasProdJobBroker.py",
            line_start=line,
            line_end=line,
        ),
        **kwargs,
    )


def _fragment_of(*nodes, version: str = VERSION) -> MapFragment:
    fragment = MapFragment(map_id=MAP_ID, derived_from=version)
    for node in nodes:
        if isinstance(node, FilterStageNode):
            fragment.filter_stages.append(node)
        elif isinstance(node, JunctionNode):
            fragment.junctions.append(node)
        elif isinstance(node, BoundaryNode):
            fragment.boundaries.append(node)
        elif isinstance(node, SubjectNode):
            fragment.subjects.append(node)
        elif isinstance(node, ReadSiteNode):
            fragment.read_sites.append(node)
    return fragment


def test_a_node_that_only_moved_is_not_a_change():
    """The reason node identity is a signature and not a position.

    ``update_job`` moved file and line between two releases; a key built from
    those would have called one boundary two.  In a refactor this is most of
    the map, so it has to generate no noise.
    """
    old = _fragment_of(_stage_for_diff("-disk", ["a < b"], line=100))
    new = _fragment_of(_stage_for_diff("-disk", ["a < b"], line=140))

    result = diff.compare(old, new)

    assert result.changes == []
    assert result.moved == ["FilterStage f:-disk"]
    assert result.unchanged == 0


def test_an_edited_condition_is_reported_as_drift():
    """The failure no gate can see: the escape hatch is gone, and a map built
    from the older release would still offer it as the reason a site survived."""
    old = _fragment_of(
        _stage_for_diff("-disk", ["size < threshold and 'skip_RSE_check' not in catchall"], 100)
    )
    new = _fragment_of(_stage_for_diff("-disk", ["size < threshold"], 100))

    result = diff.compare(old, new)

    (drifted,) = result.drift()
    assert drifted.field == "conditions"
    assert "skip_RSE_check" in drifted.before
    assert "skip_RSE_check" not in drifted.after


def _vp_stage(digest: str, conditions: list[str]) -> FilterStageNode:
    """An untagged stage, named the way ``selection.py`` names one."""
    return FilterStageNode(
        map_id=MAP_ID,
        derived_from=VERSION,
        name=f"panda:AtlasAnalJobBroker.py::doBrokerage:avoid VP queue check#{digest}",
        owner="pandajedi/jedibrokerage/AtlasAnalJobBroker.py::doBrokerage",
        conditions=conditions,
        anchor=Anchor(
            package="pandajedi",
            file="pandajedi/jedibrokerage/AtlasAnalJobBroker.py",
            line_start=100,
            line_end=100,
        ),
    )


def test_a_stage_renamed_by_its_own_conditions_is_drift_and_not_two_nodes():
    """A stage with no tag of its own is named after a digest of what it tests,
    which is right for identity -- a step testing something else is another
    step -- and defeats this report, because editing the condition renames the
    node and the edit arrives as an unrelated removal and addition.

    Two real ones were lost that way between 1.0.2 and master, both in
    ``AtlasAnalJobBroker``; this is one of them.
    """
    old = _fragment_of(
        _vp_stage("558682", ["taskSpec.ioIntensity and (not inputChunk.isMerging)"])
    )
    new = _fragment_of(_vp_stage("c02f04", ["not inputChunk.isMerging"]))

    result = diff.compare(old, new)

    assert result.added == []
    assert result.removed == []
    (drifted,) = result.drift()
    assert drifted.field == "conditions"
    assert "ioIntensity" in drifted.before
    assert "ioIntensity" not in drifted.after


def test_two_stages_of_one_funnel_changing_together_are_not_paired():
    """Once both digests have moved there is nothing left in either name to
    tell the two apart, so pairing them would report one step's new condition
    as the other's.  Reported as it was found instead."""
    old = _fragment_of(_vp_stage("aaaaaa", ["a"]), _vp_stage("bbbbbb", ["b"]))
    new = _fragment_of(_vp_stage("cccccc", ["c"]), _vp_stage("dddddd", ["d"]))

    result = diff.compare(old, new)

    assert len(result.removed) == 2
    assert len(result.added) == 2
    assert result.drift() == []


def test_a_changed_interface_is_a_change_but_not_drift():
    """Both matter; only one of them silently rewrites an explanation."""
    def boundary(values):
        return BoundaryNode(
            map_id=MAP_ID,
            derived_from=VERSION,
            name="b:pilot:update_job",
            system="pilot",
            interface="pilot_api::update_job",
            carried_values=values,
        )

    result = diff.compare(
        _fragment_of(boundary(["job_id"])), _fragment_of(boundary(["job_id", "job_status"]))
    )

    assert result.drift() == []
    (change,) = result.other()
    assert change.field == "carried_values"


def _junction_for_diff(branches: list[Branch]) -> JunctionNode:
    return JunctionNode(
        map_id=MAP_ID,
        derived_from=VERSION,
        name="j:setStatus",
        subject="JediTaskSpec.status",
        owner="pandajedi/jediorder/ContentsFeeder.py::feed",
        branches=branches,
    )


def test_branches_are_matched_by_outcome_not_position():
    """A branch keeps its outcome while the condition reaching it is edited.

    Comparing by position would render an inserted branch as "everything
    below here changed" and bury the one edit that matters.
    """
    old = _fragment_of(
        _junction_for_diff(
            [
                Branch(outcome="ready", path_condition=["nFiles > 0"], order=0),
                Branch(outcome="pending", path_condition=["else"], order=1),
            ]
        )
    )
    new = _fragment_of(
        _junction_for_diff(
            [
                Branch(outcome="broken", path_condition=["corrupt"], order=0),
                Branch(outcome="ready", path_condition=["nFiles > 2"], order=1),
                Branch(outcome="pending", path_condition=["else"], order=2),
            ]
        )
    )

    result = diff.compare(old, new)

    drifted = result.drift()
    assert [c.node for c in drifted] == ["j:setStatus → ready"]
    assert drifted[0].before == "[nFiles > 0]"
    assert drifted[0].after == "[nFiles > 2]"
    assert any(c.field == "branch" and c.after == "broken" for c in result.other())


def test_a_row_precondition_that_moves_is_reported():
    """The field that says whether a branch's outcome can land at all.

    Left off the comparison until now, and the round that settled these
    guards' binds -- ``jobStatus=:oldJobStatus`` becoming
    ``jobStatus=activated`` on 45 junctions -- was told by ``diff-map`` that
    the two builds described the same map.  The same hole ``log_sites`` had.
    """
    old = _fragment_of(
        _junction_for_diff(
            [Branch(outcome="sent", row_precondition=["jobStatus=:oldJobStatus"], order=0)]
        )
    )
    new = _fragment_of(
        _junction_for_diff(
            [Branch(outcome="sent", row_precondition=["jobStatus=activated"], order=0)]
        )
    )

    result = diff.compare(old, new)

    (change,) = [c for c in result.changes if c.field == "row_precondition"]
    assert change.before == "[jobStatus=:oldJobStatus]"
    assert change.after == "[jobStatus=activated]"


def test_a_statement_that_starts_being_read_only_in_part_is_reported():
    """A term moving out of ``conditions`` and into ``unresolved``.

    That is a statement the map stopped being able to read, which is a change
    of the same kind as a predicate gaining a value -- and with the field
    unlisted the two builds compare as identical.
    """
    old = _fragment_of(_site_with(conditions=["jobStatus=:jobStatus@12"]))
    new = _fragment_of(
        _site_with(conditions=[], unresolved=["jobStatus IN ({}) -- " + sql.BINDS_ARE_A_HOLE])
    )

    result = diff.compare(old, new)

    assert {c.field for c in result.changes} == {"conditions", "unresolved"}


def test_losing_a_polled_entry_is_reported():
    """Not cosmetic: it is the difference between a stall that clears itself
    and one that does not."""
    def junction(entries):
        node = _junction_for_diff([Branch(outcome="ready", path_condition=[], order=0)])
        node.entry_points = entries
        return node

    polled = EntryPoint(trigger="polled", entry="jediorder/ContentsFeeder.py")
    message = EntryPoint(trigger="message", entry="jedimsgprocessor/feeder.py")

    result = diff.compare(
        _fragment_of(junction([polled, message])), _fragment_of(junction([message]))
    )

    (change,) = [c for c in result.changes if c.field == "entry_points"]
    assert "polled" in change.before
    assert "polled" not in change.after


def test_losing_the_one_query_that_selects_a_value_is_reported():
    """Who selects on a value is the answer the map is asked for.

    Taken from a round that moved it for real: ``copyArchive::main`` stopped
    selecting ``JobSpec.jobStatus=closed`` because the binds it was credited
    with belonged to a later statement.  ``selected_values`` did not move --
    the value is still selected, by other functions -- so with only the value
    set compared, the two builds reported as describing the same map.  It is
    the same hole ``CONTENT_FIELDS`` exists to close.
    """
    def subject(selected_by):
        return SubjectNode(
            map_id=MAP_ID,
            derived_from=VERSION,
            name="JobSpec.jobStatus",
            spec_class="JobSpec",
            attribute="jobStatus",
            selected_values=["closed"],
            selected_by=selected_by,
        )

    result = diff.compare(
        _fragment_of(subject({"closed": ["copyArchive.py::main", "a.py::f"]})),
        _fragment_of(subject({"closed": ["a.py::f"]})),
    )

    (change,) = result.changes
    assert change.field == "selected_by"
    assert "copyArchive" in change.before
    assert "copyArchive" not in change.after


def test_every_field_a_subject_states_a_fact_in_is_compared():
    """``selected_values`` alone left three of them unwatched, and each answers
    a question the map is asked: who selects on a value, who updates rows
    already holding it, and what bounds the selecting whatever the value is."""
    fields = diff.CONTENT_FIELDS[NodeType.SUBJECT.value]

    assert {"selected_by", "updated_by", "selection_gates"} <= set(fields)


def test_added_and_removed_nodes_are_named():
    old = _fragment_of(_stage_for_diff("-disk", ["a"], 100))
    new = _fragment_of(_stage_for_diff("-rse", ["a"], 100))

    result = diff.compare(old, new)

    assert result.removed == ["FilterStage f:-disk"]
    assert result.added == ["FilterStage f:-rse"]


def test_two_identical_builds_report_as_identical():
    old = _fragment_of(_stage_for_diff("-disk", ["a < b"], 100))
    new = _fragment_of(_stage_for_diff("-disk", ["a < b"], 100))

    result = diff.compare(old, new)

    assert result.identical
    assert result.unchanged == 1


def _stored_kinds() -> dict[str, type]:
    """``{node kind: model}`` for every part of a fragment that holds the map's nodes."""
    import typing

    from bamboo.codemap.models import MAP_NODE_GROUPS

    hints = typing.get_type_hints(MapFragment)
    kinds = {}
    for group in MAP_NODE_GROUPS:
        (model,) = typing.get_args(hints[group])
        kinds[model.model_fields["node_type"].default.value] = model
    return kinds


def test_every_field_of_a_stored_node_is_compared_or_says_why_not():
    """The gate for the class ``log_sites``, ``row_precondition`` and
    ``selected_by`` each belonged to: a field nobody listed compares equal to
    itself however it changes, and each was found one round at a time.  A field
    added to a node now fails here until it is put in one list or the other --
    and a misspelt entry, which read ``None`` on both sides, fails too."""
    kinds = _stored_kinds()

    assert set(kinds) == set(diff.CONTENT_FIELDS) == set(diff.NOT_COMPARED)
    for kind, model in kinds.items():
        compared = set(diff.CONTENT_FIELDS[kind])
        excused = set(diff.NOT_COMPARED[kind]) | set(diff.COMMON_NOT_COMPARED)
        assert not compared & excused, kind
        assert compared | excused == set(model.model_fields), kind
        assert all(diff.NOT_COMPARED[kind].values()), kind


def test_every_field_of_a_branch_and_an_entry_is_compared_or_says_why_not():
    assert set(diff.BRANCH_FIELDS) | set(diff.BRANCH_NOT_COMPARED) == set(Branch.model_fields)
    assert not set(diff.BRANCH_FIELDS) & set(diff.BRANCH_NOT_COMPARED)
    assert set(diff.ENTRY_KEY) | set(diff.ENTRY_FIELDS) == set(EntryPoint.model_fields)
    assert not set(diff.ENTRY_KEY) & set(diff.ENTRY_FIELDS)


def test_every_part_of_a_fragment_is_stored_and_compared_or_says_why_not():
    """The reader and ``diff`` read one tuple of the parts holding nodes, and
    every other part of a fragment is named with the reason it is not compared."""
    from bamboo.codemap.models import MAP_NODE_GROUPS

    assert not set(MAP_NODE_GROUPS) & set(diff.NOT_COMPARED_PARTS)
    assert set(MAP_NODE_GROUPS) | set(diff.NOT_COMPARED_PARTS) == set(MapFragment.model_fields)


def test_extending_a_fragment_merges_every_part():
    """``extend`` listed the parts by hand, a second copy of the model."""
    one = MapFragment(map_id=MAP_ID, derived_from=VERSION)
    other = _fragment_of(_stage_for_diff("-disk", ["a"], 100))
    other.declaration_yields["JobSpec (x.py)"] = 3
    other.reading_stops.append(ReadingStop(reason="r"))

    one.extend(other)

    assert len(one.filter_stages) == 1
    assert one.declaration_yields == {"JobSpec (x.py)": 3}
    assert len(one.reading_stops) == 1


def test_an_edit_to_one_of_two_branches_of_one_outcome_is_drift():
    """107 of 680 junctions have an outcome reached two ways, and matched on
    the outcome alone the last branch stood for all: ``add_sub.py::main``
    could lose the condition on either way it closes a job and the report
    would have said nothing about the first."""
    old = _fragment_of(
        _junction_for_diff(
            [
                Branch(outcome="closed", path_condition=["a"], order=0),
                Branch(outcome="closed", path_condition=["b"], order=1),
            ]
        )
    )
    new = _fragment_of(
        _junction_for_diff(
            [
                Branch(outcome="closed", path_condition=["a2"], order=0),
                Branch(outcome="closed", path_condition=["b"], order=1),
            ]
        )
    )

    result = diff.compare(old, new)

    (drifted,) = result.drift()
    assert drifted.node == "j:setStatus → closed"
    assert (drifted.before, drifted.after) == ("[a]", "[a2]")
    assert diff.compare(old, old).identical


def test_a_second_branch_to_the_same_outcome_is_reported():
    one = _junction_for_diff([Branch(outcome="closed", path_condition=["a"], order=0)])
    two = _junction_for_diff(
        [
            Branch(outcome="closed", path_condition=["a"], order=0),
            Branch(outcome="closed", path_condition=["b"], order=1),
        ]
    )

    (change,) = diff.compare(_fragment_of(one), _fragment_of(two)).changes

    assert (change.field, change.before, change.after) == ("branch", "—", "closed")


def test_a_reworded_branch_message_is_reported():
    """The message is what a reader names the branch by from a log."""
    old = _fragment_of(
        _junction_for_diff([Branch(outcome="ready", messages=["set ready"], order=0)])
    )
    new = _fragment_of(
        _junction_for_diff([Branch(outcome="ready", messages=["ready now"], order=0)])
    )

    (change,) = diff.compare(old, new).changes

    assert change.field == "messages"


def test_an_entry_that_hands_over_something_else_is_reported():
    """Same door, different arguments: a guard that fires on one path only."""
    def junction(binding):
        node = _junction_for_diff([Branch(outcome="ready", order=0)])
        node.entry_points = [
            EntryPoint(
                trigger="polled",
                entry="JobGenerator.py",
                via="start",
                arg_binding=binding,
                reached_by="dispatch",
            )
        ]
        return node

    (change,) = diff.compare(
        _fragment_of(junction({"inputList": "inputList"})),
        _fragment_of(junction({"inputList": "inputList", "resource_types": "r"})),
    ).changes

    assert change.field == "arg_binding"
    assert change.node == "j:setStatus ← polled JobGenerator.py"


# ---------------------------------------------------------------------------
# Comparing two builds of the *same* source, made by two versions of this code
# ---------------------------------------------------------------------------


def _dumped(tmp_path):
    """Run ``diff-map --dump`` and return the file it wrote."""
    target = tmp_path / "before.json"
    result = click.testing.CliRunner().invoke(diff_map.main, ["--dump", str(target)])
    assert result.exit_code == 0, result.output
    return target


def test_a_dumped_build_reads_back_as_the_same_map(tmp_path, monkeypatch):
    """``--dump`` has to be lossless or the baseline argues for the change.

    A round that moves the map cannot hold both builds at once -- they need
    two checkouts of *this* repository -- so the old side is a file, and a
    field the file drops reads as a difference the round made.
    """
    built = _fragment_of(
        _stage_for_diff("-disk", ["a < b"], 100),
        _stage_for_diff("-rse", ["c"], 200),
    )
    monkeypatch.setattr(diff_map, "_build", lambda map_id, root, schema, record: built)

    target = _dumped(tmp_path)

    assert diff.compare(built, diff_map._load(target)).identical


def test_a_baseline_file_stands_in_for_the_older_build(tmp_path, monkeypatch):
    """The point of the pair: the source is fixed and the map moved anyway."""
    before = _fragment_of(_stage_for_diff("-disk", ["a < b"], 100))
    monkeypatch.setattr(diff_map, "_build", lambda map_id, root, schema, record: before)
    target = _dumped(tmp_path)

    after = _fragment_of(_stage_for_diff("-disk", ["a <= b"], 100))
    monkeypatch.setattr(diff_map, "_build", lambda map_id, root, schema, record: after)
    result = click.testing.CliRunner().invoke(
        diff_map.main, ["--baseline", str(target)]
    )

    assert result.exit_code == 0, result.output
    assert "condition drift (1)" in result.output
    assert "a <= b" in result.output


def test_comparing_a_baseline_with_a_source_tree_is_refused(tmp_path, monkeypatch):
    """Both name the other side, so taking both would silently use one.

    ``--against`` stopped being required when the two flags arrived, and the
    failure that guards against is a run with neither: two builds of the same
    source by the same code, reported as no difference at all.
    """
    monkeypatch.setattr(
        diff_map, "_build", lambda map_id, root, schema, record: _fragment_of(_stage_for_diff("-d", ["a"], 1))
    )
    target = _dumped(tmp_path)
    runner = click.testing.CliRunner()

    both = runner.invoke(
        diff_map.main, ["--baseline", str(target), "--against", str(tmp_path)]
    )
    neither = runner.invoke(diff_map.main, [])

    assert both.exit_code != 0
    assert "--baseline" in both.output and "--against" in both.output
    assert neither.exit_code != 0
    assert "--dump" in neither.output


# ---------------------------------------------------------------------------
# Reading absence correctly, and the gates that depend on it
# ---------------------------------------------------------------------------
#
# Seeing a line proves it is emitted however little of the log was read.  Not
# seeing one proves nothing unless everything the query matched came back.
# Every gate below turns on that asymmetry.


def _tag_sample(lines: list[str], filename: str = BROKER_LOG, **kwargs):
    return _sample(lines, filename, pattern=evidence.TAG_PATTERN, **kwargs)


def test_the_level_query_keeps_a_histogram_and_no_lines():
    """A 112 MB evidence file was 626,041 lines standing in for twenty rows."""
    (query,) = evidence.sample_queries({BROKER_LOG: evidence.JEDI})

    assert query.keep_lines == 0
    assert query.tail_bytes == evidence.LEVEL_TAIL_BYTES


def test_the_level_window_is_small_so_the_cap_is_not_reached():
    """``tail -c N | rg -m M`` returns the *first* M matches inside the window.

    A wide window with a cap therefore yields the oldest lines in it and comes
    back truncated, and a truncated sample cannot license "production does not
    emit this".  The window is what has to be small, not the cap.
    """
    assert evidence.LEVEL_TAIL_BYTES < evidence.DEFAULT_TAIL_BYTES
    assert evidence.LEVEL_MAX_MATCHES > evidence.DEFAULT_MAX_MATCHES


def test_a_kept_line_budget_does_not_hide_that_something_matched():
    """``matched`` is read wherever the question is "did anything match", so a
    query keeping no lines still answers it."""
    query = evidence.GrepQuery(
        pattern=evidence.ANY_LINE_PATTERN,
        log_filename=BROKER_LOG,
        service=evidence.JEDI,
        keep_lines=0,
    )
    payload = {
        "expected_machines": ["m1"],
        "results": [
            {
                "machine_name": "m1",
                "result": _log_line("INFO", "a") + "\n" + _log_line("DEBUG", "b") + "\n",
                "return_code": 0,
            }
        ],
    }

    (result,) = evidence._results_from(query, payload)

    assert result.lines == []
    assert result.matched == 2
    assert result.level_counts == {"INFO": 1, "DEBUG": 1}


def test_a_suppressed_observable_needs_a_complete_sample_to_fail():
    """The bug this exists to stop: a capped sample that happened to contain no
    DEBUG line would have been read as "production runs at INFO"."""
    fragment = MapFragment(
        map_id=MAP_ID,
        derived_from=VERSION,
        filter_stages=[_stage("-diskIO", "debug", BROKER, [BROKER_LOG])],
    )

    complete = gates.observables_are_emitted(
        fragment, _evidence(_sample([_log_line("INFO", "a")]))
    )
    partial = gates.observables_are_emitted(
        fragment, _evidence(_sample([_log_line("INFO", "a")], truncated=True))
    )

    assert not complete.passed
    assert partial.passed
    assert "sample incomplete" in partial.inconclusive[0]


def test_seeing_the_line_needs_no_complete_sample():
    """The other half of the asymmetry: positive evidence stands on its own."""
    fragment = MapFragment(
        map_id=MAP_ID,
        derived_from=VERSION,
        filter_stages=[_stage("-diskIO", "debug", BROKER, [BROKER_LOG])],
    )
    ev = _evidence(_sample([_log_line("DEBUG", "a")], truncated=True))

    result = gates.observables_are_emitted(fragment, ev)

    assert result.passed
    assert result.inconclusive == []


def test_a_tag_production_emits_with_no_stage_is_a_blind_spot():
    """The system naming a cut the map cannot explain.

    Positive evidence, so it fails the gate even from a partial read.
    """
    fragment = MapFragment(
        map_id=MAP_ID,
        derived_from=VERSION,
        filter_stages=[_stage("-disk", "info", BROKER, [BROKER_LOG])],
    )
    ev = _evidence(
        _tag_sample(
            [
                _log_line("INFO", "skip site=X criteria=-disk"),
                _log_line("INFO", "skip site=Y criteria=-newcut"),
            ],
            truncated=True,
        )
    )

    result = gates.tags_are_known(fragment, ev)

    assert not result.passed
    assert result.failures == ["production logs -newcut (1x) and the map has no stage for it"]


def test_a_tag_the_map_has_and_production_did_not_show_needs_a_complete_sample():
    """Otherwise the gate accuses the extraction of inventing a stage that is
    merely quiet."""
    fragment = MapFragment(
        map_id=MAP_ID,
        derived_from=VERSION,
        filter_stages=[
            _stage("-disk", "info", BROKER, [BROKER_LOG]),
            _stage("-rse", "info", BROKER, [BROKER_LOG]),
        ],
    )
    lines = [_log_line("INFO", "skip site=X criteria=-disk")]

    partial = gates.tags_are_known(fragment, _evidence(_tag_sample(lines, truncated=True)))
    complete = gates.tags_are_known(fragment, _evidence(_tag_sample(lines)))

    assert partial.inconclusive == []
    # Stays in ``inconclusive`` even when the sample is whole, and says "this
    # window" rather than "production never": writing the line needs the branch
    # to fire, so a quiet cut and a wrong tag look the same at any sample size.
    assert complete.inconclusive == [
        f"{BROKER_LOG}: the map has -rse and this window shows no cut using it"
    ]
    assert complete.passed
    assert complete.sample == gates.COMPLETE
    assert partial.sample == gates.PARTIAL


def _transition(task: str, status: str, stamp: str, filename: str = "panda-ContentsFeeder.log"):
    """One ``set task_status=`` line, as the knights write it."""
    return _sample(
        [
            _log_line(
                "INFO", f"<jediTaskID={task}> set task_status={status}", name="ContentsFeeder"
            )
        ],
        filename=filename,
        pattern=evidence.TRANSITION_PATTERN,
    ).model_copy(
        update={
            "lines": [
                f"2026-08-30 {stamp} ContentsFeeder: INFO     "
                f"<jediTaskID={task}> set task_status={status}"
            ]
        }
    )


def _task_status_fragment(outcomes, vocabulary=(), selected=(), runtime=()):
    branches = [Branch(outcome=o, tier=1) for o in outcomes]
    branches += [Branch(outcome=f"runtime({name})", tier=2) for name in runtime]
    return MapFragment(
        map_id=MAP_ID,
        derived_from=VERSION,
        subjects=[
            SubjectNode(
                map_id=MAP_ID,
                derived_from=VERSION,
                name=evidence.TRANSITION_SUBJECT,
                spec_class="JediTaskSpec",
                attribute="status",
                vocabulary=list(vocabulary),
                selected_values=list(selected),
            )
        ],
        junctions=[
            JunctionNode(
                map_id=MAP_ID,
                derived_from=VERSION,
                name="j",
                owner="pandajedi/jediorder/ContentsFeeder.py::feed",
                subject=evidence.TRANSITION_SUBJECT,
                branches=branches,
            )
        ],
    )


def test_a_task_history_is_rebuilt_across_files_and_machines():
    """The database keeps the current status and the one before it, so a
    sequence only exists in the log -- and one task's history is spread over the
    refiner, the generator and the post-processor, on whichever machine picked it
    up.  Ordering therefore comes from the timestamps, not from the order the
    lines came back in."""
    ev = _evidence(
        _transition("7", "running", "12:00:09,000", "panda-JobGenerator.log"),
        _transition("7", "defined", "12:00:00,000", "panda-TaskRefiner.log"),
        _transition("7", "ready", "12:00:05,000"),
        # Two knights logging the same value in a row is not a transition.
        _transition("7", "ready", "12:00:06,000"),
    )

    histories = evidence.observed_task_status(ev)

    assert [status for _stamp, status, _file in histories["7"]] == [
        "defined",
        "ready",
        "running",
    ]
    assert evidence.observed_pairs(histories) == Counter(
        {("defined", "ready"): 1, ("ready", "running"): 1}
    )


def test_a_departure_survives_a_gap_and_a_pair_does_not():
    """Seeing a task in one status and later in another proves it left the
    first, whether or not the step between was sampled.  The pair is what a gap
    invents, which is why one is evidence and the other a report."""
    histories = evidence.observed_task_status(
        _evidence(
            _transition("7", "defined", "12:00:00,000"),
            _transition("7", "running", "12:00:09,000"),
        )
    )

    assert evidence.observed_departures(histories) == Counter({"defined": 1})


def test_a_status_production_sets_and_no_branch_produces_is_a_finding():
    """What gate nine is for, and what it added over the offline check: a value
    the code declares and production sets, which no branch produces, cannot be a
    stale declaration any more."""
    fragment = _task_status_fragment(
        outcomes=["toabort"], vocabulary=["toabort", "aborted"], runtime=["newTaskStatus"]
    )
    ev = _evidence(
        _transition("7", "toabort", "12:00:00,000"),
        _transition("7", "aborted", "12:00:05,000"),
    )

    result = gates.transitions_are_explained(fragment, ev)

    assert not result.passed
    assert result.failures == [
        "production sets aborted (1x), the code declares it, and no branch in the map "
        "produces it (1 writer(s) of this subject decide the value at run time)"
    ]
    assert result.kind == gates.MAP_DEFECT


def test_an_undeclared_status_may_be_a_run_time_value():
    """Tier 2 is by design -- the writer is in the map and the outcome is not --
    so a value nothing declares is not evidence of a missing writer."""
    fragment = _task_status_fragment(outcomes=["defined"], runtime=["newTaskStatus"])
    ev = _evidence(
        _transition("7", "defined", "12:00:00,000"),
        _transition("7", "surprising", "12:00:05,000"),
    )

    result = gates.transitions_are_explained(fragment, ev)

    assert result.passed
    assert "may be a value one of the 1 run-time writer(s) computes" in result.inconclusive[0]


def test_a_status_nothing_selects_on_is_not_a_failure():
    """A task can be moved on by a junction that selects it by id or on a
    command, with no status predicate for the map to have missed."""
    fragment = _task_status_fragment(
        outcomes=["defined", "running"], selected=["defined"]
    )
    ev = _evidence(
        _transition("7", "defined", "12:00:00,000"),
        _transition("7", "running", "12:00:05,000"),
        _transition("7", "defined", "12:00:09,000"),
    )

    result = gates.transitions_are_explained(fragment, ev)

    assert result.passed
    assert any("moved tasks out of running" in row for row in result.inconclusive)


def test_conformance_passes_when_production_stays_inside_the_map():
    fragment = _task_status_fragment(
        outcomes=["defined", "ready", "running"], selected=["defined", "ready"]
    )
    ev = _evidence(
        _transition("7", "defined", "12:00:00,000"),
        _transition("7", "ready", "12:00:05,000"),
        _transition("7", "running", "12:00:09,000"),
    )

    result = gates.transitions_are_explained(fragment, ev)

    assert result.passed and result.failures == []
    assert result.checked == 3


def test_a_transposed_funnel_step_is_a_finding():
    """"Which step cut the candidates" is a question about position, so an
    order that disagrees makes every answer off by one step.

    Two traversals rather than one, because a chain that really runs its steps
    out of order does so every time, while a single reversal is what one wrap
    of a correctly-ordered chain produces.
    """
    fragment = MapFragment(
        map_id=MAP_ID,
        derived_from=VERSION,
        filter_stages=[
            _stage("-a", "info", BROKER, [BROKER_LOG], funnel_label="disk check", order=0),
            _stage("-b", "info", BROKER, [BROKER_LOG], funnel_label="memory check", order=1),
        ],
    )
    ev = _evidence(
        _sample(
            [
                _log_line("INFO", f"<jediTaskID={task}> {n} candidates passed {label}")
                for task in (1, 2)
                for n, label in ((100, "memory check"), (80, "disk check"))
            ],
            pattern=evidence.FUNNEL_PATTERN,
        )
    )

    result = gates.funnel_order_matches(fragment, ev)

    assert not result.passed
    assert "after" in result.failures[0]


def test_repeated_traversals_do_not_read_as_transpositions():
    """Brokerage walks the chain many times under one task and dataset -- once
    observed thirteen times -- and the log marks no boundary between them.

    Every attempt to cut the run into traversals cost false findings: read as
    one sequence 22, splitting on a return to the start 8, splitting on the
    size of the backward jump 2.  A majority needs no boundary, because each
    wrap is one reversed pair against many in order.
    """
    fragment = MapFragment(
        map_id=MAP_ID,
        derived_from=VERSION,
        filter_stages=[
            _stage("-a", "info", BROKER, [BROKER_LOG], funnel_label="status check", order=0),
            _stage("-b", "info", BROKER, [BROKER_LOG], funnel_label="backlog check", order=1),
        ],
    )
    # Three traversals of a two-step chain: the wrap from "backlog" back to
    # "status" is exactly the shape a size rule cannot tell from a swap.
    run = ["status check", "backlog check"] * 3
    ev = _evidence(
        _sample(
            [_log_line("INFO", f"<jediTaskID=1 datasetID=2> 5 candidates passed {label}") for label in run],
            pattern=evidence.FUNNEL_PATTERN,
        )
    )

    assert gates.funnel_order_matches(fragment, ev).passed


def test_two_chains_in_one_log_file_are_told_apart():
    """One log file is not one chain.  ``AtlasProdTaskBroker`` runs its own
    steps and then calls the job broker, whose steps are written through the log
    slot it was handed, so both chains land in the task broker's file.  Both
    begin with a step named ``status check``, so read as one chain every
    traversal contributed one pair in order and one reversed -- 4836 against
    4833 in production, a majority decided by nothing.

    The map cannot say which chains share a file; the evidence can, without
    segmenting anything.  A label only one chain uses names that chain, so the
    chains present are the ones whose unique labels appear, and a label shared
    by two of *those* is dropped.
    """
    task_log, job_log = "panda-AtlasProdTaskBroker.log", "panda-AtlasProdJobBroker.log"
    task_chain = "pandajedi/jedibrokerage/AtlasProdTaskBroker.py::runImpl"
    job_chain = "pandajedi/jedibrokerage/AtlasProdJobBroker.py::doBrokerage"
    fragment = MapFragment(
        map_id=MAP_ID,
        derived_from=VERSION,
        filter_stages=[
            _stage("-x", "info", task_chain, [task_log], funnel_label="status check", order=0),
            _stage("-y", "info", task_chain, [task_log], funnel_label="backlog check", order=1),
            _stage("-z", "info", task_chain, [task_log], funnel_label="endpoint check", order=2),
            _stage("-p", "info", job_chain, [job_log], funnel_label="status check", order=0),
            _stage("-q", "info", job_chain, [job_log], funnel_label="opportunistic check", order=1),
            _stage("-r", "info", job_chain, [job_log], funnel_label="memory check", order=2),
        ],
    )
    # What production shows, twice over: the task broker's three steps, then the
    # job broker's three, the second of which starts with the shared label.
    run = [
        "status check", "backlog check", "endpoint check",
        "status check", "opportunistic check", "memory check",
    ] * 3
    ev = _evidence(
        _sample(
            [
                _log_line("INFO", f"<jediTaskID=1> 5 candidates passed {label}")
                for label in run
            ],
            filename=task_log,
            pattern=evidence.FUNNEL_PATTERN,
        )
    )

    result = gates.funnel_order_matches(fragment, ev)

    assert result.passed, result.failures
    # Both chains were recognised as writing here, and only the shared label was
    # given up -- the unshared ones are still compared.
    assert result.checked > 0


def test_a_transposition_in_a_shared_file_is_still_found():
    """Dropping the ambiguous label must not turn the gate off: the steps only
    one of the two chains names are still compared."""
    task_log, job_log = "panda-AtlasProdTaskBroker.log", "panda-AtlasProdJobBroker.log"
    task_chain = "pandajedi/jedibrokerage/AtlasProdTaskBroker.py::runImpl"
    job_chain = "pandajedi/jedibrokerage/AtlasProdJobBroker.py::doBrokerage"
    fragment = MapFragment(
        map_id=MAP_ID,
        derived_from=VERSION,
        filter_stages=[
            _stage("-x", "info", task_chain, [task_log], funnel_label="status check", order=0),
            _stage("-y", "info", task_chain, [task_log], funnel_label="backlog check", order=1),
            _stage("-z", "info", task_chain, [task_log], funnel_label="endpoint check", order=2),
            _stage("-p", "info", job_chain, [job_log], funnel_label="status check", order=0),
        ],
    )
    # "endpoint" consistently before "backlog", against the map.
    run = ["status check", "endpoint check", "backlog check"] * 3
    ev = _evidence(
        _sample(
            [
                _log_line("INFO", f"<jediTaskID=1> 5 candidates passed {label}")
                for label in run
            ],
            filename=task_log,
            pattern=evidence.FUNNEL_PATTERN,
        )
    )

    result = gates.funnel_order_matches(fragment, ev)

    assert not result.passed
    assert "backlog check" in result.failures[0]
    # The chain is named, because a file can hold more than one.
    assert "runImpl" in result.failures[0]


def test_stages_sharing_a_step_count_as_one_position():
    """The funnel counts steps, not stages: ``AtlasAnalJobBroker`` rejects for
    two reasons under "disk check", and a position has to mean the step."""
    fragment = MapFragment(
        map_id=MAP_ID,
        derived_from=VERSION,
        filter_stages=[
            _stage("-a", "info", BROKER, [BROKER_LOG], funnel_label="disk check", order=0),
            _stage("-b", "info", BROKER, [BROKER_LOG], funnel_label="disk check", order=1),
            _stage("-c", "info", BROKER, [BROKER_LOG], funnel_label="memory check", order=2),
        ],
    )
    ev = _evidence(
        _sample(
            [
                _log_line("INFO", "<jediTaskID=1> 9 candidates passed disk check"),
                _log_line("INFO", "<jediTaskID=1> 8 candidates passed memory check"),
            ],
            pattern=evidence.FUNNEL_PATTERN,
        )
    )

    assert gates.funnel_order_matches(fragment, ev).passed


def test_an_early_exit_is_not_a_transposition():
    """The sample spans many tasks and a chain can exit early, so production
    shows a prefix or a gapped run of the map's order."""
    fragment = MapFragment(
        map_id=MAP_ID,
        derived_from=VERSION,
        filter_stages=[
            _stage("-a", "info", BROKER, [BROKER_LOG], funnel_label="disk check", order=0),
            _stage("-b", "info", BROKER, [BROKER_LOG], funnel_label="memory check", order=1),
            _stage("-c", "info", BROKER, [BROKER_LOG], funnel_label="pilot check", order=2),
        ],
    )
    ev = _evidence(
        _sample(
            [
                _log_line("INFO", "100 candidates passed disk check"),
                _log_line("INFO", "40 candidates passed pilot check"),
            ],
            pattern=evidence.FUNNEL_PATTERN,
        )
    )

    assert gates.funnel_order_matches(fragment, ev).passed


def _code_fragment() -> MapFragment:
    """A map that decodes ``taskBufferErrorCode`` and nothing else."""
    return MapFragment(
        map_id=MAP_ID,
        derived_from=VERSION,
        value_enums=[
            ValueEnumNode(
                map_id=MAP_ID,
                derived_from=VERSION,
                name="taskbuffer.ErrorCode.EC_Kill",
                namespace="taskbuffer.ErrorCode.EC",
                constant="EC_Kill",
                value=100,
            )
        ],
        enumeration_writes=[
            EnumerationWrite(
                map_id=MAP_ID,
                derived_from=VERSION,
                field="JobSpec.taskBufferErrorCode",
                constant="EC_Kill",
                namespace="taskbuffer.ErrorCode.EC",
                anchor=Anchor(package="pandaserver", file="x.py", line_start=1),
            )
        ],
    )


def _job_evidence(*jobs: dict) -> evidence.Evidence:
    return evidence.Evidence(
        fetched_at="2026-08-30T00:00:00+00:00",
        records=[evidence.JobRecords(task_id="7", jobs=list(jobs))],
        tasks_available=1,
    )


def test_a_code_the_index_cannot_decode_is_a_finding():
    """P2 means to use the index as a lookup, so a value it cannot name makes
    it answer wrongly rather than not at all."""
    ev = _job_evidence(
        {"PandaID": 1, "taskBufferErrorCode": 100},
        {"PandaID": 2, "taskBufferErrorCode": 999},
    )

    result = gates.error_codes_are_known(_code_fragment(), ev)

    assert not result.passed
    assert result.failures == [
        "production sets taskBufferErrorCode=999 (1x) and "
        "taskbuffer.ErrorCode.EC has no constant with that value"
    ]
    assert result.checked == 2


def test_the_field_decides_which_constant_a_value_means():
    """Why the binding exists at all.  ``100`` was recorded 338 times in one
    sample and meant three different things: ``EC_Kill`` in
    ``taskBufferErrorCode``, ``EC_Setupper`` in ``ddmErrorCode``, ``EC_Watcher``
    in ``jobDispatcherErrorCode``.  Without the field the index is a coin flip.
    """
    fragment = _code_fragment()
    fragment.value_enums.append(
        ValueEnumNode(
            map_id=MAP_ID,
            derived_from=VERSION,
            name="jobdispatcher.ErrorCode.EC_Watcher",
            namespace="jobdispatcher.ErrorCode.EC",
            constant="EC_Watcher",
            value=100,
        )
    )
    fragment.enumeration_writes.append(
        EnumerationWrite(
            map_id=MAP_ID,
            derived_from=VERSION,
            field="JobSpec.jobDispatcherErrorCode",
            constant="EC_Watcher",
            namespace="jobdispatcher.ErrorCode.EC",
            anchor=Anchor(package="pandaserver", file="Watcher.py", line_start=1),
        )
    )

    assert gates._decodes(fragment) == {
        "taskBufferErrorCode": {"taskbuffer.ErrorCode.EC"},
        "jobDispatcherErrorCode": {"jobdispatcher.ErrorCode.EC"},
    }
    # Both hold 100 and both decode, each through its own enumeration.
    ev = _job_evidence(
        {"PandaID": 1, "taskBufferErrorCode": 100, "jobDispatcherErrorCode": 100}
    )
    assert gates.error_codes_are_known(fragment, ev).passed


def test_a_field_the_map_binds_to_no_enumeration_is_not_checked():
    """``pilotErrorCode`` is the pilot's, and the map holds the pilot as a
    boundary on purpose.  Checking it against this index would report every
    value it carries as unknown, which says nothing about the index."""
    ev = _job_evidence({"PandaID": 1, "pilotErrorCode": 1099})

    result = gates.error_codes_are_known(_code_fragment(), ev)

    assert result.passed and result.checked == 0


def test_a_field_at_rest_is_not_a_coded_value():
    """An error-code field holds zero when there was no error, so counting it
    would make every index look as though it were missing the ordinary case."""
    ev = _job_evidence(
        {"PandaID": 1, "taskBufferErrorCode": 0},
        {"PandaID": 2, "taskBufferErrorCode": None},
    )

    result = gates.error_codes_are_known(_code_fragment(), ev)

    assert result.passed and result.checked == 0


def test_a_step_production_counts_and_the_map_lacks_is_a_finding():
    """The funnel counter's positive direction: candidates demonstrably went
    somewhere and the map has no step to name it.

    Sample-size independent -- the line proves the step -- which is why the
    other direction is not reported at all.
    """
    fragment = MapFragment(
        map_id=MAP_ID,
        derived_from=VERSION,
        filter_stages=[
            _stage("-a", "info", BROKER, [BROKER_LOG], funnel_label="disk check", order=0),
        ],
    )
    ev = _evidence(
        _sample(
            [
                _log_line("INFO", "100 candidates passed disk check"),
                _log_line("INFO", "80 candidates passed temporary problem check"),
            ],
            pattern=evidence.FUNNEL_PATTERN,
        )
    )

    result = gates.funnel_steps_are_known(fragment, ev)

    assert not result.passed
    assert result.failures == [
        "production counts a cut at 'temporary problem check' (1x) "
        "and the map has no step for it"
    ]
    # A step the map has and the window does not is never reported: reaching it
    # is what writes the line, so absence has two causes and no sample size
    # separates them.
    assert result.inconclusive == []


def test_a_step_named_after_a_run_time_value_is_matched_by_its_frame():
    """``AtlasProdTaskBroker`` names a step after a threshold it reads from
    configuration, so production writes two names for one step."""
    fragment = MapFragment(
        map_id=MAP_ID,
        derived_from=VERSION,
        filter_stages=[
            _stage(
                "-a",
                "info",
                BROKER,
                [BROKER_LOG],
                funnel_label="endpoint check with DISK_THRESHOLD={} TB",
                order=0,
            ),
        ],
    )
    ev = _evidence(
        _sample(
            [
                _log_line("INFO", "9 candidates passed endpoint check with DISK_THRESHOLD=10 TB"),
                _log_line("INFO", "4 candidates passed endpoint check with DISK_THRESHOLD=1000 TB"),
            ],
            pattern=evidence.FUNNEL_PATTERN,
        )
    )

    assert gates.funnel_steps_are_known(fragment, ev).passed


def test_one_reversed_observation_is_one_traversals_worth_of_noise():
    """Wrapping is not rare here -- pairs the map orders correctly still come
    back 4440 reversed against 6119 in order -- and every traversal that wraps
    contributes exactly one reversal.  A pair whose whole evidence is a single
    reversal is indistinguishable from one wrap, and has no majority in it.
    """
    fragment = MapFragment(
        map_id=MAP_ID,
        derived_from=VERSION,
        filter_stages=[
            _stage("-a", "info", BROKER, [BROKER_LOG], funnel_label="disk check", order=0),
            _stage("-b", "info", BROKER, [BROKER_LOG], funnel_label="memory check", order=1),
        ],
    )
    once = _evidence(
        _sample(
            [
                _log_line("INFO", "100 candidates passed memory check"),
                _log_line("INFO", "80 candidates passed disk check"),
            ],
            pattern=evidence.FUNNEL_PATTERN,
        )
    )
    assert gates.funnel_order_matches(fragment, once).passed

    # Seen again, it is no longer an anecdote.
    twice = _evidence(
        _sample(
            [
                _log_line("INFO", "<jediTaskID=1> 100 candidates passed memory check"),
                _log_line("INFO", "<jediTaskID=1> 80 candidates passed disk check"),
                _log_line("INFO", "<jediTaskID=2> 100 candidates passed memory check"),
                _log_line("INFO", "<jediTaskID=2> 80 candidates passed disk check"),
            ],
            pattern=evidence.FUNNEL_PATTERN,
        )
    )
    assert not gates.funnel_order_matches(fragment, twice).passed


def test_a_step_the_map_places_twice_cannot_testify_about_order():
    """``AtlasProdJobBroker`` runs "temporary problem check" early and returns
    when it was called for a task-brokerage hint, and otherwise runs it last.

    Order takes the first position, production mostly runs the other, and the
    chain is doing exactly what the map says -- so the label is dropped for the
    same reason a label two chains share is.
    """
    fragment = MapFragment(
        map_id=MAP_ID,
        derived_from=VERSION,
        filter_stages=[
            _stage("-a", "info", BROKER, [BROKER_LOG], funnel_label="temporary problem check", order=0),
            _stage("-b", "info", BROKER, [BROKER_LOG], funnel_label="IO check", order=1),
            _stage("-c", "info", BROKER, [BROKER_LOG], funnel_label="temporary problem check", order=2),
        ],
    )
    ev = _evidence(
        _sample(
            [
                _log_line("INFO", "100 candidates passed IO check"),
                _log_line("INFO", "80 candidates passed temporary problem check"),
            ],
            pattern=evidence.FUNNEL_PATTERN,
        )
    )

    result = gates.funnel_order_matches(fragment, ev)

    assert result.passed
    # Nothing left to compare: the other label is alone once the ambiguous one
    # is dropped, so the gate says so rather than passing on a pair it read.
    assert result.checked == 0


def test_a_template_seen_in_production_is_confirmed():
    """The only direction there is.

    A template missing from the log has two causes nothing distinguishes -- the
    wording moved on, or that branch did not fire in the window -- and writing
    the line requires the branch to fire, so no sample size separates them.  So
    only what was seen is returned.
    """
    fragment = MapFragment(
        map_id=MAP_ID,
        derived_from=VERSION,
        filter_stages=[
            _stage(
                "-disk",
                "info",
                BROKER,
                [BROKER_LOG],
                emits=["  skip site={} due to disk shortage criteria=-disk"],
                funnel_label="disk check",
            ),
            _stage(
                "-space",
                "info",
                BROKER,
                [BROKER_LOG],
                emits=["skip nucleus since disk shortage ({0} TB) criteria=-space"],
                funnel_label="space check",
            ),
        ],
    )
    ev = _evidence(
        _tag_sample([_log_line("INFO", "  skip site=X due to disk shortage criteria=-disk")])
    )

    assert gates.templates_confirmed(fragment, ev) == [f"disk check -disk ({BROKER_LOG})"]


def test_two_stages_sharing_a_tag_are_told_apart():
    """``AtlasAnalJobBroker`` emits ``-disk`` from both its scratch-disk and its
    storage-space check, so a row keyed on the tag alone printed one label twice
    and identified neither."""
    fragment = MapFragment(
        map_id=MAP_ID,
        derived_from=VERSION,
        filter_stages=[
            _stage(
                "-disk",
                "info",
                BROKER,
                [BROKER_LOG],
                emits=["skip site={} due to small scratch disk"],
                funnel_label="scratch disk check",
            ),
            _stage(
                "-disk",
                "info",
                BROKER,
                [BROKER_LOG],
                emits=["skip site={} since output endpoint undefined"],
                funnel_label="storage space check",
            ),
        ],
    )
    ev = _evidence(
        _tag_sample(
            [
                _log_line("INFO", "skip site=X due to small scratch disk criteria=-disk"),
                _log_line("INFO", "skip site=Y since output endpoint undefined criteria=-disk"),
            ]
        )
    )

    assert gates.templates_confirmed(fragment, ev) == [
        f"scratch disk check -disk ({BROKER_LOG})",
        f"storage space check -disk ({BROKER_LOG})",
    ]


def test_the_stem_is_the_longest_fixed_run_not_the_prefix():
    """``"  skip site={} due to disk shortage"`` begins with ten characters
    shared by half the file; what identifies the message is on the other side
    of the interpolation.  Taking the prefix skipped nearly every template."""
    assert gates._template_stem("  skip site={} due to disk shortage") == "due to disk shortage"
    # Below the floor, so not distinctive enough to search for.
    assert gates._template_stem("{} sites left") is None
    assert gates._template_stem("skip={}") is None
    assert gates._template_stem("{} of {}") is None


def test_returning_everything_and_writing_it_all_down_are_different():
    """Two halves of the trip.  A gate that searches the lines and concludes
    something is missing needs the stricter one, or trimming to keep_lines
    turns a kept sample into a false absence."""
    kept_all = evidence.GrepResult(
        query=evidence.GrepQuery(
            pattern=evidence.TAG_PATTERN, log_filename=BROKER_LOG, service=evidence.JEDI
        ),
        machine="m1",
        lines=["a", "b"],
        matched=2,
        return_code=0,
    )
    trimmed = kept_all.model_copy(update={"lines": ["a"]})

    assert kept_all.conclusive and kept_all.complete
    assert trimmed.conclusive and not trimmed.complete


def test_an_unconfirmed_template_is_not_reported_at_all():
    """It was reported once, as a ratio with the unconfirmed listed, and that is
    the half a reader cannot act on: a rare rejection reads as a stale template.
    Where drift *can* be detected is between two source trees -- ``diff-map``."""
    fragment = MapFragment(
        map_id=MAP_ID,
        derived_from=VERSION,
        filter_stages=[
            _stage(
                "-space",
                "info",
                BROKER,
                [BROKER_LOG],
                emits=["skip nucleus since disk shortage ({0} TB) criteria=-space"],
            )
        ],
    )
    trimmed = _tag_sample([_log_line("INFO", "something else criteria=-space")]).model_copy(
        update={"matched": 900, "truncated": True}
    )

    assert gates.templates_confirmed(fragment, _evidence(trimmed)) == []


# ---------------------------------------------------------------------------
# What the report says
#
# The gates were right and the report was not: it printed a deployment fact and
# an extraction miss with the same word, never said whether the sample it read
# was whole, and gave five incomparable counts the same label.  These pin the
# distinctions rather than the layout.
# ---------------------------------------------------------------------------


def _result(gate: str, **kwargs) -> gates.GateResult:
    return gates.GateResult(gate=gate, passed=not kwargs.get("failures"), **kwargs)


def test_a_deployment_difference_is_not_a_defect_and_does_not_fail_a_build():
    """The distinction the report exists to make.  ``code-paths-are-live``
    failing means the map is right and this deployment does not run that code;
    there is nothing to change, so it must not read as -- or exit like -- a
    genuine miss."""
    difference = _result(
        "code-paths-are-live", kind=gates.DEPLOYMENT_FACT, failures=["x is on no machine"]
    )
    defect = _result("tags-are-known", kind=gates.MAP_DEFECT, failures=["-newcut has no stage"])
    unusable = _result("log-format-recognised", kind=gates.CHECK_BROKEN, failures=["not authorized"])

    assert difference.verdict == "DIFFERS" and not difference.actionable
    assert defect.verdict == "FAIL" and defect.actionable
    assert unusable.verdict == "BROKEN" and unusable.actionable
    assert _result("funnel-order-matches").verdict == "PASS"


def test_the_real_gates_declare_which_kind_they_are():
    """Set on the gate rather than in the report, because which of the three a
    failure is depends on what the gate compared, not on how it is printed."""
    ev = _evidence(_sample([_log_line("DEBUG", "x")]), _missing("panda-Gone.log"))
    fragment = MapFragment(
        map_id=MAP_ID,
        derived_from=VERSION,
        filter_stages=[_stage("-x", "debug", BROKER, [BROKER_LOG])],
    )

    kinds = {r.gate: r.kind for r in gates.run_production(fragment, ev)}

    assert kinds["log-format-recognised"] == gates.CHECK_BROKEN
    assert kinds["code-paths-are-live"] == gates.DEPLOYMENT_FACT
    assert kinds["observables-are-emitted"] == gates.DEPLOYMENT_FACT
    assert kinds["tags-are-known"] == gates.MAP_DEFECT
    assert kinds["funnel-order-matches"] == gates.MAP_DEFECT


def test_every_count_carries_what_it_counts():
    """Query answers, log files, filter stages, tags and step pairs are not
    comparable quantities; printed as "checked" they invited the comparison."""
    ev = _evidence(_sample([_log_line("DEBUG", "x")]))
    fragment = MapFragment(
        map_id=MAP_ID,
        derived_from=VERSION,
        filter_stages=[_stage("-x", "debug", BROKER, [BROKER_LOG])],
    )

    units = {r.gate: r.unit for r in gates.run_production(fragment, ev)}

    assert units["code-paths-are-live"] == "log files"
    assert units["observables-are-emitted"] == "filter stages"
    assert "checked" not in units.values()
    assert "filter stages" in _result("observables-are-emitted", unit="filter stages").summary()


def test_the_sample_section_says_which_conclusions_are_load_bearing(capsys):
    """The worst of the report's problems was silence here: a gate that could
    not conclude anything looked exactly like one that checked and found
    nothing."""
    ev = _evidence(
        _sample([_log_line("DEBUG", "x")]),
        _tag_sample([_log_line("INFO", "skip criteria=-disk")], truncated=True),
    )

    check_map._report_sample(ev)
    out = capsys.readouterr().out

    assert "levels  complete" in out
    assert "tags    PARTIAL" in out
    assert "1 of 2 answer(s) hit a bound" in out
    # The reason a complete sample would not help either, said once.
    assert "branch has to fire to write" in out


def test_the_verdict_leads_with_the_worst_kind(capsys):
    """A check that did not run makes every other verdict unsafe to read, so it
    outranks a defect; a deployment difference asks for no change at all."""
    results = [
        _result(
            "code-paths-are-live",
            kind=gates.DEPLOYMENT_FACT,
            finding="the map describes code this deployment never runs",
            failures=["a"],
        ),
        _result(
            "tags-are-known",
            kind=gates.MAP_DEFECT,
            finding="the map is missing a cut production makes",
            failures=["b"],
        ),
    ]

    check_map._report_verdict(results)
    out = capsys.readouterr().out

    assert "1 map defect · 1 deployment difference" in out
    assert "the map is missing a cut production makes." in out


def test_a_clean_verdict_says_so(capsys):
    check_map._report_verdict([_result("funnel-order-matches")])

    assert "nothing to change" in capsys.readouterr().out


def test_the_level_table_collapses_to_the_file_that_differs(capsys):
    """Thirty rows whose whole payload was one sentence -- and the ``… 12 more``
    that truncated them was hiding one of the run's two findings."""
    ev = _evidence(
        _sample([_log_line("DEBUG", "x")], filename="panda-A.log"),
        _sample([_log_line("INFO", "y")], filename="panda-B.log"),
    )

    check_map._report_production(ev, [], [], top=10, full=False)
    out = capsys.readouterr().out

    assert "log level    1 at DEBUG, 1 at INFO" in out
    # Only the file that differs from the majority; the rest carry no
    # information one at a time.
    assert "not at DEBUG: panda-B.log" in out
    assert "panda-A.log" not in out


def test_full_puts_every_folded_row_back(capsys):
    ev = _evidence(_sample([_log_line("DEBUG", "x")], filename="panda-A.log"))

    check_map._report_production(ev, [], [f"disk check -disk ({BROKER_LOG})"], top=10, full=True)
    out = capsys.readouterr().out

    assert "panda-A.log" in out and "DEBUG=1" in out
    assert f"disk check -disk ({BROKER_LOG})" in out


def test_agreement_is_reported_one_sided(capsys):
    """No ratio: a line nobody saw may simply not have fired, and the ratio
    invited reading a rare rejection as a stale template."""
    check_map._report_production(_evidence(), [], ["a (f.log)", "b (f.log)"], top=10, full=False)
    out = capsys.readouterr().out

    assert "2 diagnostic line(s) of the map found verbatim" in out
    assert "one-sided" in out
    assert "/" not in out.split("agreement")[1].splitlines()[0]


def test_the_gate_table_names_the_question_and_marks_a_partial_sample(capsys):
    check_map._report_gates(
        [
            _result(
                "tags-are-known",
                checked=31,
                unit="observed tags",
                question="is every cut production makes in the map?",
                sample=gates.PARTIAL,
            ),
            _result("code-paths-are-live", checked=22, unit="log files", kind=gates.DEPLOYMENT_FACT),
        ]
    )
    out = capsys.readouterr().out

    assert "31 observed tags" in out
    assert "is every cut production makes in the map?" in out
    assert "[partial]" in out
    # A gate whose answer is not a sample gets no marker rather than a hedge.
    assert out.splitlines()[-1].count("[") == 0


def test_not_concluded_collapses_to_one_line_per_gate(capsys):
    check_map._report_unconcluded(
        [_result("observables-are-emitted", inconclusive=[f"row {i}" for i in range(7)])],
        full=False,
    )
    out = capsys.readouterr().out

    assert "not concluded (7)" in out
    assert "… 6 more (--full)" in out


def test_the_files_asked_include_the_ones_only_a_caller_names():
    """Otherwise a junction reached only through a caller is never checked.

    Reading ``log_files`` alone left 66 of 495 junctions with a named log that
    production was never asked about -- and the gate that loses most by it is
    ``code-paths-are-live``, since a file no machine has is the one negative
    production can prove and it cannot be proved about a file nobody asked for.
    """
    junction = JunctionNode(
        map_id=MAP_ID,
        derived_from=VERSION,
        name="j",
        subject="JediTaskSpec.status",
        owner="pandaserver/taskbuffer/db_proxy_mods/m.py::updateTaskModTimeJEDI",
        log_files=["panda-DBProxy.log"],
        caller_log_files=["panda-event_picker.log"],
        owns_logger=False,
    )
    declared = {
        "pandaserver/taskbuffer/OraDBProxy.py": "panda-DBProxy.log",
        "pandaserver/dataservice/event_picker.py": "panda-event_picker.log",
        "pandajedi/jediorder/ContentsFeeder.py": "panda-ContentsFeeder.log",
    }

    targets = check_map._targets(
        MapFragment(map_id=MAP_ID, derived_from=VERSION, junctions=[junction]), declared
    )

    assert targets == {
        "panda-DBProxy.log": evidence.SERVER,
        "panda-event_picker.log": evidence.SERVER,
    }


def test_evidence_age_and_bound_sizes_are_readable():
    assert check_map._age("2026-08-30T00:00:00+00:00").endswith("ago")
    assert check_map._age("not a date") == "age unknown"
    assert check_map._size(64 * 1024 * 1024) == "64MB"
    assert check_map._size(1 << 30) == "1GB"
    assert (check_map._count(1, "map defect"), check_map._count(0, "map defect")) == (
        "1 map defect",
        "0 map defects",
    )


def test_a_clipped_row_keeps_the_half_that_says_what_happened():
    """Which is why the gate's rows lead with their reason."""
    row = f"{BROKER_LOG} is on no machine, so -x at {BROKER} never ran here"

    assert check_map._clip(row).startswith(f"{BROKER_LOG} is on no machine")
    assert check_map._clip("short") == "short"


# --------------------------------------------------------------------------- #
# loop cuts -- the fourth filter idiom, the one with no vocabulary
# --------------------------------------------------------------------------- #

_GENERATOR = '''
class JobGenerator:
    def start(self):
        for workQueue in workQueueList:
            for resource_type in resource_types:
                cycleStr = "pid={0} vo={1} queue={2} resource_type={3}".format(
                    self.pid, vo, workQueue.queue_name, resource_type.resource_name
                )
                tmpLog_inner = MsgWrapper(logger, cycleStr)
                if active_gshare_rtypes and not active:
                    if random.uniform(0, 1) > inactive_poll_probability:
                        tmpLog_inner.debug(f"skipping {cycleStr} due to inactivity")
                        continue
                if not flagLocked:
                    tmpLog_inner.debug("skip since locked by another process")
                    continue
                self.doGenerate(workQueue, resource_type)
'''


def _cuts(*sources: tuple[str, str], covered: set[str] | None = None):
    modules = [_module(text, rel) for text, rel in sources]
    return loopcut.extract(modules, MAP_ID, VERSION, covered or set())


def test_a_guarded_continue_in_a_loop_is_a_cut_even_when_it_names_no_candidate():
    """The structure is the proof, not the wording.

    The rule first written down was "the message interpolates the loop
    variable", and it misses both of these: one interpolates ``cycleStr``,
    which is two hops from ``workQueue``, and the other interpolates nothing at
    all.  They are the guards that decide whether a queue generates jobs, which
    is the question the whole idiom was read for.
    """
    cuts, _silent = _cuts((_GENERATOR, "pandajedi/jediorder/JobGenerator.py"))

    assert [c.message for c in cuts] == [
        "skipping {} due to inactivity",
        "skip since locked by another process",
    ]
    assert cuts[1].conditions == ["not flagLocked"]
    assert cuts[0].order == 0 and cuts[1].order == 1


def test_the_prefix_the_logger_stamps_is_the_key_that_narrows_the_question():
    """Which candidate a line is about is in the prefix, not the sentence.

    ``"skip since locked by another process"`` says nothing about which queue
    was locked.  The wrapper built one line earlier says it, and it is read
    through ``.format`` because that template is what the line is stamped with.
    """
    cuts, _silent = _cuts((_GENERATOR, "pandajedi/jediorder/JobGenerator.py"))

    # ``{0}`` normalised to ``{}``: the rest of the map renders holes one way,
    # and two spellings would make a consumer build the pattern twice.
    assert cuts[1].scope_prefix == "pid={} vo={} queue={} resource_type={}"


def test_an_unguarded_continue_is_iteration_control_and_not_a_cut():
    source = (
        "def run(self):\n"
        "    for item in items:\n"
        "        tmpLog.debug(f'done with {item}')\n"
        "        continue\n"
    )
    cuts, silent = _cuts((source, "pandaserver/daemons/scripts/x.py"))

    assert cuts == [] and silent == 0


def test_a_continue_in_an_except_handler_is_not_a_cut():
    """A candidate dropped because the code raised was not tested and rejected.

    Falls out rather than being excluded by hand -- a handler is not an ``if``.
    """
    source = (
        "def run(self):\n"
        "    for item in items:\n"
        "        try:\n"
        "            self.process(item)\n"
        "        except Exception:\n"
        "            tmpLog.error(f'failed to process {item}')\n"
        "            continue\n"
    )
    cuts, silent = _cuts((source, "pandaserver/daemons/scripts/x.py"))

    assert cuts == [] and silent == 0


def test_a_message_assembled_into_a_local_first_is_the_same_line():
    """37% of matching emits are written this way, so the hop is not optional."""
    source = (
        "def run(self):\n"
        "    for dataset in datasets:\n"
        "        if dataset.status != 'ready':\n"
        "            tmpMsg = f'skip {dataset.name} due to dataset status: {dataset.status}'\n"
        "            tmpLog.debug(tmpMsg)\n"
        "            continue\n"
    )
    cuts, _silent = _cuts((source, "pandaserver/dataservice/closer.py"))

    assert [c.message for c in cuts] == ["skip {} due to dataset status: {}"]
    assert cuts[0].log_level == "debug"


def test_two_sites_wording_a_cut_alike_are_one_node_and_pool_their_guards():
    """Production cannot tell them apart either, so neither does the map.

    The message is the signature for the reason a tag is one over in the stage
    slice: it is what production prints and what a reader greps.
    """
    source = (
        "def runImpl(self):\n"
        "    for task in tasks:\n"
        "        if not self.lockA(task):\n"
        "            tmpLog.debug('skip due to lock failure')\n"
        "            continue\n"
        "        if not self.lockB(task):\n"
        "            tmpLog.debug('skip due to lock failure')\n"
        "            continue\n"
    )
    cuts, _silent = _cuts((source, "pandajedi/jediorder/JobGenerator.py"))

    assert len(cuts) == 1
    assert cuts[0].conditions == ["not self.lockA(task)", "not self.lockB(task)"]


def test_a_position_is_not_the_signature():
    """Two builds of the same cut at different lines are the same node.

    A node keyed on where it sits reports a change every time anything above it
    moves, which is the churn ``anchor`` is kept out of ``CONTENT_FIELDS`` to
    avoid.
    """
    body = (
        "    for task in tasks:\n"
        "        if task.frozen:\n"
        "            tmpLog.debug('skip since the task is frozen')\n"
        "            continue\n"
    )
    early, _ = _cuts(("def runImpl(self):\n" + body, "pandajedi/jediorder/X.py"))
    late, _ = _cuts(("def runImpl(self):\n    x = 1\n" + body, "pandajedi/jediorder/X.py"))

    assert early[0].name == late[0].name
    assert early[0].anchor.line_start != late[0].anchor.line_start


def test_a_chain_the_stage_slice_reads_is_not_read_again_here():
    """One cut, one identity.  Keyed on wording here and on the tag there."""
    source = (
        "def doBrokerage(self):\n"
        "    for site in sites:\n"
        "        if site.maxwdir < need:\n"
        "            tmpLog.debug(f'  skip {site.name} due to small scratch disk')\n"
        "            continue\n"
    )
    rel = "pandajedi/jedibrokerage/GenJobBroker.py"
    read, _ = _cuts((source, rel))
    skipped, _ = _cuts((source, rel), covered={f"{rel}::doBrokerage"})

    assert len(read) == 1 and skipped == []


def test_a_tagged_rejection_stays_with_the_slice_that_owns_the_tag():
    source = (
        "def doSomething(self):\n"
        "    for site in sites:\n"
        "        if site.blacklisted:\n"
        "            tmpLog.debug(f'  skip {site.name} criteria=-blacklisted')\n"
        "            continue\n"
    )
    cuts, silent = _cuts((source, "pandajedi/jedidog/X.py"))

    assert cuts == []
    # Counted as unnamed here rather than dropped: this slice did not name it.
    assert silent == 1


def test_the_search_key_is_the_longest_run_and_not_the_leading_one():
    """``"  skip site={} ..."`` shares its front with half the file."""
    source = (
        "def run(self):\n"
        "    for site in sites:\n"
        "        if site.offline:\n"
        "            tmpLog.debug(f'  skip {site.name} because the endpoint is unusable')\n"
        "            continue\n"
    )
    cuts, _silent = _cuts((source, "pandaserver/daemons/scripts/x.py"))

    assert cuts[0].search_key == "because the endpoint is unusable"


def test_a_key_that_also_matches_another_cut_in_the_file_is_reported():
    """No length floor: what makes a key useless is that it selects more.

    A hit on ``collection=`` cannot say which of the two cuts fired and a miss
    cannot say neither did, so the silence is not about this cut at all.
    """
    source = (
        "def run(self):\n"
        "    for collection in collections:\n"
        "        if collection.pseudo:\n"
        "            tmpLog.debug(f'collection={collection} is pseudo input ; skipped')\n"
        "            continue\n"
        "        if not collection.found:\n"
        "            tmpLog.debug(f'collection={collection} not found')\n"
        "            continue\n"
    )
    cuts, _silent = _cuts((source, "pandaserver/taskbuffer/DataCarousel.py"))
    for cut in cuts:
        cut.log_files = ["panda-DataCarousel.log"]
    fragment = MapFragment(map_id=MAP_ID, derived_from=VERSION, loop_cuts=cuts)

    assert gates.ambiguous_cut_keys(fragment) == [
        "panda-DataCarousel.log: 'collection=' also matches another cut there"
    ]


def test_two_cuts_sharing_a_key_in_different_files_are_not_ambiguous():
    """The unit is the file, because that is what a question is asked of."""
    cut = LoopCutNode(
        map_id=MAP_ID,
        derived_from=VERSION,
        name="a",
        owner="a.py::run",
        message="retry {} later",
        search_key="retry",
        log_files=["panda-DBProxy.log"],
    )
    elsewhere = cut.model_copy(
        update={"name": "b", "owner": "b.py::run", "log_files": ["panda-JediDBProxy.log"]}
    )
    fragment = MapFragment(map_id=MAP_ID, derived_from=VERSION, loop_cuts=[cut, elsewhere])

    assert gates.ambiguous_cut_keys(fragment) == []


def test_two_cuts_sharing_a_signature_fail_the_identity_gate():
    """Storing the map keeps one per name, and nothing else compares the two."""
    cut = LoopCutNode(
        map_id=MAP_ID,
        derived_from=VERSION,
        name="panda:x.py::run:skip it",
        owner="x.py::run",
        message="skip it",
    )
    fragment = MapFragment(
        map_id=MAP_ID, derived_from=VERSION, loop_cuts=[cut, cut.model_copy()]
    )

    result = gates.map_identities_are_distinct(fragment)

    assert not result.passed
    assert any("loop_cut" in failure for failure in result.failures)


def test_the_new_node_kind_is_on_every_list_that_has_to_know_about_it():
    """A kind missing from ``CONTENT_FIELDS`` compares equal to itself no matter
    what changed, and one missing from the groups is read as nothing."""
    from bamboo.codemap.models import MAP_NODE_GROUPS

    assert "loop_cuts" in MAP_NODE_GROUPS
    assert NodeType.LOOP_CUT.value in diff.CONTENT_FIELDS


def test_the_function_prints_kind_is_on_every_list_that_has_to_know_about_it():
    from bamboo.codemap import lookup
    from bamboo.codemap.models import MAP_NODE_GROUPS, FunctionPrintsNode

    assert NodeType.FUNCTION_PRINTS.value in diff.CONTENT_FIELDS
    assert "function_prints" in MAP_NODE_GROUPS
    assert lookup._MODELS[FunctionPrintsNode] is NodeType.FUNCTION_PRINTS


def test_a_functions_prints_name_the_files_its_module_writes_to(tmp_path):
    """``archiveJob`` writes ``start status=...`` into the proxy log, not into its callers'.

    The proxy modules declare no logger and the class that mixes them in
    does, so that is where their rows land.  The junction's
    ``observable_log_files`` names the callers' files instead -- where a line
    saying the value was settled appears -- and was all a reader had.
    """
    from bamboo.codemap.panda.plugin import PandaCodeMapPlugin

    sources = {
        "pandaserver/taskbuffer/db_proxy_mods/job_complex_module.py": (
            "class JobComplexModule:\n"
            "    def archiveJob(self, job):\n"
            "        tmp_log.debug(f'start status={job.jobStatus}')\n"
        ),
        "pandaserver/taskbuffer/OraDBProxy.py": (
            '_logger = PandaLogger().getLogger("DBProxy")\n'
            "class DBProxy(job_complex_module.JobComplexModule):\n"
            "    def connect(self):\n"
            "        _logger.info('connected')\n"
        ),
        "pandaserver/daemons/scripts/quiet.py": "def main():\n    tmp_log.info('nothing declared here')\n",
    }
    for rel, text in sources.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    plugin = PandaCodeMapPlugin()
    plugin.prepare(tmp_path)

    found = {node.owner: node.log_files for node in plugin._function_prints()}

    assert found == {
        "pandaserver/taskbuffer/db_proxy_mods/job_complex_module.py::JobComplexModule.archiveJob": [
            "panda-DBProxy.log"
        ],
        "pandaserver/taskbuffer/OraDBProxy.py::DBProxy.connect": ["panda-DBProxy.log"],
        "pandaserver/daemons/scripts/quiet.py::main": [],
    }


def _landed(tmp_path, sources: dict[str, str]) -> dict[str, list[tuple[int, list[str], str]]]:
    """``owner -> [(line, log_files, basis)]`` for each printing function, read with the corpus's calls."""
    import textwrap

    from bamboo.codemap.panda.plugin import PandaCodeMapPlugin
    from bamboo.codemap.panda.recognizers import sqlwrite

    for rel, text in sources.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(text))
    plugin = PandaCodeMapPlugin()
    plugin.prepare(tmp_path)
    plugin._calls = sqlwrite.corpus_calls(plugin._modules, plugin._roots)
    return {
        node.owner: [(s.line, s.log_files, s.log_files_basis) for s in node.statements]
        for node in plugin._function_prints()
    }


def test_a_logger_handed_in_lands_in_the_file_of_the_caller_that_opened_it(tmp_path):
    """``pre_action`` prints through the ``tmpLog`` ``WatchDog`` hands it, so into
    ``panda-WatchDog.log`` -- not into its own module's file.  The call is on a
    plugin chosen at run time, which the call graph places nowhere, so it is
    found by the method's name."""
    found = _landed(
        tmp_path,
        {
            "pandajedi/jediorder/WatchDog.py": """\
                logger = PandaLogger().getLogger(__name__.split(".")[-1])
                class WatchDog:
                    def start(self, impl):
                        tmpLog = MsgWrapper(logger)
                        impl.pre_action(tmpLog, "atlas")
                """,
            "pandajedi/jedidog/TypicalWatchDogBase.py": """\
                logger = PandaLogger().getLogger(__name__.split(".")[-1])
                class TypicalWatchDogBase:
                    def pre_action(self, tmpLog, vo):
                        tmpLog.info(f"rescue for vo={vo}")
                """,
        },
    )

    assert found["pandajedi/jedidog/TypicalWatchDogBase.py::TypicalWatchDogBase.pre_action"] == [
        (4, ["panda-WatchDog.log"], "logger")
    ]


_ADDER_PLUGINS = {
    "pandaserver/dataservice/adder_plugin_base.py": """\
        class AdderPluginBase:
            def __init__(self, job, params):
                self.job = job
                self.logger = None
                for key, value in params.items():
                    setattr(self, key, value)
        """,
    "pandaserver/dataservice/adder_atlas_plugin.py": """\
        from pandaserver.dataservice.adder_plugin_base import AdderPluginBase
        class AdderAtlasPlugin(AdderPluginBase):
            def __init__(self, job, **params):
                AdderPluginBase.__init__(self, job, params)
            def execute(self):
                self.logger.debug("start")
        """,
}


def _adder_gen(call: str) -> str:
    return f"""\
        from pandaserver.config import panda_config
        _logger = PandaLogger().getLogger("adder")
        class AdderGen:
            def __init__(self, job):
                self.job = job
                self.logger = LogWrapper(_logger, str(job))
            def get_plugin_class(self, vo):
                self.adder_plugin_class = panda_config.getPlugin("adder_plugins", vo)
                if self.adder_plugin_class is None:
                    from pandaserver.dataservice.adder_atlas_plugin import AdderAtlasPlugin
                    self.adder_plugin_class = AdderAtlasPlugin
                return self.adder_plugin_class
            def run(self, extra):
                self.adder_plugin_class = self.get_plugin_class(self.job.VO)
                self.adder_plugin = {call}
        """


def test_a_plugin_built_from_a_class_getplugin_chose_prints_into_its_builders_file(tmp_path):
    """``setupper_plugin_class(..., tmp_log, ...)`` builds whatever the
    configuration names, or ``SetupperAtlasPlugin``: the plugin prints through
    the ``tmp_log`` handed to its base's constructor, into ``panda-setupper.log``.
    The call was placed nowhere, so its constructor had no caller and its 110
    statements no file."""
    found = _landed(
        tmp_path,
        {
            "pandaserver/dataservice/setupper.py": """\
                from pandaserver.config import panda_config
                _logger = PandaLogger().getLogger("setupper")
                class Setupper:
                    def run(self, vo):
                        tmp_log = LogWrapper(_logger, None)
                        setupper_plugin_class = panda_config.getPlugin("setupper_plugins", vo)
                        if setupper_plugin_class is None:
                            from pandaserver.dataservice.setupper_atlas_plugin import SetupperAtlasPlugin
                            setupper_plugin_class = SetupperAtlasPlugin
                        setupper_plugin = setupper_plugin_class(self.jobs, tmp_log, resubmit=False)
                """,
            "pandaserver/dataservice/setupper_plugin_base.py": """\
                class SetupperPluginBase:
                    def __init__(self, jobs, logger, params):
                        self.jobs = jobs
                        self.logger = logger
                        for key, value in params.items():
                            setattr(self, key, value)
                """,
            "pandaserver/dataservice/setupper_atlas_plugin.py": """\
                from pandaserver.dataservice.setupper_plugin_base import SetupperPluginBase
                class SetupperAtlasPlugin(SetupperPluginBase):
                    def __init__(self, jobs, logger, **params):
                        SetupperPluginBase.__init__(self, jobs, logger, params)
                    def run(self):
                        self.logger.debug("start")
                """,
        },
    )

    assert found["pandaserver/dataservice/setupper_atlas_plugin.py::SetupperAtlasPlugin.run"] == [
        (6, ["panda-setupper.log"], "logger")
    ]


def test_a_field_installed_from_the_constructors_keywords_is_the_keyword_each_builder_passes(tmp_path):
    """``AdderPluginBase`` sets ``self.logger = None`` and then installs every
    keyword it was handed as a field, and ``AdderGen`` hands ``logger=``: the
    adder plugins print into ``panda-adder.log``.  The ``None`` written before
    the loop prints nothing, so the loop's value is the file."""
    found = _landed(
        tmp_path,
        {
            **_ADDER_PLUGINS,
            "pandaserver/dataservice/adder_gen.py": _adder_gen("self.adder_plugin_class(self.job, logger=self.logger)"),
        },
    )

    assert found["pandaserver/dataservice/adder_atlas_plugin.py::AdderAtlasPlugin.execute"] == [
        (6, ["panda-adder.log"], "logger")
    ]


def test_keywords_a_builder_passes_unread_leave_the_installed_field_partly_followed(tmp_path):
    """``**extra`` may hold a ``logger`` too, so the file found is not the only one."""
    found = _landed(
        tmp_path,
        {
            **_ADDER_PLUGINS,
            "pandaserver/dataservice/adder_gen.py": _adder_gen(
                "self.adder_plugin_class(self.job, logger=self.logger, **extra)"
            ),
        },
    )

    assert found["pandaserver/dataservice/adder_atlas_plugin.py::AdderAtlasPlugin.execute"] == [
        (6, ["panda-adder.log"], "partly")
    ]


def test_a_helper_handed_its_own_modules_logger_stays_in_that_file(tmp_path):
    """Most loggers handed in are the module's own, given to a helper beside it."""
    found = _landed(
        tmp_path,
        {
            "pandaserver/api/v1/file_server_api.py": """\
                _logger = PandaLogger().getLogger("api_file_server")
                def _get_content_length(req, tmp_logger):
                    tmp_logger.debug("length")
                def upload(req):
                    tmp_logger = LogWrapper(_logger, "upload")
                    _get_content_length(req, tmp_logger)
                """,
        },
    )

    assert found["pandaserver/api/v1/file_server_api.py::_get_content_length"] == [
        (3, ["panda-api_file_server.log"], "logger")
    ]


def test_a_logger_a_caller_may_pass_adds_that_callers_file_to_the_functions_own(tmp_path):
    """``doBrokerage`` opens its own logger unless it is given ``glLog``, and the
    task broker gives it one: its lines are in either file.  A caller that
    passes nothing leaves the default ``None``, which prints nothing anywhere."""
    found = _landed(
        tmp_path,
        {
            "pandajedi/jedibrokerage/AtlasProdJobBroker.py": """\
                logger = PandaLogger().getLogger(__name__.split(".")[-1])
                class AtlasProdJobBroker:
                    def doBrokerage(self, taskSpec, glLog=None):
                        if glLog is None:
                            tmpLog = MsgWrapper(logger)
                        else:
                            tmpLog = glLog
                        tmpLog.debug("start")
                """,
            "pandajedi/jedibrokerage/AtlasProdTaskBroker.py": """\
                from pandajedi.jedibrokerage.AtlasProdJobBroker import AtlasProdJobBroker
                logger = PandaLogger().getLogger(__name__.split(".")[-1])
                class AtlasProdTaskBroker:
                    def doBrokerage(self, taskSpec):
                        tmpLog = MsgWrapper(logger)
                        jobBroker = AtlasProdJobBroker()
                        jobBroker.doBrokerage(taskSpec, tmpLog)
                        jobBroker.doBrokerage(taskSpec)
                """,
        },
    )

    assert found["pandajedi/jedibrokerage/AtlasProdJobBroker.py::AtlasProdJobBroker.doBrokerage"] == [
        (8, ["panda-AtlasProdJobBroker.log", "panda-AtlasProdTaskBroker.log"], "logger")
    ]


def test_a_parameter_keeps_what_the_caller_passed_unless_every_path_rebinds_it(tmp_path):
    """``if tmpLog is None: tmpLog = ...`` binds the parameter on one branch, so
    the caller's logger still reaches the call; a binding on every path
    replaces it."""
    found = _landed(
        tmp_path,
        {
            "pandaserver/taskbuffer/InputChunk.py": """\
                _logger = PandaLogger().getLogger("InputChunk")
                class InputChunk:
                    def getSubChunk(self, tmpLog=None):
                        if tmpLog is None:
                            tmpLog = LogWrapper(_logger, "sub")
                        tmpLog.debug("sub chunk")
                    def own(self, tmpLog):
                        tmpLog = LogWrapper(_logger, "own")
                        tmpLog.debug("own chunk")
                """,
            "pandajedi/jediorder/JobSplitter.py": """\
                from pandaserver.taskbuffer.InputChunk import InputChunk
                logger = PandaLogger().getLogger(__name__.split(".")[-1])
                class JobSplitter:
                    def doSplit(self):
                        tmpLog = MsgWrapper(logger)
                        chunk = InputChunk()
                        chunk.getSubChunk(tmpLog)
                        chunk.own(tmpLog)
                """,
        },
    )

    assert found["pandaserver/taskbuffer/InputChunk.py::InputChunk.getSubChunk"] == [
        (6, ["panda-InputChunk.log", "panda-JobSplitter.log"], "logger")
    ]
    assert found["pandaserver/taskbuffer/InputChunk.py::InputChunk.own"] == [
        (9, ["panda-InputChunk.log"], "logger")
    ]


def test_a_factory_prints_into_the_file_of_every_subclass_that_constructs_it(tmp_path):
    """``FactoryBase.initializeMods`` prints through the logger each JEDI factory
    hands ``FactoryBase.__init__`` -- and the factories derive from
    ``FactoryBase[...]``.  With the subscripted base unread, one placed caller
    stood for all of them, and the lines read as landing in that one file."""
    found = _landed(
        tmp_path,
        {
            "pandajedi/jedicore/FactoryBase.py": """\
                class FactoryBase(Generic[T]):
                    def __init__(self, vos, logger):
                        self.logger = MsgWrapper(logger)
                    def initializeMods(self):
                        self.logger.error("wrong config definition")
                """,
            "pandajedi/jediorder/JobBroker.py": """\
                from pandajedi.jedicore.FactoryBase import FactoryBase
                logger = PandaLogger().getLogger(__name__.split(".")[-1])
                class JobBroker(FactoryBase[JobBrokerBase]):
                    def __init__(self, vos):
                        FactoryBase.__init__(self, vos, logger)
                """,
            "pandajedi/jediorder/TaskSetupper.py": """\
                from pandajedi.jedicore.FactoryBase import FactoryBase
                logger = PandaLogger().getLogger(__name__.split(".")[-1])
                class TaskSetupper(FactoryBase[TaskSetupperBase]):
                    def __init__(self, vos):
                        super().__init__(vos, logger)
                """,
        },
    )

    assert found["pandajedi/jedicore/FactoryBase.py::FactoryBase.initializeMods"] == [
        (5, ["panda-JobBroker.log", "panda-TaskSetupper.log"], "logger")
    ]


def test_a_mixins_logger_comes_down_its_constructors_and_from_the_module_that_rebinds_it(tmp_path):
    """The proxy mixins print through ``self._log_stream``, which ``OraDBProxy.DBProxy``
    hands up through ``super().__init__`` -- and ``JediDBProxy`` rebinds
    ``OraDBProxy._logger`` to its own, so in a JEDI process the same lines land in
    ``panda-JediDBProxy.log``.  A sibling mixin's constructor is not this object's."""
    found = _landed(
        tmp_path,
        {
            "pandaserver/taskbuffer/db_proxy_mods/base_module.py": """\
                class BaseModule:
                    def __init__(self, log_stream):
                        self._log_stream = log_stream
                    def create_tagged_logger(self, comment):
                        tmp_log = LogWrapper(self._log_stream, comment)
                        return tmp_log
                """,
            "pandaserver/taskbuffer/db_proxy_mods/job_complex_module.py": """\
                from pandaserver.taskbuffer.db_proxy_mods.base_module import BaseModule
                class JobComplexModule(BaseModule):
                    def __init__(self, log_stream):
                        super().__init__(log_stream)
                    def archiveJob(self, job):
                        tmp_log = self.create_tagged_logger(" /* DBProxy.archiveJob */")
                        tmp_log.debug(f"start status={job.jobStatus}")
                """,
            "pandaserver/taskbuffer/db_proxy_mods/other_module.py": """\
                from pandaserver.taskbuffer.db_proxy_mods.base_module import BaseModule
                _other = PandaLogger().getLogger("Other")
                class OtherModule(BaseModule):
                    def __init__(self):
                        super().__init__(_other)
                """,
            "pandaserver/taskbuffer/OraDBProxy.py": """\
                from pandaserver.taskbuffer.db_proxy_mods.job_complex_module import JobComplexModule
                _logger = PandaLogger().getLogger("DBProxy")
                class DBProxy(JobComplexModule):
                    def __init__(self):
                        super().__init__(_logger)
                """,
            "pandajedi/jedicore/JediDBProxy.py": """\
                from pandaserver.taskbuffer import OraDBProxy
                logger = PandaLogger().getLogger(__name__.split(".")[-1])
                OraDBProxy._logger = logger
                class DBProxy(OraDBProxy.DBProxy):
                    pass
                """,
        },
    )

    assert found["pandaserver/taskbuffer/db_proxy_mods/job_complex_module.py::JobComplexModule.archiveJob"] == [
        (7, ["panda-DBProxy.log", "panda-JediDBProxy.log"], "logger")
    ]


def test_a_mixins_logger_comes_down_the_chain_of_a_proxy_with_several_bases(tmp_path):
    """``OraDBProxy.DBProxy`` derives from thirteen mixins, and its ``super().__init__``
    runs each one's constructor in turn along its MRO -- the call graph places it at
    the first mixin only.  The second mixin's logger still arrives from the proxy."""
    found = _landed(
        tmp_path,
        {
            "pandaserver/taskbuffer/db_proxy_mods/base_module.py": """\
                class BaseModule:
                    def __init__(self, log_stream):
                        self._log_stream = log_stream
                    def create_tagged_logger(self, comment):
                        tmp_log = LogWrapper(self._log_stream, comment)
                        return tmp_log
                """,
            "pandaserver/taskbuffer/db_proxy_mods/entity_module.py": """\
                from pandaserver.taskbuffer.db_proxy_mods.base_module import BaseModule
                class EntityModule(BaseModule):
                    def __init__(self, log_stream):
                        super().__init__(log_stream)
                """,
            "pandaserver/taskbuffer/db_proxy_mods/job_complex_module.py": """\
                from pandaserver.taskbuffer.db_proxy_mods.base_module import BaseModule
                class JobComplexModule(BaseModule):
                    def __init__(self, log_stream):
                        super().__init__(log_stream)
                    def archiveJob(self, job):
                        tmp_log = self.create_tagged_logger(" /* DBProxy.archiveJob */")
                        tmp_log.debug(f"start status={job.jobStatus}")
                """,
            "pandaserver/taskbuffer/OraDBProxy.py": """\
                from pandaserver.taskbuffer.db_proxy_mods.entity_module import EntityModule
                from pandaserver.taskbuffer.db_proxy_mods.job_complex_module import JobComplexModule
                _logger = PandaLogger().getLogger("DBProxy")
                class DBProxy(EntityModule, JobComplexModule):
                    def __init__(self):
                        super().__init__(_logger)
                """,
        },
    )

    assert found["pandaserver/taskbuffer/db_proxy_mods/job_complex_module.py::JobComplexModule.archiveJob"] == [
        (7, ["panda-DBProxy.log"], "logger")
    ]


def test_a_mixins_logger_does_not_spend_its_hops_on_the_mixins_before_it(tmp_path):
    """The call graph places each mixin's ``super().__init__`` at the next mixin of the proxy;
    read one link at a time, the fifth mixin's logger would be further than a logger is followed.
    The proxy's own call stands for the chain."""
    mixins = ["First", "Second", "Third", "Fourth", "Fifth"]
    files = {
        "pandaserver/taskbuffer/db_proxy_mods/base_module.py": """\
            class BaseModule:
                def __init__(self, log_stream):
                    self._log_stream = log_stream
                def create_tagged_logger(self, comment):
                    tmp_log = LogWrapper(self._log_stream, comment)
                    return tmp_log
            """,
        "pandaserver/taskbuffer/OraDBProxy.py": "".join(
            f"from pandaserver.taskbuffer.db_proxy_mods.{name.lower()}_module import {name}Module\n" for name in mixins
        )
        + '_logger = PandaLogger().getLogger("DBProxy")\n'
        + f"class DBProxy({', '.join(f'{name}Module' for name in mixins)}):\n"
        + "    def __init__(self):\n"
        + "        super().__init__(_logger)\n",
    }
    for name in mixins:
        files[f"pandaserver/taskbuffer/db_proxy_mods/{name.lower()}_module.py"] = (
            "from pandaserver.taskbuffer.db_proxy_mods.base_module import BaseModule\n"
            f"class {name}Module(BaseModule):\n"
            "    def __init__(self, log_stream):\n"
            "        super().__init__(log_stream)\n"
            "    def archiveJob(self, job):\n"
            '        tmp_log = self.create_tagged_logger(" /* DBProxy.archiveJob */")\n'
            '        tmp_log.debug(f"start status={job.jobStatus}")\n'
        )
    found = _landed(tmp_path, files)

    assert found["pandaserver/taskbuffer/db_proxy_mods/fifth_module.py::FifthModule.archiveJob"] == [
        (7, ["panda-DBProxy.log"], "logger")
    ]


def test_each_call_is_read_by_the_logger_it_is_made_on(tmp_path):
    """One module, two loggers, two files; ``setup_logger`` opens a file as
    ``getLogger`` does; the standard library's ``logging.getLogger`` opens none,
    a call whose logger leads nowhere keeps its module's files, said so; and one
    followed to the standard library's logger has no file, even in a module
    whose own function opens one."""
    found = _landed(
        tmp_path,
        {
            "pandaserver/api/v1/pilot_api.py": """\
                _logger = PandaLogger().getLogger("api_pilot")
                pilot_logger = PandaLogger().getLogger("PilotRequests")
                def acquire_jobs(req):
                    tmp_logger = LogWrapper(_logger, "acquire_jobs")
                    tmp_logger.debug("start")
                    pilot_logger.info("acquired")
                    unknown_log.info("from somewhere")
                """,
            "pandajedi/jedimsgprocessor/panda_to_jedi_msg_processor.py": """\
                base_logger = logger_utils.setup_logger(__name__.split(".")[-1])
                class PandaToJediMsgProcPlugin:
                    def process(self, msg):
                        tmp_log = logger_utils.make_logger(base_logger, token="x", method_name="process")
                        tmp_log.info("got")
                """,
            "pandaserver/daemons/master.py": """\
                from pandaserver.daemons.utils import DaemonMaster
                def get_logger():
                    my_logger = logging.getLogger("PanDA-Daemon-Master")
                    return my_logger
                def main():
                    main_log = get_logger()
                    main_log.info("start")
                    master = DaemonMaster(logger=main_log)
                """,
            "pandaserver/daemons/utils.py": """\
                def daemon_loop():
                    base_logger = logger_utils.setup_logger("daemons")
                    tmp_log = logger_utils.make_logger(base_logger, "worker")
                    tmp_log.info("daemon worker start")
                class DaemonMaster:
                    def __init__(self, logger):
                        self.logger = logger
                    def run(self):
                        self.logger.info("master run")
                """,
            "pandaserver/workflow/snakeparser/parser.py": """\
                class Parser:
                    def __init__(self, workflow_file, logger=None):
                        if logger:
                            self._logger = logger
                        else:
                            self._logger = logging.getLogger(__name__)
                        self._logger.debug("create workflow")
                """,
            "pandaserver/taskbuffer/workflow_processor.py": """\
                from pandaserver.workflow.snakeparser.parser import Parser
                _logger = PandaLogger().getLogger("workflow_processor")
                def process(workflow_file):
                    parser = Parser(workflow_file, logger=_logger)
                """,
        },
    )

    # Handed a logger, or opening the standard library's: the line is in the
    # file only when the caller handed one, so its absence there proves nothing.
    assert found["pandaserver/workflow/snakeparser/parser.py::Parser.__init__"] == [
        (7, ["panda-workflow_processor.log"], "partly")
    ]
    assert found["pandaserver/api/v1/pilot_api.py::acquire_jobs"] == [
        (5, ["panda-api_pilot.log"], "logger"),
        (6, ["panda-PilotRequests.log"], "logger"),
        (7, ["panda-api_pilot.log"], "module"),
    ]
    assert found["pandajedi/jedimsgprocessor/panda_to_jedi_msg_processor.py::PandaToJediMsgProcPlugin.process"] == [
        (5, ["panda-panda_to_jedi_msg_processor.log"], "logger")
    ]
    assert found["pandaserver/daemons/master.py::main"] == [(7, [], "logger")]
    # The module's file is ``daemon_loop``'s logger's, and the master's lines are not in it.
    assert found["pandaserver/daemons/utils.py::daemon_loop"] == [(4, ["panda-daemons.log"], "logger")]
    assert found["pandaserver/daemons/utils.py::DaemonMaster.run"] == [(9, [], "logger")]


def _prints(owner: str, *lines: int):
    from bamboo.codemap.models import FunctionPrintsNode, PrintRow, PrintStatement

    return FunctionPrintsNode(
        map_id=MAP_ID,
        derived_from=VERSION,
        name=FunctionPrintsNode.make_name(MAP_ID, owner),
        owner=owner,
        function=owner.rsplit(".", 1)[-1],
        count=len(lines),
        statements=[
            PrintStatement(line=line, level="debug", rows=[PrintRow(text="'done'", pattern="done")])
            for line in lines
        ],
    )


def test_a_call_the_prints_leave_out_fails_the_count():
    """A reader trusts an absence only when no call is missing, so the stored
    rows are held line for line against the calls read off the tree."""
    fragment = MapFragment(
        map_id=MAP_ID,
        derived_from=VERSION,
        function_prints=[_prints("x.py::A.run", 3, 5), _prints("x.py::B.run", 7)],
        print_calls={"x.py::A.run": [3, 9], "x.py::B.run": [7], "x.py::C.run": [2]},
    )

    result = gates.prints_are_counted(fragment)

    assert not result.passed
    assert result.failures == [
        "x.py::A.run: no rows for the call(s) on [9]; rows on [5] where it makes no call",
        "x.py::C.run: no rows for the call(s) on [2]",
    ]


def test_prints_that_hold_every_call_pass_the_count():
    fragment = MapFragment(
        map_id=MAP_ID,
        derived_from=VERSION,
        function_prints=[_prints("x.py::A.run", 3, 5)],
        print_calls={"x.py::A.run": [3, 5]},
    )

    assert gates.prints_are_counted(fragment).passed


def test_the_calls_are_counted_off_the_tree_under_the_owner_the_node_carries():
    """Each function its own calls: a nested ``def`` prints under its own
    name, and ``raise socket.error(...)`` is not a print at all."""
    from bamboo.codemap.panda.plugin import _print_calls

    source = (
        "class Bridge:\n"
        "    def recv(self, spec):\n"
        "        if spec.master:\n"
        "            raise socket.error('empty packet')\n"
        "        logger.error('the child closed the socket')\n"
        "        def later():\n"
        "            logger.info('later')\n"
        "        match spec.kind:\n"
        "            case 1:\n"
        "                logger.debug('one')\n"
    )

    calls = _print_calls([_module(source, "pandaserver/taskbuffer/ConBridge.py")])

    assert calls == {
        "pandaserver/taskbuffer/ConBridge.py::Bridge.recv": [5, 10],
        "pandaserver/taskbuffer/ConBridge.py::Bridge.recv.later": [7],
    }


# ---------------------------------------------------------------------------
# The uplink a knight uses: construct a worker, dispatch it, and the work runs
# in another class.  Not a call, so no name resolution reaches it.
# ---------------------------------------------------------------------------

_THREAD_BASE = """
class WorkerThread(threading.Thread):
    def __init__(self, workerSemaphore, threadPool, logger):
        self.threadPool = threadPool

    def run(self):
        self.runImpl()
"""

_KNIGHT_WITH_WORKER = """
logger = PandaLogger().getLogger(__name__.split(".")[-1])


class TaskCommando(JediKnight):
    def start(self):
        while True:
            tmpList = self.taskBufferIF.getTasksToExecCommand_JEDI(vo, prodSourceLabel)
            taskList = ListWithLock(tmpList)
            threadPool = ThreadPool()
            thr = TaskCommandoThread(taskList, threadPool, self.taskBufferIF, self.pid)
            thr.start()
            time.sleep(60)


class TaskCommandoThread(WorkerThread):
    def __init__(self, taskList, threadPool, taskbufferIF, pid):
        WorkerThread.__init__(self, None, threadPool, logger)
        self.taskList = taskList
        self.taskBufferIF = taskbufferIF
        self.pid = pid

    def runImpl(self):
        self.settle()

    def settle(self):
        taskSpec.status = "passed"
"""


def _worker_modules(knight: str = _KNIGHT_WITH_WORKER):
    return [
        _module(_THREAD_BASE, "pandajedi/jedicore/ThreadUtils.py"),
        _module(knight, "pandajedi/jediorder/TaskCommando.py"),
    ]


def _worker_junction(knight: str = _KNIGHT_WITH_WORKER) -> JunctionNode:
    """An arm inside the worker class, anchored where the write really is."""
    line = knight.splitlines().index('        taskSpec.status = "passed"') + 1
    junction = _junction("pandajedi/jediorder/TaskCommando.py::settle")
    junction.anchor = Anchor(
        package="pandajedi",
        file="pandajedi/jediorder/TaskCommando.py",
        line_start=line,
        line_end=line,
    )
    return junction


def test_a_dispatched_worker_makes_its_construction_site_the_entry():
    """``start`` builds the worker and hands it a list; ``runImpl`` does the work.

    No call connects them -- the edge is a constructor argument and a thread
    dispatch -- so the method-name resolution the rest of the slice turns on
    reaches nothing.  Thirteen of the map's entry points sit on this shape and
    every one of them had an empty ``via``.
    """
    junction = _worker_junction()
    trigger.attach([junction], _worker_modules(), set())

    worker = [e for e in junction.entry_points if e.via == "start"]
    assert worker, [(e.trigger, e.entry, e.via) for e in junction.entry_points]
    assert worker[0].entry == "pandajedi/jediorder/TaskCommando.py"
    assert worker[0].trigger == "polled"


def test_the_handover_is_keyed_by_the_name_the_worker_body_uses():
    """``self.taskBufferIF = taskbufferIF`` -- the two names differ.

    The body of an arm says ``self.taskList``, so that is what a trace asks
    about; keying by the parameter would make every reader re-derive the
    constructor's own assignment.
    """
    junction = _worker_junction()
    trigger.attach([junction], _worker_modules(), set())

    binding = next(e.arg_binding for e in junction.entry_points if e.via == "start")
    assert binding == {
        "taskList": "taskList",
        "taskBufferIF": "self.taskBufferIF",
        "pid": "self.pid",
    }
    # ``threadPool`` goes to the base class and the body never names it, so it
    # is left out rather than reported under the parameter's name.
    assert "threadPool" not in binding


_KNIGHT_BUILDING_TWICE = _KNIGHT_WITH_WORKER.replace(
    "            thr = TaskCommandoThread(taskList, threadPool, self.taskBufferIF, self.pid)\n"
    "            thr.start()\n",
    "            thr = TaskCommandoThread(taskList, threadPool, self.taskBufferIF, self.pid)\n"
    "            thr.start()\n"
    "            other = TaskCommandoThread(urgentList, threadPool, self.taskBufferIF, self.pid)\n"
    "            other.start()\n",
)


def test_an_entry_point_says_how_many_places_hand_over_that_way():
    """The fold is the door's, and it was silent.

    An entry is keyed on ``(trigger, entry, via, reached_by)`` because the set
    of reasons a task was not picked up differs *by entry* -- so two
    constructions in one function are one entry, deliberately.  What the row did
    not say is that it is keeping one of them: ``copyArchive.py::main`` builds a
    ``Watcher`` eight times with four different ``sleepTime`` values and the map
    held the last, which is the kill-running path rather than either holding
    one.  Measured over the corpus: 177 rows across 23 doors are samples.
    """
    junction = _worker_junction(_KNIGHT_BUILDING_TWICE)
    trigger.attach([junction], _worker_modules(_KNIGHT_BUILDING_TWICE), set())

    (worker,) = [e for e in junction.entry_points if e.via == "start"]

    assert worker.sites == 2
    # Still one row, and its bindings are one site's -- the later one read.
    assert worker.arg_binding["taskList"] == "urgentList"


def test_an_entry_point_built_once_says_so():
    """One site is the common case and has to read as a fact, not a sample."""
    junction = _worker_junction()
    trigger.attach([junction], _worker_modules(), set())

    (worker,) = [e for e in junction.entry_points if e.via == "start"]

    assert worker.sites == 1


def test_a_worker_that_is_never_dispatched_opens_nothing():
    """Two message processors keep a worker and call one of its methods.

    That edge is a call, which the name resolution already follows.  Claiming
    the dispatch as well would put ``runImpl`` behind an entry that never runs
    it.
    """
    stored = _KNIGHT_WITH_WORKER.replace("            thr.start()\n", "")
    junction = _worker_junction(stored)
    trigger.attach([junction], _worker_modules(stored), set())

    assert [e for e in junction.entry_points if e.via == "start"] == []


def test_two_worker_classes_in_one_module_are_told_apart():
    """``TaskBroker`` declares two, and the owner spells neither.

    Both write ``runImpl``, so ``module::runImpl`` names both; the anchor is
    what says which class the line is in.
    """
    source = """
logger = PandaLogger().getLogger(__name__.split(".")[-1])


class TaskBroker(JediKnight):
    def start(self):
        while True:
            thr = TaskCheckerThread(taskList)
            thr.start()
            time.sleep(60)


class TaskCheckerThread(WorkerThread):
    def __init__(self, taskList):
        self.taskList = taskList

    def runImpl(self):
        taskSpec.status = "checked"


class TaskBrokerThread(WorkerThread):
    def __init__(self, other):
        self.other = other

    def runImpl(self):
        taskSpec.status = "brokered"
"""
    modules = [
        _module(_THREAD_BASE, "pandajedi/jedicore/ThreadUtils.py"),
        _module(source, "pandajedi/jediorder/TaskBroker.py"),
    ]
    checker_line = source.splitlines().index('        taskSpec.status = "checked"') + 1
    broker_line = source.splitlines().index('        taskSpec.status = "brokered"') + 1

    checker = _junction("pandajedi/jediorder/TaskBroker.py::runImpl")
    checker.anchor = Anchor(
        package="pandajedi", file="x", line_start=checker_line, line_end=checker_line
    )
    broker = _junction("pandajedi/jediorder/TaskBroker.py::runImpl")
    broker.anchor = Anchor(
        package="pandajedi", file="x", line_start=broker_line, line_end=broker_line
    )
    trigger.attach([checker, broker], modules, set())

    assert [e.via for e in checker.entry_points if e.via] == ["start"]
    # Nothing constructs ``TaskBrokerThread`` here, so its arm gets no uplink
    # even though it shares a module and a method name with one that does.
    assert [e.via for e in broker.entry_points if e.via] == []


# ---------------------------------------------------------------------------
# Where a reader that is not a junction logs.
# ---------------------------------------------------------------------------


def test_a_reader_that_is_no_junction_still_gets_a_log_site():
    """Two thirds of the owners the map names as readers own no junction.

    ``_follow_up`` looks the reader up among the *writers* and finds nothing,
    so the answer to "which log will say the query ran" is silently empty for
    164 of 241 owners.
    """
    proxy = """
class DBProxy:
    def getTasksToExecCommand_JEDI(self, vo, label):
        sqlC = 'SELECT status FROM JEDI_Tasks'
        self.cur.execute(sqlC + comment, varMap)
"""
    knight = """
logger = PandaLogger().getLogger(__name__.split(".")[-1])


class TaskCommando(JediKnight):
    def start(self):
        while True:
            self.taskBufferIF.getTasksToExecCommand_JEDI(vo, label)
            time.sleep(60)
"""
    # Declares a logger, so the proxy mixin really does inherit a file.  Without
    # it ``log_files`` is empty and the assertion below passes without ever
    # exercising the rule it is there for.
    jedi_proxy = (
        'logger = PandaLogger().getLogger(__name__.split(".")[-1])\n'
        "class DBProxy(task_module.DBProxy):\n    pass\n"
    )
    modules = [
        _module(proxy, "pandaserver/taskbuffer/db_proxy_mods/task_module.py"),
        _module(knight, "pandajedi/jediorder/TaskCommando.py"),
        _module(jedi_proxy, "pandajedi/jedicore/JediDBProxy.py"),
    ]
    fragment = MapFragment(map_id=MAP_ID, derived_from=VERSION)
    fragment.subjects.append(
        SubjectNode(
            map_id=MAP_ID,
            derived_from=VERSION,
            name="JediTaskSpec.status",
            spec_class="JediTaskSpec",
            attribute="status",
            selected_by={
                "finishing": [
                    "pandaserver/taskbuffer/db_proxy_mods/task_module.py::"
                    "getTasksToExecCommand_JEDI"
                ]
            },
        )
    )
    logfile.attach(fragment, modules)

    site = {s.owner: s for s in fragment.log_sites}
    owner = (
        "pandaserver/taskbuffer/db_proxy_mods/task_module.py::getTasksToExecCommand_JEDI"
    )
    assert owner in site
    assert site[owner].log_files == ["panda-JediDBProxy.log"]
    assert site[owner].caller_log_files == ["panda-TaskCommando.log"]
    assert site[owner].owns_logger is False
    # Both, unlike a junction.  The query writes its own diagnostics into the
    # file it inherits -- measured, 51 of the 53 such owners do -- and the
    # knight that called it writes about the call into its own.
    assert site[owner].observable_log_files() == [
        "panda-JediDBProxy.log",
        "panda-TaskCommando.log",
    ]


def test_an_owner_that_is_already_a_junction_gets_no_second_row():
    """One fact in two places is how ``log_files`` came to mean two things."""
    source = """
logger = PandaLogger().getLogger(__name__.split(".")[-1])


class Feeder:
    def feed(self):
        taskSpec.status = "finishing"
"""
    modules = [_module(source, "pandajedi/jediorder/ContentsFeeder.py")]
    owner = "pandajedi/jediorder/ContentsFeeder.py::feed"
    fragment = MapFragment(map_id=MAP_ID, derived_from=VERSION)
    fragment.junctions.append(_junction(owner))
    fragment.subjects.append(
        SubjectNode(
            map_id=MAP_ID,
            derived_from=VERSION,
            name="JediTaskSpec.status",
            spec_class="JediTaskSpec",
            attribute="status",
            updated_by={"finishing": [owner]},
        )
    )
    logfile.attach(fragment, modules)

    assert [s.owner for s in fragment.log_sites] == []


def test_a_log_site_is_a_stored_kind():
    """``DiagnosticTemplate`` and ``EnumerationWrite`` are built and unreachable.

    An investigation reads a log site, so it needs a label and a diff entry --
    and this asked for the entry without ever asking whether anything reached
    it.  ``_index`` collected seven of the eight kinds the database stores, so
    every log site compared as absent from both builds and the entry answered
    a question nobody put to it.  Which log will show a reader's query running
    is the point of the node; a build that moved it said nothing.
    """
    from bamboo.codemap.models import MAP_NODE_GROUPS

    assert "log_sites" in MAP_NODE_GROUPS
    assert NodeType.LOG_SITE.value in diff.CONTENT_FIELDS

    def site(files):
        return LogSiteNode(
            map_id=MAP_ID,
            derived_from=VERSION,
            name="pandaserver/taskbuffer/db_proxy_mods/task_standalone_module.py::f",
            owner="pandaserver/taskbuffer/db_proxy_mods/task_standalone_module.py::f",
            log_files=files,
        )

    old = MapFragment(map_id=MAP_ID, derived_from=VERSION, log_sites=[site(["panda-server.log"])])
    new = MapFragment(map_id=MAP_ID, derived_from=VERSION, log_sites=[site(["panda-jedi.log"])])

    (change,) = diff.compare(old, new).changes

    assert change.field == "log_files"
    assert "panda-server.log" in change.before
    assert "panda-jedi.log" in change.after


def test_a_call_and_a_dispatch_are_not_compared_with_each_other():
    """``ContentsFeeder`` is reached both ways, and the keys are not the same kind.

    A call binds by keyword and a dispatch by the field the worker's body
    reads, so an empty set means "positional, not bound" on one side and
    "handed nothing" on the other.  Comparing across them took the report from
    13 rows to 69 and called a difference in spelling a guard that cannot fire.
    """
    junction = _junction("pandajedi/jediorder/ContentsFeeder.py::feed")
    junction.entry_points = [
        EntryPoint(
            trigger="message",
            entry="pandajedi/jedimsgprocessor/jedi_contents_feeder_msg_processor.py",
            via="feed",
        ),
        EntryPoint(
            trigger="polled",
            entry="pandajedi/jediorder/ContentsFeeder.py",
            via="start",
            arg_binding={"taskDsList": "dsList", "pid": "self.pid"},
            reached_by="dispatch",
        ),
    ]

    assert trigger.differing_arguments([junction]) == []


def test_two_dispatch_sites_handing_over_different_arguments_are_reported():
    """The finding the uplink was built for.

    ``panda_to_jedi_msg_processor`` constructs ``JobGeneratorThread`` without
    ``resource_types`` and ``JobGenerator`` constructs it with, so the
    candidate causes differ by which one ran -- which the branch table cannot
    show and an empty ``arg_binding`` could not either.
    """
    junction = _junction("pandajedi/jediorder/JobGenerator.py::doGenerate")
    junction.entry_points = [
        EntryPoint(
            trigger="polled",
            entry="pandajedi/jediorder/JobGenerator.py",
            via="start",
            arg_binding={"inputList": "inputList", "resource_types": "resource_types"},
            reached_by="dispatch",
        ),
        EntryPoint(
            trigger="message",
            entry="pandajedi/jedimsgprocessor/panda_to_jedi_msg_processor.py",
            via="process",
            arg_binding={"inputList": "inputList"},
            reached_by="dispatch",
        ),
    ]

    reported = trigger.differing_arguments([junction])

    assert len(reported) == 1
    _subject, _owner, supplied = reported[0]
    assert supplied["pandajedi/jediorder/JobGenerator.py"] == [
        "inputList",
        "resource_types",
    ]


# --------------------------------------------------------------------------- #
# the structures a path condition does not see
# --------------------------------------------------------------------------- #


def _write_in(source: str, name: str = "f"):
    """The first attribute assignment inside function *name*."""
    func = _func(source, name)
    return next(n for n in ast.walk(func) if isinstance(n, ast.Assign))


def test_a_try_body_says_which_handler_could_have_caught_something_first():
    node = _write_in(
        "def f(self, spec):\n"
        "    try:\n"
        "        spec.status = 'ok'\n"
        "    except ValueError:\n"
        "        pass\n"
    )

    seen = pathcond.enclosing_guards(node)

    assert [(u.kind, u.detail) for u in seen] == [
        (pathcond.UNSEEN_TRY, "nothing above raised ValueError")
    ]


def test_a_handler_says_that_something_above_it_raised():
    node = _write_in(
        "def f(self, spec):\n"
        "    try:\n"
        "        self.ddmIF.getDatasetMetaData(spec.name)\n"
        "    except Exception:\n"
        "        spec.status = 'tobroken'\n"
    )

    seen = pathcond.enclosing_guards(node)

    assert [(u.kind, u.detail) for u in seen] == [
        (pathcond.UNSEEN_EXCEPT, "something above raised Exception")
    ]
    # And the body it belongs to is not also claimed: the handler did not run
    # because the body completed, it ran because the body did not.
    assert pathcond.UNSEEN_TRY not in {u.kind for u in seen}


def test_a_try_body_does_not_list_the_statements_before_the_write():
    # The paired handler is bounded; the preceding statements are not.  On the
    # installed corpus a write in a try body has a median of twelve statements
    # above it and up to 135, which is noise rather than a reason.
    node = _write_in(
        "def f(self, spec):\n"
        "    try:\n"
        "        a = one()\n"
        "        b = two()\n"
        "        c = three()\n"
        "        spec.status = 'ok'\n"
        "    except ValueError:\n"
        "        pass\n"
    )

    details = " ".join(u.detail for u in pathcond.enclosing_guards(node))

    assert "one()" not in details and "two()" not in details


def test_a_loop_header_is_recorded_because_the_guard_does_not_name_it():
    node = _write_in(
        "def f(self, specs):\n    for spec in specs:\n        spec.status = 'ok'\n"
    )

    seen = pathcond.enclosing_guards(node)

    assert [(u.kind, u.detail) for u in seen] == [
        (pathcond.UNSEEN_LOOP, "for spec in specs")
    ]


def test_an_early_exit_above_the_write_in_the_same_block_implies_its_negation():
    # Sound without dataflow: statements in a block run in order, so reaching
    # the write means the ``continue`` above it did not fire.
    node = _write_in(
        "def f(self, specs):\n"
        "    for spec in specs:\n"
        "        if spec.is_pseudo():\n"
        "            continue\n"
        "        spec.status = 'ok'\n"
    )

    seen = pathcond.enclosing_guards(node)

    assert (pathcond.UNSEEN_EARLY_EXIT, "not (spec.is_pseudo())") in [
        (u.kind, u.detail) for u in seen
    ]


def test_an_early_exit_below_the_write_implies_nothing():
    node = _write_in(
        "def f(self, specs):\n"
        "    for spec in specs:\n"
        "        spec.status = 'ok'\n"
        "        if spec.is_pseudo():\n"
        "            continue\n"
    )

    assert pathcond.UNSEEN_EARLY_EXIT not in {
        u.kind for u in pathcond.enclosing_guards(node)
    }


def test_a_branch_that_falls_through_is_not_an_early_exit():
    node = _write_in(
        "def f(self, specs):\n"
        "    for spec in specs:\n"
        "        if spec.is_pseudo():\n"
        "            spec.type = 'pseudo'\n"
        "        spec.status = 'ok'\n",
    )

    # The first assignment in the walk is the one inside the ``if``; take the
    # guarded write instead.
    func = _func(
        "def f(self, specs):\n"
        "    for spec in specs:\n"
        "        if spec.is_pseudo():\n"
        "            spec.type = 'pseudo'\n"
        "        spec.status = 'ok'\n"
    )
    node = [n for n in ast.walk(func) if isinstance(n, ast.Assign)][-1]

    assert pathcond.UNSEEN_EARLY_EXIT not in {
        u.kind for u in pathcond.enclosing_guards(node)
    }


def test_nothing_above_the_enclosing_function_is_read():
    # Module-scope statements run at import, not on the way to this write, so
    # counting them as preceding would report a reason that never applied.
    node = _write_in(
        "if BROKEN:\n"
        "    raise SystemExit(1)\n"
        "def f(self, spec):\n"
        "    spec.status = 'ok'\n"
    )

    assert pathcond.enclosing_guards(node) == []


def test_a_write_with_no_if_is_not_reported_as_unconditional():
    # 18.1% of the map's arms have no ``if`` at all and sit inside a try, a
    # loop or a with; an empty path condition there means the walk saw no
    # ``if``, not that nothing had to hold.
    node = _write_in(
        "def f(self, specs):\n"
        "    with lock():\n"
        "        for spec in specs:\n"
        "            spec.status = 'ok'\n"
    )

    kinds = [u.kind for u in pathcond.enclosing_guards(node)]

    assert pathcond.path_condition(node) == []
    assert kinds == [pathcond.UNSEEN_WITH, pathcond.UNSEEN_LOOP]


_TIMED = '''\
class TimedMethod:
    def __init__(self, method, timeout):
        self.method = method
        self.timeout = timeout

    def __call__(self, *var):
        self.result = self.method(*var)

    def run(self, *var):
        thr = threading.Thread(target=self, args=var)
        thr.start()
'''

_LOGWRAPPER = '''\
class LogWrapper:
    def __init__(self, log, prefix):
        self.logger = log
        self.prefix = prefix

    def debug(self, message):
        self.logger.debug(self.prefix + message)
'''


def test_a_class_that_calls_what_it_was_handed_is_read_as_a_forwarder():
    """Read from the declaration rather than from a list of names.  Requiring
    the class to *call* the field it kept is what separates ``TimedMethod``
    from ``LogWrapper``, which keeps a logger and calls a method *on* it."""
    modules = [
        _module(_TIMED, "pandaserver/api/v1/common.py"),
        _module(_LOGWRAPPER, "pandaserver/srvcore/LogWrapper.py"),
    ]

    assert trigger.forwarder_classes(modules) == {"TimedMethod": ("method", 0)}


def test_a_method_passed_as_a_value_reaches_the_module_that_passed_it():
    """``TimedMethod(gtb.updateJobStatus, timeout).run(...)`` runs
    ``updateJobStatus``, but the method arrives as a value and no name-based
    reading sees a call.  ``updateJobStatus`` reported that no log file named
    it, which is the junction the pilot's ``holding`` goes through."""
    modules = [
        _module(_TIMED, "pandaserver/api/v1/common.py"),
        _module(
            "def update_job(req, job_id, status):\n"
            "    while True:\n"
            "        time.sleep(1)\n"
            "    timed_method = TimedMethod(global_task_buffer.updateJobStatus, 10)\n"
            "    timed_method.run(job_id, status)\n",
            "pandaserver/api/v1/pilot_api.py",
        ),
        _module(
            "class DBProxy:\n"
            "    def updateJobStatus(self, job_id, status):\n"
            "        pass\n",
            "pandaserver/taskbuffer/db_proxy_mods/job_complex_module.py",
        ),
    ]

    reaching = trigger.reaching_modules(modules)
    entries = reaching["pandaserver/taskbuffer/db_proxy_mods/job_complex_module.py"]

    assert [entry for entry, _door, _call in entries["updateJobStatus"]] == [
        "pandaserver/api/v1/pilot_api.py"
    ]


def test_the_forwarded_edge_binds_the_arguments_of_the_later_call():
    """``TimedMethod(...)`` carries the callee and ``.run(...)`` carries its
    arguments.  Reading them off the constructor would report the timeout as
    what was passed to the method."""
    modules = [
        _module(_TIMED, "pandaserver/api/v1/common.py"),
        _module(
            "def update_job(req):\n"
            "    while True:\n"
            "        time.sleep(1)\n"
            "    timed_method = TimedMethod(global_task_buffer.updateJobStatus, 10)\n"
            "    timed_method.run(job_id=7, status='holding')\n",
            "pandaserver/api/v1/pilot_api.py",
        ),
        _module(
            "class DBProxy:\n"
            "    def updateJobStatus(self, job_id, status):\n"
            "        pass\n",
            "pandaserver/taskbuffer/db_proxy_mods/job_complex_module.py",
        ),
    ]

    reaching = trigger.reaching_modules(modules)
    _entry, _door, call = reaching[
        "pandaserver/taskbuffer/db_proxy_mods/job_complex_module.py"
    ]["updateJobStatus"][0]

    assert trigger._arg_binding(call) == {"job_id": "7", "status": "'holding'"}


def test_a_bare_attribute_passed_to_anything_else_is_not_a_forwarded_call():
    """The weaker rule -- an attribute passed as an argument -- matches 432
    sites in the corpus, and the receivers are ``len``, ``str``, ``int``,
    ``hasattr`` and ``getattr``.  Requiring a declared forwarder is what keeps
    those out without a list of exceptions."""
    modules = [
        _module(_LOGWRAPPER, "pandaserver/srvcore/LogWrapper.py"),
        _module(
            "def update_job(req):\n"
            "    while True:\n"
            "        time.sleep(1)\n"
            "    n = len(global_task_buffer.updateJobStatus)\n",
            "pandaserver/api/v1/pilot_api.py",
        ),
        _module(
            "class DBProxy:\n"
            "    def updateJobStatus(self):\n"
            "        pass\n",
            "pandaserver/taskbuffer/db_proxy_mods/job_complex_module.py",
        ),
    ]

    reaching = trigger.reaching_modules(modules)

    assert "updateJobStatus" not in reaching.get(
        "pandaserver/taskbuffer/db_proxy_mods/job_complex_module.py", {}
    )


_ADDER = '''\
class AdderGen:
    def __init__(self, taskBuffer, job_id, job_status, attempt_nr):
        self.job_status = job_status
        self.job_id = job_id

    def process_job_report(self):
        spec.status = self.job_status
'''


def test_a_class_that_decides_something_is_indexed_by_its_constructions():
    """``AdderGen`` is not a ``WorkerThread`` subclass and is not dispatched,
    so the worker uplink walks past it -- and it is where a job's status is
    settled, from a value chosen two modules away."""
    modules = [
        _module(_ADDER, "pandaserver/dataservice/adder_gen.py"),
        _module(
            "def update_job(req):\n"
            "    adder_gen = AdderGen(global_task_buffer, job_id, job_status, attempt_nr)\n"
            "    adder_gen.dump_file_report(report)\n",
            "pandaserver/api/v1/pilot_api.py",
        ),
    ]
    junctions = [_junction("pandaserver/dataservice/adder_gen.py::process_job_report")]

    reach = trigger.construction_uplinks(modules, junctions, set())

    entries = reach[("pandaserver/dataservice/adder_gen.py", "process_job_report")]
    assert [(entry, via) for entry, via, _h, _s in entries] == [
        ("pandaserver/api/v1/pilot_api.py", "update_job")
    ]
    assert entries[0][2]["job_status"] == "job_status"


def test_several_constructions_are_several_entries():
    """Three sites build ``AdderGen`` and the pilot's and the daemon's differ
    in what they pass.  Folding them would answer "who chose this" with one of
    the callers and no sign that there were others."""
    modules = [
        _module(_ADDER, "pandaserver/dataservice/adder_gen.py"),
        _module(
            "def update_job(req):\n"
            "    adder_gen = AdderGen(global_task_buffer, job_id, job_status, attempt_nr)\n",
            "pandaserver/api/v1/pilot_api.py",
        ),
        _module(
            "def main():\n"
            "    thr = AdderGen(taskBuffer, panda_id, job_status, attempt_nr)\n",
            "pandaserver/daemons/scripts/add_main.py",
        ),
    ]
    junctions = [_junction("pandaserver/dataservice/adder_gen.py::process_job_report")]

    reach = trigger.construction_uplinks(modules, junctions, set())

    assert sorted(
        entry for entry, _via, _h, _s in
        reach[("pandaserver/dataservice/adder_gen.py", "process_job_report")]
    ) == [
        "pandaserver/api/v1/pilot_api.py",
        "pandaserver/daemons/scripts/add_main.py",
    ]


def test_a_class_that_decides_nothing_is_not_indexed():
    """The restriction that replaces the two this took off.  Pairing every
    construction with every call gives 1627 pairs on the installed corpus,
    which is the generic reference edge the map has been punished for twice."""
    modules = [
        _module(
            "class Helper:\n"
            "    def __init__(self, value):\n"
            "        self.value = value\n"
            "\n"
            "    def use(self):\n"
            "        return self.value\n",
            "pandaserver/srvcore/Helper.py",
        ),
        _module(
            "def caller():\n"
            "    h = Helper(7)\n",
            "pandaserver/api/v1/pilot_api.py",
        ),
    ]

    assert trigger.construction_uplinks(modules, [], set()) == {}


def test_a_declared_spec_is_not_a_handover_of_control():
    """``classify`` and the attribution answer for a spec's fields already,
    and ``JobSpec.pack`` handing a row around is data movement."""
    modules = [
        _module(
            "class JediDatasetSpec:\n"
            "    def __init__(self, status):\n"
            "        self.status = status\n"
            "\n"
            "    def setStatus(self):\n"
            "        spec.status = self.status\n",
            "pandajedi/jedicore/JediDatasetSpec.py",
        ),
        _module(
            "def build():\n"
            "    d = JediDatasetSpec('ready')\n",
            "pandajedi/jediorder/TaskBroker.py",
        ),
    ]
    junctions = [_junction("pandajedi/jedicore/JediDatasetSpec.py::setStatus")]

    assert trigger.construction_uplinks(modules, junctions, {"JediDatasetSpec"}) == {}


def test_a_class_two_modules_declare_is_dropped_rather_than_guessed():
    """A construction site names the class and nothing else, so an ambiguous
    name would resolve the crossing by guess -- the restriction
    ``worker_classes`` already puts on a dispatched worker."""
    modules = [
        _module(_ADDER, "pandaserver/dataservice/adder_gen.py"),
        _module(_ADDER, "pandaserver/dataservice/adder_other.py"),
        _module(
            "def update_job(req):\n"
            "    adder_gen = AdderGen(global_task_buffer, job_id, job_status, attempt_nr)\n",
            "pandaserver/api/v1/pilot_api.py",
        ),
    ]
    junctions = [
        _junction("pandaserver/dataservice/adder_gen.py::process_job_report"),
        _junction("pandaserver/dataservice/adder_other.py::process_job_report"),
    ]

    assert trigger.construction_uplinks(modules, junctions, set()) == {}


_PLUGINS = '''\
class AdderPluginBase:
    def execute(self):
        pass

class AdderAtlasPlugin(AdderPluginBase):
    pass

class AdderDummyPlugin(AdderPluginBase):
    pass
'''


def test_a_run_time_class_choice_comes_back_as_the_set_it_could_be():
    """The source cannot say which class a config names, but it declares the
    shape of the answer: a fallback to a concrete class, whose base names the
    interface and whose siblings are the alternatives."""
    modules = [
        _module(_PLUGINS, "pandaserver/dataservice/adder_plugins.py"),
        _module(
            "class AdderGen:\n"
            "    def get_plugin_class(self, vo, group):\n"
            "        self.plugin_class = panda_config.getPlugin('adder_plugins', vo, group)\n"
            "        if self.plugin_class is None:\n"
            "            self.plugin_class = AdderAtlasPlugin\n"
            "        self.logger.debug(f'plugin name {self.plugin_class.__name__}')\n",
            "pandaserver/dataservice/adder_gen.py",
        ),
    ]

    found = trigger.dispatch_fanouts(modules)

    fanout = found["pandaserver/dataservice/adder_gen.py::AdderGen"][0]
    assert fanout.default == "AdderAtlasPlugin"
    assert fanout.base == "AdderPluginBase"
    assert fanout.candidates == ["AdderAtlasPlugin", "AdderDummyPlugin"]
    assert fanout.announced_by == "plugin name"


def test_an_ordinary_default_is_not_a_dispatch():
    """The structural shape on its own -- assign a call, then ``if x is None:
    x = y`` -- matches 29 sites in the corpus and most are plain defaults:
    ``maxHS06sec``, ``coreCount``, ``newScanList``.  Requiring the fallback to
    be a class the corpus declares leaves the three that are dispatches."""
    modules = [
        _module(
            "class Proxy:\n"
            "    def setHS06sec(self, start, end):\n"
            "        hs06sec = JobUtils.getHS06sec(start, end)\n"
            "        if hs06sec is None:\n"
            "            hs06sec = maxHS06sec\n",
            "pandaserver/taskbuffer/db_proxy_mods/entity_module.py",
        ),
    ]

    assert trigger.dispatch_fanouts(modules) == {}


def test_a_fallback_guarded_by_a_conjunction_still_counts():
    """``closer.py`` asks ``is None and self.job.VO == 'atlas'``.  Requiring a
    bare comparison missed it, and the extra term is a real path condition
    rather than a reason not to read the fallback."""
    modules = [
        _module(_PLUGINS, "pandaserver/dataservice/adder_plugins.py"),
        _module(
            "class Closer:\n"
            "    def perform_vo_actions(self):\n"
            "        plugin_class = panda_config.getPlugin('closer_plugins', self.job.VO)\n"
            "        if plugin_class is None and self.job.VO == 'atlas':\n"
            "            plugin_class = AdderAtlasPlugin\n",
            "pandaserver/dataservice/closer.py",
        ),
    ]

    found = trigger.dispatch_fanouts(modules)

    assert found["pandaserver/dataservice/closer.py::Closer"][0].default == "AdderAtlasPlugin"


def test_a_fan_out_nothing_prints_says_so_rather_than_implying_proof():
    """A candidate set nobody can check is three guesses wearing a bracket."""
    modules = [
        _module(_PLUGINS, "pandaserver/dataservice/adder_plugins.py"),
        _module(
            "class Setupper:\n"
            "    def run(self):\n"
            "        plugin_class = panda_config.getPlugin('setupper_plugins', vo)\n"
            "        if plugin_class is None:\n"
            "            plugin_class = AdderAtlasPlugin\n",
            "pandaserver/dataservice/setupper.py",
        ),
    ]

    found = trigger.dispatch_fanouts(modules)

    assert found["pandaserver/dataservice/setupper.py::Setupper"][0].announced_by == ""


def test_the_fan_out_reaches_the_arms_of_its_own_class_only():
    """``closer.py`` declares more than one class, and a key of
    ``module::class`` alone would hand each the other's arms."""
    modules = [
        _module(_PLUGINS, "pandaserver/dataservice/adder_plugins.py"),
        _module(
            "class Closer:\n"
            "    def perform_vo_actions(self):\n"
            "        plugin_class = panda_config.getPlugin('closer_plugins', vo)\n"
            "        if plugin_class is None:\n"
            "            plugin_class = AdderAtlasPlugin\n"
            "\n"
            "class CloserThread:\n"
            "    def run(self):\n"
            "        spec.status = 'closed'\n",
            "pandaserver/dataservice/closer.py",
        ),
    ]
    inside = _junction("pandaserver/dataservice/closer.py::perform_vo_actions")
    inside.anchor = Anchor(package="pandaserver", file="pandaserver/dataservice/closer.py", line_start=3, line_end=3)
    outside = _junction("pandaserver/dataservice/closer.py::run")
    outside.anchor = Anchor(package="pandaserver", file="pandaserver/dataservice/closer.py", line_start=9, line_end=9)

    trigger.attach_dispatch([inside, outside], modules)

    assert [d.default for d in inside.dispatch] == ["AdderAtlasPlugin"]
    assert outside.dispatch == []


def test_the_daemon_layer_speaks_through_the_task_buffer():
    """``executions`` read ``.execute`` and ``.executemany`` only, and the
    daemon scripts run 114 statements through ``querySQL``/``querySQLS`` -- 84
    of them under ``daemons/scripts``.  Without this the map could not see the
    three queries that pick a job out of ``holding``, and the verdict pointed
    at "ask whether the command arrived"."""
    source = (
        "def f():\n"
        "    sql = 'SELECT PandaID FROM ATLAS_PANDA.jobsActive4 WHERE jobStatus=:jobStatus'\n"
        "    var_map = {}\n"
        "    var_map[':jobStatus'] = 'holding'\n"
        "    status, res = taskBuffer.querySQLS(sql, var_map)\n"
    )
    func = _func(source)

    runs = sql.executions(func)

    assert [r.varmap for r in runs] == ["var_map"]
    assert "jobsActive4" in runs[0].sql


def test_the_binds_of_a_forwarded_run_are_the_ones_of_its_own_block():
    """The reason a window is needed at all.  ``copyArchive.main`` binds
    ``sql`` 23 times and ``:jobStatus`` to 11 values; pairing a statement with
    every bind in the function would have each statement claim all 253.  The
    boundaries are not guessed -- the same function writes ``var_map = {}``
    31 times."""
    source = (
        "def f():\n"
        "    sql = 'SELECT PandaID FROM ATLAS_PANDA.jobsActive4 WHERE jobStatus=:jobStatus'\n"
        "    var_map = {}\n"
        "    var_map[':jobStatus'] = 'activated'\n"
        "    status, res = taskBuffer.querySQLS(sql, var_map)\n"
        "    var_map = {}\n"
        "    var_map[':jobStatus'] = 'holding'\n"
        "    status, res = taskBuffer.querySQLS(sql, var_map)\n"
    )
    func = _func(source)

    runs = sorted(sql.executions(func), key=lambda r: r.call.lineno)
    values = [
        [ast.literal_eval(b.value) for b in sql.bound_values(func, r.varmap, ":jobStatus", r.window)]
        for r in runs
    ]

    assert values == [["activated"], ["holding"]]


def test_without_the_window_a_forwarded_run_claims_every_bind():
    """The failure this is shaped to prevent, fixed so it cannot come back
    silently: the unwindowed reading is still there, and it is still wrong for
    this form."""
    source = (
        "def f():\n"
        "    sql = 'SELECT PandaID FROM ATLAS_PANDA.jobsActive4 WHERE jobStatus=:jobStatus'\n"
        "    var_map = {}\n"
        "    var_map[':jobStatus'] = 'activated'\n"
        "    status, res = taskBuffer.querySQLS(sql, var_map)\n"
        "    var_map = {}\n"
        "    var_map[':jobStatus'] = 'holding'\n"
        "    status, res = taskBuffer.querySQLS(sql, var_map)\n"
    )
    func = _func(source)

    unwindowed = [
        ast.literal_eval(b.value)
        for b in sql.bound_values(func, "var_map", ":jobStatus")
    ]

    assert sorted(unwindowed) == ["activated", "holding"]


def _bound_at(func, key: str) -> list[list[str]]:
    """The values each run of *func*, in line order, reads for *key*."""
    return [
        sorted(ast.unparse(b.value) for b in sql.bound_values(func, run.varmap, key, run.window, at=run.call))
        for run in sorted(sql.executions(func), key=lambda r: r.call.lineno)
    ]


def test_an_execute_run_reads_only_the_binds_its_map_holds():
    """``retryTask_JEDI`` binds ``:status`` for statements on three tables, and
    every one of them read all its values: the dataset's ``ready`` and
    ``done`` became the task's.  The window stays ``None`` -- it says where a
    walk starts, which is not this question."""
    source = (
        "def f(self):\n"
        "    sqlD = 'UPDATE ATLAS_PANDA.JEDI_Datasets SET status=:status WHERE datasetID=:datasetID'\n"
        "    sqlT = 'UPDATE ATLAS_PANDA.JEDI_Tasks SET status=:status WHERE jediTaskID=:jediTaskID'\n"
        "    varMap = {}\n"
        "    varMap[':status'] = 'ready'\n"
        "    self.cur.execute(sqlD + comment, varMap)\n"
        "    varMap = {}\n"
        "    varMap[':status'] = 'running'\n"
        "    self.cur.execute(sqlT + comment, varMap)\n"
    )
    func = _func(source)

    assert _bound_at(func, ":status") == [["'ready'"], ["'running'"]]
    assert sorted(ast.unparse(b.value) for b in sql.bound_values(func, "varMap", ":status")) == [
        "'ready'",
        "'running'",
    ]
    assert [r.window for r in sql.executions(func)] == [None, None]


def _bound_by_text(func, key: str) -> list[tuple[int, str, list[str]]]:
    """For each run of *func* at each of its calls: the call's line, the text's ``SET``, and the values read for *key*."""
    return sorted(
        (
            call.lineno,
            run.sql.partition(" SET ")[2].partition(" WHERE ")[0],
            sorted(
                ast.unparse(b.value)
                for b in sql.bound_values(func, run.varmap, key, run.window, at=run._replace(calls=(call,)))
            ),
        )
        for run in sql.executions(func)
        for call in run.run_at
    )


def test_a_text_reads_only_the_binds_of_the_passes_that_build_it():
    """``updateTaskStatusByContFeeder_JEDI`` adds ``splitRule`` to its ``UPDATE``
    only when it is given a task, fills ``:status`` from the task in that case
    and ``'assigning'`` or ``'ready'`` otherwise, and runs whichever text it
    built at one call.  Read by the call alone, both texts wrote all three.
    The text with the clause is built only in the arm adding it; the text
    without it only where that arm was skipped, so not under the second ``if``
    testing the same thing."""
    source = (
        "def f(self, taskSpec, cloud):\n"
        "    sql = 'UPDATE ATLAS_PANDA.JEDI_Tasks SET status=:status'\n"
        "    if taskSpec is not None:\n"
        "        sql += ',splitRule=:splitRule'\n"
        "    sql += ' WHERE jediTaskID=:jediTaskID'\n"
        "    varMap = {}\n"
        "    varMap[':jediTaskID'] = 1\n"
        "    if taskSpec is not None:\n"
        "        varMap[':status'] = taskSpec.status\n"
        "        varMap[':splitRule'] = taskSpec.splitRule\n"
        "    elif cloud is None:\n"
        "        varMap[':status'] = 'assigning'\n"
        "    else:\n"
        "        varMap[':status'] = 'ready'\n"
        "    self.cur.execute(sql, varMap)\n"
    )
    func = _func(source)

    assert _bound_by_text(func, ":status") == [
        (15, "status=:status", ["'assigning'", "'ready'"]),
        (15, "status=:status,splitRule=:splitRule", ["taskSpec.status"]),
    ]
    at_the_call = sql.bound_values(func, "varMap", ":status", at=func.body[-1].value)
    assert sorted(ast.unparse(b.value) for b in at_the_call) == ["'assigning'", "'ready'", "taskSpec.status"]


def test_a_map_made_afresh_each_pass_carries_no_other_arm_to_the_text():
    """``insertNewJob`` makes ``varMap`` afresh for each file, sets
    ``:status`` in the two arms of one test and ``is_waiting`` in the two arms
    of another testing the same: the other arm's status, of this file or the
    one before, is in no map the text runs with."""
    source = (
        "def f(self, files):\n"
        "    for file in files:\n"
        "        varMap = {}\n"
        "        varMap[':fileID'] = file.fileID\n"
        "        if file.waiting:\n"
        "            varMap[':status'] = 'picked'\n"
        "        else:\n"
        "            varMap[':status'] = 'running'\n"
        "        sql = 'UPDATE ATLAS_PANDA.JEDI_Dataset_Contents SET status=:status'\n"
        "        if file.waiting:\n"
        "            sql += ',is_waiting=:is_waiting'\n"
        "            varMap[':is_waiting'] = 'Y'\n"
        "        else:\n"
        "            sql += ',is_waiting=NULL'\n"
        "        sql += ' WHERE fileID=:fileID'\n"
        "        self.cur.execute(sql, varMap)\n"
    )
    func = _func(source)

    assert _bound_by_text(func, ":status") == [
        (16, "status=:status,is_waiting=:is_waiting", ["'picked'"]),
        (16, "status=:status,is_waiting=NULL", ["'running'"]),
    ]


def test_a_map_kept_across_passes_carries_an_earlier_pass_bind_to_the_text():
    """A map made before the loop keeps what an earlier pass bound: the text
    the ``else`` builds names ``:a``, which only the other arm fills, and on a
    pass after one through that arm the map still holds it."""
    source = (
        "def f(self, items):\n"
        "    varMap = {}\n"
        "    varMap[':id'] = 1\n"
        "    for item in items:\n"
        "        if item.first:\n"
        "            sql = 'UPDATE ATLAS_PANDA.t SET a=:a WHERE id=:id'\n"
        "            varMap[':a'] = item.a\n"
        "        else:\n"
        "            sql = 'UPDATE ATLAS_PANDA.t SET a=:a,b=1 WHERE id=:id'\n"
        "        self.cur.execute(sql, varMap)\n"
    )
    func = _func(source)

    assert _bound_by_text(func, ":a") == [(10, "a=:a", ["item.a"]), (10, "a=:a,b=1", ["item.a"])]


def test_a_bind_every_pass_building_the_text_writes_again_is_not_the_texts():
    """``updateTaskStatusByContFeeder_JEDI`` sets ``:updateTime`` to now and
    ``:errorDialog`` to None before its ``if``; the two arms building the text
    without the clause set the time six hours back, and the arm building the
    one with it sets the task's diagnostic.  Now reaches the call only on the
    pass building the clause, None only on the passes building none."""
    source = (
        "def f(self, taskSpec, cloud):\n"
        "    sql = 'UPDATE ATLAS_PANDA.JEDI_Tasks SET status=:status,modificationTime=:updateTime,errorDialog=:errorDialog'\n"
        "    if taskSpec is not None:\n"
        "        sql += ',splitRule=:splitRule'\n"
        "    sql += ' WHERE jediTaskID=:jediTaskID'\n"
        "    varMap = {}\n"
        "    varMap[':jediTaskID'] = 1\n"
        "    varMap[':updateTime'] = timeNow\n"
        "    varMap[':errorDialog'] = None\n"
        "    if taskSpec is not None:\n"
        "        varMap[':status'] = taskSpec.status\n"
        "        varMap[':errorDialog'] = taskSpec.errorDialog\n"
        "        varMap[':splitRule'] = taskSpec.splitRule\n"
        "    elif cloud is None:\n"
        "        varMap[':status'] = 'assigning'\n"
        "        varMap[':updateTime'] = sixHoursAgo\n"
        "    else:\n"
        "        varMap[':status'] = 'ready'\n"
        "        varMap[':updateTime'] = sixHoursAgo\n"
        "    self.cur.execute(sql, varMap)\n"
    )
    func = _func(source)
    plain = "status=:status,modificationTime=:updateTime,errorDialog=:errorDialog"

    assert _bound_by_text(func, ":updateTime") == [
        (20, plain, ["sixHoursAgo", "sixHoursAgo"]),
        (20, plain + ",splitRule=:splitRule", ["timeNow"]),
    ]
    assert _bound_by_text(func, ":errorDialog") == [
        (20, plain, ["None"]),
        (20, plain + ",splitRule=:splitRule", ["taskSpec.errorDialog"]),
    ]
    at_the_call = sql.bound_values(func, "varMap", ":updateTime", at=func.body[-1].value)
    assert sorted(ast.unparse(b.value) for b in at_the_call) == ["sixHoursAgo", "sixHoursAgo", "timeNow"]


def test_a_pass_not_building_the_text_does_not_cut_what_it_carries_to_a_later_pass():
    """A pass through the arm filling ``:a`` runs on past the fill and builds
    the other text; the next pass builds this one with the map still holding
    ``:a``.  The arm's statements are no pass of this text's, but the value
    goes round the loop."""
    source = (
        "def f(self, items):\n"
        "    varMap = {}\n"
        "    varMap[':id'] = 1\n"
        "    for item in items:\n"
        "        if item.first:\n"
        "            sql = 'UPDATE ATLAS_PANDA.t SET a=:a WHERE id=:id'\n"
        "            varMap[':a'] = item.a\n"
        "            self.log(item)\n"
        "        else:\n"
        "            sql = 'UPDATE ATLAS_PANDA.t SET a=:a,b=1 WHERE id=:id'\n"
        "        self.cur.execute(sql, varMap)\n"
    )
    func = _func(source)

    assert _bound_by_text(func, ":a") == [(11, "a=:a", ["item.a"]), (11, "a=:a,b=1", ["item.a"])]


def test_a_bind_from_before_the_loop_every_pass_building_the_text_writes_again_is_not_the_texts():
    """``:a`` is 0 before the loop; every pass building the first text sets it
    again, and a pass building the second does not, so the second runs with 0
    or with what an earlier pass set."""
    source = (
        "def f(self, items):\n"
        "    varMap = {}\n"
        "    varMap[':id'] = 1\n"
        "    varMap[':a'] = 0\n"
        "    for item in items:\n"
        "        if item.first:\n"
        "            sql = 'UPDATE ATLAS_PANDA.t SET a=:a WHERE id=:id'\n"
        "            varMap[':a'] = item.a\n"
        "        else:\n"
        "            sql = 'UPDATE ATLAS_PANDA.t SET a=:a,b=1 WHERE id=:id'\n"
        "        self.cur.execute(sql, varMap)\n"
    )
    func = _func(source)

    assert _bound_by_text(func, ":a") == [(11, "a=:a", ["item.a"]), (11, "a=:a,b=1", ["0", "item.a"])]


def test_one_text_run_from_two_arms_keeps_each_call_its_own_binds():
    """``activateJob`` writes the same ``sqlJob`` in each arm and runs it there:
    one run holds both calls, and its pieces name one of the two assignments,
    so a piece is read only where it can be on a pass through the call.  And
    ``archiveJob`` writes ``sqlFMod`` before its loop and the same text again
    in one arm: a text written twice adds no clause."""
    source = (
        "def f(self, job, ok, jobs):\n"
        "    if ok:\n"
        "        sqlJob = 'UPDATE ATLAS_PANDA.jobParamsTable SET jobParameters=:param WHERE PandaID=:PandaID'\n"
        "        varMap = {}\n"
        "        varMap[':PandaID'] = job.PandaID\n"
        "        varMap[':param'] = job.jobParameters\n"
        "        self.cur.execute(sqlJob, varMap)\n"
        "    else:\n"
        "        sqlJob = 'UPDATE ATLAS_PANDA.jobParamsTable SET jobParameters=:param WHERE PandaID=:PandaID'\n"
        "        varMap = {}\n"
        "        varMap[':PandaID'] = job.PandaID\n"
        "        varMap[':param'] = job.newParameters\n"
        "        self.cur.execute(sqlJob, varMap)\n"
        "    sqlF = 'UPDATE ATLAS_PANDA.filesTable4 SET modificationTime=:modificationTime WHERE PandaID=:PandaID'\n"
        "    for one in jobs:\n"
        "        if one.failed:\n"
        "            sqlF = 'UPDATE ATLAS_PANDA.filesTable4 SET modificationTime=:modificationTime WHERE PandaID=:PandaID'\n"
        "            varMap = {}\n"
        "            varMap[':PandaID'] = one.PandaID\n"
        "            varMap[':modificationTime'] = one.modificationTime\n"
        "            self.cur.execute(sqlF, varMap)\n"
    )
    func = _func(source)

    # Each assignment of the text is a run of its own, holding both calls.
    assert {(line, tuple(values)) for line, _set, values in _bound_by_text(func, ":param") if line != 21} == {
        (7, ("job.jobParameters",)),
        (13, ("job.newParameters",)),
    }
    in_the_loop = {tuple(values) for line, _set, values in _bound_by_text(func, ":PandaID") if line == 21}
    assert in_the_loop == {("one.PandaID",)}


def test_one_text_built_in_two_arms_for_one_call_keeps_both_arms_binds():
    """The same text written in each arm and run once after the ``if`` is one
    run, its pieces naming one arm's assignment: a text is built by every way
    that renders it, and a statement only some ways run decides nothing."""
    source = (
        "def f(self, ok, job):\n"
        "    if ok:\n"
        "        sql = 'UPDATE ATLAS_PANDA.jobParamsTable SET jobParameters=:param WHERE PandaID=:PandaID'\n"
        "        varMap = {}\n"
        "        varMap[':PandaID'] = job.PandaID\n"
        "        varMap[':param'] = job.jobParameters\n"
        "    else:\n"
        "        sql = 'UPDATE ATLAS_PANDA.jobParamsTable SET jobParameters=:param WHERE PandaID=:PandaID'\n"
        "        varMap = {}\n"
        "        varMap[':PandaID'] = job.PandaID\n"
        "        varMap[':param'] = job.newParameters\n"
        "    self.cur.execute(sql, varMap)\n"
    )
    func = _func(source)

    assert _bound_by_text(func, ":param") == [(12, "jobParameters=:param", ["job.jobParameters", "job.newParameters"])]


def test_a_map_rebuilt_in_each_arm_keeps_both_values():
    """The window cannot say this: it opens at the nearest ``varMap = {}``,
    which is the second arm's, and loses the first."""
    source = (
        "def f(self, broken):\n"
        "    sqlT = 'UPDATE ATLAS_PANDA.JEDI_Tasks SET status=:status WHERE jediTaskID=:jediTaskID'\n"
        "    if broken:\n"
        "        varMap = {}\n"
        "        varMap[':status'] = 'broken'\n"
        "    else:\n"
        "        varMap = {}\n"
        "        varMap[':status'] = 'ready'\n"
        "    self.cur.execute(sqlT + comment, varMap)\n"
    )
    func = _func(source)

    assert _bound_at(func, ":status") == [["'broken'", "'ready'"]]
    assert [
        ast.unparse(b.value) for b in sql.bound_values(func, "varMap", ":status", sql.binding_window(func, "varMap", 9))
    ] == ["'ready'"]


def test_a_run_over_a_list_of_maps_reads_the_binds_appended_to_it():
    """``prepareTasksToBeFinished_JEDI`` fills the input datasets' statuses in
    one loop and runs them in another.  The loop target is a new map, so the
    binds above it are cancelled; what the run reads is what was live at the
    ``append``."""
    source = (
        "def f(self, datasets):\n"
        "    sqlDIU = 'UPDATE ATLAS_PANDA.JEDI_Datasets SET status=:status WHERE datasetID=:datasetID'\n"
        "    varMap = {}\n"
        "    varMap[':status'] = 'unrelated'\n"
        "    varMapList = []\n"
        "    for datasetID, done in datasets:\n"
        "        varMap = {}\n"
        "        if done:\n"
        "            varMap[':status'] = 'done'\n"
        "        else:\n"
        "            varMap[':status'] = 'failed'\n"
        "        varMapList.append(varMap)\n"
        "    for varMap in varMapList:\n"
        "        self.cur.execute(sqlDIU + comment, varMap)\n"
    )
    func = _func(source)

    assert _bound_at(func, ":status") == [["'done'", "'failed'"]]


def test_a_bind_below_the_run_does_not_reach_it_round_the_loop_past_a_rebuild():
    """``copyArchive.main``'s loop starts with ``var_map = {}``, checks the
    site and then binds the job query further down.  The next iteration's
    run cannot see that bind: the rebuild at the top of the body comes first."""
    source = (
        "def f(sites):\n"
        "    sqlS = 'SELECT status FROM ATLAS_PANDA.SiteData WHERE site=:site'\n"
        "    sqlJ = 'SELECT PandaID FROM ATLAS_PANDA.jobsActive4 WHERE computingSite=:site'\n"
        "    for site in sites:\n"
        "        var_map = {}\n"
        "        var_map[':site'] = site\n"
        "        taskBuffer.querySQLS(sqlS, var_map)\n"
        "        var_map = {}\n"
        "        var_map[':site'] = site.upper()\n"
        "        taskBuffer.querySQLS(sqlJ, var_map)\n"
    )
    func = _func(source)

    assert _bound_at(func, ":site") == [["site"], ["site.upper()"]]


def test_a_test_on_a_name_reassigned_since_does_not_part_a_bind_from_its_run():
    """Read as written, the unlock sits under ``not toSkip`` and under
    ``not (not toSkip)``, and its own bind could reach nothing.  The test
    above it described ``toSkip`` before the reassignment."""
    source = (
        "def f(self, tasks):\n"
        "    sqlTUU = 'UPDATE ATLAS_PANDA.JEDI_Tasks SET lockedBy=NULL WHERE jediTaskID=:jediTaskID AND status=:status'\n"
        "    for task in tasks:\n"
        "        toSkip = False\n"
        "        varMap = {}\n"
        "        varMap[':status'] = 'top'\n"
        "        if not toSkip:\n"
        "            if task.bad:\n"
        "                toSkip = True\n"
        "            if not toSkip:\n"
        "                pass\n"
        "            else:\n"
        "                varMap = {}\n"
        "                varMap[':status'] = task.status\n"
        "                self.cur.execute(sqlTUU + comment, varMap)\n"
    )
    func = _func(source)

    assert _bound_at(func, ":status") == [["task.status"]]


def test_a_bind_reading_its_own_key_keeps_the_one_above():
    """``varMap[":diag"] = truncate(varMap[":diag"])`` shortens the text
    above it rather than replacing it, and the text is the diagnostic
    template the map files against the column."""
    source = (
        "def f(self, batch):\n"
        "    sqlSUP = 'UPDATE ATLAS_PANDA.jobsActive4 SET supErrorDiag=:diag WHERE PandaID=:PandaID'\n"
        "    varMap = {}\n"
        "    varMap[':diag'] = f'invalid batchID={batch}'\n"
        "    varMap[':diag'] = JobSpec.truncateStringAttr('supErrorDiag', varMap[':diag'])\n"
        "    self.cur.execute(sqlSUP + comment, varMap)\n"
    )
    func = _func(source)

    assert _bound_at(func, ":diag") == [
        ["JobSpec.truncateStringAttr('supErrorDiag', varMap[':diag'])", "f'invalid batchID={batch}'"]
    ]


# ---------------------------------------------------------------------------
# An annotated assignment is an assignment
# ---------------------------------------------------------------------------


def test_an_annotated_dict_is_the_list_of_tables_it_is():
    """The shape that cost the map eleven facts.  ``getDispatchDatasetsPerUser``
    writes one table in the literal and adds two more by subscript; upstream
    annotated the literal and changed nothing else about the function, so the
    three tables went unnamed and the statement was reported as "table supplied
    at run time and not resolved"."""
    source = (
        "def f(self):\n"
        "    tableStatMap: dict[str, Any] = {'jobsDefined4': ['defined', 'assigned']}\n"
        "    if not onlyActive:\n"
        "        tableStatMap['jobsActive4'] = None\n"
        "        tableStatMap['jobsArchived4'] = None\n"
    )
    func = _func(source)

    assert sorted(sql._sequence_values(func, "tableStatMap")) == [
        "jobsActive4",
        "jobsArchived4",
        "jobsDefined4",
    ]


def test_a_bind_written_as_an_annotated_dict_literal_is_read():
    """The harvester, worker and data-carousel modules write their binds as
    dict literals, and master annotates 45 of those keys across 12 files.
    ``harvester_workers.pilotStatus`` was a subject that existed only through
    them, so reading ``ast.Assign`` alone took the whole node away."""
    source = (
        "def f(self):\n"
        "    var_map: dict[str, Any] = {':site': site, ':status': 'running'}\n"
    )
    func = _func(source)

    assert [
        ast.literal_eval(b.value) for b in sql.bound_values(func, "var_map", ":status")
    ] == ["running"]


def test_an_annotated_empty_dict_restarts_the_binding_window():
    """The window only ever moves later, so reading one more assignment takes
    binds *away* -- and that is the point.  ``copyArchive.main`` writes
    ``var_map: dict[str, Any] = {}`` on the line above a forwarded call, so the
    map was attributing every earlier bind to a statement whose var map had
    just been emptied."""
    source = (
        "def f():\n"
        "    sql = 'SELECT PandaID FROM ATLAS_PANDA.jobsActive4 WHERE jobStatus=:jobStatus'\n"
        "    var_map = {}\n"
        "    var_map[':jobStatus'] = 'activated'\n"
        "    var_map: dict[str, Any] = {}\n"
        "    status, res = taskBuffer.querySQLS(sql, var_map)\n"
    )
    func = _func(source)

    runs = sql.executions(func)

    assert [
        ast.literal_eval(b.value)
        for run in runs
        for b in sql.bound_values(func, run.varmap, ":jobStatus", run.window)
    ] == []


def test_a_declaration_without_a_value_writes_nothing():
    """``tables: list[str]`` states a type and assigns nothing.  Read as a write
    it would hand the name whatever the *next* statement assigns -- and here the
    sequence would be withdrawn entirely, because there is no value to read."""
    source = "def f():\n    tables: list[str]\n    tables = ['jobsActive4']\n"
    func = _func(source)

    assert sql._sequence_values(func, "tables") == ["jobsActive4"]


def test_an_annotated_literal_is_a_value_the_name_can_hold():
    """``_literal_values`` accumulates rather than withdrawing, so an annotated
    initial value it could not see made the answer quietly smaller instead of
    absent.  34 names on master have exactly this shape."""
    source = (
        "def f():\n"
        "    table: str = 'jobsActive4'\n"
        "    if archived:\n"
        "        table = 'jobsArchived4'\n"
    )
    func = _func(source)
    name = ast.parse("table", mode="eval").body

    assert sorted(sql._literal_values(func, name)) == ["jobsActive4", "jobsArchived4"]


def test_an_augmented_assignment_is_still_a_fragment_not_a_statement():
    """``pathcond.targets_of`` leaves ``AugAssign`` out on purpose.  Folding it in would
    make ``sql += ' AND x=1'`` look like the whole statement rather than a piece
    of it."""
    source = (
        "def f(self):\n"
        "    sql = 'SELECT PandaID FROM ATLAS_PANDA.jobsActive4 '\n"
        "    sql += 'WHERE jobStatus=:jobStatus'\n"
        "    self.cur.execute(sql + comment, varMap)\n"
    )
    func = _func(source)

    assert [ast.unparse(target) for node in ast.walk(func) for target in pathcond.targets_of(node)] == ["sql"]
    assert _held(func, "sql") == ["SELECT PandaID FROM ATLAS_PANDA.jobsActive4 WHERE jobStatus=:jobStatus"]

def test_an_annotated_local_is_a_value_the_name_can_hold():
    """``assigned_expressions`` is what the emit pass follows one hop through.

    ``server/panda.py`` annotates the message it is about to log, so without
    this the 403 it writes has no text: the call resolves to nothing and the
    line vanishes from the map rather than arriving unexplained."""
    source = (
        "def f():\n"
        "    if not validate_method(method_name, api_module, version):\n"
        "        error_message: str | None = f'method {method_name} is forbidden'\n"
        "        tmp_log.error(error_message)\n"
    )
    func = _func(source)

    held = pathcond.assigned_expressions(func)["error_message"]

    assert [ast.unparse(e) for e in held] == ["f'method {method_name} is forbidden'"]


def test_a_declaration_binds_nothing_to_the_name_it_declares():
    """``found: list[str]`` states a type.  Counting it as a binding would have
    the name hold whatever the next statement assigns, one line early."""
    source = "def f():\n    found: list[str]\n    found = compute()\n"
    func = _func(source)

    held = pathcond.assigned_expressions(func)["found"]

    assert [ast.unparse(e) for e in held] == ["compute()"]


def test_an_annotated_initialiser_is_the_definition_behind_a_bare_test():
    """``if not ret_req:`` names no predicate; the write it tests does.

    ``data_carousel_module`` takes the row count into an annotated local and
    then branches on it, which is the corpus's way of saying "the compare-and-
    set lost".  Four of its methods are written exactly this way."""
    source = (
        "def f(self):\n"
        "    self.cur.execute(sql_update + comment, var_map)\n"
        "    ret_req: int = self.cur.rowcount\n"
        "    if not ret_req:\n"
        "        tmp_log.warning('already terminated; cannot be cancelled ; skipped')\n"
    )
    func = _func(source)

    definition = pathcond.single_definition(func, "ret_req")

    assert definition is not None and ast.unparse(definition) == "self.cur.rowcount"


def test_a_name_written_twice_explains_no_branch_however_it_is_spelled():
    """Two writes are fan-out, and one of them is not the reason for a branch.

    This is the half of the ``AnnAssign`` reading that *withdraws* an answer,
    and it is the contract working rather than a fact going missing.  Reading
    only ``ast.Assign``, ``misc_standalone_module`` had ``if res:`` annotated
    with ``res := []`` -- the write in that test's own ``else``, which runs
    because the test failed.  Nine of the ten substitutions this withdraws
    pointed at a statement that runs after the test or inside its else."""
    source = (
        "def f(self):\n"
        "    res: Sequence[Any] = self.cur.fetchone()\n"
        "    if res:\n"
        "        tmp_log.debug(f'task has status: {res[0]}')\n"
        "    else:\n"
        "        res = []\n"
        "        tmp_log.debug('task not found')\n"
    )
    func = _func(source)

    assert pathcond.single_definition(func, "res") is None


def test_a_scope_is_read_once_and_every_caller_gets_answers_of_its_own():
    """The index kept on a scope answers as a pass over it would, however often asked.

    The three readings walk different nodes -- a nested function's writes count
    for names and not for ``self.<field>`` -- and a caller editing the list it
    was handed must not change what the next caller is told."""
    source = (
        "def f(self, x):\n"
        "    a = 1\n"
        "    a += 2\n"
        "    b, c = g()\n"
        "    d: int = 3\n"
        "    self.status = 'x'\n"
        "    self.status += 'y'\n"
        "    self.pair, e = h()\n"
        "    def inner():\n"
        "        a = 4\n"
        "        self.status = 'inner'\n"
    )
    func = _func(source)

    def passed_over(name):
        return [
            ast.unparse(node.value)
            for node in ast.walk(func)
            for target in pathcond.targets_of(node)
            for element in (target.elts if isinstance(target, ast.Tuple) else [target])
            if isinstance(element, ast.Name) and element.id == name
        ]

    for name in ("a", "b", "c", "d", "e", "x"):
        found = passed_over(name)
        expected = found[0] if len(found) == 1 else None
        definition = pathcond.single_definition(func, name)
        assert (ast.unparse(definition) if definition is not None else None) == expected, name
    assigned = pathcond.assigned_expressions(func)
    assert {name: [ast.unparse(v) for v in values] for name, values in assigned.items()} == {
        "a": ["1", "2", "4"],
        "d": ["3"],
    }
    status = pathcond.attribute_expressions(func, "status")
    assert [ast.unparse(v) for v in status] == ["'x'", "'y'"]
    assert [ast.unparse(v) for v in pathcond.attribute_expressions(func, "pair")] == ["h()"]

    assigned["a"].clear()
    status.clear()

    assert len(pathcond.assigned_expressions(func)["a"]) == 3
    assert len(pathcond.attribute_expressions(func, "status")) == 2


def test_a_branch_is_not_explained_by_the_write_its_else_makes():
    """The same site read through ``path_condition``, which is what the map stores.

    The unit above says the definition is withdrawn; this says the guard the
    map writes down no longer carries the backwards annotation."""
    source = (
        "def f(self):\n"
        "    res: Sequence[Any] = self.cur.fetchone()\n"
        "    if res:\n"
        "        tmp_log.debug(f'task has status: {res[0]}')\n"
        "    else:\n"
        "        res = []\n"
    )
    func = _func(source)
    inside = next(
        n for n in ast.walk(func) if isinstance(n, ast.Call) and ast.unparse(n).startswith("tmp_log")
    )

    assert pathcond.path_condition(inside) == ["res"]


def test_a_row_count_taken_into_an_annotated_local_is_still_a_row_count():
    """The line saying whether a compare-and-set landed is what `emit` looks for.

    ``data_carousel_module`` writes ``ret_req: int = self.cur.rowcount`` in four
    of its methods, and eleven locals across the corpus are spelled that way.
    Without this the junction carries the message and not the fact that the
    message reports how many rows moved."""
    source = (
        "def f(self):\n"
        "    self.cur.execute(sql_update + comment, var_map)\n"
        "    ret_req: int = self.cur.rowcount\n"
        "    tmp_log.debug(f'done with {ret_req} rows')\n"
    )
    func = _func(source)

    assert emit._row_counts(func) == {"ret_req"}


def test_a_construction_hands_over_what_an_annotated_field_receives():
    """``Finisher.__init__`` writes ``self.dataset: DatasetSpec | None = dataset``.

    ``datasetManager`` calls ``Finisher(taskBuffer, None, job)``, so what it
    hands over for that field is ``None``.  Keyed by the field rather than the
    parameter, which is why the annotated write has to be read: without it the
    entry point looked as though it supplied no dataset argument at all."""
    source = (
        "class Finisher:\n"
        "    def __init__(self, taskBuffer, dataset, job=None):\n"
        "        self.taskBuffer = taskBuffer\n"
        "        self.dataset: DatasetSpec | None = dataset\n"
        "        self.job: JobSpec | None = job\n"
    )
    init = next(
        n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.FunctionDef)
    )
    call = ast.parse("Finisher(taskBuffer, None, job)", mode="eval").body

    assert trigger._handover(call, init) == {
        "taskBuffer": "taskBuffer",
        "dataset": "None",
        "job": "job",
    }


def test_an_annotated_module_constant_is_a_module_constant():
    """A dispatch table the reader cannot explain still belongs in the denominator.

    ``processor.HANDLERS``, ``data_carousel_ops.OPERATIONS`` and
    ``jedi_config._SECTION_CLASSES`` are all annotated, and all three were
    invisible rather than unexplained -- which is the failure the coverage
    figure exists to prevent."""
    tree = ast.parse(
        "HANDLERS: dict[str, Callable] = {'grep': _handle_grep}\n"
        "EC_Kill: Final[int] = 100\n"
        "declared: list[str]\n"
    )

    found = {name: type(node).__name__ for name, node in errorcode._iter_module_constants(tree)}

    assert found == {"HANDLERS": "AnnAssign", "EC_Kill": "AnnAssign"}


# ---------------------------------------------------------------------------
# Two arms of one statement are two statements, however the caller arrived
# ---------------------------------------------------------------------------


def _unattached(source: str, name: str = "f"):
    """Parse without attaching parents, the way half this function's callers do."""
    tree = ast.parse(source)
    return next(
        n
        for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name
    )


def test_two_arms_stay_two_statements_when_no_caller_attached_parents():
    """``boundary`` and ``trigger`` call ``executions`` and attach nothing.

    The exclusive reading in ``_run_variants`` is answered off the ancestor
    chain, and a tree without one does not raise -- every fragment reads as
    unconditional, so both arms are concatenated into a statement nobody runs
    and the two the code does run are never seen.
    """
    source = (
        "def f(self):\n"
        "    sql = 'SELECT a FROM ATLAS_PANDA.t1 '\n"
        "    if wanted:\n"
        "        sql += 'WHERE b=:b '\n"
        "    else:\n"
        "        sql += 'WHERE c=:c '\n"
        "    self.cur.execute(sql, varMap)\n"
    )

    assert sorted(r.sql for r in sql.executions(_unattached(source))) == [
        "SELECT a FROM ATLAS_PANDA.t1 WHERE b=:b ",
        "SELECT a FROM ATLAS_PANDA.t1 WHERE c=:c ",
    ]

# ---------------------------------------------------------------------------
# A statement is not run by every call in its function
# ---------------------------------------------------------------------------
#
# The variants of a name are read across the whole function and were then
# paired with every execution in it.  ``copyArchive.main`` assigns ``sql``
# forty-four times and runs it twenty-three, so 21 of the 31 statements the
# build reported as having a table name it could not resolve were one
# statement, claimed at 21 call sites that never run it -- and resolved at the
# one that does.


_TWO_ARMS = (
    "def f(self):\n"
    "    if wanted:\n"
    "        sql = 'SELECT a FROM ATLAS_PANDA.t1 '\n"
    "        self.cur.execute(sql, varMap)\n"
    "    else:\n"
    "        sql = 'SELECT b FROM ATLAS_PANDA.t2 '\n"
    "        self.cur.execute(sql, varMap)\n"
)


def test_a_call_under_one_arm_does_not_run_the_other_arm_s_statement():
    """Both halves of the rule at once.

    The ``else`` binding is *below* the ``if`` call, so position rules it out
    there; the ``if`` binding is above the ``else`` call and only the path
    condition separates them.  Neither reading alone leaves each call with one
    statement.
    """
    runs = sorted(sql.executions(_func(_TWO_ARMS)), key=lambda r: r.call.lineno)

    assert [r.sql for r in runs] == [
        "SELECT a FROM ATLAS_PANDA.t1 ",
        "SELECT b FROM ATLAS_PANDA.t2 ",
    ]


def test_the_arms_are_kept_apart_without_a_caller_attaching_parents():
    """The rule is read off the ancestor chain, and half the callers bring none."""
    runs = sorted(sql.executions(_unattached(_TWO_ARMS)), key=lambda r: r.call.lineno)

    assert [r.sql for r in runs] == [
        "SELECT a FROM ATLAS_PANDA.t1 ",
        "SELECT b FROM ATLAS_PANDA.t2 ",
    ]


def test_a_statement_assigned_after_the_call_is_not_what_it_ran():
    source = (
        "def f(self):\n"
        "    sql = 'SELECT a FROM ATLAS_PANDA.t1 '\n"
        "    self.cur.execute(sql, varMap)\n"
        "    sql = 'SELECT b FROM ATLAS_PANDA.t2 '\n"
        "    self.cur.execute(sql, varMap)\n"
    )

    runs = sql.executions(_func(source))
    first = [r.sql for r in runs if r.call.lineno == 3]

    assert first == ["SELECT a FROM ATLAS_PANDA.t1 "]


def test_a_loop_holding_both_keeps_a_statement_assigned_after_the_call():
    """The half of the position rule that is not about position.

    Every pass but the first runs the statement the foot of the body left
    behind, so dropping it would lose what the loop actually executes.
    """
    source = (
        "def f(self):\n"
        "    sql = 'SELECT a FROM ATLAS_PANDA.t1 '\n"
        "    for row in rows:\n"
        "        self.cur.execute(sql, varMap)\n"
        "        sql = 'SELECT b FROM ATLAS_PANDA.t2 '\n"
    )

    runs = sql.executions(_func(source))

    assert sorted(r.sql for r in runs) == [
        "SELECT a FROM ATLAS_PANDA.t1 ",
        "SELECT b FROM ATLAS_PANDA.t2 ",
    ]


def test_two_arms_building_the_same_statement_do_not_lose_the_second_one():
    """Rule 39, and ``activateJob`` is where it costs four statements.

    The fragments are keyed by the text they render, so two arms writing the
    same statement leave only the first arm's fragment behind -- and asking
    *that* whether it reaches the call in the second arm answers no.  Asking
    every fragment that renders the text answers yes, which is what the code
    does.  Filling cannot pool them the same way: ``_braces_are_literal`` is
    answered per statement, for the reason its own docstring gives.

    Each call still sees the statement once per variant rendering it, because
    the variants are listed and not set -- 11 duplicate pairs over the corpus,
    which this round leaves alone.  What it fixes is the second call seeing it
    at all.
    """
    source = (
        "def f(self):\n"
        "    if wanted:\n"
        "        sqlF = 'UPDATE ATLAS_PANDA.filesTable4 SET x=:x '\n"
        "        self.cur.execute(sqlF, varMap)\n"
        "    else:\n"
        "        sqlF = 'UPDATE ATLAS_PANDA.filesTable4 SET x=:x '\n"
        "        self.cur.execute(sqlF, varMap)\n"
    )

    runs = sql.executions(_func(source))

    assert {r.call.lineno for r in runs} == {4, 7}
    assert {r.sql for r in runs} == {"UPDATE ATLAS_PANDA.filesTable4 SET x=:x "}


def test_a_fragment_both_arms_share_is_appended_to_each_arm():
    """``getNumberJobsUser``: the ``+=`` after an ``if``/``else`` that assigns in each arm belongs to both."""
    source = (
        "def f(self, workingGroup):\n"
        "    if workingGroup is not None:\n"
        "        sql = 'SELECT COUNT(*) FROM ATLAS_PANDA.jobsActive4 WHERE workingGroup=:workingGroup '\n"
        "    else:\n"
        "        sql = 'SELECT COUNT(*) FROM ATLAS_PANDA.jobsActive4 WHERE workingGroup IS NULL '\n"
        "    sql += 'AND NOT jobStatus IN (:failed,:merging) '\n"
        "    self.cur.execute(sql, varMap)\n"
    )

    runs = sql.executions(_func(source))

    assert sorted(r.sql for r in runs) == [
        "SELECT COUNT(*) FROM ATLAS_PANDA.jobsActive4 WHERE workingGroup IS NULL AND NOT jobStatus IN (:failed,:merging) ",
        "SELECT COUNT(*) FROM ATLAS_PANDA.jobsActive4 WHERE workingGroup=:workingGroup AND NOT jobStatus IN (:failed,:merging) ",
    ]


def test_an_arm_that_uses_the_tested_name_does_not_join_the_other_arm():
    """``getActiveDebugJobs``: one execution of an ``if`` enters one arm.

    The arm calls ``workingGroup.upper()`` and assigns ``compactDN``, and a
    reading that drops a test once its name is given a new value since -- the
    one the walk had -- let the two arms meet and built a statement holding
    both ``UPPER(workingGroup) IN`` and ``prodUserName=``, which no run sends.
    """
    source = (
        "def f(self, workingGroup, dn):\n"
        "    sql = 'SELECT PandaID FROM ATLAS_PANDA.jobsActive4 WHERE '\n"
        "    if workingGroup is not None:\n"
        "        sql += 'UPPER(workingGroup) IN (:wg1) AND '\n"
        "        varMap[':wg1'] = workingGroup.upper()\n"
        "        workingGroup = None\n"
        "    else:\n"
        "        sql += 'prodUserName=:prodUserName AND '\n"
        "    sql += 'specialHandling=:sh '\n"
        "    self.cur.execute(sql, varMap)\n"
    )

    assert sorted(r.sql for r in sql.executions(_func(source))) == [
        "SELECT PandaID FROM ATLAS_PANDA.jobsActive4 WHERE UPPER(workingGroup) IN (:wg1) AND specialHandling=:sh ",
        "SELECT PandaID FROM ATLAS_PANDA.jobsActive4 WHERE prodUserName=:prodUserName AND specialHandling=:sh ",
    ]


def test_a_call_inside_a_test_is_not_a_new_value_after_it():
    """A test spanning lines holds calls below its ``if`` line, and they ran before the test did."""
    func = _func(
        "def f(spec, other):\n"
        "    if (\n"
        "        spec.isExpress()\n"
        "        and other\n"
        "    ):\n"
        "        a = 1\n"
        "    if not (spec.isExpress() and other):\n"
        "        b = 2\n"
    )
    first, second = (n for n in ast.walk(func) if isinstance(n, ast.Assign))

    assert pathcond.parted(first, second, exits=False)


def test_a_new_value_in_an_arm_the_pass_does_not_enter_leaves_two_tests_contradicting():
    """``prepareTasksToBeFinished_JEDI``: ``toSkip = True`` sits in the ``else`` of the ``if`` above the first test.

    A pass that decided ``new`` under the first ``not toSkip`` never ran that
    ``else``, so the second ``not toSkip`` still holds and the unlock below it
    does not run with ``new``.  Counted by line alone, the new value read the
    contradiction away and the unlock's log line went to the branch.
    """
    func = _func(
        "def f(status):\n"
        "    toSkip = False\n"
        "    if status in ('running', 'ready'):\n"
        "        if not toSkip:\n"
        "            new = 'defined'\n"
        "    else:\n"
        "        toSkip = True\n"
        "    if not toSkip:\n"
        "        write = 1\n"
        "    else:\n"
        "        unlock = 1\n"
    )
    assigned = {n.targets[0].id: n for n in ast.walk(func) if isinstance(n, ast.Assign)}

    assert pathcond.parted(assigned["new"], assigned["unlock"], exits=False)


def test_a_test_and_its_negation_are_matched_as_they_mean_not_as_spelled():
    """``ppEventServiceJob``: ``if noNewJob:`` and, further down, ``if not noNewJob:`` cannot both hold.

    Their conditions are written ``noNewJob`` and ``not noNewJob``, and only an
    ``else`` writes ``not (noNewJob)``, so compared as spelled the two read as
    one run.  A bare name's annotation is left out the same way; a name given a
    new value between the two tests still parts nothing.
    """
    func = _func(
        "def f(noNewJob):\n"
        "    if noNewJob:\n"
        "        a = 1\n"
        "    if not noNewJob:\n"
        "        b = 2\n"
    )
    first, second = (n for n in ast.walk(func) if isinstance(n, ast.Assign))

    assert pathcond.parted(first, second, exits=False)
    assert pathcond.exclusive(["noNewJob"], ["not noNewJob"])
    assert pathcond.exclusive(["done  [done := n > 0]"], ["not (done)  [done := n > 0]"])
    assert pathcond.negates("x == 1", "not x == 1")
    assert not pathcond.negates("x == 1", "not x == 2")
    assert not pathcond.exclusive(["noNewJob"], ["noNewJob"])

    func = _func(
        "def f(items):\n"
        "    if not items:\n"
        "        return\n"
        "    a = 1\n"
        "    items = [one for one in items if one]\n"
        "    if not items:\n"
        "        b = 2\n"
    )
    first, second = (n for n in ast.walk(func) if isinstance(n, ast.Assign) and n.targets[0].id in "ab")

    assert pathcond.exclusive(pathcond.reachable_when(first), pathcond.reachable_when(second))
    assert not pathcond.parted(first, second)


def test_a_comparison_is_the_negation_of_its_opposite():
    """``AdderAtlasPlugin.update_outputs`` fills a value under the ``else`` of ``if
    file.lfn not in alt`` and reads it under ``if file.lfn in alt``: the two arms
    cannot meet.  ``==`` / ``!=``, ``in`` / ``not in`` and ``is`` / ``is not`` are
    each other's negation, read either way round, and ``not not x`` is ``x``;
    ``<`` and ``>=`` are not (a set's order, a NaN).
    """
    assert pathcond.negates("numSlots == -1", "numSlots != -1")
    assert pathcond.negates("numSlots != -1", "numSlots == -1")
    assert pathcond.negates("x in L", "x not in L")
    assert pathcond.negates("x not in L", "x in L")
    assert pathcond.negates("x is None", "x is not None")
    assert pathcond.negates("x", "not not not x")
    assert pathcond.exclusive(["x not in L"], ["x in L"])
    assert not pathcond.negates("x < 1", "x >= 1")
    assert not pathcond.negates("x == 1", "x != 2")

    func = _func(
        "def f(lfn, alt):\n"
        "    if lfn not in alt:\n"
        "        a = 1\n"
        "    else:\n"
        "        b = 2\n"
        "    if lfn in alt:\n"
        "        c = 3\n"
    )
    assigned = {n.targets[0].id: n for n in ast.walk(func) if isinstance(n, ast.Assign)}

    assert pathcond.parted(assigned["a"], assigned["c"], exits=False)
    assert not pathcond.parted(assigned["b"], assigned["c"], exits=False)


def test_a_write_reaches_a_read_round_a_loop_whose_arms_one_pass_keeps_apart():
    """``parse_condition_string``: the ``else`` of ``if not items`` replaces ``s``, and the next pass's ``if`` hands it on.

    One pass runs one arm, so the two arms part within it, and a reader asking
    what one pass runs still hears that.  The loop binds ``items`` afresh, so
    the next pass can take the other arm and read the replaced string.  Only
    the way round counts: where every arm binds the name again before the
    read, the write the first test chose does not reach the second test's
    ``else``; and a flag no pass changes keeps every pass in its arm.
    """
    func = _func(
        "def f(start):\n"
        "    s = start\n"
        "    while True:\n"
        "        items = find(s)\n"
        "        if not items:\n"
        "            return convert(s)\n"
        "        else:\n"
        "            s = s.replace(items[0], 'x')\n"
    )
    first, replaced = (n for n in ast.walk(func) if isinstance(n, ast.Assign) and n.targets[0].id == "s")
    use = next(n for n in ast.walk(func) if isinstance(n, ast.Return))

    assert pathcond.surviving_writes([first, replaced], use, as_run=True) == [first, replaced]
    assert pathcond.parted(replaced, use)
    assert not pathcond.compatible(replaced, use, current=True)

    func = _func(
        "def f(specs):\n"
        "    for spec in specs:\n"
        "        if spec.isMaster():\n"
        "            n = 1\n"
        "        else:\n"
        "            n = 2\n"
        "        if spec.isMaster():\n"
        "            m = 0\n"
        "        else:\n"
        "            m = n\n"
    )
    one, two = (n for n in ast.walk(func) if isinstance(n, ast.Assign) and n.targets[0].id == "n")
    use = next(n for n in ast.walk(func) if isinstance(n, ast.Assign) and isinstance(n.value, ast.Name))

    assert pathcond.surviving_writes([one, two], use, as_run=True) == [two]

    func = _func(
        "def f(items, flag):\n"
        "    a = None\n"
        "    for item in items:\n"
        "        if flag:\n"
        "            a = item\n"
        "        else:\n"
        "            b = a\n"
    )
    before, inner = (n for n in ast.walk(func) if isinstance(n, ast.Assign) and n.targets[0].id == "a")
    use = next(n for n in ast.walk(func) if isinstance(n, ast.Assign) and n.targets[0].id == "b")

    assert pathcond.surviving_writes([before, inner], use, as_run=True) == [before]


def test_a_call_carrying_its_own_statement_has_no_fragment_to_place():
    """"No fragment reaches this call" and "there is no fragment" are
    different answers.  Thirty pairs in the corpus write the statement into
    the call, and reading the second as the first drops the daemon layer."""
    func = _func(
        "def f():\n"
        "    if wanted:\n"
        "        status, res = taskBuffer.querySQLS("
        "'SELECT a FROM ATLAS_PANDA.t1', var_map)\n"
    )

    assert [r.sql for r in sql.executions(func)] == ["SELECT a FROM ATLAS_PANDA.t1"]


def test_a_statement_thrown_away_before_the_call_is_not_what_it_ran():
    """What position alone leaves behind.

    ``copyArchive.main`` binds ``sql`` at line 532 and runs it hundreds of
    lines below, and the binding does reach that call -- what stops it is the
    plain ``sql = ...`` in between, each of which discards the statement
    before it.  Nine of its statements were still read as unresolved after
    the reachability rule for exactly that reason.
    """
    source = (
        "def f(self):\n"
        "    sql = 'SELECT a FROM ATLAS_PANDA.t1 '\n"
        "    self.cur.execute(sql, varMap)\n"
        "    sql = 'SELECT b FROM ATLAS_PANDA.t2 '\n"
        "    self.cur.execute(sql, varMap)\n"
    )

    runs = sql.executions(_func(source))
    second = [r.sql for r in runs if r.call.lineno == 5]

    assert second == ["SELECT b FROM ATLAS_PANDA.t2 "]


def test_a_rebinding_under_an_arm_the_call_is_not_in_throws_nothing_away():
    """One arm's rebinding is not the other arm's."""
    source = (
        "def f(self):\n"
        "    sql = 'SELECT a FROM ATLAS_PANDA.t1 '\n"
        "    if wanted:\n"
        "        sql = 'SELECT b FROM ATLAS_PANDA.t2 '\n"
        "    else:\n"
        "        self.cur.execute(sql, varMap)\n"
    )

    assert [r.sql for r in sql.executions(_func(source))] == [
        "SELECT a FROM ATLAS_PANDA.t1 "
    ]


def test_two_independent_choices_make_four_statements_not_five_halves():
    """``insertTaskParams_JEDI`` builds one INSERT across two unrelated ``if``s.

    The arms were grouped by connected component, and the two groups touch --
    the inner ``if`` repeats the outer test, so one arm of each is exclusive
    with one arm of the other.  Merged into a single alternative, exactly one
    fragment of the five survives per variant, and every statement comes out
    with an opening parenthesis and no closing one or the reverse.  ``writes``
    reads none of them, which is how the map lost ``T_TASK.priority`` while
    reporting six statements it could not name a column of.
    """
    source = (
        "def f(self):\n"
        "    sql = 'INSERT INTO ATLAS_DEFT.T_TASK (a,b,c) VALUES '\n"
        "    if self.backend == 'oracle':\n"
        "        sql += '(SEQ.nextval,'\n"
        "    else:\n"
        "        sql += '(:nextval,'\n"
        "    sql += ':status,'\n"
        "    if parent_tid is None:\n"
        "        if self.backend == 'oracle':\n"
        "            sql += 'SEQ.currval) '\n"
        "        else:\n"
        "            sql += ':currval) '\n"
        "    else:\n"
        "        sql += ':parent_tid) '\n"
        "    self.cur.execute(sql, varMap)\n"
    )
    head = "INSERT INTO ATLAS_DEFT.T_TASK (a,b,c) VALUES "

    assert sorted(r.sql for r in sql.executions(_func(source))) == sorted(
        [
            head + "(SEQ.nextval,:status,SEQ.currval) ",
            head + "(SEQ.nextval,:status,:parent_tid) ",
            head + "(:nextval,:status,:currval) ",
            head + "(:nextval,:status,:parent_tid) ",
        ]
    )


# --------------------------------------------------------------------------- #
# where an inline value is anchored
# --------------------------------------------------------------------------- #


def _branches_of(source: str, subject: str):
    """``{(outcome, path condition)}`` for one subject, from one module."""
    _s, junctions, _c, _u, _conf, _att = _sql_extract(source)
    return {
        (b.outcome, tuple(b.path_condition))
        for j in junctions
        if j.subject == subject
        for b in j.branches
    }


def test_a_statement_appended_in_four_steps_reads_like_the_same_one_written_whole():
    """The tool used to answer differently depending on how the text was typed.

    ``assigns_in`` reads one fragment at a time, which works for ``SET
    col=value`` because that pair is adjacent, and cannot work for an
    ``INSERT``: its columns and its values are two parenthesised runs and the
    correspondence between them is positional, so it straddles whatever line
    breaks the author chose.  Four ``+=`` and one implicit concatenation are
    the same statement under the same guard, and the map has to say so.
    """
    appended = (
        "class M:\n"
        "    def insertUser(self, name):\n"
        "        sqlAdd = 'INSERT INTO ATLAS_PANDAMETA.users '\n"
        "        sqlAdd += '(ID,NAME,NCURRENT) '\n"
        "        sqlAdd += \"VALUES(SEQ.nextval,:name,\"\n"
        "        sqlAdd += '0) '\n"
        "        self.cur.execute(sqlAdd + comment, varMap)\n"
    )
    whole = (
        "class M:\n"
        "    def insertUser(self, name):\n"
        "        sqlAdd = ('INSERT INTO ATLAS_PANDAMETA.users '\n"
        "                  '(ID,NAME,NCURRENT) '\n"
        "                  \"VALUES(SEQ.nextval,:name,\"\n"
        "                  '0) ')\n"
        "        self.cur.execute(sqlAdd + comment, varMap)\n"
    )

    assert _branches_of(appended, "users.NCURRENT") == {("0", ())}
    assert _branches_of(appended, "users.NCURRENT") == _branches_of(whole, "users.NCURRENT")


def test_a_value_appended_under_a_test_carries_that_test():
    """The anchor is the fragment the value is in, not the head of the run.

    Grouping fragments that share a guard would put this value on the
    unconditional head and drop the ``if`` -- which is the reason the anchor
    exists at all.
    """
    source = (
        "class M:\n"
        "    def updateOne(self, wanted):\n"
        "        sqlU = 'UPDATE ATLAS_PANDA.jobsActive4 SET jobStatus=:jobStatus'\n"
        "        if wanted:\n"
        "            sqlU += \",lockedby='jedi'\"\n"
        "        sqlU += ' WHERE PandaID=:PandaID '\n"
        "        self.cur.execute(sqlU + comment, varMap)\n"
    )

    # Table-qualified: ``lockedby`` is not a declared attribute of the spec
    # this fixture holds, which is beside the point being made here.
    assert _branches_of(source, "jobsActive4.lockedby") == {("jedi", ("wanted",))}


def test_an_upsert_reports_both_arms_with_their_own_reasons():
    """One local, two arms, one column, one value -- and two different reasons.

    ``assigns_in`` reads only ``SET``, so the ``INSERT`` arm found no fragment
    of its own and matched the sibling ``UPDATE``'s.  The two then collapsed in
    the ``(outcome, path condition)`` merge and the report stated the retry's
    conditions as necessary for a value the fresh claim also writes.
    """
    source = (
        "class M:\n"
        "    def claim(self, row):\n"
        "        if row is None:\n"
        "            sql = (\"INSERT INTO ATLAS_PANDA.jobsActive4 \"\n"
        "                   \"(PandaID,jobStatus) \"\n"
        "                   \"VALUES (:PandaID,'running') \")\n"
        "            self.cur.execute(sql + comment, varMap)\n"
        "        elif row[0] == 'pending':\n"
        "            sql = (\"UPDATE ATLAS_PANDA.jobsActive4 \"\n"
        "                   \"SET jobStatus='running' \"\n"
        "                   \"WHERE PandaID=:PandaID \")\n"
        "            self.cur.execute(sql + comment, varMap)\n"
    )

    assert _branches_of(source, "JobSpec.jobStatus") == {
        ("running", ("row is None",)),
        ("running", ("not (row is None)", "row[0] == 'pending'")),
    }


def test_a_statement_written_into_the_call_is_anchored_at_the_call():
    """The daemons hand the text straight to the task buffer, so there is no
    local to assemble and no fragment to anchor at.

    Anchoring at the call is wrong for a statement built from fragments --
    ``getTasksToExecCommand_JEDI`` appends its two forms in two arms of one
    ``if``, so the fragment carries the reason and the call does not -- and it
    is the only possibility here, where the call *is* where the text is.
    """
    source = (
        "class M:\n"
        "    def sweep(self, allFinished):\n"
        "        if allFinished:\n"
        "            self.taskBuffer.querySQLS(\n"
        "                \"UPDATE ATLAS_PANDA.jobsActive4 SET jobStatus='holding' \"\n"
        "                \"WHERE PandaID=:PandaID \", var_map)\n"
    )

    assert _branches_of(source, "JobSpec.jobStatus") == {("holding", ("allFinished",))}


def test_a_statement_built_from_fragments_is_not_anchored_at_its_call():
    """The control for the clause above, and the older finding it must not undo.

    ``getTasksToExecCommand_JEDI`` appends its two forms of one update in the
    two arms of an ``if``, so the fragment carries the condition and the call
    does not.  A call-anchored answer here would report both forms as running
    under whatever guarded the ``execute``.
    """
    source = (
        "class M:\n"
        "    def pick(self, release):\n"
        "        sqlU = 'UPDATE ATLAS_PANDA.jobsActive4 '\n"
        "        if release:\n"
        "            sqlU += \"SET jobStatus='activated' \"\n"
        "        else:\n"
        "            sqlU += \"SET jobStatus='holding' \"\n"
        "        sqlU += ' WHERE PandaID=:PandaID '\n"
        "        self.cur.execute(sqlU + comment, varMap)\n"
    )

    assert _branches_of(source, "JobSpec.jobStatus") == {
        ("activated", ("release",)),
        ("holding", ("not (release)",)),
    }


def test_the_value_is_found_after_the_table_name_is_filled_in():
    """Offsets have to be taken on the text ``writes`` read, not before it.

    ``_fold_filled`` interpolates a table name the loop supplies, which changes
    the length of the fragment carrying it.  Offsets measured on the unfilled
    text drift by that difference and point at the wrong column as soon as the
    hole is wider or narrower than ``{}``.
    """
    source = (
        "class M:\n"
        "    def sweep(self, wanted):\n"
        "        tables = ['ATLAS_PANDA.jobsActive4', 'ATLAS_PANDA.jobsDefined4']\n"
        "        for tableName in tables:\n"
        "            sqlU = f'UPDATE {tableName} '\n"
        "            sqlU += \"SET jobStatus='holding',lockedby='jedi' \"\n"
        "            sqlU += ' WHERE PandaID=:PandaID '\n"
        "            self.cur.execute(sqlU + comment, varMap)\n"
    )

    # One statement per table the loop supplies, and the value found in both.
    # The two fills differ in width by a character, so the second fragment
    # starts at 31 in one variant and 32 in the other; an offset taken before
    # filling would land outside it.
    assert _branches_of(source, "jobsActive4.jobStatus") == {("holding", ())}
    assert _branches_of(source, "jobsDefined4.jobStatus") == {("holding", ())}


# --------------------------------------------------------------------------- #
# the verb a statement uses on a row
# --------------------------------------------------------------------------- #


def test_creating_a_row_is_not_the_same_verb_as_changing_one():
    """An INSERT is an object entering the system; an UPDATE is one moving.

    Both were reported as ``written``, so nothing downstream could ask where a
    kind of row comes into existence.  ``sqltext.deletes``'s own docstring already
    said why that matters -- a DELETE followed by an INSERT on a command table
    means a second command silently replaced one that was never picked up --
    and the map had nowhere to put it.
    """
    assert sqlwrite._table_verbs("INSERT INTO ATLAS_PANDA.jobsActive4 (a) VALUES (1)") == [
        ("created", "jobsActive4")
    ]
    assert sqlwrite._table_verbs("UPDATE ATLAS_PANDA.jobsActive4 SET a=1") == [
        ("updated", "jobsActive4")
    ]
    assert sqlwrite._table_verbs("DELETE FROM ATLAS_PANDA.jobsActive4 WHERE a=1") == [
        ("deleted", "jobsActive4")
    ]
    assert sqlwrite._table_verbs("SELECT a FROM ATLAS_PANDA.jobsActive4") == [
        ("read", "jobsActive4")
    ]


def test_one_statement_can_both_create_and_read_the_same_kind_of_row():
    """``INSERT INTO a SELECT FROM a`` does both, and says so."""
    statement = (
        "INSERT INTO ATLAS_PANDA.jobsArchived4 (PandaID,jobStatus) "
        "SELECT PandaID,jobStatus FROM ATLAS_PANDA.jobsActive4"
    )

    assert sqlwrite._table_verbs(statement) == [
        ("created", "jobsArchived4"),
        ("read", "jobsActive4"),
    ]


def test_a_table_filled_by_an_insert_select_is_not_maintained_elsewhere():
    """Whether anything here maintains a table does not depend on the column
    list being readable.

    ``writes`` needs ``(columns) VALUES (values)`` because it answers which
    column gets which value.  ``log_task_attempt_start`` fills ``TASK_ATTEMPTS``
    from a ``SELECT``, ``metric_collector.update`` inserts positionally with no
    column list at all, and ``updateCache_JEDI`` interpolates its column list --
    five statements in the corpus, and every one of them made its table look
    like a dependency on something outside.
    """
    source = (
        "class M:\n"
        "    def logAttempt(self, taskID):\n"
        "        sqlI = 'INSERT INTO ATLAS_PANDA.TASK_ATTEMPTS (jediTaskID,attemptNr) '\n"
        "        sqlI += 'SELECT jediTaskID,attemptNr FROM ATLAS_PANDA.JEDI_Tasks '\n"
        "        self.cur.execute(sqlI + comment, varMap)\n"
        "    def readAttempt(self, taskID):\n"
        "        sqlS = 'SELECT a.attemptNr FROM ATLAS_PANDA.JEDI_Tasks t,ATLAS_PANDA.TASK_ATTEMPTS a '\n"
        "        sqlS += 'WHERE t.jediTaskID=a.jediTaskID AND t.status=:status '\n"
        "        self.cur.execute(sqlS + comment, varMap)\n"
    )
    modules = [_module(source, "pandaserver/taskbuffer/db_proxy_mods/task_module.py")]

    assert "task_attempts" not in boundary.tables_never_written(modules)


# --------------------------------------------------------------------------- #
# a kind of row, keyed by the table it lives in
# --------------------------------------------------------------------------- #

_ROW_LIFETIME = """
class M:
    def issueCommand(self, site):
        sqlI = "INSERT INTO ATLAS_PANDA.HARVESTER_COMMANDS "
        sqlI += "(command_id,status) VALUES (:command_id,'new') "
        self.cur.execute(sqlI + comment, varMap)

    def pickUpCommand(self, site):
        sqlS = "SELECT command_id FROM ATLAS_PANDA.HARVESTER_COMMANDS "
        sqlS += "WHERE status=:status "
        self.cur.execute(sqlS + comment, varMap)

    def dropCommand(self, site):
        sqlD = "DELETE FROM ATLAS_PANDA.harvester_commands WHERE command_id=:command_id "
        self.cur.execute(sqlD + comment, varMap)

    def readConfig(self, site):
        sqlC = "SELECT a.name FROM ATLAS_PANDA.HARVESTER_COMMANDS c,ATLAS_PANDAMETA.schedconfig a "
        sqlC += "WHERE c.site=a.name AND c.status=:status "
        self.cur.execute(sqlC + comment, varMap)
"""


def _entities_of(source: str):
    modules = [_module(source, "pandaserver/taskbuffer/db_proxy_mods/task_module.py")]
    attributor = attribution.SpecAttributor(
        progress.spec_attributes(modules), attribution.class_bases(modules)
    )
    attributor.learn_table_classes(modules)
    side = sqlwrite.read_side(modules, attributor)
    return {node.name: node for node in sqlwrite.entity_nodes(side.entities, MAP_ID, VERSION)}


def test_a_kind_of_row_exists_without_a_spec_class_to_name_it():
    """Keying entities on the spec class answered for seven of the forty-eight
    tables this corpus inserts into.

    The forty-one without one are where "something arrived" means anything:
    HARVESTER_COMMANDS, PRODSYS_COMM, SQL_QUEUE, async_requests,
    Job_Output_Report, users, SiteData, T_TASK, jobs_StatusLog.  A command
    reaching a harvester *is* a row appearing in a table, and the map could not
    say where that happens.
    """
    entities = _entities_of(_ROW_LIFETIME)

    assert "harvester_commands" in entities
    assert entities["harvester_commands"].spec_class is None


def test_the_three_writing_verbs_are_three_different_claims():
    """Creating, changing and removing a row are not one fact."""
    entity = _entities_of(_ROW_LIFETIME)["harvester_commands"]

    assert [o.split("::")[-1] for o in entity.created_by] == ["issueCommand"]
    assert [o.split("::")[-1] for o in entity.deleted_by] == ["dropCommand"]
    assert entity.updated_by == []
    assert sorted(o.split("::")[-1] for o in entity.read_by) == ["pickUpCommand", "readConfig"]


def test_one_table_spelled_two_ways_is_one_kind_of_row():
    """``HARVESTER_COMMANDS`` and ``harvester_commands`` are one table, so the
    row created in one and removed in the other has one lifetime."""
    entity = _entities_of(_ROW_LIFETIME)["harvester_commands"]

    assert entity.created_by and entity.deleted_by


def test_a_table_this_corpus_only_reads_is_a_boundary_and_not_a_kind_of_row():
    """The read-only tables already have a node, and it is not this one.

    ``tables_never_written`` finds them and ``extract_selection_gates`` makes
    each an unbound boundary -- a dependency on something outside.  Recording
    them here as well would put one fact in two node types, which is the shape
    ``log_files`` and ``opened_by`` already cost this map twice.  Measured on
    the installed corpus, the read-only set and ``tables_never_written`` are
    the same set.
    """
    entities = _entities_of(_ROW_LIFETIME)

    # Joined, never written here: a dependency, not a kind of row PanDA keeps.
    assert "schedconfig" not in entities


_SCOUT_SOURCE = '''
class Proxy:
    def get_scout_data(self, task_id):
        returnMap = {}
        returnMap["ramCount"] = ram
        returnMap["walltime"] = wall
        returnMap["newNG"] = ng
        return True, returnMap, extra

    def set_scout_data(self, task_id):
        ok, scoutData, extra = self.get_scout_data(task_id)
        varMap = {}
        varMap[":jediTaskID"] = task_id
        sqlTSD = f"UPDATE {schema}.JEDI_Tasks SET "
        for scoutKey, scoutVal in scoutData.items():
            if scoutKey in ["newNG"]:
                continue
            tmpScoutKey = f":{scoutKey}"
            varMap[tmpScoutKey] = scoutVal
            sqlTSD += f"{scoutKey}={tmpScoutKey},"
        sqlTSD = sqlTSD[:-1]
        sqlTSD += " WHERE jediTaskID=:jediTaskID "
        self.cur.execute(sqlTSD + comment, varMap)
'''


@contextlib.contextmanager
def _in_build(source: str, name: str, rel: str = "pandaserver/taskbuffer/db_proxy_mods/scout.py"):
    """*name* from *source*, read the way a build reads it: through the corpus's calls."""
    modules = [_module(source, rel)]
    func = next(
        node
        for node in ast.walk(modules[0].tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
    )
    with _with_calls(modules):
        yield func


def test_a_column_named_by_a_mapping_key_is_read_from_the_callee():
    """The scout's columns are stated in the method that built the mapping.

    ``setScoutJobData_JEDI`` appends ``f"{scoutKey}={tmpScoutKey},"`` per key,
    so reassembly saw ``"{}={},"`` and the statement named no column at all --
    which is why the map had the scout writing nothing of ``JediTaskSpec``.
    """
    with _in_build(_SCOUT_SOURCE, "set_scout_data") as func:
        statements = [w for text in _held(func, "sqlTSD") for w in sqltext.writes(text)]

    (statement,) = statements
    assert sorted(statement.columns) == ["ramCount", "walltime"]


def test_a_key_the_loop_passes_over_is_not_a_column():
    """``if scoutKey in ["newNG"]: continue`` -- the statement never writes it."""
    with _in_build(_SCOUT_SOURCE, "set_scout_data") as func:
        columns = {c for text in _held(func, "sqlTSD") for w in sqltext.writes(text) for c in w.columns}

    assert "newNG" not in columns


def test_a_skip_written_as_a_prefix_test_is_read_too():
    """``if not k.startswith("walltime") and not k.startswith("ram")`` is the
    other spelling, and reading only ``in [...]`` has the merge loop writing
    sixteen columns where it writes four."""
    source = _SCOUT_SOURCE.replace(
        '            if scoutKey in ["newNG"]:',
        '            if not scoutKey.startswith("wall"):',
    )
    with _in_build(source, "set_scout_data") as func:
        columns = {c for text in _held(func, "sqlTSD") for w in sqltext.writes(text) for c in w.columns}

    assert columns == {"walltime"}


def test_a_literal_before_the_key_stays_in_the_column_name():
    """``merge{mergeScoutKey}`` writes ``mergewalltime``, a different column."""
    source = _SCOUT_SOURCE.replace(
        '            sqlTSD += f"{scoutKey}={tmpScoutKey},"',
        '            sqlTSD += f"merge{scoutKey}={tmpScoutKey},"',
    )
    with _in_build(source, "set_scout_data") as func:
        columns = {c for text in _held(func, "sqlTSD") for w in sqltext.writes(text) for c in w.columns}

    assert columns == {"mergeramCount", "mergewalltime"}


def test_the_placeholder_the_loop_computes_is_resolved_with_the_column():
    """Both halves come from the same key, so both resolve or neither does.

    Left as a hole the value reads as an expression rather than a bind, and the
    walk cannot follow it to ``varMap[tmpScoutKey] = scoutVal``.
    """
    with _in_build(_SCOUT_SOURCE, "set_scout_data") as func:
        (statement,) = [w for text in _held(func, "sqlTSD") for w in sqltext.writes(text)]
        bound = [ast.unparse(b.value) for b in sql.bound_values(func, "varMap", ":ramCount")]

    assert statement.columns["ramCount"].kind == "bind"
    assert statement.columns["ramCount"].text == ":ramCount"
    assert bound == ["scoutVal"]


def test_the_mapping_is_read_from_the_method_the_call_lands_in():
    """Found by name in the file, the first ``get_scout_data`` the walk met was
    read -- here another class's, naming a column the statement never writes."""
    source = (
        "class Other:\n"
        "    def get_scout_data(self, task_id):\n"
        "        returnMap = {}\n"
        "        returnMap['cpuTime'] = t\n"
        "        return True, returnMap, extra\n"
    ) + _SCOUT_SOURCE
    with _in_build(source, "set_scout_data") as func:
        columns = {c for text in _held(func, "sqlTSD") for w in sqltext.writes(text) for c in w.columns}

    assert columns == {"ramCount", "walltime"}


def test_a_mapping_that_arrives_as_a_parameter_is_the_callers_to_state():
    """Five of the corpus's mapping loops are this shape; filling them in from
    one caller would make the statement say something false about the others."""
    source = (
        "def update(self, criteria):\n"
        "    sql = f'UPDATE {schema}.JEDI_Tasks SET '\n"
        "    for tmpKey, tmpVal in criteria.items():\n"
        "        sql += f'{tmpKey}=:{tmpKey},'\n"
        "    self.cur.execute(sql + comment, varMap)\n"
    )
    func = _func(source, "update")

    columns = {c for text in _held(func, "sql") for w in sqltext.writes(text) for c in w.columns}

    assert columns == set()
    assert sql.key_column_census(func) == sql.KeyColumnCensus(
        loops=1, resolved=0, from_parameter=1
    )


# ---------------------------------------------------------------------------
# The executions nothing was read out of, and who named a guarded write's rows
# ---------------------------------------------------------------------------

_SKIPPING = '''
class TaskModule:
    def queryTasksToBePending_JEDI(self, sql_query, params_map, reason):
        comment = " /* queryTasksToBePending_JEDI */"
        self.cur.execute(sql_query + comment, params_map)
        taskIDs = self.cur.fetchall()

    def getSiteInfo(self):
        sql = "SELECT siteid, cloud FROM ATLAS_PANDA.schedconfig"
        self.cur.execute(sql)

    def setSession(self):
        self.cur.execute("ALTER SESSION SET TIME_ZONE='UTC'")

    def inline(self):
        self.cur.execute("SELECT LAST_INSERT_ID()", {})

    def unbound(self):
        self.cur.execute(SQL_FROM_ELSEWHERE, {})

    def peekJob(self, table):
        sql = sql1_0 % (columns, table) + sql1_1
        self.cur.execute(sql + comment, varMap)
'''

_WRAPPER = '''
class WrappedCursor:
    def execute(self, sql, varDict=None):
        return self.cur.execute(sql, varDict)
'''


def _skips(source: str, name: str) -> list:
    tree = ast.parse(source)
    func = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    skipped: list = []
    runs = sql.executions(func, skipped)
    return runs, skipped


@pytest.mark.parametrize(
    "name, reason, detail",
    [
        ("queryTasksToBePending_JEDI", sql.SKIP_HANDED_IN, "sql_query"),
        ("unbound", sql.SKIP_UNBOUND, "SQL_FROM_ELSEWHERE"),
        ("peekJob", sql.SKIP_UNREAD_NAME, "sql"),
    ],
)
def test_an_execution_nothing_is_read_out_of_is_said_with_its_reason(name, reason, detail):
    """Each of these was a silent ``continue``: no node, no stop, no count."""
    runs, skipped = _skips(_SKIPPING, name)

    assert runs == []
    assert [(s.reason, s.detail) for s in skipped] == [(reason, detail)]


def test_a_wrapper_handing_on_its_callers_statement_is_said_as_theirs():
    """``WrappedCursor.execute`` runs what every caller wrote, and the callers
    are where it is read -- handing it on is not a gap, and is said so."""
    runs, skipped = _skips(_WRAPPER, "execute")

    assert runs == []
    assert [(s.reason, s.detail) for s in skipped] == [(sql.SKIP_CALLERS, "sql")]


_RUNNERS = '''
class Proxy:
    def getClobObj(self, sql, varMap):
        self.cur.execute(sql + comment, varMap)

    def executemanySQL(self, sql, varMaps):
        self.cur.executemany(sql, varMaps)

    def queryTasksToBePending_JEDI(self, sql_query, params_map):
        self.cur.execute(sql_query, params_map)
        self.cur.execute("UPDATE ATLAS_PANDA.JEDI_Tasks SET status=:s", {":s": "pending"})


class TaskBuffer:
    def executemanySQL(self, sql, varMaps):
        return self.proxy.executemanySQL(sql, varMaps)

    def getSiteInfo(self):
        sql = "SELECT siteid FROM ATLAS_PANDA.schedconfig"
        return self.proxy.getClobObj(sql, {})

    def clean(self, maps):
        self.executemanySQL("DELETE FROM ATLAS_PANDA.Metrics WHERE id=:id", maps)
'''


def test_a_function_running_its_own_parameter_as_the_statement_is_a_runner_read_at_its_callers(tmp_path):
    """``getClobObj(sql, varMap)`` runs ``self.cur.execute(sql, varMap)``: a call of it runs the statement its caller
    writes, and is read there, as ``execute`` is; ``TaskBuffer.executemanySQL`` hands its own on to the proxy's, a
    list of maps as ``executemany`` takes.  A reading with no corpus knows the cursor's and the task buffer's."""
    from bamboo.codemap.gitsource import blob_sha

    root = tmp_path / "pandaserver"
    root.mkdir()
    (root / "db.py").write_text(_RUNNERS)
    tree = ast.parse(_RUNNERS)
    pathcond.attach_parents(tree)
    module = SourceModule(
        package="pandaserver", rel_path="pandaserver/db.py", tree=tree, source=_RUNNERS, blob_sha=blob_sha(_RUNNERS)
    )
    functions = {
        f"{cls.name}.{func.name}": func
        for cls in tree.body
        if isinstance(cls, ast.ClassDef)
        for func in cls.body
        if isinstance(func, ast.FunctionDef)
    }

    assert set(sql.runners()) == {"execute", "executemany", "querySQL", "querySQLS"}
    with sql.across_calls(sqlwrite.corpus_calls([module], {"pandaserver": root})):
        runners = sql.runners()
        (site_info,) = sql.executions(functions["TaskBuffer.getSiteInfo"])
        (clean,) = sql.executions(functions["TaskBuffer.clean"])
        sql.executions(functions["Proxy.getClobObj"], handing := [])
        many = sqlwrite._runs_many(clean)

    assert {name: (one.many, one.own) for name, one in runners.items()} == {
        "execute": (False, False),
        "executemany": (True, False),
        "getClobObj": (False, False),
        "executemanySQL": (True, False),
        "queryTasksToBePending_JEDI": (False, True),
    }
    assert (site_info.sql, site_info.varmap) == ("SELECT siteid FROM ATLAS_PANDA.schedconfig", None)
    assert (clean.sql, clean.varmap, many) == ("DELETE FROM ATLAS_PANDA.Metrics WHERE id=:id", "maps", True)
    assert [(s.reason, s.line) for s in handing] == [(sql.SKIP_CALLERS, 4)]


def test_the_skipped_statements_become_stops_with_what_they_read():
    """A statement not read says its tables are unknown; one run with no bind map
    is read, and a statement about no rows -- ``ALTER SESSION`` -- is no stop at all."""
    tree = ast.parse(_SKIPPING)
    stops = {
        func.name: sqlwrite.statement_stops(func, "pandaserver/x.py")
        for func in ast.walk(tree)
        if isinstance(func, ast.FunctionDef)
    }

    (handed,) = stops["queryTasksToBePending_JEDI"]
    (unbound,) = stops["unbound"]
    assert (handed.reads, handed.reads_known) == ([], False)
    assert (unbound.reads, unbound.reads_known) == ([], False)
    assert stops["getSiteInfo"] == []
    assert stops["setSession"] == []


def test_the_answers_stops_are_the_builds_stops():
    """``statement_stops`` is what ``derive-strategy`` reads one function with,
    and ``extract`` what the build reads the corpus with: the read side of the
    two must be the same list, or the gate and the answer disagree."""
    source = _SKIPPING + '''
    def getJobs(self, where):
        sqlP = "SELECT PandaID FROM ATLAS_PANDA.jobsActive4 WHERE "
        sqlP += where
        self.cur.execute(sqlP + comment, varMap)
'''
    rel = "pandaserver/taskbuffer/db_proxy_mods/task_module.py"
    modules = [_module(_SPECS, "pandaserver/taskbuffer/Specs.py"), _module(source, rel)]
    attributor = attribution.SpecAttributor(
        progress.spec_attributes(modules), attribution.class_bases(modules)
    )
    attributor.learn_table_classes(modules)
    *_rest, stops = sqlwrite.extract(modules, MAP_ID, VERSION, attributor)

    tree = modules[1].tree
    again = [
        stop
        for func in ast.walk(tree)
        if isinstance(func, ast.FunctionDef)
        for stop in sqlwrite.statement_stops(func, rel)
    ]
    built = [s for s in stops if s.reason != sqlwrite.PRECONDITION_BIND_UNSETTLED]
    key = lambda s: (s.owner, s.statement, s.reason, s.detail)  # noqa: E731
    assert sorted(map(key, again)) == sorted(map(key, built))
    assert any(s.reason == sql.SKIP_HANDED_IN for s in built)


def test_a_statement_a_class_defined_in_a_function_runs_is_no_stop_of_that_function():
    """``datasetManager.main`` defines its threads' classes in its own body, and their ``run`` methods execute the
    statements.  ``main`` builds none of them, so read as its own calls each was a stop -- "no statement this
    function builds reaches this call" -- and a gap under ``main`` in the answer; ``run`` is where they are read."""
    source = (
        "def main(taskBuffer):\n"
        "    class CloserThr:\n"
        "        def run(self):\n"
        "            sql = 'UPDATE ATLAS_PANDA.Datasets SET status=:status WHERE vuid=:vuid'\n"
        "            varMap = {':status': 'closed', ':vuid': self.vuid}\n"
        "            self.proxy.cur.execute(sql, varMap)\n"
        "    CloserThr().run()\n"
    )
    functions = {func.name: func for func in ast.walk(ast.parse(source)) if isinstance(func, ast.FunctionDef)}

    skipped: list = []
    assert sql.executions(functions["main"], skipped) == []
    assert skipped == []
    assert sqlwrite.statement_stops(functions["main"], "pandaserver/daemons/scripts/datasetManager.py") == []
    assert [run.call.lineno for run in sql.executions(functions["run"])] == [6]


def test_a_writes_row_keys_are_the_where_terms_it_does_not_guard():
    (write,) = sqltext.writes(
        "UPDATE ATLAS_PANDA.jobsActive4 SET jobStatus=:newStatus "
        "WHERE PandaID=:PandaID AND jobStatus=:oldStatus AND NOT x=:y "
        "AND t.jediTaskID = :jediTaskID AND status IN (:a,:b)"
    )

    assert write.preconditions == {"jobStatus": "jobStatus=:oldStatus"}
    assert write.keys == {"PandaID": ":PandaID", "jediTaskID": ":jediTaskID"}
    assert write.key_lists == {}


def test_a_write_naming_its_rows_by_a_helpers_in_list_keeps_the_list_beside_its_keys():
    """``fileID IN ({names})`` from ``get_sql_IN_bind_variables`` reads as one bind (``sql.in_lists``); a list on a
    column the statement writes is still its guard, and a negated one names no row."""
    (write,) = sqltext.writes(
        "UPDATE ATLAS_PANDA.JEDI_Dataset_Contents SET status=:status WHERE jediTaskID=:taskID "
        "AND fileID IN (:file__in) AND status IN (:old__in) AND NOT datasetID IN (:ds__in)"
    )

    assert write.keys == {"jediTaskID": ":taskID"}
    assert write.key_lists == {"fileID": ":file__in"}
    assert write.preconditions == {"status": "status IN (:old__in)"}


def _keyed_corpus(tmp_path, files: dict[str, str]):
    """Write *files* under *tmp_path*: their modules, the corpus's calls over them and the reach record, as the build
    reads a guarded write's keys (:func:`sqlwrite.attach_key_sources`)."""
    from bamboo.codemap import reach
    from bamboo.codemap.panda import plugin

    modules = []
    for rel, text in files.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        modules.append(_module(text, rel))
    calls = sqlwrite.corpus_calls(modules, {top: tmp_path / top for top in {rel.split("/")[0] for rel in files}})
    with sql.across_calls(calls):
        record = reach.ReachGraph(reach.graph(calls, MAP_ID, VERSION, plugin._DatabaseLeaves()))
    return modules, calls, record


@pytest.mark.parametrize(
    "expression, expected",
    [
        ("job.PandaID", "jobs"),
        ("tid", "taskCloudMap"),
        ("row_id", ""),
        ("self.datasets[0].vuid", "self.datasets"),
        ("str(jediTaskID)", "jediTaskID"),
        # A method no reading places: what it makes of its argument is its own value (U1), and the method no field.
        ("self.getNext(jediTaskID)", ""),
        ("self.cur.rowcount", "self.cur"),
    ],
)
def test_a_value_is_filled_from_the_parameter_or_field_of_self_the_carry_brings_it_from(tmp_path, expression, expected):
    """Through a local or a loop; a fetch from the cursor is the database's, not the field's; what a method no reading
    places makes of what it is handed is not, and the method is no field."""
    rel = "pandaserver/taskbuffer/proxy.py"
    source = (
        "class Proxy:\n"
        "    def f(self, jobs, taskCloudMap, jediTaskID):\n"
        "        for job in jobs:\n"
        "            pass\n"
        "        for tid, cloud in taskCloudMap.items():\n"
        "            pass\n"
        "        rows = self.cur.fetchall()\n"
        "        for row_id, _x in rows:\n"
        "            pass\n"
        f"        value = {expression}\n"
    )
    modules, calls, record = _keyed_corpus(tmp_path, {rel: source})
    func = next(node for node in ast.walk(modules[0].tree) if isinstance(node, ast.FunctionDef) and node.name == "f")
    assign = func.body[-1]
    with sql.across_calls(calls):
        filled = sqlwrite._Filled(sqlwrite._Reading(calls, set(), record), rel, func)
        handed = filled.handed(assign.value, assign)

    assert (handed[0] if handed else "") == expected


_KEYED = '''
class TaskModule:
    def setCloudToTasks_JEDI(self, taskCloudMap):
        sqlU = "UPDATE ATLAS_PANDA.JEDI_Tasks SET status=:newStatus WHERE jediTaskID=:jediTaskID AND status=:oldStatus "
        for jediTaskID, cloud in taskCloudMap.items():
            varMap = {}
            varMap[":jediTaskID"] = jediTaskID
            varMap[":newStatus"] = "ready"
            varMap[":oldStatus"] = "assigning"
            self.cur.execute(sqlU + comment, varMap)

    def rescuePickedFiles_JEDI(self, vo):
        sqlT = "SELECT jediTaskID FROM ATLAS_PANDA.JEDI_Tasks WHERE vo=:vo "
        self.cur.execute(sqlT + comment, {":vo": vo})
        resList = self.cur.fetchall()
        sqlU = "UPDATE ATLAS_PANDA.JEDI_Tasks SET status=:newStatus WHERE jediTaskID=:jediTaskID AND status=:oldStatus "
        for (jediTaskID,) in resList:
            varMap = {}
            varMap[":jediTaskID"] = jediTaskID
            varMap[":newStatus"] = "ready"
            varMap[":oldStatus"] = "picked"
            self.cur.execute(sqlU + comment, varMap)

    def lockJobOutputReport(self, panda_id, attempt_nr, pid):
        sqlGL = "SELECT PandaID, attemptNr FROM ATLAS_PANDA.Job_Output_Report WHERE lockedBy=:lockedBy "
        self.cur.execute(sqlGL + comment, {":lockedBy": pid})
        resGL = self.cur.fetchall()
        sqlUL = "UPDATE ATLAS_PANDA.Job_Output_Report SET lockedBy=:lockedBy WHERE PandaID=:PandaID AND attemptNr=:attemptNr AND lockedBy IS NULL "
        for panda_id, attempt_nr in resGL:
            varMap = {}
            varMap[":PandaID"] = panda_id
            varMap[":attemptNr"] = attempt_nr
            varMap[":lockedBy"] = pid
            self.cur.execute(sqlUL + comment, varMap)


class DatasetModule:
    def main(self):
        class Closer:
            def __init__(self, datasets):
                self.datasets = datasets

            def run(self):
                for name in self.datasets:
                    pass

        sql = "SELECT name FROM ATLAS_PANDA.Datasets WHERE status=:status "
        self.cur.execute(sql + comment, {":status": "doing"})
        resReset = self.cur.fetchall()
        sqlU = "UPDATE ATLAS_PANDA.Datasets SET status=:newStatus WHERE name=:name AND status=:oldStatus "
        for (name,) in resReset:
            varMap = {}
            varMap[":name"] = name
            varMap[":oldStatus"] = "doing"
            varMap[":newStatus"] = "running"
            self.cur.execute(sqlU + comment, varMap)
'''


def test_a_guarded_write_says_whether_its_caller_named_the_rows(tmp_path):
    """The shapes the "nothing here reads these rows" answer held: a parameter's rows, the function's own rows, and
    two of its own rows that read as handed when a key's name was followed by its spelling -- a parameter's name the
    rows rebind, and a loop of a class the function defines."""
    rel = "pandaserver/taskbuffer/db_proxy_mods/task_module.py"
    modules, calls, record = _keyed_corpus(tmp_path, {"pandaserver/taskbuffer/Specs.py": _SPECS, rel: _KEYED})
    attributor = attribution.SpecAttributor(
        progress.spec_attributes(modules), attribution.class_bases(modules)
    )
    attributor.learn_table_classes(modules)
    keys: list[RowKeySource] = []
    with sql.across_calls(calls):
        _subjects, junctions, *_rest = sqlwrite.extract(modules, MAP_ID, VERSION, attributor, row_keys=keys)
        assert all(source.handed_by == "" for source in keys)
        assert sqlwrite.attach_key_sources(keys, junctions, calls, record) == len(keys) == 4

    by = {source.owner.rpartition("::")[2]: source for source in keys}
    assert by["setCloudToTasks_JEDI"].handed_by == "taskCloudMap"
    assert by["setCloudToTasks_JEDI"].keys == ["jediTaskID=:jediTaskID"]
    assert by["setCloudToTasks_JEDI"].columns == ["status"]
    assert by["rescuePickedFiles_JEDI"].handed_by == ""
    assert by["lockJobOutputReport"].keys == ["PandaID=:PandaID", "attemptNr=:attemptNr"]
    assert by["lockJobOutputReport"].handed_by == ""
    assert by["main"].keys == ["name=:name"]
    assert by["main"].handed_by == ""


_FLUSH_MORE = '''
class JobModule:
    def updateJob(self, job: JobSpec, inJobsDefined):
        varMap = job.valuesMap(onlyChanged=True)
        varMap[":PandaID"] = job.PandaID
        sql1 = f"UPDATE ATLAS_PANDA.jobsActive4 SET {job.bindUpdateChangesExpression()} WHERE PandaID=:PandaID "
        self.cur.execute(sql1 + comment, varMap)

    def registerTaskInOneShot_JEDI(self, jediTaskID, taskSpec: JediTaskSpec, oldTaskStatus):
        varMap = taskSpec.valuesMap(useSeq=False, onlyChanged=True)
        varMap[":jediTaskID"] = jediTaskID
        varMap[":preStatus"] = oldTaskStatus
        sql = f"UPDATE {panda_config.schemaJEDI}.JEDI_Tasks SET {taskSpec.bindUpdateChangesExpression()} WHERE "
        sql += "jediTaskID=:jediTaskID AND status=:preStatus "
        self.cur.execute(sql + comment, varMap)

    def updateTaskOld_JEDI(self, taskSpec: JediTaskSpec, jediTaskID, oldStatus=None):
        varMap = taskSpec.valuesMap(useSeq=False, onlyChanged=True)
        varMap[":jediTaskID"] = jediTaskID
        sql = f"UPDATE {panda_config.schemaJEDI}.JEDI_Tasks SET {taskSpec.bindUpdateChangesExpression()} "
        sql += "WHERE jediTaskID=:jediTaskID "
        if oldStatus is not None:
            names, old_map = get_sql_IN_bind_variables(oldStatus, prefix=":old_", value_as_suffix=True)
            sql += f"AND status IN ({names}) "
            varMap.update(old_map)
        self.cur.execute(sql + comment, varMap)
'''


def _flushed(tmp_path, caller: str, rel: str = "pandajedi/jediorder/Knight.py", more: dict | None = None):
    """The build's steps a flush call's records come out of, over *caller* and the proxies (and *more* modules): the
    attribute slice's arms, the SQL slice's records of the methods' own statements, the calls' (flush.attach), then
    their sources and writes and the objects no write persists (sqlwrite.attach_key_sources).
    ``(KeyFroms, RowKeySources)``."""
    modules, calls, record = _keyed_corpus(
        tmp_path,
        {
            "pandaserver/taskbuffer/Specs.py": _SPECS,
            "pandaserver/taskbuffer/db_proxy_mods/task_module.py": _FLUSH_PROXY,
            "pandaserver/taskbuffer/db_proxy_mods/job_module.py": _FLUSH_MORE,
            **(more or {}),
            rel: caller,
        },
    )
    _s, arms, *_rest = progress.extract(modules, MAP_ID, VERSION)
    attributor = attribution.SpecAttributor(progress.spec_attributes(modules), attribution.class_bases(modules))
    attributor.learn_element_types(modules)
    attributor.learn_self_attributes(modules)
    attributor.learn_table_classes(modules)
    froms: list[KeyFrom] = []
    keys: list[RowKeySource] = []
    with sql.across_calls(calls), sqltext.with_schema(None, {}, {}):
        _subjects, junctions, *_rest = sqlwrite.extract(modules, MAP_ID, VERSION, attributor, key_froms=froms)
        flush.attach(arms, modules, attributor, row_keys=keys, key_froms=froms)
        sqlwrite.attach_key_sources(
            keys, junctions + arms, calls, record, key_froms=froms, modules=modules, attributor=attributor
        )
    return [one for one in froms if one.owner.startswith(rel)], [one for one in keys if one.owner.startswith(rel)]


def test_a_flush_call_names_its_row_as_the_method_fills_it(tmp_path):
    """A flush call writes the row the method's statement names, filled as the method fills it, through the call:
    ``updateTask_JEDI`` names it by the criteria it is handed, ``updateJob`` by its spec's ``PandaID``, and
    ``registerTaskInOneShot_JEDI`` by its parameter ``jediTaskID``, not the spec's -- and a term on a column the
    caller writes is its compare-and-set, no name.  Each writes what the caller assigns on the spec before the call."""
    caller = '''
class Agent:
    def run(self, taskSpec: JediTaskSpec, jobSpec: JobSpec, jediTaskID, taskStatus):
        taskSpec.status = "running"
        self.taskBufferIF.updateTask_JEDI(taskSpec, {"jediTaskID": jediTaskID}, oldStatus=[taskStatus])
        jobSpec.jobStatus = "failed"
        self.taskBufferIF.updateJob(jobSpec, True)
        taskSpec.status = "defined"
        self.taskBufferIF.registerTaskInOneShot_JEDI(jediTaskID, taskSpec, "ready")
'''
    froms, keys = _flushed(tmp_path, caller)

    lines = caller.split("\n")
    running, failed, defined = (
        next(n for n, line in enumerate(lines, 1) if f'= "{value}"' in line)
        for value in ("running", "failed", "defined")
    )
    read = sorted(
        (one.statement - running, one.table, one.term, one.object, [(s.kind, s.name) for s in one.sources], one.writes)
        for one in froms
    )
    assert read == [
        (1, "JEDI_Tasks", 'jediTaskID=criteria["jediTaskID"]', "taskSpec", [("parameter", "jediTaskID")],
         {"JediTaskSpec.status": [running]}),
        (3, "jobsActive4", "PandaID=:PandaID", "jobSpec", [("parameter", "jobSpec")], {"JobSpec.jobStatus": [failed]}),
        (5, "JEDI_Tasks", "jediTaskID=:jediTaskID", "taskSpec", [("parameter", "jediTaskID")],
         {"JediTaskSpec.status": [defined]}),
    ]
    assert sorted((one.statement - running, one.keys, one.handed_by) for one in keys) == [
        (1, ['jediTaskID=criteria["jediTaskID"]'], "jediTaskID"),
        (5, ["jediTaskID=:jediTaskID"], "jediTaskID"),
    ]


def test_a_guarded_flush_is_handed_its_row_by_the_criteria_that_name_it(tmp_path):
    """``updateTask_JEDI`` takes the spec's own ``jediTaskID`` out of the write and names the row by the criteria
    it is handed, so who handed the row is who filled the criteria: the spec's parameter, a field of ``self`` for a
    fresh spec holding a message at another field (``TaskRefinerThread.runImpl``'s ``tmpTaskSpec``), and nobody for a
    literal -- read by the spec, that one was "taskSpec"."""
    caller = '''
class TaskRefiner:
    def runImpl(self, taskSpec: JediTaskSpec, taskStatus, errorText):
        taskSpec.status = "tobroken"
        self.taskBufferIF.updateTask_JEDI(taskSpec, {"jediTaskID": taskSpec.jediTaskID}, oldStatus=[taskStatus])
        tmpTaskSpec = JediTaskSpec()
        tmpTaskSpec.oldStatus = errorText
        tmpTaskSpec.status = "broken"
        self.taskBufferIF.updateTask_JEDI(tmpTaskSpec, {"jediTaskID": self.taskID}, oldStatus=[taskStatus])
        taskSpec.status = "failed"
        self.taskBufferIF.updateTask_JEDI(taskSpec, {"jediTaskID": 1}, oldStatus=[taskStatus])
'''
    _froms, keys = _flushed(tmp_path, caller)

    assert [(one.columns, one.handed_by) for one in sorted(keys, key=lambda one: one.statement)] == [
        (["status"], "taskSpec"),
        (["status"], "self.taskID"),
        (["status"], ""),
    ]


def test_a_flush_guard_is_what_each_call_hands_it(tmp_path):
    """A flush's guard on a parameter of the method is settled at each call, by what that call hands it:
    ``JobGenerator.runImpl``'s list is its guard, and another caller's is not; a value read off the row stays the
    bind (a compare-and-set); a status tested above the call (``if taskStatus in ["registered", "staged"]`` before
    ``registerTaskInOneShot_JEDI``) is what the test says.  Said only where every guarded call of the function
    settles alike, and on the branches the call writes: ``TaskRefiner.runImpl``'s two calls are alternatives, and a
    branch's guards a conjunction."""
    caller = '''
class Agent:
    def generate(self, taskSpec: JediTaskSpec, jediTaskID):
        taskSpec.status = "running"
        self.taskBufferIF.updateTaskOld_JEDI(taskSpec, jediTaskID, oldStatus=["ready", "scouting"])

    def refine(self, taskSpec: JediTaskSpec, jediTaskID, taskStatus):
        taskSpec.status = "ready"
        self.taskBufferIF.updateTaskOld_JEDI(taskSpec, jediTaskID, oldStatus=[taskStatus])

    def register(self, taskSpec: JediTaskSpec, jediTaskID, taskStatus):
        if taskStatus in ["registered", "staged"]:
            taskSpec.status = "defined"
            self.taskBufferIF.registerTaskInOneShot_JEDI(jediTaskID, taskSpec, taskStatus)

    def both(self, taskSpec: JediTaskSpec, jediTaskID, taskStatus):
        if taskStatus in ["registered", "staged"]:
            taskSpec.status = "defined"
            self.taskBufferIF.registerTaskInOneShot_JEDI(jediTaskID, taskSpec, taskStatus)
        else:
            taskSpec.status = "broken"
            self.taskBufferIF.updateTaskOld_JEDI(taskSpec, jediTaskID, oldStatus=[taskStatus])

    def partly(self, taskSpec: JediTaskSpec, jediTaskID):
        taskSpec.status = "running"
        self.taskBufferIF.updateTaskOld_JEDI(taskSpec, jediTaskID, oldStatus=["ready"])
        taskSpec.status = "done"
'''
    rel = "pandajedi/jediorder/Agent.py"
    modules, calls, _record = _keyed_corpus(
        tmp_path,
        {
            "pandaserver/taskbuffer/Specs.py": _SPECS,
            "pandaserver/taskbuffer/db_proxy_mods/task_module.py": _FLUSH_PROXY,
            "pandaserver/taskbuffer/db_proxy_mods/job_module.py": _FLUSH_MORE,
            rel: caller,
        },
    )
    _s, arms, *_rest = progress.extract(modules, MAP_ID, VERSION)
    attributor = attribution.SpecAttributor(progress.spec_attributes(modules), attribution.class_bases(modules))
    attributor.learn_element_types(modules)
    attributor.learn_self_attributes(modules)
    with sql.across_calls(calls):
        flush.attach(arms, modules, attributor)

    guards = {
        junction.owner.rpartition("::")[2]: {b.outcome: b.row_precondition for b in junction.branches}
        for junction in arms
        if junction.owner.startswith(rel) and junction.subject == "JediTaskSpec.status"
    }
    assert guards == {
        "generate": {"running": ["status IN (ready,scouting)"]},
        "refine": {"ready": ["status IN (:old___in)"]},
        "register": {"defined": ["status IN (registered,staged)"]},
        # Two calls settling apart: each call's guard is the other's branch's too, so neither is said settled.
        "both": {
            "defined": ["status IN (:old___in)", "status=:preStatus"],
            "broken": ["status IN (:old___in)", "status=:preStatus"],
        },
        # The status set after the call is no write of it: the bind, as the method has it.
        "partly": {"running": ["status=ready"], "done": ["status IN (:old___in)"]},
    }


_UNSAVED = '''
def updateJobs(jobs):
    return len(jobs)


class Agent:
    def fail(self):
        job = JobSpec()
        job.jobStatus = "failed"
        updateJobs([job])

    def build(self):
        fileSpec = JediFileSpec()
        fileSpec.status = "ready"
        return fileSpec

    def note(self):
        job = JobSpec()
        job.jobStatus = "closed"
        tmpLog.debug(f"closed {job.jobStatus}")

    def bind(self):
        job = JobSpec()
        job.jobStatus = "holding"
        self.cur.execute("UPDATE ATLAS_PANDA.jobsActive4 SET jobStatus=:s " + comment, {":s": job.jobStatus})

    def mark(self, job: JobSpec):
        job.jobStatus = "throttled"
        updateJobs([job])
'''


def test_an_object_no_write_persists_says_what_its_fields_reach(tmp_path):
    """An arm no write here persists has a record of no term saying what its field reaches next -- a call of the
    corpus it is handed to (and where it lands), a statement it is bound in, a return, the object the function was
    handed (the caller holds it after) -- or that nothing the map reads does: with none, a reader would take the arm
    as known not to be on a statement's rows.  One record per use: a parameter's object handed on is both."""
    rel = "pandajedi/jediorder/Agent.py"
    froms, _keys = _flushed(tmp_path, _UNSAVED, rel)

    lines = _UNSAVED.split("\n")
    failed, ready, closed, holding, throttled = (
        next(n for n, line in enumerate(lines, 1) if f'= "{value}"' in line)
        for value in ("failed", "ready", "closed", "holding", "throttled")
    )
    read = sorted(
        (one.owner.rpartition("::")[2], one.object, one.term, one.unread, one.handed_to, one.sources, one.writes)
        for one in froms
    )
    assert read == [
        ("bind", "job", "", KEY_BOUND, [], [], {"JobSpec.jobStatus": [holding]}),
        ("build", "fileSpec", "", KEY_RETURNED, [], [], {"JediFileSpec.status": [ready]}),
        ("fail", "job", "", KEY_HANDED, [f"{rel}::updateJobs"], [], {"JobSpec.jobStatus": [failed]}),
        ("mark", "job", "", KEY_HANDED, [f"{rel}::updateJobs"], [], {"JobSpec.jobStatus": [throttled]}),
        ("mark", "job", "", KEY_KEPT, [], [], {"JobSpec.jobStatus": [throttled]}),
        ("note", "job", "", KEY_NO_USE, [], [], {"JobSpec.jobStatus": [closed]}),
    ]



_LOOPED_UNSAVED = '''
def updateJobs(jobs):
    return len(jobs)


class Agent:
    def activate(self):
        job = JobSpec()
        for fileSpec in job.Files:
            fileSpec.status = "ready"
        updateJobs([job])

    def build(self):
        job = JobSpec()
        for fileSpec in job.Files:
            fileSpec.status = "built"
        return job

    def finalize(self, job: JobSpec):
        for fileSpec in job.Files:
            fileSpec.status = "merging"

    def fail(self, jobs: list[JobSpec]):
        for job in jobs:
            for fileSpec in job.Files:
                fileSpec.status = "failed"

    def note(self):
        job = JobSpec()
        for fileSpec in job.Files:
            fileSpec.status = "noted"
'''


def test_a_field_set_on_a_loops_element_reaches_where_its_container_goes(tmp_path):
    """The element a loop takes is its container's: a field set on it goes where what holds the container goes --
    ``datasetManager.run``'s ``for tmpFile in tmpJob.Files: tmpFile.status = "ready"`` before ``tmpJob`` is handed
    to ``activateJobs``, ``doGenerateBuild``'s ``jobSpec`` returned, a parameter's files, a parameter's jobs' files
    (on up through the outer loop) -- and nowhere only where nothing takes the container on."""
    rel = "pandajedi/jediorder/Agent.py"
    froms, _keys = _flushed(tmp_path, _LOOPED_UNSAVED, rel, more={"pandaserver/taskbuffer/JobSpec.py": _ADDERS})

    lines = _LOOPED_UNSAVED.split("\n")
    ready, built, merging, failed, noted = (
        next(n for n, line in enumerate(lines, 1) if f'= "{value}"' in line)
        for value in ("ready", "built", "merging", "failed", "noted")
    )
    read = sorted(
        (one.owner.rpartition("::")[2], one.object, one.unread, one.handed_to, one.writes)
        for one in froms
        if not one.term
    )
    assert read == [
        ("activate", "fileSpec", KEY_HANDED, [f"{rel}::updateJobs"], {"FileSpec.status": [ready]}),
        ("build", "fileSpec", KEY_RETURNED, [], {"FileSpec.status": [built]}),
        ("fail", "fileSpec", KEY_KEPT, [], {"FileSpec.status": [failed]}),
        ("finalize", "fileSpec", KEY_KEPT, [], {"FileSpec.status": [merging]}),
        ("note", "fileSpec", KEY_NO_USE, [], {"FileSpec.status": [noted]}),
    ]


def test_a_plus_equals_arm_says_what_its_field_reaches(tmp_path):
    """A ``+=`` that adds to a value and leaves is an arm of its own (the attribute slice's runtime branch), and like
    an assignment's it has a record of what its field reaches next: ``job`` was handed in, so the caller holds it."""
    rel = "pandajedi/jediorder/Agent.py"
    source = (
        "class Agent:\n"
        "    def tag(self, job: JobSpec):\n"
        "        job.jobStatus = 'a'\n"
        "        job.jobStatus += 'b'\n"
    )
    froms, _keys = _flushed(tmp_path, source, rel)

    read = sorted(
        (one.owner.rpartition("::")[2], one.object, one.term, one.unread, one.handed_to, one.sources, one.writes)
        for one in froms
    )
    assert read == [("tag", "job", "", KEY_KEPT, [], [], {"JobSpec.jobStatus": [4]})]

_KEY_FROM = '''
class TaskModule:
    def throttleTasks(self, vo):
        sqlS = "SELECT jediTaskID FROM ATLAS_PANDA.JEDI_Tasks WHERE vo=:vo AND status=:status "
        self.cur.execute(sqlS + comment, {":vo": vo, ":status": "running"})
        resS = self.cur.fetchall()
        sqlT = "UPDATE ATLAS_PANDA.JEDI_Tasks SET status=:newStatus WHERE jediTaskID=:jediTaskID AND status=:oldStatus "
        for (jediTaskID,) in resS:
            varMap = {}
            varMap[":jediTaskID"] = jediTaskID
            varMap[":newStatus"] = "throttled"
            varMap[":oldStatus"] = "running"
            self.cur.execute(sqlT + comment, varMap)

    def setMaxAttempt(self, taskID, fileIDs):
        file_var_names_str, file_var_map = get_sql_IN_bind_variables(fileIDs, prefix=":file")
        varMap = {}
        varMap[":taskID"] = taskID
        varMap[":status"] = "ready"
        varMap.update(file_var_map)
        sql = f"UPDATE ATLAS_PANDA.JEDI_Dataset_Contents SET status=:status WHERE jediTaskID=:taskID AND fileID IN ({file_var_names_str}) "
        self.cur.execute(sql + comment, varMap)

    def resetTask(self, status):
        varMap = {}
        varMap[":status"] = status
        varMap[":jediTaskID"] = 123
        sql = "UPDATE ATLAS_PANDA.JEDI_Tasks SET status=:status WHERE jediTaskID=:jediTaskID "
        self.cur.execute(sql + comment, varMap)

    def lostKey(self, status):
        varMap = {}
        varMap[":status"] = status
        sql = "UPDATE ATLAS_PANDA.JEDI_Tasks SET status=:status WHERE jediTaskID=:jediTaskID "
        self.cur.execute(sql + comment, varMap)
'''


def test_a_writes_row_naming_terms_say_where_their_binds_were_filled_from(tmp_path):
    """A term each: its run's line, the subjects it writes with their lines, and its sources -- the rows of the
    function's own statement, a parameter, an IN list's parameter, a literal -- or why none was read."""
    rel = "pandaserver/taskbuffer/db_proxy_mods/task_module.py"
    modules, calls, record = _keyed_corpus(tmp_path, {"pandaserver/taskbuffer/Specs.py": _SPECS, rel: _KEY_FROM})
    attributor = attribution.SpecAttributor(
        progress.spec_attributes(modules), attribution.class_bases(modules)
    )
    attributor.learn_table_classes(modules)
    keys: list[RowKeySource] = []
    froms: list[KeyFrom] = []
    with sql.across_calls(calls):
        _subjects, junctions, *_rest = sqlwrite.extract(
            modules, MAP_ID, VERSION, attributor, row_keys=keys, key_froms=froms
        )
        sites = sqlwrite.read_site_nodes(sqlwrite.read_side(modules, attributor).statements, MAP_ID, VERSION)
        assert all(not one.sources for one in froms)
        read = sqlwrite.attach_key_sources(keys, junctions, calls, record, key_froms=froms, read_sites=sites)
    assert read == len(froms) + len(keys)

    by = {(one.owner.rpartition("::")[2], one.term): one for one in froms}
    assert sorted(by) == [
        ("lostKey", "jediTaskID=:jediTaskID"),
        ("resetTask", "jediTaskID=:jediTaskID"),
        ("setMaxAttempt", "fileID IN (:file__in)"),
        ("setMaxAttempt", "jediTaskID=:taskID"),
        ("throttleTasks", "jediTaskID=:jediTaskID"),
    ]
    lines = _KEY_FROM.split("\n")

    def at(text: str) -> int:
        return next(number for number, line in enumerate(lines, 1) if text in line)

    throttle = by[("throttleTasks", "jediTaskID=:jediTaskID")]
    (written,) = [j for j in junctions if j.owner == throttle.owner]
    assert throttle.statement == at("self.cur.execute(sqlT")
    assert throttle.writes == {written.subject: [at('":newStatus"] = "throttled"')]}
    (site,) = [s for s in sites if s.owner == throttle.owner]
    assert site.anchor.line_start == at("self.cur.execute(sqlS")
    assert [one.model_dump() for one in throttle.sources] == [
        {
            "kind": "rows",
            "line": at('":jediTaskID"] = jediTaskID'),
            "name": site.name,
            "site_owner": site.owner,
            "site_line": site.anchor.line_start,
            "paths": ["()"],
            "column": "jedi_tasks.jeditaskid",
            "attr": "",
            "column_unread": "",
            "same_rows": False,
            "never_at": [],
            "via": [],
            "reaches": [],
            "reaches_open": [],
        }
    ]
    assert throttle.unread == ""
    assert [(s.kind, s.name) for s in by[("setMaxAttempt", "jediTaskID=:taskID")].sources] == [("parameter", "taskID")]
    assert [(s.kind, s.name) for s in by[("setMaxAttempt", "fileID IN (:file__in)")].sources] == [
        ("parameter", "fileIDs")
    ]
    assert [(s.kind, s.name) for s in by[("resetTask", "jediTaskID=:jediTaskID")].sources] == [("constant", "123")]
    lost = by[("lostKey", "jediTaskID=:jediTaskID")]
    assert (lost.sources, lost.unread) == ([], "no bind found")


_TWO_PLACES = '''
class TaskModule:
    def restartTasks(self, vo):
        sqlD = "SELECT jediTaskID FROM ATLAS_PANDA.JEDI_Tasks WHERE vo=:vo AND status=:status "
        sqlR = "SELECT jediTaskID FROM ATLAS_PANDA.JEDI_Tasks WHERE vo=:vo AND status=:status AND splitRule IS NULL "
        sqlT = "UPDATE ATLAS_PANDA.JEDI_Tasks SET status=:newStatus WHERE jediTaskID=:jediTaskID AND status=:oldStatus "
        self.cur.execute(sqlD + comment, {":vo": vo, ":status": "defined"})
        resD = self.cur.fetchall()
        for (jediTaskID,) in resD:
            varMap = {}
            varMap[":jediTaskID"] = jediTaskID  # KEY D
            varMap[":newStatus"] = "ready"  # NEW D
            varMap[":oldStatus"] = "defined"
            self.cur.execute(sqlT + comment, varMap)  # RUN D
        self.cur.execute(sqlR + comment, {":vo": vo, ":status": "ready"})
        resR = self.cur.fetchall()
        for (jediTaskID,) in resR:
            varMap = {}
            varMap[":jediTaskID"] = jediTaskID  # KEY R
            varMap[":newStatus"] = "defined"  # NEW R
            varMap[":oldStatus"] = "ready"
            self.cur.execute(sqlT + comment, varMap)  # RUN R
'''


def test_a_text_run_from_two_places_with_its_map_built_anew_is_a_key_record_for_each(tmp_path):
    """restartTasksForContentsUpdate_JEDI's ``sqlTD`` at 1855 and 1914: one text and one map name, so one run
    (:attr:`sql.Execution.calls`), but each place keys its write by another statement's rows -- a record each, its
    source and its write the binds reaching that place."""
    rel = "pandaserver/taskbuffer/db_proxy_mods/task_module.py"
    modules, calls, record = _keyed_corpus(tmp_path, {"pandaserver/taskbuffer/Specs.py": _SPECS, rel: _TWO_PLACES})
    attributor = attribution.SpecAttributor(
        progress.spec_attributes(modules), attribution.class_bases(modules)
    )
    attributor.learn_table_classes(modules)
    froms: list[KeyFrom] = []
    with sql.across_calls(calls):
        _subjects, junctions, *_rest = sqlwrite.extract(modules, MAP_ID, VERSION, attributor, key_froms=froms)
        sites = sqlwrite.read_site_nodes(sqlwrite.read_side(modules, attributor).statements, MAP_ID, VERSION)
        sqlwrite.attach_key_sources([], junctions, calls, record, key_froms=froms, read_sites=sites)
    lines = _TWO_PLACES.split("\n")

    def at(text: str) -> int:
        return next(number for number, line in enumerate(lines, 1) if text in line)

    (written,) = [j for j in junctions if j.owner.endswith("::restartTasks")]
    site = {s.anchor.line_start: s.name for s in sites if s.owner.endswith("::restartTasks")}
    found = sorted(
        (one.statement, one.writes, [(s.kind, s.name, s.line) for s in one.sources])
        for one in froms
        if one.term == "jediTaskID=:jediTaskID"
    )
    assert found == [
        (
            at("# RUN D"),
            {written.subject: [at("# NEW D")]},
            [("rows", site[at("execute(sqlD")], at("# KEY D"))],
        ),
        (
            at("# RUN R"),
            {written.subject: [at("# NEW R")]},
            [("rows", site[at("execute(sqlR")], at("# KEY R"))],
        ),
    ]


_READ_TWICE = '''
class TaskModule:
    def countTasks(self, vo):
        sqlC = "SELECT COUNT(*) FROM ATLAS_PANDA.JEDI_Tasks WHERE vo=:vo AND status=:status "
        varMap = {}
        varMap[":vo"] = vo
        varMap[":status"] = "defined"
        self.cur.execute(sqlC + comment, varMap)  # RUN D
        (nDefined,) = self.cur.fetchone()
        varMap = {}
        varMap[":vo"] = vo
        varMap[":status"] = "ready"
        self.cur.execute(sqlC + comment, varMap)  # RUN R
        (nReady,) = self.cur.fetchone()
        return nDefined, nReady
'''


def test_a_select_run_from_two_places_with_its_map_built_anew_is_a_read_site_for_each(tmp_path):
    """insertnRunningInSiteData's ``sqlCh`` at 1713 and 1751: one text and one map name, so one run
    (:attr:`sql.Execution.calls`), read once at every call -- one site asking both values.  Each place is a
    statement of its own, asking what the binds reaching it hold, and so is its key record."""
    rel = "pandaserver/taskbuffer/db_proxy_mods/task_module.py"
    modules, calls, record = _keyed_corpus(tmp_path, {"pandaserver/taskbuffer/Specs.py": _SPECS, rel: _READ_TWICE})
    attributor = attribution.SpecAttributor(
        progress.spec_attributes(modules), attribution.class_bases(modules)
    )
    attributor.learn_table_classes(modules)
    froms: list[KeyFrom] = []
    with sql.across_calls(calls):
        _subjects, junctions, *_rest = sqlwrite.extract(modules, MAP_ID, VERSION, attributor, key_froms=froms)
        sites = sqlwrite.read_site_nodes(sqlwrite.read_side(modules, attributor).statements, MAP_ID, VERSION)
        sqlwrite.attach_key_sources([], junctions, calls, record, key_froms=froms, read_sites=sites, modules=modules)
    lines = _READ_TWICE.split("\n")

    def at(text: str) -> int:
        return next(number for number, line in enumerate(lines, 1) if text in line)

    asked = {s.anchor.line_start: s.selects for s in sites if s.owner.endswith("::countTasks")}
    assert asked == {at("# RUN D"): ["JEDI_Tasks.status=defined"], at("# RUN R"): ["JEDI_Tasks.status=ready"]}
    keyed = sorted(
        (one.statement, [(s.kind, s.name) for s in one.sources])
        for one in froms
        if one.owner.endswith("::countTasks") and one.term == "status=:status"
    )
    assert keyed == [(at("# RUN D"), [("constant", "'defined'")]), (at("# RUN R"), [("constant", "'ready'")])]


def test_a_run_read_at_its_own_call_is_one_object_at_every_ask():
    """The readers keep what they read by the run (``_text_run``, ``bound_sites``), so ``_read_runs`` hands the
    same runs each time it is asked; a new run at every ask read them all again."""
    func = ast.parse(_READ_TWICE).body[0].body[0]
    first, again = sqlwrite._read_runs(func), sqlwrite._read_runs(func)
    assert [run.calls for run in first] == [(run.call,) for run in first] and len(first) == 2
    assert all(one is other for one, other in zip(first, again, strict=True))


_BATCHED = '''
from pandacommon.pandautils.PandaUtils import batched


class TaskModule:
    def updateStaged(self, task_dict, chunk_size=500):
        sqlU = "UPDATE ATLAS_PANDA.JEDI_Tasks SET status=:new_status "
        sqlU += "WHERE jediTaskID=:jediTaskID AND status=:old_status "
        varMap = dict()
        varMap[":old_status"] = "staging"
        varMap[":new_status"] = "pending"
        with_id = {}
        for name, (task_id, _other) in task_dict.items():
            with_id[name] = (task_id, _other)
        for one_batch in batched(with_id.items(), chunk_size):
            varMaps = []
            for name, (task_id, _other) in one_batch:
                tmp_varMap = varMap.copy()
                tmp_varMap[":jediTaskID"] = task_id
                varMaps.append(tmp_varMap)
            self.cur.executemany(sqlU + comment, varMaps)
'''


_ROUTED = '''
class TaskModule:
    def retry(self, jediTaskID, getNew):
        varMap = {}
        varMap[":jediTaskID"] = jediTaskID
        varMap[":status"] = "ready"
        if not getNew:
            sqlU = "UPDATE ATLAS_PANDA.JEDI_Tasks SET status=:status WHERE jediTaskID=:jediTaskID "
            self.cur.execute(sqlU + comment, varMap)
        sqlB = "UPDATE ATLAS_PANDA.JEDI_Tasks SET oldStatus=:status WHERE jediTaskID=:jediTaskID "
        if getNew:
            self.cur.execute(sqlB + comment, varMap)
        else:
            self.cur.execute(sqlB + comment, varMap)

    def partly(self, jediTaskID, getNew):
        varMap = {}
        varMap[":jediTaskID"] = jediTaskID
        varMap[":status"] = "ready"
        if not getNew:
            sqlP = "UPDATE ATLAS_PANDA.JEDI_Tasks SET status=:status WHERE jediTaskID=:jediTaskID "
            self.cur.execute(sqlP + comment, varMap)

    def rebinding(self, jediTaskID, getNew, forced):
        getNew = getNew or forced
        varMap = {}
        varMap[":jediTaskID"] = jediTaskID
        varMap[":status"] = "ready"
        if not getNew:
            sqlR = "UPDATE ATLAS_PANDA.JEDI_Tasks SET status=:status WHERE jediTaskID=:jediTaskID "
            self.cur.execute(sqlR + comment, varMap)

    def uncalled(self, jediTaskID, getNew):
        varMap = {}
        varMap[":jediTaskID"] = jediTaskID
        varMap[":status"] = "ready"
        if not getNew:
            sqlN = "UPDATE ATLAS_PANDA.JEDI_Tasks SET status=:status WHERE jediTaskID=:jediTaskID "
            self.cur.execute(sqlN + comment, varMap)

    def daily(self):
        self.retry(1, True)
        self.partly(1, True)
        self.rebinding(1, True, False)

    def weekly(self):
        self.retry(2, True)
        self.partly(2, False)


def outside(proxy):
    proxy.partly(3, True)
'''


def _routed(tmp_path):
    """:data:`_ROUTED`'s KeyFroms by the variable of their text, through :func:`sqlwrite.attach_routes`."""
    rel = "pandaserver/taskbuffer/db_proxy_mods/task_module.py"
    modules, calls, record = _keyed_corpus(tmp_path, {"pandaserver/taskbuffer/Specs.py": _SPECS, rel: _ROUTED})
    attributor = attribution.SpecAttributor(progress.spec_attributes(modules), attribution.class_bases(modules))
    attributor.learn_table_classes(modules)
    froms: list[KeyFrom] = []
    with sql.across_calls(calls):
        _subjects, junctions, *_rest = sqlwrite.extract(modules, MAP_ID, VERSION, attributor, key_froms=froms)
        sqlwrite.attach_key_sources([], junctions, calls, record, key_froms=froms, read_sites=[])
        sqlwrite.attach_routes(froms, calls)
    lines = _ROUTED.split("\n")

    def at(text: str) -> str:
        return f"{rel}:{next(number for number, line in enumerate(lines, 1) if text in line)}"

    by = {}
    for one in froms:
        variable = lines[one.statement - 1].split("execute(")[1].split(" ")[0]
        by.setdefault(variable, []).append(one)
    return by, at


def test_a_write_says_the_routes_on_which_it_never_runs(tmp_path):
    """``retryJob``'s old-row ``UPDATE`` runs only ``if not getNewPandaID``, and every placed call hands ``True``: the
    record says each route it is out on, and that no call of the function goes unplaced -- a statement no placed
    caller runs.  ``partly`` is out on the route that hands ``True`` alone, and a call by its name on an undeclared
    receiver leaves its routes open."""
    by, at = _routed(tmp_path)

    (old_row,) = by["sqlU"]
    assert (old_row.routes, old_row.routes_open) == (2, False)
    assert sorted(old_row.out_on) == [[at("self.retry(1, True)")], [at("self.retry(2, True)")]]
    (partly,) = by["sqlP"]
    assert (partly.routes, partly.routes_open, partly.out_on) == (2, True, [[at("self.partly(1, True)")]])


def test_a_write_run_from_two_places_is_a_record_for_each_out_where_it_is(tmp_path):
    """One text and one map run from two places is a record per place: ``if getNew`` runs on both routes, every
    placed caller handing ``True``, and its ``else`` on neither -- the write is held by both, so a reader dropping
    the place that never runs keeps it."""
    by, at = _routed(tmp_path)

    runs, never = sorted(by["sqlB"], key=lambda one: one.statement)
    assert (runs.routes, runs.out_on) == (2, [])
    assert (never.routes, never.out_on) == (2, [[at("self.retry(1, True)")], [at("self.retry(2, True)")]])
    assert runs.writes == never.writes


def test_a_flag_the_function_binds_again_rules_out_no_route(tmp_path):
    """``getNew = getNew or forced``: what the caller handed is not what the test reads."""
    by, _at = _routed(tmp_path)

    (rebound,) = by["sqlR"]
    assert (rebound.routes, rebound.out_on) == (1, [])


def test_a_write_of_a_function_no_call_places_has_no_route(tmp_path):
    """Not that it never runs: an entry point, a thread's ``run``, a daemon's ``main`` is called by no line, and is
    reached from outside the corpus -- a route the corpus's calls do not hold (``Calls.missed``)."""
    by, _at = _routed(tmp_path)

    (uncalled,) = by["sqlN"]
    assert (uncalled.routes, uncalled.routes_open, uncalled.out_on) == (0, True, [])


_REREAD_SPECS = """
class JediTaskSpec(object):
    _attributes = ("jediTaskID", "status", "parent_tid")

    def pack(self, values):
        for i in range(len(self._attributes)):
            attr = self._attributes[i]
            val = values[i]
            object.__setattr__(self, attr, val)
"""

_REREAD = '''
from pandaserver.taskbuffer.JediTaskSpec import JediTaskSpec

comment = " /* x */"


class TaskModule:
    def toFinish(self):
        varMap = {}
        varMap[":status1"] = "prepared"
        varMap[":status2"] = "scouted"
        sqlRT = "SELECT tabT.jediTaskID,tabT.status,tabT.parent_tid FROM ATLAS_PANDA.JEDI_Tasks tabT "
        sqlRT += "WHERE tabT.status IN (:status1,:status2) "
        self.cur.execute(sqlRT + comment, varMap)
        resList = self.cur.fetchall()
        retTasks = []
        for jediTaskID, taskStatus, parentID in resList:
            if taskStatus == "scouted":
                varMap = {}
                varMap[":jediTaskID"] = jediTaskID
                varMap[":newStatus"] = "running"
                sqlSC = "UPDATE ATLAS_PANDA.JEDI_Tasks SET status=:newStatus WHERE jediTaskID=:jediTaskID "
                self.cur.execute(sqlSC + comment, varMap)
            else:
                varMap = {}
                varMap[":jediTaskID"] = jediTaskID
                sqlTS = "SELECT jediTaskID,status,parent_tid FROM ATLAS_PANDA.JEDI_Tasks WHERE jediTaskID=:jediTaskID "
                self.cur.execute(sqlTS + comment, varMap)
                resTS = self.cur.fetchone()
                taskSpec = JediTaskSpec()
                taskSpec.pack(resTS)
                retTasks.append(taskSpec)
                varMap = {}
                varMap[":jediTaskID"] = taskSpec.jediTaskID
                sqlDS = "SELECT datasetID FROM ATLAS_PANDA.JEDI_Datasets WHERE jediTaskID=:jediTaskID "
                self.cur.execute(sqlDS + comment, varMap)
                varMap = {}
                varMap[":jediTaskID"] = parentID
                sqlP = "SELECT status FROM ATLAS_PANDA.JEDI_Tasks WHERE jediTaskID=:jediTaskID "
                self.cur.execute(sqlP + comment, varMap)
        return retTasks

    def twice(self, jediTaskID, again):
        varMap = {}
        varMap[":jediTaskID"] = jediTaskID
        sqlW = "SELECT status FROM ATLAS_PANDA.JEDI_Tasks WHERE jediTaskID=:jediTaskID "
        if again:
            self.cur.execute(sqlW + comment, varMap)
        else:
            self.cur.execute(sqlW + comment, varMap)

    def kinds(self, cols):
        self.cur.execute("SELECT * FROM ATLAS_PANDA.JEDI_Tasks WHERE status=:s", {":s": "ready"})
        every = self.cur.fetchall()
        for row in every:
            self.cur.execute("SELECT status FROM ATLAS_PANDA.JEDI_Tasks WHERE jediTaskID=:a", {":a": row[0]})
        self.cur.execute("SELECT COUNT(*),MAX(jediTaskID) FROM ATLAS_PANDA.JEDI_Tasks WHERE status=:s", {":s": "x"})
        res = self.cur.fetchone()
        self.cur.execute("SELECT status FROM ATLAS_PANDA.JEDI_Tasks WHERE jediTaskID=:b", {":b": res[1]})
        self.cur.execute("SELECT {0} FROM ATLAS_PANDA.JEDI_Tasks WHERE status=:s".format(cols), {":s": "y"})
        one = self.cur.fetchone()
        self.cur.execute("SELECT status FROM ATLAS_PANDA.JEDI_Tasks WHERE jediTaskID=:c", {":c": one[2]})
        spec = JediTaskSpec()
        spec.pack(one)
        self.cur.execute("SELECT status FROM ATLAS_PANDA.JEDI_Tasks WHERE jediTaskID=:h", {":h": spec.lockedBy})
        self.cur.execute("SELECT status FROM ATLAS_PANDA.JEDI_Tasks WHERE jediTaskID=:i", {":i": spec.parent_tid})

    def texts(self, defined):
        if defined:
            sqlA = "SELECT jediTaskID,status FROM ATLAS_PANDA.JEDI_Tasks WHERE status=:s "
        else:
            sqlA = "SELECT jediTaskID,parent_tid+1 FROM ATLAS_PANDA.JEDI_Tasks WHERE status=:s "
        self.cur.execute(sqlA + comment, {":s": "done"})
        resA = self.cur.fetchall()
        for row in resA:
            self.cur.execute("SELECT status FROM ATLAS_PANDA.JEDI_Tasks WHERE jediTaskID=:d", {":d": row[0]})
            self.cur.execute("SELECT status FROM ATLAS_PANDA.JEDI_Tasks WHERE jediTaskID=:e", {":e": row[1]})
        if defined:
            sqlB = "SELECT jediTaskID FROM ATLAS_PANDA.JEDI_Tasks WHERE status=:s "
        else:
            sqlB = "SELECT jediTaskID FROM ATLAS_PANDA.JEDI_Datasets WHERE status=:s "
        self.cur.execute(sqlB + comment, {":s": "done"})
        resB = self.cur.fetchall()
        for row in resB:
            self.cur.execute("SELECT status FROM ATLAS_PANDA.JEDI_Tasks WHERE jediTaskID=:f", {":f": row[0]})
        sqlC = "SELECT * FROM (SELECT jediTaskID,status FROM ATLAS_PANDA.JEDI_Tasks WHERE status=:s "
        sqlC += "ORDER BY jediTaskID) WHERE rownum<=:n "
        self.cur.execute(sqlC + comment, {":s": "done", ":n": 10})
        resC = self.cur.fetchall()
        for row in resC:
            self.cur.execute("SELECT status FROM ATLAS_PANDA.JEDI_Tasks WHERE jediTaskID=:g", {":g": row[0]})
        if defined:
            sqlK = "SELECT jediTaskID FROM ATLAS_PANDA.JEDI_Tasks WHERE status=:s "
        else:
            sqlK = "SELECT jediTaskID FROM ATLAS_PANDA.JEDI_Tasks_Old WHERE status=:s "
        self.cur.execute(sqlK + comment, {":s": "done"})
        resK = self.cur.fetchall()
        for row in resK:
            self.cur.execute("SELECT status FROM ATLAS_PANDA.JEDI_Tasks WHERE jediTaskID=:k", {":k": row[0]})
        sqlO = "UPDATE ATLAS_PANDA.JEDI_Tasks_Old SET status=:st,parent_tid=:pt WHERE jediTaskID=:t "
        self.cur.execute(sqlO + comment, {":st": "done", ":pt": None, ":t": 1})

    def handed(self, taskSpec, taskID, tasks):
        varMap = {}
        varMap[":jediTaskID"] = taskSpec.parent_tid
        self.cur.execute("SELECT status FROM ATLAS_PANDA.JEDI_Tasks WHERE jediTaskID=:jediTaskID" + comment, varMap)
        self.cur.execute("SELECT status FROM ATLAS_PANDA.JEDI_Tasks WHERE jediTaskID=:m", {":m": taskID})
        for task in tasks:
            self.cur.execute("SELECT status FROM ATLAS_PANDA.JEDI_Tasks WHERE jediTaskID=:n", {":n": task.jediTaskID})

    def restart(self):
        sqlTL = "SELECT jediTaskID,status FROM ATLAS_PANDA.JEDI_Tasks WHERE status IN (:s1,:s2) "
        self.cur.execute(sqlTL + comment, {":s1": "defined", ":s2": "ready"})
        resTL = self.cur.fetchall()
        sqlTD = "UPDATE ATLAS_PANDA.JEDI_Tasks SET status=:newStatus WHERE jediTaskID=:jediTaskID "
        for jediTaskID, taskStatus in resTL:
            if taskStatus == "defined":
                varMap = {}
                varMap[":jediTaskID"] = jediTaskID
                varMap[":newStatus"] = "defined"
                self.cur.execute(sqlTD + comment, varMap)
            else:
                varMap = {}
                varMap[":jediTaskID"] = jediTaskID
                varMap[":newStatus"] = "defined"
                self.cur.execute(sqlTD + comment, varMap)
        self.cur.execute("SELECT jediTaskID FROM ATLAS_PANDA.JEDI_Tasks WHERE status=:s" + comment, {":s": "ready"})
        resTW = self.cur.fetchall()
        for (jediTaskID,) in resTW:
            varMap = {}
            varMap[":jediTaskID"] = jediTaskID
            varMap[":newStatus"] = "defined"
            self.cur.execute(sqlTD + comment, varMap)
'''


def _reread(tmp_path):
    """The KeyFroms of :data:`_REREAD`'s statements, read sources and all, with a schema giving the tables' keys, and
    a function of the source to find a line by its text."""
    rel = "pandaserver/taskbuffer/db_proxy_mods/task_module.py"
    modules, calls, record = _keyed_corpus(
        tmp_path, {"pandaserver/taskbuffer/JediTaskSpec.py": _REREAD_SPECS, rel: _REREAD}
    )
    attributor = attribution.SpecAttributor(progress.spec_attributes(modules), attribution.class_bases(modules))
    attributor.learn_table_classes(modules)
    # As the build hands them: the attributor the attribute slice reads with, which learns no table, and the
    # tables' classes from the one that did.
    audited = attribution.SpecAttributor(progress.spec_attributes(modules), attribution.class_bases(modules))
    schema = {
        "jedi_tasks": frozenset({"jeditaskid", "status", "parent_tid"}),
        "jedi_tasks_old": frozenset({"jeditaskid", "status", "parent_tid"}),
        "jedi_datasets": frozenset({"jeditaskid", "datasetid"}),
    }
    keys = {
        "jedi_tasks": frozenset({"jeditaskid"}),
        "jedi_tasks_old": frozenset({"jeditaskid"}),
        "jedi_datasets": frozenset({"jeditaskid", "datasetid"}),
    }
    froms: list[KeyFrom] = []
    with sql.across_calls(calls), sqltext.with_schema(schema, keys, {}):
        _subjects, junctions, *_rest = sqlwrite.extract(modules, MAP_ID, VERSION, attributor, key_froms=froms)
        sites = sqlwrite.read_site_nodes(sqlwrite.read_side(modules, attributor).statements, MAP_ID, VERSION)
        sqlwrite.attach_key_sources(
            [],
            junctions,
            calls,
            record,
            key_froms=froms,
            read_sites=sites,
            table_class=attributor.class_for_table,
            modules=modules,
            attributor=audited,
        )
    lines = _REREAD.split("\n")

    def at(text: str) -> int:
        return next(number for number, line in enumerate(lines, 1) if text in line)

    return [one for one in froms if one.term], sites, at


def test_a_read_statements_terms_say_which_rows_fill_them_and_whether_they_are_the_same_rows(tmp_path):
    """getTasksToBeFinished_JEDI's shape (the querier's A11, K28): a statement's rows fill the key of a read of the
    same table by its key -- the same rows, read again -- of a write of them, of a read of another table by them, and
    of a read of the same table by another column; each source says the column, and the values the statement
    selects at which the keyed statement never runs, by the test on the rows' status on its way."""
    froms, sites, at = _reread(tmp_path)
    by = {(one.statement, one.term): one for one in froms}
    (rt,) = [site for site in sites if site.anchor.line_start == at("self.cur.execute(sqlRT")]
    assert sorted(rt.selects) == ["JediTaskSpec.status=prepared", "JediTaskSpec.status=scouted"]

    def rows(line: int) -> list[tuple]:
        (key_from,) = [one for (statement, term), one in by.items() if statement == line and "jediTaskID" in term]
        return [
            (s.name.rsplit("@", 1)[1], s.column, s.attr, s.column_unread, s.same_rows, s.never_at)
            for s in key_from.sources
            if s.kind == "rows"
        ]

    ts = by[(at("self.cur.execute(sqlTS"), "jediTaskID=:jediTaskID")]
    assert (ts.reads, ts.writes, ts.table, ts.attr, ts.unread) == (True, {}, "JEDI_Tasks", "JediTaskSpec.jediTaskID", "")
    (site_ts,) = [site for site in sites if site.anchor.line_start == ts.statement]
    assert site_ts.owner == ts.owner
    rt_line = str(rt.anchor.line_start)
    # Read again: the task's own rows, by its key, from the key column -- and not at the value that takes the if.
    assert rows(ts.statement) == [
        (rt_line, "jedi_tasks.jeditaskid", "JediTaskSpec.jediTaskID", "", True, ["JediTaskSpec.status=scouted"])
    ]
    # A write of the same rows, under the if: never at the other value.
    assert rows(at("self.cur.execute(sqlSC")) == [
        (rt_line, "jedi_tasks.jeditaskid", "JediTaskSpec.jediTaskID", "", True, ["JediTaskSpec.status=prepared"])
    ]
    # Another table by the rows read again: a link, the field of the packed object its column by name.
    ts_line = str(ts.statement)
    assert rows(at("self.cur.execute(sqlDS")) == [
        (ts_line, "jedi_tasks.jeditaskid", "JediTaskSpec.jediTaskID", "", False, [])
    ]
    # The same table by another column: the parent's row, not these.
    assert rows(at("self.cur.execute(sqlP")) == [
        (rt_line, "jedi_tasks.parent_tid", "JediTaskSpec.parent_tid", "", False, ["JediTaskSpec.status=scouted"])
    ]
    # The status list is a term of the statement's own, filled by literals.
    rt_terms = [one for one in froms if one.statement == rt.anchor.line_start]
    assert [(one.term, [s.kind for s in one.sources]) for one in rt_terms] == [
        ("status IN (:status1, :status2)", ["constant", "constant"])
    ]


def test_a_read_run_from_two_places_is_a_read_site_and_a_keyfrom_at_each(tmp_path):
    """One text with one map run from two calls is one run, and each call is a statement of its own: a read site at
    each, and its KeyFrom at that site's line, so a reader joining a source's statement to the read site finds
    each."""
    froms, sites, at = _reread(tmp_path)
    twice = [one for one in froms if one.owner.endswith("::twice")]
    lines = sorted(one.anchor.line_start for one in sites if one.owner.endswith("::twice"))
    assert [(one.statement, one.term, [(s.kind, s.name) for s in one.sources]) for one in twice] == [
        (line, "jediTaskID=:jediTaskID", [("parameter", "jediTaskID")]) for line in lines
    ]
    assert lines[0] == at("self.cur.execute(sqlW") and len(lines) == 2


def test_a_rows_source_says_why_it_names_no_column(tmp_path):
    """``SELECT *``, an aggregate, a position a hole's columns fill, a field of an object packed from a hole that the
    table has no column of: the rows fill the key, and which column is not said, with why -- a field it has is its
    column."""
    froms, _sites, _at = _reread(tmp_path)
    kinds = {one.term: one for one in froms if one.owner.endswith("::kinds")}
    assert {
        term: [(s.kind, s.column, s.column_unread) for s in kinds[term].sources]
        for term in ("jediTaskID=:a", "jediTaskID=:b", "jediTaskID=:c", "jediTaskID=:h", "jediTaskID=:i")
    } == {
        "jediTaskID=:a": [("rows", "", "select *")],
        "jediTaskID=:b": [("rows", "", "an expression")],
        "jediTaskID=:c": [("rows", "", "in a hole")],
        "jediTaskID=:h": [("rows", "", "no item")],
        "jediTaskID=:i": [("rows", "jedi_tasks.parent_tid", "")],
    }


def test_a_value_s_column_is_read_on_each_select_list_of_the_statement_s_texts(tmp_path):
    """One line, two texts by an ``if``: a value is the column both lists give at its place, and where they give
    two -- another column, another table -- which is not said, with why, and the subject where both are one (a
    table of the same class); a ``*`` over one derived table is its list (``getJobs``'s ``SELECT * FROM (SELECT
    PandaID, ...) WHERE rownum<=:n``)."""
    froms, _sites, _at = _reread(tmp_path)
    texts = {one.term: one for one in froms if one.owner.endswith("::texts")}
    task = "JediTaskSpec.jediTaskID"
    assert {
        term: [(s.kind, s.column, s.attr, s.column_unread) for s in texts[term].sources]
        for term in ("jediTaskID=:d", "jediTaskID=:e", "jediTaskID=:f", "jediTaskID=:g", "jediTaskID=:k")
    } == {
        "jediTaskID=:d": [("rows", "jedi_tasks.jeditaskid", task, "")],
        "jediTaskID=:e": [("rows", "", "", "texts differ")],
        "jediTaskID=:f": [("rows", "", "", "texts differ")],
        "jediTaskID=:g": [("rows", "jedi_tasks.jeditaskid", task, "")],
        "jediTaskID=:k": [("rows", "", task, "texts differ")],
    }


def test_a_statement_run_from_several_calls_never_runs_at_a_value_only_where_the_call_handed_the_value_does_not(
    tmp_path,
):
    """restartTasksForContentsUpdate_JEDI's shape: one statement run in both arms of a test on the rows' status and
    in a loop over another statement's rows, each call with its own bind -- the call another statement's rows reach
    runs at every value, and does not keep the arms' sources from saying theirs."""
    froms, sites, at = _reread(tmp_path)
    (tl,) = [site for site in sites if site.anchor.line_start == at("self.cur.execute(sqlTL")]
    assert sorted(tl.selects) == ["JediTaskSpec.status=defined", "JediTaskSpec.status=ready"]
    tl_line = str(tl.anchor.line_start)
    found = sorted(
        (s.line, s.name.rsplit("@", 1)[1] == tl_line, s.never_at)
        for one in froms
        if one.owner.endswith("::restart") and one.term == "jediTaskID=:jediTaskID"
        for s in one.sources
        if s.kind == "rows"
    )
    assert [(same, never) for _line, same, never in found] == [
        (True, ["JediTaskSpec.status=ready"]),
        (True, ["JediTaskSpec.status=defined"]),
        (False, []),
    ]


def test_a_select_list_reads_a_name_holding_a_hole_as_the_hole_and_a_star_over_one_derived_table_as_its_list():
    """``{}{}`` is a hole spelled as two (a column list built in a loop, read on a pass that writes two), not a
    column of that name; ``SELECT * FROM (SELECT a, b ...)`` returns the derived table's items, a ``*`` over a union
    or a table stays one."""
    hole = sqltext.SelectItem(sqltext.ITEM_HOLE, "jobsactive4")
    assert sqltext.select_items("SELECT {}{} FROM ATLAS_PANDA.jobsActive4 WHERE PandaID=:PandaID") == (hole,)
    assert sqltext.select_items("SELECT {},{} FROM ATLAS_PANDA.jobsActive4 WHERE PandaID=:PandaID") == (
        hole,
        sqltext.SelectItem(sqltext.ITEM_AFTER_HOLE),
    )
    derived = "SELECT * FROM (SELECT PandaID,jobStatus FROM ATLAS_PANDA.jobsActive4 tab WHERE x=:x) WHERE rownum<=:n"
    assert sqltext.select_items(derived) == (
        sqltext.SelectItem(sqltext.ITEM_COLUMN, "jobsactive4", "pandaid"),
        sqltext.SelectItem(sqltext.ITEM_COLUMN, "jobsactive4", "jobstatus"),
    )
    star = sqltext.SelectItem(sqltext.ITEM_STAR)
    assert sqltext.select_items("SELECT * FROM (SELECT a FROM t1 UNION SELECT a FROM t2)") == (star,)
    assert sqltext.select_items("SELECT * FROM ATLAS_PANDA.jobsActive4 WHERE PandaID=:p") == (star,)



def test_a_key_a_parameter_s_field_fills_says_the_field_as_a_subject(tmp_path):
    """archiveJob 1718's shape (P2's own): the key is filled with a field of what the caller hands -- the parameter's
    own, or of an element of it -- and the source names the parameter and, as a subject, the field; a parameter
    handed whole is what the callers hand, no subject."""
    froms, _sites, _at = _reread(tmp_path)
    handed = {one.term: one for one in froms if one.owner.endswith("::handed")}
    assert {
        term: [(s.kind, s.name, s.paths, s.attr) for s in handed[term].sources]
        for term in ("jediTaskID=:jediTaskID", "jediTaskID=:m", "jediTaskID=:n")
    } == {
        "jediTaskID=:jediTaskID": [("parameter", "taskSpec", ["()"], "JediTaskSpec.parent_tid")],
        "jediTaskID=:m": [("parameter", "taskID", ["()"], "")],
        "jediTaskID=:n": [("parameter", "tasks", ["()"], "JediTaskSpec.jediTaskID")],
    }


_TABLES_SPECS = """
class JobSpec(object):
    _attributes = ("PandaID", "jobStatus", "creationTime", "modificationTime")
"""

_TABLES = '''
comment = " /* x */"


class JobModule:
    def insertJob(self, pandaID):
        sqlI = "INSERT INTO ATLAS_PANDA.jobsActive4 (PandaID,jobStatus,creationTime,modificationTime) "
        sqlI += "VALUES (:PandaID,:jobStatus,CURRENT_DATE,CURRENT_DATE) "
        self.cur.execute(sqlI + comment, {":PandaID": pandaID, ":jobStatus": "activated"})

    def insertOther(self, pandaID):
        varMap = {}
        varMap[":PandaID"] = pandaID
        varMap[":jobStatus"] = "defined"
        varMap[":ct"] = naive_utcnow()
        varMap[":mt"] = naive_utcnow()
        sqlD = "INSERT INTO ATLAS_PANDA.jobsActive4 (PandaID,jobStatus,creationTime,modificationTime) "
        sqlD += "VALUES (:PandaID,:jobStatus,:ct,:mt) "
        self.cur.execute(sqlD + comment, varMap)

    def touch(self, pandaID):
        sqlU = "UPDATE ATLAS_PANDA.jobsActive4 SET jobStatus=:s,modificationTime=CURRENT_DATE WHERE PandaID=:p "
        self.cur.execute(sqlU + comment, {":s": "running", ":p": pandaID})

    def both(self):
        sqlR = "SELECT PandaID FROM ATLAS_PANDA.jobsActive4 WHERE jobStatus=:s "
        sqlR += "UNION SELECT PandaID FROM ATLAS_PANDAARCH.jobsArchived WHERE jobStatus=:s "
        self.cur.execute(sqlR + comment, {":s": "failed"})
        return self.cur.fetchall()
'''


def test_a_table_record_says_its_key_as_the_class_spells_it_its_family_and_its_rows_creation(tmp_path):
    """The live-case harness's fetch tables (P2R): a table holding a class's rows names its row by the class's
    attribute, is read with its archive in one UNION, and the class's column every write of which makes the row
    and puts the time now -- by the statement's clock or a bound one -- is its creation; one an UPDATE writes is
    not.  The archive is the family's table holding the rows that no INSERT writes, partitioned by time."""
    rel = "pandaserver/taskbuffer/db_proxy_mods/job_module.py"
    modules, calls, record = _keyed_corpus(
        tmp_path, {"pandaserver/taskbuffer/JobSpec.py": _TABLES_SPECS, rel: _TABLES}
    )
    attributor = attribution.SpecAttributor(progress.spec_attributes(modules), attribution.class_bases(modules))
    attributor.learn_table_classes(modules)
    held = frozenset({"pandaid", "jobstatus", "creationtime", "modificationtime"})
    schema = {"jobsactive4": held, "jobsarchived": held}
    keys = {"jobsactive4": frozenset({"pandaid"})}
    dated = {"jobsactive4": frozenset({"creationtime", "modificationtime"}), "jobsarchived": frozenset()}
    froms: list[KeyFrom] = []
    with sql.across_calls(calls), sqltext.with_schema(schema, keys, dated):
        _subjects, junctions, *_rest = sqlwrite.extract(modules, MAP_ID, VERSION, attributor, key_froms=froms)
        sites = sqlwrite.read_site_nodes(sqlwrite.read_side(modules, attributor).statements, MAP_ID, VERSION)
        sqlwrite.attach_key_sources(
            [], junctions, calls, record, key_froms=froms, read_sites=sites,
            table_class=attributor.class_for_table, modules=modules, attributor=attributor,
        )
        entities = [
            EntityNode(
                map_id=MAP_ID, derived_from=VERSION, name="jobsactive4", tables=["jobsActive4"],
                spec_class="JobSpec", created_by=[f"{rel}::JobModule.insertJob"],
            )
        ]
        made = tables.records(
            modules, entities, sites, froms, attributor, attributor.class_for_table, schema, keys,
            {"jobsarchived": ("range", ("modificationtime",))},
            {"jobsactive4": ("ATLAS_PANDA",), "jobsarchived": ("ATLAS_PANDAARCH",)},
        )
    by = {one.name: one for one in made}
    written = sorted(
        (one.owner.rsplit("::", 1)[1], line)
        for one in froms
        for line in one.writes.get("JobSpec.creationTime", [])
    )
    assert [owner for owner, _line in written] == ["insertJob", "insertOther"]
    assert sorted(by) == ["jobsactive4", "jobsarchived"]
    live, archive = by["jobsactive4"], by["jobsarchived"]
    assert (live.rows_of, live.identity, live.identity_attrs, live.inserted, live.family, live.creation) == (
        "JobSpec", ["pandaid"], ["JobSpec.PandaID"], True, ["jobsarchived"], ["JobSpec.creationTime"],
    )
    assert (live.schema_names, live.partitioned_by, live.spellings) == (["ATLAS_PANDA"], [], ["jobsActive4"])
    assert (
        archive.rows_of, archive.identity, archive.inserted, archive.family, archive.schema_names,
        archive.partition_kind, archive.partitioned_by, archive.partitioned_by_attrs, archive.creation,
    ) == (
        "JobSpec", [], False, ["jobsactive4"], ["ATLAS_PANDAARCH"],
        "range", ["modificationtime"], ["JobSpec.modificationTime"], ["JobSpec.creationTime"],
    )


def test_the_ddl_says_a_tables_partition_and_the_schema_creating_it(tmp_path):
    """``PARTITION BY RANGE ("MODIFICATIONTIME")`` after a table's body is its, not the next table's; the schema
    is the one its ``CREATE TABLE`` names."""
    (tmp_path / "ATLAS_PANDA.sql").write_text(
        'CREATE TABLE "ATLAS_PANDA"."JOBSARCHIVED4"\n   (\t"PANDAID" NUMBER(11,0),\n\t"MODIFICATIONTIME" DATE\n   )\n'
        '  PARTITION BY RANGE ("MODIFICATIONTIME")\n (PARTITION "P1" VALUES LESS THAN (MAXVALUE)) ;\n'
        'CREATE TABLE "ATLAS_PANDA"."JOBSACTIVE4"\n   (\t"PANDAID" NUMBER(11,0)\n   ) ;\n'
    )
    (tmp_path / "ATLAS_PANDAARCH.sql").write_text(
        'CREATE TABLE "ATLAS_PANDAARCH"."JOBSARCHIVED"\n   (\t"PANDAID" NUMBER(11,0),\n\t"JEDITASKID" NUMBER\n   )\n'
        '  PARTITION BY LIST ("JEDITASKID", "PANDAID")\n (PARTITION "P1" VALUES (DEFAULT)) ;\n'
    )
    (tmp_path / "ATLAS_PANDA_VIEW.sql").write_text('CREATE OR REPLACE VIEW "ATLAS_PANDA"."V" AS SELECT 1 FROM dual;\n')
    assert sqltext.read_partitions(tmp_path) == {
        "jobsarchived4": ("range", ("modificationtime",)),
        "jobsarchived": ("list", ("jeditaskid", "pandaid")),
    }
    assert sqltext.read_declaring(tmp_path) == {
        "jobsarchived4": ("ATLAS_PANDA",),
        "jobsactive4": ("ATLAS_PANDA",),
        "jobsarchived": ("ATLAS_PANDAARCH",),
    }


def test_a_union_pairs_the_tables_its_arms_read_in_one_place():
    """The arms' one sources, or the sources of one alias -- not every table of a joined arm with every other, not
    a WITH name."""
    assert sqltext.union_places(
        "SELECT PandaID FROM ATLAS_PANDA.jobsArchived4 WHERE x=:x "
        "UNION SELECT PandaID FROM ATLAS_PANDAARCH.jobsArchived"
    ) == (("jobsarchived", "jobsarchived4"),)
    assert sqltext.union_places(
        "SELECT f.lfn FROM filesTable4 f, jobsActive4 j WHERE f.PandaID=j.PandaID "
        "UNION SELECT f.lfn FROM filesTable4 f, jobsDefined4 j WHERE f.PandaID=j.PandaID"
    ) == (("jobsactive4", "jobsdefined4"),)
    assert sqltext.union_places("WITH w AS (SELECT a FROM t1) SELECT a FROM w UNION SELECT a FROM t2") == ()

_ACROSS = '''
from pandaserver.taskbuffer.JediTaskSpec import JediTaskSpec

comment = " /* x */"


class TaskModule:
    def toFinish(self):
        self.cur.execute("SELECT jediTaskID,status FROM ATLAS_PANDA.JEDI_Tasks WHERE status=:s" + comment, {":s": "done"})
        resList = self.cur.fetchall()
        for jediTaskID, taskStatus in resList:
            taskSpec = self.readTask(jediTaskID)
            self.finish(taskSpec)
            self.lockTask(jediTaskID)

    def elsewhere(self, someID):
        taskSpec = self.readTask(someID)
        self.other(taskSpec)

    def toCheck(self):
        self.cur.execute("SELECT jediTaskID,status FROM ATLAS_PANDA.JEDI_Tasks WHERE status=:s" + comment, {":s": "ok"})
        res = self.cur.fetchall()
        return res

    def checker(self):
        for jediTaskID, taskStatus in self.toCheck():
            taskSpec = self.readTask(jediTaskID)
            self.finish(taskSpec)

    def readTask(self, jediTaskID):
        varMap = {}
        varMap[":jediTaskID"] = jediTaskID
        sqlTS = "SELECT jediTaskID,status,parent_tid FROM ATLAS_PANDA.JEDI_Tasks WHERE jediTaskID=:jediTaskID "
        self.cur.execute(sqlTS + comment, varMap)
        res = self.cur.fetchone()
        taskSpec = JediTaskSpec()
        taskSpec.pack(res)
        return taskSpec

    def lockTask(self, jediTaskID):
        varMap = {}
        varMap[":jediTaskID"] = jediTaskID
        sqlLK = "SELECT status FROM ATLAS_PANDA.JEDI_Tasks WHERE jediTaskID=:jediTaskID FOR UPDATE "
        self.cur.execute(sqlLK + comment, varMap)
        varMap = {}
        varMap[":jediTaskID"] = jediTaskID
        varMap[":status"] = "locked"
        sqlUL = "UPDATE ATLAS_PANDA.JEDI_Tasks SET status=:status WHERE jediTaskID=:jediTaskID "
        self.cur.execute(sqlUL + comment, varMap)

    def finish(self, taskSpec):
        varMap = {}
        varMap[":jediTaskID"] = taskSpec.jediTaskID
        varMap[":status"] = "finished"
        sqlF = "UPDATE ATLAS_PANDA.JEDI_Tasks SET status=:status WHERE jediTaskID=:jediTaskID "
        self.cur.execute(sqlF + comment, varMap)

    def other(self, taskSpec):
        varMap = {}
        varMap[":jediTaskID"] = taskSpec.jediTaskID
        varMap[":status"] = "other"
        sqlO = "UPDATE ATLAS_PANDA.JEDI_Tasks SET status=:status WHERE jediTaskID=:jediTaskID "
        self.cur.execute(sqlO + comment, varMap)
'''


def test_rows_reaching_a_key_through_calls_say_the_way_and_where_the_rows_read_again_go(tmp_path):
    """getTaskWithID_JEDI's shape: a statement's ids handed to a helper reading each task again by its key, the task
    returned and handed on; and to a junction's owner that reads them in a bind (the walk crosses to it, not through).
    The source says the call it came by, the column, the same rows -- and where the task read again goes back up that
    call alone, not to what another caller of the helper hands it to."""
    rel = "pandaserver/taskbuffer/db_proxy_mods/task_module.py"
    modules, calls, record = _keyed_corpus(
        tmp_path, {"pandaserver/taskbuffer/JediTaskSpec.py": _REREAD_SPECS, rel: _ACROSS}
    )
    attributor = attribution.SpecAttributor(progress.spec_attributes(modules), attribution.class_bases(modules))
    attributor.learn_table_classes(modules)
    audited = attribution.SpecAttributor(progress.spec_attributes(modules), attribution.class_bases(modules))
    schema = {"jedi_tasks": frozenset({"jeditaskid", "status", "parent_tid"})}
    froms: list[KeyFrom] = []
    landings: dict[str, list] = {}
    with sql.across_calls(calls), sqltext.with_schema(schema, {"jedi_tasks": frozenset({"jeditaskid"})}, {}):
        _subjects, junctions, *_rest = sqlwrite.extract(modules, MAP_ID, VERSION, attributor, key_froms=froms)
        sites = sqlwrite.read_site_nodes(sqlwrite.read_side(modules, attributor).statements, MAP_ID, VERSION)
        sqlwrite.attach_crossings(sites, modules, junctions, calls, record, landings=landings)
        sqlwrite.attach_key_sources(
            [],
            junctions,
            calls,
            record,
            key_froms=froms,
            read_sites=sites,
            table_class=attributor.class_for_table,
            modules=modules,
            attributor=audited,
            landings=landings,
        )
    lines = _ACROSS.split("\n")

    def at(text: str, nth: int = 1) -> int:
        return [number for number, line in enumerate(lines, 1) if text in line][nth - 1]

    (source_site,) = [site for site in sites if site.owner.endswith("::toFinish")]

    def rows(text: str) -> list[tuple]:
        (key_from,) = [one for one in froms if one.statement == at(text) and one.term == "jediTaskID=:jediTaskID"]
        return [
            (one.via, one.column, one.same_rows, [owner.rsplit("::", 1)[1] for owner in one.reaches])
            for one in key_from.sources
            if one.kind == "rows" and one.name == source_site.name
        ]

    assert rows("self.cur.execute(sqlTS") == [
        ([f"{rel}:{at('taskSpec = self.readTask(jediTaskID)')}"], "jedi_tasks.jeditaskid", True, ["finish"])
    ]
    # In an owner crossed to: the read there, and the write.
    assert rows("self.cur.execute(sqlLK") == [
        ([f"{rel}:{at('self.lockTask(jediTaskID)')}"], "jedi_tasks.jeditaskid", True, [])
    ]
    assert rows("self.cur.execute(sqlUL") == [
        ([f"{rel}:{at('self.lockTask(jediTaskID)')}"], "jedi_tasks.jeditaskid", True, [])
    ]
    # Returned to a caller first, then handed down: the column is still the returning statement's.
    (back_site,) = [site for site in sites if site.owner.endswith("::toCheck")]
    (key_from,) = [
        one for one in froms if one.statement == at("self.cur.execute(sqlTS") and one.term == "jediTaskID=:jediTaskID"
    ]
    assert [
        (one.via, one.column, one.attr, one.same_rows)
        for one in key_from.sources
        if one.kind == "rows" and one.name == back_site.name
    ] == [
        (
            [
                f"{rel}:{at('for jediTaskID, taskStatus in self.toCheck()')}",
                f"{rel}:{at('taskSpec = self.readTask(jediTaskID)', 2)}",
            ],
            "jedi_tasks.jeditaskid",
            "JediTaskSpec.jediTaskID",
            True,
        )
    ]


_JOINED = '''
comment = " /* x */"


class TaskModule:
    def joined(self):
        sql = "SELECT d.datasetID FROM ATLAS_PANDA.JEDI_Datasets d, ATLAS_PANDA.JEDI_Dataset_Contents c "
        sql += "WHERE d.jediTaskID=c.jediTaskID AND d.datasetID=c.datasetID AND c.status=:s "
        sql += "AND NOT EXISTS (SELECT 1 FROM ATLAS_PANDA.JEDI_Tasks t WHERE t.jediTaskID=d.jediTaskID) "
        self.cur.execute(sql + comment, {":s": "ready"})
        sqlP = "SELECT a.jediTaskID FROM ATLAS_PANDA.JEDI_Tasks a LEFT OUTER JOIN ATLAS_PANDA.JEDI_Tasks b "
        sqlP += "ON a.parent_tid=b.jediTaskID WHERE a.status=:s AND a.status=a.oldStatus "
        self.cur.execute(sqlP + comment, {":s": "running"})

    def set(self, jediTaskID):
        sql = "UPDATE ATLAS_PANDA.JEDI_Tasks SET status=:status,parent_tid=:p WHERE jediTaskID=:jediTaskID "
        self.cur.execute(sql + comment, {":status": "x", ":p": 1, ":jediTaskID": jediTaskID})
'''


def test_a_read_site_says_each_equality_joining_two_of_its_sources(tmp_path):
    """The harness's P2 by joins: each column = column between two sources, a subquery's and an outer join's ``ON``
    among them, each side's source as the statement names it -- so the two columns of a composite key share one
    pair, and two aliases of one table are two sources -- and its subject where the table's class has the column.
    Not a column compared with its own source's."""
    rel = "pandaserver/taskbuffer/db_proxy_mods/task_module.py"
    modules = [_module(_REREAD_SPECS, "pandaserver/taskbuffer/JediTaskSpec.py"), _module(_JOINED, rel)]
    attributor = attribution.SpecAttributor(progress.spec_attributes(modules), attribution.class_bases(modules))
    attributor.learn_table_classes(modules)
    sites = sqlwrite.read_site_nodes(sqlwrite.read_side(modules, attributor).statements, MAP_ID, VERSION)
    by = {site.owner.rsplit("::", 1)[1] + "@" + str(site.anchor.line_start): site for site in sites}
    joined = [site for name, site in by.items() if name.startswith("joined@")]
    assert [[(one.a, one.b, one.a_source, one.b_source, one.a_attr, one.b_attr) for one in site.joins] for site in joined] == [
        [
            ("jedi_dataset_contents.datasetid", "jedi_datasets.datasetid", "c", "d", "", ""),
            ("jedi_dataset_contents.jeditaskid", "jedi_datasets.jeditaskid", "c", "d", "", ""),
            ("jedi_datasets.jeditaskid", "jedi_tasks.jeditaskid", "d", "t", "", "JediTaskSpec.jediTaskID"),
        ],
        [
            (
                "jedi_tasks.jeditaskid",
                "jedi_tasks.parent_tid",
                "b",
                "a",
                "JediTaskSpec.jediTaskID",
                "JediTaskSpec.parent_tid",
            )
        ],
    ]


def test_a_key_filled_from_a_batch_is_the_batched_values_not_the_batch_size(tmp_path):
    """``updateInputFilesStaged_JEDI``'s shape: a map per file, copied from one filled before the loop, appended to the
    list ``executemany`` takes, the file's key from a batch of a dict the caller handed.  ``batched`` is not the
    corpus's, and a call no reading places holds the rows where something handed to it holds them whole -- so the
    batch size, an int handed whole, read as every key's source, and the dict, handed as items, as none."""
    rel = "pandaserver/taskbuffer/db_proxy_mods/task_module.py"
    modules, calls, record = _keyed_corpus(tmp_path, {"pandaserver/taskbuffer/Specs.py": _SPECS, rel: _BATCHED})
    attributor = attribution.SpecAttributor(
        progress.spec_attributes(modules), attribution.class_bases(modules)
    )
    attributor.learn_table_classes(modules)
    keys: list[RowKeySource] = []
    froms: list[KeyFrom] = []
    with sql.across_calls(calls):
        _subjects, junctions, *_rest = sqlwrite.extract(
            modules, MAP_ID, VERSION, attributor, row_keys=keys, key_froms=froms
        )
        sqlwrite.attach_key_sources(keys, junctions, calls, record, key_froms=froms, read_sites=[])

    assert {
        (branch.outcome, tuple(branch.row_precondition)) for junction in junctions for branch in junction.branches
    } == {("pending", ("status=staging",))}
    (key_from,) = [one for one in froms if one.term == "jediTaskID=:jediTaskID"]
    assert key_from.unread == ""
    assert {(source.kind, source.name) for source in key_from.sources} == {("parameter", "task_dict")}


_BORN = '''
class JobModule:
    def holdJob(self, pandaID):
        sqlH = "UPDATE ATLAS_PANDA.jobsActive4 SET jobStatus=:jobStatus WHERE PandaID=:PandaID "
        self.cur.execute(sqlH + comment, {":jobStatus": "holding", ":PandaID": pandaID})

    def killJob(self, pandaID):
        sqlS = "SELECT PandaID,jobStatus FROM ATLAS_PANDA.jobsActive4 WHERE PandaID=:PandaID "
        self.cur.execute(sqlS + comment, {":PandaID": pandaID})
        res = self.cur.fetchone()
        job = JobSpec()
        job.PandaID = res[0]
        job.jobStatus = "cancelled"
        sqlA = f"INSERT INTO ATLAS_PANDA.jobsArchived4 ({JobSpec.columnNames()}) "
        sqlA += JobSpec.bindValuesExpression()
        self.cur.execute(sqlA + comment, job.valuesMap())

    def archiveJob(self, job: JobSpec):
        job.jobStatus = "finished"
        sqlA = f"INSERT INTO ATLAS_PANDA.jobsArchived4 ({JobSpec.columnNames()}) "
        sqlA += JobSpec.bindValuesExpression()
        self.cur.execute(sqlA + comment, job.valuesMap())

    def addReport(self, pandaID, attemptNr):
        sqlR = "INSERT INTO ATLAS_PANDA.Job_Output_Report (PandaID,attemptNr,status) VALUES (:PandaID,:attemptNr,:status) "
        self.cur.execute(sqlR + comment, {":PandaID": pandaID, ":attemptNr": attemptNr, ":status": "new"})

    def lockProcess(self, vo):
        sqlD = "DELETE FROM ATLAS_PANDA.Process_Lock WHERE vo=:vo "
        self.cur.execute(sqlD + comment, {":vo": vo})
        sqlI = "INSERT INTO ATLAS_PANDA.Process_Lock (vo,lockedBy) VALUES (:vo,:lockedBy) "
        self.cur.execute(sqlI + comment, {":vo": vo, ":lockedBy": "me"})

    def addEvents(self, pandaID, nEvents):
        sqlE = "INSERT INTO ATLAS_PANDA.Events (PandaID,eventID,status) VALUES (:PandaID,:eventID,:status) "
        varMaps = []
        for i in range(nEvents):
            varMaps.append({":PandaID": pandaID, ":eventID": i + 1, ":status": "ready"})
        self.cur.executemany(sqlE + comment, varMaps)

    def addStdOut(self, pandaID, stdOut, exists):
        sqlU = "UPDATE ATLAS_PANDA.jobsDebug SET stdOut=:stdOut WHERE PandaID=:PandaID "
        sqlI = "INSERT INTO ATLAS_PANDA.jobsDebug (PandaID,stdOut) VALUES (:PandaID,:stdOut) "
        if exists:
            sql = sqlU
        else:
            sql = sqlI
        varMap = {}
        varMap[":PandaID"] = pandaID
        varMap[":stdOut"] = stdOut
        self.cur.execute(sql + comment, varMap)

    def moveJob(self, pandaID, spare: JobSpec):
        varMap = spare.valuesMap()
        sqlS = "SELECT PandaID,jobStatus FROM ATLAS_PANDA.jobsActive4 WHERE PandaID=:PandaID "
        self.cur.execute(sqlS + comment, {":PandaID": pandaID})
        res = self.cur.fetchone()
        job = JobSpec()
        job.PandaID = res[0]
        sqlA = f"INSERT INTO ATLAS_PANDA.jobsArchived4 ({JobSpec.columnNames()}) "
        sqlA += JobSpec.bindValuesExpression()
        varMap = job.valuesMap()
        self.cur.execute(sqlA + comment, varMap)

    def addFiles(self, files):
        varMaps = []
        for lfn in files:
            fileSpec = FileSpec()
            fileSpec.lfn = lfn
            varMaps.append(fileSpec.valuesMap())
        sqlF = f"INSERT INTO ATLAS_PANDA.filesTable4 ({FileSpec.columnNames()}) "
        sqlF += FileSpec.bindValuesExpression()
        self.cur.executemany(sqlF + comment, varMaps)

    def newTask(self):
        sqlT = "INSERT INTO ATLAS_PANDA.JEDI_Tasks (jediTaskID,status) "
        sqlT += "VALUES (ATLAS_PANDA.JEDI_TASK_ID_SEQ.nextval,:status) RETURNING jediTaskID INTO :newID "
        self.cur.execute(sqlT + comment, {":status": "defined", ":newID": None})

    def retryJob(self, pandaID):
        sql1 = "SELECT PandaID,jobStatus FROM ATLAS_PANDA.jobsActive4 WHERE PandaID=:PandaID "
        for iTry in range(3):
            try:
                varMap = {}
                varMap[":PandaID"] = pandaID
                self.cur.execute(sql1 + comment, varMap)
                res = self.cur.fetchone()
                job = JobSpec()
                job.PandaID = res[0]
                sql1 = f"INSERT INTO ATLAS_PANDA.jobsActive4 ({JobSpec.columnNames()}) "
                sql1 += JobSpec.bindValuesExpression() + " RETURNING PandaID INTO :newPandaID"
                varMap = job.valuesMap()
                varMap[":newPandaID"] = self.cur.var(varNUMBER)
                self.cur.execute(sql1 + comment, varMap)
                return True
            except Exception:
                continue

    def addContents(self, nFiles):
        sqlFID = "SELECT ATLAS_PANDA.JEDI_DATASET_CONT_FILEID_SEQ.nextval FROM "
        sqlFID += "(SELECT level FROM dual CONNECT BY level<=:nIDs) "
        self.cur.execute(sqlFID + comment, {":nIDs": nFiles})
        resFID = self.cur.fetchall()
        newFileIDs = []
        for (fileID,) in resFID:
            newFileIDs.append(fileID)
        varMaps = []
        for fileID in newFileIDs:
            fileSpec = JediFileSpec()
            fileSpec.fileID = None
            if fileID is not None:
                fileSpec.fileID = fileID
                varMaps.append(fileSpec.valuesMap())
        sqlC = f"INSERT INTO ATLAS_PANDA.JEDI_Dataset_Contents ({JediFileSpec.columnNames()}) "
        sqlC += JediFileSpec.bindValuesExpression()
        self.cur.executemany(sqlC + comment, varMaps)

    def copyContents(self, jediTaskID, fileID):
        sqlGI = "SELECT * FROM ATLAS_PANDA.JEDI_Dataset_Contents WHERE jediTaskID=:jediTaskID AND fileID=:fileID "
        self.cur.execute(sqlGI + comment, {":jediTaskID": jediTaskID, ":fileID": fileID})
        resGI = self.cur.fetchone()
        sqlJI = "INSERT INTO ATLAS_PANDA.JEDI_Dataset_Contents VALUES ("
        varMap = {}
        for columDesc, columVal in zip(self.cur.description, resGI):
            keyName = f":{columDesc[0]}"
            varMap[keyName] = columVal
            sqlJI += f"{keyName},"
        sqlJI = sqlJI[:-1] + ") "
        self.cur.execute(sqlJI + comment, varMap)

    def linkJobs(self, workerID, pandaIDs, stale):
        for pandaID in pandaIDs:
            sqlL = "INSERT INTO ATLAS_PANDA.Rel_Jobs (workerID,PandaID) VALUES (:workerID,:PandaID) "
            self.cur.execute(sqlL + comment, {":workerID": workerID, ":PandaID": pandaID})
        for pandaID in stale:
            sqlX = "DELETE FROM ATLAS_PANDA.Rel_Jobs WHERE workerID=:workerID AND PandaID=:PandaID "
            self.cur.execute(sqlX + comment, {":workerID": workerID, ":PandaID": pandaID})

    def addRequests(self, dc_req_specs):
        for dc_req_spec in dc_req_specs:
            sqlR = f"INSERT INTO ATLAS_PANDA.data_carousel_requests ({dc_req_spec.columnNames()}) "
            sqlR += f"{dc_req_spec.bindValuesExpression()} RETURNING request_id INTO :new_request_id "
            var_map = dc_req_spec.valuesMap(useSeq=True)
            var_map[":new_request_id"] = self.cur.var(varNUMBER)
            self.cur.execute(sqlR + comment, var_map)
'''


def test_an_insert_says_whether_it_makes_its_row_or_moves_one_of_its_class(tmp_path):
    """BI, with the querier: an INSERT always makes a row of its table; S1 asks whether the entity was a row of its
    class before.  Moved: its key carried from a row of the class read here (killJob's archive); rewritten: the
    function deletes the row its whole key names first; new: a key column filled by RETURNING or a sequence, computed
    here, or carried from something not of the class (a scalar handed in, a table no class maps); not said: an object
    of the class handed in (who made its key is the caller's).  A key's dated columns time a version, and are left
    out (jobsArchived4's MODIFICATIONTIME).  An object's INSERT writes the subjects the function assigns on it.
    A run that updates the row or inserts it is not said; the object is the one whose map reaches a call of the run,
    or whose maps the list ``executemany`` takes holds; its field is what the function assigns to it where the map
    is made, read through the value assigned -- a sequence's next values are a source of their own.  An INSERT
    spelling no columns gets one record per key column all the same: ``RETURNING``'s is new, the others not read.
    (``holdJob`` is there for the class ``jobsActive4`` holds: the attributor learns it from a write's columns.)"""
    rel = "pandaserver/taskbuffer/db_proxy_mods/job_module.py"
    modules, calls, record = _keyed_corpus(tmp_path, {"pandaserver/taskbuffer/Specs.py": _SPECS, rel: _BORN})
    attributor = attribution.SpecAttributor(
        progress.spec_attributes(modules), attribution.class_bases(modules)
    )
    attributor.learn_table_classes(modules)
    keys = {
        "jobsarchived4": frozenset({"pandaid", "modificationtime"}),
        "jobsactive4": frozenset({"pandaid"}),
        "job_output_report": frozenset({"pandaid", "attemptnr"}),
        "process_lock": frozenset({"vo"}),
        "events": frozenset({"pandaid", "eventid"}),
        "jedi_tasks": frozenset({"jeditaskid"}),
        "jobsdebug": frozenset({"pandaid"}),
        "filestable4": frozenset({"row_id"}),
        "jedi_dataset_contents": frozenset({"fileid"}),
        "data_carousel_requests": frozenset({"request_id"}),
        "rel_jobs": frozenset({"workerid", "pandaid"}),
    }
    froms: list[KeyFrom] = []
    _s, attribute_junctions, *_rest = progress.extract(modules, MAP_ID, VERSION)
    with sql.across_calls(calls), sqltext.with_schema(None, keys, {"jobsarchived4": frozenset({"modificationtime"})}):
        _subjects, junctions, *_rest = sqlwrite.extract(modules, MAP_ID, VERSION, attributor, key_froms=froms)
        sites = sqlwrite.read_site_nodes(sqlwrite.read_side(modules, attributor).statements, MAP_ID, VERSION)
        sqlwrite.attach_key_sources(
            [], junctions + attribute_junctions, calls, record, key_froms=froms, read_sites=sites,
            table_class=attributor.class_for_table,
        )

    marks = {
        (one.owner.rpartition("::")[2], one.table.lower(), one.term): (one.new_row, one.new_row_by)
        for one in froms
        if one.new_row is not None or one.new_row_by
    }
    assert marks == {
        ("killJob", "jobsarchived4", "PandaID=job.PandaID"): (False, "moved within its class"),
        ("archiveJob", "jobsarchived4", "PandaID=job.PandaID"): (None, "handed in as its class"),
        ("addReport", "job_output_report", "PandaID=:PandaID"): (True, "not of its class"),
        ("addReport", "job_output_report", "attemptNr=:attemptNr"): (True, "not of its class"),
        ("lockProcess", "process_lock", "vo=:vo"): (False, "rewritten"),
        ("addEvents", "events", "PandaID=:PandaID"): (True, "computed"),
        ("addEvents", "events", "eventID=:eventID"): (True, "computed"),
        ("newTask", "jedi_tasks", "jediTaskID RETURNING"): (True, "returning"),
        ("addStdOut", "jobsdebug", "PandaID=:PandaID"): (None, "updates or inserts"),
        ("moveJob", "jobsarchived4", "PandaID=job.PandaID"): (False, "moved within its class"),
        ("addFiles", "filestable4", "row_id"): (None, "not read"),
        # sql1 holds the INSERT at the SELECT's call too, round the retry loop: one run of two calls, the object's map
        # reaching the second.  The text before its RETURNING reaches the first as well; it shares the mark.
        ("retryJob", "jobsactive4", "PandaID RETURNING"): (True, "returning"),
        ("retryJob", "jobsactive4", "pandaid"): (True, "returning"),
        # The field as assigned where the map is made -- not its reset to None above -- from the sequence's values.
        ("addContents", "jedi_dataset_contents", "fileID=fileSpec.fileID"): (True, "sequence"),
        ("copyContents", "jedi_dataset_contents", "fileid"): (None, "not read"),
        ("addRequests", "data_carousel_requests", "request_id RETURNING"): (True, "returning"),
        # A DELETE of the whole key after the INSERT deletes other rows: the one inserted is still new.
        ("linkJobs", "rel_jobs", "workerID=:workerID"): (True, "not of its class"),
        ("linkJobs", "rel_jobs", "PandaID=:PandaID"): (True, "not of its class"),
    }
    # The map reaching the run is job's, not the spare's assigned above it; the files' maps are the list's.
    assert {one.object for one in froms if one.owner.endswith(("::moveJob", "::addFiles")) and one.new_row_by} == {
        "job",
        "fileSpec",
    }
    assert {one.object for one in froms if one.owner.endswith(("::retryJob", "::addRequests")) and one.object} == {
        "job",
        "dc_req_spec",
    }
    (filled,) = [one for one in froms if one.owner.endswith("::addContents")]
    assert [(source.kind, source.name) for source in filled.sources] == [("sequence", "JEDI_DATASET_CONT_FILEID_SEQ")]
    lines = _BORN.split("\n")
    (killed,) = [one for one in froms if one.owner.endswith("::killJob") and one.table.lower() == "jobsarchived4"]
    assert killed.object == "job"
    assert killed.writes == {
        "JobSpec.PandaID": [next(n for n, line in enumerate(lines, 1) if "job.PandaID = res[0]" in line)],
        "JobSpec.jobStatus": [next(n for n, line in enumerate(lines, 1) if '"cancelled"' in line)],
    }


_SEEDS = '''
class TaskModule:
    def getRandomSeed(self, jediTaskID, nNew):
        sqlFR = "SELECT fileID,status FROM ATLAS_PANDA.JEDI_Dataset_Contents WHERE jediTaskID=:jediTaskID "
        self.cur.execute(sqlFR + comment, {":jediTaskID": jediTaskID})
        var_maps = []
        for resFR in self.cur.fetchall():
            tmpFileSpec = JediFileSpec()
            tmpFileSpec.pack(resFR)
            varMap = {}
            varMap[":jediTaskID"] = jediTaskID
            varMap[":fileID"] = tmpFileSpec.fileID
            varMap[":status"] = "picked"
            var_maps.append(varMap)
        sqlFU = "UPDATE ATLAS_PANDA.JEDI_Dataset_Contents SET status=:status "
        sqlFU += "WHERE jediTaskID=:jediTaskID AND fileID=:fileID "
        self.cur.executemany(sqlFU + comment, var_maps)
        var_maps = []
        for i in range(nNew):
            tmpFileSpec = JediFileSpec()
            tmpFileSpec.status = "picked"
            varMap = tmpFileSpec.valuesMap()
            var_maps.append(varMap)
            tmpFileSpec.status = "ready"
        sqlFI = f"INSERT INTO ATLAS_PANDA.JEDI_Dataset_Contents ({JediFileSpec.columnNames()}) "
        sqlFI += JediFileSpec.bindValuesExpression()
        self.cur.executemany(sqlFI + comment, var_maps)

    def addFiles(self, lfns):
        kept = []
        for lfn in lfns:
            fileSpec = JediFileSpec()
            fileSpec.status = "staging"
            kept.append(fileSpec)
        varMaps = []
        for fileSpec in kept:
            varMaps.append(fileSpec.valuesMap())
        sqlFI = f"INSERT INTO ATLAS_PANDA.JEDI_Dataset_Contents ({JediFileSpec.columnNames()}) "
        sqlFI += JediFileSpec.bindValuesExpression()
        self.cur.executemany(sqlFI + comment, varMaps)

    def addOne(self, first):
        sqlFI = f"INSERT INTO ATLAS_PANDA.JEDI_Dataset_Contents ({JediFileSpec.columnNames()}) "
        sqlFI += JediFileSpec.bindValuesExpression()
        if first:
            fileSpec = JediFileSpec()
            fileSpec.status = "defined"
            varMap = fileSpec.valuesMap()
            self.cur.execute(sqlFI + comment, varMap)
        else:
            fileSpec = JediFileSpec()
            fileSpec.status = "pending"
            varMap = fileSpec.valuesMap()
            self.cur.execute(sqlFI + comment, varMap)
'''


def test_an_object_write_is_what_its_map_holds_when_it_runs(tmp_path):
    """getRandomSeed_JEDI's shape, the querier's finding on AK-1: one list name taken by two executemany calls, made
    again between them, and a field set after the object's valuesMap() is taken.  The UPDATE of the old seeds runs
    with the first list -- its maps are dicts, no object's -- and the INSERT of the new seeds writes what each map
    held when it was made: "picked", not the "ready" set on the spec after.  A spec taken again from a list it was
    kept in is the same spec (insertFilesForDataset_JEDI), and an INSERT run from two places writes both places'
    (registerTaskInOneShot_JEDI)."""
    rel = "pandaserver/taskbuffer/db_proxy_mods/task_module.py"
    modules, calls, record = _keyed_corpus(tmp_path, {"pandaserver/taskbuffer/Specs.py": _SPECS, rel: _SEEDS})
    attributor = attribution.SpecAttributor(
        progress.spec_attributes(modules), attribution.class_bases(modules)
    )
    attributor.learn_table_classes(modules)
    froms: list[KeyFrom] = []
    _s, attribute_junctions, *_rest = progress.extract(modules, MAP_ID, VERSION)
    with sql.across_calls(calls), sqltext.with_schema(None, {"jedi_dataset_contents": frozenset({"fileid"})}, {}):
        _subjects, junctions, *_rest = sqlwrite.extract(modules, MAP_ID, VERSION, attributor, key_froms=froms)
        sites = sqlwrite.read_site_nodes(sqlwrite.read_side(modules, attributor).statements, MAP_ID, VERSION)
        sqlwrite.attach_key_sources(
            [], junctions + attribute_junctions, calls, record, key_froms=froms, read_sites=sites,
            table_class=attributor.class_for_table,
        )

    lines = _SEEDS.split("\n")
    picked_bind, picked_field, ready = (
        next(n for n, line in enumerate(lines, 1) if spelled in line)
        for spelled in ('":status"] = "picked"', 'tmpFileSpec.status = "picked"', '"ready"')
    )
    updated = [one for one in froms if one.term == "fileID=:fileID"]
    assert [(one.object, one.writes) for one in updated] == [("", {"JediFileSpec.status": [picked_bind]})]
    (inserted,) = [one for one in froms if one.new_row_by and one.owner.endswith("::getRandomSeed")]
    assert inserted.object == "tmpFileSpec"
    assert inserted.writes == {"JediFileSpec.status": [picked_field]}
    assert ready not in inserted.writes["JediFileSpec.status"]
    staging, defined, pending = (
        next(n for n, line in enumerate(lines, 1) if f'fileSpec.status = "{status}"' in line)
        for status in ("staging", "defined", "pending")
    )
    written = {one.owner.rpartition("::")[2]: one.writes for one in froms if one.new_row_by}
    assert written["addFiles"] == {"JediFileSpec.status": [staging]}
    assert written["addOne"] == {"JediFileSpec.status": [defined, pending]}
