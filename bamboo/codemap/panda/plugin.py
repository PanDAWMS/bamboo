"""PanDA Code Map plugin.

One map covers ``pandaserver`` *and* ``pandajedi``: they are separate Python
packages but a single distribution (``panda-server-source``) that always
deploys at one version.  A JEDI knight decides and a server-side proxy writes,
so the two halves of a single junction routinely sit in different packages --
splitting the map by package would turn those handoffs into cross-system
boundaries and demand version bindings for something that is not another
system.

Extraction is stdlib ``ast`` only.  A census over both versions showed ~92% of
writes to promoted subjects resolve with no interprocedural dataflow at all,
so a heavier analysis framework would buy the remaining few percent at the
cost of a build-time dependency.
"""

from __future__ import annotations

import ast
import hashlib
import logging
import time
from pathlib import Path
from typing import Optional

from bamboo.codemap import calls as calls_mod
from bamboo.codemap import reach, reading, trace
from bamboo.codemap.astwalk import walk
from bamboo.codemap.base import CodeMapPlugin
from bamboo.codemap.gitsource import blob_sha
from bamboo.codemap.gitsource import describe as _git_describe
from bamboo.codemap.models import (
    LOG_FILES_OF_MODULE,
    Anchor,
    AnnotationAudit,
    CallStop,
    FunctionPrintsNode,
    JunctionNode,
    MapFragment,
    SourceModule,
    SubjectNode,
    pooled_frames,
)
from bamboo.codemap.panda import mustprints, objects, promotion, runs, sql, sqltext
from bamboo.codemap.panda.attribution import (
    SpecAttributor,
    annotation_readings,
    class_bases,
    spec_annotation_forms,
)
from bamboo.codemap.panda.pathcond import (
    attach_parents,
    functions_with_owner,
    qualified_name,
)
from bamboo.codemap.panda.recognizers import (
    alias,
    boundary,
    emit,
    errorcode,
    flush,
    logfile,
    loopcut,
    outbound,
    progress,
    selection,
    sqlwrite,
    tables,
    trigger,
)

logger = logging.getLogger(__name__)


def _print_calls(modules: list[SourceModule]) -> dict[str, list[int]]:
    """``owner -> the line of each logging call the function makes itself``, off the syntax tree.

    The other reading of ``FunctionPrintsNode.statements``: no skeleton, no
    message, only the calls -- each the value of a statement of its own, so
    the statement's line.  Spelled with the owner the node carries.
    """
    lines: dict[str, list[int]] = {}
    for module in modules:
        attach_parents(module.tree)
        for func, _cls in functions_with_owner(module.tree):
            calls = emit.own_logging_arguments(func)
            if calls:
                lines[f"{module.rel_path}::{qualified_name(func)}"] = sorted(
                    argument.parent.parent.lineno  # type: ignore[attr-defined]
                    for argument in calls
                )
    return lines


def _print_landings(
    modules: list[SourceModule],
    reader: Optional[logfile.LoggerFiles],
) -> dict[tuple[str, int], logfile.Landing]:
    """``(owner, statement line) -> where that logging call's lines land``.

    Over the build's own trees, the same calls :func:`_print_calls` reads, so
    that the call graph finds a caller in a tree it has parsed.  Empty without
    a call graph: every statement then takes its module's files.
    """
    found: dict[tuple[str, int], logfile.Landing] = {}
    if reader is None:
        return found
    for module in modules:
        attach_parents(module.tree)
        for func, _cls in functions_with_owner(module.tree):
            owner = f"{module.rel_path}::{qualified_name(func)}"
            for argument in emit.own_logging_arguments(func):
                call = argument.parent  # type: ignore[attr-defined]
                key = (owner, call.parent.lineno)
                if key not in found:
                    found[key] = reader.of_call(module.rel_path, func, call)
    return found


def _merge_uplinks(*sources: dict) -> dict:
    """Pool the crossings, keeping every entry.

    Two readings of the same kind of edge -- a dispatched worker and a plain
    construction -- and a class can be both: ``ContentsFeederThread`` is a
    ``WorkerThread`` subclass that is also constructed with its rows.  Pooled
    rather than chosen between, since ``attach`` keys entries by door and will
    fold whatever genuinely coincides.
    """
    merged: dict = {}
    for source in sources:
        for key, entries in source.items():
            merged.setdefault(key, []).extend(entries)
    return merged


def _merge_junctions(junctions: list[JunctionNode]) -> list[JunctionNode]:
    """Combine junctions sharing a name, keeping every branch.

    Two recognizers can reach the same write site -- ``updateTaskStatus...``
    sets ``taskSpec.status`` *and* binds ``:status`` -- and both readings are
    real branches of the same junction rather than rival descriptions of it.
    """
    merged: dict[str, JunctionNode] = {}
    for junction in junctions:
        existing = merged.get(junction.name)
        if existing is None:
            merged[junction.name] = junction
            continue
        known = {(b.outcome, tuple(b.path_condition)): b for b in existing.branches}
        for branch in junction.branches:
            held = known.get((branch.outcome, tuple(branch.path_condition)))
            if held is not None:
                held.value_frames = pooled_frames(held.value_frames, branch.value_frames)
                held.other_lines = sorted(
                    {*held.other_lines, *[line for line in (branch.line, *branch.other_lines) if line is not None]}
                    - {held.line}
                )
                continue
            branch.order = len(existing.branches)
            existing.branches.append(branch)
        if existing.structural_subject is None:
            existing.structural_subject = junction.structural_subject
    return list(merged.values())


def _unique_subjects(subjects: list[SubjectNode]) -> list[SubjectNode]:
    """Collapse subjects the slices found independently, pooling their evidence."""
    merged: dict[str, SubjectNode] = {}
    for subject in subjects:
        existing = merged.get(subject.name)
        if existing is None:
            merged[subject.name] = subject
            continue
        for criterion in subject.criteria:
            if criterion not in existing.criteria:
                existing.criteria.append(criterion)
        if subject.vocabulary and not existing.vocabulary:
            existing.vocabulary = subject.vocabulary
    return list(merged.values())

PACKAGES = ("pandaserver", "pandajedi")

# Test code is not the system's behaviour: :func:`calls.is_test_module`, the
# one rule the build and the call index share.
_is_test_module = calls_mod.is_test_module


def _schema_stamp(root: Path, schema: dict[str, frozenset[str]], keys: dict[str, frozenset[str]]) -> str:
    """``<root>: <n> table(s), <k> key(s), <digest>`` -- the digest over the definition files, as read."""
    digest = hashlib.sha256()
    for path in sorted(root.glob("*.sql")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return f"{root}: {len(schema)} table(s), {len(keys)} key(s), {digest.hexdigest()[:16]}"


class _Lap:
    """How long each stage of a build took, logged as it ends: what to make faster is chosen from where the time
    goes, and a profiler's shares are not a run's (tmaeno, 2026-10-27: the build's time is the bottleneck)."""

    def __init__(self) -> None:
        self._at = time.monotonic()

    def __call__(self, stage: str) -> None:
        now = time.monotonic()
        logger.info("PandaCodeMapPlugin: %s took %.1f s", stage, now - self._at)
        self._at = now


class PandaCodeMapPlugin(CodeMapPlugin):
    """Builds the ``panda`` Code Map from installed or checked-out source."""

    def __init__(self) -> None:
        self._emits = 0
        self._glossed = 0
        self._flush_guards = 0
        self._roots: dict[str, Path] = {}
        self._version: str = ""
        self._modules: list[SourceModule] = []
        self._calls = None
        self._stop_disagreements: list[str] = []
        self._schema: Optional[dict[str, frozenset[str]]] = None
        self._keys: Optional[dict[str, frozenset[str]]] = None
        self._dated: Optional[dict[str, frozenset[str]]] = None
        self._partitions: dict[str, tuple[str, tuple[str, ...]]] = {}
        self._declaring: dict[str, tuple[str, ...]] = {}
        self._schema_stamp = "none"
        self._recorded: Optional[calls_mod.Recorded] = None
        self._record_stamp = "none"

    @property
    def map_id(self) -> str:
        return "panda"

    def prepare(
        self,
        source_root: Optional[Path] = None,
        schema_root: Optional[Path] = None,
        instance_record: Optional[Path] = None,
    ) -> str:
        """Resolve the source snapshot, parse it, and return the version stamp."""
        if schema_root is not None:
            self._schema = sqltext.read_schema(schema_root)
            self._keys = sqltext.read_keys(schema_root)
            self._dated = sqltext.read_dated(schema_root)
            self._partitions = sqltext.read_partitions(schema_root)
            self._declaring = sqltext.read_declaring(schema_root)
            self._schema_stamp = _schema_stamp(Path(schema_root), self._schema, self._keys)
        if instance_record is not None:
            self._recorded, self._record_stamp = calls_mod.read_recorded(Path(instance_record))
        self._roots = self._resolve_roots(source_root)
        if not self._roots:
            raise FileNotFoundError(
                "Neither pandaserver nor pandajedi found. Install the 'panda' "
                "extra or pass an explicit source root."
            )
        self._version = self._resolve_version(source_root)
        self._modules, self._skipped_tests = self._parse_modules(self._roots)
        logger.info(
            "PandaCodeMapPlugin: parsed %d module(s) from %s",
            len(self._modules),
            self._version,
        )
        return self._version

    def run(self) -> MapFragment:
        """Extract the fragment.  Requires :meth:`prepare` to have run.

        Every slice reads statements through the corpus's calls (see
        :func:`sql.across_calls`), so that two slices reading one statement
        read the same text.
        """
        if not self._modules:
            raise RuntimeError("prepare() must be called before run()")
        lap = _Lap()
        self._calls = sqlwrite.corpus_calls(self._modules, self._roots, self._recorded)
        lap("the corpus's calls")
        with sql.across_calls(self._calls), sqltext.with_schema(self._schema, self._keys, self._dated):
            return self._run()

    @property
    def schema_stamp(self) -> str:
        """Which table definitions decided the unqualified columns, or ``none``."""
        return self._schema_stamp

    @property
    def record_stamp(self) -> str:
        """Which instance's record the calls read module names by (:func:`calls.read_recorded`), or ``none``."""
        return self._record_stamp

    def _run(self) -> MapFragment:

        lap = _Lap()
        fragment = MapFragment(
            map_id=self.map_id,
            derived_from=self._version,
            instance_record="" if self._recorded is None else self._record_stamp,
        )
        enums, enum_coverage = errorcode.extract(self._modules, self.map_id, self._version)
        fragment.value_enums.extend(enums)
        fragment.coverage.extend(enum_coverage)

        boundaries, boundary_coverage = boundary.extract(
            self._modules, self.map_id, self._version
        )
        fragment.boundaries.extend(boundaries)
        fragment.coverage.extend(boundary_coverage)

        channels, channel_coverage = boundary.extract_shared_tables(
            self._modules, self.map_id, self._version
        )
        fragment.boundaries.extend(channels)
        fragment.coverage.extend(channel_coverage)

        # The same crossing found from the read side rather than the schema
        # qualifier: a table that bounds a query's reach and that nothing here
        # keeps current.  Kept out of the coverage matrix because there is no
        # denominator to be a fraction of -- every join either names such a
        # table or does not.
        self._never_written = boundary.tables_never_written(self._modules)
        gates = boundary.extract_read_only_tables(
            self._modules,
            self.map_id,
            self._version,
            self._never_written,
            {channel.interface.split(".")[-1] for channel in channels},
        )
        fragment.boundaries.extend(gates)
        # What PanDA sends out: the other direction, read from the interface
        # classes' methods and the calls placed at them.
        if self._calls is not None:
            fragment.boundaries.extend(
                outbound.extract(self._modules, self._calls, self.map_id, self._version)
            )

        filter_stages, selection_coverage, self._unexplained_steps = selection.extract(
            self._modules, self.map_id, self._version
        )
        fragment.filter_stages.extend(filter_stages)
        fragment.coverage.extend(selection_coverage)

        # The same cut everywhere the code does not declare a chain: a guard in
        # a loop, a ``continue``, and a sentence.  Told which functions the
        # stage slice already reads so that a brokerage rejection is not given a
        # second identity keyed on its wording instead of its tag.
        cuts, self._silent_cuts = loopcut.extract(
            self._modules,
            self.map_id,
            self._version,
            {stage.owner for stage in filter_stages},
        )
        fragment.loop_cuts.extend(cuts)

        # Which enumeration a constant belongs to, by its bare name.  A name
        # two enumerations share cannot decode anything, so it is dropped
        # rather than guessed at -- the one such name in the corpus belongs to
        # neither of the modules that declare error codes.
        by_constant: dict[str, set[str]] = {}
        for enum in enums:
            by_constant.setdefault(enum.constant, set()).add(enum.namespace)
        enumerations = {
            constant: next(iter(spaces))
            for constant, spaces in by_constant.items()
            if len(spaces) == 1
        }

        lap("enumerations and boundaries")
        subjects, junctions, progress_coverage, written_text, bindings = progress.extract(
            self._modules, self.map_id, self._version, enumerations
        )
        fragment.subjects.extend(subjects)
        fragment.junctions.extend(junctions)
        fragment.coverage.extend(progress_coverage)
        fragment.diagnostics.extend(written_text)
        fragment.enumeration_writes.extend(bindings)
        self._tagged_branches = sum(1 for j in junctions for b in j.branches if b.tags)
        self._unclaimed_tags = progress.unclaimed_tags(self._modules, junctions)

        lap("the attribute slice")
        # The SQL slice needs an attributor that already knows the table map,
        # which is learned from the whole corpus rather than from one module.
        declarations = progress.spec_attributes(self._modules)
        self._declarations = declarations
        # Every declaring class and how many columns came back, so that a form
        # this reader does not understand fails a gate instead of vanishing into
        # `unresolved`.  Counted per class rather than per assignment: the four
        # classes using the typed form also assign the older name, deriving it,
        # and that derived line legitimately yields nothing on its own.
        yields: dict[str, set[str]] = {}
        for cls, rel_path, names in progress.spec_declarations(self._modules):
            yields.setdefault(f"{cls} ({rel_path})", set()).update(names)
        fragment.declaration_yields = {where: len(names) for where, names in yields.items()}
        attributor = SpecAttributor(declarations, class_bases(self._modules))
        attributor.learn_self_attributes(self._modules)
        attributor.learn_return_types(self._modules)
        attributor.learn_import_aliases(self._modules)
        self._table_conflicts = attributor.learn_table_classes(self._modules)

        # Audit the classes the map takes on trust from an annotation.  Needs an
        # attributor equipped the same way the progress slice's was, since both
        # readings are the ordinary resolution -- one of the annotated container,
        # one of what the code stores in it.
        audited = SpecAttributor(declarations, class_bases(self._modules))
        audited.learn_element_types(self._modules)
        audited.learn_self_attributes(self._modules)
        audited.learn_return_types(self._modules)
        audited.learn_import_aliases(self._modules)

        # Writes the attribution drops for having a receiver whose class is a
        # run-time choice.  Collected here rather than counted inside the
        # progress slice so that the reading lives in one place: the slice asks
        # the attributor, and so does this.
        spec_attribute_names = (
            set().union(*declarations.values()) if declarations else set()
        )
        self._runtime_dispatch = [
            f"{module.rel_path}:{target.lineno} {ast.unparse(target)}"
            for module in self._modules
            for func, _owner in functions_with_owner(module.tree)
            for target in walk(func)
            if isinstance(target, ast.Attribute)
            and isinstance(target.ctx, ast.Store)
            and target.attr in spec_attribute_names
            and isinstance(target.value, ast.Name)
            and audited.built_at_runtime(target.value.id, func)
        ]
        fragment.annotation_readings = [
            AnnotationAudit(
                where=row.where,
                kind=row.kind,
                container=row.container,
                stated=row.stated,
                put_in=sorted(row.put_in),
                read=row.read,
                audited=row.audited,
                unsettled=sorted(row.unsettled),
            )
            for row in annotation_readings(
                self._modules,
                audited,
                set().union(*declarations.values()) if declarations else set(),
            )
        ]
        # And whether every annotation that names a spec class was legible at
        # all.  The audit above can only report on annotations it managed to
        # read; a form the reader does not understand yields no row, which is
        # the same blindness ``spec-declarations-are-read`` was written for.
        fragment.spec_annotation_forms = spec_annotation_forms(self._modules, audited)

        lap("attributors and annotation audits")
        alias_subjects, alias_junctions, alias_coverage = alias.extract(
            self._modules, self.map_id, self._version, declarations, attributor
        )
        fragment.subjects.extend(alias_subjects)
        fragment.junctions.extend(alias_junctions)
        fragment.coverage.extend(alias_coverage)

        # The same relationship read the other way -- a helper that returns the
        # value the caller writes.  Kept separate in the coverage matrix because
        # the two have different denominators.
        returned_subjects, returned_junctions, returned_coverage = alias.extract_producers(
            self._modules, self.map_id, self._version, declarations, attributor
        )
        fragment.subjects.extend(returned_subjects)
        fragment.junctions.extend(returned_junctions)
        fragment.coverage.extend(returned_coverage)

        lap("the alias slices")
        (
            sql_subjects,
            sql_junctions,
            sql_coverage,
            self._uncovered_tables,
            bound_text,
            self._unexplained_writes,
            reading_stops,
        ) = sqlwrite.extract(
            self._modules,
            self.map_id,
            self._version,
            attributor,
            row_keys=fragment.row_key_sources,
            key_froms=fragment.key_froms,
        )
        # On the fragment rather than on the plugin, because a gate reads the
        # fragment: "which junctions guard on a value nothing selects on" is a
        # question whose answer is half in the nodes and half in what the
        # build could not read, and a gate that could only see the first half
        # would report every unreadable ``WHERE`` as a disagreement.
        fragment.reading_stops.extend(sqlwrite.distinct_stops(reading_stops))
        self._stop_disagreements = _stops_read_again(
            self._modules, self._roots, fragment.reading_stops
        )
        fragment.subjects.extend(sql_subjects)
        fragment.junctions.extend(sql_junctions)
        fragment.coverage.extend(sql_coverage)
        fragment.diagnostics.extend(bound_text)

        lap("the SQL slice")
        # The read side of the same statements: which values something selects
        # rows on, and whose rows it selects at all.  The values are attached to
        # the subject rather than kept apart, so the invariants can compare them
        # with what the junctions write; the rows become their own nodes,
        # because a pair of class and attribute can only describe a column.
        side = sqlwrite.read_side(self._modules, attributor, self._calls)
        # The read side's denominator.  Without it a query form the slice
        # does not recognise can go missing without a single number moving.
        read_cov, self._unexplained_reads = sqlwrite.read_coverage(self._modules)
        fragment.coverage.extend(read_cov)
        selected = side.values
        fragment.entities.extend(
            sqlwrite.entity_nodes(side.entities, self.map_id, self._version, attributor)
        )
        # One node per statement, beside the per-function answer the subject
        # already carries.  A function name says which log to read; the
        # statement says where in the code, which is what a row that is not
        # moving needs -- ``copyArchive.main`` asks for ``jobStatus='holding'``
        # from three statements with three different time limits.
        fragment.read_sites.extend(
            sqlwrite.read_site_nodes(side.statements, self.map_id, self._version)
        )
        self._unreadable_tables = side.unreadable
        self.read_watch = side.watch
        gated = sqlwrite.selection_gates(self._modules, attributor, self._never_written)
        for subject in fragment.subjects:
            readers = selected.get(subject.name, {})
            subject.selected_values = sorted(readers)
            subject.selected_by = {
                value: sorted(use.selected_by)
                for value, use in sorted(readers.items())
                if use.selected_by
            }
            subject.updated_by = {
                value: sorted(use.updated_by)
                for value, use in sorted(readers.items())
                if use.updated_by
            }
            subject.selection_gates = sorted(gated.get(subject.name, ()))

        # A function that writes a status both ways produces one junction from
        # each recognizer under the same name.  Storage merges on the name, so
        # without this the second silently replaces the first's branches.
        lap("the read side")
        fragment.junctions = _merge_junctions(fragment.junctions)
        fragment.subjects = _unique_subjects(fragment.subjects)
        cut = progress.spec_length_limits(self._modules)
        for subject in fragment.subjects:
            if subject.qualifier_kind == "spec":
                subject.truncated_to = cut.get((subject.spec_class, subject.attribute))

        # Which other kind of row one of the owner's queries asks for in the
        # same statement as the rows it decides about.  After the merge because
        # it reads the junction's subject, and restricted to the statement
        # because the owner-wide version is the join this corpus has twice
        # charged for: it would open 113 junctions where the statement opens 31.
        for junction in fragment.junctions:
            here = junction.subject.rpartition(".")[0]
            junction.joined_entities = sorted(
                {
                    entity
                    for group in side.joins.get(junction.owner, ())
                    if here in group
                    for entity in group
                }
                - {here}
            )

        # After the merge, because it annotates branches rather than making
        # them: the decision a knight makes in memory is already a junction,
        # and what this adds is the row its flush needed to land.  Reads the
        # callers with ``audited`` rather than ``attributor`` -- it asks which
        # class an attribute write is on, which is the progress slice's own
        # question, and that slice learns element types before asking it.
        lap("merging")
        self._flush_guards = flush.attach(
            fragment.junctions,
            self._modules,
            audited,
            row_keys=fragment.row_key_sources,
            # Each flush call's records, read from the methods' own the SQL slice made.
            key_froms=fragment.key_froms,
        )
        if self._flush_guards:
            logger.info(
                "PandaCodeMapPlugin: %d branch(es) told what the row had to say for "
                "the flush that persists them",
                self._flush_guards,
            )

        # Promotion last, because criterion 3 reads the extracted branches.
        # Everything the slices found is kept; the criteria say which subjects
        # the code treats as deciding something.
        lap("flush")
        criteria = promotion.criteria_for(
            fragment,
            promotion.gated_fields(self._modules),
            progress.declared_vocabularies(
                self._modules, progress.spec_attributes(self._modules)
            ),
        )
        criteria = promotion.close_over_passthrough(fragment, criteria)
        self._unmarked = promotion.apply(fragment, criteria)

        # A column name the code builds from a mapping key.  Where the keys
        # cannot be read the statement names no column, which in the map looks
        # exactly like a statement that writes none -- so the count is carried
        # to the build report rather than left as that silence.
        self.key_columns = sql.KeyColumnCensus(
            *(
                sum(part)
                for part in zip(
                    *(
                        sql.key_column_census(func)
                        for module in self._modules
                        for func, _owner in functions_with_owner(module.tree)
                    ),
                    strict=True,
                )
            )
        )

        # After promotion, which marks the subjects and drops none: the reach
        # figure is over every junction the slices produced.
        # Classified once and handed to both readers of it: the trigger slice
        # builds entry points from it, and the log sites need the same fact to
        # say whether a reader re-evaluates.
        lap("promotion")
        cadence = trigger.classify(
            self._modules, {b.interface.split(".")[-1] for b in channels}
        )
        self._uplinks = trigger.worker_uplinks(self._modules)
        # The second crossing, and it needs the junctions the first does not:
        # a dispatched worker is recognisable from the corpus alone, while a
        # plain construction is only worth indexing where the class it builds
        # decides something.
        self._built = trigger.construction_uplinks(
            self._modules, fragment.junctions, set(declarations)
        )
        self._reached = trigger.attach(
            fragment.junctions,
            self._modules,
            {b.interface.split(".")[-1] for b in channels},
            _merge_uplinks(self._uplinks, self._built),
        )
        # Why the rest have none.  A bare ratio is one number covering three
        # situations, only one of which is work this slice could do, and the
        # report cannot tell them apart from the ratio alone.
        self._unreached = trigger.unreached_reasons(
            fragment.junctions,
            self._modules,
            {b.interface.split(".")[-1] for b in channels},
        )
        # What each junction consults, as opposed to what starts it.  The one
        # edge out of a junction that a reader can follow without resolving a
        # name, and the only route from an arm to the aggregate it decided on.
        self._consults = trigger.attach_calls(fragment.junctions, self._modules)
        lap("the trigger slice")
        # Who calls each owner, by what the receiver is declared to be -- the
        # other direction from ``calls``, and the one a transition's movers
        # are answered from.
        if self._calls is not None:
            _attach_called_from(fragment.junctions, self._calls)
            fragment.calls_considered, fragment.call_stops, fragment.open_calls = _call_stops(self._calls, self._modules)
            # Where every function can go on to, for the reach a mover is
            # answered from (:mod:`bamboo.codemap.reach`): its own record,
            # because a run asks it of hundreds of movers.
            fragment.reach_graph.extend(
                reach.graph(
                    self._calls,
                    self.map_id,
                    self._version,
                    _DatabaseLeaves(),
                    {module.rel_path: module.blob_sha for module in self._modules},
                )
            )
        lap("callers and the reach record")
        # Where a row a statement keeps is carried on to: after the reach
        # record, whose holes and database calls it reads, over every junction
        # the map keeps.
        # Read by the build's report rather than logged, so that the census
        # appears where the other read-side counts do and nothing changes for
        # the readers that only check the map.  The gap between the middle two
        # is the finding: an empty ``crosses_to`` on a statement that plainly
        # hands its rows on would otherwise read as "these rows go nowhere".
        record = reach.ReachGraph(fragment.reach_graph) if self._calls is not None else None
        # Where the walk reaches the database through calls with a statement's rows in its binds: the KeyFroms'
        # sources across calls, read in the next step.
        landings: dict[str, list] = {}
        self.crossings = sqlwrite.attach_crossings(
            fragment.read_sites,
            self._modules,
            fragment.junctions,
            self._calls,
            record,
            landings=landings,
        )
        lap("crossings")
        # Where each write's row keys were filled from, and who handed a guarded write its keys: read by the carry,
        # which reads a call where it lands, so after the reach record and with it.
        sqlwrite.attach_key_sources(
            fragment.row_key_sources,
            fragment.junctions,
            self._calls,
            record,
            key_froms=fragment.key_froms,
            read_sites=fragment.read_sites,
            # Whose rows a read site's tables hold, for an INSERT's mark: off the attributor that learned the tables.
            table_class=attributor.class_for_table,
            # The objects no write persists: their arms read as the attribute slice reads them.
            modules=self._modules,
            attributor=audited,
            landings=landings,
        )
        # The routes of calls on which each write never runs, its records all made (the objects' last).
        sqlwrite.attach_routes(fragment.key_froms, self._calls)
        lap("key sources")
        # What the schema and the corpus say of each table named, for a reader asking rows: the write records all
        # made, read inside the calls and the schema as the read sites were.
        fragment.tables = tables.records(
            self._modules,
            fragment.entities,
            fragment.read_sites,
            fragment.key_froms,
            attributor,
            attributor.class_for_table,
            self._schema or {},
            self._keys or {},
            self._partitions,
            self._declaring,
        )
        lap("tables")
        # Which classes a run-time dispatch could have picked.  A candidate
        # set, never an attribution: P1-11 refused to name the single class a
        # run-time receiver is, and this says "one of these", which is the
        # shape the map already trades in.
        self._dispatching = trigger.attach_dispatch(fragment.junctions, self._modules)

        # Which file each node's diagnostics land in, for every junction the
        # map keeps, as the trigger reach is.
        self._log_files = logfile.attach(fragment, self._modules, cadence)
        self._declared_files = logfile.declared_files(self._modules)

        lap("dispatch and log files")
        # After the log files, because the file an emit lands in is resolved
        # from the module that writes the line.  Reads the callers with
        # ``audited`` for the reason the flush pass does: it asks which class
        # an attribute belongs to, which is the progress slice's question.
        self._emits = emit.attach(fragment, self._modules, audited)
        if self._emits:
            logger.info(
                "PandaCodeMapPlugin: %d branch(es) carry the line the code writes",
                self._emits,
            )

        lap("emits")
        # Last, because it keys on the anchor every other pass has finished
        # placing.  Chooses the text a reader would be given; reading it is
        # nobody's business at build time.
        self._glossed = reading.attach(fragment, self._modules)

        lap("glosses")
        # What every function prints, as the skeleton reads it with nothing
        # asked: the rows the querier walked each function for.  Counted a
        # second way off the syntax tree, so that a call the skeleton leaves
        # out fails ``prints-are-counted`` instead of reading as a line the
        # code cannot print.
        fragment.function_prints.extend(self._function_prints())
        fragment.print_calls = _print_calls(self._modules)
        # Which of those prints every path to an arm passes, and which of their
        # holes hold a column of the row the arm writes: what tells the run that
        # wrote a value from the runs that only printed the same lines.
        self._must_prints = mustprints.attach(
            fragment.junctions, fragment.function_prints, self._modules, self._declarations
        )
        logger.info("PandaCodeMapPlugin: %d branch(es) have a print every path to them passes", self._must_prints)
        lap("prints")
        if self._calls is not None:
            # Where each function puts the objects it names and which of their
            # columns it assigns: what bamboo.codemap.objects.follow answers
            # from, and what a history pair's row below is read with.
            assigned, bound, handed, written = objects.attach(fragment, self._modules, self._calls)
            logger.info(
                "PandaCodeMapPlugin: %d assign(s), %d binding(s), %d call binding(s) of objects, %d write(s)",
                assigned,
                bound,
                handed,
                written,
            )
            # Where each function's runs start, what each call of a history writer
            # can record, and the calls that write a history row in the same pass
            # as an arm: what gives a value the map sees set a time, and the log
            # lines of the run it was set in.
            started, recorded, paired = runs.attach(fragment, self._modules, self._calls)
            logger.info(
                "PandaCodeMapPlugin: %d function(s) start a run, %d history call(s), %d branch(es) with a history pair",
                started,
                recorded,
                paired,
            )

        lap("objects and runs")
        logger.info(
            "PandaCodeMapPlugin: %d enumeration(s), %d boundary/boundaries, "
            "%d subject(s), %d junction(s)",
            len(enums),
            len(boundaries),
            len(fragment.subjects),
            len(fragment.junctions),
        )
        return fragment

    def _function_prints(self) -> list[FunctionPrintsNode]:
        """One node per function that prints, read by :func:`trace.function_prints`.

        The walk reads each file again, from the same root, and a file whose
        text differs from the build's reading of it means the corpus moved
        under the build: the build is void, and saying so beats storing rows
        from one tree beside nodes from another.
        """
        shas = {module.rel_path: module.blob_sha for module in self._modules}
        declared = logfile.declared_files(self._modules)
        inherited = logfile.inherited_files(self._modules, declared)
        calls = getattr(self, "_calls", None)
        landings = _print_landings(
            self._modules, logfile.LoggerFiles(calls, declared, inherited) if calls is not None else None
        )
        nodes: list[FunctionPrintsNode] = []
        for found in trace.function_prints(self._roots, list(shas)):
            if found.blob_sha != shas[found.file]:
                raise RuntimeError(
                    f"{found.file} changed while the map was being built; "
                    "build again from a tree nothing writes to"
                )
            own = logfile.files_of(found.file, declared, inherited)
            for statement in found.statements:
                landing = landings.get((found.owner, statement.line))
                statement.log_files = landing.files if landing else list(own)
                statement.log_files_basis = landing.basis if landing else LOG_FILES_OF_MODULE
            nodes.append(
                FunctionPrintsNode(
                    name=FunctionPrintsNode.make_name(self.map_id, found.owner),
                    map_id=self.map_id,
                    derived_from=self._version,
                    owner=found.owner,
                    cls=found.cls,
                    function=found.func.name,
                    count=len(found.statements),
                    statements=found.statements,
                    log_files=sorted({file for statement in found.statements for file in statement.log_files}),
                    anchor=Anchor(
                        package=found.file.split("/", 1)[0],
                        file=found.file,
                        line_start=found.func.lineno,
                        line_end=found.func.end_lineno,
                        blob_sha=found.blob_sha,
                    ),
                )
            )
        return nodes

    @property
    def emits(self) -> int:
        """Branches given the log line the code writes when they fire."""
        return self._emits

    @property
    def glossed(self) -> int:
        """Nodes whose enclosing function the map could locate and key."""
        return self._glossed

    @property
    def flush_guards(self) -> int:
        """Branches told what the row had to say for their flush to land."""
        return self._flush_guards

    @property
    def module_count(self) -> int:
        """How many modules were analysed."""
        return len(self._modules or [])

    @property
    def excluded_module_count(self) -> int:
        """How many test modules were left out of the analysis."""
        return getattr(self, "_skipped_tests", 0)

    @property
    def silent_cuts(self) -> int:
        """Guarded ``continue``s in a loop that drop a candidate without saying why."""
        return getattr(self, "_silent_cuts", 0)

    @property
    def stop_disagreements(self) -> list[str]:
        """Functions whose stops the answer would read differently from the build -- see ``_stops_read_again``."""
        return self._stop_disagreements

    @property
    def unexplained_steps(self) -> list[str]:
        """Funnel steps that count a cut the slice could not find a reason for."""
        return getattr(self, "_unexplained_steps", [])

    @property
    def unexplained_writes(self) -> list:
        """Written columns the sql-write slice counted and could not account for.

        Each carries the reason the branch that gave up recorded, and the name of
        the junction it would have joined -- unresolved here on purpose.  Whether
        that junction exists is a question about the whole fragment, so it is
        answered where the fragment is, rather than by this plugin holding a
        second copy of what it built.
        """
        return getattr(self, "_unexplained_writes", [])

    @property
    def unexplained_reads(self) -> list:
        """Statements the sql-read slice counted and did not see run.

        Its reasons name the call form the reader declined, not a value's
        origin: this slice's denominator exists so that a query form nobody
        taught it cannot go missing silently, and a count alone says a form is
        missing without saying which one.
        """
        return getattr(self, "_unexplained_reads", [])

    @property
    def trigger_reach(self) -> tuple[int, int]:
        """``(junctions with an entry point, total)``."""
        return getattr(self, "_reached", (0, 0))

    @property
    def unreached_junctions(self) -> dict[str, list]:
        """``{why it has no entry point: junctions}``."""
        return getattr(self, "_unreached", {})

    @property
    def consulting(self) -> int:
        """Junctions whose owner calls something on ``self``."""
        return getattr(self, "_consults", 0)

    @property
    def log_file_reach(self) -> tuple[int, int]:
        """``(nodes whose log file is known, total)``."""
        return getattr(self, "_log_files", (0, 0))

    @property
    def worker_uplinks(self) -> dict[tuple[str, str], list]:
        """Which ``(module, method)`` a dispatched worker's entry reaches.

        Kept from the build so the report does not walk the tree a second time
        for a figure the attachment already had in hand.
        """
        return getattr(self, "_uplinks", {})

    @property
    def worker_classes(self) -> int:
        """How many dispatched-worker classes the corpus declares.

        The denominator the uplink count is silent without: no uplinks and no
        worker classes is a corpus that dispatches nothing, and no uplinks with
        nine of them is the reader having lost the door.
        """
        return len(trigger.worker_classes(self._modules)) if self._modules else 0

    @property
    def declared_log_files(self) -> dict[str, str]:
        """``{module: log filename}`` for modules that declare their own logger.

        Exposed so a caller can work out which machine group writes a file --
        the module that declares the logger is the one that names it, and its
        package is what places it.
        """
        return getattr(self, "_declared_files", {})

    @property
    def unpromoted(self) -> tuple[int, int]:
        """``(subjects, junctions)`` no promotion criterion marks -- kept, with empty ``criteria``."""
        return getattr(self, "_unmarked", (0, 0))

    @property
    def uncovered_tables(self) -> set[str]:
        """Tables written by the code that hold no spec class."""
        return getattr(self, "_uncovered_tables", set())

    @property
    def unreadable_tables(self) -> dict[str, int]:
        """``{owner: statements whose table name the reading could not read}``.

        Apart from :attr:`uncovered_tables`, which is about tables that hold no
        spec: this is about statements whose table nobody could name at all.
        Pooled into that list they read as one finding, and only one of the two
        is a gap in the extraction.
        """
        return getattr(self, "_unreadable_tables", {})

    @property
    def table_conflicts(self) -> dict[str, set[str]]:
        """Tables whose column evidence named more than one spec class."""
        return getattr(self, "_table_conflicts", {})

    @property
    def tagged_branches(self) -> int:
        """Branches that name a machine-readable tag for their own decision."""
        return getattr(self, "_tagged_branches", 0)

    @property
    def unclaimed_tags(self) -> list[tuple[str, list[str]]]:
        """Tags named where subjects are settled that no branch claims.

        The second reading of the same fact: the code names an action, and the
        branch table says which branch takes it.  A tag with no claimant is
        either a decision this slice does not model or a message about
        something the code no longer does -- the corpus has one of the latter.
        """
        return getattr(self, "_unclaimed_tags", [])

    @property
    def runtime_dispatch_writes(self) -> list[str]:
        """Writes dropped because the receiver's class is chosen at run time.

        Reported rather than left silent.  These are not junctions -- nothing
        says the object is a spec -- but the reason is an inference of this
        reader's rather than something the code states, and a drop nobody can
        see is how ``unresolved`` came to absorb "not read" once before.
        """
        return getattr(self, "_runtime_dispatch", [])

    # -- source discovery ------------------------------------------------- #

    @staticmethod
    def _resolve_roots(source_root: Optional[Path]) -> dict[str, Path]:
        """Return ``{package: root}``.

        With *source_root* the packages come from that directory, which is how
        a named release is mapped instead of whatever happens to sit next to
        bamboo.  Otherwise the installed distribution is used, resolved via
        ``importlib.metadata`` rather than ``sys.modules`` so that a
        monkey-patched import cannot redirect it.
        """
        roots: dict[str, Path] = {}
        if source_root is not None:
            base = Path(source_root).expanduser().resolve()
            for pkg in PACKAGES:
                if (base / pkg).is_dir():
                    roots[pkg] = base / pkg
            return roots

        from importlib.metadata import (
            Distribution,
            PackageNotFoundError,
            packages_distributions,
        )

        pkg_to_dist = packages_distributions()
        for pkg in PACKAGES:
            for dist_name in pkg_to_dist.get(pkg, []):
                try:
                    dist = Distribution.from_name(dist_name)
                except PackageNotFoundError:
                    continue
                pkg_dir = Path(dist.locate_file("")).resolve() / pkg
                if pkg_dir.is_dir():
                    roots[pkg] = pkg_dir
                    break
        return roots

    @staticmethod
    def _resolve_version(source_root: Optional[Path]) -> str:
        """Return the stamp recorded on every node of this build.

        The stamp has to identify the *contents* that were analysed.  A
        filesystem path does not: the same path holds different code tomorrow,
        so two builds would carry the same stamp while describing different
        systems -- which defeats the point of stamping at all, and silently
        breaks the version diffing that condition drift depends on.

        For a git checkout the description is used instead.  ``--dirty``
        matters: a map built from a modified working tree cannot be
        reproduced, and a stamp that hides that is worse than no stamp.
        """
        if source_root is not None:
            root = Path(source_root).expanduser().resolve()
            described = _git_describe(root)
            return f"git:{described}" if described else f"source-root:{root}"

        from importlib.metadata import (
            Distribution,
            PackageNotFoundError,
            packages_distributions,
        )

        for dist_name in packages_distributions().get(PACKAGES[0], []):
            try:
                return f"{dist_name} {Distribution.from_name(dist_name).version}"
            except PackageNotFoundError:
                continue
        return "unknown"

    @staticmethod
    def _parse_modules(roots: dict[str, Path]) -> tuple[list[SourceModule], int]:
        """Parse and hash every ``.py`` once; files that will not parse are skipped.

        Returns the modules and how many test modules were left out, because the
        exclusion is a decision the report should state rather than a detail in
        a log line.

        Recognizers are run over the whole tree rather than a file list, so a
        module added upstream is picked up on the next build with no
        configuration change.

        The content hash is of what was actually read, not of what git has
        committed -- in a modified working tree those differ, and an anchor
        that pointed at the committed bytes would describe code the build never
        saw.
        """
        modules: list[SourceModule] = []
        skipped_tests = 0
        for pkg, root in roots.items():
            for path in sorted(root.rglob("*.py")):
                rel_path = f"{pkg}/{path.relative_to(root)}"
                if _is_test_module(rel_path):
                    skipped_tests += 1
                    continue
                # The process's one tree for the file, so the build and the
                # walks it makes hold the same nodes -- see ``calls.parsed``.
                held = calls_mod.parsed(path)
                if held is None:
                    logger.debug("skipping unparseable module: %s", path)
                    continue
                source, tree = held
                modules.append(
                    SourceModule(
                        package=pkg,
                        rel_path=rel_path,
                        tree=tree,
                        source=source,
                        blob_sha=blob_sha(source),
                    )
                )
        if skipped_tests:
            # Spelled out because the bare count read as a warning.  Test code
            # is not the system's behaviour: a status a test assigns is not a
            # transition PanDA makes, and including them put most of the
            # ambiguous write sites in the corpus.
            logger.info(
                "excluded %d test module(s) from analysis "
                "(test code is not the system's behaviour)",
                skipped_tests,
            )
        return modules, skipped_tests


#: A logger's methods.  What a log call lands in says nothing about a value,
#: and counting them would make a third of the calls the stops are drawn from.
_LOG_CALLS = frozenset({"debug", "info", "warning", "error", "critical", "exception"})


#: The classes PanDA's proxies talk to the database through.
_DRIVER_CLASSES = frozenset({"WrappedCursor", "WrappedOracleConn", "WrappedPostgresConn"})
#: The names a cursor or a connection is held under, a field or a local --
#: ``provenance.DATABASE_INTERFACES`` names ``cur`` the same way.
_DRIVER_NAMES = frozenset({"cur", "conn"})


class _DatabaseLeaves:
    """Which calls are PanDA's database, for :func:`reach.graph` (:data:`reach.Leaves`).

    A statement the map reads is read at the call that runs it
    (:func:`sql.executions`), and a write's arms are in the function making
    that call: the call is the database, and the closure goes no further --
    unless the runner runs statements of its own as well (:attr:`sql.Runner.own`:
    ``queryTasksToBePending_JEDI`` updates the tasks the ``SELECT`` it is handed
    picks), and is followed as any call.  A runner handing on the statement
    its function was handed is the database too: the statement is its
    callers', read where they run it (:data:`sql.SKIP_CALLERS`).
    A runner call it read nothing out of may be a write, and is a hole.  A
    call on the driver -- a cursor or a connection, ``fetchall``,
    ``commit`` -- goes only to the database; but a runner call on it whose
    statement nobody read is no leaf, because what it runs is read nowhere.
    """

    def __init__(self) -> None:
        self._read: dict[int, tuple[ast.AST, set[int], dict[int, str]]] = {}

    def __call__(self, file: str, func: ast.AST, call: ast.Call, placed) -> Optional[tuple[str, str]]:
        called = call.func
        runners = sql.runners()
        if isinstance(called, ast.Attribute) and called.attr in runners:
            executed, skipped = self._of(func)
            if id(call) in executed:
                if runners[called.attr].own:
                    return None
                return "database", "runner_runs_only_sql"
            if call.lineno in skipped:
                if skipped[call.lineno] == sql.SKIP_CALLERS:
                    # ``JediTaskBuffer.queryTasksToPreassign_JEDI`` hands its caller's statement on to the proxy's,
                    # which updates the tasks it picks: followed, as a run of it is.
                    return None if runners[called.attr].own else ("database", "runner_runs_callers_sql")
                return "hole", f"the statement is not read: {skipped[call.lineno]}"
            if _on_driver(call, placed):
                return "hole", "the statement this runs is read nowhere"
            return None
        if _on_driver(call, placed):
            return "database", "driver_talks_only_to_db"
        return None

    def _of(self, func: ast.AST) -> tuple[set[int], dict[int, str]]:
        held = self._read.get(id(func))
        if held is None or held[0] is not func:
            skipped: list = []
            executed: set[int] = set()
            if isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
                try:
                    executed = {id(one.call) for one in sql.executions(func, skipped)}
                except Exception as error:  # noqa: BLE001 -- every runner call here is then a hole
                    logger.warning("reach: executions failed in %s: %s", getattr(func, "name", "?"), error)
            held = (func, executed, {skip.line: skip.reason for skip in skipped})
            self._read[id(func)] = held
        return held[1], held[2]


def _on_driver(call: ast.Call, placed) -> bool:
    """Whether *call* is made on a cursor or a connection: held as ``cur`` / ``conn``, or placed at a driver class's method."""
    root = call.func.value if isinstance(call.func, ast.Attribute) else None
    while isinstance(root, (ast.Attribute, ast.Subscript, ast.Call)):
        if (
            isinstance(root, ast.Attribute)
            and root.attr in _DRIVER_NAMES
            and isinstance(root.value, ast.Name)
            and root.value.id in ("self", "cls")
        ):
            return True
        root = root.func if isinstance(root, ast.Call) else root.value
    if isinstance(root, ast.Name) and root.id in _DRIVER_NAMES:
        return True
    return any(target.cls is not None and target.cls.name in _DRIVER_CLASSES for target in placed.targets)


def _call_stops(calls, modules: list[SourceModule]) -> tuple[int, list[CallStop], list[CallStop]]:
    """``(considered, stops, opened)``: the build's calls to its own functions, those ``calls`` cannot place, and those it places open.

    A call placed open is not a stop -- the walk enters the definitions it
    names -- but it may land somewhere else as well, and before it had a list
    of its own the 452 of them that X4 placed open simply left the stops.

    Drawn from the calls whose name some function or class of the build's
    modules defines -- a call to ``json.loads`` names nothing the walk could
    enter, so it is not a stop.  Over the build's modules only, test code
    left out by the build's own rule, and over the calls in functions: one in
    a module's own body is where a walk ends, not a call it goes into.
    """
    files = {module.rel_path for module in modules}
    defined = {
        node.name
        for module in modules
        for node in walk(module.tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    considered = 0
    stops: list[CallStop] = []
    opened: list[CallStop] = []
    for name, sites in sorted(calls.every_call().items()):
        if name not in defined or name in _LOG_CALLS:
            continue
        for site in sites:
            if site.file not in files or site.in_module:
                continue
            considered += 1
            placed = calls.callees(site.file, site.func, site.call)
            if placed.targets and not placed.unplaced:
                continue
            (opened if placed.targets else stops).append(
                CallStop(
                    reason=placed.unplaced,
                    file=site.file,
                    line=site.call.lineno,
                    owner=site.owner,
                    callee=ast.unparse(site.call.func)[:80],
                )
            )
    stops.sort(key=lambda stop: (stop.file, stop.line, stop.callee))
    opened.sort(key=lambda stop: (stop.file, stop.line, stop.callee))
    return considered, stops, opened


def _attach_called_from(junctions, calls) -> None:
    """Fill ``JunctionNode.called_from`` for every junction, one owner at a time."""
    from bamboo.codemap.calls import Target

    by_owner: dict[str, list[str]] = {}
    for junction in junctions:
        if junction.owner not in by_owner:
            file, _, name = junction.owner.partition("::")
            tree = calls.types.tree(file)
            found: list[str] = []
            for func in walk(tree) if tree is not None else ():
                if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)) or func.name != name:
                    continue
                if junction.anchor is not None and not (
                    func.lineno <= junction.anchor.line_start <= (func.end_lineno or func.lineno)
                ):
                    continue
                owned = calls.types.method_class(file, func)
                found = calls.called_from(Target(file, func, owned[1] if owned else None))
                break
            by_owner[junction.owner] = found
        junction.called_from = by_owner[junction.owner]


def _stops_read_again(modules, roots, stops) -> list[str]:
    """``owner: build [...] / answer [...]`` for each function the two readings of its stops disagree on.

    The answer does not store the stops: ``derive-strategy`` reads a mover's
    statements again with :func:`sqlwrite.statement_stops`, over a reader of
    the one file it was asked about (``strategy._mover_gaps``).  Two readings
    of one thing are how two counts come to disagree, so the build reads
    every function that way as well and says where they part.  The row
    precondition stops are the write's, and ``statement_stops`` leaves them to
    ``extract`` -- they are not compared.
    """
    from bamboo.codemap.calls import Calls, Types

    built: dict[str, set[tuple]] = {}
    for stop in stops:
        if stop.reason == sqlwrite.PRECONDITION_BIND_UNSETTLED:
            continue
        built.setdefault(stop.owner, set()).add((stop.reason, stop.detail, stop.line))
    found: list[str] = []
    for module in modules:
        calls = Calls(Types(roots, {module.rel_path: (module.blob_sha, module.tree)}))
        with sql.across_calls(calls):
            for func, _cls in functions_with_owner(module.tree):
                owner = f"{module.rel_path}::{func.name}"
                again = {
                    (stop.reason, stop.detail, stop.line)
                    for stop in sqlwrite.statement_stops(func, module.rel_path)
                }
                if again != built.get(owner, set()) and (again or owner in built):
                    found.append(
                        f"{owner}: build {sorted(built.get(owner, set()) - again)[:2]}"
                        f" / answer {sorted(again - built.get(owner, set()))[:2]}"
                    )
    return sorted(set(found))
