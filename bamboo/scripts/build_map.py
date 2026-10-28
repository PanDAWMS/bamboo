"""``bamboo build-map`` — extract a Code Map from a target system's source.

An offline build step, one per release of the target: a build parses the whole
source tree, and a Code Map's lifetime is tied to that *code version*, not to an
incident, which is what makes its stamp meaningful.

The map is kept as a dump -- the whole fragment, written to a file -- and not in
the graph database.  Every reading of it is a lookup in the dump itself
(:class:`~bamboo.codemap.lookup.CodeMap` on :func:`~bamboo.codemap.store.load_dump`),
with no database between; and each case is read against the map of the
corpus that wrote its log, one file per pin, where the database held one build at
a time.  Neo4j keeps the incident graph only.

Usage::

    # Build from the installed distribution, and report
    bamboo build-map

    # Build a specific release from a checkout, and keep it
    bamboo build-map --source-root /path/to/panda --dump map.json
"""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Optional

import click

from bamboo.codemap import gates
from bamboo.codemap.factory import available_map_ids, get_code_map_plugin
from bamboo.codemap.models import SCOPE_OF_STOPS, MapFragment
from bamboo.codemap.panda import sql, sqltext
from bamboo.codemap.panda.attribution import AMBIGUOUS_FORM

logger = logging.getLogger(__name__)


def _report(
    fragment: MapFragment,
    results: list[gates.GateResult],
    top: int,
    silent_cuts: int = 0,
) -> bool:
    """Print the build summary and return True when every gate passed."""
    click.echo(f"\nmap_id       : {fragment.map_id}")
    click.echo(f"derived_from : {fragment.derived_from}")

    totals = gates.slice_totals(fragment)
    if totals:
        click.echo("\nslice coverage:")
        for slice_name, counts in sorted(totals.items()):
            candidates = counts["candidates"]
            explained = counts["explained"]
            pct = f"{explained * 100 // candidates}%" if candidates else "-"
            click.echo(
                f"  {slice_name:<14} {explained:>5}/{candidates:<5} ({pct:>4})  "
                f"in {counts['files']} file(s)"
            )

    rows = gates.coverage_matrix(fragment)
    weak = [r for r in rows if r[4] < 1.0]
    if weak:
        # Files where candidates were found but not explained use a different
        # idiom; naming them is the point, since an unreported gap reads as
        # full coverage.
        click.echo(f"\nlowest-coverage files ({len(weak)} below 100%):")
        for slice_name, file, candidates, explained, ratio in weak[:top]:
            click.echo(
                f"  {ratio:6.0%}  {slice_name:<12} {explained:>4}/{candidates:<4} {file}"
            )
        if len(weak) > top:
            click.echo(f"  … {len(weak) - top} more")

    if fragment.junctions:
        bases = Counter(j.attribution for j in fragment.junctions)
        total = sum(bases.values())
        click.echo("\njunction attribution:")
        for basis in ("certain", "container", "structural", "unresolved"):
            count = bases.get(basis, 0)
            click.echo(f"  {basis:<12} {count:>5}  ({count * 100 // total if total else 0}%)")
        open_forms = sorted(
            f"{where} {form.removeprefix(AMBIGUOUS_FORM)}"
            for where, form in fragment.spec_annotation_forms.items()
            if form.startswith(AMBIGUOUS_FORM)
        )
        if open_forms:
            # A form the reader understands and declines to answer.  Reported
            # here rather than in the gate's note, which only prints on a
            # failure -- and these are on the passing side by design, so the
            # note would be the one place nobody sees them.
            click.echo(f"  annotations naming a spec and left open ({len(open_forms)}):")
            for row in open_forms[:top]:
                click.echo(f"    {row}")
            if len(open_forms) > top:
                click.echo(f"    … {len(open_forms) - top} more")

        thin = gates.unresolved_attributes(fragment)
        if thin:
            # Each of these is a candidate for one line of type annotation
            # upstream, which is the intended remedy -- cheaper and more honest
            # than another inference rule here.
            click.echo("  attributes with unresolved writes (annotation candidates):")
            for attribute, writes in thin[:top]:
                click.echo(f"    {attribute:<20} {writes} write(s)")
            if len(thin) > top:
                click.echo(f"    … {len(thin) - top} more")

        sinks = gates.unreachable_values(fragment)
        if sinks:
            # The plan's "in-edges but no out-edge" invariant, as a report:
            # a terminal status is supposed to be a sink and nothing in the
            # source says which ones those are.
            click.echo(f"  written but nothing selects rows on ({len(sinks)} subject(s)):")
            for subject, values in sinks[:top]:
                click.echo(f"    {subject:<30} {values}")
            if len(sinks) > top:
                click.echo(f"    … {len(sinks) - top} more")

        outside = gates.carried_from_outside(fragment)
        if outside:
            # A terminal, in the same sense as a boundary: the walk stops and
            # the answer is complete rather than missing.
            click.echo(f"  carried from a field the map does not explain ({len(outside)}):")
            for subject, source in outside[:top]:
                click.echo(f"    {subject} <- {source}")

        gated = sorted(
            {gate for subject in fragment.subjects for gate in subject.selection_gates}
        )
        if gated:
            # The same shape as the line above, on the read side: a query's
            # reach depends on a table nothing here keeps current, so a row
            # can be missed for a reason no branch condition mentions.
            click.echo(f"  queries bounded by a table nothing here writes ({len(gated)}):")
            for table in gated[:top]:
                users = sorted(
                    s.name for s in fragment.subjects if table in s.selection_gates
                )
                click.echo(f"    {table:<34} gates {len(users)} subject(s)")

        raced = sorted(
            {
                junction.subject
                for junction in fragment.junctions
                for branch in junction.branches
                if branch.row_precondition
            }
        )
        if raced:
            # Neither a defect nor a gate: a compare-and-set is how PanDA keeps
            # two knights off one row.  Shown because losing that race is the
            # one failure the code announces and the row does not take, so a
            # log line saying the value was set is not proof that it was.
            writes = sum(
                1
                for junction in fragment.junctions
                for branch in junction.branches
                if branch.row_precondition
            )
            click.echo(
                f"  writes conditional on the row's own prior value "
                f"({writes} branch(es) over {len(raced)} subject(s)):"
            )
            for subject in raced[:top]:
                guards = sorted(
                    {
                        guard
                        for junction in fragment.junctions
                        if junction.subject == subject
                        for branch in junction.branches
                        for guard in branch.row_precondition
                    }
                )
                click.echo(f"    {subject:<34} {', '.join(guards[:3])}")
            if len(raced) > top:
                click.echo(f"    … {len(raced) - top} more")

        drift = gates.outcomes_outside_declared_subsets(fragment)
        if drift:
            # Not a gate: the declared lists are purpose-built subsets, so an
            # outcome outside them is normal.  Shown because a value no
            # declaration mentions is still worth a glance.
            click.echo(f"  outcomes no declared list mentions ({len(drift)}):")
            for subject, outcome, where in drift[:top]:
                click.echo(f"    {subject} = {outcome!r}  ({where})")
            if len(drift) > top:
                click.echo(f"    … {len(drift) - top} more")

    if fragment.filter_stages:
        chains = Counter(s.owner for s in fragment.filter_stages)
        click.echo(f"\nfilter stages: {len(fragment.filter_stages)} in {len(chains)} chain(s)")
        for owner, count in chains.most_common(top):
            click.echo(f"  {count:>3}  {owner}")
        blind = gates.unexplainable_rejections(fragment)
        if blind:
            # What a rejection did not log cannot be checked afterwards, so
            # the map should not promise "read the log" for these.
            click.echo(
                f"  rejections logging none of what they tested ({len(blind)}):"
            )
            for tag, inputs, where in blind[:top]:
                click.echo(f"    {tag:<22} tests {inputs}  ({where})")
            if len(blind) > top:
                click.echo(f"    … {len(blind) - top} more")

    if fragment.loop_cuts:
        owners = Counter(c.owner for c in fragment.loop_cuts)
        scoped = sum(1 for c in fragment.loop_cuts if c.scope_prefix)
        placed = sum(1 for c in fragment.loop_cuts if c.log_files)
        click.echo(
            f"\nloop cuts: {len(fragment.loop_cuts)} in {len(owners)} loop-holding "
            f"function(s) · {placed} with a log file · {scoped} scoped by a logger prefix"
        )
        if silent_cuts:
            # No ratio: most guarded ``continue``s in the corpus are iteration
            # control in pure computation rather than filtering, so they are the
            # wrong denominator.  The count still belongs on the page -- it says
            # how much of the cutting happens with the map unable to name it.
            click.echo(
                f"  {silent_cuts} more drop a candidate without logging a reason"
            )
        for owner, count in owners.most_common(top):
            click.echo(f"  {count:>3}  {owner}")
        if len(owners) > top:
            click.echo(f"    … {len(owners) - top} more")
        ambiguous = gates.ambiguous_cut_keys(fragment)
        if ambiguous:
            # Not a defect: the cut and its guard are right, and only a
            # production question built on the wording is weak.  Named because
            # a hit on such a key cannot say which cut fired, and a miss cannot
            # say none did -- silence that is not about this cut at all.
            click.echo(
                f"  search keys that also match another cut in the same file "
                f"({len(ambiguous)}):"
            )
            for line in ambiguous[:top]:
                click.echo(f"    {line}")
            if len(ambiguous) > top:
                click.echo(f"    … {len(ambiguous) - top} more")

    readable = [n for n in list(fragment.junctions) + list(fragment.filter_stages) if n.gloss_key]
    unreadable = [
        n for n in list(fragment.junctions) + list(fragment.filter_stages) if not n.gloss_key
    ]
    if readable:
        shared = Counter(n.gloss_key for n in readable)
        click.echo(
            f"\ncode to read: {len(readable)} node(s) in {len(shared)} function(s)"
        )
        click.echo(
            f"  most shared: {shared.most_common(1)[0][1]} node(s) in one function "
            "-- read once, every arm in it is explained together"
        )
        if unreadable:
            # Not a failure to fix: a write at module scope has no enclosing
            # function, so there is nothing to hand a reader.  Said out loud
            # because a silent zero is how "not read" hides inside "no key".
            where = Counter(n.owner for n in unreadable)
            click.echo(f"  no enclosing function ({len(unreadable)}):")
            for owner, count in where.most_common(top):
                click.echo(f"    {count:>3}  {owner}")

    if fragment.diagnostics:
        fields = Counter(d.field for d in fragment.diagnostics)
        distinct = {d.template for d in fragment.diagnostics}
        click.echo(
            f"\ndiagnostic templates: {len(fragment.diagnostics)} "
            f"({len(distinct)} distinct) in {len(fields)} field(s)"
        )
        for field, count in fields.most_common(top):
            click.echo(f"  {count:>3}  {field}")
        if len(fields) > top:
            click.echo(f"    … {len(fields) - top} more field(s)")
        # Said out loud because it is the one part of the map promotion does
        # not filter: these are here precisely because their fields are not
        # subjects, and a reader who assumed otherwise would expect a junction.
        click.echo(
            "  outside promotion: a free-text field is not a subject, and this "
            "answers\n  \"who wrote this line\" rather than \"why is it this value\""
        )

    if fragment.enumeration_writes:
        bound: dict[str, set[str]] = {}
        for write in fragment.enumeration_writes:
            bound.setdefault(write.field, set()).add(write.namespace)
        constants = {w.constant for w in fragment.enumeration_writes}
        click.echo(
            f"\nenumeration bindings: {len(fragment.enumeration_writes)} write(s) "
            f"of {len(constants)} constant(s) into {len(bound)} field(s)"
        )
        for field, spaces in sorted(bound.items()):
            click.echo(f"  {field}  <-  {', '.join(sorted(spaces))}")
        # The same note the templates carry, for the same reason, plus what
        # this one buys: values are reused across enumerations, so a code seen
        # in a record cannot be decoded without knowing which field held it.
        click.echo(
            "  outside promotion: an index, not a claim that the field is a "
            "subject.\n  It is what makes a value observed in a record decodable, "
            "since the\n  index is keyed on (namespace, value) and values repeat "
            "across namespaces"
        )

    if fragment.boundaries:
        systems = Counter(b.system for b in fragment.boundaries)
        click.echo(
            "\nboundaries: "
            + ", ".join(f"{system}={count}" for system, count in systems.most_common())
        )
        sent = [b for b in fragment.boundaries if b.direction == "outbound"]
        if sent:
            verbs = Counter(verb for b in sent for verb in b.verbs)
            click.echo(
                f"  outbound ({len(sent)} method(s), {sum(len(b.call_sites) for b in sent)} call site(s)): "
                + ", ".join(f"{verb}={count}" for verb, count in verbs.most_common())
                + "  -- verbs by the far side's method names"
            )
        shared = [b for b in fragment.boundaries if b.transport == "shared_table"]
        if shared:
            # Listed rather than counted: a shared table is investigated by
            # querying it, so the columns and the verbs are the investigation.
            # "DELETE, INSERT" on a command table is a finding on its own.
            click.echo(f"  through a shared table ({len(shared)}):")
            for channel in shared:
                click.echo(
                    f"    {channel.interface}  [{', '.join(channel.operations)}]"
                    f"  in {len(channel.carried_values)} / out {len(channel.handed_over)}"
                )

        thin = gates.unobservable_boundaries(fragment)
        if thin:
            # What crossed a boundary and was never logged cannot be recovered
            # afterwards, so the limit is worth knowing before an incident
            # turns on it rather than during one.
            click.echo(f"  logging under half of what they receive ({len(thin)}):")
            for interface, observed, carried in thin[:top]:
                click.echo(f"    {observed:>3}/{carried:<3}  {interface}")
            if len(thin) > top:
                click.echo(f"    … {len(thin) - top} more")

    click.echo("\ngates:")
    all_passed = True
    for result in results:
        click.echo(f"  {result.summary()}")
        if not result.passed:
            all_passed = False
            for failure in result.failures[:top]:
                click.echo(f"      - {failure}")
            if len(result.failures) > top:
                click.echo(f"      … {len(result.failures) - top} more")
        if result.note and not result.passed:
            # A note explains why a failure matters; on a passing gate it is
            # prose nobody needs, and it made the least interesting gates as
            # loud as the ones that found something.
            click.echo(f"      note: {result.note}")
    return all_passed


def _readers(fragment: MapFragment) -> set[str]:
    """Functions the map records as selecting rows of anything.

    Both resolutions, because both are a read: a predicate on a promoted
    attribute says which rows and what was asked of them, and an entity read
    says only which rows.  Counting the first alone made the helper that
    fetches a task's jobs on the join key look like a function that consults
    nothing.
    """
    return {
        owner
        for subject in fragment.subjects
        for owners in (subject.selected_by or {}).values()
        for owner in owners
    } | {owner for entity in fragment.entities for owner in entity.read_by}


def _report_triggers(fragment: MapFragment, plugin: object, top: int) -> None:
    """Print how junctions are reached, and what follows from it.

    Plugin-specific, like the table findings: what counts as an entry point is
    a fact about how the target system is deployed, not about the map model.
    """
    from bamboo.codemap.panda.recognizers import trigger

    reached, total = getattr(plugin, "trigger_reach", (0, 0))
    if not total:
        return
    kinds = Counter(e.trigger for j in fragment.junctions for e in j.entry_points)
    click.echo(
        f"\nentry points: {reached}/{total} junction(s) reached  ("
        + ", ".join(f"{k}={v}" for k, v in kinds.most_common())
        + ")"
    )

    # Why the others have none.  The log-file gap is already enumerated below;
    # this one was a ratio, which reads as one backlog of the same kind when
    # it is three different answers -- and the largest of them is the one-hop
    # rule working, not a backlog at all.  Saying so here is what keeps the
    # next reader from "fixing" it.
    unreached = getattr(plugin, "unreached_junctions", {})
    if unreached:
        click.echo(f"  no entry point ({total - reached}), by what the map could tell:")
        for reason, junctions in sorted(
            unreached.items(), key=lambda item: (-len(item[1]), item[0])
        ):
            click.echo(f"    {len(junctions):>4} {reason}")
            by_module = Counter(j.owner.split("::")[0] for j in junctions)
            for module, count in by_module.most_common(3):
                click.echo(f"           {module:<54} {count}")
            if len(by_module) > 3:
                click.echo(f"           … {len(by_module) - 3} more module(s)")
        if trigger.CALLER_STARTS_NOTHING in unreached:
            click.echo(
                f"    {trigger.CALLER_STARTS_NOTHING!r} is the one-hop rule in"
                " recognizers/trigger.py holding, not a backlog: reach is a"
                " name match, and chaining name matches compounds them into a"
                " claim"
            )

    # What each junction consults, which is the edge outward rather than the
    # edges inward.  Reported because an empty ``calls`` has to be readable as
    # "this owner calls nothing on itself" and not as "nobody looked" -- a
    # label absorbing the second meaning is how 27 writes once sat unexamined
    # under ``unresolved``.
    consulting = getattr(plugin, "consulting", 0)
    readers = _readers(fragment)
    reaching = sum(
        1 for j in fragment.junctions if any(target in readers for target in j.calls)
    )
    crossing = sum(
        1
        for j in fragment.junctions
        if any(
            not target.startswith(j.owner.partition("::")[0] + "::") for target in j.calls
        )
    )
    click.echo(
        f"consults: {consulting}/{total} junction(s) call something, "
        f"{crossing} of them into another module, "
        f"{reaching} reaching a function that selects rows"
    )

    # Where to look, which is a different question from what starts it: the
    # proxy mixins declare no logger and the knights that call them do.
    own = [j for j in fragment.junctions if j.log_files]
    borrowed = [j for j in fragment.junctions if not j.log_files and j.caller_log_files]
    silent = [j for j in fragment.junctions if not j.log_files and not j.caller_log_files]
    click.echo(
        f"log files: {len(own)} junction(s) from their own logger, "
        f"{len(borrowed)} from a caller's, {len(silent)} from neither"
    )

    # The same question for the owners that read or write a value and settle
    # nothing.  Two thirds of the readers the map names are these, and until
    # they carried the fact the answer to "which log will show the query
    # running" was silently empty for every one of them.
    sites = fragment.log_sites
    if sites:
        called = [s for s in sites if s.caller_log_files]
        alone = [s for s in sites if s.log_files and not s.caller_log_files]
        mute = [s for s in sites if not s.observable_log_files()]
        click.echo(
            f"  reader log sites: {len(sites)} owner(s) that settle nothing, "
            f"{len(called)} also named by a caller, {len(alone)} only by their own "
            f"or inherited logger, {len(mute)} by neither"
        )
        if mute:
            # The only group here that is a gap.  A reader nobody can watch is
            # a question the map cannot be asked, and that has to be visible
            # rather than implied by a smaller total.
            by_module = Counter(s.owner.split("::")[0] for s in mute)
            click.echo("    no log file named by the source:")
            for module, count in by_module.most_common(top):
                click.echo(f"      {module:<56} {count} owner(s)")
            if len(by_module) > top:
                click.echo(f"      … {len(by_module) - top} more module(s)")
    # Every line every function can print, which a reader lays over a log
    # region.  A call refused outright still has its row, with the reason, and
    # is counted apart: it is a line the region can show and the map cannot
    # match.
    prints = fragment.function_prints
    if prints:
        calls = [statement for node in prints for statement in node.statements]
        several = sum(1 for statement in calls if len(statement.rows) > 1)
        refused = sum(1 for statement in calls if not any(row.pattern for row in statement.rows))
        prefixed = sum(1 for statement in calls if any(row.prefix for row in statement.rows))
        click.echo(
            f"  function prints: {len(calls)} logging call(s) over {len(prints)} function(s), "
            f"{several} printing more than one way, {refused} refused outright, "
            f"{prefixed} with the logger's prefix"
        )
    # The other side of the same statements: where the query that would carry a
    # row onwards actually is.  Reported beside the log sites because the two
    # answer one question between them -- which log to read, and where in the
    # code -- and because a reader the map names with no statement recorded is
    # what ``readers-have-a-statement`` fails on.
    reads = fragment.read_sites
    if reads:
        many = [r for r in reads if len(r.selects) > 1]
        windowed = [r for r in reads if r.bind_window]
        conditioned = [r for r in reads if r.conditions]
        terms = sum(len(r.conditions) for r in reads)
        click.echo(
            f"  read sites: {len(reads)} statement(s) over "
            f"{len({r.owner for r in reads})} owner(s), {len(many)} asking on more "
            f"than one value, {len(windowed)} with a bind window"
        )
        # What a row had to be besides holding the value, and the stretch of
        # the function that answers for it.  Both counted because both are
        # answers where they are empty: a statement with no conditions asks
        # only for the value, and that is what the reader wanted to know.
        click.echo(
            f"    {terms} condition(s) over {len(conditioned)} statement(s); "
            f"{len(reads) - len(conditioned)} ask on the value alone"
        )
        consulting = [r for r in reads if r.consults]
        click.echo(
            f"    {sum(len(r.consults) for r in reads)} value(s) a subquery checks rows by, "
            f"over {len(consulting)} statement(s); schema {getattr(plugin, 'schema_stamp', 'none')}"
        )
        # Where reading a statement's terms stopped, by reason: one that does
        # not parse is read not at all, a column whose table is not decided
        # is left out of a statement read otherwise, and a build given no
        # schema says so here rather than reading fewer columns quietly.
        stopped = Counter(
            stop.rsplit(" -- ", 1)[-1]
            for r in reads
            for stop in r.unresolved
            if stop.rsplit(" -- ", 1)[-1] in _TERM_STOPS
        )
        for reason, count in stopped.most_common():
            click.echo(f"    {count:4}  {reason}")
        # Where the rows go after the statement has them: dropped one by one,
        # not there at all, or handed to something the map already answers
        # for.  Counted separately because an empty ``exits`` is an answer --
        # most statements have none -- while an empty ``crosses_to`` on a
        # statement that clearly hands its rows on is a resolution that failed.
        exited = [r for r in reads if r.exits]
        silent_exit = sum(1 for r in reads for e in r.exits if e.startswith("no-rows:"))
        crossed = [r for r in reads if r.crosses_to]
        click.echo(
            f"    {sum(len(r.exits) for r in reads)} exit(s) over "
            f"{len(exited)} statement(s), {silent_exit} of them because no row "
            f"came back; {sum(len(r.crosses_to) for r in reads)} crossing(s) "
            f"over {len(crossed)} statement(s)"
        )
        # The handed-back crossing shape leaves the file, so what it could not
        # resolve is not visible in the nodes at all: a statement whose rows
        # reach a ``return`` and whose caller does nothing the map answers for
        # keeps the same empty field as one whose rows go nowhere.  Said here
        # so the two are not read as the same answer.
        crossings = getattr(plugin, "crossings", None)
        if crossings is not None:
            click.echo(
                f"    {crossings.handing_back} statement(s) hand their rows "
                f"back to a caller, {crossings.resolved} of which reach a "
                f"junction; {crossings.opened} statement(s) have a call on the "
                "way, or a caller back, their calls cannot place"
            )
        # What each statement returns, so that an aggregate's crossings are
        # read as a value going on and an empty result as one not read.
        results = Counter(r.result or "not read" for r in reads)
        click.echo(
            "    returning "
            + ", ".join(f"{kind} {count}" for kind, count in sorted(results.items()))
        )
        # The write side's version of the same silence: a column name built
        # from a mapping key leaves the statement naming no column, which reads
        # as a statement that writes none.
        keys = getattr(plugin, "key_columns", None)
        if keys is not None and keys.loops:
            click.echo(
                f"    {keys.loops} SQL fragment(s) name their column from a "
                f"mapping key: {keys.resolved} read here, {keys.from_parameter} "
                "left to the caller that supplies the mapping"
            )
        # Reported beside them because it is what makes the exits findable:
        # the loop opens inside the region and runs to its own end, and where
        # that end is past the region the exits are only reachable through
        # this stretch.
        handled = [r for r in reads if r.rows_handled]
        outlives = sum(
            1 for r in reads if r.rows_handled and r.region and r.rows_handled[1] > r.region[1]
        )
        click.echo(
            f"    {len(handled)} statement(s) work their rows through a loop, "
            f"{outlives} of which runs past the region"
        )
        busiest = Counter(r.owner for r in reads).most_common(top)
        for owner, count in busiest:
            if count < 2:
                break
            click.echo(f"    {owner:<58} {count} statement(s)")
    if silent:
        # Not a defect: a base class logging through a caller's MsgWrapper has
        # no file to name.  Listed because an unnamed file is an observation
        # the map cannot offer, and that has to be visible rather than implied.
        by_module = Counter(j.owner.split("::")[0] for j in silent)
        click.echo("  no log file named by the source:")
        for module, count in by_module.most_common(top):
            click.echo(f"    {module:<58} {count} junction(s)")
        if len(by_module) > top:
            click.echo(f"    … {len(by_module) - top} more module(s)")

    fragile = trigger.fragile_subjects(fragment.junctions)
    if fragile:
        # The question the map exists to answer for a stalled task: nothing
        # re-evaluates these, so waiting will not help.
        click.echo(f"  no self-repairing trigger reaches ({len(fragile)}):")
        for subject, triggers in fragile[:top]:
            click.echo(f"    {subject:<34} only {', '.join(triggers)}")

    # What a knight handed the class its work actually runs in.  Not a call, so
    # nothing resolved by name reaches it; before this an arm in a worker could
    # be named and what handed it its input could not.
    uplinks = getattr(plugin, "worker_uplinks", {})
    if not uplinks and getattr(plugin, "worker_classes", 0):
        # The loud half of the same fact.  With no door resolved the whole
        # uplink is off and the map goes back to naming an arm without naming
        # what handed it its input -- which reads, from the report alone,
        # exactly like a corpus that dispatches nothing.
        click.echo(
            "  worker uplinks: none -- the dispatched-worker base class is "
            "declared but the method it enters could not be read"
        )
    if uplinks:
        owners = {j.owner for j in fragment.junctions}
        landed = {key for key in uplinks if "::".join(key) in owners}
        sites = {
            (entry, via) for entries in uplinks.values() for entry, via, _b, _s in entries
        }
        click.echo(
            f"  worker uplinks: {len(sites)} dispatch site(s), reaching "
            f"{len(uplinks)} worker method(s) of which {len(landed)} own a junction"
        )

    # A run-time class choice the source can bound but not settle.  Reported
    # with the line that proves it, since a candidate set nobody can check is
    # three guesses wearing a bracket.
    fanned = [j for j in fragment.junctions if j.dispatch]
    if fanned:
        sets = {
            (d.default, tuple(d.candidates), d.announced_by)
            for j in fanned
            for d in j.dispatch
        }
        silent = sum(1 for _d, _c, said in sets if not said)
        click.echo(
            f"  run-time dispatch: {len(sets)} fan-out(s) over {len(fanned)} junction(s)"
            + (f", {silent} of them printing no class name" if silent else "")
        )
        for default, candidates, said in sorted(sets):
            click.echo(
                f"    {default:<24} -> {len(candidates)} candidate(s): "
                f"{', '.join(candidates)}"
                + (f"   ask the log for {said!r}" if said else "   nothing prints it")
            )

    differing = trigger.differing_arguments(fragment.junctions)
    if differing:
        # Structure, not trivia: an argument one entry omits is a guard that
        # cannot fire on that path, so the candidate causes differ by entry.
        click.echo(f"  entries handing over different arguments ({len(differing)}):")
        for subject, owner, supplied in differing[:top]:
            click.echo(f"    {subject} at {owner.split('::')[-1]}")
            for entry, args in supplied.items():
                click.echo(f"      {entry.split('/')[-1]:<38} {args}")


def write_dump(fragment: MapFragment, path: Path) -> None:
    """Write *fragment* where a reader -- the querier, or a later build of other code -- can read it back.

    The whole fragment, not a summary of it: a baseline is only worth having
    if the comparison it feeds is the same one two live builds would get, and
    a field left out here reads afterwards as a difference the round made.
    """
    path.write_text(fragment.model_dump_json())


#: Why reading a statement's terms stopped -- see :func:`sqltext.terms`.
_TERM_STOPS = frozenset(
    {
        sqltext.NOT_PARSED,
        sqltext.NO_SCHEMA,
        sqltext.OWNER_UNDECIDED,
        sqltext.JSON_PATH,
        sqltext.UNKNOWN_QUALIFIER,
    }
)

#: Shared by every command that builds a map: a build without the definitions
#: reads fewer columns, and two builds compared must have read the same ones.
schema_root_option = click.option(
    "--schema-root",
    default=None,
    metavar="PATH",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help=(
        "The target system's table definitions -- for PanDA, panda-database's "
        "schema/oracle.  They say which table an unqualified column of a join "
        "belongs to; without them such a column is reported, not read."
    ),
)
#: Shared the same way: two builds compared must have read the same record.
instance_record_option = click.option(
    "--instance-record",
    default=None,
    metavar="FILE",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help=(
        "One deployment's recorded values of module names -- for PanDA, the "
        "plugin map its configuration made (bamboo.codemap.calls.read_recorded).  "
        "Another deployment may configure other values; without a record the "
        "names are read as the code says them.  The dump says which record."
    ),
)


@click.command("build-map")
@click.option(
    "--map-id",
    default="panda",
    show_default=True,
    help="Which Code Map to build.",
)
@click.option(
    "--source-root",
    default=None,
    metavar="PATH",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help=(
        "Analyse packages under PATH instead of the installed distribution. "
        "Use this to map a specific release rather than whatever happens to be "
        "installed next to bamboo."
    ),
)
@schema_root_option
@instance_record_option
@click.option(
    "--dump",
    "dump_to",
    default=None,
    metavar="PATH",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Write the map there, as the querier reads it (MAP_DUMP).  Without it the build only reports.",
)
@click.option(
    "--strict",
    is_flag=True,
    help=(
        "Exit non-zero if any gate fails.  Off by default because a gate "
        "failure means two independent expressions of the same fact disagree "
        "-- often the upstream code, not the extraction -- and that is a "
        "signal to look, not a reason to refuse the build."
    ),
)
@click.option("--top", default=10, show_default=True, help="Rows per listing.")
@click.option("-v", "--verbose", is_flag=True, help="DEBUG logging.")
def main(
    map_id: str,
    source_root: Optional[Path],
    schema_root: Optional[Path],
    instance_record: Optional[Path],
    dump_to: Optional[Path],
    strict: bool,
    top: int,
    verbose: bool,
) -> None:
    """Extract a Code Map, run its gates, and write it to a dump where asked."""
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO)

    try:
        plugin = get_code_map_plugin(map_id)
    except ValueError as exc:
        raise click.ClickException(
            f"{exc}  Available maps: {', '.join(available_map_ids())}"
        ) from exc

    try:
        plugin.prepare(source_root, schema_root, instance_record)
    except FileNotFoundError as exc:
        raise click.ClickException(str(exc)) from exc

    fragment = plugin.run()
    results = gates.run_all(fragment)
    all_passed = _report(fragment, results, top, getattr(plugin, "silent_cuts", 0))

    # Plugin-specific findings.  Kept out of the fragment because they describe
    # the target system rather than the map: a table holding no spec is not a
    # node, and a contradiction about one is a fact about PanDA's schema.
    conflicts = getattr(plugin, "table_conflicts", {})
    if conflicts:
        all_passed = False
        click.echo(f"\n[FAIL] table-class-agrees: {len(conflicts)} table(s) disagree")
        for table, classes in sorted(conflicts.items()):
            click.echo(f"      - {table} reads as {sorted(classes)}")
        click.echo("      note: a table holds one kind of row; two answers means one reading is wrong.")
    unmarked = getattr(plugin, "unpromoted", (0, 0))
    if any(unmarked):
        # Not a coverage gap: an attribute no predicate gates, no vocabulary
        # declares and no closed literal set fills is one the code writes and
        # nothing says decides anything.  Kept, with no criteria; the count
        # says how much of the map a reader of the criteria leaves out.
        click.echo(
            f"\nno criterion marks: {unmarked[0]} subject(s), {unmarked[1]} junction(s) "
            "— kept, criteria empty"
        )
    disagreeing = getattr(plugin, "stop_disagreements", None)
    if disagreeing is not None:
        # The answer reads a mover's stops again rather than from the map;
        # a function the two readings part on answers differently from what
        # the build reported.
        if disagreeing:
            all_passed = False
            click.echo(
                f"\n[FAIL] stops-read-the-same-way: {len(disagreeing)} function(s) the answer"
                " would read differently from the build"
            )
            for row in disagreeing[:top]:
                click.echo(f"      - {row}")
        else:
            click.echo("\n[PASS] stops-read-the-same-way: the answer re-reads every function's stops as the build did")
    unexplained = getattr(plugin, "unexplained_steps", [])
    if unexplained:
        # The funnel will report candidates disappearing here and the map has
        # nothing to say about where they went -- the one way this slice can be
        # quietly wrong, so it is named rather than left in a ratio.
        all_passed = False
        click.echo(
            f"\n[FAIL] selection-steps-explained: {len(unexplained)} step(s) count a cut "
            "with no readable reason"
        )
        for step in unexplained[:top]:
            click.echo(f"      - {step}")

    _report_triggers(fragment, plugin, top)

    tagged = getattr(plugin, "tagged_branches", 0)
    unclaimed = getattr(plugin, "unclaimed_tags", [])
    if tagged or unclaimed:
        click.echo(f"\nbranches naming their own decision: {tagged}")
        if unclaimed:
            # The other half of the same comparison.  A tag with no branch is
            # either a decision this slice does not model or a message about
            # something the code stopped doing -- and the corpus has one of the
            # latter, a block logging ``action=set_exhausted`` with the write
            # on the next line commented out.
            click.echo(
                f"  tags named where subjects are settled that no branch claims "
                f"({len(unclaimed)}):"
            )
            for where, tags in unclaimed[:top]:
                click.echo(f"    {where}  {' '.join(tags)}")
            if len(unclaimed) > top:
                click.echo(f"    … {len(unclaimed) - top} more")

    dispatched = getattr(plugin, "runtime_dispatch_writes", [])
    if dispatched:
        # Dropped writes rather than junctions: nothing here says the receiver
        # is a spec.  Printed because that is this reader's inference and not
        # something the code states, and a drop nobody can see is how
        # ``unresolved`` came to absorb "not read" once already.
        click.echo(
            f"\nreceivers whose class is a run-time choice ({len(dispatched)} write(s), "
            "not attributed):"
        )
        for row in dispatched[:top]:
            click.echo(f"  {row}")
        if len(dispatched) > top:
            click.echo(f"  … {len(dispatched) - top} more")

    if fragment.entities:
        # The row-level vocabulary, alongside the column-level one.  Read-only
        # and write-only are both worth saying out loud: the first is a kind of
        # row this map can be asked about but never explain the state of, and
        # the second is one nothing here ever consults.
        # A table nothing here writes is not listed: it is a boundary, and
        # ``extract_read_only_tables`` already reports it as one.
        write_only = [e.name for e in fragment.entities if not e.read_by]
        never_made = [
            e.name for e in fragment.entities if not e.created_by and e.read_by
        ]
        click.echo(
            f"\nentities: {len(fragment.entities)} kind(s) of row, "
            f"{sum(len(e.read_by) for e in fragment.entities)} function(s) reading, "
            f"{sum(len(e.created_by) for e in fragment.entities)} creating, "
            f"{sum(len(e.updated_by) for e in fragment.entities)} updating, "
            f"{sum(len(e.deleted_by) for e in fragment.entities)} deleting"
        )
        if write_only:
            click.echo("  written but never read here:  " + ", ".join(sorted(write_only)))
        if never_made:
            # Rows this map moves but never makes: whatever puts them there is
            # outside what was read, so an investigation that gets this far has
            # to leave the corpus.
            click.echo("  changed but never created here:  " + ", ".join(sorted(never_made)))

    unreadable = getattr(plugin, "unreadable_tables", {})
    if unreadable:
        # Separate from the list below, which is about tables that hold no spec.
        # These are statements whose table nobody could name, and the two read
        # as one finding when they share a line -- only this one is a gap.
        click.echo(
            f"\ntable name supplied at run time and not resolved "
            f"({sum(unreadable.values())} statement(s) in {len(unreadable)} function(s)):"
        )
        for owner, count in sorted(unreadable.items(), key=lambda row: (-row[1], row[0]))[:top]:
            click.echo(f"  {owner:<70} {count}")
        if len(unreadable) > top:
            click.echo(f"  … {len(unreadable) - top} more")

    uncovered = getattr(plugin, "uncovered_tables", set())
    if uncovered:
        # Not a gap in extraction: these are written by the code and hold no
        # spec, so nothing about them can become a subject.  Named because
        # "the map is silent here" is worth knowing before an incident needs it.
        click.echo(f"\ntables written but holding no spec ({len(uncovered)}):")
        click.echo("  " + ", ".join(sorted(uncovered)[:top]))
        if len(uncovered) > top:
            click.echo(f"  … {len(uncovered) - top} more")

    _report_unexplained(fragment, plugin, top)

    if dump_to is None:
        click.echo("\nnothing written: pass --dump PATH to keep the map.")
    else:
        write_dump(fragment, dump_to)
        click.echo(f"\nwrote {dump_to}")

    if not all_passed and strict:
        raise SystemExit(1)


def _report_unexplained(fragment, plugin, top: int) -> None:
    """Say which candidates the SQL slices counted and could not account for.

    A ratio at 89% says a tenth is missing and not which tenth, and this slice
    carried 140 of those across five rounds while every number beside it stayed
    still.  So the slices name them, split two ways.

    **Gap against accounted for.**  The same distinction the line above draws
    for a table that holds no spec.  On this corpus only the read side has any:
    a statement handed to a caller to run is not a gap in the function that
    assembled it.  Every unaccounted-for *write* is a gap, and getting there
    took a correction -- 51 of them were first read as "the varmap comes from
    outside, so this code decides nothing", and the varmap turned out to be a
    list handed to ``executemany`` with every value filled right there.
    ``hs_scrapers`` does write five of seven columns from a row of a parsed HTML
    table, and those stay on the gap side: the reading that would tell them from
    a helper filling the same map does not exist yet, and guessing would put a
    judgement where a measurement belongs.

    Being accounted for does not remove a candidate from the denominator, either
    way: changing what counts as a candidate would move ``slice coverage`` and
    hide a judgement inside a number that looks measured.

    **A missing junction against a missing arm.**  For a write, ``explained``
    and "this column joins an anchored junction" are the same event, so a
    column with no outcome either leaves a junction unbuilt -- the map cannot
    offer that value at all -- or leaves one of the ways of setting it off a
    junction that otherwise looks complete.  The second is the dangerous one:
    pruning is by elimination, so an explanation that never becomes a candidate
    lets a wrong one survive with confidence.
    """
    from bamboo.codemap.panda.recognizers import sqlwrite

    built = {junction.name for junction in fragment.junctions}
    reported = (
        ("sql-write", getattr(plugin, "unexplained_writes", [])),
        ("sql-read", getattr(plugin, "unexplained_reads", [])),
    )
    for label, rows in reported:
        if not rows:
            continue
        # Split on the reason's class, so a reason that names an instance after
        # a colon -- the call form it declined -- still lands in its class.
        gaps = [r for r in rows if r.reason.split(":")[0] not in sqlwrite.ACCOUNTED_FOR]
        click.echo(
            f"\n{label} candidates not accounted for: {len(rows)}"
            f"   ({len(gaps)} a gap in the extraction,"
            f" {len(rows) - len(gaps)} not)"
        )
        for reason, count in Counter(r.reason for r in rows).most_common():
            mark = " " if reason.split(":")[0] in sqlwrite.ACCOUNTED_FOR else "*"
            here = [r for r in rows if r.reason == reason]
            where = Counter(r.file for r in here).most_common(1)[0]
            click.echo(
                f"  {mark} {count:>4}  {reason}"
                f"   (most in {where[0]}, {where[1]})"
            )
        if label == "sql-write":
            missing_arm = [r for r in rows if r.junction in built]
            click.echo(
                f"    of these, {len(missing_arm)} leave a way of setting a value off a"
                " junction that already exists; the rest leave no junction at all"
            )
            for row in missing_arm[:top]:
                click.echo(f"      - {row.owner}  {row.table}.{row.column}")
    click.echo("  * = a gap in the extraction")

    stops = fragment.reading_stops
    if stops:
        # Not a denominator and not a ratio.  These sit on statements the
        # slices otherwise read: what they say is that one clause, one
        # fragment or one bind is decided somewhere this build does not go, so
        # a consumer shown the statement is shown where to read next.  Until
        # this list existed a truncated ``WHERE`` was indistinguishable from
        # no ``WHERE`` -- ``getJobs``, the whole of the pilot's job request,
        # was stored as a ``SELECT`` that filters on nothing.
        where = Counter(r.owner for r in stops)
        click.echo(
            f"\nwhere the reading stopped: {len(stops)} in "
            f"{len(where)} function(s)"
        )
        # The scope, said where the number is: a count read as "where the map
        # stops" is taken for all of it, and it is the SQL reader's alone.
        click.echo(f"  {SCOPE_OF_STOPS}")
        for reason, count in Counter(r.reason for r in stops).most_common():
            here = [r for r in stops if r.reason == reason]
            files = Counter(r.file for r in here).most_common(1)[0]
            click.echo(
                f"  {count:>4}  {reason}   (most in {files[0]}, {files[1]})"
            )
            # The expression, not the class of expression.  A reader told
            # "a fragment does not render" knows only that something is
            # wrong; told ``build_where_clause()`` it knows what to open.
            for detail, seen in Counter(r.detail for r in here).most_common(3):
                click.echo(f"        {seen:>3}x  {detail}")
            if reason == sql.SKIP_HANDED_IN:
                # By function, every one: the text is in the caller, so the
                # function is the only coordinate there is, and a handful
                # folded to three is the silence this line exists to end.
                for owner, seen in sorted(Counter(r.owner for r in here).items()):
                    click.echo(f"        {seen:>3}   {owner}")
        skipped = [r for r in stops if r.reason in sql.SKIP_REASONS]
        if skipped:
            click.echo(
                f"    of these, {len(skipped)} are executions no statement was read out"
                f" of at all, in {len({r.owner for r in skipped})} function(s);"
                " the rest stopped inside a statement that was read"
            )
        for owner, count in where.most_common(top):
            click.echo(f"  {count:>4}  {owner}")
        if len(where) > top:
            click.echo(f"    … {len(where) - top} more function(s)")

    _report_call_stops(fragment, top)
    watch = getattr(plugin, "read_watch", None)
    if watch:
        owner, routes = watch.get("most_routes", ("", 0))
        click.echo(
            f"\nread side, every caller read: most routes into one function {routes} ({owner or 'none'}); "
            f"settling bind values from callers took {watch.get('settle_seconds', 0.0):.1f} s"
        )

    differences = gates.mover_differences(fragment)
    if differences:
        # Every one by name, not folded to ``top``.  The gate beside this used
        # to pass with the count in a note, and ``getJobs`` -- the statement
        # it was added to expose -- was one of thirteen that nobody could see.
        # The gaps are listed first because they are work; the answers are
        # listed because an answer nobody can read is a claim nobody can check.
        click.echo(
            f"\nwho moves a value, read two ways: {len(differences)} function(s) whose "
            "write requires a value its queries do not ask for"
        )
        for reason in gates.MOVER_ORDER:
            here = [d for d in differences if d.reason == reason]
            if not here:
                continue
            mark = "*" if reason in gates.MOVER_FAILURES else " "
            click.echo(f"  {len(here):>4}{mark} {reason}")
            for d in here:
                click.echo(
                    f"          {d.owner}  {d.subject}={d.value}"
                    + (f"  {d.where}" if d.where else "")
                )
        click.echo("  * = a gap in the map, or a statement read in full that asks for nothing")

    # The slices with a gap and no reasons, said out loud rather than left to a
    # reader to notice by subtraction.  Each keeps its own notion of what a
    # candidate is, so a reason vocabulary for one does not transfer, and
    # claiming these were classified would be the more expensive mistake.
    told = {label for label, rows in reported if rows}
    rest = Counter()
    for row in fragment.coverage:
        if row.slice_name not in told and row.candidates > row.explained:
            rest[row.slice_name] += row.candidates - row.explained
    if rest:
        click.echo(
            f"\nnot classified this round: {sum(rest.values())} candidate(s)"
            f" in {len(rest)} slice(s) whose recognizers report no reason"
        )
        for slice_name, count in rest.most_common():
            click.echo(f"    {count:>4}  {slice_name}")


def _report_call_stops(fragment: MapFragment, top: int) -> None:
    """The other list of where the map stops: the calls ``calls`` cannot place.

    Beside the SQL reader's stops and apart from them, because the count above
    was read as where the map stops at all.  A call nobody can place is one
    the walk does not enter and no reader follows a value through, and the
    reason is what would place it -- a receiver whose class nothing declares
    wants a declaration, a ``__getattr__`` wants the class it forwards to.
    """
    stops = fragment.call_stops
    if not fragment.calls_considered:
        return
    where = Counter(stop.owner for stop in stops)
    click.echo(
        f"\nwhere calls could not be followed: {len(stops)} of {fragment.calls_considered} "
        f"calls to this corpus's functions, in {len(where)} function(s)"
    )
    click.echo("  (calls the map cannot place: the walk does not enter them; a logger's are left out)")
    for reason, count in Counter(stop.reason for stop in stops).most_common():
        here = [stop for stop in stops if stop.reason == reason]
        files = Counter(stop.file for stop in here).most_common(1)[0]
        click.echo(f"  {count:>5}  {reason}   (most in {files[0]}, {files[1]})")
        for callee, seen in Counter(stop.callee for stop in here).most_common(3):
            click.echo(f"        {seen:>4}x  {callee}")
    for owner, count in where.most_common(top):
        click.echo(f"  {count:>5}  {owner}")
    if len(where) > top:
        click.echo(f"    … {len(where) - top} more function(s)")
    opened = fragment.open_calls
    click.echo(
        f"\nwhere calls were placed open: {len(opened)} of {fragment.calls_considered} "
        f"calls to this corpus's functions"
    )
    click.echo("  (entered at the definitions they name, and possibly landing elsewhere too)")
    for reason, count in Counter(stop.reason for stop in opened).most_common():
        here = [stop for stop in opened if stop.reason == reason]
        files = Counter(stop.file for stop in here).most_common(1)[0]
        click.echo(f"  {count:>5}  {reason}   (most in {files[0]}, {files[1]})")
        for callee, seen in Counter(stop.callee for stop in here).most_common(3):
            click.echo(f"        {seen:>4}x  {callee}")


if __name__ == "__main__":
    main()
