"""What changed between two builds of the same map.

Every other check compares the map with something else -- the source it came
from, or the logs the deployment writes.  Neither catches a threshold moving.
A comparison against the source agrees with whatever the source now says, and
a condition is not echoed to any log, so a build made from the wrong release
will explain a decision using a number that has since changed and sound
entirely convincing doing it.  Two builds compared against each other is the
only thing that shows it.

**Nodes are matched by semantic signature, never by position.**  That choice
was made when ``update_job`` moved from ``JobDispatcher.py:975`` to
``api/v1/pilot_api.py:332`` between two releases -- a key of file and line
would have called that one boundary two, or lost it.  Here the same decision
pays twice: matching by signature is what lets a moved-but-unchanged node be
reported as exactly that, which is the common case in a refactor and the one
that should generate no noise at all.

**Condition drift is the headline.**  A changed ``path_condition`` or filter
``conditions`` is the failure the gates are blind to, so it is separated from
everything else rather than listed among renamed fields and new columns.
"""

from __future__ import annotations

import re
from typing import Any, Optional

from pydantic import BaseModel, Field

from bamboo.codemap.models import MAP_NODE_GROUPS, MapFragment
from bamboo.models.graph_element import NodeType

# Keyed off the enum rather than spelled out, because spelling them out is how
# this went wrong once: the label is ``JunctionPoint`` and a hand-written
# "Junction" matched nothing, so junction branches were never compared and
# every junction reported as unchanged whatever had happened to it.
_JUNCTION = NodeType.JUNCTION_POINT.value

# Fields that describe what a node *does*, per kind.  Everything else is
# identity, provenance or position: ``anchor`` moves whenever a line is added
# above it, ``derived_from`` differs by definition, and ``references`` is a
# count that churns with edits elsewhere in the file.  Including any of them
# would bury the changes that matter in changes that never do.
CONTENT_FIELDS: dict[str, tuple[str, ...]] = {
    NodeType.VALUE_ENUM.value: ("namespace", "constant", "value", "comment"),
    # ``selected_by``, ``updated_by`` and ``selection_gates`` are here because
    # leaving them out was a hole of exactly the kind the whole module exists
    # to close.  A round that took ``copyArchive::main`` off
    # ``JobSpec.jobStatus[closed]`` -- who selects on a value, which is the
    # answer the map is asked for -- was told the two builds described the same
    # map, because the value *set* had not moved, only everyone who acts on it.
    NodeType.SUBJECT.value: (
        "spec_class",
        "qualifier_kind",
        "attribute",
        "criteria",
        "vocabulary",
        "selected_values",
        "selected_by",
        "updated_by",
        "selection_gates",
        "truncated_to",
    ),
    NodeType.ENTITY.value: (
        "tables",
        "spec_class",
        "read_by",
        "created_by",
        "updated_by",
        "deleted_by",
    ),
    # ``owner`` is the signature, so it cannot differ between two nodes being
    # compared.
    # ``triggers`` is how the reader's module is started, which is what
    # ``_follow_up`` reads to say whether anything will look at the row again.
    NodeType.LOG_SITE.value: ("log_files", "caller_log_files", "owns_logger", "triggers"),
    # ``owner`` and the line are the signature here, so what is left to differ
    # is what the statement asks for.  A predicate gaining a value is the read
    # side's equivalent of a branch gaining an outcome: it is the difference
    # between a status something carries a row out of and one it does not.
    NodeType.READ_SITE.value: (
        "selects",
        "excludes",
        "refuses",
        "consults",
        "keys",
        "order_by",
        "limits",
        "call_sites",
        "tables",
        "bind_window",
        "conditions",
        # What the statement was read only in part on.  Listed for the same
        # reason ``conditions`` is: a term moving out of ``conditions`` and
        # into here is a statement the map stopped being able to read, and
        # with the field unlisted the two builds would compare as identical.
        "unresolved",
        "region",
        "rows_handled",
        "exits",
        "crosses_to",
    ),
    # ``calls``, ``joined_entities`` and ``dispatch`` are what the owner
    # consults, reads beside its rows and constructs to run, and the two log
    # fields are the ones ``LOG_SITE`` already compares for the same owner: all
    # five were outside this list with nothing saying why, which is the hole
    # :data:`NOT_COMPARED` now closes.
    _JUNCTION: (
        "subject",
        "owner",
        "log_files",
        "caller_log_files",
        "owns_logger",
        "attribution",
        "structural_subject",
        "calls",
        "called_from",
        "joined_entities",
        "dispatch",
    ),
    NodeType.BOUNDARY.value: (
        "system",
        "kind",
        "transport",
        "interface",
        "carried_values",
        "handed_over",
        "operations",
        "access_conditions",
        "observable_values",
        "accepts_arbitrary",
        "version_binding",
        "resolution",
        "direction",
        "verbs",
        "calls_out",
        "call_sites",
    ),
    NodeType.FILTER_STAGE.value: (
        "owner",
        "criteria_tag",
        "funnel_label",
        "order",
        "conditions",
        "inputs",
        "emits",
        "log_level",
        "log_files",
    ),
    # ``message`` is the signature, so it cannot differ between two nodes being
    # compared; ``loop_line`` is a position and moves whenever anything above it
    # does, which is the churn ``anchor`` is left out to avoid.
    NodeType.LOOP_CUT.value: (
        "owner",
        "search_key",
        "scope_prefix",
        "order",
        "conditions",
        "inputs",
        "log_level",
        "log_files",
    ),
    # ``statements`` holds each call's line: a line added above the function
    # moves every one of them, which between two builds of one tree is nothing.
    NodeType.FUNCTION_PRINTS.value: (
        "cls",
        "function",
        "count",
        "statements",
    ),
}

_IDENTITY = "identity: the node's signature or part of it, so equal on both sides"
_PROVENANCE = "provenance: which build wrote the node, different by definition"
_POSITION = "position: moves whenever a line is added above it"
_GLOSS_KEY = (
    "a hash of the function's text: moves with an edited comment, and keys a "
    "reading rather than stating anything the function does"
)

#: The fields of every stored node that are *not* compared, each with why.
#: Together with :data:`CONTENT_FIELDS` they are the whole model -- a test holds
#: the two against ``model_fields`` -- so a field added to a node fails until
#: someone says which list it belongs in.  Until this existed a field nobody
#: listed compared equal to itself however it changed, and ``log_sites``,
#: ``row_precondition`` and ``selected_by`` were each found that way, a round
#: at a time; a misspelt one read ``None`` on both sides and passed as well.
COMMON_NOT_COMPARED: dict[str, str] = {
    "id": _IDENTITY,
    "name": _IDENTITY,
    "node_type": _IDENTITY,
    "map_id": _IDENTITY,
    "derived_from": _PROVENANCE,
    "description": "prose for the vector index, which the code map does not write",
    "metadata": "free-form, and the code map does not write it",
}
NOT_COMPARED: dict[str, dict[str, str]] = {
    NodeType.VALUE_ENUM.value: {
        "anchor": _POSITION,
        "references": (
            "a count of sites that churns with edits elsewhere in the file; "
            "zero is the fact, and value-enum-referenced gates it"
        ),
    },
    NodeType.SUBJECT.value: {},
    NodeType.ENTITY.value: {},
    NodeType.LOG_SITE.value: {"owner": _IDENTITY},
    NodeType.READ_SITE.value: {"owner": _IDENTITY, "anchor": _POSITION},
    _JUNCTION: {
        "anchor": _POSITION,
        "gloss_key": _GLOSS_KEY,
        "branches": "compared one by one: see BRANCH_FIELDS",
        "entry_points": "compared one by one: see ENTRY_KEY and ENTRY_FIELDS",
    },
    NodeType.BOUNDARY.value: {"anchor": _POSITION},
    NodeType.FILTER_STAGE.value: {"anchor": _POSITION, "gloss_key": _GLOSS_KEY},
    NodeType.LOOP_CUT.value: {
        "message": _IDENTITY,
        "loop_line": _POSITION,
        "anchor": _POSITION,
    },
    NodeType.FUNCTION_PRINTS.value: {"owner": _IDENTITY, "anchor": _POSITION},
}

#: What a branch says, compared between two branches of one outcome.  ``tags``
#: and ``messages`` are the signature production writes when the branch fires,
#: so a reworded message is a branch a reader can no longer name from a log.
#: ``value_frames`` is what an observed value rules the branch out by.
BRANCH_FIELDS: tuple[str, ...] = (
    "path_condition",
    "row_precondition",
    "tier",
    "emits",
    "log_level",
    "tags",
    "messages",
    "value_frames",
)
BRANCH_NOT_COMPARED: dict[str, str] = {
    "outcome": "the key branches are matched on",
    "line": _POSITION,
    "order": _POSITION,
}

#: An entry's identity is the door, as :class:`EntryPoint` states it; what it
#: hands over, and through how many sites, is what it says.
ENTRY_KEY: tuple[str, ...] = ("trigger", "entry", "via", "reached_by")
ENTRY_FIELDS: tuple[str, ...] = ("arg_binding", "sites")

#: The parts of a fragment that hold no stored node.  ``diff`` compares what the
#: stored map claims; these are the build's report on itself, which it prints.
_REPORT = "not stored: the build's report on itself, printed by the build"
NOT_COMPARED_PARTS: dict[str, str] = {
    "map_id": _IDENTITY,
    "derived_from": _PROVENANCE,
    "diagnostics": _REPORT,
    "enumeration_writes": _REPORT,
    "coverage": _REPORT,
    "reading_stops": (
        "what the build could not say, not what the map claims -- "
        "see ReadingStop, and the answer reads them again from the source"
    ),
    "call_stops": "the calls the build could not place -- see CallStop",
    "calls_considered": _REPORT,
    "row_key_sources": _REPORT,
    "print_calls": "the second reading prints-are-counted checks the stored rows against",
    "declaration_yields": _REPORT,
    "annotation_readings": _REPORT,
    "spec_annotation_forms": _REPORT,
}

# The two fields that hold a decision's reasoning.  A change here is what no
# other check can see, so it is reported on its own.
DRIFT_FIELDS = frozenset({"path_condition", "conditions"})

# Long values are truncated for display; the point of a diff line is to say
# *that* something moved and roughly where, not to reproduce both versions.
_RENDER_LIMIT = 120


def _render(value: Any) -> str:
    """A stable, comparable rendering of a field value."""
    if isinstance(value, (list, tuple)):
        text = "[" + ", ".join(_render(item) for item in value) + "]"
    elif isinstance(value, BaseModel):
        text = _render(value.model_dump())
    elif isinstance(value, dict):
        text = "{" + ", ".join(f"{k}={_render(v)}" for k, v in sorted(value.items())) + "}"
    else:
        text = str(value)
    return text


def _abbreviate(text: str) -> str:
    return text if len(text) <= _RENDER_LIMIT else text[: _RENDER_LIMIT - 1] + "…"


class Change(BaseModel):
    """One behavioural difference in a node both builds contain."""

    node: str
    kind: str
    field: str
    before: str
    after: str

    @property
    def is_drift(self) -> bool:
        return self.field in DRIFT_FIELDS

    def render(self) -> str:
        return (
            f"{self.node} [{self.field}]\n"
            f"      - {_abbreviate(self.before)}\n"
            f"      + {_abbreviate(self.after)}"
        )


class MapDiff(BaseModel):
    """Everything that differs between two builds."""

    old_version: str
    new_version: str
    added: list[str] = Field(default_factory=list)
    removed: list[str] = Field(default_factory=list)
    changes: list[Change] = Field(default_factory=list)
    moved: list[str] = Field(default_factory=list)
    unchanged: int = 0

    def drift(self) -> list[Change]:
        """Changes to the reasoning behind a decision."""
        return [c for c in self.changes if c.is_drift]

    def other(self) -> list[Change]:
        return [c for c in self.changes if not c.is_drift]

    @property
    def identical(self) -> bool:
        return not (self.added or self.removed or self.changes)


def _key(node: Any) -> tuple[str, str]:
    """Identity: the node kind and its semantic signature.

    Kind is part of the key because the signatures are only unique within one
    -- a subject and the junction that writes it can share a name.
    """
    return (str(node.node_type.value), node.name)


def _index(fragment: MapFragment) -> dict[tuple[str, str], Any]:
    nodes: dict[tuple[str, str], Any] = {}
    # Every kind ``store_fragment`` writes, from the one tuple both read.
    # ``log_sites`` was once missing from a list written out here, so
    # ``CONTENT_FIELDS[LOG_SITE]`` was answering a question nobody asked and all
    # 294 of them compared as absent from both builds.
    for group in MAP_NODE_GROUPS:
        for node in getattr(fragment, group):
            nodes[_key(node)] = node
    return nodes


def _unmatched(olds: list[Any], news: list[Any], fields: tuple[str, ...]) -> tuple[list, list]:
    """What is left of two lists once every item with its twin on the other side is taken.

    Twins are equal on *fields*.  What is left is in source order, so a pair
    made from it is the nth survivor on each side.
    """
    def said(item: Any) -> str:
        return _render([getattr(item, field) for field in fields])

    left = list(olds)
    right = list(news)
    for item in list(left):
        twin = next((other for other in right if said(other) == said(item)), None)
        if twin is not None:
            left.remove(item)
            right.remove(twin)
    return left, right


def _branch_changes(name: str, old: Any, new: Any) -> list[Change]:
    """Compare a junction's branches, matched by outcome.

    Branch identity is ``(owner, outcome)`` -- the plan's choice, and the one
    that makes drift visible: a branch keeps its outcome while the condition
    that reaches it is edited, which is exactly the case a positional
    comparison would render as "everything after line 40 changed".

    **An outcome can have several branches**, and did on 107 of 680 junctions:
    ``add_sub.py::main`` closes a job two ways.  Matched on the outcome alone
    the last of them stood for all, and an edit to any other compared as no
    change.  Branches of one outcome that say the same thing on both sides are
    set aside; what is left is paired in source order and compared field by
    field, so an edited condition is still drift and not a removal and an
    addition.
    """
    before: dict[str, list[Any]] = {}
    after: dict[str, list[Any]] = {}
    for branches, index in ((old.branches, before), (new.branches, after)):
        for branch in branches:
            index.setdefault(branch.outcome, []).append(branch)
    changes: list[Change] = []
    for outcome in sorted(set(before) - set(after)):
        changes.append(
            Change(node=name, kind=_JUNCTION, field="branch", before=outcome, after="—")
        )
    for outcome in sorted(set(after) - set(before)):
        changes.append(
            Change(node=name, kind=_JUNCTION, field="branch", before="—", after=outcome)
        )
    for outcome in sorted(set(before) & set(after)):
        # ``row_precondition`` is here because leaving it out was the same hole
        # the note above ``log_sites`` in ``_index`` records: a field nobody
        # listed compares equal to itself however it changes.  The round that
        # settled these guards' binds -- ``jobStatus=:oldJobStatus`` becoming
        # ``jobStatus=activated`` on 45 junctions -- was told the two builds
        # described the same map.  It is also the field that says whether a
        # branch's outcome can land at all, so drift in it is not cosmetic.
        olds, news = _unmatched(before[outcome], after[outcome], BRANCH_FIELDS)
        for old_branch, new_branch in zip(olds, news, strict=False):
            for field in BRANCH_FIELDS:
                old_value = _render(getattr(old_branch, field))
                new_value = _render(getattr(new_branch, field))
                if old_value != new_value:
                    changes.append(
                        Change(
                            node=f"{name} → {outcome}",
                            kind=_JUNCTION,
                            field=field,
                            before=old_value,
                            after=new_value,
                        )
                    )
        for _ in olds[len(news):]:
            changes.append(
                Change(node=name, kind=_JUNCTION, field="branch", before=outcome, after="—")
            )
        for _ in news[len(olds):]:
            changes.append(
                Change(node=name, kind=_JUNCTION, field="branch", before="—", after=outcome)
            )
    return changes


def _entry_changes(name: str, old: Any, new: Any) -> list[Change]:
    """Compare how a junction is reached.

    A junction losing its polled entry is not a cosmetic difference: it is the
    difference between a stall that clears itself and one that does not.  And
    an entry that stays but hands over something else is a guard that fires
    on one path and not another -- ``arg_binding`` was left out of this, so
    that could move with the report saying nothing.
    """
    def key(entry: Any) -> tuple:
        return tuple(getattr(entry, field) for field in ENTRY_KEY)

    changes: list[Change] = []
    doors_before = sorted({key(e) for e in old.entry_points}, key=str)
    doors_after = sorted({key(e) for e in new.entry_points}, key=str)
    if doors_before != doors_after:
        changes.append(
            Change(
                node=name,
                kind=_JUNCTION,
                field="entry_points",
                before=_render(doors_before),
                after=_render(doors_after),
            )
        )
    for door in sorted(set(doors_before) & set(doors_after), key=str):
        olds, news = _unmatched(
            [e for e in old.entry_points if key(e) == door],
            [e for e in new.entry_points if key(e) == door],
            ENTRY_FIELDS,
        )
        if len(olds) != len(news):
            # Two entries at one door is not a shape the trigger slice makes;
            # said rather than paired, since there is nothing to pair them by.
            changes.append(
                Change(
                    node=f"{name} ← {door[0]} {door[1]}",
                    kind=_JUNCTION,
                    field="entry_points",
                    before=f"{len(olds)} unmatched",
                    after=f"{len(news)} unmatched",
                )
            )
        for old_entry, new_entry in zip(olds, news, strict=False):
            for field in ENTRY_FIELDS:
                old_value = _render(getattr(old_entry, field))
                new_value = _render(getattr(new_entry, field))
                if old_value != new_value:
                    changes.append(
                        Change(
                            node=f"{name} ← {door[0]} {door[1]}",
                            kind=_JUNCTION,
                            field=field,
                            before=old_value,
                            after=new_value,
                        )
                    )
    return changes


def _node_changes(kind: str, name: str, old: Any, new: Any) -> list[Change]:
    if kind not in CONTENT_FIELDS:
        # Silence here is the dangerous failure: a kind nobody listed compares
        # equal to itself no matter what changed, and the diff then reports a
        # rewritten node as unchanged.  A new node kind must add its fields.
        raise KeyError(f"{kind} has no content fields; add them to CONTENT_FIELDS")
    changes: list[Change] = []
    for field in CONTENT_FIELDS[kind]:
        old_value = _render(getattr(old, field, None))
        new_value = _render(getattr(new, field, None))
        if old_value != new_value:
            changes.append(
                Change(node=name, kind=kind, field=field, before=old_value, after=new_value)
            )
    if kind == _JUNCTION:
        changes.extend(_branch_changes(name, old, new))
        changes.extend(_entry_changes(name, old, new))
    return changes


#: The digest ``selection.py`` appends to a filter stage that carries no tag of
#: its own -- six hex characters of the conditions it tests.
_DIGEST_SUFFIX = re.compile(r"#[0-9a-f]{6}$")


def _resignatured(
    gone: list[tuple[str, str]], arrived: list[tuple[str, str]]
) -> list[tuple[tuple[str, str], tuple[str, str]]]:
    """Pairs that are one node whose signature moved with its own conditions.

    ``selection.py`` names an untagged filter stage after a digest of what it
    tests, and defends that in its own docstring: a step testing something else
    is another step.  It is right about identity and it defeats this report.
    The one change no other check can see -- a condition quietly edited --
    arrives here as a removal and an addition that read as unrelated, and two
    real edits were lost exactly that way:

        - avoid VP queue check#558682    + avoid VP queue check#c02f04
        - input data check#d05145        + input data check#573efe

    So the identity stays where it is and the reporting is what changes.
    """

    def stem(key: tuple[str, str]) -> Optional[tuple[str, str]]:
        base, replaced = _DIGEST_SUFFIX.subn("", key[1])
        return (key[0], base) if replaced else None

    left: dict[tuple[str, str], list[tuple[str, str]]] = {}
    right: dict[tuple[str, str], list[tuple[str, str]]] = {}
    for keys, index in ((gone, left), (arrived, right)):
        for key in keys:
            if (base := stem(key)) is not None:
                index.setdefault(base, []).append(key)
    return [
        (olds[0], news[0])
        for base, olds in left.items()
        # Only an unambiguous pair.  Two steps of one funnel whose conditions
        # both changed have nothing left in the name to tell them apart, and
        # guessing would report one step's new condition as the other's.
        if len(olds) == 1 and len(news := right.get(base, [])) == 1
    ]


def _anchor_ref(node: Any) -> Optional[str]:
    anchor = getattr(node, "anchor", None)
    return anchor.as_ref() if anchor else None


def compare(old: MapFragment, new: MapFragment) -> MapDiff:
    """Diff two builds of the same map.

    Both sides are built from source by the same plugin, so anything that
    differs here is a difference in the code -- which is what makes the result
    readable as "the release changed this" rather than "the extractor is
    unstable".
    """
    before = _index(old)
    after = _index(new)
    diff = MapDiff(old_version=old.derived_from, new_version=new.derived_from)

    gone = sorted(set(before) - set(after))
    arrived = sorted(set(after) - set(before))
    for old_key, new_key in _resignatured(gone, arrived):
        changes = _node_changes(new_key[0], new_key[1], before[old_key], after[new_key])
        if not changes:
            # Nothing to say about the pair, so leave it as it was found rather
            # than have a node disappear from the report altogether.
            continue
        gone.remove(old_key)
        arrived.remove(new_key)
        diff.changes.extend(changes)
    for key in gone:
        diff.removed.append(f"{key[0]} {key[1]}")
    for key in arrived:
        diff.added.append(f"{key[0]} {key[1]}")

    for key in sorted(set(before) & set(after)):
        kind, name = key
        changes = _node_changes(kind, name, before[key], after[key])
        if changes:
            diff.changes.extend(changes)
        elif _anchor_ref(before[key]) != _anchor_ref(after[key]):
            # The case the semantic signature exists for: the code moved and
            # says the same thing.  Counted rather than listed one by one,
            # because in a refactor this is most of the map.
            diff.moved.append(f"{kind} {name}")
        else:
            diff.unchanged += 1
    return diff
