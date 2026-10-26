"""Which attributes are worth asking "why is it this?" about -- said on each subject, not a filter on the map.

The spec classes declare 421 attributes and the tables add more, but most are
identifiers, timestamps and counters.  The criteria below say which of them the
code treats as deciding something, and :func:`apply` writes them on the subject
(``SubjectNode.criteria``).  A subject no criterion marks is kept with none, and
so are the junctions writing it: a reader that wants the deciding fields reads
the criteria, and one asking about any field the code writes finds it.

**It was a filter once**: only the subjects a criterion marked, and the
junctions writing them, were kept -- on pin k 131 of 676 candidate subjects and
679 of 2001 junctions.  Keeping every candidate showed two defects the filter
had hidden: crossings resolved by a callee's bare name (266 became 450, 98 of
them to ``WorkQueue.pack``) and no edge into ``JobSpec.__setattr__``, which sets
``stateChangeTime``.  And a filter sees only what the criteria see: criterion 3
dropped ``JediTaskSpec.ramCount`` once a ``None`` it counted stopped landing,
and criterion 5, reading a filter stage's input names, does not see
``JobSpec.cloud`` reach a stage by its value.

Three criteria mark a subject as deciding, any one enough.  They were derived by
census over the whole corpus, and each was narrowed by watching something
absurd come out on top:

**1. Gated by a SQL selection predicate.**  ``WHERE t.status IN ('ready',
'running')`` means the field decides whether another component proceeds, which
is what makes "why is it this?" a question worth asking.  The narrowing was
threefold -- ``WHERE PandaID=:PandaID`` is a lookup key, ``WHERE
t.jediTaskID=f.jediTaskID`` is a join, and a quote inside ``IN (SELECT ...)``
belongs to the subquery -- and each was found because ``lfn``, then
``jediTaskID``, ranked first.

**2. A declared vocabulary.**  The rarest and strongest: the class states the
value set outright.

**3. The writes form a closed set of literals.**  Two distinct literal values
at least, and no more than forty: an identifier is written from a sequence or a
row and offers none, while a free-text field offers a different string every
time.  ``jediTaskID`` is the case this was narrowed against and the *range* is
what excludes it -- measured, 91 of the dropped subjects have an attribute
ending in ``ID`` and none of them reaches two distinct literals.

This once also required the literal writes to be a *majority*, on the reading
that ``jediTaskID`` needed it.  It did not, and the share was measured doing a
different job: it kept out ``JobSpec.jobParameters`` and ``JobSpec.metadata``
-- payloads assembled at run time -- while also keeping out
``JediTaskSpec.errorDialog`` at 16 literal writes of 33, which is the map's own
symptom-to-cause hop, and ``JediTaskSpec.ramCount``.  Worse, its denominator
was *the writes this map managed to explain*, not the writes the code makes,
so reading one more statement demoted a subject although PanDA had not changed:
that is how ``setScoutJobData_JEDI`` becoming readable removed
``JediTaskSpec.ramCount`` and its three junctions from the map.  Dropping the
share adds 15 subjects and 141 junctions, measured over the corpus.

Two further signals (the field is mentioned in logs; it is written under a
guard) fire on 69% and 79% of everything and so corroborate rather than
promote.

A fourth criterion marks nothing on its own: it closes the set under the
passthrough edge, so a subject a marked one copies its value from is marked as
well.  Without it the backward walk's first hop out of ``JediTaskSpec.status``
lands on a subject no criterion says decides anything.

A fifth restates the first where the first cannot see.  Criterion 1 reads SQL
predicates, and brokerage gates in Python: ``nucleus``, ``minRamCount``,
``ioIntensity`` and ``maxwdir`` decide whether a site survives the filter chain
without appearing in any ``WHERE``.  Taking them from the selection slice keeps
the evidence as strong -- the code tagged the rejection and counted the cut.

A junction carries no criteria of its own: it is read through its subject's.
"""

from __future__ import annotations

import ast
import re
from collections import Counter

from bamboo.codemap.models import MapFragment, SourceModule
from bamboo.codemap.panda.attribution import UNRESOLVED

WHERE_GATE = "1:state-gate-in-where"
DECLARED_VOCABULARY = "2:declared-vocabulary"
CLOSED_LITERAL_SET = "3:closed-literal-set"
PASSTHROUGH_SOURCE = "4:carried-into-a-promoted-subject"
FILTER_GATE = "5:gates-a-filter-stage"

# ``passthrough(JediTaskSpec.oldStatus)`` -- the subject a branch carries its
# value from.
_PASSTHROUGH = re.compile(r"^passthrough\((.+)\)$")

# Two values are a set; forty are a free-form field wearing one.  This is the
# whole of criterion 3 -- see the module docstring for the share test that used
# to stand beside it and what measurement removed it.
_CLOSED_SET_RANGE = range(2, 41)

# ``WHERE status='ready'`` / ``AND type IN ('input','pseudo_input')``.  The
# literal is what distinguishes a state gate from a lookup: a predicate against
# a bind variable selects a row, a predicate against literals selects a state.
_PREDICATE = re.compile(
    r"(?:WHERE|AND|OR)\s+(?:[a-zA-Z_]\w*\.)?([a-zA-Z_]\w*)\s*"
    r"(?:(?:=|<>|!=|<|>)\s*'[^']*'|\bIN\b\s*\((?P<inlist>[^)]*)\))",
    re.IGNORECASE,
)


def gated_fields(modules: list[SourceModule]) -> Counter:
    """Count fields a SQL predicate tests against literal values."""
    counts: Counter = Counter()
    for module in modules:
        for node in ast.walk(module.tree):
            if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                continue
            if "WHERE" not in node.value.upper():
                continue
            for match in _PREDICATE.finditer(node.value):
                inlist = match.group("inlist")
                if inlist is not None:
                    if "'" not in inlist:
                        continue  # ``IN (:a,:b)`` -- a lookup, not a state gate.
                    if "SELECT" in inlist.upper():
                        continue  # The quote belongs to the subquery.
                counts[match.group(1)] += 1
    return counts


def filter_gated_fields(fragment: MapFragment) -> set[str]:
    """Return the attributes a filter stage's exclusion condition reads.

    Criterion 1 asks whether a field decides that another component proceeds,
    and answers it from SQL: ``WHERE t.status IN ('ready','running')``.  That
    misses every field brokerage gates, because brokerage gates in Python --
    ``nucleus``, ``coreCount``, ``minRamCount``, ``ioIntensity``, ``walltime``,
    ``maxwdir``, ``pledgedCPU`` decide whether a site survives the chain and
    none of them appears in a predicate.

    Reading them from the selection slice rather than from Python conditions at
    large is what makes this evidence instead of a guess: a census of names
    compared against literals in an ``if`` matched 72 known column names over
    540 sites, led by ``key``, ``name`` and ``value``.  A filter stage's
    condition is different in kind -- the code tagged the rejection and counted
    the cut, so it has stated that this test excludes candidates.
    """
    return {name for stage in fragment.filter_stages for name in stage.inputs}


def criteria_for(
    fragment: MapFragment,
    gated: Counter,
    vocabularies: dict[tuple[str, str], set[str]],
) -> dict[str, list[str]]:
    """Return ``{subject name: criteria satisfied}`` for every candidate.

    Criterion 3 reads the extracted junctions rather than rescanning the
    source: they already hold every outcome of every write site, which is
    exactly the evidence -- and it means a subject reached through SQL is
    judged by the same rule as one reached through an attribute.

    It counts the *distinct* literal outcomes and nothing else.  Anything
    weighed against the number of junctions would be weighed against how much
    of the corpus this build managed to read, which moves when the reader
    improves and not when PanDA changes.
    """
    values: dict[str, set[str]] = {}
    for junction in fragment.junctions:
        for branch in junction.branches:
            if branch.tier == 1:
                values.setdefault(junction.subject, set()).add(branch.outcome)

    filtered = filter_gated_fields(fragment)
    found: dict[str, list[str]] = {}
    for subject in fragment.subjects:
        criteria: list[str] = []
        if gated.get(subject.attribute):
            criteria.append(WHERE_GATE)
        if subject.attribute in filtered:
            criteria.append(FILTER_GATE)
        if vocabularies.get((subject.spec_class, subject.attribute)):
            criteria.append(DECLARED_VOCABULARY)
        if len(values.get(subject.name, set())) in _CLOSED_SET_RANGE:
            criteria.append(CLOSED_LITERAL_SET)
        if criteria:
            found[subject.name] = criteria
    return found


def close_over_passthrough(
    fragment: MapFragment, criteria: dict[str, list[str]]
) -> dict[str, list[str]]:
    """Promote the subjects a promoted subject carries its value from.

    ``UPDATE JEDI_Tasks SET status=oldStatus`` is how a task leaves ``pending``,
    so the branch reads ``passthrough(JediTaskSpec.oldStatus)``.  On its own
    ``oldStatus`` satisfies no criterion -- no predicate compares it to a
    literal, and every write to it is a copy or a ``NULL`` -- so it would be
    dropped, and the branch would point at a subject the map does not contain.

    That is not a tidy gap.  The backward walk exists to answer "why is it this
    value", and its very first hop out of the flagship symptom would land on
    nothing.  A subject that decides a promoted one is worth asking about by the
    same argument that promoted the first, so the set is closed under the edge.
    """
    known = {subject.name for subject in fragment.subjects}
    promoted = dict(criteria)
    while True:
        added = False
        for junction in fragment.junctions:
            if junction.subject not in promoted:
                continue
            for branch in junction.branches:
                match = _PASSTHROUGH.match(branch.outcome)
                if match is None or match.group(1) in promoted:
                    continue
                if match.group(1) not in known:
                    # ``currentPriority`` is carried from ``taskPriority``, and
                    # nothing in this map writes ``taskPriority`` -- it arrives
                    # with the task.  Promoting the name anyway would put a
                    # criterion on a node that does not exist; the edge is a
                    # provenance terminal, which ``gates`` reports as one.
                    continue
                promoted[match.group(1)] = [PASSTHROUGH_SOURCE]
                added = True
        if not added:
            return promoted


def apply(fragment: MapFragment, criteria: dict[str, list[str]]) -> tuple[int, int]:
    """Write on each subject the criteria that fire on it -- none where none does -- and keep every one.

    Returns ``(subjects, junctions)`` no criterion marks: the subjects with
    empty ``criteria`` and the junctions writing them.  A junction whose
    subject could not be settled is not among them: it has no subject to
    judge, and is a gap the attribution reports.
    """
    for subject in fragment.subjects:
        subject.criteria = criteria.get(subject.name, [])
    marked = {subject.name for subject in fragment.subjects if subject.criteria}
    return (
        sum(1 for subject in fragment.subjects if not subject.criteria),
        sum(
            1
            for junction in fragment.junctions
            if junction.subject not in marked and junction.attribution != UNRESOLVED
        ),
    )
