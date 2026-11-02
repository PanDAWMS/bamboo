"""What the querier says, as data: a STOP, an END or a branch's label is a :class:`Said` -- a kind of :data:`TEMPLATES`
and its facts -- and its English is made here when it is read: out.txt as the run goes, fieldtest's one line, any
reader of result.json.  result.json keeps the records, not the sentences (the map's rule for its whys, RK1; the
analyzer handover's out/odata/odata-design.md), so a rewording is an edit here and no run again.

A fact is JSON (a string, a number, a list, an object, None), another :class:`Said`, or a value JSON does not hold,
kept as a :class:`Lit` (its ``repr`` and ``str``) so a template's ``!r`` and plain field render it as the walk wrote
it.  A fact may still be a sentence a part makes whose own step comes later: a string, rendered as it is.  A template
is :meth:`str.format`'s, with three more specs: ``{x:join=SEP}`` joins a list's items' ``str``, ``{x:rjoin=SEP}``
their ``repr`` (a record's, its sentence's), and ``{x:gap_texts}`` says a list of the map's gap records
(``ReaderGap``) in the map's words.
"""

from __future__ import annotations

import json
import string
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Said:
    """One sentence as data: *kind* names its template in :data:`TEMPLATES`, *facts* fill it.  The facts are taken as
    they are when it is said, as the f-string it replaces took them: a list is copied, a value JSON does not hold
    becomes a :class:`Lit` there."""

    kind: str
    facts: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        assert self.kind in TEMPLATES, self.kind
        object.__setattr__(self, "facts", {k: _taken(v) for k, v in self.facts.items()})

    def __str__(self) -> str:
        return text(self)

    def __format__(self, spec: str) -> str:
        return format(text(self), spec)


@dataclass(frozen=True)
class Lit:
    """A value JSON does not hold, as the walk wrote it: *r* for ``!r``, *s* otherwise."""

    r: str
    s: str

    def __repr__(self) -> str:
        return self.r

    def __str__(self) -> str:
        return self.s

    def __format__(self, spec: str) -> str:
        return format(self.s, spec)


class _Formatter(string.Formatter):
    def format_field(self, value: Any, format_spec: str) -> str:
        if format_spec.startswith("join="):
            return format_spec[len("join=") :].join(str(v) for v in value)
        if format_spec.startswith("rjoin="):
            return format_spec[len("rjoin=") :].join(repr(text(v) if isinstance(v, Said) else v) for v in value)
        if format_spec == "gap_texts":
            return "; ".join(_gap_texts(value))
        return format(value, format_spec)


def _gap_texts(gaps: list[dict]) -> list[str]:
    """The map's words for a function's gaps (``strategy.gap_texts``), from their records as result.json keeps them."""
    from bamboo.codemap import strategy
    from bamboo.codemap.models import ReaderGap

    return strategy.gap_texts([ReaderGap.model_validate(g) for g in gaps])


_FORMAT = _Formatter()


def text(said: Said | str) -> str:
    """The English of *said*: a :class:`Said`'s template filled with its facts; a sentence not made data yet, as is."""
    if isinstance(said, str):
        return said
    return _FORMAT.vformat(TEMPLATES[said.kind], (), said.facts)


def at(step: str, what: Said | str) -> Said:
    """*what* said at *step*: ``H3: ...``."""
    return Said("at-step", {"step": step, "what": what})


def to_json(said: Said | str) -> Any:
    """*said* for result.json: ``{"kind", "facts"}``; a sentence not made data yet, as is."""
    if isinstance(said, str):
        return said
    return {"kind": said.kind, "facts": {k: _json(v) for k, v in said.facts.items()}}


def _taken(v: Any) -> Any:
    if v is None or isinstance(v, (bool, int, float, str, Said, Lit)):
        return v
    if type(v) is list:
        return [_taken(x) for x in v]
    if type(v) is dict and all(isinstance(k, str) for k in v):
        return {k: _taken(x) for k, x in v.items()}
    return Lit(repr(v), str(v))


def _json(v: Any) -> Any:
    if isinstance(v, Said):
        return {"said": to_json(v)}
    if isinstance(v, list):
        return [_json(x) for x in v]
    if isinstance(v, Lit):
        return {"lit": [v.r, v.s]}
    if isinstance(v, dict):
        return {"obj": {k: _json(x) for k, x in v.items()}}
    return v


def from_json(d: Any) -> Said | str:
    """:func:`to_json` read back."""
    if isinstance(d, str):
        return d
    return Said(d["kind"], {k: _value(v) for k, v in d["facts"].items()})


def _value(v: Any) -> Any:
    if isinstance(v, dict):
        if "obj" in v:
            return {k: _value(x) for k, x in v["obj"].items()}
        return from_json(v["said"]) if "said" in v else Lit(*v["lit"])
    if isinstance(v, list):
        return [_value(x) for x in v]
    return v


def same_read_back(said: Said | str) -> bool:
    """Whether *said* reads back from its JSON to the same sentence: what result.json keeps says what the run said."""
    return text(from_json(json.loads(json.dumps(to_json(said))))) == text(said)


def outcome_text(outcome: dict) -> str | None:
    """result.json's ``outcome`` as the run's last lines said it: its END lines (its ``record``, an ``ends``), or its
    STOP line (its ``record``, the stop, and ``hand_used``)."""
    record = outcome.get("record")
    if outcome["kind"] == "end":
        return None if record is None else text(from_json(record))
    used = outcome.get("hand_used") or []
    return f"STOP {text(from_json(record))}; HAND used {len(used)} time(s) {used}"


#: every kind a :class:`Said` can be, and its template
TEMPLATES: dict[str, str] = {
    "at-step": "{step}: {what}",
    "a-run-that-shows-ran": "a run that shows {fn_name}:{chosen_line} ran ({why})",
    "left-by-elimination-no-pass": "left by elimination, no pass of it at the window",
    "no-pass-at-the-window": "no pass of it at the window; {left}",
    "no-run-shows-the-line-ran": "{said} ({by_elimination}no run shows the line ran; {how}), {at}{holes}",
    "by-elimination": "by elimination: ",
    "not-searched-there": "; not searched there: {holes:join=; }",
    "a-method-the-guard-reads-through": "a method the guard reads through{conditions}",
    "its-conditions": " (its conditions: {unread:join=, })",
    "values-the-guard-reads": "the values its guard reads at its run{conditions}",
    "no-guard-to-read": "{fn}:{line} has no guard of its own to read: what made it run is its callers'",
    "guard-not-read-held": "{guard!r} not read ({keys:join=; })",
    "its-guards-hold-with-held": "{how}; {held}",
    "its-guards-hold-held": "its guards hold: {held}",
    "held-column-read": "{guards:rjoin=, } read in the rows file's {section} as fetched at {as_of}{shown}: the column as the pass left it, no later write of it read (KNOWN_ISSUES.md K49)",
    "held-route-shown": ", on {route}, the route its run shows",
    "held-read-false": "{guard!r} read false in the rows file's {section} as fetched at {as_of}",
    "held-key": "{use} {value!r} on {routes:join=, } ({why})",
    "held-key-not-read": "{use} not read on {routes:join=, } ({why})",
    "held-as-fetched": "held as fetched{gone}",
    "not-held-as-fetched": "not held as fetched{gone}",
    "held-gone": "; {gone:join=; }",
    "held-change": "{kind} {key} at {at}",
    "held-change-did-not-run": "{change} did not run: {test!r} at {at} read {read}",
    "held-may-be-changed": "may be changed by {changes:join=, }",
    "held-no-section": "the rows file holds no {section}",
    "held-way-part": "a part of the value read ({steps})",
    "held-way-handed": "what {owner} is handed as {name} is not read",
    "held-way-stopped": "a way stopped ({kind})",
    "held-way-ends": "a way ends at {kind} {name}",
    "held-way-after-not-read": "the changes after the read not read ({stops:join=, })",
    "stops-at-a-terminal": "{label}:{line} stops at {terminal} ({detail}){loose}",
    "loose-not-looked-up": "; the walk does not say what {loose} was looked up with",
    "a-mover-with-a-run-to-hand": "a mover with a run of the key in a log to hand{window}{unsearched}",
    "in-parens": " ({what})",
    "not-searched-all-the-window": "; not searched all the window for {unsearched}",
    "a-pass-in-the-window-that-holds-the-key": "a pass of {func} in the window that holds the key: none of its {unkeyed} there prints it",
    "a-pass-of-the-mover-in-the-window": "a pass of the mover in the window",
    "the-same-tables": "the walk's stop and this call to name the same tables (the stop: {named}, the call: {at_call}){unread}",
    "and-read-every-proxy-method": " and read every proxy method (the stop does not read {named_unread})",
    "the-where-and-binds-of-the-call": "the WHERE and binds of the call that moved it on{failure}",
    "lost-on-order": "lost on order: {where} returns the first rows of {other!r} or of {took!r}; under the first, at the move ({moved}) {total} row(s){mine_rank} at the call's WHERE held {observed!r} ahead of it ({how}){at_least}; its group left {observed!r}: {left} (this row to {next_value!r}){as_now}; the calls that passed it over are not read (their orders and binds); the call that took it used {took!r}",
    "at-least-within-an-hour": " -- at least: the earliest entry counted, {first}, is within an hour of the lookback {lookback}",
    "read-as-now": "; read as now: {as_now:join=, }",
    "ends": "END {ends:join=\nEND }",
    "stop-line": "STOP {stop}",
    "excluded-line": "excluded {stop}",
    "numbered": "{n} ({said})",
    "arm-label": "{arm} ({rows})",
    # a try's row (rerun._row_said) and an arm's rows (rerun._at_rows)
    "row-statuslog": "{status!r} {time}",
    "row-made": "the row's making {time}",
    "row-run": "a run of the key {time}",
    "row-life": "the object's life {time}",
    "at-rows": "{rows:join=, }",
    "wrote-at": "wrote at {row}",
    "may-have-at": "may have at {row}",
    "wrote": "wrote",
    "may-have": "may have",
    # H3's end: what was written last, by which line, how and when
    "written-last-by": "{subject}={observed!r} written last by {fn}:{line}{via}",
    "called-at": ", called at {fn}:{line}",
    "its-guards-hold-in-the-run": "its guards hold in the run's values",
    "its-guards-hold-in-the-rows": "its guards hold in the rows' values",
    "it-has-no-guard": "it has no guard",
    "in-the-pass-of-the-row": "in the pass of the statuslog row {status!r} at {time}",
    "at-time": "at {time}",
    "in-window": "in {lo} .. {hi}",
    "log-on": "{log} on {machines:join=, }",
    "left-by-elimination": "by elimination",
    "it-may-have": "it may have: {premises:join=; }",
    "shown-to-write": "shown to write",
    "guard-read-false": "{guard!r} read false",
    "guard-not-read": "{guard!r} not read",
    # a window's why (windows.Asking.why)
    "window-at-transition": "the statuslog's {status!r} at {time}, {where}",
    "on-the-rows-host": "on {host}, the row's host",
    "on-every-machine-no-host": "on every machine (the row names no host)",
    "rows-carry-no": "the statuslog rows carry no {column}",
    "last-row-holds-another": "the last statuslog row ({status!r} at {time}) holds {column}={value!r}: written after it, and no row closes the window",
    "first-row-no-creation": "the first statuslog row ({status!r} at {time}) holds {column}={value!r} and the row's creation time is not known",
    "value-held": "{column}={value!r} {since} to the statuslog's {status!r} at {time}, on every machine (no row says which run wrote it)",
    "after-the-row-holding": "after the statuslog's {status!r} at {time} ({column}={value!r})",
    "from-the-creation": "from the row's creation at {created}",
    "value-not-a-time": "{column}={value!r}, which its writers set to the time they run, is not a time",
    "value-now": "{column}={value!r}, the time its writer ran: every writer of it writes the time it runs (the last write), on every machine",
    "creation-not-known": "the row's creation time is not known",
    "read-time-not-known": "when the value was read is not known",
    "lifetime": "the object's life, from the row's creation at {lo} to when the value was read at {hi}, on every machine, for {strings:rjoin=, }",
    "no-as-of": "the statuslog says no as_of (when it was read)",
    "no-row-for": "the statuslog has no row for {observed!r}",
    "last-part": "{observed!r} since {since}; from {bound} {span} to {end}",
    "its-row-less": "its row less",
    "as-of-less": "as_of less",
    "the-move-less": "the move less",
    "the-statuslogs-as-of": "the statuslog's as_of {as_of}",
    "the-move-at": "the move {move}",
    "rest-of": "{observed!r} since {since}; the rest of the window, from its row less {slack} to the last {span}, for a mover with no run of its own in those",
    "hours": "{h} h",
    "minutes": "{m} min",
    "the-time-handed-for": "the time handed for {value}, on every machine",
    "a-run-of-the-key-f3": "a run of {key} at {first}..{last} (F3), on every machine",
    "from-the-stays-start": "from the stay's start to the move, {how}",
    "for-marks": "for {marks:rjoin=, }",
    "for-the-key-alone": "for the key alone (its mark holds none)",
    # what an arm H2 left rests on (rerun.Premise)
    "its-condition-not-read": "its condition {guard!r} not read",
    "functions-between-not-read": "the {n} function(s) between the call and it not read",
    "its-run-does-not-show": "its run does not show the line ran ({reason})",
    "a-birth-no-run": "a birth: it wrote as the row was made, at {time}; no run is read",
    "logs-not-searched-whole-over": "its logs were not searched whole over {why}: it may have written in what was not",
    "no-line-holds-the-key": "no line of {logs:join=, } in the records holds {key}: its runs are not shown to name it",
    # L1's stay (rerun._stay_of, _stays_said)
    "a-statuslog-row-for": "a statuslog row for {observed!r}",
    "a-stay-beginning-at": "a stay of {observed!r} beginning at {since} (--since)",
    "one-stay-or-since": "one stay of {observed!r}, or --since naming the one asked: the statuslog holds {stays}",
    "stays": "{n} stays: the first {first}, the last {last}, the longest that ended {longest}",
    "stay-on": "{begins} (on)",
    "stay-lasted": "{begins} ({hours:.2f} h)",
    # census's stop (census.py)
    "classes-define": "{n} classes define {method}: {owners!r}",
    # a run's line (records.Run.said)
    "run-line": "{where} token={token} {n} record(s) {first} .. {last} (key on {found}{cut}{asked})",
    "log-at-machine": "{log}@{machine}",
    "read-as-cut": "; read as cut: {cut}",
    "asked-for-its-rows": "; asked for {method}'s rows and WARNING and above",
    "mover-label": "mover {owner}",
    "others-excluded": "; the other {n_excluded} arm(s) left excluded ({excluded:join=, })",
    "one-arm-ended": "{ends:join= / }{rests}{older}{others}",
    "rests-on": "; it rests on: {premises:join=; }",
    "or-older": "; or at an older row it may have written at: {older:join=, }",
    "the-one-arm-left": "the one arm left",
    "one-of-the-arms-left": "one of the {n_left} arm(s) left",
    "arm-ended": "{n} {arm}: ended",
    "arm-open": "{n} {arm}: open ({kinds:join=, })",
    "a-call-behind": "a call behind {q_name}",
    "a-complete-key-set-for": "a complete key set for {fg_other}",
    "a-facade-s-the-method-the-walk": "a facade's {callee}, the method the walk's stop names, and its calls on the proxy",
    "a-final-state-for-every-job-in": "a final state for every job in {q_name} before the value moved",
    "a-frame-at-the-corpus-boundary-or": "a frame at the corpus boundary, or an innermost frame in the corpus",
    "a-guard-the-passes-made-false-against": "a guard the passes made false against every arm (none shown against arm {unshown:join=, })",
    "a-history-of-to-ask-the-window": "a history of {subject} to ask the window of",
    "a-job-in-at-the-value-not": "a job in {at_call} at {tables_at}, the value not moved by {as_of} (none: the arm kept off was open)",
    "a-legitimate-wait-emptied-at-and-the": "a legitimate wait: {q_name} emptied at {last} and the value moved at {moved_time}; each group waited on is a question of its own (not followed)",
    "a-log-of-to-hand": "a log of {func} to hand ({files})",
    "a-log-to-look-for-a-run": "a log to look for a run of a writer of {subject} in ({funcs}: the map names none; KNOWN_ISSUES.md K25)",
    "a-make-call-whose-function-printed-the": "a make call whose function printed the object's name, in a run holding one of the job's rows and naming the job",
    "a-pass-of-at-the-move-to": "a pass of {owner}:{chosen_line} at the move, to read what moved it on in",
    "a-pass-of-before-the-one-that": "a pass of {func} before the one that moved it on at {q_before}: none from {since}, asked {asking_why} -- the delay is in what calls {func_2} on the row, not followed here (KNOWN_ISSUES.md K45)",
    "a-pass-of-before-the-one-that-2": "a pass of {func} before the one that moved it on at {q_before}: none found from {since}, asked {asking_why}, and not ruled out: {q_unsure:join=; } (KNOWN_ISSUES.md K45)",
    "a-pass-printing-no-key-not-found": "a pass printing no line with the key is not found",
    "files-not-searched-all-the-window": "{files:join=, } not searched all the window",
    "files-not-to-hand": "{files:join=, } not to hand",
    "a-pass-that-wrote-at-the-statuslog": "a pass that wrote {subject}={value!r} at the {of_rows} statuslog row(s) its writers' passes can write (none shows a writer's pass: KNOWN_ISSUES.md K38)",
    "a-print-of-s-text-and-binds": "a print of {where}'s text and binds in its function (the map's printed_by)",
    "a-run-of-at": "a run of {key} at {when}",
    "a-run-that-shows-ran-to-read": "a run that shows {fn_name}:{chosen_line} ran, to read its guards in ({left}; its conditions: {unread:join=, })",
    "a-settled-comparison-of-two-values": "a settled comparison of two values",
    "a-statuslog-row-after": "a statuslog row after {q_observed!r}",
    "a-statuslog-row-after-with-what-the": "a statuslog row after {after} with what the kept-off arm writes {q_outcomes}",
    "a-statuslog-row-for-to-start-the": "a statuslog row for {observed!r} to start the window at",
    "a-statuslog-row-the-writers-of-can": "a statuslog row the writers of {subject} can write",
    "a-statuslog-time-for": "a statuslog time for {value}: {asking_why}",
    "a-statuslog-time-for-2": "a statuslog time for {value}",
    "a-term-of-the-call-s-where": "a term of the call's WHERE on {subject} alone, which the history's value stands for ({f} reads {fact} too)",
    "a-time-for": "a time for {value}: {asking_why}",
    "a-traceback-through-in-the-run": "a traceback through {fn_name} in the run",
    "a-wait-still-open-no-statuslog-row": "a wait still open: no statuslog row after {after} by {as_of}, and {q_name} would hold the {n_named} job(s) in {at_call} at {tables_at}: {named_item}{fact}; each is a question of its own (not followed)",
    "a-way-on-from": "a way on from {name} = {step_value}",
    "a-way-to-compare-the-operands-of": "a way to compare the operands of {fg_kind!r}",
    "a-window-to-ask-the-logs-over": "a window to ask the logs over: {asking_why}",
    "a-writer-of-in-the-map": "a writer of {subject} in the map",
    "a-writer-of-in-the-map-to": "a writer of {subject} in the map, to say when it took {value!r}",
    "an-arm-of-s-own-that-writes": "an arm of {func}'s own that writes another value (S2 reads no arm of what it calls: KNOWN_ISSUES.md K14)",
    "an-arm-of-that-writes-another-value": "an arm of {func} that writes another value than {q_observed!r}",
    "an-arm-that-could-have-written-each": "an arm that could have written {subject}={value!r}: each of the {n_entries} left is excluded ({items:join=; }; KNOWN_ISSUES.md K38)",
    "an-input-the-two-sides-part-on": "an input the two sides part on, ending in a stop",
    "arm-wrote-in-the-passes-not-the": "arm {arm_line} wrote {values} in the passes, not the value observed alone, yet the value stayed {observed!r}: not read",
    "ask-s-of-the-logs-not-in": "{n} ask(s) of the logs not in the records (written to {asks})",
    "empty-before-the-value-moved": "{q_name} empty before the value moved",
    "every-proxy-method-reaches-at-this-call": "every proxy method {callee} reaches at this call read (not read: {unread}; the stop: {fact})",
    "every-writer-of-on-a-history-row": "every writer of {subject} on a history row of the object, a birth or marking its runs by the object's key, to say when it took {value!r} ({n_unplaced} of {n_found} not: KNOWN_ISSUES.md K37)",
    "exactly-one-test-that-came-out-the": "exactly one test that came out the other way",
    "it-did-not-write-at-any-row": "{where}: it did not write at any row it may have written at",
    "its-own-group-among-the-rows-that": "its own group among the rows that left {miss_observed!r} in the window",
    "of-branch-es-stopped": "{n_stops} of {n_q_branches} branch(es) stopped: {stops:join=; }",
    "of-the-arm-s-left-stopped-on": "{n_unrecorded} of the {n_entries} arm(s) left stopped on what the records do not answer yet: {stopped:join=; }",
    "one-alternative-and-one-guard-against-it": "one alternative and one guard against it",
    "one-arm-that-wrote-last-of-left": "one arm that wrote {subject}={value!r} last, of {n_arms} left: {said:join=; } (KNOWN_ISSUES.md K38)",
    "one-binding-of-in-every-pass": "one binding of {name} in every pass",
    "one-binding-of-the-key-for-the": "one binding of the key, for the same row, whose value is in the key set",
    "one-exit-every-pass-took": "one exit every pass took",
    "one-expression-of-the-line-that-raises": "one expression of the line that raises {ch_exc_type}",
    "one-function-among-the-makers": "one function among the makers",
    "one-guard-against-the-nearest-arm-s": "one guard against the nearest arm(s)",
    "one-input-of-the-guard-besides-the": "one input of the guard besides the loop's row",
    "one-name-the-guard-reads": "one name the guard reads",
    "one-order-the-call-that-moved-it": "one order the call that moved it on used, in its SQL",
    "one-other-order-the-statement-can-be": "one other order the statement can be built with",
    "one-thing-is-computed-from": "one thing {expr} is computed from",
    "ran-in-every-pass-and-wrote-the": "{fn_name}:{arm_line} ran in every pass ({n_q_passes}) and wrote {name} = {observed!r}, the value observed: the mover kept the value",
    "reading-s-assumption-at-that-moment-no": "reading 2's assumption at that moment: no final job in {at_call}, and none not yet final outside them",
    "reading-to-give-every-count-the-passes": "reading 2 to give every count the passes printed",
    "s-print-at-among-its-rows": "{func}'s print at {printed_line} among its rows",
    "said": "{what} = {found}",
    "shown-in-its-run": "{said} (shown in its run; {how}), {at}: {question_describe}",
    "the-answer-to-holding-the-column-s": "the answer to {ask!r} holding the column(s) {lacking} (it holds {held}; run the SQL above)",
    "the-answers-to-in-the-rows-file": "the answers to {asks} in the rows file",
    "the-client-builds": "the client {inner_rel} builds",
    "the-container-is-drawn-from": "the container {fg_operand} is drawn from",
    "the-frame-s-source-line-matches-the": "the frame's source line matches the corpus",
    "the-if-whose-test-is": "the if whose test is {guard!r}",
    "the-implicit-else-of": "the implicit else of {i_lineno}",
    "the-job-rows-read-at-the-counts": "the job rows read at the counts' moment ({tables_at}; they were read at {rows_at})",
    "the-job-rows-to-hold-the-row": "the job rows to hold the {n_held} row(s) the counts put in {at_call} (they hold {n_open_rows})",
    "the-key-set-the-walk-gives-consistent": "the key set the walk gives, consistent with the exception",
    "the-loop-that-binds": "the loop that binds {make_first_arg}",
    "the-make-call-of-the-object": "the make call of the object",
    "the-map-s-answer-to-what-moves": "the map's answer to what moves the value on",
    "the-map-s-condition-trees-and-routes": "the map's condition trees and routes of calls (R2, 2079b2a and later)",
    "the-map-s-prints-of-a-statement": "the map's prints of a statement and their cut (printed_by, sqltext.cut and unfilled; R2, 2079b2a and later)",
    "the-object-s-life-to-ask-over": "the object's life, to ask {names} over: {life_why}",
    "the-object-the-exception-names-seen-earlier": "the object the exception names, seen earlier in the run",
    "the-outside-says-the-object-is-missing": "the outside says the object is missing",
    "the-rebuilt-inputs-raise-what-the-log": "the rebuilt inputs raise what the log shows",
    "the-row-itself-once-in-its-own": "the row itself, once, in its own group of those that held {miss_observed!r} at the move under the call's WHERE (the answer counts it {fact} time(s) there, {fact_2} in all; the terms read on the row as it is now: {fact_3})",
    "the-rows-file-s-row-of-its": 'the rows file\'s row of {cls} (its "classes" say which row is which class)',
    "the-rows-returned-in-each-pass-the": "the rows {callee} returned in each pass (the rows file has none)",
    "the-run-of-at": "the run of {candidate_func} at {when}",
    "the-run-s-value-of-every-bind": "the run's value of every bind of the call's WHERE (not written in: {unfilled})",
    "the-same-guards-against-the-arms-in": "the same guards against the arms in every pass",
    "the-statement-at": "the statement at {frame_func}:{frame_line}",
    "the-statement-that-moved-the-value-on": "the statement that moved the value on returns the first rows of an order",
    "the-statuslog-read-after-the-last-pass": "the statuslog read after the last pass ({after}; it was read at {as_of})",
    "the-tables-read-after-the-last-pass": "the tables read after the last pass ({after}; they were read at {tables_at})",
    "the-tables-rows-of-the-key-by": "the tables' rows of the key by table and jobStatus at one moment (the rows file has no counts)",
    "the-time-the-counts-were-read-their": "the time the counts were read (their as_of), to name the jobs in the tables then",
    "the-time-the-statuslog-was-read-its": "the time the statuslog was read (its as_of), to say the value had not moved by then",
    "the-value-of-a-field-of-the": "the value of {what} (a field of the {fact} row: {why})",
    "the-value-of-arm-wrote-in-every": "the value of {name} arm {arm_line} wrote in every pass: a row after it in its block that prints it, nothing binding {name_2} between",
    "the-whole-traceback-of-the-search-cut": "the whole traceback of {record_span} (the search cut it after 200 lines)",
    "what-arm-wrote-yet-the-value-stayed": "what arm {arm_line} wrote, yet the value stayed {observed!r}: one value known at run time (it writes {arm_outcomes})",
    "what-is-computed-from-in-the-walk": "what {expr} is computed from, in the walk",
    "when-the-row-was-made-for-the": "when the row was made, for the birth {names} (its creation column, attr_time.creation_of)",
    "which-call-set-it": "which call set it",
    "which-candidate-ran": "which candidate ran",
    "which-way-it-went": "which way it went",
    "which-way-pass-went": "which way pass {n} went",
    "why-took-it-only-at-and-it": "why {name} took it only at {q_moved}: {where} {held}, and it returns no first rows of an order (not read)",
    "written-last-by-which-the-data-does": "{subject}={value!r} written last by {who}, which the data does not decide: {listed:join=; }{others} (KNOWN_ISSUES.md K38)",
    # a mover's premises (judgement 3): prints_naming, every_log_asked, _searched_whole
    "prints-no-key-column": "the rows file says no column of the row holds the key",
    "prints-no-arm-of-its-own": "no arm of its own: what it calls prints, if anything",
    "prints-no-line-before": "no line runs before the arm(s) at {lines} on every path",
    "prints-read-whole-none-names": "the lines before the arm(s) at {lines} were read whole and none names the written row's {columns}: a search by the key cannot show a run absent",
    "prints-not-read-before": "the map did not read the lines before [{unread:rjoin=, }]",
    "line-hole": "{line}: {hole}",
    "prints-not-known-named": "no line before the arm(s) at {lines} is known to name the written row's {columns}: the map did not read their names whole (KNOWN_ISSUES.md K19)",
    "prints-name-the-key": "a line before every arm at {lines} names the written row's {columns}",
    "logs-the-maps-reading": "the logs are the map's reading's, not where it places the prints (ROWS_OWN_FILES, ROWS_FROM)",
    "logs-no-prints-of-its-own": "the map has no prints of its own for it",
    "logs-placed-by": "the map places some of its prints by {bases}, not by their logger: where they land is not known whole",
    "logs-in-no-file": "the map places its prints in no file",
    "logs-not-asked": "not asked: {missing}",
    "logs-every-file-asked": "every file the map places its prints in was asked: {files}",
    "window-no-log-asked": "no log was asked",
    "window-door-over-files": "the door over files does not say what of the window a file holds",
    "window-not-searched-all": "{logs:join=, } not searched all the window",
    "window-asks-not-answered": "{n} ask(s) not answered yet",
    "window-searched-from-only": "searched from {start} only",
    "window-searched": "{start} .. {end} searched in {to_hand}, none of it unsearched",
    "window-not-asked": "not asked",
    # a mover's gaps, in the map's words
    "map-could-not-read-all": "the map could not read all of it ({gaps:gap_texts})",
    # a mover's reasons (S1 / L2)
    "births": "{births:join=, }",
    "birth-by-at": "{by} at {at}",
    "birth-by": "{by}",
    "births-not-arms": "its writes at {made} make the row (births: {births}): not arms (KNOWN_ISSUES.md K29)",
    "kept-def-not-found": "its def is not found",
    "kept-calls-into": "it calls into {reached}",
    "kept-may-call-into": "it may call into {cut}, whose callers the map lists only in part",
    "kept-hands-rows-on": "it hands its rows on",
    "kept-rows-hidden": "the map could not read where all its rows go",
    "kept-nothing-reached": "no call it makes reaches a writer of another value within the two hops the map lists, and a write further down is not seen (KNOWN_ISSUES.md K14)",
    "no-arm-kept": "no arm of its own, kept: {kept:join=; }",
    "reach-own-arms": "reach: its def holds an arm moving it off by the map's junctions {own_arms}, though S1 counts none: kept",
    "reach-reaches-writers": "reach: reaches {n} writer(s) of another value: {names}",
    "reach-left-open": "reach: reaches no write of another value, but {holes} call(s) on the way are left open, e.g. {first_holes}",
    "reach-rows-hidden": "reach: reaches no write of another value and leaves nothing open, but the map could not read where all its rows go: kept",
    "reach-closed-by-name": "; {n} call(s) closed by name, e.g. {first_closed}",
    "reach-dropped": "no arm of its own, and reaches no write of another value with nothing left open (closure {closure}; assuming {assumptions}{closed}): dropped",
    "its-waker": "not a mover, but it writes {writes:join=; } for rows selected at the value, which {movers:join=, } asks: its waker (the map's wakers; KNOWN_ISSUES.md K39)",
    "writes-at": "{writes} at {line}",
    "rows-select-reach-none": "the rows {selects:join=, } select(s) reach none of its arms {arms}: the map names each one's row by a key they do not fill",
    "reach-writes-no-other-by-key": "{whose}, and the reach from it, those arms left out, writes no other value with nothing left open (closure {closure}; assuming {assumptions}): a writer that takes a row by its key (KNOWN_ISSUES.md K16)",
    "kept-by-reach": "{whose}; kept: {said}",
    "the-reach-was-not-asked": "the reach was not asked",
    "refused-at-the-moment-read": "refused at the moment read, one moment of the window: not dropped",
    "prints-nothing-not-followed": "{n} arm(s) that write another value {arms}; the map has no prints of its own for it, so no run in a log is its own: not followed (KNOWN_ISSUES.md K25)",
    "over-the-whole-window": " over the whole window",
    "from-the-last-part-by-key": " from {start}, the window's last part, which alone is asked of a writer that takes a row by its key (KNOWN_ISSUES.md K16)",
    "did-not-run-judgement-3": "no run of its own found in {to_hand}{over}: it did not run (judgement 3)",
    "no-run-found-not-ruled-out": "no run of its own found in {to_hand}{over}: not followed, though not ruled out (KNOWN_ISSUES.md K19)",
    # a mover's reasons (MT, L2)
    "statement-kind": "{where} {kind}{holds}",
    "holds-for-the-row": ", holds for the row {spans}",
    "all-the-window": "all the window",
    "spans-from-to": "{spans:join=, }",
    "from-to": "from {a} to {b}",
    "writes-refuse-a-row": "its writes of another value refuse a row holding {observed!r}: {arms:join=; }",
    "writes-only-on-a-row-holding": "{line} writes {outcome} only on a row holding {column} in {values}",
    "statements-refused": "{said:join=; }",
    "no-arm-through-calls": "no arm of its own: it writes, if at all, through what it calls (KNOWN_ISSUES.md K14)",
    "taken-by-its-key": "its write requires the value: taken by its key (KNOWN_ISSUES.md K15)",
    "asks-rows-by-run-time-value": "{where} asks rows by a value known at run time ({keys:join=, }): never surely true",
    "not-compared": "{where}: {subject} not compared ({why})",
    "after-only-why": "written at the move or later; its row here holds only the value after",
    "rows-no-column-why": "its row here holds no {attr}",
    "rows-no-row-why": "no row of {owner}",
    "history-no-column-why": "its status history here holds no {attr}",
    "logs-not-asked-not-ruled-out": "its logs are not asked: a run not found would not rule it out ({failing:join=; }) (KNOWN_ISSUES.md K15)",
    "premise-why": "{premise}: {why}",
    "ran-not-read-for-late": "ran: a run of the key in {logs} between {window:join= and }, and L1 names another mover: which way its passes went is not read for late (KNOWN_ISSUES.md K15)",
    "asks-of-its-logs-not-recorded": "{n} ask(s) of its logs not in the records ({why}): whether it ran is not known yet",
}
