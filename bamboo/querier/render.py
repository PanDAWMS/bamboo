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
    "no-run-shows-the-line-ran": "{said} ({by_elimination}no run shows the line ran; {how}), {at}{holes}{store}{counted}{clock}",
    "clock-writers": "; chosen over the other positive arms, each writing the clock: {others:join=; }",
    "clock-writer-in": "{name} was last only where its write fell within {second} (its run {first} .. {last})",
    "clock-writer-off": "{name}'s run ({first} .. {last}) holds no write at {second}",
    "counted-over-linked": ("; the rows file's {cls} rows linked by {via} count it: {n} of {m} with {atoms:join= and } "
                            "({per:join=, } read as those rows), {base} {op} {n} x {amount} = {total}, {match}; its "
                            "condition as the map writes it, {rest:join=; }, read in those atoms alone"),
    "the-observed-value": "the observed value",
    "not-the-observed-value": "not the observed value {observed!r}",
    "by-elimination": "by elimination: ",
    "not-searched-there": "; not searched there: {holes:join=; }",
    "its-conditions": " (its conditions: {unread:join=, })",
    "values-the-guard-reads": "the values its guard reads at its run{conditions}",
    "values-held-on": "where the values its guards held on come from, its run showing it ran ({held:join=; })",
    "values-held-on-from": (
        "where the values its guards held on come from, its run showing it ran ({held:join=; }); where each term's ways "
        "end: {terms:join=; }; its entry: {entry}; {left}"
    ),
    "ran-where-every-value-ends": (
        "{fn}:{line} ran, its run showing it, and every value its guards held on ends ({held:join=; }): "
        "{terms:join=; }; its entry: {entry}"
    ),
    "term-ends": "{term} -> {ends:join=, }",
    "way-end": "{end} ({why}){at}{returned}{route}{via}{changes}{split}",
    "told-apart-by-an-if": (
        ", the constants {call} hands told apart by the if at {ifs:join=, } ({guards:rjoin=, }), read by the map's "
        "records ({asked}): its terms {terms:join=; }"
    ),
    # what a split says of its caller's run (Split.asked): not asked, or what Q5's ask of it answered
    "split-run-not-asked": "its run not asked",
    "split-run-no-log": "its caller's run not asked which binding ran (Q5): no log of it to hand",
    "split-run-unexamined": "its caller's run asked which binding ran (Q5), not answered",
    "split-run-none-shown": "its caller's run asked which binding ran (Q5): it shows none",
    "split-run-several-shown": "its caller's run asked which binding ran (Q5): it shows more than one",
    "way-end-at": " at {at}",
    "way-end-at-handler": " at {at}, in the except handler at {handler} (S3: which exception)",
    "way-end-at-caught": " at {at}, read off the exception the except handler at {handler} caught (K: its run's traceback)",
    "way-end-unread": " at {at}: {operand!r} decided the test on its run",
    "way-end-written": " at {at}: its {part} as written there, of the object made at {made}",
    "made-by-a-change": (
        ", changed between it and the read by {change}, {how}; read as an arm at {site}: its terms {terms:join=; }; its "
        "entry: {entry}"
    ),
    "made-by-a-change-unread": ", changed between it and the read by {change}, {how}; at {site}, its guards ({guards:join=, }) not read",
    "change-shown-by-its-run": "its run showing it",
    "change-left-by-elimination": (
        "left by elimination (it stores what the run printed, its function's rows are in the run, and every other "
        "change is shown not to have run or to store another value)"
    ),
    "change-left-start-other": (
        "left by elimination (it stores what the run printed, its function's rows are in the run, and every other "
        "change is shown not to have run or to store another value), and what the object held of {part} at its start "
        "({start:join= or }) is not what the run printed"
    ),
    "change-left-start-same": (
        "left by elimination (it stores what the run printed, its function's rows are in the run, and every other "
        "change is shown not to have run or to store another value), but what the object held of {part} at its start "
        "({start:join= or }) is what the run printed too: the run does not tell the change from the start"
    ),
    "change-left-start-unread": (
        "left by elimination (it stores what the run printed, its function's rows are in the run, and every other "
        "change is shown not to have run or to store another value), but what the object held of {part} at its start "
        "is not read{why}"
    ),
    "no-field-from": " (the map holds no FieldFrom of {field})",
    "changed-between": ", changed between it and the read: {changes:join=; }",
    "a-change": "{at}{made}",
    "made-by": " made by {via:join= -> }",
    "returned-by": ", returned{line} by {route:join= -> }",
    "returned-from": " from callee line {line}",
    "came-in-by": ", in by {route:join= -> }",
    "bound-via": ", through the field bindings at {via:join= -> }",
    "first-term-stops": "it stops at {term!r} (in {guard!r})",
    "first-term-stops-further": "it stops at {term!r} (in {guard!r}){made}{s3}",
    "first-term-stops-at-a-raise": (
        "it stops at {term!r} (in {guard!r}){made}, at the {type} its run raised at {at}, caught by the except handler"
        " at {handler} (K: its run's traceback)"
    ),
    "raised-at-other": "{at} ({other}: its run's code, not the pin's)",
    "reached-down": ", in {steps:join=, then }",
    "the-arm-of-a-change": "the arm that made the change at {site}",
    "the-if-telling-constants-apart": "the if at {at} that tells apart the constants {call} hands",
    "bound-in-a-handler": ", bound in the except handler at {lines:join=, } (S3: which exception)",
    "terms-end-entry-stops": "every term ends; its entry does not",
    "for-a-call-handing-a-value": "for the call at {site} that may have handed the value",
    "for-a-binding-a-call-hands": "for which binding the call at {site} hands ran (Q5)",
    "for-a-call-that-may-have-entered-it": "for the call at {site} that may have entered it",
    "for-a-call-across-a-send": "for the call at {site} that may have sent the value across the send at {send}",
    "for-the-rows-after-a-call-across-a-send": "for the rows the call at {site} prints after it returns, across the send at {send}",
    "for-the-entry-a-send-entered": "for the run of {entry} the send at {send} entered, its first line closing the caller's window",
    "guard-held-on": "{guard!r} held{on}",
    "held-on-values": " on {values:join=, }",
    "term-printed": "{term} = {value!r} ({where})",
    "printed-values-read-false": "its run's printed values read {guard!r} false{on}, though its run shows it ran",
    "no-guard-to-read": "{fn}:{line} has no guard of its own to read: what made it run is its callers'",
    "the-only-way-to-the-value": (
        "{fn}:{line}'s value {value!r} comes from {callee}(): the one of its {n} way(s) to it the values read leave, its "
        "run showing it ran ({terms:join=; })"
    ),
    "ways-left-to-the-value": "{fn}:{line}'s value {value!r} comes from {callee}(): {k} of its {n} way(s) to it the values read leave ({ways:join= | })",
    "no-way-to-the-value-holds": "{fn}:{line}'s value {value!r} comes from {callee}(): each of its {n} way(s) to it reads a term false ({ways:join= | }){shown}",
    "though-its-run-shows-it-ran": ", though its run shows it ran",
    "one-way": "{terms:join=, }",
    "guard-held-the-only-way": "{guard!r} held, the only way to the value",
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
    "a-count-under-the-order-that-took-it": "a count of the rows ahead of it under {took!r}, the order the call that took it sorted by (KNOWN_ISSUES.md K48: not built)",
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
    # W10: a branch's arm no run shows ran, with no guard of its own
    "written-last-by-if-it-ran": "{subject}={observed!r} written last by {fn}:{line}{via} if it ran",
    "it-has-no-guard-callers-unread": "it has no guard of its own, and its callers' conditions, which made it run or not, are not read",
    "called-at": ", called at {fn}:{line}",
    "its-guards-hold-in-the-run": "its guards hold in the run's values",
    "its-guards-hold-in-the-rows": "its guards hold in the rows' values",
    "its-guards-held-as-it-ran": "its guards hold as its run shows it ran ({guards:join=, }){rest}",
    "guard-quoted": "{guard!r}",
    "the-rest-in-the-run": "; the others in the run's values",
    "its-guards-held-the-only-way": "its guards held, the only way to the value ({guards:join=, }){rest}",
    "the-rest-in-the-rows": "; the others in the rows' values",
    "the-only-writer-left": "the one writer of the value left, none dropped by a guard: ",
    "it-has-no-guard": "it has no guard",
    "in-the-pass-of-the-row": "in the pass of the statuslog row {status!r} at {time}",
    "as-the-row-was-made": "as the row was made, at {time}",
    "at-a-try-s-row": "at {row}",
    "at-time": "at {time}",
    "in-window": "in {lo} .. {hi}",
    "log-on": "{log} on {machines:join=, }",
    "left-by-elimination": "by elimination",
    "it-may-have": "it may have: {premises:join=; }",
    "shown-to-write": "shown to write",
    "guard-read-false": "{guard!r} read false",
    "guard-read-false-on": "{guard!r} read false{on}",
    "a-printed-count-of-none": "its statement changed no row, as the count printed at {line} says (E-4)",
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
    "near-value": "{column}={value!r} less and plus {slack}, from {lo} to {hi}: when it ran, if it wrote the value, on every machine, for {strings:rjoin=, }",
    "near-ran": "{column}={value!r}, a clock plus {offset} s: it ran at {ran} if it wrote the value; from {lo} to {hi}, {slack} either side, on every machine, for the key",
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
    # P43: the run that made the row (rerun._birth_said)
    "a-birth-its-run": "a birth: it wrote as the row was made, at {time}; the run that made it: {runs:join=; }{more} "
                       "({funcs:join=, }, the INSERT its write is born by); whether the arm ran in that pass is not read",
    "more-runs": " and {n} more",
    "for-the-birth": "the run of {funcs:join=, } that made the row at {time}, on every machine",
    "logs-not-searched-whole-over": "its logs were not searched whole over {why}: it may have written in what was not",
    "no-line-holds-the-key": "no line of {logs:join=, } in the records holds {key}: its runs are not shown to name it",
    # the run that recorded the row (rerun._recorder_read, KNOWN_ISSUES.md K67)
    "its-pass-recorded-the-row": "its pass at the row is {run}: {methods:join=, }'s run of {key} ran inside its call at "
                                 "{site}; whether the line ran in it is not read",
    "its-call-at-the-row": "its call at {site} ran while the row was written: the pass printed at {before} just before "
                           "the call and at {after} just after it, the row between; {callee}, which it calls there, prints "
                           "nothing, so this is the call's place, not a run of it",
    "no-recorder-run-at-the-row": "{methods:join=, } ran no run of {key} near the row on {machine} ({logs:join=, }, read "
                                  "whole there): no pass through its call that records the row wrote it",
    "for-the-recorders-pass": "the pass of {func} around its call at {call}, for {marks:rjoin=, }",
    "for-the-rows-recorder-anywhere": "for the key in the runs of {methods:join=, } that record the row ({status}, {time}) within EDGE of it, on every machine (no row says which)",
    "for-the-rows-recorder": "the run of {methods:join=, } that records the statuslog's {status!r} at {time}, on {machine}, "
                             "the row's host",
    # L1's stay (rerun._stay_of, _stays_said)
    "a-statuslog-row-for": "a statuslog row for {observed!r}",
    "a-stay-beginning-at": "a stay of {observed!r} beginning at {since} (--since)",
    "one-stay-or-since": "one stay of {observed!r}, or --since naming the one asked: the statuslog holds {stays}",
    "stays": "{n} stays: the first {first}, the last {last}, the longest that ended {longest}",
    "stay-on": "{begins} (on)",
    "stay-lasted": "{begins} ({hours:.2f} h)",
    # a run's line (records.Run.said)
    "run-line": "{where} token={token} {n} record(s) {first} .. {last} (key on {found}{cut}{asked})",
    "log-at-machine": "{log}@{machine}",
    "read-as-cut": "; read as cut: {cut}",
    "read-alone": "; read alone: its logger writes no mark, so the token is the line's first word{then}",
    "asked-for-its-rows": "; asked for {method}'s rows and WARNING and above",
    "mover-label": "mover {owner}",
    "others-excluded": "; the other {n_excluded} arm(s) left excluded ({excluded:join=, })",
    "one-arm-ended": "{ends:join= / }{rests}{older}{others}",
    "rests-on": "; it rests on: {premises:join=; }",
    "an-end-resting-on": "{what}{rests}",
    "the-one-arm-left-moved-it-on": "the one arm H2 left moved it on, {arm} ({premises:join=; }; KNOWN_ISSUES.md K38)",
    "the-pass-changed-it": "the pass changed it: {arm} is shown to write it, and which of it and the pass's other writes {others:join=, } wrote it last is not read",
    "or-older": "; or at an older row it may have written at: {older:join=, }",
    "the-one-arm-left": "the one arm left",
    "one-of-the-arms-left": "one of the {n_left} arm(s) left",
    "arm-ended": "{n} {arm}: ended",
    "arm-open": "{n} {arm}: open ({kinds:join=, })",
    "a-branch-not-excluded": "a branch that was not excluded (each of the {n_branches} was)",
    "a-call-behind": "a call behind {q_name}",
    "a-complete-key-set-for":"a complete key set for {fg_other}",
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
    # U1, U2: the callers of a mover none of whose passes took the row (KNOWN_ISSUES.md K55)
    "a-caller-the-map-lists": "a caller of {func} that called_from lists (KNOWN_ISSUES.md K55)",
    "caller-branch": "the caller {caller} at {lines}",
    # Q1 part 2: a caller's passes read by its callees' runs (rerun._call_tied)
    "for-the-callers-callees": "for the runs of the key of {caller}'s {n} callee(s) that print it, over {start} .. {end}",
    "for-the-step-s-run": "for {callee}'s own run, from {start} to {end}",
    "a-caller-pass-by-its-callees": "{machine} {first} .. {last}: {callees:join=, } ({way})",
    "it-called-the-mover": "it called the mover",
    "not-into-the-mover": "not into the mover",
    "a-callee-run-is-the-caller-s": "each run of {callee} read is {caller}'s call: {n} other function(s) reach {callee} within 3 calls, and that one of them ran it on the row then is not excluded",
    "a-callee-marked-by-a-reference": "each run of {callee} read is marked by {attr}, which holds a {cls}'s key by the foreign key(s) the schema declares ({keys:join=; }): a run marked by the key is read as the key's (E-1)",
    "one-process-a-machine": "one process of {caller} on {machine}: the two runs are one pass of it",
    "took-it-in-the-next-step": "{mover} took the row at {at} on {machine} in the step after {callee}'s in one pass of {caller}, which calls {callee} at {lines} before {mover} at {mover_lines}: the stay began in that step, at {began}, {rest}; its passes read by its callees: {passes:join=; }; premises: {premises:join=; }",
    "the-step-ended-at": "and the step went on to {last}, {rest} s after",
    "the-step-s-end-not-read": "and where the step ended is not read ({callee}'s run under its own mark is not in the records)",
    "the-caller-s-passes-by-its-callees": "{caller}'s passes on the row, read by its callees' runs: {passes:join=; }; premises: {premises:join=; }; not known: {unknown}",
    "why-each-pass-took-its-way": "why each pass took its way, which {caller}'s own log ({files:join=, }) would say",
    "a-caller-s-run-printing-the-key": "a run of {caller} (calling at {lines}) printing the key: no mark of it holds the key and no row of it prints a key column (KNOWN_ISSUES.md K55)",
    "over-the-movers-window": "from {start} to {end}, {how}",
    "over-the-stay-for-the-lock": "over the stay ({start} .. {end}), for the runs of {lock} on the row (P55)",
    "every-pass-stopped-the-lock-width": "every pass ({n}) of {func} in the window stopped in a call before the value it set reached the row ({calls:join=, }); {caller} calls on toward it at {line} only past {lock}'s take, and {lock}'s calls on the row took it at {takes:join=, } ({gaps:join=, } apart) and refused it {refused} time(s): its term reads lockedTime < now - {width} on the way {caller} calls it (the map), and every refusal came within {width} of the take before it, every later take past {width} after the one before it -- the lock a pass took kept the others off for its width; no unlock's run is read (P55)",
    "every-pass-stopped-the-lock-read": "every pass ({n}) of {func} in the window stopped in a call before the value it set reached the row ({calls:join=, }); {caller} calls on toward it at {line} only past {lock}'s take, and {lock}'s calls on the row took it at {takes:join=, } ({gaps:join=, } apart) and refused it {refused} time(s), each refusal after a take and before the next: the others were kept off while a lock taken before them stood; why each later take came when it did is not read: the width the lock's term reads is not settled on the way the run came (the map), nor an unlock read (P55)",
    "waker-writes-unshown": "the writes {writes:join=, } print no row a run would show them by: an earlier one of theirs is not ruled out (C)",
    "every-pass-stopped-in-a-call": "every pass ({n}) of {func} in the window stopped in a call before the value it set reached the row ({calls:join=, }): what kept another pass from taking the row between -- a lock its caller takes, its selection -- is not read here (P55)",
    "caller-called-mover-not-found": "{mover}'s pass on the row inside {caller}'s call at {line}: every pass of {caller} on the row called there, and no pass of {mover} holding the key was found; its run inside the call is not read (KNOWN_ISSUES.md K55)",
    "one-call-statement-at": "one statement calling at {site}, or two with the same guards ({n} found)",
    "caller-bare-mark-holds-the-key": "its runs' mark is the key alone, on every line",
    "caller-marks-hold-the-key": "its marks {marks:rjoin=, } hold the key on every line",
    "caller-asked-by-the-key-alone": "asked by the key alone, a pass of it printing no line with the key is not found",
    "no-pass-of-the-caller-handed-the-row": "no pass of {caller} on the row from {start} to {end} ({n} read) handed it to {mover} by its call(s) at {lines}: {why:join=; }{after}; premises: {premises:join=; }",
    "a-handing-prints-the-element": "every way to an append filling the list its call(s) at {lines} hand prints row(s) {rows}, which begin with the element's key: a pass handing the key's element prints a record of it",
    "a-handing-prints-no-element": "a list its call(s) at {lines} hand is filled by no append every way to which prints the element's key: a pass handing the key's element may print no record of it",
    "the-mover-s-last-pass": "; {mover}'s last pass on the row ended at {last}: {way}",
    "passes-left-by": "its {n} pass(es) in the stay left by {exit}",
    "passes-ran-no-arm": "its {n} pass(es) in the stay ran none of its arms, leaving by one of {left_by} or running on past them",
    "passes-took-none": "its {n} pass(es) in the stay took none of its candidates",
    "a-pass-of-the-caller-on-the-row": "a pass of {caller} on the row from {start} to {end}: none found, and not ruled out: {failing:join=; } (KNOWN_ISSUES.md K55)",
    "why-the-caller-ran-on-the-row-only": "what made {caller} run on the row: none of its passes from {start} to {end} took it, and what makes it run on the row -- its own selection, or a caller past hop {hop} of the map's two -- is not read here (KNOWN_ISSUES.md K55)",
    "for-an-agent-s-cycles": "for {agent}'s cycles around its call to {callee}, by {strings:rjoin=, } (its lines hold no key), on every machine",
    "ran-when-an-agent-s-cycle-took-it": "{caller} ran on the row when {agent}'s cycles took it: {takes:join=; }{before:join=}{kept:join=}{refused}{idle}; premises: {premises:join=; }{unknown:join=}",
    "for-a-callee-by-a-linked-value": "for {callee}'s runs by the row's linked {column} values, from {start} to {end}",
    "moved-by-the-caller-through-a-linked-callee": "{caller} moved it: its pass on {machine} ran {callee} (its call at {lines}) from {start}, its run marked by the row's linked {column}, and then {mover} at {at}; of the mover's callers only {caller}'s calls reach {callee} (the map's calls) -- next: what called {caller}; premises: {premises:join=; }",
    "for-the-worker-s-passes-before": "for {worker}'s passes on other rows on {machine}, by {head!r} (its run mark before the key)",
    "the-worker-s-passes-before": "; on {machine}, between the selection's return at {start} and that pass at {end}, {worker} ran on {n} other row(s){span}{how}",
    "the-worker-s-passes-before-asked": "; what ran on {machine} between the selection's return at {start} and that pass at {end} is asked",
    "idle-cycles": "; on {machines:join=, } no cycle selected from {start} to {end}: {items:join=; }",
    "an-idle-cycle": "{machine}'s cycle {mark} printed its line {line} at {start} and its next, {next_line}, at {end}: it was in {calls:join=, } meanwhile",
    "passed-over-refused": "; the {n} selection(s) on {machines} machine(s) from {start} to {end} before the take did not take it: {callee}'s statement refused the row then ({terms:join=; })",
    # Q1 batch 2 part 2: the passed-over selections P5 does not read all of
    "passed-over-left": "; the {n} selection(s) on {machines} machine(s) from {start} to {end} before the take did not take it{parts:join=}",
    "passed-refused-part": " -- {n} refused the row by {callee}'s terms then ({terms:join=; })",
    "passed-shown-part": " -- {n} dropped it after the select at {way}, printed by {prints:join=, } ({times:join=, })",
    "passed-excluded-part": " -- {rest}not {items:join=; }",
    "excluded-term-true": "{entry} (true for the row then)",
    "excluded-step-unprinted": "{way} ({prints:join=, } names the row in none of them, searched)",
    "which-left-it-out": "; not known: what left the row out of each of the other {n} selection(s), one of {alternatives:join=; } -- no log prints which",
    "left-by-a-term": "{entry} (not read either way then: {why:join=; })",
    "term-open-arms": "{arms:join=, } may have written its columns in the stay, printing no line naming the row",
    "term-reached-writes": "{arms:join=, } reached the row before it, no line saying whether they wrote it",
    "left-by-a-term-turning": "{entry} (it turned within the selection; the SELECT's own moment is not printed)",
    "left-by-a-joined-key": "{terms:join= or } (against {sources:join=, }, another table's column)",
    "left-by-a-handed-key": "{terms:join=, } (the values the agent handed the selection, its cycles printing no label)",
    "left-by-the-cap": "its cap ({limits:join=, }: which rows a capped selection leaves out)",
    "left-by-a-silent-step": "{way} after the select (decided at {decided}, printing no line naming the row)",
    "handed-keys-as-the-row-s": "the selections read are those {agent} made with the row's {terms:join=, } (its cycles for {label})",
    "for-the-later-steps": "for the ways after {func}'s select that print the row's key ({prints:join=, }), from {start} to {end}",
    "an-agent-s-take": "{machine}'s cycle {mark} took it ({callee} ran between {after} and {at} and returned {count} row(s)), and the pass on it began at {first}{moved}",
    "and-it-moved-it-on": ", the one that moved it on",
    # Q1 P17: a selection that took the row with no pass of the worker on it after (rerun._unrun_takes)
    "an-agent-s-unrun-take": "{machine}'s cycle {mark} took it ({callee} ran between {after} and {at} and returned {count} row(s); {where} names the row), and no pass of {caller} on it followed there before {until}",
    "a-cycle-s-lock-on-the-row": "{machine}'s {callee} between {after} and {at} returned the row and its lock ({sets}) changed it: {where} prints the statement's binds naming the row, and neither the rows returned nor those locked print",
    "a-cycle-s-lock-changed-it": "{machine}'s lock between {after} and {at} ({sets}) changed the row: {where} prints the lock's binds, not the rows it changed",
    "ran-while-a-take-did": "the selection(s) that ran while a take did ({items:join=, }) did not take it as the take's lock came first, which their lines do not tell",
    "a-selection-s-span": "{machine} {start} .. {end}",
    "no-pass-after-a-cycle-s-take":"no pass of {caller} on the row ran on {machine} after that take but those read",
    "why-a-cycle-left-it": "; not known: why {machine}'s cycle did not hand the row on after taking it, which {files:join=, } on {machine} from {at} would say",
    "kept-off-by-the-lock": "; the {n} selection(s) on {machines} machine(s) from {start} to {end} did not take it: {callee} skips a row locked less than {span} ago ({term}; {bound}) and its take locks it ({sets})",
    "one-process-a-label": "one process of {agent} selects for {label} on each of {machines:join=, } (no two of its cycles there overlap; two processes taking turns would not show)",
    "another-way-in-not-read": "{worker}'s parts given at {at:join=, } (another way in, a message processor's) are not read",
    # M3(a)-cap: the first selection below its cap after the statement's turn took the row
    "ran-when-the-first-selection-below-its-cap-took-it": "{caller} ran on the row when {agent}'s first selection below its cap took it: {where} returns at most {cap} row(s) and began to return the row at {turn}; from then {full} selection(s) on {machines} machine(s) returned {cap} (rows ahead of it) and {empty} none; {machine}'s cycle {mark} returned {count} at {at}, fewer than {cap}: every row selectable then, and the pass on it began at {first}; premises: {premises:join=; }",
    # L4: the mover's first pass after its statement turned for the row took it
    "took-it-at-its-first-pass-after-the-turn": "{func} took the row at its first pass after {where} turned for it: {false} surely false for the row from the stay's start {since} to {turn}, so no pass of it that ended before then took it ({before} read, on {on:join=, }); no pass of it began between {turn} and its pass on {machine} from {first}, which took it at {moved}, on the {n} machine(s) its log was asked of ({logs:join=, }); premises: {premises:join=; }",
    "its-marks-hold-the-row-s-values": "the passes that may have selected the row are those whose marks hold the row's {pairs:join=, }, as the rows file holds them",
    "held-from-the-stay-s-start": "{column} held the value the stay began with: none of {arms:join=, } wrote it in the stay",
    "an-empty-selection-lost-the-rows": "the {n} selection(s) that returned none ran beside one of {callee} that took the same rows",
    "the-row-s-label-by-the-take-s-place": "{agent}'s lines name no label: the selection for the row's {keys:join=, } (read at as_of) is selection {place} of each cycle's {width}, the only one cycle {mark} returned rows at",
    "another-width-another-label": "the {n} selection(s) of {agent}'s cycles of another width than {width} are another label's (one of the row's below its cap would have taken it)",
    # A2: a mover keyed only by rows another table's wakers write runs on the row when one is written for its key
    "over-the-stay-for-its-wakers": "over the stay from {start} to {end}, for the key: the writes of the rows {reader} selects (the map's wakers)",
    # P45: the key handed in by the callers a selection's rows reach (rerun._waker_turn, agents.crossed_readers)
    "handed-on-by": "handed on by {callers:join=, }, which its rows reach",
    "the-key-from-the-callers": "its writes carry no key of their own (the object is handed in): that the key is a row {reader} selects is read from the map's crosses_to to {callers:join=, }",
    "ran-on-a-row-its-wakers-wrote": "{mover} moved it on {after} s after {waker}:{line} wrote {what} at {at} on {machine}{through}: it takes the row only by the rows {reader} selects ({column}); {earlier}; premises: {premises:join=; }",
    "a-row-holding": "a row for it holding {subject}={value}",
    "a-row-for-it": "a row for it",
    "through-the-entry": ", entered from {entries:join=, }",
    "through-requests": ", entered by a request to {routes:join= or } (every way in is an endpoint's: an outside request made the row)",
    "through-no-one-entry": "",
    "no-earlier-write-in-the-stay": "no waker wrote one for its key earlier in the stay (from {since}; their logs searched whole on every machine)",
    "earlier-writes-in-the-stay": "its wakers wrote {n} more for its key earlier in the stay ({at:join=, }), each refused or of another value (not read)",
    "its-first-pass-after-the-write": "{mover}'s first pass after the write took it (its passes print no key: not read)",
    "h-ran-after-other-rows-moved-on": "{mover} moved it on {delta} s after {where}'s {polarity} term on {inner} began to hold at {at}: the last of the row's {n} {cls} row(s) by {via} to reach one of {values:join=, } was {key}, {value} at {at} on {machine}; the stay from {since} waited on them{later} -- next: {next}; premises: {premises:join=; }",
    "h-made-later": " ({n} {cls} row(s) made after the move not read)",
    "for-the-waited-rows-birth": "for {cls} {key}'s birth by {makers:join=, } near {at}, on every machine",
    "waited-birth-asked": "{cls} {key}'s birth: the runs of {makers:join=, } holding it are asked ({n} ask(s))",
    "waited-birth-run": "{cls} {key} was made by {maker}'s run at {first} on {machine}",
    "waited-birth-unread": "{cls} {key}'s birth: no run of {makers:join=, } holding it is read near its creation",
    "before-the-birth-for-its-wakers": "over the 10 minutes before the {cls} was made, from {start} to {end}, for the key: the writes of the rows {reader} selects (the map's wakers)",
    "waited-trigger": "its maker's selection {where} takes rows its wakers write: the last such write for the key in the 10 minutes before the {cls} was made was {waker}:{line}'s ({values:join=, }) at {at} on {machine}",
    "waited-trigger-asked": "its maker's selection {where}: its wakers' writes for the key from {since} to {until} are asked ({n} ask(s))",
    "waited-trigger-unread": "its maker's selection {where}: its wakers' writes are not read ({why})",
    "waited-trigger-none": "its maker's selection {where}: no waker of it wrote for the key from {since} to {until}",
    "waited-life": "{cls} {key}'s history: {life:join=, }",
    "h-next-late": "why {cls} {key}{made} held {held} from {since} to {until}, the longest of its {status} stays in that span",
    "h-made-at": " (made at {at}, after the stay began)",
    "h-next-stuck": "why {cls} {key} has not left {value} since {since}, its {status}",
    "h-next-none": "why {cls} {key} had not moved on (its history holds no row)",
    "h-rows-it-waits-on": "{where}'s {polarity} term on {inner} did not hold up to {until}, the window's end: {n} of the row's {total} {cls} row(s) by {via} not in one of {values:join=, } then ({open:join=, }): it waits on them -- next: {next}; premises: {premises:join=; } (KNOWN_ISSUES.md K77)",
    # P25 (i): a count the selection groups turned (rerun._count_turn)
    "h-count-turned": "{mover} took it {delta} s after {where}'s count at {line} turned: {test} holds at the {name} the take printed ({k}; {printed}) and not at {less}; of the row's {n} {cls} row(s) by {via}, the one that brought it to {k} by getting a status under which {counted}={value!r} is written was {key}, {status} at {at} on {machine}{later}; premises: {premises:join=; }",
    "h-rows-stand-for-the-count": "the rows the count is over are the row's; its {cls} rows by {via} stand for them, one per {cls} row (which of them hold {counted} rows is not read)",
    "h-count-ways-not-read": "the writing arm's other atoms ({atoms:join=; }) are not read: a way through them may write it at another status",
    "h-moved-by-column": "by the writers' accumulations, row by row, a column moves only under some of those values: {columns:join=; } (the map's Branch.accumulated; the others as recorded)",
    "h-carried-by": "the term's columns ({inner:join=, }) are written by {writers:join=, }, which {carriers:join=, } reach as they record {status} (the map's calls; read by function, not by its arms)",
    "h-rows-stand-for-the-term-s": "the {inner} rows the term counts are the row's; its {cls} rows by {via} stand for them (which of them hold those rows is not read)",
    "h-a-later-write-not-read": "a write of the term's columns later than its row's move (a queued update) is not read: the move came {delta} s after it",
    "an-outside-writer-of-the-table": "{system} writes {table} too, with no log here",
    "a-waker-s-write-in-the-stay": "a row {reader} selects written for the key by its wakers ({wakers:join=, }) from {since} to {before}, before {mover} moved it on: none in PanDA's logs, searched whole{outside} (KNOWN_ISSUES.md K55)",
    "no-row-its-wakers-wrote-in-the-window": "{mover} takes the row only by the rows {reader} selects ({column}), and its wakers ({wakers:join=, }) wrote none for the key from {since} to {until}, the window's end, every one's log searched whole and the table written here alone: no row it takes came for the key (A2s); premises: {premises:join=; }",
    "the-wakers-write-every-row-it-takes": "the map's wakers of {reader} are every writer of the rows it selects",
    "a-waker-s-write-in-the-window": "a row {reader} selects written for the key by its wakers ({wakers:join=, }) from {since} to {until}, the window's end, with {mover} not having taken it: none in PanDA's logs, searched whole{outside} (KNOWN_ISSUES.md K55)",
    # M3(a)-2: a time column closed by a run of the row printing it; a pass reaching a selection told by its key line
    "held-as-printed": "{column} held {value} until {at}: {where} printed it then on {machine}, of the row by its key, and the arms that may write it in the stay ({arms:join=, }) write the clock",
    "for-a-print-of-the-column": "from {start} to {end}, for the key: {where}'s print of {column}",
    "ran-at-the-first-pass-reaching-it-after-its-turn": "{caller} ran on the row at its first pass to reach {where} after its turn {turn}: {where} returns every row it holds (no cap), and a pass reaching it prints each row's key at {line} before any exit; no line of {caller}'s with the key from {turn} to {first}, its logs searched whole on every machine; premises: {premises:join=; }",
    "a-pass-reaching-it-after-its-turn": "a pass of {caller} reaching {where} from its turn {turn} to {end}, the window's end: none -- each would print the row's key at {line}, and no line of {caller}'s holds it, its logs searched whole on every machine; premises: {premises:join=; } (KNOWN_ISSUES.md K55)",
    "a-pass-stopped-before-it-not-told": "a pass that stopped before {where} (an error before it) is not told from one that did not run: its lines hold no key",
    "written-outside-too": "; {writers:join=, }",
    "written-here-alone": "",
    # U3: which call ran an arm with no guard of its own (KNOWN_ISSUES.md K56)
    "the-arm-s-run-mark": "which row {fn}'s call handed it: no mark the map records for {fn} reads a row's identity in its run {token!r} (KNOWN_ISSUES.md K56)",
    "a-call-handing-the-mark-on": "a call of {fn}'s whose caller's run marks hold the row its run's mark holds: none of {sites:join=, } does (KNOWN_ISSUES.md K56)",
    "over-the-arm-s-run": "over the arm's run, {start} .. {end}, for {marks:rjoin=, }",
    "a-caller-s-run-holding-the-arm-s": "a run of {callers:join=, } holding {fn}'s run whole ({start} .. {end}, on its machine): {n} run(s) found, none holds it (KNOWN_ISSUES.md K56)",
    "which-call-of-the-caller": "which of {caller}'s calls at {lines} ran {fn}: one run of {caller} holds {fn}'s, and which call it was is not read (KNOWN_ISSUES.md K56)",
    "the-call-has-no-guard": "{caller}:{line} calls {fn} with no guard of its own: what made it run is {caller}'s callers' (KNOWN_ISSUES.md K56)",
    "the-call-that-ran-it": "{fn}:{line} ran inside {caller}'s call at {call_line}, which ran under its guards: {what} (KNOWN_ISSUES.md K56)",
    "files-not-searched-all-the-window": "{files:join=, } not searched all the window",
    "files-not-to-hand": "{files:join=, } not to hand",
    "a-pass-that-wrote-at-the-statuslog": "a pass that wrote {subject}={value!r} at the {of_rows} statuslog row(s) its writers' passes can write (none shows a writer's pass: KNOWN_ISSUES.md K38)",
    "a-print-of-s-text-and-binds": "a print of {where}'s text and binds in its function (the map's printed_by)",
    "a-run-of-at": "a run of {key} at {when}",
    "writes-in-a-batch-no-run-held": "  {func}:{line} writes in a batch (assigned in the loop at {loop}, handed on to {call} at "
                                     "{statement}): no run of the key held from the row before ({before}) to {when}; not asked",
    "writes-in-a-batch-its-run": "  {func}:{line} writes in a batch (assigned in the loop at {loop}, handed on to {call} at "
                                 "{statement}): its run is the key's latest from the row before ({before}) to {when}, "
                                 "{first}..{last} in {log}, the log searched with no hole; it rests on no other pass of it "
                                 "printing the key between",
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
    "every-arm-runs-only-in-an-except": "every arm of {fn_name} runs only in an except handler (at {handlers:join=, }), and every pass printed row {rows:join=, } with nothing after it in the try that may raise",
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
    "shown-in-its-run": "{said} (shown in its run; {how}), {at}: {question_describe}{store}{counted}{clock}",
    # what an attribute's END leaves unread of its value reaching the row (KNOWN_ISSUES.md K41)
    "row-written-later-not-read": "; its row is written by a later statement or call, which is not read (KNOWN_ISSUES.md K41)",
    "failure-caught-not-read": "; its statement's failure is caught in {fn} (except at {line}) and the call returns: that it "
                               "wrote is not read (KNOWN_ISSUES.md K41)",
    "the-answer-to-holding-the-column-s": "the answer to {ask!r} holding the column(s) {lacking} (it holds {held}; run the SQL above)",
    "the-answers-to-in-the-rows-file": "the answers to {asks} in the rows file",
    "the-client-builds": "the client {inner_rel} builds",
    "the-counts-place-every-job": "the counts to place every job in or out of {at_call} (they leave {unplaced}, partly in them by its jobStatus)",
    "the-container-is-drawn-from": "the container {fg_operand} is drawn from",
    "the-frame-s-source-line-matches-the": "the frame's source line matches the corpus",
    "the-if-whose-test-is": "the if whose test is {guard!r}",
    "the-implicit-else-of": "the implicit else of {i_lineno}",
    "the-job-rows-read-at-the-counts": "the job rows read at the counts' moment ({tables_at}; they were read at {rows_at})",
    "the-job-rows-read-from-the-counts": "the job rows read at or after the counts' moment ({tables_at}; they were read up to {rows_at})",
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
    "logs-the-maps-reading": "the logs are the map's reading's, not where it places the prints (ROWS_FROM)",
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
    # bundle G, G-3: the stay's start, where a writer taking the row by its key may have run as the call that began it did
    "the-stay-s-start-by-key": "from {start} to {end}, the stay's start, where a writer taking the row by its key may have run as the call that began the stay did (G-3)",
    "did-not-run-judgement-3": "no run of its own found in {to_hand}{over}: it did not run (judgement 3)",
    "no-run-found-not-ruled-out": "no run of its own found in {to_hand}{over}: not followed, though not ruled out (KNOWN_ISSUES.md K19)",
    # a mover's reasons (MT, L2)
    "statement-kind": "{where} {kind}{holds}",
    "holds-for-the-row": ", holds for the row {spans}",
    "all-the-window": "all the window",
    "spans-from-to": "{spans:join=, }",
    "from-to": "from {a} to {b}",
    "writes-refuse-a-row": "its writes of another value refuse a row holding {observed!r}: {arms:join=; }",
    "not-reached-from-the-value-s-reads": "no read of {observed!r}'s rows reaches it ({reads:join=, }); only reads of other values do, and no API endpoint: not a mover off {observed!r} (M2)",
    "writes-only-on-a-row-holding": "{line} writes {outcome} only on a row holding {column} in {values}",
    "statements-refused": "{said:join=; }",
    "no-arm-through-calls": "no arm of its own: it writes, if at all, through what it calls (KNOWN_ISSUES.md K14)",
    "taken-by-its-key": "its write requires the value: taken by its key (KNOWN_ISSUES.md K15)",
    "asks-rows-by-run-time-value": "{where} asks rows by a value known at run time ({keys:join=, }): never surely true",
    "not-compared": "{where}: {subject} not compared ({why})",
    "after-only-why": "written at the move or later; its row here holds only the value after",
    "after-only-before-why": "written at the move or later; its row here holds only the value after, and the map does not settle it at {value!r}: {why:join=; }",
    "before-no-start": "no arm writing the value is read",
    "before-written-without": "{arms:join=, } write the value without writing it the time they run",
    "before-may-be-written": "{arms:join=, } may write it while the value holds",
    "born-only-held": "{subject} read from the row after the move: every arm writing it makes the row ({arms:join=, }), so it held that value all the stay",
    "written-only-before-held": "{subject} read from the row after the move: no arm writing it lands on the row from {since} to the reading -- its key's selections return no row of the object, or the writes persisting what it sets take the row in other states: {arms:join=; } (I-4, K-4b), so it held that value from then on",
    "held-before": "{subject} read as {value} all the stay:the time of the statuslog row of {observed!r}, to within the pass of {starts:join=, } that wrote both; {out} other line(s) writing it left out{premises:join=}",
    # Q1 part 1: a time column read over the stay from the records (rerun._timeline)
    "for-the-column-s-writes": "for the writes of {where} over {start} .. {end}",
    # S-3: a tested name's value, read off its callee's print inside the pass (rerun._callee_print)
    "for-a-callee-s-print": "for {callee}'s print at {line} of what its call at {at} gives, inside the pass {first} .. {last}",
    "false-on-the-values-read": "{fn_name}'s test at {line} was false in every pass ({n}) on the values read, by {parts:rjoin=, }{whatever}: the arms were kept off there; why those values were what they were is not read here",
    "whatever-they-held": ", whatever {unread:rjoin=, } held",
    "timeline-read": "{subject} over the stay ({start} .. {end}) and the {slack} min before it, from the records: {writes}{rowvalue}{confirmed}{witnessed}{tied}{notrun}{absent}{opens}{quiet:join=}{begun:join=}",
    # K-2 (a): a reached write read as run, its statement's status value the statuslog shows then
    "timeline-confirmed": "; read as run, the statuslog showing their statement's own status value then: {writes:join=; }",
    "timeline-confirmed-write": "{at} by {arm} ({by} writes {value} {statement}; the statuslog's row at {then})",
    # K-3 (b): a write whose clock a later print of the column's value is, read as run, the value held to the print
    "timeline-witnessed": "; read as run, a later print showing the value it wrote: {writes:join=; }",
    "timeline-witnessed-write": "{at} by {arms:join= or } (its clock then is {value}, which {where} on {machine} printed as the row's at {printed}: none wrote it between)",
    # K-3 (c): a write whose statement's write of another column of the row a later print shows run
    "timeline-tied": "; read as run, its statement's write of another column of the row shown run by a later print: {writes:join=; }",
    "timeline-tied-write": "{at} by {arm} (its statement writes {column} by {arms:join=/}, {value} then, which {where} on {machine} printed at {printed})",
    "timeline-quiet": "; {arms:join=, } did not write it from {since} to {until}: their statements write {column} too, which a print shows held its value then",
    # K-2 (b): a reached write whose key's selection takes other states than the row's then
    "its-selection-takes-other-states": "its key's selection {site} takes {takes:join=, } only, the statuslog holding {state} then",
    "timeline-begun-by": "; the stay begun by {arms:join=, } (the other arms writing the value shown not to then, judgement 3)",
    "timeline-begun-unread": "; {arms:join=, }, which may have begun the stay, print nothing naming the row as it began, while another arm that may have does",
    "timeline-begun-runtime": "; {arms:join=, }, writing the status from a value known only at run time, printed the row as the stay began: read as having written the value seen (a premise)",
    "timeline-begun-runtime-premise": "; that reads the {n} arm(s) writing the status from a value known only at run time, of which no write naming the row was read as it began, as not having begun it (a premise)",
    "none-began-the-stay": "none of {arms:join=, } began the stay (they print nothing naming the row as it began, while another arm that may have does)",
    "timeline-writes": "written at {writes:join=; }",
    "timeline-no-write": "no write of it read",
    "timeline-write": "{at} by {arms:join=/} ({how}: {where} on {machine}{before})",
    # bundle F, F-5: the row's birth, at its creation column's value
    "timeline-birth-write": "{at} by {arms:join=/} ({how}: the row's birth, at {where}{before})",
    "timeline-before-the-stay": ", before the stay",
    "timeline-absent": "; shown not written by {arms:join=, } (judgement 3)",
    "timeline-row-value": "; the row's value {value}, read at {as_of}, was last written at {at} by {arms:join= or } ({how}){apart}, none writing it since",
    "timeline-row-value-clock": "each writes the clock with one offset",
    # K-1: a column whose arms write other values too, its value matched to the write the records show
    "timeline-row-value-matched": "its clock then, the write the records show; an arm writing the clock later would hold a later time, and {named:join=, } (writing another value) named as having not written it again",
    "timeline-row-value-matched-clock": "its clock then, the write the records show; an arm writing the clock later would hold a later time",
    # bundle F, F-3: a writer M3(a)-1 left out of the stay is not the row value's
    "timeline-row-value-apart": "; not {apart:join=, or }, left out of the stay",
    "timeline-not-run": "; read as not run in their pass: {writes:join=; }",
    # K-3 (e): a write its pass does not reach, as the judge reads a run (rule 4)
    "its-must-rows-not-in-the-pass": "{why}",
    "timeline-not-run-write": "{at} by {arm} on {machine} ({guards:join=, })",
    "timeline-opens": "; not all read, so they may write it: {arms:join=, }",
    "timeline-read-so-far": "the records show {n} write(s) of it in the stay ({m} more in the {slack} min before it) and {k} arm(s) not writing it",
    "timeline-gap-row-value": "the row's value, read at {as_of}, shows none wrote it from {at} on",
    "false-on-the-timeline": "on its {n} write(s) read in the stay ({m} more in the {slack} min before it), {absent} arm(s) shown not to write it and the others writing at any moment: false for the row all the window",
    "none-of-wrote-it-besides": "none of {arms:join=, } wrote {column} in the stay, besides the {n} write(s) the records show",
    "a-run-time-value-not-none": "the value {arms:join=, } write(s) to {column}, known only at run time, is not None (the map does not rule None out)",
    "timeline-term-held": "{term}: surely held for the row {spans}, on the writes the records show",
    "timeline-term-false-if": "{term}: false for the row at every moment of the stay where {said} is at least {least} (in units of {unit} s): on its {n} write(s) read in the stay ({m} more in the {slack} min before it), {opens} arm(s) writing at any moment, the column was never more than {age} old there (E-5)",
    "timeline-term-held-if": "{term}: surely held for the row {spans}, on the writes the records show, where {named:join= and }",
    "none-of-wrote-it": "none of {arms:join=, } wrote {column} in the stay",
    "none-of-began-it": "none of {arms:join=, } began the stay",
    "timeline-term-not-held": "{term}: surely held at no moment of the stay on the writes the records show (a value not read leaves it open)",
    # K-3 (d): a term comparing a time column with the clock, where it surely failed
    "timeline-term-failed": "{term}: surely false for the row {spans}, on the writes the records show",
    "timeline-term-failed-if": "{term}: surely false for the row {spans}, on the writes the records show, where {named:join= and }",
    # K-3 (a): a term IS NULL or IS NOT NULL on its column's states
    "timeline-state-term": "{term}: {parts:join=; }, on the writes the records show{where}",
    "state-term-held": "surely held for the row {spans}",
    "state-term-failed": "surely false for the row {spans}",
    "state-term-where": ", where {named:join= and }",
    "held-for-the-row-if": "held for the row {spans} where {premises:join=; }",
    "before-premise-runtime": "; premise: {arms:join=, } write the status from a value known at run time, read as not writing {observed!r}",
    "rows-no-column-why": "its row here holds no {attr}",
    "rows-no-row-why": "no row of {owner}",
    "history-no-column-why": "its status history here holds no {attr}",
    "logs-not-asked-not-ruled-out": "its logs are not asked: a run not found would not rule it out ({failing:join=; }) (KNOWN_ISSUES.md K15)",
    "premise-why": "{premise}: {why}",
    "ran-followed-for-late": "ran: a run of the key in {logs} between {window:join= and }, and L1 names another mover: followed on its own, which way its passes in the stay went (KNOWN_ISSUES.md K15)",
    "no-pass-of-in-the-stay": "no pass of {func} with the key in the stay ({since} .. {moved}): its runs fall outside it, so it did not take the row there (KNOWN_ISSUES.md K15)",
    "mover-that-moved-it-label": "mover {owner}, which moved it on",
    # bundle F, F-1: the mover's own passes before the one that moved it, whose selection took the row
    "mover-earlier-takes-label": "mover {owner}, its passes before the one that moved it on, whose selection took the row",
    "held-at-times-not-known-taken-before": "held for the row at times not known; {takes:join=; }: which way each pass read from its mark's records went is read on its own (S2)",
    "held-at-times-not-known-taken-before-unread": "held for the row at times not known; {takes:join=; }",
    "took-it-before": "{func}{lines:join=, } took it before, at {at} on {machine}",
    "took-it-before-uncut": "{func}{lines:join=, } took it before, at {at} on {machine}, in a run of the key's records with no record of its entry row, not cut into passes: which way that pass went is not read (no mark of it to ask its pass by)",
    # bundle H, H-2 (P5): a selection that stamps its own term's column each time it takes the row, by the writes read
    "stamps-itself-by-writes": "{func} writes {subject} with the clock itself on the rows it takes (at {arms:join=, }), and every write of it the records show in the stay ({n}, {first} .. {last}) is one of those: each pass of it that took the row stamped it again, so it took the row no more often than its term's gap (on the writes the records show)",
    # bundle H, H-1 (F-1b): a take's pass asked by the mover's marks' heads on its machine around it
    "the-take-s-pass-by-heads": "{func}'s pass at {at} on {machine}, by its mark from {first} to {last}",
    "took-it-before-heads-asked": "{func}{lines:join=, } took it before, at {at} on {machine}: its pass is asked by its mark there from {first} to {last} ({n} ask(s))",
    "took-it-before-heads-uncut": "{func}{lines:join=, } took it before, at {at} on {machine}: its mark's records there from {first} to {last} hold no pass of it cut around the take: which way that pass went is not read",
    # bundle F, F-4: a take the row's value of a time column the mover's own arms alone wrote shows
    "took-it-by-the-row-value": "{func} took it before, at {at}, writing {column} by {arms:join= or } (the row's value): its pass on {machine} from {first} to {last}",
    "took-it-by-the-row-value-asked": "{func} took it before, at {at}, writing {column} by {arms:join= or } (the row's value): its pass then is asked by its mark on {on} from {first} to {last} ({n} ask(s))",
    "every-machine": "every machine",
    "every-jedi-machine": "every JEDI machine",
    "not-asked-off-jedi": "{func}'s passes are not asked of {logs:join=, }, machines the records name another service's: every route into it passes {entered:join=, }, in pandajedi, whose code runs in JEDI's processes alone (H-3)",
    "took-it-by-the-row-value-others": "{func}'s logs on {n} other machine-file(s) around {at} are not in the records and not asked: another pass of it there writing {column} then is not ruled out",
    "took-it-by-the-row-value-unread": "{func} took it before, at {at}, writing {column} by {arms:join= or } (the row's value): its pass then is not read",
    "unprinted-take": "and that pass printed no key at {line}, which {func} prints for a row it selects and keeps: its condition, {condition}, was false for the row then (read: {read:join=, }; open: {open:join=, }); premises: {premises:join=; }",
    "unprinted-take-count-zero": "and that pass printed no key at {line}, which {func} prints for a row it selects and keeps: its condition, {condition}, holds for every {name} from 1 to {n} (read: {read:join=, }), so {name} was 0 then; premises: {premises:join=; }",
    "as-the-print-needs": "{atom} was {held}, as the print needs it",
    "unprinted-read": "{name}={value!r} {why}",
    "unprinted-the-stay": "(the row's value in the stay)",
    "unprinted-the-rows-file": "(the rows file's, read at {as_of} with no statuslog row between; a write that leaves none is not read, K22)",
    "for-the-key-before-the-moving-pass": "for the key alone in {func}'s prints' logs, from the stay's start {since} to its moving pass at {before}",
    "asks-of-its-logs-not-recorded": "{n} ask(s) of its logs not in the records ({why}): whether it ran is not known yet",
}
