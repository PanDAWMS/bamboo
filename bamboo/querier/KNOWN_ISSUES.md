# The querier: known issues

Where the querier is known to fall short, how each shortfall shows in its output, why it is not fixed
now, and what should bring it back.  Each item names the code it is about, and that code points back
here as `KNOWN_ISSUES.md K<n>`.  An item comes off this list in the commit that fixes it, with its
pointers; ids are not reused.

**Shows** says what a reader of a run's output sees: *a stop* -- the run stops and says why (the safe
way); *said* -- the output says it, and the run goes on; *silent* -- nothing in the output marks it.
**This list is not a census: an issue missing from it is not evidence that the querier is right.**

## S5 -- what the call returned (`rerun.Rerun.rows`)

### K1. Reading 2 holds for one class of call only

It counts the jobs not yet final at each pass, from the job status histories, and takes those to be
what the call returned.  So it assumes:

- the rows are jobs (PandaID, `job_statuslog`, `job_rows`);
- PanDA's table life, written in `rerun.py` and not taken from the map: a job sits in `ACTIVE_TABLES`
  until it reaches one of `FINAL_STATES`;
- the call's WHERE is its key alone.  A call that also filters on the status, such as
  `getQueuedPandaIDsWithTask_JEDI` (`jobStatus NOT IN (...)`), is counted as if it did not;
- the pass printed `len(<name>)`, and the arm waits for that list to empty (the END's criterion 4).

*Shows:* a stop -- where the call reads a table outside `ACTIVE_TABLES`, where reading 2's count
differs from a printed one, or where the counts break the assumption at their moment
(`at_one_moment`).
*Why not now:* the cases that reach S5 all call one method (`getPandaIDsWithTask_JEDI(jediTaskID,
True)` in TaskCommando); a general rule shaped on them would fit them only.
*Revisit:* build order 2 (tmaeno, 2026-10-15: no longer "when other calls reach S5"), with the evaluator L2 and S1
read statements by (`_true_spans`).  First count, from the map, the calls an S4 stop at the database
can name, by class (the tables, the kinds of condition, whether the method returns the rows it
fetched).  The general form would be parts: the map's tables and conditions of the call; the rows at a
time, per row kind, from its status history and the tables it moves through; the conditions evaluated
on those rows.  Reading 2 is that with job rows and no condition but the key.

### K2. S5's premise is not checked

The premise: the count a pass printed is the number of rows the statement returned by the key.  A
facade may return something else: `peekJobs` returns one entry per job ID asked, found or not.

*Shows:* a stop where it has been seen (`peekJobs` reads tables outside `ACTIVE_TABLES`); latent
elsewhere.
*Revisit:* with K1; whether a method returns the rows it fetched is the map's to read.

### K4. Through the file door, S2 reads the passes after the statuslog's `as_of`

It drops them only where the door needs a window (`Rerun.passes`).  Reading 2 at such a pass takes
each job's last state before the histories' `as_of`.

*Shows:* mostly a stop at the fit (a job that moved after `as_of` makes the counts differ); silent only
where such moves cancel out.  The END of a wait still open needs the statuslog and the counts read after
the last pass.
*Revisit:* drop those passes through either door, or need each pass before the histories' `as_of`.

## Windows (`windows.py`)

### K5. `STUCK_SPAN` (6 h) is an account, not a measurement

A mover retries every few hours (tmaeno, 2026-10-13), so S1 asks a stuck window's last 6 h first.  Since build order
2a it sizes only that first ask: a mover with no run of its own there is asked over the rest of the window
(`windows.rest`), so one that retries less often costs a second round of asks rather than going unseen.

*Shows:* the line "no run of its own from ...: asking the rest of the window" and the rest's window.
*Revisit:* when the second round's asks cost too much; measure the movers' retry gaps.

### K6. H6 has no window for a make call in JEDI's code

The window is the Datasets row's `creationdate`, which only the server's setupper inserts.

*Shows:* a stop at H6 that says so (through the record door; the file door needs no window).
*Revisit:* when a case needs it: another time to anchor JEDI's makers by.

## The record door (`recorded.py`)

### K7. `EDGE` (10 min) is the windows' slack, not a measured gap

A run whose records stop for longer than `EDGE` across a log rotation is read as whole while it is not.

*Shows:* silent (the module docstring says it; a run's output does not).
*Revisit:* measure, per log, the longest gap between the records of one run, e.g. the passes of a
task's mark.

### K8. Every machine that lists the log's name is asked

`Ask.service` is always `"unsettled"`: the rule that would name a log's service by its entry process is
not built.

*Shows:* nothing wrong; it costs asks on machines that cannot hold the run.
*Revisit:* when the asks' cost matters.

### K9. JEDI's log clocks were measured against the database on some of its machines only

*Shows:* said (a note on every log read from JEDI's machines).

### K13. The round that asks a run's mark is not limited to the window

The records are not cut to the window (a widened ask reuses them), so the key's records include runs of
other days, and `RecordedLogs._read` asks the mark of each of them that does not hold the key.
*Shows:* nothing wrong; it costs asks (a mark of a run a day before the window, on each machine).
*Why not now:* limiting it to the marks whose key records reach the window +- `EDGE` would miss the end of
a run that goes on longer than `EDGE` past its last record of the key -- the case the mark round is for
(a run that names the key at its start and writes the row at its end).  How long such runs last is not
measured.
*Revisit:* measure, per log, how long a mark's run goes on past its last record of the key.

## S1, S2 -- the movers of a stuck value (`Rerun.stuck`, `Rerun.passes`)

### K14. A mover that writes through what it calls is not followed

S1 counts a mover's arms in its own function only; `TaskBuffer.killJobs` writes through the proxy's
`killJob`, `archiveJob` and `retryJob`, and counts none.

- Such a mover that ran stops at S2: S2 and S3 judge the mover's own arms, and read none of what it calls.
- S1 drops no mover with no arm of its own (`kept_without_arms`).  It says why each is kept: the map's
  `JunctionNode.called_from` shows a call from it into a writer of another value, such a writer's list
  is cut, it hands its rows on, or the map has a gap for it (`through_calls`); failing all of these, that
  `called_from` holds two hops, so a write three or more calls down is not seen.  "Not listed" is not
  "does not reach" (bamboo-85, 2026-10-14).  So movers that cannot move the value stay in, and their
  logs are asked: `JobSpec.jobStatus=running` keeps 7 such, 5 of them statements read by hand as only
  counting.

*Shows:* a stop at S2 for a mover that writes through its calls; each kept mover's reason in S1's output.
*Revisit:* following a mover's calls to their arms (S2 per callee); the map's answer to what a function
reaches -- the writes and the holes on the way (an open call, a call not placed, one outside the corpus,
a hand-off to a thread, queue or plugin) -- dropping a mover only where it reaches none and has no hole.

### K16. S1 lists no writer that takes a row by its key alone

S1's movers are the statements' (`movers_of`): the functions the rows of a statement selecting the value reach, and
the writers whose write requires the value (`guarded_by`).  A writer called with the key -- the pilot's report into
`updateJobStatus`, a kill command into `killJob` -- is in neither for `JobSpec.jobStatus=running`, whose
`guarded_by` is empty.  L2 lists them (`by key`, K15).

*Shows:* silent in S1 (such a writer is not asked).
*Revisit:* build order 2: S1 takes them too, with the absence rule of K15 deciding which of them did not run.

### K17. Stuck reads the movers' statements at one moment only

The rows file's values, `after` among them, are the row's columns as read when the file was built: the rows file
says when (each value's `as_of` and `after`'s, the client's clock, a lower bound), a moment near the window's end but
not the statuslog's `as_of`, which is the database's clock and another instant (Fr2: 13:11:50 and 13:11:59; e1,
2026-10-15).  S1 reads the statements at the moment the values were read; a rows file that does not say is read at the
statuslog's `as_of`, and the header says so.  A column may have held other values earlier in the window.  So S1's
readings (`_read_at_as_of`) drop no mover and remove no ask: a mover that rewrites the column it compares refuses the
row between its passes -- TaskCommando's `getTasksToExecCommand_JEDI@4591` asks `modificationtime < now - 5 min` and
runImpl sets modificationtime on each pass, so at G's as_of the row is refused, though runImpl took it 70 times
(tmaeno, 2026-10-15).

*Shows:* said (each reading is marked "a reading, not a drop").
*Revisit:* with the map's work on which columns are written in a window, and by whom.

### K19. S1 does not follow a mover with no run of its own found

S1 follows a mover only where a log asked for it holds a run of the key in the window asked (`_of_mover`).  One whose
every ask was answered and none held such a run is not followed -- but not ruled out either: absence needs the rule
of K15 (the whole window searched, every log its process writes to, and every run that reaches the arm bound to
print a string the asks search by).  Since build order 2a S1 searches the whole window for such a mover (K5); the
third premise needs the map's prints that every path to an arm passes (asked 2026-10-16), and the asks search by the
question's key, which a mover's runs need not print (the setupper names datasets, not PandaIDs).  result.json says, for
each mover with no run found, which of the three hold (`premises`: `window`, `logs`, `prints`); `prints` is never known
yet, so none is ruled out.
`JobSpec.jobStatus=running` (Fr2) had five such over the last 6 h: `Watcher.run`, `copyArchive.main`,
`TaskBuffer.killJobs`, `analy_pmerge_jobs_wait_time` and `get_site_strr_stats`.

*Shows:* said ("no run of its own found in [...] over the whole window: not followed, though not ruled out");
result.json's verdict `no-run-found`, with its `premises`.
*Revisit:* build order 2a: judgement 3, the absence rule, replaces it once the map answers.

## L2 -- the movers of a late value (`Rerun.missed`, `Rerun._movers_in_window`)

### K15. A writer that takes a row by its key is not ruled out

Every writer with an arm that can move the value (`arms_of`) is listed `by key`: whether it ran on the key in the
window is read from its runs there, which L2 does not ask.  Ruling one out needs more than finding no run: the whole
window searched in every log its process writes to (a DB proxy method writes `panda-DBProxy.log` or
`panda-JediDBProxy.log` by the process that called it), and its runs bound to name the key.

*Shows:* said (`by key`, counted among "not ruled out").
*Revisit:* build order 2: the absence rule, the logs narrowed by the callers' processes (the map's upward union; two
hops of `called_from` until then), the window's asks, and S2-S4 over the window for a writer that ran.

### K18. "Could have" needs every term known true

`_statement_true` takes a statement as surely returning the row only where each term is known true; never for a
statement that asks rows by a run-time value (the map's `keys`) or a term on a status history.  (A column holding
NULL is never true either -- SQL's NULL fails every comparison but `IS NULL` -- and `_leaf` says so: that is a
refusal, not a limit.)

*Shows:* said: such a mover is `undecided` or `by key`, not `could have`.
*Revisit:* status-history terms with the column values over the window (the map's (b)).

## H steps

### K10. H7 stops on rows files whose job row holds only the key

The rows builder puts the job's other columns under `after`, which the steps' environment does not
read (`Rerun.base_env`).

*Shows:* a stop at H7 at a branch whose guard reads a job column.
*Revisit:* with the map's work on which values hold at the time of a decision, and the shape of the
rows file's `rows` and `after` with the builder.

### K11. After H4 the next question is always H5, the traceback

`Rerun.set_by` returns `Raised` only.  A call H4 chose that does not fail by an exception has no other
way on.

*Shows:* a stop at H5 ("a traceback through <function> in the run").
*Revisit:* when a case reaches H4 with a value set without an exception.

## `which_ran`

### K12. A candidate whose several logs hold none of its runs is given the last log's reason only

`judge` writes the reason per log, each over the one before.

*Shows:* said, incompletely.
