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
For S1 (and S2 after it) the round is narrower (`recorded._partial`, 2026-10-26): S1 names the functions whose runs it
reads, and a DB proxy method's mark of none of them is not asked, nor one of theirs whose key records come in
pieces `EDGE` or more apart -- a mark the method's calls share (`getTasksToExecCommand_JEDI < vo=None label=None >`),
whose run is every call in the window: G's round asked 492 marks, 9 of them a writer's, and that one was cut at
1 MB in all 18 files (bamboo-e1).  Such a mark's key records are read as a run cut at both ends (`Log.partial`: rule
8 reads no row missing from it), so a mover whose run it is is followed on positive evidence only.  A shared mark
seen in one piece is still asked, and comes back cut.

## S1, S2 -- the movers of a stuck value (`Rerun.stuck`, `Rerun.passes`)

### K14. A mover that writes through what it calls is not followed

S1 counts a mover's arms in its own function only; `TaskBuffer.killJobs` writes through the proxy's
`killJob`, `archiveJob` and `retryJob`, and counts none.

- Such a mover that ran stops at S2: S2 and S3 judge the mover's own arms, and read none of what it calls.
- S1 asks the map's reach of each mover with no arm of its own (`Rerun._reach_of`, `reach_said`): from the def holding
  its statement -- every def of its name where it takes rows another selects -- does any function it can go on into
  hold an arm moving the value off?  It drops the mover (`reaches-no-write`) only where the answer is `[]`: no write
  reached and no call on the way left open.  It keeps one whose own defs hold an arm the junctions place there though
  S1 counts none, and one where the map could not read where all its rows go (a gap that may hide a write the reach has
  no junction for: any `ReaderGap` but a statement read only in part, whose gap is which rows it takes -- `hides_where`).
  Every gap is said, a part-read one too (`map-gap`, in the map's words).  A kept
  mover says why as before (`kept_without_arms`) and what the reach found: the writers reached, or the calls left open.
  A dump with no `reach_graph` keeps them all, and says so once.
- Kept for a call left open is not "can move it": the answer is `None`, not a write.

*Shows:* a stop at S2 for a mover that writes through its calls; each kept mover's reasons and reach in S1's output; a
dropped one's line, and its `reach` in result.json.
*Revisit:* following a mover's calls to their arms (S2 per callee).  The map closing holes (its precision items after
reach v1) moves movers from kept to dropped with no change here.

### K27. A value computed from a statement's rows in Python is read as the rows

S1 takes no mover from a statement the map says aggregates (`ReadSiteNode.result`), but that fact is the SQL's own.
A function that selects rows and computes from them in Python hands on values, yet its statement's `result` says rows,
and the map's carry holds a value computed from a part as the part, so its crossings are read as handing rows (the
map's U1, 2026-10-26).  `getScoutJobData_JEDI` 658 / 670 select one row per finished scout job and return medians,
maxima and averages of their fields; the callers reading them, `setScoutJobData_JEDI` and `retryTask_JEDI`, are
crossings, and S1 keeps both as movers of `JobSpec.jobStatus=finished` (K14), though neither moves a job's status.  No
case of mine asks `finished` (2026-10-26); not counted map-wide.

*Shows:* a kept mover in S1 that the rows reach only as a value computed from them.
*Revisit:* when the map says per crossing whether it hands rows or values computed from them (its U1).

### K28. A write keyed by a statement's rows is the statement's whatever column fills the key

Past a selecting statement's region, S1 takes an arm of the owner's as the statement's where the map says the
statement's rows fill the key the arm's write names its row by (`keyed_arms`, the map's `KeyFrom`, 2026-10-27):
`prepareTasksToBeFinished_JEDI` keys the writes of its later loop by the `jediTaskID` its statements at 3723 and 3768
select.  The map says where in a value the rows sit, not which column of them fills the key (its ii and U1), so a key
computed from the rows, or filled from a column naming another row, reads as the rows: the owner is kept as a mover
where it may not be one.  Nor is an arm's path condition read against the rows' values, here as in the region:
`getTasksToBeFinished_JEDI`'s 1705 runs only for a `scouted` row (`if taskStatus == "scouted":`), yet it is 1676's
for each value 1676 selects, and the function is kept as a mover of `prepared`, `passed`, `tobroken` and `toabort`,
which it hands on to the agent that moves them (`post_process_tasks`, through its return; on pin k, 2026-10-27).  An
arm whose key the map did not read (no bind map, no bind found, a value not followed) keeps the owner too, said.  So
does an arm set on an object no write in the function persists (the map's KeyFrom of no term, AK-2: the object's
fields handed to a call, bound in a statement, returned, kept in what the function was handed, or no use found):
there is no key here to read, and the write that persists the value, if any, is the caller's or a callee's; it is
said with the object and where its fields go, not as a key the map could not read.  On x13ak8k (the map's 5fab70b),
over every subject: 3 (statement, value) pairs, none of `JediTaskSpec.status` or `JobSpec.jobStatus` --
`datasetManager.run`@359's `FileSpec.status` `failed` / `merging` (713 / 745 / 958: a file of the job the function
hands on, which the map does not read as the job's, no use found) and `getJobs`@2743's `FileSpec.type` `output`
(2786: handed to a call and returned).  Not seen: rows reaching the key through a call (the map's v:
`setScoutJobDataToTasks_JEDI` names 3631's row by `taskSpec.jediTaskID`, the task `getTaskWithID_JEDI` returns for a
`jediTaskID` 3605 selects, and the arm reads as not 3605's), and an arm no KeyFrom names (`datasetManager.run`'s
858, `tmpFile.status = "ready"`; `getCommands`' 1824, an `IN` list of binds made by a join) -- such an arm is the
statement's only inside its region, as before.

*Shows:* the mover's line "<statement> selects the value, and its rows fill the key of the write(s) at [...]", or "...
the map could not read what fills the key of the write(s) at [...] (<terms and why>): kept", or "... no write in the
function persists the value(s) set at [...] (<object> (<where its fields go>)): kept" (the two joined by ", and "
where a statement has both).
*Revisit:* with the map's per-column carry (ii, U1); with its v, rows and objects followed across a call (a field
set on an element of a container handed on, a key of an object a call returns).

### K29. A birth is no arm only where the map says every write on its lines makes the row

A write that makes the row moves no row off the value, yet the map holds an INSERT's values as junction branches,
and S1 read each as an arm: `insertFilesForDataset_JEDI` INSERTs files ready, staging or pending (455 / 887 / 892) and
was kept as a mover of `JediFileSpec.status=finished` and `ready`; births stood among the writers by_key_of asks (the
movers oracle's rounds 3 and 4, 2026-10-27).  Now `arms_of` leaves out a branch each of whose lines (`line`,
`other_lines`) is written only by KeyFroms the map marks as making a new row (`births_of`, `is_birth`: the map's
`new_row`, its statement's mark on each KeyFrom of an INSERT), and says so with the map's why where the owner is
still listed (K29 in S1's entry and L2's line).  A writer each of whose writes of another value is a birth has no arm
left: it leaves the movers by key (K16) and L2's list as any writer with no arm does, and no line says it --
`getTasksToRefine_JEDI` (1184, registered) at `JediTaskSpec.status=finishing`, `ppEventServiceJob` (5556 / 5558 /
5560) and `makeFakeCoJumbo` (4509) at `JobSpec.jobStatus=activated` and `running`.  A line no KeyFrom names, an INSERT the map did not read, an entity moved within its class (an archive),
a row rewritten (DELETE and INSERT of its whole key), a call that updates the table too, and a line an UPDATE writes
too stay arms: errs on keeping.  An object of the table's class handed in is not followed to where its key was made
(None): `insertNewJob`'s new jobs, whose PandaID `TaskBuffer.storeJobs` takes from the sequence, stay arms.

The same fact answers who wrote a statuslog row (H2, F1): a write that makes the row writes no row that was there, so
where the statuslog holds a row of the key before the one asked, a reading's birth lines are no candidates, and a
reading left with none goes (`_unborn`, `held.row_before`): `ppEventServiceJob`'s new jobs, INSERTed activated with a
sequence's PandaID, did not write a job's 'activated' after its 'defined'.  For the row's first status, a tie at the
second the log keeps, or a value held beside the status (`JobSpec.computingSite`), every reading stays.

K14's keep reads the same fact: the map's reach (`reach.moving_off`) reads arms off the junctions' branches as
`arms_of` does but knows no KeyFrom, so a mover whose own def holds births alone was kept -- "its def holds an arm
moving it off by the map's junctions ..., though S1 counts none" -- `updateJobStatus` at `FileSpec.type=input` and
`pseudo_input` by its new FileSpec of type zipinput (434, KeyFrom 438).  The reach is handed the junctions without
their births (`unborn_junctions`), for a mover's own def and for the writers its calls reach alike.  On M 5aca5a4
no verdict moved by it: `updateJobStatus` is kept now because its reach leaves 2 calls open
(`push_job_status_message`'s `mb_proxy.send`, ...), and 27 reach texts lose writers of births alone
(`ppEventServiceJob`, `makeFakeCoJumbo`, `increaseSeqNumber_JEDI`, ...).

*Shows:* "<owner>: its writes at [...] make the row (births: sequence at 623): not arms"; result.json's `births`; a
writer with births alone is absent from the movers; H2 / F1: "the key's row was there before 'activated' at ..., so
its writes that make a row are no candidates: ppEventServiceJob [5556, 5560] (births: sequence at 5706)".
*Revisit:* when the map follows an object handed in to where its key was made.

### K30. A statement no placed caller runs persists nothing

The map says of each KeyFrom the routes into its statement's function (`routes`, as `Calls.routes` places them), those
on which the statement never runs by the constants the route hands (`out_on`: an if holding it, an exit before it),
and whether a route is left open (`routes_open`: a call by the function's name placed nowhere, a route cut).
`retryJob`'s in-place UPDATE (4146) is under `if not getNewPandaID:`, and every placed caller (TaskBuffer 1127,
copyArchive 704 / 814 through TaskBuffer 815) hands True: the old row keeps its status, its only write the error code
and diag (4202-4207), and what 4016 sets (`job.jobStatus = "activated"`) goes to the new row's INSERT alone.  S1 kept
retryJob a mover of six statuses by 4146 (the movers oracle's O5): `failed` by its keyed way, `assigned`, `defined`,
`pending`, `waiting` and `starting` by the rows `lockJobsForReassign`@846 and copyArchive's `main`@696 hand it.  Now
S1, H2 and F1 read no KeyFrom whose statement never runs from a placed caller (`never_runs`: routes > 0, every route
in `out_on`, none open; `key_froms`, the one place they read the records): 4016 is then written by a birth alone, no
arm (K29).  Measured on M's dump with the field put on its 5 such records: retryJob leaves `failed`; at the five
statuses handed to it, it stays as K14's (no arm of its own; its reach leaves 5 calls open: `file.pack`,
`file.bindUpdateChangesExpression`, `file.valuesMap` on an element of `job.Files`, ...), so the oracle's O5 holds
there until the reach places those calls.  H2 / F1: 4016 is no candidate for an 'activated' after a row before
(A-rec, N-rec: 7 -> 6 readings).  A statement with no route placed (the map leaves it open: reached from outside,
called from outside the corpus), one some route runs, and a dump without the fields are read as before.  A caller
outside the map's view (a test handing `simul=True`, a REST call) may still run such a statement: S1 errs on
dropping there, by the map's word that no placed caller does.

*Shows:* the writer is absent from the movers, as one with births alone (K29); H2 / F1: its line among the births a
later status leaves out.
*Revisit:* a statement some routes run, read per way: none of S1's rows rests on one on pin k (the map's iv,
2026-10-28), and the way would need the hand-off call's line (`lockJobsForReassign`@846's rows reach `retryJob` at
copyArchive 814, not from `lockJobsForReassign`).

### K16. S1 asks a writer that takes a row by its key over the window's last part only

S1's movers are the statements' (`movers_of`): the functions the rows of a statement selecting the value reach, and
the writers whose write requires the value (`guarded_by`).  A writer called with the key -- the pilot's report into
`updateJobStatus`, a kill command into `killJob` -- is in neither for `JobSpec.jobStatus=running`, whose
`guarded_by` is empty.  S1 takes them as L2 does (`by_key_of`: a writer with an arm moving the value off that no
statement gives rows) and follows the runs found, but asks their logs over the window's last part (`STUCK_SPAN`)
alone, not the rest a mover's is asked over: they are mostly the DB proxy's, and the rest is days of
`panda-DBProxy.log` on every server machine -- Fr2's 3.7 days on 8 machines about 35 GB grepped (tmaeno, 2026-10-26).
So a writer that ran and stopped before the last part is not named, and none of them is ruled out (K19's window).
`by_key_of` is every writer no statement gives rows, not only the per-key ones: on pin k (dce1e07's dump) 32 for
`jobStatus=running` and 28 for `status=finishing`, bulk methods whose selection the map does not link and the
setupper's helpers among them, against 3 and 1 movers.

*Shows:* said ("... from <the last part's start>, the window's last part, which alone is asked of a writer that takes
a row by its key"); result.json's `reached` `key`.
*Revisit:* when a stuck value needs the writer that ran and stopped: the rest asked, the smaller logs first.

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
print a string the asks search by).  Since build order 2a S1 searches the whole window for such a mover (K5).  The
third premise (`prints_naming`) holds where, before every arm, a line the map says runs on every path
(`Branch.must_prints`) has a hole it says holds the written row's key (`must_names`, the row's key being the rows
file's column holding the key asked).  The map reads such names off an attribute write's message only: it pairs
neither a run's mark (`PandaID={}`) nor an SQL arm's key bind with the written row (`mustprints`), so an arm with must
prints and no such name leaves the premise unknown.  On pin k (dce1e07's dump, out/p3 of the analyzer handover) the
premise can hold for 2 of the 22 arms of the movers S1 keeps for `JobSpec.jobStatus` (datasetManager's run 735,
retryJob 4016) and none of the 57 for `JediTaskSpec.status`; a mark holds a key column of the class on 7 and 39, the
written row's or not -- retryChildTasks_JEDI's is the parent's, its arm writes the children.  And most of the owners
S1 keeps (11 of 18, 9 of 13) have an arm whose runs print no row's key -- a bulk run marked `vo={} label={}` or by
time, or no line before it on every path: a search by the key cannot show such a mover absent at all.  result.json
says, for each mover with no run found, which of the three hold (`premises`: `window`, `logs`, `prints`), and
`did-not-run` where all do.
`JobSpec.jobStatus=running` (Fr2) had five such over the last 6 h: `Watcher.run`, `copyArchive.main`,
`TaskBuffer.killJobs`, `analy_pmerge_jobs_wait_time` and `get_site_strr_stats`.

*Shows:* said ("no run of its own found in [...] over the whole window: not followed, though not ruled out", or "...:
it did not run (judgement 3)"); result.json's verdict `no-run-found` or `did-not-run`, with its `premises`.
*Revisit:* when the map pairs a run's mark and an SQL statement's key bind with the written row.

### K25. S1 does not follow a function that prints nothing of its own

A function the map has no prints of its own for (`prints_nothing`) has no run of its own in a log, so S1 does not
follow it.  Before, a run found in its logs was read as its own wherever the run's mark named no method (`_of_mover`),
and S2 then took the function's one arm by elimination with nothing read: `JediTaskSpec.setOnHold`, no logs of its own,
was looked for in the selecting statements' -- TaskCommando's, which never calls it -- and Fo's TaskCommando run became
"every pass left by setOnHold:807".  Where such a function's write moves the value, the map places it at each call as
an arm of the caller's (JobGenerator.runImpl's 642 `pending`, setOnHold's own test in its path condition marked
`[in setOnHold()]`, K24), and the caller's run is read for it.  One whose write no caller's junction places is not
read at all.  On pin k (dce1e07's dump): 2 of the 60 writers that take a row by its key in Fr2's and G's values
(setOnHold, adder_gen's finalize_job_status) and 1 of the movers S1 keeps over the census (TaskBuffer.updateJobs, one
value of `JobSpec.jobStatus`).

*Shows:* said ("the map has no prints of its own for it, so no run in a log is its own: not followed"); result.json's
verdict `prints-nothing`.
*Revisit:* where a caller's junction does not place the function's write.

### K26. S3 reads what an arm that ran wrote only from a row after it in its own block

Where every pass leaves by an arm that writes the subject (S2 takes the arm itself), no guard kept the passes off
it, and S3 reads what the arm wrote: a `runtime(<name>)` outcome, by a row after the arm in its block that prints
`<name>`, nothing binding it between (`lines_after`).  The value observed in every pass ends the answer: the mover
ran and wrote it again -- Fr2's updateJobStatus, arm 241 in 12 passes, "old=running new=running", the pilot's
reports.  A literal outcome, several outcomes at the line, a name printed only before the arm or past a binding, or a
value other than the one observed stops S3 and says which.  The pass-wide printed values leave such a name out on
purpose (`stable_holes`, rule 17): updateJobStatus binds `jobStatus` again at 228, between 51's `status={jobStatus}`
and the arm.

*Shows:* END "...:241 ran in every pass (12) and wrote jobStatus = 'running', the value observed: the mover kept the
value"; else stopped, with what is not read.
*Revisit:* when an arm that ran writes a name no row after it prints.

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

## Row values at a step's time (`held.py`)

### K22. A row value read at another time than the step's run is not read

The rows file holds the row as it was read (`as_of`, days after the case), and a step reads the row as it was
when its run read it (`held.at_run`, through `Rerun.base_env` with the run's start).  A value the builder read
is taken for the run only where it is the row's key (`--key`); every other one, and every column of the rows
file's `after`, is left unread and says why: when it was read and how many statuslog rows lie between, a count of
rows and not of writes, since a write that moves no status leaves none (bamboo-68, 2026-10-18).  Which
writes between can change a column -- the run each statuslog row was written by, the writes that leave no row,
and the attributes and SET columns each run can change on the object -- is the map's to say
(rows-at-time-design.md §4).  Two readings stay as before: the hand's values (no `as_of`) are read whatever the
time, and so are a job's file rows (`files`), though the builder read them at `as_of` too: their writes are the
job's runs, and they wait for the same facts (tmaeno, 2026-10-18).  The row's creation time is read from `after`
whatever the time, as set once when the row is made (`Rerun._created`, H2's bounds): a premise the map does not
state yet.

Before such a stop, F1 (`Rerun.rows_between`) reads the first of the three: for each statuslog row between the
run's start and `as_of`, the run that wrote it, asked and judged as H2 asks and judges its own row -- the map's
writers of the row's status, their logs at the row's time on the row's host, by the key, every row's asks before
one stop (tmaeno, 2026-10-23, so that the logs of A's rows are recorded before they expire).  A run that prints
the key in none of its records is not found there.  On dc440d8 no writer is such: of the 120 arms with a same or
can pair at level 0 (`JobSpec.jobStatus` 53, `JediTaskSpec.status` 67), the writer's run carries the key in its
mark, or the function spells it in a print, for all but 7, and each of the 7, read in the code, is found by the
key: Watcher.run's mark is filled by `str(pandaID)`; the post-processors' run is their caller's,
PostProcessorThread.post_process_tasks, marked `<jediTaskID={}>`; Setupper.update_jobs prints counts only, in
Setupper.run's run marked by a time stamp, and the rows its arms lead to are written in activateJob's run, marked
`PandaID={}` (2026-10-24, analyzer-handover/out/lim-count.txt).  Asking the row's second instead would not be
enough: it returns every record of that second, and with no key in them the records do not tell which run wrote
the row -- that needs a run matched to the row by another column its mark holds (getJobs' `siteName` against the
row's `computingSite`).

*Shows:* a stop where a step needs such a value (on the record door, H7's implicit else at 785 for A and N, R5's
`job.jobMetrics` for K), saying when the value was read, after F1's section; silent for the file rows and the
creation time.
*Revisit:* with the map's facts: the attributes a run can set on an object and the columns a SET fills with
another value than the one at hand; the writes that leave no row, with the select and the caller each is reached
from, which can set a writer aside with no log asked (updateJobMetrics_JEDI takes only jobs a select found
finished); for F1, a writer whose runs print no key (count again when the pin moves), with its run matched to
the row by another column of the mark.

## H steps

### K20. H2 asks a value the history's rows hold over a window, on every machine

For a subject the status history's rows hold beside the status (`JobSpec.computingSite`), H2 asks the
logs from the row before the last stretch of rows holding the value (or the row's creation) to the
stretch's first row (`windows.value_held`).  A row holding the value says it was there by then, not
which run wrote it, so no machine is named; the pairs of a writer and the history row its run writes,
which would name the row's time and host, are the map's to say (build2-design.md 7, the second stage).
A value written after the last row has no row to close the window, and a task's creation time is not
checked (`parts.History.created`): both stop.  A subject no history holds stops as before.

*Shows:* a stop at H2 naming the column, the last row or the creation time; asks of every machine.
*Revisit:* when the map says which history row a writer's run writes.

### K11. After H4 the next question is always H5, the traceback

`Rerun.set_by` returns `Raised` only.  A call H4 chose that does not fail by an exception has no other
way on.

*Shows:* a stop at H5 ("a traceback through <function> in the run").
*Revisit:* when a case reaches H4 with a value set without an exception.

## `which_ran`

### K21. Runs of the key in several logs are one run only where one ran inside a call the other made

H2 reads a run of another log as a part of the question's run (`which_ran.inside_call`) where both name the same
machine, the map's reach goes from a function printed in the outer run into the inner run's candidate through a
call c of that function, and the inner run's records fall between the outer run's last record before c and its
first after (c's `must_rows`): DBProxy's archiveJob inside the adder's updateJobs (A, K on the record door).  Every
such c is tried, at each of its lines (`ReachCall.lines`), each asked of the map's reach on its own (`rerun._called_at`):
the adder's process_job_report reaches archiveJob from 400 and 471.  A log line names no process, so this is
inferred: another process on that machine calling the same function for the same key while the call was open would
be joined too.  Calls between the same two prints (the two arms of an `if`) cannot be told apart by the records, so
each whose pass holds the inner run is named.  A call with no print every path runs before it, or none after it
before the next statement that can leave the block (`must_rows` stops there), leaves the runs apart -- 400 above has
neither, so only 471 places archiveJob -- and so does the file door, whose files name no machine.  Apart, no run is
the question's, as before (Fr1, 2026-10-14).  The line before still says each candidate is judged on its own run.

*Shows:* "one run: <log> <first>..<last> ran inside the call at <file:line>[ or <file:line> ...] of the run in <log>
(between <record> and <record>), on <machine>", and rule 3 "runs inside the call at ..." for its candidates.
*Revisit:* when a log line names its process or thread; when the map says which prints precede a statement.

### K23. A function whose rows land in several files is read in one run of one of them

A candidate is judged on one run, of one log.  A row the map places only in other files than that run's is neither
shown nor missing there (`Row.lands_in`, `rows.absence_trusted`): its lines are not read, so it neither excludes the
candidate nor proves it.  The runs of the other files are not joined to the candidate's, as K21 joins runs only
inside a call.  On ced60ed, 2 of the 1093 functions with prints have rows in different files (`rows.split_files`):
pilot_api's `acquire_jobs` and `update_job`, in panda-PilotRequests.log and panda-api_pilot.log.  Neither writes: on
dc440d8 the map's reach gives them no assignment and no write, so neither is a candidate of H1, H2, F1 or S1, and no
step meets the limit today (2026-10-24).

*Shows:* said: "<function>'s rows land in different files [...]: a row is read only in a run of a file it lands in (KNOWN_ISSUES.md K23)".
*Revisit:* when a writer's rows land in several files (compare `rows.split_files` with the writers when the pin
moves); when one pass's runs in several files can be joined by the key, the time and the machine.

### K24. S3 names no guard that no test above the arm says, and reads no callee's test nor a later write's

S3 (`Rerun.blocked`) names the guard the passes made false against the nearest arm, then goes on from the `if` that
tests it (`guard_if`): one above the arm whose test is the guard, or for `not (t)` one testing t with the arm in its
`else`.  A branch's path condition also holds guards no test above the arm says.  A local's branch carries the
bind's guards beside the assignment's, its line the assignment.  A later write that replaces the value adds the
negation of its test (`pathcond.literal_values`; the map's PC adds more for fields, setter calls and map keys).  Named,
such a guard stops S3 at "the if whose test is ...", so S3 withholds, never answers wrong.  The dump holds each entry
as text only, and one function can hold several `if`s of the same text (`storeJobs`' `match is not None` at 267, 288
and 305), so the querier cannot place them itself.  A guard the map marks `  [in f()]` (`alias._produced`) is tested
in a method the function calls, in that method's names.  One the map's PC marks `  [at N]` is a later write's test
reading a name given a new value after the branch's write (the write's own place among them), its names as they
are when line N's test runs; a pass holds one value per name for the whole pass (`stable_holes`), which may have
been printed before the write, and read so the entry could be false where the later write never ran.  S3 counts
either neither false nor true (`E.read_elsewhere`), as any guard the passes leave unknown: on an `[at N]` entry S3
is as it was before PC, and can name a guard whose flip alone would not move the value.  On 5e84330's dump
(`tools/s3_guards.py`, 1,202 branches, 2,803 entries) 80 entries of about 31 branches have no test above the arm,
207 of 43 branches are marked `[in f()]` and none `[at N]` (PC not built); S3 meets `[in f()]` guards on
JediTaskSpec.status at registered, rerefine and staged (TaskRefiner.runImpl), none of them in a case of mine
(2026-10-25).

*Shows:* stopped: "the if whose test is '<guard>'" for a guard with no test above the arm; a marked guard in an arm's
"needs [...]" and never in "false in the passes".
*Revisit:* when S3 must go on from such a guard: the map would record each entry's place (its `if`'s line, or the
replacing write's, and for a callee's test the method and how its names map to the caller's); for an `[at N]`
entry, when a reader holds a name's value at a line rather than for the whole pass.
