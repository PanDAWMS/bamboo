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

### K6. H6 finds a maker only in a run that holds one of the job's rows

H6 asks the make calls' logs at each of the job's statuslog rows, at the row's second on its host
(`windows.rows_until`), and takes a run there that printed the name at a make call and names the job.  A maker that
wrote no row of the job -- a run that made the object before the job was created, and wrote none of its rows -- is
not seen.  The window was the Datasets row's `creationdate` before; that row is of the first run to use the vuid
(insertDataset writes none where one is), not of this job's, and gave JEDI's make calls no window at all.  The door
over files knows no file's machine: a run in any file given that holds the row's second is taken, and the row's
line says "in the files given, their machine not known", not the row's host.

*Shows:* a stop at H6, "a make call whose function printed the object's name, in a run holding one of the job's
rows and naming the job", and the line of each row asked.
*Revisit:* when a case's maker wrote no row of its job.

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
A mark of theirs whose rows below WARNING all start with fixed text (`rows.mark_heads`) is asked for those rows and
every line at WARNING and above (`recorded._narrowed`, 2026-10-31; the harness's to-querier-63): a call of
getTasksToBeProcessed_JEDI was 382,661 lines in MT-ready-stuck's round 5, 381,600 of them its callee's, one per task
listed, and a second of them stopped the recorder's paging.  A run of such a mark holds no other function's line below
WARNING, and rule 8 and E6 read such a row as not searched (`Run.holds`; said on the run's line).  A message the
shapes read as a leading hole (an f-string's expression, a conditional) leaves the mark asked whole: insertNewJob's
rows start with `job.PandaID`.  A narrowed ask is not read from a record of the whole mark cut at 1 MB over its window
(`recorded._cut_over`): it is asked anew, so the next round reads what the cut left unread -- where the record says what
was read for each cut string (fieldtest ca43fb9's `read`), only if a string the ask needs is cut, and an ask that
drops the string that filled the stopped second is read from it whole (`recorded._read_for`).  H2 still asks every mark
whole (it reads a candidate's rows in any run of the key, and a writer handed its caller's logger prints under the
caller's mark); by the harness's count over rounds 1-5 (to-querier-64), only getTasksToBeProcessed_JEDI's marks come
near a page's stop, and a round that stops on one is left to the recorder's split of the stopped second.

### K40. A truncated file with no `searched` is read up to its last record, whichever string's

A file is `truncated` where any string of its ask was cut at 1 MB, while the ask's other strings ran on to the file's
end; read up to its last record returned, it counts as searched the stretch past a cut string's last record, where
that string's records were never fetched (and the cut record's second).  On pin m's records (the harness, 10-30 20:4x):
157 of 315 truncated entries end on a record after a cut string's last, 99 of those over the ask's window --
MR-T-53354737-queuedTime reads aipanda081's panda-JediDBProxy.log-20261029.gz as searched at 10-29 07:43:04, though
`getTasksToRefine_JEDI < vo=None label=managed >` was cut at 10-28 07:52:08.  A record saying where every string was
read (`searched`) is read only there, and a later round writes such records again with `replaces` (tmaeno: "OK", 10-30
20:5x).
*Shows:* silent: the "not searched" lines of H2 and S2 leave the machine out, and a log read as searched whole with no
run of the key there reads as no writer's pass.
*Revisit:* when the harness's round rewrites every truncated entry with `searched`; the records snapshots taken
before it keep the overread.

## The rows ask (`Rerun.ask_rows`)

### K36. Rows the rows file does not have are asked, and nothing of the querier's answers them

Where a step stops for rows the rows file does not have, it says them as an ask a fetch can answer, written to
`--rows-asks` (none unless given), each once, `{"rows": {...}, "step", "why"}` (the harness's rows ask, stage 1,
10-29).  Four places ask: a walk's loop over a row's collection the file lacks (`Walk.elements`: `collection`, the
row by its class and the key the dump's tables name its identity by, and the attribute; the element's class is
`null`, not known here), S5's call whose rows, history or counts the file lacks (`statement`: the proxy's statements
the call runs as `<file>:<line>`, the tables it reads, each argument's value in each pass -- `"unknown"` where the
run's is not known -- and the passes' times), L3's answers the file lacks (`sql`: the question's name and SQL, a
GroupsNow's `db_read` form -- table, WHERE, the columns grouped by, the key it takes the least of -- and none for the
window over the history), and a statement S1 or L2 compares with a class the file has no row of (`Rerun.ask_linked`,
the 4th place, the harness's to-querier-57: `collection` with the attribute `null` and the class named, `where`,
the class's columns the statement's joins compare with the row's, with the row's values -- the task's datasets by
jediTaskID -- and `via`, each pair by name as the rows file's `linked` names one (`from` the row's class, `to` the
class asked); one per pair of sources the joins meet by).  The 4th place does not stop: the statement stays `not
settled`, as before.  It asks only where the rows could change the verdict (not for a statement `refused`), and
not for a class joined to the row only through another table, a class-less table (`T_TASK`: its join side names
no attribute) or a join column the row does not hold.  Nor does it ask rows the file's `linked` holds by one of
those pairs for the row's value (a job's files, `JobSpec.PandaID -> FileSpec.PandaID`): S1 reads one row per class
(`rows`), not `linked` or `files`, so its "no row of FileSpec" means none in `rows`; it says the file holds them,
which no step compares yet.  On the harness's stuck cases of 10-29 (runs/20261029-150512-m-r1)
its 51 not-settled judgements lacking rows (JediDatasetSpec 50, FileSpec 1) all join the row directly
(`out/nrow/nr0-summary.txt`); JediFileSpec and T_TASK appear there only in refused ones; the FileSpec one's rows
are in its file's `linked` (the harness's reading of f99a537's ask, 10-29).  The fetch that answers
them is the harness's (stage 2); until it reads them the file keeps its sections as now.  A stop no other rule
names, with rows asked, has the code `rows` -- not `asks`, which the harness records the logs for.  An iterable the
walk does not read as a row's attribute asks nothing.

L3's `db_read` form gives the WHERE as db_read's conditions, `[column, op]` or `[column, op, value]`, all of which
hold (the harness's ask 4, 10-29: a WHERE in text is not passed to db_read), where every term is one column tested
against a value: each term read as the map reads a WHERE (`sqltext.row_tests`), a bind's value the run's as its
print holds it, a quoted constant a text and another a number (`parts.condition`).  db_read takes `=`, `!=`, `<`,
`<=`, `>`, `>=`, `in`, `is null`, `is not null`.  A term that is none -- an OR, `NOT IN`, an expression of columns,
the database's clock, another column, a value that is no number or text -- keeps the form's WHERE text, and L3 says
each such term under the SQL ("db_read takes no ..."): a term is said whole or not at all, since dropping one
counts rows the call did not pass.  db_read refuses a value whose type is not its column's (`bad_value`); the
querier gives the run's, not knowing the column's type.  On pin m one statement reaches L3, getJobs@2404, and only
D and D-rec of the querier's and the harness's cases reach it, with three terms, all said.  The harness's records
hold 293 of getJobs' prints: 291 of that shape; 2 (the unified pilot's, in N-rec and A-rec) with
`(minRamCount<=:minRamCount OR minRamCount=0)` and `(maxDiskCount<=:maxDiskCount OR maxDiskCount=0)`, which keep
the text (out/dw/dw-predictions.txt).

*Shows:* `rows_asks` in result.json (`path`, `count`); a stop coded `rows`; "-- db_read takes no ..." under L3's SQL.
*Revisit:* when the harness's fetch answers them (stage 2); when the rows file's old sections go (stage 3).  The 4th
place's answer is many rows of a class (a task's datasets), and S1 compares one row of each class: reading it needs
a comparison that takes the row where some joined row holds the terms, which is not built.  When a case reaches L3
with an OR: db_read could still answer it, the OR's columns grouped by and the groups that fail the OR dropped by the
querier, or the OR's branches asked apart where no row can pass two; neither is built.

## S1, S2 -- the movers of a stuck value (`Rerun.stuck`, `Rerun.passes`)

### K14. A mover that writes through what it calls is not followed

S1 counts a mover's arms in its own function only; `TaskBuffer.killJobs` writes through the proxy's
`killJob`, `archiveJob` and `retryJob`, and counts none.

- Such a mover that ran stops at S2: S2 and S3 judge the mover's own arms, and read none of what it calls.
- S1 asks the map's reach of each mover with no arm of its own (`Rerun._reach_of`, `reach_said`): from the def holding
  its statement -- every def of its name where it takes rows another selects -- does any function it can go on into
  hold an arm moving the value off?  It drops the mover (`reaches-no-write`) only where the answer is `[]`: no write
  reached and no call on the way left open.  The reach closes a call left open where no function of the corpus of
  its name can lead to a write (`closed_by_name`, the map's O1); a drop that rests on that names the calls closed.  It keeps one whose own defs hold an arm the junctions place there though
  S1 counts none, and one where the map could not read where all its rows go (a gap that may hide a write the reach has
  no junction for: any `ReaderGap` but a statement read only in part, whose gap is which rows it takes -- `hides_where`).
  Every gap is said, a part-read one too (`map-gap`, in the map's words).  A kept
  mover says why as before (`kept_without_arms`) and what the reach found: the writers reached, or the calls left open.
  A dump with no `reach_graph` keeps them all, and says so once.
- Kept for a call left open is not "can move it": the answer is `None`, not a write.
- A mover whose statements' rows reach none of its arms is asked so too, those arms left out of where the reach ends
  (K28); where it writes nothing so, it is a writer that takes a row by its key (K16).

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
select.  Whatever column of the rows fills the key counts, and an arm's path condition is not read against the rows'
values, here as in the region.  The map's v says both on each `rows` source -- the select item filling the key
(`column`), whether the write names the rows themselves by the table's whole key (`same_rows`), the values for which
it never runs (`never_at`) -- and S1 does not read them here: on x13v4k (M 03c3237), of the 55 arms keyed so over
six subjects, 47 are keyed by a `same_rows` source, 5 by a part of the key from the same column (the rows and more:
`throttleUserJobs` 1746, `unThrottleUserJobs` 1844, `restartTasksForContentsUpdate_JEDI` 1839 / 1884,
`resetFileStatusInJEDI` 2083), 3 by a column the map did not read (`insertFilesForDataset_JEDI` 1116 / 1143), none
by another column and none at a value in `never_at`: reading them would move no row.  The example here before,
`getTasksToBeFinished_JEDI`'s 1705 under `if taskStatus == "scouted":`, kept as 1676's at `prepared`, `passed`,
`tobroken` and `toabort`, holds no more: its branch says the row precondition `status=scouted`, so `arms_of` leaves
it out at those values (`refused_arms`).  An
arm whose key the map did not read (no bind map, no bind found, a value not followed) keeps the owner too, said.  So
does an arm set on an object no write in the function persists (the map's KeyFrom of no term, AK-2: the object's
fields handed to a call, bound in a statement, returned, kept in what the function was handed, or no use found):
there is no key here to read, and the write that persists the value, if any, is the caller's or a callee's; it is
said with the object and where its fields go, not as a key the map could not read.  On x13ak8k (the map's 5fab70b),
over every subject: 3 (statement, value) pairs, none of `JediTaskSpec.status` or `JobSpec.jobStatus` --
`datasetManager.run`@359's `FileSpec.status` `failed` / `merging` (713 / 745 / 958: a file of the job the function
hands on, which the map does not read as the job's, no use found) and `getJobs`@2743's `FileSpec.type` `output`
(2786: handed to a call and returned).  Rows reaching another function's key through calls (v's sources with `via`)
are not read as keying its arms: on x13v4k, 16 such records, each of a function S1 has as a mover of the value
already.  Not seen: an arm no KeyFrom names (`datasetManager.run`'s 858, `tmpFile.status = "ready"`; `getCommands`'
1824, an `IN` list of binds made by a join) -- such an arm is the statement's only inside its region, as before.

A statement of the owner's that takes its rows by a key the statement's rows fill hands them on to a write keyed by
its own rows (`key_chain`, through the map's read KeyFroms): `resetInputToReGenCoJumbo_JEDI` sets ready at 5038 the
files of 5026's rows, 5026 takes the job's files by the PandaIDs of 5009's rows, and 5009 takes those by 5001's
`datasetID` and `fileID` -- another table at each step, so neither the one hop nor `read_again`'s same rows saw that
5035 is 5001's (the movers re-scan's O6, 2026-10-29).  Whatever column fills a key counts at every step.  The other
way, an arm whose row the map names by a key it read whole, at every step, that the rows do not fill is not the
statement's.  Where every way S1 has of a mover is a statement of its own whose rows reach none of its arms so
(`elsewhere_arms`, `elsewhere_of`), S1 asks the map's reach of it as of a mover with no arm of its own (K14), those
arms left out of where the reach ends (`Rerun._rows_elsewhere`); where the reach writes no other value with nothing
left open, the mover is a writer that takes a row by its key (K16), said: `updateJobStatus`'s 294 selects the
co-jumbo jobs `waiting` and sets their modificationTime alone, 262 the jobset's `activated` ES jobs alike, and its one
arm, 241, writes the pilot's own job by the `PandaID` it is handed.  On M d62d62d's census (lm10) the class holds 5
(value, mover) rows: those two and `updateJob` at `JediFileSpec.status=running` (596 checks input files in one branch,
618 writes output files in its `elif`) move to by key; `insertFilesForDataset_JEDI` at `finished` stays (its reach
leaves 2 calls open), said; `resetInputToReGenCoJumbo_JEDI` at `running` is keyed through its chain.  An arm whose
key, or a key on the way, the map did not read keeps the mover as before.

Rows that can be the statement's count as its own (`same_rows`; the map, 2026-10-31): a statement on the way that
takes one table by a term on a column the selecting statement compares too, every such term of the two filled from
the same sources and none a value the map did not follow.  `insertFilesForDataset_JEDI` sets lost at 1113 files of
757's rows, the dataset's, and 923 selects the dataset's ready and pending files by the same `jediTaskID` and
`datasetID` (datasetSpec's): a file 923 selects can be the one 1113 writes, and read as another row S1 moved the writer
to by key at `ready`.  A term only one of them compares narrows its rows, not where they come from.  On C4b
(x13xiiiC4bm), over the nine subjects the movers census asks, 18 (value, statement, write term) rows are keyed so and
not by the statement's own rows; that one moved S1's verdict.  The others' owners are movers by another way --
`throttleUserJobs` at `activated`, `unThrottleUserJobs` at `throttled`, `resetFileStatusInJEDI` at `finished`,
`insertFilesForDataset_JEDI` at `pending`, `_read_unprocessed_inputs` at `ready` -- and only the mover's line can
move.  A write whose terms are filled from the values the selecting statement compares, with no statement's rows, is
not read so: `updateJob`'s 619 names its file by the `job` file's fields 597 compares, but 597 and 619 are an `if` and
its `elif`, which S1 is not given; read alike, 619 would be kept wrongly.  A term the two fill otherwise is not
read either (`insertFilesForDataset_JEDI`'s 283 takes the parent task's files by `parent_tid`), nor rows of another
table naming the same object (`getTasksToExecCommand_JEDI`'s 4339, a command naming its task: K16's writer by key).

*Shows:* the mover's line "<statement> selects the value, and its rows fill the key of the write(s) at [...]" -- with
"(<arm>: <term> by <function>@<statement>'s rows, which take the same table by the same values)" where only rows
that can be the statement's fill it --, or "...
the map could not read what fills the key of the write(s) at [...] (<terms and why>): kept", or "... no write in the
function persists the value(s) set at [...] (<object> (<where its fields go>)): kept" (the two joined by ", and "
where a statement has both); for rows that reach none of the arms, "the rows <statements> select(s) reach none of its
arms [...]: ..." -- "a writer that takes a row by its key" or "; kept: <the reach>" --, the entry's reason `K16` or
`K14` with the reach.
*Revisit:* when a census on a new map finds an arm keyed by another column or at a value in `never_at` (v's fields
make both readable); with objects followed across a call (a field set on an element of a container handed on, a key of
an object a call returns); with the branches of one turn read as exclusive, when terms filled by the same values can
be read alike too.

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

An arm on an object no row holds is no arm either.  The map's OS (ix) says on a branch that the records prove its
object is held by no write of its class's tables, or by none that takes it whole or by the attribute (`off_row`,
`no_write` or `not_taken`: a spec the function makes and drops, `getTaskWithJumbo_JEDI`'s useJumbo, `checkAllEventsDone`'s
FileSpec), and its readers -- derive-strategy's readings (H1), `reach.moving_off` -- leave it out.  The querier's
door to the writers (`Rerun._writers_of`, which S1, L2 and the mover census read) leaves those branches out too
(`on_rows`), and a junction left with none, its owner no writer.  On ix's map (x13ixO1m, 20 such arms, none on a
status) one census row moves: `checkAllEventsDone` at `FileSpec.type=input`, kept by its arm 1275, has no arm of its
own and reaches no write, so it is dropped (out/ix/or-predictions.txt).  A dump built before OS has no `off_row`.

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
dropping there, by the map's word that no placed caller does.  The other way, a route the map places counts
whatever calls the function it comes from: `makeTaskPending_JEDI`'s one route is JEDI's facade
(`JediTaskBuffer.makeTaskPending_JEDI`, routes 1, none open, no `out_on`), and the facade's one call is in a comment
(`AtlasTaskWithholderWatchDog` 118), so S1 keeps it a mover of `JediTaskSpec.status` `ready`, `running` and `scouting`
by its guard (the movers oracle's O2, 2026-10-25; read again 2026-10-29).  Telling a chain no entry runs from one an
entry outside the corpus does needs the callers above it to an entry point (the map's upward union, K15).

*Shows:* the writer is absent from the movers, as one with births alone (K29); H2 / F1: its line among the births a
later status leaves out; a writer with no caller above its route stays a mover, said nowhere.
*Revisit:* a statement some routes run, read per way: none of S1's rows rests on one on pin k (the map's iv,
2026-10-28), and the way would need the hand-off call's line (`lockJobsForReassign`@846's rows reach `retryJob` at
copyArchive 814, not from `lockJobsForReassign`).

### K31. Rows read again are handed on whatever test the call runs under, and to a function that may write other rows

S1 follows a selecting statement's rows into a statement that reads them again (`read_again`; the map's v: a read
KeyFrom whose `rows` source is the selecting statement's, `same_rows`, the value not in its `never_at`) and hands them
on from there as from a selecting statement's region, to writers only, one hop: `getTasksToBeFinished_JEDI`@1676 ->
1731 -> `post_process_tasks`, a mover of `passed`, `prepared`, `tobroken` and `toabort` (the movers oracle's A11).
It overclaims in two ways, as the region does:
- a call is taken whatever test it runs under.  `prepareTasksToBeFinished_JEDI`@3723 -> 3896 (`sqlRT`) ->
  `setScoutJobData_JEDI`, whose call at 3958 runs under `taskSpec.status == "scouting" or jediTaskID in
  set_scout_data_only or (taskSpec.status == "ready" and ...)`; `set_scout_data_only` holds only tasks `sqlEA` adds,
  and 3723's are not among them (`jediTaskID not in jediTaskIDstatusMap`), so for 3723's rows the call runs at
  `scouting` and `ready` alone, and S1 keeps `setScoutJobData_JEDI` a mover of `merging`, `preprocessing`,
  `throttled` and `running` too.  What decides it is a set's membership, filled under a test on another statement's
  rows: a `never_at` on the call, read as a source's is (a test on a name holding the rows' column), would rule none
  of the four out.
- a function reached may write other rows than those it is handed: `reassignJobsInPreassignedTask_JEDI`@5755 ->
  `killJob` 991 (through 5759) -> `killEventServiceConsumers` and `killUnusedEventServiceConsumers`, which kill the
  job's consumers, kept as movers of `JobSpec.jobStatus` `activated`.
On x13v4k (M 03c3237), over six subjects: 12 (value, mover) pairs come -- 6 right (`post_process_tasks` x4,
`setScoutJobData_JEDI` at `scouting` and `ready`), 4 and 2 as above -- and no other mover, why, arm or reach moves;
of the cases, D's L2 alone, where `killEventServiceConsumers` and `killUnusedEventServiceConsumers`, by key already,
are given 5755's rows.  Handed on to every function the statement reaches, writers or not, 35 more pairs came, each for
K14 to drop or keep by its reach, and D's L2 no longer ruled out `updateRelatedEventServiceJobs` (its arm refuses
`activated`) nor `recordStatusChange`: hence writers only.  A miss costs more than an overclaim here: a mover kept
is asked, one left out is not looked at again.

A selecting statement's own crossings no longer overclaim so where the map reads the routes back up: a crossing takes
only the values those routes select (`crossing_values`, the map's `crossing_selects`, XB, 2026-10-29).  1898 in
`_get_tasks_datasets_with_unprocessed_inputs` adds `tabT.status IN (statusForJobGenerator())` only where no
`target_tasks` is given, and its rows reach `setCloudToTasks_JEDI` only back up TaskBroker 259, which hands
`simTasks`: S1 kept it a mover of `JediTaskSpec.status` `preprocessing`, `running`, `scouting` and `topreprocess`, and
no more; `getPandaIDsWithTaskID`@95 filters failed / cancelled / closed only under `unsuccessful_only`, which task_api
432 hands `killJobs` False.  An owner the map gives no values (reached from the statement's function, by a way no
route matches, or from a statement with no route read) takes all the statement selects, as before.  A re-read's
crossings are not narrowed so: the map gives the field on a selecting statement, and a statement reading rows again
by their key selects no value of its own.

*Shows:* the mover's line "<statement> selects the value, and <statement> reads its rows again and hands them to it",
or "..., and they fill the key of the write(s) at [...]".
*Revisit:* with a reading of the test a call runs under on the rows' values, through what a set the function fills
holds; with whether a function reached writes the rows it is handed (its write's KeyFrom).

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
setupper's helpers among them, against 3 and 1 movers.  A mover whose rows reach none of its arms, and whose reach
writes nothing else, is taken as one of these (K28): `updateJobStatus` at `waiting`, its 241 the pilot's own job.

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

### K25. S1 does not follow a function that prints nothing of its own, and H2 stops where no writer has a log

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

H2 reads such a writer, where the map's history pair places its arm (`attr_time.arms`), at the pair's caller: the
call on the way to it (`HistoryPair.point`) is the candidate, judged on the caller's run (`Rerun._h2_candidates`,
`StandIn`; attr-time-shape.md 5).  The call is shown run by a row on every path through it, or by two of the caller's
rows printed in a row that only a path through it prints (`which_ran.rows_around`, `rows.passes_through`:
TaskRefinerThread.runImpl's 205 and 563 around 211, extractCommon).  That says the writer was called, not that its arm
wrote: the arm's own path condition is read in the row's values, and what is not read -- or, above level 1, the
functions between the call and the writer -- is left open, a premise of "may have written" (K38).  Counted on the
harness's attribute cases (pin m, x13M110m, 2026-10-29, out/attr/attr-pass-summary.txt): 41 of the 46 arms that print
nothing are placed, at 24 calls; at 10 of them two rows can show the call ran, at all 24 two rows can stand around it.
A writer no pair places is asked in the logs its reading names (`rows.own_log_files`).  Where no writer of the subject
has a log -- each prints nothing, no pair places it and its reading names none -- H2 stops on that before it asks
(`_writer_logs`); before, it asked no log and stopped on "a run of <key> at <t>", as if the logs had held none.  Only
where the map places the prints (`ROWS_FROM=graph`, `ROWS_OWN_FILES`): else the readings' logs stand.

*Shows:* said ("the map has no prints of its own for it, so no run in a log is its own: not followed"); result.json's
verdict `prints-nothing`.  H2: the caller's line among the candidates, "rows a and b printed in a row, which only a
path through it prints"; "a log to look for a run of a writer of <subject> in (<functions>: the map names none;
KNOWN_ISSUES.md K25)".
*Revisit:* where a caller's junction does not place the function's write; for H2, when the functions between a
level-2 pair's call and the writer are read.

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

### K43. A -late question whose value has several stays needs `--since`

L1 asked the first stay of the value; a value can come back many times (a task between pending and running), and a
case picks one stay -- the case set's rule, the longest stay begun in the 5.5 hours before its SQL -- so 5 of pin m's
17 -late cases were answered about another stay (MT-exhausted-late: the 240 h from 10-19 for the 1.7 h of 10-29;
MT-defined-late's stay is the 65th of 71).  `--since` names the stay by the time_extended of the row that begins it
(`rerun._stay_of`; tmaeno: "OK", 10-31, and not to pick up every pending <-> running bounce).
*Shows:* a stop at L1 that names the stays (how many; the first, the last, the longest) where the value has more than
one and no stay is named.  The cases carry it once the harness writes `question.since` and fieldtest hands it on.
*Revisit:* when a question means every stay, or the longest, rather than one the case names.

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
checked (`parts.History.created`): both stop.  A subject no history records or holds takes no time from
the status's rows, though its values may be spelled as the status's: until 10-29 H2 took the status's row
holding the value as the subject's time (`JediTaskSpec.superStatus` 'done', three of the harness's cases;
`oldStatus` too).  Its writers' arms are asked instead (K37).

*Shows:* a stop at H2 naming the column, the last row or the creation time; asks of every machine.
*Revisit:* when the map says which history row a writer's run writes.

### K37. An attribute no history records is timed by its writers' arms, and only where every arm says the time

`attr_time` (the harness's ask b, tmaeno: "please do", 10-29): where every arm H1 names writes the time it runs
(`runtime(CURRENT_DATE)`, `runtime(naive_utcnow())`), the value is the write's time and H2 asks the run at it, on
every machine (`windows.value_now`); else each arm is placed on the object's history by the map's HistoryPair (row
same or can), the statuslog rows the placed passes can write are asked in one round (rows holding the statuses the
history call records where they are literals, any row otherwise), and read newest first (K38): the first whose pass a
writer is shown to write in is the write that left the value, the answer (below).  Only the last write is
asked; one of the same value before it is not.  An arm on no row of the object -- its pairs are another object's,
the map does not follow the object to its pass's history call, or it has none -- may have written the value after any
row.  Since 2026-10-31 two such arms are read all the same (tmaeno: "build both", 13:2x).  A birth (`born`, K29: the
INSERT that makes the row) wrote as the row was made: its row is the scan's last, at the row's creation (the map's
`tables[].creation`, the rows file's value), asking nothing, an arm that may have written there and is never the
answer by elimination -- getTasksToRefine_JEDI's INSERT INTO JEDI_Tasks (1209) writes taskName as a uuid4 placeholder
the value then held need not be.  An arm whose function marks its runs by the object's key (`marked_by_key`: a
`function_prints` run whose mark's key holds the class's identity) is asked over the object's life, from the row's
creation to when the value was read, on every machine (F3, `windows.lifetime`), for its mark as its rows' prefix
prints it with the key (`key_marks`: "changeTaskSplitRulePanda < jediTaskID=53351604 >", "; jediTaskID=53351078;
siteID="), not the key alone -- a task's ID is on every line of the calls listing it (getTasksToBeProcessed_JEDI), and
the door would then ask each such mark whole (K13; the harness, 10-31 14:00).  Each run found there whose mark holds
the key is read as a row is, by its first record's time among the rows (`_runs_as_rows`), asked for the same marks; a
run naming the key in its lines alone is another call's.  A life with no run, its logs shown to
name the key and searched whole, is no pass of those writers, and one not searched whole leaves them open, newer than
any row.  Any other unplaced arm stops H2 at "every writer of <subject> on a history row of the object, a birth or
marking its runs by the object's key ... (k of n not: K37)" before anything is asked: its absence from a log cannot be
shown.  Counted on C4b's map (out/k37c, tools/k37_inserts.py): of the 264 unplaced arms of jobs and tasks, 9 are the
one birth's (getTasksToRefine_JEDI's INSERT), and of the other 255, 133 are in a function marking its runs by the
object's key, 85 by another object's (changeTaskPriorityPanda's <jediTaskID> for a job's currentPriority), 26 have
no run of their own (printing through a caller, K25), 11 a mark with no key (doForPriorityMassage's "label=user").
It also stops where H1 names no writer (a value the pilot reports through updateJob's SET, which the map does not
read).  A name bound to
the time earlier in the pass (`timeNow`, `now_time`) or the time less a span (`CURRENT_DATE-1`) is not read as
the time.  Counted on the harness's 57 (attr-time-shape.md): every arm placed 23, some 13, none 18, no writer 3;
the map not following JobGeneratorThread.doGenerate's jobSpec back to runImpl's storeJobs leaves 12 of the 18
(asked of the map, 10-29).

A row is asked, and read, of the logs of the writers whose passes can write it alone (`attr_time.row_writers`,
`readings_of`; the harness's to-querier-61, step 1, tmaeno: "go with 1", 10-30): before, every row was asked of every
writer's logs, so a writer that can write any row kept every log in every window.  No row is left out by it.  Which
rows a writer's pass can write is still read from the history call's recorded statuses only: a pass that records the
status at run time can write any row, though it moves the object on only from the statuses it picks it up at --
TaskRefiner's pass from registered, rerefine and staged.  Telling those from a writer that takes its row by a key from
outside (updateJobStatus, the pilot's PandaID) needs the key's source of the pass's status write (step 2, not built:
S1's census lists TaskRefiner.runImpl as a writer by key at 31 task values, and updateJobStatus as a kept mover at
activated and waiting, so neither its kept movers nor its writers by key tell them apart).

The pass H2 answers with is when the attribute took the value (attr-time-shape.md 6, P5; tmaeno: "OK", 10-30 15:03):
result.json's `wrote` names the arm, whether its run shows the line ran (`shown`; else H2 left it by elimination: the
one writer of a value that is its time), the call it is read at for a writer that prints nothing (K25), the statuslog
row whose pass wrote it (none for a value held or a value that is its time), the window asked and the run, and H3 ends
there (END) where the arm has no guard or every guard holds in the run's values.  A condition not read, or read false,
is named in H3's stop; a guard read through a method goes on to H4 as for a status, whose steps say why the arm ran,
not when.  Before, an attribute went on as a status does and stopped at H3 "a method the guard reads through" for an
arm with no guard: 12 of the 14 attribute cases H2 answered on the harness's round-2 snapshot.

An arm left by elimination names no run (`wrote.run` null), and its guards are read in the rows' values alone, its
H3 stopping at "a run that shows <arm> ran" where they do not all hold: no run shows it, and the run the judge read it
on is any of the key's at the window, on any machine -- a server's panda-DBProxy.log run (insertTaskParamsPanda, a
day long, the key on one line) for rescueUnLockedTasksWithPicked_JEDI and get_tasks_for_periodic_action, JEDI's
(the harness on xii's T, 10-30 20:2x; tmaeno: "fix it in xii").  Its END names the writer's logs not searched at the
window instead.

*Shows:* "-- row n of m" in H2 and its judgement per row, "<writer>:<line>: n row(s)" for each of several writers;
"END <subject>=<value> written last by <arm>[, called at <call>] (shown in its run; it has no guard | its guards hold
in the run's values), in the pass of the statuslog row <status> at <time>: <run>", or "(by elimination: no run shows
the line ran; it has no guard | its guards hold in the rows' values), at <time>[; not searched there: <log> on
<machines>]"; "<arm> ...; a birth: it wrote as the row was made" or "; its runs are marked by the object's key",
"F3: <arms> asked over the object's life, from the row's creation at <t> to when the value was read at <t>",
"-- row n of m: a run of <key> at <first>..<last> (F3)", a possibility "may have at the row's making <t>" or "at the
object's life <t>" (result.json's possibility rows carry `f3`: made, run, span or open); a stop "every writer of
<subject> on a history row of the object, a birth or marking its runs by the object's key ... (k of n not: K37)",
"the object's life, to ask <arms> over: <why>", "when the row was made, for the birth <arm> ...", "a writer of
<subject> in the map", or at H3 "a method the guard reads through (its conditions: '<condition>' not read)" or "a run
that shows <arm> ran, to read its guards in (by elimination; its conditions: ...)".
*Revisit:* when the map follows an object back to its caller (the JobGenerator way); when F3 is to read a writer
marking its runs by another object's key (the job's jediTaskID, which its row holds); when a writer only reads the
row into its spec (getTasksToRefine_JEDI and the like) is told apart.

### K38. H2's scan of the rows names the arms that wrote last or may have, and goes on from none of them but one

Over the statuslog rows a writer's pass can write (K37), newest first, each row's pass is read three ways
(`Rerun._scan`, `_row_at`, attr-time-shape.md 5; tmaeno: "a Fork over the possibilities, I'd say", 10-29): an arm shown to
write ends the scan; a row whose logs, searched whole, hold no run of the key, or whose arms are excluded, has no
write and goes; an arm the run does not show ran (kept), whose own condition is not read (K25's stand-ins), or whose
logs were not searched whole may have written and stays, with what is open named.  Before (querier-at), a row where
the judge chose no writer was passed over as "no writer ran" -- for a judgement that could not decide too -- and an
older row's write was taken as the last.  Where one arm is shown to write and nothing newer may have, H2 goes on
from it; else it stops naming each arm left with its rows, code `one-of`, and result.json's
`possibilities` lists them (arm, rows, premises).  Arms, not rows, are the possibilities: the steps after H2 explain
an arm, and an arm's passes differ only in the run read.  An arm whose passes are all older than the newest shown
write is not among them.  None read on any row: "a pass that wrote ... (none shows a writer's pass)".  A single
window (a status's row, a value held, a value that is its time) is judged as before, and a stand-in chosen there
is read the same way.  A stand-in's arm shown to write goes on as an arm that prints does (attr-time-shape.md 6,
P2; tmaeno: "the order does not matter much, so let's keep clearing the second stage", 10-30 14:47): H3 reads its
own guards, on the caller's run, whose passes and record after the call stand for it (`Wrote.via`).  One writer
that prints nothing can be called twice in one pass, once shown and once not (setSuperStatus_JEDI from
updateTask_JEDI and from getTasksToBeFinished_JEDI): the arm and the row are one, so the answer stands and the
other calls are said; which of them was last is not read.  The steps after H2 run for no arm of several: the
second stage's fork, whose branches are the arms and whose undecided branches end open rather than stopping the
run, is not built.

No run of the key at a row is no write only for a writer whose log some row's records show naming the key -- a run of
it there, on any machine (`Rerun._named`).  JobGenerator's pass marks its lines `<jediTaskID=... datasetID=...>` and
prints no PandaID, so its log holds no run of a job it made: the 12 job attributes it writes were read as no writer's
pass at every row (the harness's to-querier-62; tmaeno: "please do (b)", 10-30 13:0x).  A writer not shown so
may have written at each row with no run (`_unshown`), its premise "no line of <log> in the records holds <key>".
Asking its log by the key its mark holds (the job's jediTaskID, which rows.json has) needs the map to say what fills a
print's prefix; asked of the map 10-30.  Which of a pass's arms wrote last where several did is not read either: the
order of their points in the caller (TaskRefiner.runImpl calls extractCommon at 211 and doRefine at 284, in its
loop over tasks) says it only where no loop inside the pass holds both, so it waits for the run's records to say it.

*Shows:* "-- row n of m", "wrote: <arm>" / "may have written: <arm> (<premises>)" per row, "no run of <key> at
<time>: no writer's pass of a log that names it" where a writer's log is not shown naming it; "<arm> wrote, read at
<call>; in the same pass also <calls>, the same arm: which call was last is not read"; a stop "one arm that
wrote <subject>=<value> last, of k left: <arm> (wrote at 'status' time | may have at ...); ... (KNOWN_ISSUES.md K38)".
*Revisit:* with the second stage (the arms' fork, and H2's single window's "which candidate ran" with it); when the
map says what fills a print's prefix (then ask a writer's log by its mark's key).

### K39. S1 names what a mover's rows wait on only where the map states it, and does not ask whether it ran

A mover's statement can select its rows by another column too, one other functions write: ContentsFeeder's 130
takes a task at defined only while its dataset is toupdate or defined, and restartTasksForContentsUpdate_JEDI's 1844
sets the dataset toupdate for the tasks its 1830 selects at defined.  The map records these on the read site
(`ReadSiteNode.wakers`, ix's 6 b, A only), and `wakers_of` takes, for each statement giving a mover rows, those the
map states (the branch's outcome is the value) and whose write is handed rows selected at the value (`keyed_at`).
S1 says each under its mover where the mover is not dropped, and says a mover it drops as reaching no write that is
one of them as their waker -- restart at task defined and at dataset toupdate, which S1 dropped with nothing more
before.  A waker keyed by a statement whose function moves the subject on from the value itself is said a candidate:
that pass may take the row on before the mover sees it (prepareTasksToBeFinished_JEDI's 4001 / 4006 for JobGenerator's
1871 at task running, keyed by its own 3694; the map's own doubt).  Not said: a write the map does not state (tier
2), one keyed by rows of other values, and what the map does not read -- a command row (getAchievedTasks_JEDI's finish
into PRODSYS_COMM, read by TaskCommando's 4339: the map's B) and a Python test on a row read by key
(AtlasAnalWatchDog's `tmpFileSpec.status == "finished"`: C).  So no waker said is not "nothing wakes it".  S1 does not
ask whether a waker ran for the row, nor read the row's other column against the term.  On pin m (x13ixA1m) the
census has 17 kept movers naming 41 wakers, 13 of them with a candidate, and 4 dropped movers said as wakers, all at
the task's and the dataset's statuses.

*Shows:* "<mover>: its rows wait also on <waker>@<line> writing <subject>=<value>, which <reader> asks ('<term>'),
for rows selected at the value" (" -- a candidate: ..."); "<waker>: not a mover, but it writes ...: its waker";
result.json's mover entries' `wakers`, and a `K39` reason on the dropped one.
*Revisit:* with step 2 -- asking a waker's runs for the row, and the row's other column against the term (on tmaeno's
word); and with the map's B and C (map_todo MV).

### K11. After H4 the next question is always H5, the traceback

`Rerun.set_by` returns `Raised` only.  A call H4 chose that does not fail by an exception has no other
way on.

*Shows:* a stop at H5 ("a traceback through <function> in the run").
*Revisit:* when a case reaches H4 with a value set without an exception.

### K33. A traceback of another version of the code is placed by its frames' source text, where it is unique

A run's traceback prints the lines of the code it ran, and the pin is the code the case is read against: since the
pin moved to panda-server 32907fe6 (tag 1.1.0 and #841, 2026-10-29), the replay cases' logs (written by 1.0.x on
09-28 .. 10-14) are read against other code, and a log of 1.1.0 itself is too: #841 moves lines in 8 files
(`adder_gen.py`, `job_complex_module.py`, `base_module.py`, ...).  `records._place` reads a frame whose line holds
other text at the one line of its function -- every def of its name in the file -- holding the source the traceback
printed, keeps the traceback's line (`Frame.moved_from`), and H5 says it ("frame <file>:<the traceback's line>
<function> is line <n> of the pin by its source text").  `rows.raised_in` places such a frame the same way.  A
source line printed twice in the function, or not found there, leaves the frame unplaced, as before -- the version's
own lines are not known -- and H5 says, on the line before its stop ("the frame's source line matches the corpus"),
the lines of the function holding it ("at no line" or "at lines <a>, <b>").  The text matching says the same
statement, not the same code around it: a guard that changed between the versions is read as the pin has it.

No case's frame has moved so far.  K's stop on the new pin is not a moved line: its statement is gone.  8ec5f109
("Fix KeyError in decompose_id_map for alternatively staged-out outputs", 2026-09-30) is K's own failure fixed, and
pin m's `decompose_id_map` holds the printed `sub_to_dataset_map[tmp_dataset]` at no line.  So K and K-rec, the only
cases that reach R1-R5, stop at H5 on the pin.  The other position the cases are keyed by is a rows file's answer
to L3 ("getJobs@<line> ..."): `data/D-rows.build.py` (the analyzer handover's, not this repo's) writes D's answers
under the statement's line on both pins.

*Shows:* H5's line for each frame placed so; H5's "holds the printed source at no line" before a stop where the run's
statement is not the pin's.
*Revisit:* when a case's logs come from a version whose function no longer holds the printed line, or holds it twice
(a pin of the case's own version); when another step keys a reading by a line of the code; when a current case
reaches R1-R5.

### K34. An exception relayed from another process is read as raised inside the corpus

`pandajedi/jedicore/Interaction.py`'s `__call__` hands a method call to a child process over a pipe and raises the
class the child returned with the child's string (`raise retException(strException)`, line 272 on pin m).  The
traceback the parent prints ends at that line, in the corpus, while the exception came from what the child called:
in an example of 10-29, `AtlasTaskSetupper.doSetup`'s `registerNewDataset` failed by Rucio's ORA-20101.  H5 takes an
innermost frame in the corpus as the line that raised and goes on to R1-R5, which would read the relay's `raise` as
panda's own failure; the boundary is the pipe, as WrappedCursor -> oracledb is one H5 sees.  No case has reached it,
and that example likely would not (not run): doSetup's except returns a code and sets no status, so H5 would find
no traceback through the function H4 chose.  A traceback raised by panda's own line comes from a bug and is rare
(tmaeno, 10-29); R1-R5's cases are K and K-rec, read on the pin of their version (8e24aa1a-f2), since K's bug is
fixed on the current pin (K33).

*Shows:* R1 reading `Interaction.py`'s `raise retException(strException)` as what raised.
*Revisit:* when a case's traceback ends in a relay of another process's exception.

### K35. A binding in a loop over rows the investigation does not have is unknown

`false_guard.Walk.elements` gives the rows a loop around a step ran for, from the rows file's collections.
A collection the file does not have (None) is kept apart from one of no rows (`[]`): a binding of a name
in a loop over the first is an origin of unknown value and reach, so the name's other bindings are not
taken as all it held (`Walk._choose` takes none of them alone); a keyed write there leaves the key set
opaque and the container not rebuilt, as before.  A loop over no rows ran no write: the container gains
nothing from it, and its key set can be whole.  The file's rows are taken as the run's, as for any
collection.  Until 10-29 `origins` left such a binding out and a loop over no rows read as one over rows
not had.  `G.part` still matches the printed key by value, and a binding of unknown value beside it does not
make the match two.  H7's implicit else read without the job's file row says the row is missing before its
stop.

*Shows:* "@<line> runs in a loop over rows this investigation does not have" among a name's origins in R2.
*Revisit:* when the harness's fetch answers the walk's ask for the collection (K36, stage 2).

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
before the next statement that can leave the block (`must_rows` stops there: the next statement calls too, or the
prints after it are in arms), is placed by the run instead (`which_ran._between_records`): the outer function's two
records in a row around the inner run, none of its rows read between them, and c on a path from the one's print to
the other's that prints nothing else (`rows.runs_between`, the function's own statements: blocks, arms, loops, a
call raising to its handlers).  Other calls can run between the two records, as between c and its print before
already; only those the map's reach takes to the inner's function are named, so the join rests on the reach for the
rest.  On x13vi1k such calls are half of what the join tries between two logs of a status's writers, and 9 of
JediTaskSpec.status's 33 values had no other; a path holds 129 of the 132 lines with no print after, 1 of the 21 with
none before (out/h2b/h2b-predictions.txt, h2b2 and hb1, 2026-10-29).  A call no such path holds, or a function that
prints nothing, leaves the runs apart: the adder's 400 (the prints above it each return, so only 471 places
archiveJob).  So does the file door, whose files name no machine.  Apart, no run is the question's, as before (Fr1,
2026-10-14).  The line before still says each candidate is judged on its own run.

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

An arm's guards are `pathcond.path_condition`'s, the tests of the `if`s enclosing it, and since 2026-10-31 each early
exit's above it that `pathcond.enclosing_guards` names (`if t: return` puts `not (t)` on what follows;
`rerun.arm_guards`), S3 going on from that `if` (`early_exit_if`): check_output_file_metadata's arms 200 / 209 after
the return at 181 the pass took are kept off by `not (self.job_status in [...])`.  S4 then stops at "one name the
guard reads" (it says "reads []"): it follows bare names, and this guard reads the attribute `self.job_status`, bound in
`__init__` (48, `self.job_status = self.job.jobStatus`), which its walk in the function does not reach (s3a,
MJ-transferring-stuck on records-cut; the only such S4 in the four sets).  An arm in an `except` handler has
no guard to show false -- it is reached only where something above raised -- and S3 stops at "a guard the passes made
false against every arm (none shown against arm N (in an except handler))", the need it could not meet, not a claim
(parse_job_output_report 820, MJ-merging-stuck on records-cut, 7a05464).  And a
pass that leaves a loop at its terminator is read as leaving at that exit: setupper's assign_dispatch_data_blocks
walks `self.jobs + [None]` and leaves at 324 (`if job is None: break`) after every job, so S2's "left by 324" and S3's
{'job is None': True} are the call's, not the key's iteration (the investigator's round 2, 2026-10-31).  Neither
gives a wrong answer: S3 stops.

*Shows:* stopped: "the if whose test is '<guard>'" for a guard with no test above the arm; a marked guard in an arm's
"needs [...]" and never in "false in the passes".
*Revisit:* when S3 must go on from such a guard: the map would record each entry's place (its `if`'s line, or the
replacing write's, and for a callee's test the method and how its names map to the caller's); for an `[at N]`
entry, when a reader holds a name's value at a line rather than for the whole pass.  S3 to read enclosing_guards and
to say its stop as the need it is (promised to the map, 2026-10-31); a loop's terminator, when S2 reads a key's
iteration rather than the call.

### K41. P reads the arm's own function's rows, by what a call is handed, and a positive says the arm ran

P (`printed.py`) reads a run against the arm's own function only.  Of the 103 pairs of possibilities the
investigator read in the possibilities task (round 1, 2026-10-30), the print that tells them apart sits in the arm's
function for 96 and in a function it calls for 7 (doBasicPostProcess's getTaskStatus_JEDI 2596 and
get_event_statistics 4633 in task_standalone_module; archiveJob's check_fine_grained_processing 1945): those are not
read.  What a call changes is read from what it is handed -- by name, and below ``self`` by
chain (`printed.self_effects`, the core's (c)) -- so an object stored elsewhere earlier and changed through that alias
is not seen, nor is a local bound to the target's object in the walked function (``job = self.job`` then
``job.jobStatus = ...`` binds ``job.jobStatus``, not the target).  (c) resolves ``self.m()`` with the map's
``calls.Types`` (the class, its corpus subclasses, each along its order): a base outside the corpus listed before the
class in a subclass's bases, defining the same name, is not seen.  The second P (the core's (d), `printed.Reaching`)
rests on the candidates being every writer of the observed value: on a map without the writers of a value known at
run time only (RD, codemap-rd), a call that writes the target from such a value is not seen.

A positive says the arm ran in the run.  Where the value or its row is written by a later call of the pass (for a
status the map's `HistoryPair`, reachable and not taken; for an attribute set in memory, the call that stores the
object), a test in the callee can refuse the write and the caller go on unaware: ContentsFeeder's
updateTaskStatusByContFeeder_JEDI writes nothing where the task is locked by another process or its status is not one
for contents update (task_complex_module 1324-1331), and the caller's print after the call reads the same.  On
x13xiiC2m (the 65 texts, the attribute 60, MT; 2026-10-31), of the 14 arms judged positive, 9 are DB proxy methods that
write the value with their own statement (5 of them leave the history row to a later recordStatusChange or
record_task_status_change); 2 are stored by the adder's updateJobs -- process_job_report 441 (32 texts) and
adder_atlas_plugin execute 170 (6), whose runs print rows only after updateJobs returned success; 2 by TaskRefiner's
registerTaskInOneShot_JEDI -- runImpl 211 (14 texts) and AtlasAnalTaskRefiner doRefine 260 (1), where a refusal prints
"failed to register the task to JEDI in a single shot", in none of the 14 runs, an absence P does not read; and
doBasicPostProcess 114 (endTime, 1 text) by updateTask_JEDI, whose result it does not read: one arm in one text that
no row can tell.

*Shows:* H2's verdicts: P's reason, and "read with the other verdicts: a call reaching no writer of '<value>' left
standing does not write it" for a positive of the second P.
*Revisit:* when a pass's runs of a called function's log are read as the caller's (K21's join, inside the call); when
the map holds a call's write conditions; when RD lands.

### K42. A record two rows print alike proves neither, and rule 6 reads a caller's guard on any object of the class

`rows.Seen.proven` counts a record only where the function reads it with that row alone (`Seen.unambiguous`, after
`rows.claim`): activateJob prints its files' UPDATE alike at 92 (in the arm that activates) and 124 (in the one that
does not), and such a record had made 71 positive (the investigator's round 2, 2026-10-31).  The record proves
neither row even where the other's path is ruled out on its own, or its place in the run rules the other out:
parse_job_output_report prints "parse_job_output_report end" at 750 (before the early return 751) and at 966.  On
records-cut (out/pv1) MJ-merging-stuck and MJ-transferring-stuck lose 967's positive, though 751 is excluded by rule 4
and the record follows rows 869-873, which 750 cannot follow; those branches stop at S2.

Rule 6 (`which_ran.rewriter`) solves the callers' guards too (`callers_guards`) and reads `<x>.<attr>` in them as the
key's object whatever x is: archiveJob 1783's `job.jobStatus == 'finished'` tests the merge job, while
updateUnmergedJobs 1236 writes the jobs it merged.  On records-cut before pv1 it excluded 1236 in MJ-merging-late (the
key 7390523881, the merge job 7390910703).  With the row's second read run by run (pv1) the env no longer settles
that guard there and 1236 is positive; the defect is not fixed.

*Shows:* "row(s) [..] print only on a path through it" names only rows a record is read by alone; "rule 6: reached
only when <x>.<attr> is already ..." with a caller's guard.
*Revisit:* when an ambiguous record can be given to the one row the run's order or another rule leaves (P's order
check reads rows read alone too); rule 6, when the map says which object a caller's guard tests against the callee's
argument.

## The walks (`asked.py`)

### K32. The walks read the code alone, not the instance record the map was built with

The querier's walks (`asked.walk`: the map's `trace.Asker` over `asked._types()`; under `ROWS_FROM=walk` the rows'
skeleton too) read the corpus with `Types(ROOTS, {})`: a module name holds what the code says.  The map built the
dump with an instance's record (`MapFragment.instance_record`, `calls.read_recorded`), which the dump names by its
path and digest only.  On x13vi1k the record is ATLAS production's `g_pluginMap`, empty, so `panda_config.getPlugin`
returns None at its three callers (adder_gen 174, setupper 109, closer 126) and the ATLAS plugin class is taken; the
walks read getPlugin's `type[Any] | None`, so the plugin constructor's binding names every plugin class of the name
and says the call may land somewhere the reading cannot name, where the dump's reach has the ATLAS one.  The walks
name more, never less: the steps and the questions are the same.  By the mover tables no S4 walk starts in those
files -- the census's movers there have no arm (AdderAtlasPlugin's process_subscriptions, register_files and
update_outputs, Closer.run, of FileSpec.type) -- and the 307 guards S4 walks from in 75 movers' functions read the
same with the record and without, as do two rounds of every question from their arms (out/ir/ir-predictions.txt,
ir1 and ir2, 2026-10-29).  The steps that read the plugins otherwise -- H1 to H6, F1, S1's reach -- read the dump.
Not measured: R2's and value_diff's walks, which start where a case's traceback or diff is, past two rounds.

*Shows:* silent that the record is not read; where a stop line prints the binding (S4, R2, value_diff), its "returns
from ..." names every plugin class and "the call may also land somewhere this reading cannot name".
*Revisit:* when the census gives a mover in a plugin's file an arm, the record holds more than `g_pluginMap`, or a
case's answer rests on such a binding: the map would put the record's values in the dump, and `asked._types()` would
hand them to `Types`.
