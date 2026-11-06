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

### K13. The round that asks a run's mark reaches only the key's records near the window

The records are not cut to the window (a widened ask reuses them, and a rotated file's record holds the key's lines of
its whole day), so the key's records include runs of other hours.  Since 2026-11-04 `RecordedLogs._read` asks the mark
that does not hold the key only of a key record within `EDGE` of the window.  Before, it asked every such mark, over
the window: in the analyzer handover's six sets 25 of the round's 65 asks named a mark with no line of the key in the
window +- 10 min -- 15 runs after it (periodic marks: getTasksToBeProcessed_JEDI 7, getTasksToExecCommand_JEDI 4), 7 on
both sides and none in it, 3 before (insertNewJob's marks at the job's creation, 7 min to 11.6 h before a stuck
window) -- and of the harness's round 20 most were AtlasProdWatchDog's stamps and setCloudToTasks_JEDI's mark 4-8 h
after MT-staged-late's window (out/l2bk/mark_round_windows.py).  None was a run that named the key before the window's
slack and went on into it.
*Shows:* such a run -- one that names the key at its start, more than `EDGE` before the window, and writes the row
inside it -- is not asked: the window holds its key-less lines only, and they are read as no run of the key.  Such a
mark's run is the key's records alone; since 2026-11-05 it is read as cut, as a run is judged whole by what was searched
for the strings that return its records (`Log.whole` takes the mark), and nothing searched for the mark's.  Before, it
was judged by what was searched for any string, and with no entry row among its records a candidate's pass spanned from
a key record before the window to one after it: in m-cut2's MT-defined-late, reactivatePendingTasks_JEDI:4704 was kept
at 12:30:50 on its records at 11:29, 13:06 and 13:17, where the whole mark showed none of its 133 passes there.  That
arm is no candidate: its UPDATE requires status 'pending' (the map's `row_precondition`), and the row before held
'defined'.  H2 does not read the precondition; S1 and L2 do (`arms_of`).
*Revisit:* measure, per log, how long a mark's run goes on past its last record of the key; the functions whose pass
prints the key that far before its write are the map's to name (below).
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
(`recorded._settled`): its strings hold the cut one and more, read fewer lines a second, and may pass where it stopped,
so it is asked anew -- where the record says what was read for each cut string (fieldtest ca43fb9's `read`), only if a
string the ask needs is cut, and an ask that drops the string that filled the stopped second is read from it whole
(`recorded._read_for`).  An ask of the cut string itself is not asked inside its record's window +- `EDGE` (2026-11-05):
the recording step reads a cut string on from where it stopped there (`--again`), or cannot -- MT-ready-stuck's and
Tr's getTasksToBeProcessed_JEDI marks, cut at the second of their own call in a rotated file, 10 asks the harness found
no round could fill (to-querier-120) -- and the part left is a hole of the mark's run.  H2 and F1 name their
candidates to the door as S1 does (2026-11-02): they judge a candidate on its passes alone, and a run whose mark names
another method is none of them (`which_ran.passes_at`, `of_mover`, K47), so a mark of a method that is no candidate is
not asked.  Before, they asked every mark whole, for a writer handed its caller's logger, whose rows under the caller's
mark no judge read: 6 of the harness's round 16's 7 asks were `push_job_status_message < ... >`, a mark of no
candidate, whose run is one line already held, each a grep of a whole DBProxy log of 3.0-3.2 GB, 3 of them timed out
(to-querier-107).  A candidate's mark they still ask whole (`Logs.log`'s *whole*): neither read as cut for the calls
sharing it nor asked for its own rows alone, which S1 needs over a stay's hours.  At a row's moment a shared mark's run
is one call, and read as cut it lost the must rows MT-exhausted-late's choice of getTasksToExecCommand_JEDI:4442 was
shown by (13 records at 11:14:22-23, whole in the record); counting only its pieces near the window instead asked the
mark over hours once the run was widened, and S1 then asked getTasksToBeProcessed_JEDI's marks over a stuck window's 6
hours (both measured, the analyzer handover's out/k13).  A candidate's rows under a caller's method mark (a logger
handed in) are read by neither H2 nor S1; how many such rows the map holds is not counted yet.
A run near an end of what was searched widens the window past it once, and the end is one the asks over the window
itself searched (`recorded._runs`'s *own*, 2026-11-03).  Before, it was any record at hand: a mark asked for a row's
moment was answered from the record of the round before's widened ask (`_within` prefers the machine's record), and a
periodic method's mark (`rescueUnLockedTasksWithPicked_JEDI < vo=atlas label=user >`, a pass every few minutes) is one
run as far as any record reaches, so each round asked it EDGE further back: MR-T-53351604-rescueTime's 081 went 13:09:14,
12:59:13, 12:49:12, 12:39:11 in rounds 5-18, and round 6 took three machines to their files' starts (the harness's
round 18).  A run reaching past the window's own search is read as cut there.  So is one reaching past the search of
an ask answered from other asks' records (`recorded._held`, 2026-11-05): those records are not the window's own -- a
widened ask's record among them would widen again each round -- while a new ask's record would be, and widen once.
The answer needs every second of the ask's window searched, not EDGE past it.  Since 2026-11-04 only a run whose mark
holds the key is widened.  A mark that does not hold the key is shared: a periodic method's, the setupper's stamp,
or the first word of a line with no mark, which the records read as its mark (ContentsFeeder's "failed to get taskSpec
for jediTaskID=..." -> `failed`).  Its run is read as cut at that end, and the log's notes say how many.  The pass that
decides sits within a second of its row: MT-toretry-late's 19:19:29.607 to .611, rescueTime's 082 13:09:13.450 to
13:09:14.128 (the harness), and superStatus's hour of prepareTasksToBeFinished_JEDI decided nothing.
*Revisit:* a function whose pass prints the key only farther before its write than EDGE.  Its window then holds only
key-less lines of the run, and widening is what ties the run to the key.  The map is to say which functions do this,
from their prints and the write (asked by the harness, 2026-11-04); the door does not read it yet.  And a run near an
end of what other asks' records searched, read as cut where a new ask's record would have widened past it: count them
on the measured sets before deciding whether such an answer should widen from the files meeting the window alone.

### K40. A truncated file with no `searched` is read up to its last record, whichever string's

A file is `truncated` where any string of its ask was cut at 1 MB, while the ask's other strings ran on to the file's
end; read up to its last record returned, it counts as searched the stretch past a cut string's last record, where
that string's records were never fetched (and the cut record's second).  On pin m's records (the harness, 10-30 20:4x):
157 of 315 truncated entries end on a record after a cut string's last, 99 of those over the ask's window --
MR-T-53354737-queuedTime reads aipanda081's panda-JediDBProxy.log-20261029.gz as searched at 10-29 07:43:04, though
`getTasksToRefine_JEDI < vo=None label=managed >` was cut at 10-28 07:52:08.  A record saying where every string was
read (`searched`) is read only there, and a later round writes such records again with `replaces` (tmaeno: "OK", 10-30
20:5x).  Such a record answers no ask but one of its own strings from 2026-11-05 (`recorded._settled`): another ask is
asked anew rather than read past the cut.
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
run's is not known -- and the passes' times), L3's answers the file lacks (`sql`: the question's name, its SQL and
its structure, `question` (`parts.asked`): the part, the moment or the window, the terms read then and those read
now, the columns grouped by, the key and how it ranks, the columns an answer holds -- and no `db_read` form, both
being windows over the history: K48), and a statement S1 or L2 compares with a class the file has no row of (`Rerun.ask_linked`,
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

*Shows:* `rows_asks` in result.json (`path`, `count`); a stop coded `rows`.
*Revisit:* when the harness's fetch answers them (stage 2); when the rows file's old sections go (stage 3).  The 4th
place's answer is many rows of a class (a task's datasets), and S1 compares one row of each class: reading it needs
a comparison that takes the row where some joined row holds the terms, which is not built.

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

S1 follows a mover only where a log asked for it holds a run of the key in the window asked (`of_mover`).  One whose
every ask was answered and none held such a run is not followed -- but not ruled out either: absence needs the rule
of K15 (the whole window searched, every log its process writes to, and every run that reaches the arm bound to
print a string the asks search by).  Since build order 2a S1 searches the whole window for such a mover (K5).  The
third premise (`prints_naming`) holds where, before every arm, a line the map says runs on every path
(`Branch.must_prints`) has a hole it says holds the written row's key (`must_names`, the row's key being the rows
file's column holding the key asked).  Since xviii the map reads such names in the message, the run's mark and an
SQL arm's bind map (`MustName.part`), and says whether it read them over every must print whole
(`Branch.must_names_whole`).  Where it did and none names the key, the premise does not hold (False): a run need not
print what the asks search by.  Where it did not, an arm with must prints and no such name leaves the premise
unknown (None).  On xviii's dump (out/j3l of the analyzer handover, l3e's runs), 411 of the 1114 S1 movers with
premises move from None to False so (True 311, False 271, None 121 stay); no verdict moves, since `did-not-run`
needs True.  The False errs one way only: a print holding the key where the map does not see it (TaskCommando.py:155's
mark holding the value the key is assigned from; a hole printing a whole object, `str(job)`) keeps a mover from
being ruled out, never rules one out.

Judgement 3 is one part, S1's and L2's (`Rerun._absence`): its asks over the window's last part, the part before it
for a mover not reached by its key, and its premises.  The window is one rule with its end a parameter
(`windows.last_part`, `rest_of`): stuck's as_of, late's move.  L2's use is K15's.

Before xviii, on pin k (dce1e07's dump, out/p3 of the analyzer handover), with the names read off an attribute
write's message only, the
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
it did not run (judgement 3)"); result.json's verdict `no-run-found` or `did-not-run`, with its `premises` (prints:
"the lines before the arm(s) at [...] were read whole and none names the written row's [...]").
*Revisit:* when the map sees a key a print holds through another value (a key chain, a whole object printed).

### K25. S1 does not follow a function that prints nothing of its own, and H2 stops where no writer has a log

A function the map has no prints of its own for (`prints_nothing`) has no run of its own in a log, so S1 does not
follow it.  Before, a run found in its logs was read as its own wherever the run's mark named no method (`of_mover`),
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
rows printed in a row that only a path through it prints (`which_ran.rows_around`, the map's `flow.passes_through`:
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

### K15. A writer that takes a row by its key is ruled out only by judgement 3, and one that ran is not read

Every writer with an arm that can move the value (`arms_of`) gets judgement 3, S1's part (K19), over the stay
(`Rerun._late_absence`; tmaeno, 2026-11-01).  The window is S1's rule with the move for as_of: from the stay's row
less SLACK, no earlier than the move less STUCK_SPAN, to the move, and the part before it for a writer not reached
by its key.  Ruling one out needs more than finding no run: the whole window searched in every log its process
writes to (a DB proxy method writes `panda-DBProxy.log` or `panda-JediDBProxy.log` by the process that called it), and
its runs bound to name the key.
- Its logs are asked only where the two premises the map decides hold (every log to hand, every run printing the
  key); otherwise a run not found would rule nothing out, so it stays `by key` (or `could have`) with the premise
  said.
- On today's cases (l3e's results, xviii's dump, out/j3l): 63 of the 360 (case, mover) pairs L2 leaves are asked (66 hold both
  premises on the records door; D runs on the file door with two logs).  A
  stay longer than six hours (D 152.6 h, Fr1 93.2 h) is asked over its last six for a writer reached by its key, so
  nothing is ruled out there.
- An ask no record answers is kept for the asks file and the walk goes on: L2's movers are the answer's context, so a
  late case ENDs with asks.
- A writer that ran on the key in the stay is said, not followed: which way its passes went -- why it passed the row
  over -- is not read for late.

*Shows:* said ("by key -- ...; its logs are not asked: a run not found would not rule it out (...)", "ran -- ...; ran: a
run of the key in [...] between ... and ..., and L1 names another mover", "asks unanswered -- ...; N ask(s) of its logs
not in the records (...)", "did not run", "no run found", "not searched", "prints nothing"); result.json's movers'
`verdict` and `premises`; "did not run" is not counted among "not ruled out".
*Revisit:* S2-S4 over the stay for a writer that ran (the writer that moved a late value on is read so, judgement 4,
K45); a stay longer than six hours asked whole; the logs narrowed by the callers' processes.

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

### K48. L3 counts the rows held ahead at the move, some of their terms as they are now

L3 (`Rerun.ordered`) reads a value that moved on late because the statement that took the row returns only the first
rows of one of its orders.  It counted the rows that pass the call's WHERE now (GroupsNow), grouped by the other
order's columns with the key left out, so a late case got today's rows, not the ones it lost to, and an order of the
key alone ('pandaid asc', MJ-activated-late) left no column and stopped.  Now (tmaeno: "OK", 11-01):
- the order, one rule: its items before the key group the rows, the key ranks them within a group, and the items
  after it decide nothing (the key is unique).  The key alone is one group.  An order without the key leaves its
  ties in an order the database picks: its own group is said, not counted ahead;
- late asks the rows that held the value at the move, the window's end: those whose last history row before it is
  the value (`parts.HeldAt`).  The call's terms on the subject go (the history's value stands for them); a term
  reading the subject with another column stops.  Terms on columns the history holds are read on that row; the
  others on the row as it is now (A1), and the END names them;
- the window's moves (`parts.MovedOn`) count, per group, those with a key ahead of the row's, under the same terms:
  the ones the history holds on the history row each left, the others on the row as it is now.  Before, the terms
  on columns the history does not hold were dropped there (MJ-activated-late's resource_type and RAM and disk
  terms), and the held ones were read on any row in the window;
- checks: the row itself is counted once, in its own group, or the run stops naming the terms read as now; an
  earliest entry within an hour of the lookback says the count is "at least";
- the ask carries the question's structure beside its SQL (K36).  Neither question has a db_read form:
  jobs_StatusLog is not on db_read's allowlist (panda-server asyncprocess/db_read/tables.py), and
  job/get_status_history takes keys, not a value and a site.  So the answers are a hand's SQL, or a fetch by the
  parts, until PanDA answers the structure.

What stays:
- the order's columns are read as they are now (currentpriority, gshare); a term read as now that changed since
  the move (a time column) counts other rows than the call did;
- the window's moves are asked of the keys with a history row in the window under the held terms: a key whose rows
  there all differ from the row it left (its site changed at the move) is not counted;
- the lookback: 8 days before the move (`L3_LOOKBACK`: copyArchive.py:509-512 kills an activated job created more than
  7 days before; D's group waited up to 169.1 h).  A row that entered the value earlier is not counted;
- only the call that took the row is read: its WHERE and binds, and the other order counted.  The calls that passed
  it over -- each picks its order by a draw (entity_module getSortingCriteria: by age about one in nine, else by
  gshare) and its binds from the pilot's request -- are not read;
- stuck does not reach L3: no call of a stuck row's statement is read (S1 asks the logs for the key, which a call that
  passed the row over does not print), and M2's "takes it" reads no first-rows statement in the five sets.  Its way
  in, and its count now from the row tables, are a design of their own (part B).

*Shows:* L3's two questions under their names "... held V at the move, by the order's items before the key" and
"... left V in the window, by the value it went to and the key"; "at the move (T): N row(s) ahead of it (G in groups
ahead, K of its own group's M with a key ahead)"; the END's "read as now: ..." and "-- at least: ...".
*Revisit:* when a stuck case's first-rows statement is read (part B); when the calls that passed the row over are
read; when PanDA answers a history held at a moment.

## Row values at a step's time (`held.py`)

### K49. A guard over a column the rows file holds is read in it, by the map's ways and the direction of their changes

Where H3 reads an attribute arm's guards and the run's values decide one not, it is read in the rows file's decoded
column where its name's ways end there (`heldcol`; the analyzer handover's out/tpm/q0/q0-step0.md; tmaeno: "D1",
2026-11-02 23:1x): `task_params`, `jedi_taskparams.taskparams` as fieldtest 5b6d2e9 fetches it.  The map gives the ways
(GuardFrom and ParameterFrom), the changes of the object on each before the read (HeldChange) and after it, before a
write gives the column back (`HeldSource.after`).  A key's presence (`'K' in m`) is read by the direction of each
change: a store, an inner change or a `*` store adds a key or needs it there; a delete, a method, an escape or a call
not read may remove one.  The rows hold K: it was there at the read on a way, unless something may add K after the
read or remove it before; the rows lack K: it was absent, unless something may add K before or remove it after.
Whether the pass wrote the column back is not read: either way these hold.  A key's value (`m['K']`, a literal) is read
as the rows hold it where nothing changes K before or after.  A change counts unless a test on its way reads false: the
tests deciding each call of its `via` and its `at`, early returns included, each read by the map's GuardFrom of that
test the same way, two deep -- TaskRefinerBase 915's `*` store runs only past doPreProRefine 927's
`"preproSpec" not in taskParamMap: return`, and no task case's rows hold preproSpec.  Each way is a route; one the run
shows by a line only its own functions print (the functions holding the calls not every route goes through) is read
alone, else every route is read and a key is decided where they agree.  Every answer rests on the column being read
as the pass left it: no later write of it is read (a later refine's write-back, updateTaskParams_JEDI 1749).  Before
(querier-q), H3 stopped at these guards with "a method the guard reads through (its conditions: ... not read)": 10 of
m-cut2's 13 attribute stops there were extractCommon's guards over the task's parameters.

Not read:
- a test of another form, a part of the value (a way with *steps*), a way ending elsewhere, or whose changes after the
  read were not read (an after-stop other than `no_meeting`): not read on that route;
- a change after the read is counted on every route, whichever route its *via* goes through (more counted, never
  fewer);
- the route by the run's "refining with <class>" (TaskRefiner 282), which every m-cut2 task case prints: it needs the
  map's record of JEDI's factory dispatch (its JD); until then a route is shown in 1 of the 10 (AtlasAnal's "tweaked
  architecture"), and coreCount 53345980 stays read on one route only.

*Shows:* in H3, "'<guard>' read in the rows file's task_params as fetched at <as_of>: True | False | UNKNOWN[, on
<route>, the route its run shows] (KNOWN_ISSUES.md K49)" and each key's reading per route ("'K' in m True on <routes>
(held as fetched[; <change> did not run: <test> at <line> read <value>])", "... not read on <routes> (may be changed
by <changes>)"); the END "its guards hold: '<guard>' read in the rows file's task_params as fetched at <as_of>...: the
column as the pass left it, no later write of it read"; a stop's condition "'<guard>' not read (<each key per route>)".
*Revisit:* when the map records JEDI's factory dispatch (JD); when another column is fetched decoded into the rows file
(`HELD`).

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
there (END) where the arm has no guard or every guard holds in the run's values -- or, for an arm its run shows ran,
where none is read false (K54).  A condition read false, or not read for an arm no run shows ran, is named in H3's
stop; a guard read through a method goes on to H4 as for a status, whose steps say why the arm ran, not when.
Before, an attribute went on as a status does and stopped at H3 "a method the guard reads through" for an arm with no
guard: 12 of the 14 attribute cases H2 answered on the harness's round-2 snapshot.

An arm left by elimination names no run (`wrote.run` null), and its guards are read in the rows' values alone, its
H3 stopping at "a run that shows <arm> ran" where they do not all hold: no run shows it, and the run the judge read it
on is any of the key's at the window, on any machine -- a server's panda-DBProxy.log run (insertTaskParamsPanda, a
day long, the key on one line) for rescueUnLockedTasksWithPicked_JEDI and get_tasks_for_periodic_action, JEDI's
(the harness on xii's T, 10-30 20:2x; tmaeno: "fix it in xii").  Its END names the writer's logs not searched at the
window instead.

*Shows:* "-- row n of m" in H2 and its judgement per row, "<writer>:<line>: n row(s)" for each of several writers;
"END <subject>=<value> written last by <arm>[, called at <call>] (shown in its run; it has no guard | its guards hold
in the run's values | its guards hold as its run shows it ran ('<condition>', ...)[; the others in the run's values]),
in the pass of the statuslog row <status> at <time>: <run>", or "(by elimination: no run shows
the line ran; it has no guard | its guards hold in the rows' values), at <time>[; not searched there: <log> on
<machines>]"; "<arm> ...; a birth: it wrote as the row was made" or "; its runs are marked by the object's key",
"F3: <arms> asked over the object's life, from the row's creation at <t> to when the value was read at <t>",
"-- row n of m: a run of <key> at <first>..<last> (F3)", a possibility "may have at the row's making <t>" or "at the
object's life <t>" (result.json's possibility rows carry `f3`: made, run, span or open); a stop "every writer of
<subject> on a history row of the object, a birth or marking its runs by the object's key ... (k of n not: K37)",
"the object's life, to ask <arms> over: <why>", "when the row was made, for the birth <arm> ...", "a writer of
<subject> in the map", or at H3 "the values its guard reads at its run (its conditions: '<condition>' read false |
not read)" (a guard through no method; one over a column the rows file holds is read there, K49), "a method the guard
reads through (...)" (one through a method, read on at H4) or "a run that shows <arm> ran, to read its guards in (by
elimination; its conditions: ...)".  A status's arm its run shows ran stops, through no method, at "where the values
its guards held on come from, its run showing it ran (...)" (K54; "the values its guard reads at its run" for one no
run of its own shows), and with no guard at "<arm> has no guard of its own to read: what made it run is its callers'"
(both "a method the guard reads through" before querier-q; the analyzer handover's todo, 2026-11-02 22:47:54).
*Revisit:* when the map follows an object back to its caller (the JobGenerator way); when F3 is to read a writer
marking its runs by another object's key (the job's jediTaskID, which its row holds); when a writer only reads the
row into its spec (getTasksToRefine_JEDI and the like) is told apart.

### K38. H2's scan of the rows names the arms that wrote last or may have, and each walks on where none is the one

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
other calls are said; which of them was last is not read.

Where H2 decides nothing, each arm left walks on from H3 (`Rerun.arms_left`, the second stage's first half): at the
newest row it may have written at, and where an attribute's guard reads false there, at its next; with none left it is
excluded.  A branch ends (END), stops (open), or is excluded.  What is open in it has kinds, given where it is found,
not read off its text (`Premise`): not in the data (its run, in logs searched whole, does not show the line ran), not
read (its runs not shown to name the key, a condition, the functions between a call and the arm, a birth's run, and a
stop's own, unless coded asks or rows), not recorded (logs not searched whole, a stop on asks or rows).  The arms not
excluded are the answer: one that ended is the END, with what it rests on and the older rows it may have written at
instead; else END "written last by one of the N arm(s) left, which the data does not decide", result.json's
`outcome.code` `one-of` and each arm in `branches` (its row, premises, kind, kinds, text, and what it put in
result.json, such as H3's `wrote`).  None left: a stop, no arm could have.  An arm stopped on asks or rows stops the
run on them once every arm is walked, so that one round records them.  L1 (`--late`) does not walk the arms it is left
with: its stop names them as before.  H2's single window, where the judge chooses no candidate ("which candidate
ran"), does not fork either: most of its arms are unexamined (no pass of the key at the window, K47), and are dropped
only where the logs are searched whole and every way that takes the row prints the key (the map's K19).

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
wrote <subject>=<value> last, of k left: <arm> (wrote at 'status' time | may have at ...); ... (KNOWN_ISSUES.md K38)"
(L1's); "k arm(s) left: each walks on from H3", "==== n of k: <arm> (<rows>)" and "==== n: reached the end | STOP ... |
excluded ..." per arm, "<arm>'s guard <g> read false in the rows' values at <row>: it did not write there; its next
row, <row>"; an END "written last by one of the k arm(s) left, which the data does not decide: 1 <arm>: ended; 2 <arm>:
open (<kinds>); ... (KNOWN_ISSUES.md K38)", or the one arm's END "...; it rests on: <premises>; or at an older row it may
have written at: <rows>"; a stop "an arm that could have written ...: each of the k left is excluded".
*Revisit:* with the second stage's second half (H2's single window's "which candidate ran", after the map's K19) and
L1's arms; when the map says what fills a print's prefix (then ask a writer's log by its mark's key).

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

### K54. An arm its run shows ran ran under its guards; a status's says the values they held on, where its run printed them

H2's positive (a row printed only on a path through the site, P's proof, or P read with the other verdicts) says the
site ran in the pass, so its path condition held there.  H3 takes that for a guard the run's values leave unread
(`Rerun.wrote_by`, design D1's G, out/after-q/d1s-design.md; tmaeno: "C2", 11-03 17:30).  It is not taken:
- for a stand-in (K25), whose run is its caller's and shows the caller's call, not the arm;
- where a guard reads false: the value decides, as before (K37: an older row, or excluded);
- where a guard reads through a method: H4 reads which call set it, saying why the arm ran.

An attribute then ends at H3 (K37): "its guards hold as its run shows it ran (...)".  A status does not: what it asks
is why the arm ran, and that is where the values its guards held on came from.  Where none is a method, H3 stops on
it, each guard with the values its run printed (V, `Rerun._held_on`, `changed.printed_at`):
- a hole of the term, or `str()` of it, at a row before the guard's `if` or after it, nothing between able to change
  it (`changed.holds`, R's rule (ii));
- or a hole of a chain one assignment after the `if` binds to the term, the term holding up to it and the chain from
  it to the row (setupper 1107 `job.ddmErrorDiag = disp_error[disp]`, printed at 1108);
- read as the map's evaluator reads a printed value (`put_printed`).  Text is not ordered: a guard that orders a value
  read as text is not read (a printed time, with or without microseconds).  A guard the values read false is a stop
  of its own: the run says it held, so the reading or the positive is wrong.
- In a loop that holds the site, only the records the key is on are read, and only where each of the loop's rows
  printed once among them: that is the key's one pass, and the loop is read as gone round once (`flow`'s *once*).
  datasetManager's FinisherThr runs over its jobs, setupper over its jobs.

Measured as u1 against train xx's T (xxT, 11-03 18:51; out/d1/u1-predictions.txt, every prediction held), it moves
5 walks in 9 cases and nothing else:
- hs06sec and queuedTime in m-cut, m-cut2 and attr go from STOP H3 to END H3 (6 cases);
- the 3 m-failed statuses stop at H3 with their values:
  - datasetManager run:735: nTokens 1, nOkTokens 0, endTime 2026-10-31 12:45:38, timeOut 2026-10-31 13:40:26.260951;
  - Watcher run:117: modificationTime 2026-11-02 11:59:24, endTime NULL;
  - setupper subscribe_dispatch_data_block:1105: disp_error[disp] "... could not register subscription", through
    1107.

*Shows:* "END ... (shown in its run; its guards hold as its run shows it ran ('<condition>', ...)[; the others in the
run's values])"; "STOP H3: where the values its guards held on come from, its run showing it ran ('<condition>' held
on <term> = '<value>' (row <fn>:<line> at <span>[, as <chain> bound at <file>:<line>]), ...; ...)"; "STOP H3: its run's
printed values read '<condition>' false on ..., though its run shows it ran".
*Revisit:* when S, the values' sources by the map's trace (d1s-design.md, tmaeno "C2"), is built: a status ends there.

## `which_ran`

### K53. A print that begins with a hole is read on the line its logger wrote

A logger with no mark of its own writes a message whose first word is the print's first hole: datasetManager's
FinisherThr.run prints `f"{job.PandaID} Finisher : Kill"`.  The log's parser (`records.PREFIX`) takes any first
word for the instance token, so the record is token "7392812614" and message "Finisher : Kill".  The map's pattern
begins at the space after the hole (`\ Finisher\ :\ Kill`), so it never found the message, and no run of the writer
was readable.  `Row.texts` reads a print that begins with a hole on the message and, failing that, on token + " " +
message, the line as written; `matches` and `capture` take both, and the hole's value is read there.

- *Only a print that begins with a hole.*  On the line, a pattern can also find a mark's own words.  For example
  `tmpLog.debug("update")` in updateDatasetLocality_JEDI found "update" in its never-closed mark's name
  `updateDatasetLocality_JEDI < taskID=...`.  For a print beginning with a hole, the token is that hole's value.
- *Measured by the records* (out/tk/tk_probe.py, every answer of records, records-cut, records-cut2 and
  records-snap, the rotated files' included).  Of the map's 235 statements whose print begins with a hole, 8 rows in
  52 records are misread, all datasetManager's: FinisherThr.run 696, 707, 709, 727, 730 and 753, ActivatorThr.run
  863, and ActivatorWithRuleThr.run 963.  22 rows are read on the message, since their loggers have a mark.
- *Measured by the walks* (tk1 against ck1).  Only MJ-transferring-stuck's text moves, in m-cut and m-cut2: its
  datasetManager pass now reads its printed values, and run:735 is excluded with [730] not in the run, in place of
  all four of its must rows.

*Shows:* nothing of its own; the rows of such a print are found and captured.
*Revisit:* when the map's pattern for a markless logger's print says where the message begins (the token split
off), or the parser knows which loggers have a mark.

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
the other's that prints nothing else (the map's `flow.runs_between`: one flow, whose ways include a statement raising
to its handlers).  Other calls can run between the two records, as between c and its print before
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
(parse_job_output_report 820, MJ-merging-stuck on records-cut, 7a05464).  Since 2026-11-04 such an arm is kept off
where every pass's rows show its `try` raised nothing (`rerun.raised_nothing`): a row in the try's body with nothing
after it there that may leave (`rows.may_divert`, as `must_rows` reads what follows a row), no loop between, and a
pass that is one call.  Every arm so kept off, the branch is excluded, and a fork counts it neither an end nor a stop
(setupper_atlas_plugin's run 212: row 206, its try's last statement).  `may_divert` does not see a subscript or an
attribute that raises, as `must_rows` does not.  And a
pass that leaves a loop at its terminator is read as leaving at that exit: setupper's assign_dispatch_data_blocks
walks `self.jobs + [None]` and leaves at 324 (`if job is None: break`) after every job, so S2's "left by 324" and S3's
{'job is None': True} are the call's, not the key's iteration (the investigator's round 2, 2026-10-31).  Neither
gives a wrong answer: S3 stops.  A pass whose judgement excluded every candidate, the arms and the exits alike, took
none of them (`rerun.took_none`, 2026-11-04): it ran to its end or left by a raise in what it calls, and S2 used to
stop at "which way pass N went" (setupper_atlas_plugin's run, whose one arm 212 its row 206 excludes).  S3 then has
no exit's path to settle a guard, and reads the arms' guards in the rows the passes printed only (`_ways_printed`).
setup_jumbo_jobs's pass took none only because `rows.must_rows` gave its `if len(self.jumbo_jobs) == 0: return`
(1967) the "start" after the `if` (1968), which only the other way prints; since 2026-11-04 a `return` or `raise` has
no row after it, and S2 takes 1967.  S3 then reads `not (len(self.jumbo_jobs) == 0)` false against its three arms,
and S4 stops at "one binding of len in every pass": it reads the builtin the guard calls as a name to bind, and not
the attribute (MJ-assigned-stuck, MJ-defined-stuck on m-cut2).

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

### K42. A record two rows print alike proves neither, of one function or of two, and rule 6 reads a caller's guard on any object of the class

`rows.Seen.proven` counts a record only where the function reads it with that row alone (`Seen.unambiguous`, after
`rows.claim`): activateJob prints its files' UPDATE alike at 92 (in the arm that activates) and 124 (in the one that
does not), and such a record had made 71 positive (the investigator's round 2, 2026-10-31).  The record proves
neither row even where the other's path is ruled out on its own, or its place in the run rules the other out:
parse_job_output_report prints "parse_job_output_report end" at 750 (before the early return 751) and at 966.  On
records-cut (out/pv1) MJ-merging-stuck and MJ-transferring-stuck lose 967's positive, though 751 is excluded by rule 4
and the record follows rows 869-873, which 750 cannot follow; those branches stop at S2.

Nor does a record another function of the dump prints whole (`rows.whole_writers`: a row of its `function_prints`
the map reads the record as, mark and message anchored at both ends, `trace.reads_whole`, in the run's log):
`rows._seen` places it with no row.  `rows.claim` sets only the functions it is given against each other, and a
sibling plugin that writes nothing the question asks is not among them.  AtlasProdTaskRefiner.doRefine prints
"start taskType=..." and "done" (107, 174) under the mark AtlasAnalTaskRefiner.doRefine prints them with (205,
280), and the two had made AtlasAnal's doRefine:260 positive by `rows_around` in MR-T-53344406-reqID, a run that
printed "refining with AtlasProdTaskRefiner" (q1, 2026-11-03).  On r1 it is unexamined: no pass of it holds a
record of the key.  Read on samples from the map's rows, it was the one of q1's 33 positives by rows (out/ra); the
measure found a second, which samples cannot show: runImpl's "start" (138), found by search, had read AtlasAnal's
"start taskType=anal" (doRefine 205, called at 284) as its own, and with 282 made runImpl:284 positive in
MR-T-53351604-splitRule.  It is kept now: the callee's line that does show 284 ran is not read (K41).  A row the map cannot read whole (`reads_whole` None: no capture,
or a mark its logger may write that the map did not read) does not count: PostProcessor.start's "done" and
TaskRefiner.start's "start" have no prefix for that reason, and counted, they would take every "done" and
"start" of their logs, "doBasicPostProcess done with taskStatus=..." (122) among them, which they cannot print.

Rule 6 (`which_ran.rewriter`) solves the callers' guards too (`callers_guards`) and reads `<x>.<attr>` in them as the
key's object whatever x is: archiveJob 1783's `job.jobStatus == 'finished'` tests the merge job, while
updateUnmergedJobs 1236 writes the jobs it merged.  On records-cut before pv1 it excluded 1236 in MJ-merging-late (the
key 7390523881, the merge job 7390910703).  With the row's second read run by run (pv1) the env no longer settles
that guard there and 1236 is positive; the defect is not fixed.

*Shows:* "row(s) [..] print only on a path through it" names only rows a record is read by alone, and that no other
function prints whole; "rule 6: reached only when <x>.<attr> is already ..." with a caller's guard.
*Revisit:* when an ambiguous record can be given to the one row the run's order or another rule leaves (P's order
check reads rows read alone too), or to the one function the run's other records leave (the map's JD: the
factory's dispatch, read with "refining with <class>"); when the map reads a row whole where its mark is not read;
rule 6, when the map says which object a caller's guard tests against the callee's
argument.

### K44. A row shorter than MIN_PROOF proves only as its whole message no other function can print there

`rows.Seen.proves` reads a record found at fewer than `MIN_PROOF` (8) literal characters as proof of its row only where
the record's message is the row's text whole and no other function can have printed it there (`rows.line_writers`:
a row of the dump's `function_prints` whose map pattern is found in the text, whose prefix stands where the line
starts or which has none, and whose lines can land in the run's log).  "start" inside "restart", a traceback another
function prints with the caller's logger, a record the search cut, or a row whose map rows carry no prefix proves
nothing below 8 characters.  A row of holes alone is not counted as another's line: it can print any text at any
length, which MIN_PROOF does not read either.

On M cbe29b6's map this proved no row of the DB proxy logs: activateJob's "start" and "done" under
`activateJob < PandaID=... >` (the harness's to-querier-74, MJ-assigned-late) could also be the lines of 5 methods that
print the same words there, whose rows the map gave no prefix because each makes its logger with different marks on
its paths (calculateRWwithPrio_JEDI's two `create_tagged_logger` calls with two tags; 91 rows with a pattern in the
DB proxy logs).  Since xvi the map gives a print row per mark (PFX), and those two words have activateJob alone as
writer there: MJ-assigned-late and MJ-defined-late go past H2 (the analyzer handover's out/xvit).

A DB proxy method's mark is the name its `comment` gives (`create_tagged_logger`).  On the pin, 12 of the 474 methods
that call it pass a comment naming another method.  S1 reads a run as a mover's by that name (`of_mover`), so their
runs are not read as their own, and updateTaskModTimeJEDI's (`updateTaskErrorDialogJEDI < <jediTaskID=N> >`) are read
as updateTaskErrorDialogJEDI's.  The rows are not misled: the map's prefixes tell those two tags apart, and
configurator_read_panda_sites's mark, the same as configurator_read_sites's, makes both its writers.  panda-server
fixed the 12 comments (tmaeno/panda-server 4f89a3d7, 2026-10-31); no release holds it yet.

*Shows:* a short row's record among "row(s) [..] print only on a path through it"; else the row is kept, not positive.
*Revisit:* when production runs a release with the fixed comments (the records keep the old names until then).

### K45. A late value's mover is read over its own passes before the move, and no further back

Where the run that moved a late value on is of a mover that takes the row by its key -- no statement of its own
selects on the value (L2's statements) -- why it moved it only then is in its own passes before (judgement 4,
`Rerun._late_passes`).  Its logs are asked from the stay's start to the move for its marks holding the key
(`key_marks`), as F3 asks a writer's life; a bare key over hours takes every listing of a DB proxy's log.  S2 reads
the passes that end before the second the statuslog shows it moving on, and S3 what kept them from moving it.

- A mover whose mark holds no key (`setCloudToTasks_JEDI`'s time) is asked by the key alone (tmaeno: "B", 11-01):
  a pass of it that prints no line with the key is not found, so finding none rules nothing out.
- The second is the statuslog's, the database's clock, against the log's: a pass of the same second as the move is
  read as the move's.
- With no pass before it, the delay is in what calls it on the row: U1 reads its callers' passes before the move
  (K55), whether or not every line of its runs holds the key -- a caller's pass that called it there finds a pass of
  it its own mark did not.  Where the map lists no caller of it, or the window is not known (the door over files),
  none found is said: "none from" where every line of its runs holds the key and the window was searched whole in
  every log of it, else with why it may still have run.
- The other writers L2 lists get judgement 3 (K15); one that ran in the window and did not move it is said, not read.

*Shows:* "no pass of <func> took the row from <since> to <t>: its callers' passes there are read", then U1 and U2
(K55); with no caller listed, "STOP S2: a pass of <func> before the one that moved it on at <t>: none from <since>,
asked <how> -- the delay is in what calls <func> on the row, not followed here", or "...: none found from <since>,
asked <how>, and not ruled out: <why>", code K45; else S2's and S3's lines over the passes.
*Revisit:* with K55.

### K55. A mover that did not take the row is read up its callers, two hops, by their runs of the key

Where no pass of a mover took the row in the window -- a late mover's passes before the move (K45), a mover's passes
in a stuck value's window (S2), or a statement of a late mover's own that took the row only at the move (L2's
judgement 1) -- what kept it off the row is in what calls it on the row.  U1 reads the callers the map's
`called_from` lists for it, each on its own (a fork, as S1's movers), and U2 one caller's passes on the row in the
window, as S2 reads a mover's, its calls into the mover the arms.

- The caller of an entry (`outer -> inner`, the inner the call into the mover) is, of its two hops inner first, the
  first whose run marks hold the key (`key_marks`), else the first a row of which prints a key column (asked by the
  key alone, as K45's mover whose mark holds none), else the innermost that prints, which stops: its runs print no
  key.  A facade (TaskBuffer's, JediTaskBuffer's) prints nothing, and a plugin's dispatch (doPostProcess,
  doRefine) marks no key; the caller behind it is the one a run shows.  Their own way to the inner call is not read:
  on pin m the 11 such calls of the 19 movers first read have no guard.
- Every pass of the caller on the row took another way: S3 reads the guard of its call, the statement the map's
  `call_statements` picks for the hop (two on the line with different guards are not read), with the pass's values.
- A caller with no entry row is read on its runs whole, as S2 reads a mover's (K47): ContentsFeeder's
  feed_contents_to_tasks prints under one mark of the task on a machine for hours (1621 records, 08:26 to 03:34 the
  next day), and U2 judges that as one pass.  Where such passes differ, U2 stops on which way one went; were they to
  agree, S3 would read several calls as one.
- A pass of it called the mover there, and no pass of the mover holding the key was found: the mover's run inside the
  call is not read (a mark holding no key: setCloudToTasks_JEDI's, doBasicPostProcess's), and it stops.
- No pass of it on the row: it did not run on the row only where the window was searched whole, every log of it was
  asked and every line of its runs holds the key (its marks do; one asked by the key alone rules nothing out).  Then
  U1 goes up once, from a caller at hop 1 to its callers at hop 2 the same entries list; past the two hops, or for one
  at hop 2, what makes it run on the row -- its own selection (a poller's SELECT, a message), or a caller further up
  -- is not read.

*Shows:* "U1 what called the mover on the row" with each caller, its hop and how it is asked; "U2 which way the
caller's passes went on the row", then S2's lines over its passes and S3's over its call's guard; or "STOP U1/U2:
..." with code K55.
*Revisit:* when the map lists more than two hops, or a facade's guard is read; when a mover's run inside its caller's
call is read by the caller's pass span (its mark's own strings, no key).

### K56. An arm with no guard of its own is read at the call that ran it, by what the call handed its run's mark

A status's arm its run shows ran (H2's positive, K54) with no guard of its own ran because a call of its function's
ran it, and H3 has nothing to read.  U3 reads which call: the arm's run is marked by what that call handed it, not by
the row's key -- updateUnmergedJobs < PandaID=... > is the merge job's, and the unmerged job's key is on one line of
it -- so the caller's run is asked by that value.

- The run's mark (the map's `runs` of the arm's function) gives each hole's value and the expression it prints
  (`job.PandaID`); the map's `call_bindings` give, for each call `called_from` lists (the hop into the function), the
  caller's name handed to that parameter (`job` at archiveJob 1783, `jobSpec` at propagateResultToJEDI 4822).  A
  caller's run marks by that name (`jobSpec.PandaID`) hold the same value; a call handing it no name a mark of the
  caller's prints is not asked, and with none, U3 stops.
- The callers' runs are asked over the arm's run, on its machine.  A caller's run holds the arm's whole where the
  call ran it, and of those the innermost (latest begun) is the call's: archiveJob's run holds propagateResultToJEDI's,
  which holds updateUnmergedJobs's when 4822 called it.  A run cut by the window holds nothing, and U3 stops naming
  how many runs it found.  One run of a caller holding the arm's at two calls of it does not say which, and U3 stops.
- The call ran, so its guard held (K54): the call statement's path condition (`call_statements`, one statement on the
  line) is read as H3 reads an arm's, with where the values it held on were printed in the caller's run, and the
  walk stops there as a status's H3 does.  A call with no guard of its own says its callers made it run, past the
  hop read.
- Not read: which call a value carried through the arm tells.  updateUnmergedJobs:1236 copies the caller's
  job.jobStatus, and 1783 is under `job.jobStatus == "finished"`, so a value other than 'finished' did not come from
  1783 -- with no caller's run read.

*Shows:* "U3 which call ran the arm" with the mark's values, each call's handed name and marks, the run holding the
arm's and the call's guards; or "STOP U3: ..." with code K56.
*Revisit:* when the carried value picks the call; when an arm's early exits are its guards (the map's branch
condition, `pathcond.reachable_when`), which takes killUnusedEventServiceConsumers:986 out of this class.

### K57. A run is at the window only where a record of it lies near, not where its span holds it

H2 placed a run, or a pass of one, at the window where its span held it.  `passes_at` cuts a run at its function's
entry rows, but the records at hand are those the asks returned.  Where they hold no entry row, the run stays whole,
and it spans every pass of its mark between its first and last record.  So it is "at" any time between them.

MT-scouted-late (JediTaskSpec.status 'scouted' -> 'running' at 18:49:05) is the example.  TaskRefiner's run
`< jediTaskID=53199089 >` held only the child task's two "pending until parent task 53199080 is done" lines, at
15:53:23 and 22:35:59.  runImpl:257 was positive on it, beside getTasksToBeFinished_JEDI:1672, which made the write.
The census of every H2 judgement over the querier's sets found 7 verdicts in 4 cases on such runs, 6 kept and 1
positive.  All 7 are separate passes under one mark, with no record within 10 min of the window: reactivatePendingTasks_JEDI's
`< vo=atlas label=... >` with records hours apart, and JobGenerator's `<jediTaskID=... datasetID=...>` over 16 hours.
No excluded verdict was on such a run.

Now `which_ran.at_window` places a run at the window where its span holds it and one of its records lies within
EDGE (`recorded.EDGE`) of it -- the door's part near the window.  Inside EDGE the records at hand are what the asks
searched, so a pass at the window has a record near it unless it printed nothing there.
- `passes_at` keeps a pass only where it is at the window this way.  A run that spans the window with none near is
  said so ("N span(s) it with no record within 10 min of it"), and its candidate is left unexamined.
- The runs H2 gathers at the row's second (`_wrote_at`, `_row_at`), and F1's (`rows_between`) and H7's (`missing`),
  keep the span (`which_ran.spans`).  They decide the question's run, and a question's run is read whole, not cut into
  passes.  With the far runs dropped there, MR-T-53351604-rescueTime's one run left in JediDBProxy.log, the
  `< vo=atlas label=user >` mark's 297 passes, became the question's run and was read as one (rule 20's warning),
  where each candidate had been read on its pass at 13:09:13.  MR-T-53348259-rescueTime is read so already (187
  passes).
- Judgement 3 (`_runs_of`) still reads a mover as run where a run of its spans its window (`which_ran.spans`).
  There, "not at the window" would be absence: a mover could be ruled out, and the window's rest asked.

A pass that prints nothing within EDGE either side of its write -- a slow call between two prints -- is now not at
the window, and its candidate is unexamined there: a positive lost, not a wrong one.

*Shows:* "no pass of the key at ...: ... none of its N pass(es) of F is at the window (N span(s) it with no record
within 10 min of it)".
*Revisit:* where a function's pass is silent for longer than EDGE around a write (the map's reach of a slow call);
when judgement 3's twin is decided; when a question's run is cut into the pass at the window, which lets the runs at
the row's second be gathered by records near it too.

### K58. H2 can read another object's run as the key's run, where the run prints the key in another role

A run is the key's for H2 where a record of it holds the key.  It can be another object's run that prints the key
in another role.  TaskRefiner's child-task run `< jediTaskID=53199089 >` printed the parent's key 53199080 in row
262's `parent_tid` hole, and runImpl:257, which writes the child's `impl.taskSpec.status`, was read on it (MT-scouted-late).
K57 removes that run there, since it was not at the window.  The census found no other case of the class.

A rule would read a run as the key's for an arm only where the key is in a hole that holds the key of the row the arm
writes -- the run's mark, a local, a loop's element, or an SQL bind map.  The map types a hole by the column it holds
(row 262's `parent_tid` as `JediTaskSpec.jediTaskID`), not by which row's key it is.  It names the written row only
for an arm's must prints (`Branch.must_names`, row "written"): for 52 of the 242 positive or kept verdicts of the
census.  257 has no such name, and 616's reading of the mark's `jediTaskID` cannot show it holds the value the write
counts.  On today's map the rule would cut the tie of the other 190, of which 4 are wrong.

*Revisit:* when a case shows another object's run at the window; the map would then be asked to name, for every row
of an arm's function, which holes hold the written row's key.

### K59. A bare mark with a blank is cut by the map's marks of its log, not at its first word

`records.PREFIX` takes the first word for the mark where no bracket ends it.  A LogWrapper prefix or a MsgWrapper
token holding a blank was read short:
- `update_job PandaID={job_id} PID={os.getpid()}` (pilot_api.py:513) was the mark `update_job`: every pilot's call,
  with the key in the message.  U2's mark round (`recorded._read`) asked `update_job` on api_pilot.log over U2's
  window, 5.6 h on two machines each, in MJ-merging-stuck and MJ-transferring-stuck (the harness's round 21 re-run).
  The key's own ask had returned the call whole (Start, executing, Done).
- AtlasAnalWatchDog's ` #ATM #KV doForPreStaging label=user` was `#ATM`, every ATM action's lines (MT-throttled-late,
  7 asks).
The map holds 86 LogWrapper and 16 MsgWrapper runs whose mark has a blank and no bracket.

Now `records.cut_mark` cuts a line again by the map's marks of the runs printing into its log: the marks of the runs a
statement prints through, its own and those handed to it, under the files the map follows its logger to (basis
`logger`), of the builders that write the mark bare (LogWrapper, MsgWrapper).  Where one starts the line and goes past
PREFIX's cut, ending at a blank, it is the mark; the longest wins.  Both readers of a line (`read_log`,
`recorded._record`) call it.
- update_job's run is one call, the key in its mark, so the mark round does not ask it.
- `#ATM`'s is `#ATM #KV doForPreStaging label=user`.  It is still a shared mark, which the mark round asks (below).
- A hole is one word.  A value holding a blank leaves the line as PREFIX cut it.
- A mark with no literal (`{}`, `{} {}`: 283 (mark, log) pairs in the map, the time stamps' empty marks among them)
  is not read: it would cut any word.  make_logger's and create_tagged_logger's marks are bracketed, and PREFIX reads them.

Left:
- A line with no mark at all keeps its first word as the mark: a bare getLogger's `Sent Watcher {id}` in
  copyArchive.log (MJ-sent-late), base_module's `rollback EC:...` in DBProxy.log (MJE-7394762518, MJE-7397173613).
  The map's marks cannot tell such a word from the value of a mark that is a hole alone (DBProxy.log has `{}`).  The
  row that printed the line can.
- The mark round asked a mark the calls share over the whole window: a JEDI worker's `process <host_pid-thread>`
  (MT-scouting-stuck, 5.3 h), `getTasksToExecCommand_JEDI < vo=None label=None >` (MT-passed-late's L2), and
  `#ATM #KV doForPreStaging label=user`.  It asks one near the key's records now (K60).

*Shows:* nothing of its own: the run under the whole mark (`token=update_job PandaID=... PID=...`).
*Revisit:* with the line with no mark; when the map says how each builder prints its mark.

### K60. A mark the calls share is asked only near the key's records, one ask a machine

The mark round (`recorded._read`) asks each mark under which a key record lies, the mark not holding the key, over the
window.  One the calls share returns every call's lines over it: a JEDI worker's `process <host_pid-thread>`
(MT-scouting-stuck, 5.3 h), getTasksToBeProcessed_JEDI's tag on MT-ready-stuck's snapshot (9 asks, 51.4 h).

Now a mark that every map mark of a run printing into the log reading it says the calls share -- none of its holes
holds a column that is a table's identity (`TableRecord.identity_attrs`), and it is no time stamp (`RUN_KEY_TIME`) --
is asked only within 2 EDGE of the key's records under it on its machine, inside the window.  The windows of one
machine are joined where they overlap, one ask each with their marks' strings (`recorded._calls_share`,
`_share_index`, over `records.runs_into`, which K59's `_bare_marks` reads too).  An asker that reads a run at a row's
moment (`whole`) asks as before; so does a mark no map mark reads, and one holding a table's identity.
- Not asking such a mark (a probe, sh1) lost the key's own pass under it: getTasksToExecCommand_JEDI's 13 records at
  09:33 in MT-exhausted-late, and the arms MT-exhausted-stuck and MT-paused-stuck excluded by it came back kept.
- One ask a machine and mark (a probe, sh2) made more asks: MT-ready-stuck 9 -> 34.

Measured as sh3 (this rule on 58db119's files, xxii's T) against mk1 (58db119):
- No verdict, END or stop moves: 144 cases, the 65, attr, MT.  The asks move, with the notes on them, and
  MT-ready-stuck's STOP S1 count (9 -> 10 asks).
- The asks of the key's runs under a mark not holding the key, all sets: 41 over 129.7 h -> 42 over 45.2 h.
  MT-scouting-stuck 5.3 h -> 0.3 h, MT-ready-stuck 51.4 h -> 10.9 h.

Left:
- The rule's premise: a key's run under a shared mark prints its lines within 2 EDGE of the key's records.  A run
  printing them farther apart is read cut.  None in the sets.
- A shared mark whose key records spread over the window is asked over it: MT-throttled-late's
  `#ATM #KV doForPreStaging label=user`, on every machine every few minutes from before the window to after it (7 asks,
  12.2 h).

*Shows:* the note "<n> mark(s) the calls share (the map) asked only within 20 min of the key's records", and the asks'
why "..., near them: a mark the calls share".
*Revisit:* when the map says where a call's run starts and ends, so that an ask is one call's span.

### K61. The early exits above a statement are its guards, for every reader

The map's branches say the tests a write runs under with the early exits above it (`pathcond.arm_condition`: an
`if X: return` puts `not (X)` on what follows, `  [at N]` where a name it reads is given a new value between the exit
and the write).  The querier's own readers computed `pathcond.path_condition` from the corpus, the `if`s around a
statement alone, and S3 added `enclosing_guards`' exits to the branches' guards itself, so a guard `X  [at N]` stood
beside its bare twin `X`, which S3 then read in the pass's values.

Now every reader of a statement's guards reads `pathcond.arm_condition`: judge's step 5 and rule 6 (`rewriter`,
`callers_guards`), H2's stand-ins, H3, U3's call guards, S3's exit taken (`blocked`) and the rows it printed on the way
(`_ways_printed`).  An exit's guard is read at its own `if` (`_held_on`, the lines `pathcond.exits_above_at` gives), a
guard `[at N]` only at line N: in a pass's or the rows' values it is not read (`E.read_elsewhere`).  S3 reads the
branches' guards as the map writes them and goes on from an exit's `if` by its line (`early_exit_if`).
- An arm whose only guards are exits has guards now: H3 reads them, as an `if` around it, and goes to U3 only for an
  arm with none.  MJE-7398540916's killUnusedEventServiceConsumers:986 (`not (notToDelete)`, `not (not deletedFlag)`)
  stops at H3, its exits held with nothing printed; before, U3 asked for the run of archiveJob that called it (K56).
- An arm left that "may have" written, whose exits the rows' values do not read, stops for a run that shows it ran,
  as one with an unread `if`: the doGenerate / doGenerateBuild arms of the MR-J attributes (`not (not readStat or
  taskParamMap is None)`, `not (not tmpStat)`), which ended on "it has no guard".  Their one-of answers stand, the arms
  open.  A writer read on its caller's run whose exit is unread may have written: extractCommon:501's `not (workQueue
  is None)` (MR-T resource_type).
- An exit that reads a method the evaluator does not run goes to H4: setupper:1105's `job.isCancelled()`
  (MF-7398155528), which H4 finds no call setting.

Measured as ex1 (this on codemap-xxiv bf88f71 + K60, dump x13xxivXm) against the same tree without it (xb1):
- No run crashed.  The case outcomes: END 100 -> 100 (one-of 61 -> 63, plain 39 -> 37), U3's K56 stop 5 -> 4, H3's
  one more.  Within the walks one-of arms ended 43 -> 13, open 99 -> 133.
- No verdict of judge moves; adder_gen process_job_report:395, excluded by rule 4 already, gains its exit's test read
  false (the job merging).
- S3's bare twins are gone: MJ-assigned-stuck and MJ-defined-stuck stop at S3, not one step on at S4 from the twin
  `not (len(self.jumbo_jobs) == 0)` read false at the write (its names are as they are at 1966).

Left:
- The ENDs that rested on not seeing an exit now rest on an open arm.  Ending an arm that may have written on its
  unread guards as premises is another rule, for `if`s and exits alike.
- An arm whose guards its run shows held, with nothing printed, stops at H3; U3 would ask on.
- A method an exit reads (`isCancelled()`) is read only once the evaluator runs the map's runnables.

*Shows:* the exits in "guards of f:N: [...]", each read at its own `if` in "held on ..."; S3's "needs [...]" without a
bare twin.
*Revisit:* with the evaluator running the map's runnables; with the rule above for an arm left with unread guards.

### K46. A loop is read as gone round once only where every way back prints a row the pass lacks

S2 (`which_ran.judge`) reads a row and a site in one loop as possibly printed in different rounds of it
(`rows.share_loop`): a printed row's arms then do not leave the site out, and rule 3 does not order them.  A loop
that no way through its body runs to the end of, each of whose `continue`s comes after a row inside it that the
pass did not print, went round once in that pass (`rows.once_loops`, `which_ran.once_loops_of`).  Two shapes:
- the DB proxy's retry, `for iTry in range(nTry)`, with no "retry: {iTry}" in the run (on pin m: activateJob,
  updateJobStatus, updateJob, retryJob, keepJob, peekJob, peekJobLog, addMetadata, countFilesWithMap; and
  workflow_core's process_step_checking);
- a loop whose body always breaks (`for (clob,) in self.cur: ... break`).

S3 (`Rerun.blocked`) also reads the path condition of each row the pass printed outside any other loop
(`Rerun._ways_printed`), beside the exit's.  MJ-defined-late's activateJob row 111, in the else of 70, says
`not (len(res) == 0 or allOK)`; the exit the pass left by (153, `return True`) is reached either way.

What is still read as a loop that may go round:
- a loop that goes round by running off the end of its body: getJobs 2378 (a retry over candidates, which breaks
  when it got one), and the retries of updateInFilesReturnPandaIDs, updateOutFilesReturnPandaIDs and setGUIDs,
  whose handlers end with neither a `continue` nor a `return`;
- a loop over items (`for file in job.Files`), which prints nothing on its way back;
- any loop in a run of several calls.  S3 reads no row's path condition in such a run either.

*Shows:* "row N printed and its arms [...] leave L out (the loop at M went round once: its way back prints row(s)
[R], not in the run)"; "settled by the rows the pass printed on the way [N]: {...}".  `WHICH_RUN_ONCE_LOOPS=0` and
`RERUN_S3_ROWS=0` read as before.
*Revisit:* a way back that runs off the body's end and still must print (a handler that prints "retry" and ends).

### K47. A candidate is read on its function's pass that holds the key at the window, not on a run that spans it

H2 (`which_ran.judge`) read a candidate on whichever run of the key in its log spanned the window, and took the one
with the most rows proven (`run_holding`).  A mark that names no key spans every pass of a worker.
rescueUnLockedTasksWithPicked_JEDI's `< vo=atlas label=user >` run on aipanda081 held 155 passes, with the key in
one at 19:11.  It was read for the write at 13:09:14 that aipanda082's pass made.  A run found alone was taken as it
was, another method's included (insertNewJob read on updateWorkers').  On m-cut2, 11 of 93 choices were such runs
(the analyzer handover's out/wr2).

Now `which_ran.passes_at` reads the runs that span the window in four steps:
- it drops a run whose mark names another method (`of_mover`, S1's rule);
- it cuts a run into the function's passes at the records the map reads as its entry row whole (`rows.entries`,
  `trace.reads_whole`: the row's mark, then its message, anchored at both ends), the records before the first a pass
  of their own.  The row's pattern is searched, and `start` finds a loop's `start 3/10 jediTaskID=...`
  (prepareTasksToBeFinished_JEDI:3854, getTasksToExecCommand_JEDI:4346) and a callee's `start taskType=anal` through
  the same logger (doRefine under TaskRefiner's runImpl).  Cut there, the pass at the window lacked the rows before
  it, and rule 4 excluded runImpl:211 (MR-T-53351389-transUses).  The querier's own patterns, read whole, lost real
  entries: Fm's map pattern for TaskCommando runImpl:129 `start for {commandStr}` drops the trailing hole, and a
  LogWrapper's `<setup_destination> ` mark stands inside the message.  A record the map cannot read whole -- its row
  has no capture, or its logger is not bound in the function -- does not cut: the passes it may begin stay one, and
  the word on the cut counts them.  The rule-20 warning and S2's passes of a mover read the same entries;
- it keeps a pass whose span holds the window and that has a record within EDGE of it (`at_window`, K57);
- of these it keeps one that holds the key: in one of its records, or in a run of the key inside one of its calls
  (`inside_call`, with a callee the map's rows say printed that run's key records:
  reactivatePendingTasks_JEDI's `< vo=atlas ... >` pass, inside which release_task_on_hold's
  `< jediTaskID=... >` ran).

A candidate with none left is unexamined, with what each run lacked.  With the list complete and one candidate left,
the judge chooses it by elimination even when it is unexamined, but only in a window that holds the write itself
(`windows.Asking.holds_write`): the value is the time its writer ran (`value_now`).  A statuslog row asked is a
guess at when the value was written, and with no pass there the next row is asked, as before.
MR-T-53354352-actionTime had stood only because its one candidate was read as kept on insertTaskParamsPanda's run.
An attribute's END then names no run; a status, or the move of a late value, stops for want of the pass.

S2 (`Rerun.passes`, which way each pass of a mover went) reads a pass by the same test (`which_ran.holds_key`).
A mover's run of the key under a mark with no key is a poller's whole day.  On m-cut2 S2 had read
getTasksToExecCommand_JEDI's 1149 passes in MT-scouting-stuck, none of them holding the key; 234 with 1 holding it in
MT-exhausted-stuck, 188 with 1 in MT-paused-stuck, and 258 with none in Tr.  A pass of prepareTasksToBeFinished_JEDI
there was read positive at 3994 and 4095 on the lines of tasks 49758808 and 53354973.  The passes without the key are
counted and not read; with none left S2 stops for a pass that holds it.

What stays:
- A log line names no process.  A run inside the pass's call may be a concurrent thread's in the same seconds, as
  for `inside_call` everywhere.
- A function with no entry row is read whole, and so are the passes of one whose entry the map cannot read whole.
- An entry the map reads as not whole where it is does not cut.  The map's capture reads a hole up to the first
  character of the literal after it, so a value holding that character fails: killJob:840's `user=([^\ ]*)\ wg=`
  against a DN with blanks (MJ-pending-late; one pass either way there).  In the xviii dump 3169 of the 7662 capture
  hole groups stop at a blank.  The map's to decide (told 2026-11-01, w1e).
- The prefix proof of a record below MIN_PROOF (`Seen`, K44) still reads the row's own patterns whole, not the map's.
- A pass that prints no record of the key may still have taken the row without printing it.  Whether every way that
  takes the row prints the key is the map's (K19's must_names_whole); until it is read, "not among the rows it took"
  is not said.
- A pass that holds the key is read whole, the other rows' turns of its loop included (getTasksToExecCommand_JEDI
  goes round the tasks it selected).  Reading the key's turn alone needs the loop over the rows and its body's first
  row, the shared flow part S1: for H2 and S2 together, after S1.
- The question's run (`_question_run`, one run of one log at the window) is still taken by its span.  On m-cut2 no
  such run is off the window (out/wr1).

*Shows:* "no pass of the key at (...): <run>: another method's mark; <run>: none of its N pass(es) of f is at the
window; <run>: its pass at the window holds no record of the key, nor a run of it inside a call"; and "no run: no pass
of f at T shows it ran".  In S2: "N pass(es) of f in the window print no record of the key, nor run it inside a call:
not read", and "STOP S2: a pass of f in the window that holds the key: none of its N there prints it".  On the cut:
"cut at entry row L of f: N pass(es)", or "entry row L of f is read whole at none of its N record(s) in the run (M
the map cannot read whole): the run is not split" -- whether the map can read the row whole is the row's, so M is N
or the parenthesis is not there.
*Revisit:* a function with no entry row whose runs span hours, or whose entry the map cannot read whole; the
question's run of a mark with no key; the prefix proof (K44) on the map's whole reading.

### K50. A row before the site implies it only where nothing on its way to the site may leave

`rows.implying_rows` reads a row before the site as proof only where nothing on its way to the site can leave: no
call but a logger's and no jump (`may_divert`).  An attribute or a subscript that raises is not counted, as it is
not anywhere in the rule.  The row is at the top of the site's block, or nested in an earlier statement it falls
out of (`rows._falls_out`, MF-7397944082 and MF-7396944927, the harness's finding, 2026-11-02): handle_failed_job
prints 370 ("job_failure_postprocessing is back") last in a `try` of an `if`, and 374 follows that `if`.  363
("AdderGen.run will call job_failure_postprocessing") does not: the call after it may leave.  Of q1's 108 kept
candidates, 5 gain such a row (out/ra/d2_kept.out); on r1 374 is positive in 4 m-failed walks and no other moves.  In
MF-7397695467 and MF-7398402231 it is positive next to parse_job_output_report:919, and H2 chooses neither ("2
positive"): both wrote "failed", 374 later in the pass (process_job_report 421, then finalize_job_status at 446).
On q1 919 was chosen alone with 374 kept and the walks went on to H4.  None falls out of a loop's body (it goes back
to the loop), of a `finally` (a raise may be passing through), of a `with` (its exit is a call) or of a `match` (not
read).

*Shows:* "row(s) [..] print only on a path through it" with the nested row.
*Revisit:* when the rule reads the map's flow (`flow.passes_through`'s ways, a raise anywhere) rather than its own.
H2 now orders one pass's candidates by the flow (K51): 374 is a stay in all four walks.

### K51. Of one pass's writes the last that changed the value is the answer, and one that kept it is a stay

`changed.last_change` (rule R) reads the candidates H2 leaves in one pass: those of the question's run, and of the
runs inside a call it made.  The value the pass writes is the one it assigned last, so the answer is the last write
that changed it.  A write of the value already there is a stay, as the census reads a row written again with its own
value (the harness's reading of adder_gen at the pin, 2026-11-03).

- *The order.*  One function of the pass, the anchor, holds the others or calls on into them (the map's reach,
  `Rerun._called_at`), and its rows read as one call.  A run inside a call sits at that call.  Two places are in
  order where the flow runs one after the other and never the reverse.  Any pair out of order, and R says nothing.
- *The value before a write.*  It is read from a printed hole of the chain the write assigns, at a row before it, or
  from the constant an earlier write that ran assigned.  Either counts only where nothing on the way binds the chain
  or hands what holds it to a call that can change it, by P's reading of a call (`printed.self_effects`).
  - The way runs through the rest of the source's function, the anchor between the two places, and the write's own
    function before it.
  - The frames between the anchor and a function reached through others are not read (finalize_job_status, between
    process_job_report and handle_failed_job).
  - *What keeps it on the way* (CK, 2026-11-03).  An assignment of the value's own constant to the chain keeps it.  So
    does a call P reads as able to change it, where two things hold.  First, the call cannot bind what holds the chain
    anew: for a chain of self, `self_effects` binds none of its prefixes; else the chain is the field of a name the
    call is handed.  Second, the map's reach from that call alone (`Rerun._keeps`, a copy of the row as
    `_called_at` asks) reaches no arm assigning the field another value in memory (`reach.moving_off_in_memory`,
    births left out), and leaves no hole; holes closed by their name are counted and said.  This rests on the reach's
    premises, and on the junctions as every assignment of the field by its name.  A write by a name made at run time
    is none: `JobSpec.pack`'s `object.__setattr__`; 421's closure holds one, peekJob's, on a JobSpec it makes.
- *The value a write assigns.*  Its constant, or a printed hole of the expression it assigns, unbound between.
- *The answer.*
  - A write with the observed value before it is a stay, and one assigning another value is no write of it: both are
    excluded.
  - Read from the last back, each write left may be the last change, down to one that ran with another value before
    it.
  - One left that ran is chosen ("the last change in the pass").  More are named, and none is chosen ("N may have
    changed it last in the pass: ...").

In the adder's pass, row 444 prints `job.jobStatus=failed jobStatus=failed` before finalize in the 4 m-failed adder
walks.  So handle_failed_job:374 is a stay: nothing on the way into it binds `self.job.jobStatus`, and its one call
is handed `self.taskBuffer`, `self.job.PandaID` and `self.job.attemptNr`.  The change itself is not read.  Between
416 and 444 the pass calls parse_job_output_report (421) and execute_plugin (425).  P's reading of a call reads both
as changing `self.job.jobStatus`: 820, 885, 919 and 959 bind it, and 425 hands `self.job` to the plugin.  Since the
map's RB (codemap-xx e0f6fce placed the `.get` calls at extract_executor_metadata:710 and update_outputs:393), the
reach from each reaches no arm assigning jobStatus another value than 'failed' in memory: 421 closes one call by its
name (DBProxyPool.getProxy:79 `proxy.wakeUp`), 425 eight (update_outputs' appends on dict elements).  So in
MF-7397695467 and MF-7398402231, 919's 'failed' carries to 441 (959 sets it 'failed' again; the calls at 895, 910
and 425 keep it), 441 is a stay, and the walks name 1813, 919 and 416.  In MF-7396944927 and MF-7397944082, 416 and
441 are both kept and no value reaches 441 (no row before it prints the status), so 1813, 441 and 416 stay named.
1813, archiveJob's UPDATE after the archive, assigns a statement's bind, so no value before it is read.  In the 65's
A-rec, K-rec and N-rec, 444 prints `jobStatus=finished`: 416 assigns 'finished' and is no write of 'failed', and 441
is the last change, as before (416's 'finished' does not carry: 421 reaches parse_job_output_report's 'failed' arms).

The Watcher's pass: run:117 sets `job.jobStatus = "failed"`, and archiveJob:1813 runs inside the call at 135.  P's
reading counts `EventServiceUtils.isEventServiceJob(job)` (129) and `getEventStat(job.jediTaskID, ...)` (130) between
them as calls handed the job, and the reach from each assigns jobStatus nothing in memory (130 closes one call by its
name).  So 1813 is a stay and 117 the last change: MF-7397938541 stops at H3 on 117's guard, MJ-sent-late at L2
(m-cut) and S2 (m-cut2), as before R.  updateJobs:654 assigns the job handed on in another class: it is not read as
the same object, and R leaves it as H2 did.

Measured (rr1 against r1, map x13xixB14m, out/s/s-predictions.txt), every prediction held:
- outcomes: MF-7396944927 and MF-7397944082 H3 -> H2; MF-7397938541 H3 -> H2; MJ-sent-late L2 (m-cut) and S2 (m-cut2)
  -> H2;
- text only: MF-7397695467 and MF-7398402231 (H2 as before), and 32 of the 65's runs (416 excluded, "the last change
  in the pass");
- attr and MT: the same.

With CK (ck1 against rb2, map x13xxB3m read with codemap-xx 5ccd571, out/ck/ck1-predictions.txt), every prediction
held: MF-7397938541 H2 -> H3, MJ-sent-late H2 -> L2 (m-cut) and H2 -> S2 (m-cut2); MF-7397695467 and MF-7398402231
"4 may" -> "3 may" (441 a stay); everything else the same, the 65, attr and MT included.

*Shows:* "rule R: a stay, ... already held ... before it (...)", "rule R: it assigns ... no write of ...", and the
choice's "the last change in the pass" or "N may have changed it last in the pass: ...".  A way kept by CK adds "but to
'<value>' again at <file:line>" and "the calls at <file:line>, ... reach no assignment of <field> but '<value>' (the
map's reach, N call(s) closed by their name)".
*Revisit:*
- with the status at the pass's start (the statuslog's row before the question's, under the premise that no status
  change went unlogged): 416's guard is then read, and the adder's MF-7396944927 and MF-7397944082, where no value
  reaches 441, may move;
- when a write through a statement's binds (1813) is read for the row's value before it;
- when the frames between a join's call and the write inside it are read: in the Watcher's pass, updateJobs (between
  run's call at 135 and archiveJob) assigns jobStatus in memory at 649, 652 and 654 for a jumbo job, and the stay of
  1813 rests on that frame unread;
- when the map gives a write by a name made at run time (`JobSpec.pack`) as an arm, or says which object it writes.

### K52. A writer that writes its loop's objects after the loop is read on its latest run before the row

H2 (`_wrote_at`), its scan (`_row_at`) and F1 read a candidate on the key's runs at the row's second.  A batched
writer assigns each object in a loop and writes them all by one call after it, so its run of the key ends before the
row's second.  datasetManager's FinisherThr.run prints "<PandaID> Finisher : Kill" (730) and sets the job failed
(735) in the loop at 662, then calls updateJobs (761).  In MF-7392812614 the run ended at 13:40:29, and the row was
written at 13:40:42.  So the candidate had "no run of the key" and stayed unexamined.

- *A batched writer* (`rerun.batched`) is one where the map's KeyFrom of the candidate's function says the object
  assigned at its line is handed to a call (`KEY_HANDED`), and a loop of the function holds the line and not the call.
- *Its run* (`Rerun._batched`) is the key's latest run in its logs between the row before and the row's second.  It
  counts only where the records hold it and the log was searched with no hole.  It is handed to the judge as the
  candidate's run, and judged as any run is.  It rests on no other pass of the writer printing the key between.
  FinisherThr runs in a pool of 3, and another thread's pass that printed nothing of the key is not seen.
- *None held*: the candidate stays as it was, and the walk says so.  Nothing is asked: the records' answers search
  each log's whole current file, so in m-failed only MF-7392812614's setupper writers lack the span (2 days back to
  10-31).  An ask back to the row before would stop that walk before H2 judges (tmaeno, W1a', 2026-11-03).

Measured (dw2 against tk1, map x13xxB3m read with codemap-xx 5ccd571, out/d3/dw2-predictions.txt), every
prediction held.  MF-7392812614 goes from STOP H2 to STOP H3: run:735 is handed its run, is positive by row 730, and
is chosen by positive evidence.  H3 then stops on its guards (`not (nTokens <= nOkTokens)`, `endTime < timeOut`), whose
values 707 and 727 print but H3 does not read yet.  Text only: a line per batched candidate held no run, in the other
6 m-failed, MJ-sent-late (m-cut, m-cut2) and the 65's A-db, A-rec, K-db, K-rec, N-rec with their rounds.  attr and MT
are the same.  The run is read only with K53: without it, 735's rows found nothing in it.

*Shows:* "<func>:<line> writes in a batch (assigned in the loop at N, handed on to <call> at M): its run is the key's
latest from the row before (...) to ..., ... in <log>, the log searched with no hole; it rests on ...", or "...: no
run of the key held from the row before (...) to ...; not asked".
*Revisit:* where H2 cannot choose without a batched writer's run that is not held: then ask back to the row before,
by the writer's mark where it has one.

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
