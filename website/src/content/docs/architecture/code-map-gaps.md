---
title: "Code Map: Known gaps"
description: Where the map's reading of SQL statements is known to fall short, how each gap shows in what the map outputs, and how much of the corpus it touches.
---

The map reads each SQL statement PanDA runs by running the Python that builds it, on every
path, over values ([Building the map](/bamboo/architecture/code-map-building/)). That
reading is not exact. Some of its gaps are being worked on; others are accepted, because
closing them would cost more than they hurt. Either way, whoever reads the map — a person or
the querier — needs to know they are there, because **an entry the map got wrong looks
exactly like one it got right**.

This page lists them: what goes wrong, how it shows in the map's output, how often, and
whether anything is planned.

:::caution[What this page covers]
Only the SQL reader: the statements (`read_sites`), the terms some of a statement's builds
hold (`added_under`), the row preconditions on junction branches, and the columns the write
side reads. It describes the reader that runs the function (`BAMBOO_SQL_EVAL=1`); the
reader that reassembles statements from their shapes has gaps of its own that are not
listed. Other slices are not covered yet. **A gap missing from this page is not evidence
that the map is right.**
:::

## How to read the tables

- **The sections are by how a gap shows**, because in production the output is all a
  reader has: *Silent* — nothing marks it, and the entry reads like any other; *Said
  wrongly* — the map reports something, but not the real cause or place; *Said* — the map
  reports it, and the text it reports is quoted; *In the build report only* — `build-map`
  counts it, and the graph does not carry it; *Latent* — not wrong today, and nothing would
  say so if it became wrong.
- **Size** is counted on panda-server `2cd8c424` with bamboo `14ad4d6` and
  `BAMBOO_SQL_EVAL=1`, from `bamboo diff-map --dump` or from a census over the same reader,
  unless the row says otherwise. As everywhere on these pages, the command is the authority:
  a later build moves these figures.
- **Status** is *open* (nothing decided), *deferred* (decided to do later) or *accepted*
  (decided not to fix; the row says why).

## Silent

These matter most: nothing in the output tells a reader to doubt the entry.

| Gap | Example | Size | Status |
|---|---|---|---|
| **A value paired with a table its branch does not read.** `selects` gathers the values of every build, and a value one branch fills is paired with the term of every build — including one whose table that branch never reads. Where the builds differ in the term rather than the table, `added_under` does say so (`no build holds it`, under *Said*). | `get_num_jobs_with_status_by_nucleus` selects `JOBSDEFINED_SHARE_STATS.jobStatus=transferring`, but with `transferring` it always reads `JOBS_SHARE_STATS`. The function's routes (`call_readings`) are right. | 1 found; not counted | open |
| **A value an `if` rewrites keeps no test for the branch that does not.** A value written before an `if` and rewritten under it reaches the statement when the `if` does not hold, but that is no test *above* the earlier write, so the place it gets lacks it. | `analy_pmerge_jobs_wait_time` writes `('activated', 'sent')` before `if site_workflow ...` and `('activated', 'sent', 'starting')` under it. With no other test above it the map says `no test chooses it` (under *Said wrongly*); with one, the place shows that test alone. | not counted | open |
| **Two branches that append the same text count as one.** The reader hands one statement per distinct text, so a term two branches append alike is placed under one branch's tests. A value formatted into the statement is the exception: every assignment of it that reaches the call is a place. | — | not counted; none of the places the reassembly found is missing | open |
| **A repeated part with several templates is read as one pass of each, together.** What a loop of unknown length appends is one part holding one pass of each text it can append; texts no single pass appends together are read side by side — the union. | `getJobs` orders by `(CASE {} ELSE {} END)` from two `WHEN` templates | 37 of 136 repeated parts have more than one template | accepted: a part decided at run time is read for the columns and values it can hold, not for which of them go together |
| **A loop that reads the text it builds is read as one or two passes.** Only a loop whose body does nothing with the name but append to it is read as a repeated part. | a spec's `columnNames` (`if ret != "": ret += ","`): the column lists of `SELECT` and `INSERT INTO` | 74 calls in 38 functions, 70 of them through `columnNames` | deferred |
| **The evaluator's own approximations.** Values a loop carries between passes are forgotten (numbers always); the tests that decide a guard are not tracked; a dict is tracked by literal keys only; lists are values, not aliases; a nested function runs only when the resolver calls it; a loop over a sequence of unknown length is taken to run at least once. Each is documented in `bamboo/codemap/panda/sqleval.py`. Where one leaves a value unknown, the statement holds a hole and that is said; where it leaves a value wrong, nothing is. | — | not counted | accepted: the stated limits of the evaluator's design |
| **What needs every statement says less past the cap.** A call can run more statements than the reader expands (512). Past it, the values every statement refuses or restricts (`refuses`, `restricts`) are read off the statements the reader draws, a disjunction of conditions across statements is not said, and neither is a precondition only some statements carry (`row_precondition_when`). Whether some build lacks a term (`added_under`) is always read off the statements drawn. | `getJobs`, `duplicateFilesForReuse_JEDI` | 2 calls | accepted: the cap bounds the build's time |
| **A conditional precondition is approximate.** `row_precondition_when` comes from the statements the reader draws, so a combination it does not draw is not said. "Under C the fragment carrying the term is appended" is read off the path, and a `return` or exception between C and the fragment breaks it. A local flag (`updateAttemptNr is True`) is given as written, which a reader cannot evaluate. | `updateJobStatus` appends `AND NOT jobStatus=:ngStatus` only under `jobStatus == "transferring"` | not counted | open |
| **A flush method's guard under a test on its parameter is said for every caller.** A method that saves a spec through `bindUpdateChangesExpression` gives its callers' branches the terms of its `WHERE` as the row they need (`row_precondition`), leaving out a term a parameter decides for a caller that does not pass the parameter. The parameter is found by looking for the term's text in the method's source, so a term the method formats (`f"status IN ({names})"`) is tied to none, and every caller is told it. | `updateTask_JEDI` appends `status IN (:old___in)` only `if oldStatus is not None`. `AtlasProdWatchDog.doActionForReassign` passes no `oldStatus`, yet its branches carry the term, and the answer for `JediTaskSpec.status` `ready` prints "only lands on a row where status IN (:old___in)" under it. | 15 branches in 5 functions; the 26 branches whose callers pass `oldStatus` are right. No other flush method's guard is of this kind | open |

## Said wrongly

The map reports something here, so a reader is warned — but about the wrong thing.

| Gap | Shows as | Size | Status |
|---|---|---|---|
| **A test does choose the term.** (a) The table is chosen in another fragment than the term's. (b) A value written before an `if` is rewritten under it (*Silent*, above). (c) One build of the statement does not parse, so the term looks missing from that build. The other six of this reason are right: a loop over a constant list runs every one of its statements (`get_job_statistics_per_site_label_resource` 2, `copyArchive`'s `main` 4). | `no test chooses it` | (a) 2, `get_num_jobs_with_status_by_nucleus`; (b) 1, `analy_pmerge_jobs_wait_time`; (c) 1, `getFilesInDatasetWithID_JEDI`'s order `fileID` — with no condition its text is `WHERE  ORDER BY` | open |
| **A call in a nested function is reported again from the function around it.** The nested function is read on its own and its statements are there; the outer one reports the same calls as unreached. | `no statement this function builds reaches this call` | 9, `datasetManager`'s `main` | open |
| **A statement held in a dict is reported as unreached** rather than as unreadable text. | `no statement this function builds reaches this call` | 1, `bulk_insert_new_jobs` | open |
| **A condition common to a `UNION`'s branches repeats a test.** It is placed under every branch's `if`: true, but the same test once per branch. | — | `getPandaIDsWithTaskID`'s `scout_only`, four times | open |

The same reason is right once: `AtlasTaskWithholderWatchDog` runs a statement under
`if dry_run:` after `dry_run = False`, and no path reaches it.

## Said

What the map reports at the statement, with the reason it gives. The counts are entries of
`reading_stops` unless a row says read sites; `read_sites[].unresolved` carries the same
reasons on each read site, with the part not read.

| Reason | Count | What it usually is |
|---|---|---|
| `the row precondition names a bind nothing here fills` | 91 | a precondition whose bind the function does not fill: `status IN (:ost1,:ost2)`, `locked_by=:locked_by` |
| `the WHERE names its bind variables through an interpolation` | 50 | binds or a bind's name built at run time: `jediTaskID IN ({})`, `:w_{}`, `rownum<={}` |
| `the statement runs with no bind map, and this build reads a statement only with its binds` | 29 | |
| `the WHERE names its column through an interpolation` | 11 | columns from a caller's dict — `criteria`, `map`, `data` — in 7 functions (`updateTask_JEDI`, `getTasksWithCriteria_JEDI`, `queryDatasetWithMap` …). In all, 21 loops in 15 functions build columns from a parameter's keys — in `SET` and `SELECT` lists too, where nothing is said — and no reading is handed the caller's dict (counted at `f9ca59f`) |
| `no statement this function builds reaches this call` | 11 | right once; see *Said wrongly* |
| `the statement is handed in by the caller, so its text is not in this function` | 7 | |
| `the WHERE takes whole conditions through an interpolation` | 2 | |
| `the statement is a name bound here to text this build cannot render` | 1 | |
| `a path into a JSON column, which this build does not read` | 26 read sites | `read_sites[].unresolved` only |
| `the statement does not parse` | 3 read sites | `read_sites[].unresolved` only |
| `no build holds it` | 4 terms | `added_under`: the pairing under *Silent*, said — `getScoutJobData_JEDI` pairs the input types with the dataset table's term and the merge types with the file table's |

### In the build report only

Columns the write side cannot give a value. `build-map` prints them under
`sql-write candidates not accounted for`; the graph holds no entry for them. A column without
a value either leaves its junction unbuilt, or leaves one way of setting the value off a
junction that otherwise looks complete — the dangerous case, which the report lists by name.
Counted as distinct (function, table, column).

| Reason | Count | What it usually is |
|---|---|---|
| `the varmap is one of a list handed to executemany` | 81 | `configurator_write_ddm_endpoints`, `updateEventRanges`, `insertNewJob`: bind maps filled per row and appended to a list |
| `nothing in this function binds this placeholder` | 22 | |
| `the placeholder is spelled by an expression` | 14 | `propagateResultToJEDI`'s `varMap[f":{tmpKey}"]` over a local list; `updateJobStatus`'s `memory_leak`; `updateJob` |
| `the execute call names no varmap` | 9 | |
| `the placeholder is bound without its leading colon` | 4 | |
| `bound outside the window this execution owns` | 2 | |

## Latent

Nothing is wrong here today, but nothing would say so if it became wrong.

- **Two readers still read the plain text.** `flush.py` and `trace.py` match a statement by
  regular expression rather than reading what the evaluator hands them. They agree today,
  because a repeated part reads as `{}` to both.
- **Stops spell an `IN` list as its loop's template**: `IN (:pt_{})`, where the reassembly
  said `IN ({})`. A reader that matches stop text by pattern sees a different spelling.

## Keeping this page honest

The sizes are counted at the versions stated above and are not checked: re-count them with
the command when the reader changes. The reasons the tables quote are checked:
`tests/test_docs.py` fails when one is no longer a text the code gives, so a renamed reason
cannot leave this page describing output the map no longer produces.
