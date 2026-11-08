# WITNESS_PREREG — layer A kill gate, preregistered

status: DRAFT until sealed. the seal is the sha256 of this file as written to line 1 of `seal.jsonl` in the live
store (`C:/Users/heyzo/clawd/witness/seal.jsonl`, `{event:"prereg_sealed", file, sha256, sealed_at}`), the first
time the witness opens that store, BEFORE the tail ever stamps a line of the live ledger
`C:/Users/heyzo/clawd/gate/ledger.jsonl`. `seal.jsonl` is append-only and never derived from anything else;
`index.json` copies the seal from it (never from a previous index), and `witness.cjs check` exits 1 if `seal.jsonl`
is absent, if this file's sha256 differs from the sealed one, or if `index.json` disagrees with `seal.jsonl`. the
sealed sha256 is itself stamped in the first live batch as `proofs/prereg_<sha256>.ots`, so the seal has a bitcoin
block time of its own that a stranger can check (until that block arrives, "sealed before the tail ran" rests on
the lab's word and `seal.jsonl`'s order). after the seal this file is not edited; an erratum is a separate file
beside it, dated.

written 2026-11-08 by the build agent for item 1a of THE_PLAN (wt/_dream_2026_11_08/THE_PLAN.md §2 tier 1); revised
the same day after the review (orphans, window, tail lag, seal, wording), still before any live run. the operator's
act is one: approving the scheduled task in SCHEDULED_TASK.md. nothing here moves anything the agent reads during
the window flobi approved; the witness reads the ledger and writes only under its own store.

## 1. what is being measured

the object: every entry of the gate ledger, identified by its `entry_id` (sha256 of the line's JCS core; PROTOCOL.md
§3). the witness stamps each entry_id through the opentimestamps aggregators and later reads the bitcoin block the
calendars commit it into.

the clocks, named:

| field | whose clock | what it is |
|---|---|---|
| `line_timestamp` | the gate's (the writer's statement, PROTOCOL §1) | the entry's own `timestamp`; the store carries it for every line, withheld lines included |
| `seen_at` | the tail's (the lab's word) | when the witness read the complete line |
| `stamped_at` | the tail's (the lab's word) | when the aggregators answered with a pending commitment |
| `block_time` | the bitcoin block header's `time`, read from two explorers that must agree | the only clock here the lab does not run. it is the miner's: bitcoin accepts a header time up to about 2 h ahead of network time and above the median of the previous 11 blocks, so it sits within about 2 h of consensus wall time. buckets under 6 h carry that slop; a negative `block_time − line_timestamp` can appear for an honest line and is printed, not hidden |

words: a pending `.ots` file is a calendar's RECEIPT — a url and a commitment, no time a stranger can check. a PROOF
is the file once it holds a bitcoin attestation. "stamp latency" (`stamped_at − seen_at`) is the lab's own
measurement of its own process and is labelled so wherever it is printed; one batch is one measurement, however many
lines share it. "tail lag" (`seen_at − line_timestamp`) is process evidence: how long after the write the witness
saw the line (about 5 min when the task runs; hours after an outage; days for the lines written before the witness
started). "outside latency" (`block_time − line_timestamp`) is the number the exhibit's sentence rests on.

## 2. the opening window, and the exact command

the opening 7 live days: from the first `seen_at` the live store records (day 1, UTC) through the end of day 7 UTC.
the denominator for bars (a) and (b) is the set of entries whose `line_timestamp` falls inside those 7 days, counted
from the ledger bytes at the close of day 9 UTC (48 h after the window, so bar (b) can be read). entries written
during the window but seen by the tail after day 7 (an outage) still belong to the denominator: a line the tail did
not see in time is a miss, not an exclusion. REPORT.md prints the tail-lag counts so an outage leaves a number.

the lines written BEFORE the first live run (734 at the time of writing, timestamps 2026-10-24..11-08) are stamped
retroactively by the first batch; their outside latency is days by construction and they are OUTSIDE the window
denominator. REPORT.md prints them as "lines stamped retroactively (seen more than 1 h after their own timestamp):
r of N" and computes their median separately from the live lines' median.

the measurement is this command, run at the close of day 9 against a copy of the ledger taken then:

    node witness.cjs report --store C:/Users/heyzo/clawd/witness --ledger <copy of ledger.jsonl at the close of day 9>
                            --since <day 1 00:00:00Z> --until <day 8 00:00:00Z>

`--since` is day 1 at 00:00:00Z (the UTC day of the first live `seen_at`), `--until` is day 8 at 00:00:00Z
(exclusive). with a window given, REPORT.md prints a "kill gate" section: the window and its denominator on their own
line; the in-window lines not yet 48 h old at report time (0 at day 9); the in-window lines stamped retroactively;
bar (a)'s median with its count and the not-computable rule; bar (b)'s count verbatim; and the gate verdict from (a)
and (b), with bar (c) named as not computed. without a window the report says the gate is not evaluated. the REPORT.md
written at day 9 is published with the counts whether the gate fires or not.

## 3. the kill gate — three bars, any one fires it

layer A is called FAILED, and the counts published as the result, if at the close of day 9 any of these holds:

(a) **median outside latency over 24 h.** median over the in-window entries that have a bitcoin attestation verified
against two agreeing explorers, of `block_time − line_timestamp`, is greater than 86 400 s. (entries with no
attestation are not in this median; they are counted under (b).) if fewer than 20 in-window entries have an
attestation the median is printed with its count and bar (a) is read as not computable, which is also a FAIL. the
report's line: "bar (a) median outside latency over the attested in-window lines: <median> (over k) — did not fire /
FIRED / NOT COMPUTABLE".

(b) **more than 5 of every 100 lines with no bitcoin attestation within 48 h.** count the in-window entries whose
latest state at the close of day 9 is anything other than `bitcoin` with `block_time ≤ line_timestamp + 172 800 s`
(so: `pending`, `stamp_failed`, `explorers_disagree`, `bitcoin_mismatch`, no record, or `bitcoin` with a later
block). if that count × 100 > 5 × denominator, the bar fires. the report's line: "bar (b) lines with no bitcoin
attestation with block_time ≤ line_timestamp + 48 h: k of n — did not fire / FIRED". printed as "k of n", never as a
percentage alone. (`explorers_disagree` and `bitcoin_mismatch` are revisited on every upgrade, so a transient
explorer lag counts only if it is still unresolved at day 9.)

(c) **a stranger fails to pin 95 of 100 sampled lines.** a stranger is a machine holding no lab files (a fresh vm or a
cloud sandbox) or a named person outside the lab, never a process on the box. this bar is RUNNABLE ONLY AFTER PART 2
publishes the store beside the ledger on the exhibit branch (`ledger/witness/index.json`, `ledger/witness/proofs/`);
its clock starts when those urls exist and the stranger is given them, and the urls are written into the stranger's
transcript. they draw 100 published entries by a rule fixed before drawing (the first 100 entries in the window
denominator whose seq ≡ 0 mod 3, in seq order; fewer if the denominator is small, then all), take each `entry_id`
from the published ledger, fetch `proofs/<entry_id>.ots` from the published store, and run
`node ots-cli.js verify -i -d <entry_id> <file>.ots` (javascript-opentimestamps 0.4.9) or
`ots verify -d <entry_id> <file>.ots` (python client) against an explorer or their own node. a line is PINNED when
the tool prints a bitcoin block height for that digest. a PIN FAILURE is: the tool runs, exits 0 or 1, and prints no
block for that digest (pending, or no attestation). a TOOLING FAILURE is anything else (install fails, explorer
rate-limits, network down, file not fetched) and is retried once; if it persists it is counted and published as a
tooling failure, not a pin failure. the bar fires when fewer than 95 of the 100 are pinned, pin failures and
unresolved tooling failures both counting against; both counts are published separately. the stranger's command
transcript is published beside the counts.

the bars are the builder's, set before the tail ran live, from the lab's three prior stamping events (42 min, 57 min
and 1 h 49 min from stamp to block; anchors/index.json against mempool.space header times, read 2026-11-08) plus the
5-minute schedule and one assumed outage day in seven. they are not tuned after the window opens.

## 4. what is published either way

at the close of day 9: REPORT.md with every count and its denominator, as `report` prints it: the counts section
(commitments, pending, bitcoin, disagree, mismatch, none, failed, orphans — store digests no longer in the chain —
and seqs with more than one entry_id in the store); tail lag (buckets, median, retroactive count, run gaps); stamp
latency (batches beside lines; both medians); outside latency (buckets, all-lines and live/retro medians, the 48 h
miss count); pending age (older than 48 h, older than 7 d); the runs section from runs.jsonl (runs that found the
ledger not re-deriving, runs that found a partial last line, runs with a stamp error, runs that found orphans); the
lists (seqs with no commitment, seqs with stamp_failed, calendars by uri with the count of lines naming each, the
seal with its record and its own proof's state); and the kill-gate section. the stranger's transcript and both of its
counts for bar (c).

## 5. the sentence the exhibit may carry if no bar fires

> every published line with a timestamp from <day 1> to <day 8> has a calendar commitment (k1 of n); k of n carry a
> bitcoin attestation within 24 h of the line's own timestamp (block header time minus the gate's timestamp); r
> lines written before the witness started carry a proof dated by their first block, not their own day.

with k1, k, n and r read from the kill-gate section of the day-9 REPORT.md. the stamp latency (the lab's own clock)
is NOT in the sentence; it stays in REPORT.md under its "the lab's word" heading.

## 6. the sentences it may never carry

- "within seconds" or "within minutes" (a pending file carries no time; the lab's three prior events reached their
  blocks in 42 min, 57 min and 1 h 49 min). the stamp latency median, whatever it reads, never enters an exhibit
  sentence: a calendar's acceptance carries no time
- "every line carries a proof" while pending files are counted: a pending file is a receipt; the proof is the
  attested file. the exhibit counts commitments and attestations separately
- "the lab cannot rewrite" (a proof pins a digest's existence before a block; the lab can stamp two chains and
  nothing in a calendar shows it; what the witness shrinks is the window, and what it leaves behind is the orphan
  count and the duplicated seqs in the store — printed, not hidden)
- "signed" (nothing here is signed; PROTOCOL §7)
- "provably", "first", "no one"
- any rate without its denominator printed beside it
- any sentence about the words, truth, authorship or completeness of a line (a send with no line gets no witness)

## 7. what firing means

the build is called failed in THE_PLAN's ledger of items; the counts are published as the result; the tail may keep
running (its records are still evidence) but the exhibit carries no layer-A sentence until a repaired layer with its
own preregistration is sealed. a bar that did not fire is reported as "did not fire: k of n", never as "passed".
