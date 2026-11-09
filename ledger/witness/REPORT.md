# witness report — layer A

written 2026-11-09T05:30:06.709Z from `witness`; ledger head `3f9fe35a1b276ed34933b75cc0cd075383b8bd5e89ea93aec603cc62f116cf3b`

every number below is a count with its denominator. the tail's clock (seen_at, stamped_at) is the lab's word;
line_timestamp is the gate's own clock; block_time is a bitcoin block header's time read from two explorers that
must agree — the miner's time, within about 2 h of consensus wall time, so buckets under 6 h carry that slop.
a pending .ots file is a calendar's receipt and carries no time a stranger can check; a proof is the file once it
holds a bitcoin attestation. "proof" below means that.

## counts

- lines in the ledger (published + withheld bridged by digest): 792 (792 published, 0 withheld; partial last line: 0 bytes)
- lines with a calendar commitment (a stamp request accepted by at least 2 calendars; carries no time): 792 of 792
- lines pending (commitment, no bitcoin block yet): 49 of 792
- lines with a bitcoin attestation verified against two explorers (a proof a stranger can check): 743 of 792
- lines where the two explorers disagreed on the header (retried every upgrade): 0 of 792
- lines whose attestation did not match the block's merkle root (retried every upgrade): 0 of 792
- lines with no commitment and no attestation: 0 of 792
- lines whose last stamp attempt failed (retried next run): 0 of 792
- store entries whose digest is no longer in the chain (orphans; a rewrite leaves these): 0 of 792
- seqs with more than one entry_id in the store (a rewrite leaves these too): 0

## tail lag — seen_at − line_timestamp (the tail's clock minus the gate's: process evidence, not outside evidence)

- within 10 min: 13 of 792
- within 1 h: 24 of 792
- within 24 h: 64 of 792
- over 24 h: 728 of 792
- median: 4 d 8 h (over 792)
- lines stamped retroactively (seen more than 1 h after their own timestamp): 768 of 792
- lines with no line_timestamp in the input (withheld, bridged by digest only): 0 of 792
- runs with a gap over 10 min to the previous run (a stopped task leaves a count here): 3 of 9; largest gap 5 h 21 min

## stamp latency — stamped_at − seen_at (the tail's own clock: the lab's word, not an outside time)

- batches (distinct seen_at/stamped_at pairs; one batch is ONE measurement, printed once per line): 5 for 792 lines
- within 10 s: 792 of 792 lines
- within 60 s: 792 of 792 lines
- within 10 min: 792 of 792 lines
- over 10 min: 0 of 792 lines
- failed: 0 of 792
- median over lines: 1 s (over 792); median over batches: 2 s (over 5)

## outside latency — block_time − line_timestamp (block header clock minus the writer's clock)

- within 1 h: 0 of 792
- within 6 h: 11 of 792
- within 24 h: 15 of 792
- over 24 h: 728 of 792
- none yet (no bitcoin attestation): 49 of 792
- negative (block before the line's own timestamp; a clock question, the miner's slop): 0 of 743
- median over all attested lines: 4 d 9 h (over 743)
- median over lines stamped live (seen within 1 h of their timestamp): 1 h 21 min (over 9); over lines stamped retroactively: 4 d 9 h (over 734)
- lines with line_timestamp at least 48 h before this report and no bitcoin attestation with block_time ≤ line_timestamp + 48 h: 708 of 728 (seqs 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60 … (+648))

## pending age — now − stamped_at (the tail's clock; a dropped commitment and a two-hour wait differ here)

- pending older than 48 h: 0 of 49
- pending older than 7 d: 0 of 49
- median age of pending: 29 min 56 s (over 49)

## runs — from runs.jsonl

- tail runs: 10; upgrades: 14; first run 2026-11-08T21:01:32.514Z, last run 2026-11-09T05:30:02.285Z
- runs that found the ledger not re-deriving (nothing stamped): 0 of 10
- runs that found a partial last line: 0 of 10
- runs with a stamp error: 0 of 10
- runs that found orphans (store digests not in the chain): 0 of 10

## lists

- seqs with no commitment and no attestation: none
- seqs with stamp_failed as last state: none
- calendars named in pending attestations (lines naming each, of 792): https://alice.btc.calendar.opentimestamps.org 792; https://bob.btc.calendar.opentimestamps.org 792; https://btc.calendar.catallaxy.com 792; https://finney.calendar.eternitywall.com 792
- prereg seal: WITNESS_PREREG.md sha256 43f315594728a7f21f9685cf19810b5303628198d0507c6eece8da51e716503b sealed 2026-11-08T21:28:19.041Z (seal.jsonl line 1); its own proof: bitcoin block 975108 time 1794179175

## kill gate (WITNESS_PREREG.md §3)

the kill gate is not evaluated by this report (no --since/--until). bar (c) is a stranger's transcript and is never computed here.
