# CORRECTION — PREREG_pmnist_robustness_2026_10_11: what the receipt's seeds 0 and 1 enter, rounding, what is committed and in which order, the order of the pre-run checks, the seal and its anchors, and night 3's prompt, fixed before any data

**status: a correction and a set of reading rules for a frozen document, written before any data and to be sealed
beside it (anchor 7b in `papers/checksum/SEALS_2026_09_13.md`). The PREREG is not edited; its seal digest is
unchanged (row 7, `2068e6733c1d5d2c4305cddc89e6e5f2ddd3671406e8bcf56332251d05db648e`, frozen at `3598e59d`).
Everything here answers the lab's two reviews of the sealed text (2026-10-11 and 2026-10-12). No run, no smoke and
no leaf exists, and nothing here reads a number from the run. The author of this file has not opened the sealed
untied receipt (`pmnist_untied_result.json`).**

Sections 1 to 7 carry the content of the section 15 ("Amendment, before any data (2026-10-12)") that commit
`98072bed` appended to the PREREG and that SEALS row 8 sealed, in the same order (its 15.1 to 15.7), restated for the
route the operator chose on 2026-10-12 (section 6). Sections 5 and 6 are rewritten for that route, and section 1
carries the corrections of the lab's second review. Row 8's text survives only in its blob
(`git show 98072bed:papers/frequency-resonance/PREREG_pmnist_robustness_2026_10_11.md`); where it and this file
differ, this file governs, and row 8 is not to be anchored.

**No bar moves**: no gate, value, operator, metric path, row, arm, seed, count, permutation derivation, threshold or
schedule changes. The PREREG's bytes are row 7's, so the gates block's sha256 as `styxx.protocol` computes it
(`gates_sha256`) is `578924f8e776573cc82db8fbbd8bb9cf84106b81171a5ba6d5783e422c96e854`, as it was at `3598e59d` and
at `98072bed`.

## 1. corrected: the receipt's seeds 0 and 1 enter one gated metric by value

Three sentences of the PREREG are wrong as written, and stay as they are: the opening's "those two enter no gated
metric (§4)"; §3's "they feed `anchor_max_abs_dev` (reported) and **no gated metric**"; and the last clause of
`G_C_anchor`'s power basis, "seeds 0 and 1 are the receipt's own and enter no gate", which sits inside the frozen
gates block. This section corrects all three in prose. No byte of the block changes, and a power basis decides no
row.

**By value.** The six re-run trainings (FREE, CLAMPED and REAL2 at seeds 0 and 1 on permutation 1234) are 4000-step
runs, and `degenerate_fraction` (§4) is degenerate 4000-step runs over 60: the sixty 4000-step runs, the six among
them, and not the five 12,000-step runs, which §4 leaves out of it. The runner counts them (its pool for that metric
is every 4000-step leaf at H = 64). By value the six enter exactly one gated metric, `degenerate_fraction`
(`G_D_stable`), and no other: not the floor and not `anchor_gap_newseed` (both read anchor seeds 2, 3, 4), not any
permutation-averaged gate (held-out permutations only), not `gap_long` (its own 12,000-step runs). This is kept,
not changed in code: `G_D_stable` is a validity gate over the sixty 4000-step runs, and dropping six of them would
let a collapse of the re-run go uncounted. The bar stays at 0.10.

**As leaves.** "And no other" holds for the six runs' values only. Like every leaf, the six leaves count toward
`G_P_plumbing`: `plumbing_ok` reads 0 if any of them is absent, reads back module values other than the
schedule's, names other imported-runner blobs, another data digest, another PREREG digest or another digest of this
file (section 5), lacks `preflight_ok`, carries a curve of the wrong length or spacing, has a final accuracy unequal
to its last curve point, or is not the leaf log's latest entry for its name (§9.4, §9.5).

**What a reader of the sealed receipt knows.** The committed `run_pmnist_untied.py` trains FREE, CLAMPED and REAL2
again at seeds 0 and 1 on permutation 1234 and writes each re-run's final accuracy (`test_acc` per arm and seed): the
sealed receipt holds six re-run accuracies, not the "two" the opening says. Four of the six configurations, FREE and
CLAMPED at seeds 0 and 1, have public values in `pmnist_ablation_result.json` (`test_acc`: FREE 0.924 and 0.915,
CLAMPED 0.5529 and 0.6617); the sealed receipt's re-runs agree with those only up to device nondeterminism, which
that receipt's `anchor_max_abs_dev` measures on the two arms' seed means (§7 lists
`INVALID__plumbing_anchors_drifted`, a deviation above 0.03, among the verdicts it can carry). REAL2 at seeds 0 and 1
has no public value. Whoever reads the sealed receipt before this run therefore knows six of this design's sixty-five
final accuracies, each up to device nondeterminism, and could foresee at most 6 of the 60 runs that
`degenerate_fraction` counts.

**Withdrawn.** This branch's CHANGELOG entry as written at `4e3d8be7` and reworded at `e8b91392`, and those two
commits' messages, said that four of the six accuracies in the sealed receipt "are already public" and that its
reader learns two new numbers. That is withdrawn: the four public numbers are the 2026-07-23 run's, the sealed
receipt's are re-runs, and how closely they agree is a fact about a receipt no one writing this has read.

The RESULT repeats this section beside the PREREG's three sentences, citing this correction.

## 2. rounding before scoring

Every gated metric is computed in double precision from the leaves and written to `result.<name>` rounded once to
four decimals (Python `round(x, 4)`; a value that sits half-way in decimal is decided by its binary double, as
`round` decides it); `Experiment.score` compares the rounded number. A metric within 0.00005 of a bar is therefore
decided by its rounded value: `real2_minus_clamped_perm_mean` = 0.03004 is written 0.03 and passes `<= 0.03`;
`gap_perm_mean` = 0.14996 is written 0.15 and passes `>= 0.15`; `gap_perm_ci_lo` = 0.00004 is written 0.0 and fails
`> 0.0`. `degenerate_fraction` is k/60 and `plumbing_ok` is 0 or 1, so the rounding cannot move either across its
bar. A final accuracy is a count over 10,000 test images, a multiple of 0.0001, so a leaf's four-decimal
`final_acc` loses nothing. The convention is the committed untied runner's (`round(..., 4)` on every metric before
`Experiment.score`). The non-finite rule of §9.7 applies before the rounding. Reported numbers are rounded the same
way (the bootstrap's two variances to six decimals).

## 3. what is committed, and the order of writes

With the RESULT, in its commit: every leaf (`pmnist_robustness_runs/p*.json`), the leaf log
(`pmnist_robustness_runs/leaf_chain.jsonl`) and the receipt (`pmnist_robustness_result.json`). The receipt inlines
every leaf under `runs` and carries the numbers on its own; the leaves and the log are committed so that the
receipt's `leaf_chain` block can be re-derived from bytes. Not committed: `_mnist.npz` (gitignored, §1), the smoke
receipt (`pmnist_robustness_smoke.json`, gitignored by the lab's `papers/**/*_smoke.json` rule; INVALID by type) and a
leaf's temporary file (`*.tmp` in the leaf directory, gitignored). The leaf directory is marked `-text` in
`.gitattributes`, so no checkout rewrites a leaf that the log names by its sha256, and the aggregate hashes each leaf
with CRLF folded to LF (a leaf as written holds no CR byte), the rule §1 applies to the imported runners.

Order of writes: the runner appends the log line for a leaf and fsyncs it, then writes the leaf to `<leaf>.tmp`,
fsyncs it, and renames it into place. A crash between the two leaves a log line with no leaf: the night re-runs that
run, the log carries a second line for the name, and the receipt lists it under `leaf_chain.rewrites`, visible and
not fatal. (Written the other way round, a crash between the two left a leaf the log never named, which reads
`plumbing_ok` = 0 until the leaf is deleted by hand and the run redone, with no trace in the log.) A log whose last
line is torn (a crash inside the append) is refused by name the next time it is read, and a night reads it before
any of its runs trains; nothing repairs it silently, and the RESULT reports any repair made by hand. Every leaf, the
smoke receipt and the receipt are strict JSON: a loss the training loop prints as `nan`, `inf` or anything
unparseable is written as null and flagged by `nonfinite_loss` (the run is degenerate, §6), and any other non-finite
number is written as null and its path listed under `nonfinite_fields`. §8's "loses at most one run" holds.

## 4. the order of the pre-run checks

The §9.4 list is unchanged, with item 3 read as section 5 reads it; this is the order it runs in. Before torch is
imported: a night 2 or 3 without `--anchors` or `--fallback-perms` is refused (section 5 says why `--beacon` alone no
longer starts one); items 1–3 (the two runner blobs, the data pin, the PREREG's blob and this file's, `Experiment`
and `check_metrics()`); the leaf log, read whole (section 3); the smoke receipt's presence; item 6 (the derivation;
`--anchors` is verified over the network here, for both rows of section 5); the night's projected schedule and its
refusals (§8). Then the committed runners are imported (a box without torch is refused by name, not by a
traceback), the committed constants are checked, and item 7 (CUDA available and at least 6 GiB free) runs before any
tensor is placed on the device: `R.redteam()`'s scan check (item 4), the earliest tensor the run puts on the card,
comes after the guard, then item 5 (construction on CPU). The smoke runs the same order without the night-only steps.

## 5. the seal, the anchors, and what the runner pins

The PREREG is not edited. Its seal is SEALS row 7: the sha256 of its git blob, `2068e673…`, frozen at `3598e59d`.
The file at HEAD is those 56,960 bytes, restored at `3a505c26` after `98072bed` had appended a section 15 to it and
`1ef6d242` had sealed the result as row 8. Wherever the PREREG names its own blob digest (the opening lines; §3's
"blob sha256 of this file" and "this file's digest"; §9.4 item 3's "the sealed blob"; §9.7's `prereg_blob_sha256`),
it means row 7's digest, as it did when it was written.

This file is sealed on its own, as row 7b: the sha256 of its git blob, frozen at the commit that adds it, which adds
nothing else. Row 8 (`18aa265c…`) is not to be anchored. A memo carrying row 8's digest draws nothing and seals
nothing this run reads; if one exists, the RESULT records it.

The beacon of §3 is the block hash of the slot of the earliest confirmed memo `styxx sealed-prereg <row 7's digest>`
that the creator wallet signed, as `python -m styxx.clock verify` prints it on a line that reads ANCHORED. Nights 2
and 3 do not start before rows 7 and 7b both read ANCHORED, or the labelled sha256-of-tag fallback of §3 is declared,
as the beacon-draw run waits for rows 6 and 6b. Night 1 (permutation 1234 only) needs neither, and neither does the
smoke. The operator anchors row 7's memo, then row 7b's, and no memo for row 8.

What `run_pmnist_robustness.py` does under this file, which its tests assert:

- it carries row 7's digest as `PREREG_BLOB_SHA256` and this file's as `CORRECTION_BLOB_SHA256`, and refuses before
  data unless the PREREG and this file on disk, LF-normalised, hash to them: §9.4 item 3 is read as naming both
  files, every leaf and the receipt carry `prereg_blob_sha256` and `correction_blob_sha256`, and `plumbing_ok`
  re-checks both, in the leaves and on disk;
- under `--anchors` it runs `styxx.clock.verify` over the file, requires a line reading ANCHORED for row 7's digest
  and one for row 7b's, refuses lines for row 7's digest that read ANCHORED with different beacons, and takes the
  beacon from row 7's line;
- `--beacon <hex>` starts a night 2 or 3 only beside `--anchors`, and must equal the beacon so verified: a hex copied
  from a verify line shows that row 7 read ANCHORED, not that row 7b did. `--derive-perms --beacon <hex>` still
  prints the five permutations a beacon gives, and starts nothing.

`styxx.protocol.Experiment` records as `prereg_commit` the commit that added the PREREG (`3598e59d`; it reads
`git log --diff-filter=A`), whose bytes are again the PREREG's at HEAD; the bytes a receipt was scored under are
named by `prereg_blob_sha256` and `correction_blob_sha256`.

## 6. why a CORRECTION file and not an appended section: the operator's route, 2026-10-12

§12 says an amendment before the seal is a dated entry appended under a §15 heading and never an edit of the text
above it, and that after the seal there are no amendments, only a new PREREG. The lab's documents carry two readings
of "the seal". In the PREREG's own words the SEALS row is the seal: its opening ("Seal: the sha256 of the **git
blob** of this file", "recorded in that table in the commit after this one. Anchor: at the operator's discretion
(§13)"), §12 ("sealed as a git blob per `papers/checksum/SEALS_2026_09_13.md` in the next commit; anchored at the
operator's discretion") and §13 decision 5 ("commit and seal; anchor"), with the rule section of
`SEALS_2026_09_13.md` ("A seal is the sha256 of the **git blob** of the frozen file"). Read that way, row 7
(`689ee88c`) is the seal and the section 15 appended at `98072bed` came after it. The section 15 sealed in row 8
read the seal as the earliest confirmed memo, by the earliest-memo rule of the same file; no anchor of row 7 is
recorded (`papers/charon/anchors.jsonl` does not exist in this tree), so in that sense the entry preceded the seal.

On 2026-10-12 the operator chose neither appended section but the lab's precedent for a SEALS row not yet anchored
and a run not started: a separate CORRECTION file, sealed as its own row, with the PREREG not edited
(`papers/checksum/CORRECTION_prereg_deploy_quant_H1_2026_09_13.md`, row 3b;
`papers/checksum/CORRECTION_prereg_beacon_draw_2026_09_14.md`, row 6b). Under that route the question does not
arise: the PREREG carries row 7's bytes and nothing appended to them, and this file amends no text of it; it
corrects three sentences in prose and fixes readings beside a frozen document, as rows 3b and 6b do. Row 8 is not to
be anchored. The RESULT records the operator's choice and cites this file.

Decision 2 (three seeds per held-out permutation) is not reopened. A change after rows 7 and 7b are anchored, or
after any leaf exists, is a new PREREG under a new name.

## 7. night 3 and `--accept-schedule`

At the 18.0-minute refusal threshold night 3's 27 runs project 8.1 h, as the §8 table prints, which is over the
8.0-hour window; night 3 therefore refuses without `--accept-schedule` whenever the smoke projects more than 17.78
minutes per 4000-step run (8.0 × 60 / 27). Nights 1 and 2 are 24 run-equivalents each (a 12,000-step run counts
three), 7.2 h at 18.0 minutes; their window refusal would need over 20 minutes per run, where the per-run threshold
has already refused. The operator should expect the prompt on night 3 near the threshold; no threshold moves.

## what this does not change

The architecture, the data, the arms, the seeds, the permutations and their derivation, the counts, the metrics, the
gates block, the bars, the outcome table, the thresholds and the schedule stand as frozen at `3598e59d`. The
PREREG's seal is unchanged. This file is sealed on its own (7b), and nights 2 and 3 do not start before both 7 and 7b
read ANCHORED, as the 2026-09-14 correction was sealed beside its PREREG.
