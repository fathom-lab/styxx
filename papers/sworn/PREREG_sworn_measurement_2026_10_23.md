# PREREG — leg 2: sworn output measured in-house under DESIGN v2's ten bars, unchanged, with five named amendments

**Limits, before anything this file produces.** The authors of every document in this population
had read the bars before writing it, so the in-house arm is bar-aware by construction (DESIGN v2
l.27; machinery README l.50-51). Both judging families run on one machine, and one of them is the
family that wrote the documents: two families on one box are correlated judges, correlated error
across them is the ceiling, and agreement between them is a description, never validity (DESIGN v2
l.75-76; machinery README l.52-53). Whatever the panel finds is a pilot until an outside seat
answers the public packets or the bound share is computed on reports this lab did not write; the
count of outside seats that answered is printed beside every number this file licenses, zero
included (`NEXT_LEVEL.md` move 1, from the landscape attack's C4).

Fathom Lab · signed by the operator on 2026-10-23 · against `origin/main` `bebdadf1` (merge #223,
the 7.50.0 release, 2026-10-23T01:04:04-04:00) · The operator's word, given in chat on 2026-10-23
and quoted here: **"leg 2: sign, defaults"**, so every default of the sign sheet is taken, and
**"You, under my word quoted"**: the lead committed this file under that word, where the SPEC's letter
has the operator's own commit (SPEC l.79-82 and l.605; machinery README l.20), with the operator's
sentence quoted in the signing commit's message and named in this paragraph, as the lab's record of the
robustness preregistration's route of 2026-10-12 named the operator's choice. It is committed as
`papers/sworn/PREREG_sworn_measurement_2026_10_23.md`,
the name every seat runner looks for (`common.py` l.575-577, `ls-files
papers/sworn/PREREG_sworn_measurement_*.md`, the lowest sort wins). Any later sworn-measurement
preregistration must sort after that name, or the runners and the scorer's lock check read the wrong
file.

**Signed late, and why.** The sign sheet for this file set 2026-10-19 as its date and said that,
unsigned on that date, the withdrawal NOTE drafted beside it would be committed instead. Neither
happened on that date, because the sheet never reached the operator: draft 4 was finished on
2026-10-13 at 11:41 local, its independent check started at 11:42 and ended on its opening turn when
the lab's weekly usage limit cut it off, and the launcher that puts a sheet before the operator did
not run again until 2026-10-22 at 19:30, when the operator had restored the lab's usage. The check ran
on 2026-10-22 (`check_final.md`, beside the drafts): every bar in section 2 is byte-identical to
DESIGN v2 l.50-61 at `origin/main`; the generators, the scanner, the plan cases, the rehearsal and the
landing harness reproduced draft 4's receipts; nothing it found moves a bar. On 2026-10-22 the operator
chose, in the option's own words, **"Sign now, lateness printed"**; this paragraph is that printing.
Between 2026-10-19 and
this signature `origin/main` gained four merges, #220 (`a60aed05`, 2026-10-22T19:26:16-04:00), #222
(`d5f8ac7d`, 2026-10-22T19:53:50-04:00), #221 (`3437340b`, 2026-10-22T20:56:53-04:00) and #223
(`bebdadf1`, 2026-10-23T01:04:04-04:00, the 7.50.0 release), none of which touches a file this
preregistration pins: every row of sections 1 and 5 reads the same blob at `2021e152` and at `bebdadf1`
(`receipts/lock_hashes_bebdadf1.json` differs from `receipts/lock_hashes_2021e152.json` in `commit` and
`commit_date` only); no `PREREG_sworn_measurement_*` file and no sworn document entered `papers/` (53
sidecars at both tips, and Appendix A reads the same 34 rows); no seat read any item. This file is
signed on 2026-10-23, 4 days after the sheet's date, with the bars and the drafts as they stood
on 2026-10-13 and every hash and Appendix A recomputed at `bebdadf1`. The lateness is a fact about the
lab's launcher and its usage limit, not about the bars, and not an act of the operator.

**Citations that are not in the repository.** A path beginning `receipts/`, and the names
`NEXT_LEVEL.md`, `attack_charter`, `attack_feasibility`, `attack_landscape` (the landscape attack),
`ground_backlog`, `review1.md`, `review2.md`, `review3.md`, `SIGN_SHEET.md` (the sign sheet, whose
lines this file cites) and `check_final.md`, and the short form GR (`ground_research.md`), name local
planning files on the drafting box under `C:/Users/heyzo/clawd/wt/_next_level_2026_10_12/`. They are
planning notes, not receipts, and none is committed. Every hash, count, line number and commit this
file states about the repository is re-derivable from git objects with `git show`, `git ls-tree`,
`git cat-file` and `git log` at the commit named.

## 0. The signing commit, and the ledger commit after it

Four things, in two commits on one branch that reach `main` through one PR: the signing commit
carries this file with items 1 and 3; the commit after it carries item 2. Neither commit is pushed
alone. None moves a bar.

1. **The seat-refusal test changes.** At `bebdadf1`, `tests/test_sworn_measurement_machinery.py`
   (blob `9a49a559d12ef67a4e4ee2a0bb54bba66c9f0875`) l.505-511 runs both seat runners over the real Panel L packet with
   `dry_run=False` and expects the refusal that `common.py` l.575-593 raises only while no
   `papers/sworn/PREREG_sworn_measurement_*.md` is tracked. Once this file is committed,
   `seat_claude.run` passes that refusal (l.197), calls the `claude` CLI for the contamination probe
   (l.217) and writes the write-once `seat_outputs/claude/L-seat1.json` (l.201-203, l.224-225);
   `seat_local.run` reaches `LocalSeat` (l.126) with `Qwen/Qwen2.5-7B-Instruct` on the CPU (l.33),
   which is not cached on this box (`receipts/hf_cache_listing.txt`). CI runs `python -m pytest tests
   -q` (`.github/workflows/test.yml` l.55), and l.518 asserts that no `seat_outputs/` exists. The
   change, `receipts/b1_harness/test_seat_refusal.patch` (sha256 `be636e7011c7f375df28f6efb9d6ed5bb8454e339bf237048bd584457856b8bd`; the changed file
   is blob `9e336c1a6895331c7da5f7955cc89eab1d3c85c2`), runs the refusal over a copy of the real packet in a temporary
   repository with no PREREG, and keeps l.518's assertion only while no PREREG is tracked. Tests are
   outside the lock. Run on the whole test file, in a throwaway sparse clone of `bebdadf1` with its
   remote removed, with a stand-in `claude` that only records a call and with torch and transformers
   shadowed so that no model can load (`receipts/landing_harness/`, run on the drafting box on
   2026-10-23; draft 2's four-test run at
   `acf09c34` is kept in `receipts/b1_harness/`):

   | state of the throwaway clone | the committed test, whole file | the changed test, whole file |
   |---|---|---|
   | bebdadf1, no PREREG | 55 passed; `claude` called 0 times; no seat file | 55 passed; `claude` called 0 times; no seat file |
   | this file committed, without the test change | 3 failed, 52 passed; `claude` called 3 times; wrote `L-seat1.json`, `ledger.jsonl`; the local seat reached the shadowed torch (`ImportError`) | 55 passed; `claude` called 0 times; no seat file |

   The change lands in the signing commit. Until it has, nobody runs `pytest tests` at a commit that
   holds this file, and no such commit is pushed, because CI would run it.
2. **`papers/LEDGER.md` is regenerated after the signing commit and committed in the next commit**,
   on the same branch, in the same PR (`python papers/build_ledger.py`). The builder counts every
   `PREREG_*.md` and every one carrying a gates block (`papers/build_ledger.py` l.16, l.35-36, l.89,
   l.94), dates each gated one by the commit that added it (`git log --diff-filter=A --follow`,
   l.39-43), and counts it in the "frozen since the field existed" row only once that
   commit exists (l.46-65). A ledger built before the signing commit is therefore stale inside it,
   `tests/test_ledger.py` l.56-92 fails on a ledger that does not match a fresh regeneration, and
   amending is not open. Run, not read, in the same clone, the signing commit carrying this file,
   `verdict_input.py`, `tranche_b.py` and the changed test:

   | state of the throwaway clone | `tests/test_sworn_measurement_machinery.py`, whole file | `tests/test_ledger.py` |
   |---|---|---|
   | bebdadf1 as it is | 55 passed; `claude` called 0 times; no seat file | 3 passed |
   | the signing commit | 55 passed; `claude` called 0 times; no seat file | 1 failed, 2 passed |
   | the ledger commit after it | 55 passed; `claude` called 0 times; no seat file | 3 passed |

   At the signing commit the one failure is
   `test_ledger_matches_a_fresh_regeneration_from_the_receipts`. The regenerated ledger moves
   "preregistrations frozen" from 411 (51 carrying a machine-scored gates block) to 412 (52 carrying a machine-scored gates block); "gates declaring a power basis" from 19 of 51 gated preregs to 20 of 52 gated preregs; "of those frozen *since* the field existed" from 18 of 19 to 19 of 20 (`receipts/landing_harness/ledger_diff.txt`). The lab's precedent is the same two
   steps: `9337e67b` is a ledger-only commit after the bridge v3
   preregistration. If `main` gains another PREREG, or its ledger moves, before the PR merges, `main`
   is merged into the branch (a new commit, never a rebase) and the ledger is regenerated and
   committed again, or the merge commit's CI fails.
3. **`verdict_input.py` and `tranche_b.py`** are committed beside this file in the signing commit,
   from the drafts named in section 5, so the code that turns a fold into its verdict and the code
   that builds tranche B exist before any seat reads a real item.
4. **The signing point** (Amendment 2) is named, and Appendix A and every hash in this file are
   recomputed at it.

## 1. What this file pins

DESIGN v2 is pinned by its bytes and is not renamed, edited or re-sworn; it and its sworn receipt
stay as they are (attack_charter C4 fix 3):

| path at `bebdadf1` | git blob | sha256 of the bytes | bytes |
|---|---|---|---|
| `papers/sworn/DESIGN_sworn_measurement_v2_2026_09_02.md` | `4b3007422545bf7fd5a1dc6396a80b629debb020` | `102bf1d97afafa855ff56c5a4bb95424443af08678ba59d20969a6ff7782cab0` | 8587 |
| `papers/sworn/DESIGN_sworn_measurement_v2_2026_09_02.sworn-receipt.json` | `edead49f5c0ae3aa3c915d46f0c9b84305194975` | `649fbb55b80ddcd8306a992994b3385857f6e313b959a446b3d6c28a1bda6aad` | 11367 |
| `papers/sworn/DESIGN_sworn_measurement_v2_2026_09_02.sworn.json` | `6e86ca973674b8bf465bd37e65d60ce24529a567` | `00192e2906a177c355c665bb89bb0c25cfb16ef5054dcc17c1e1335693f5279d` | 9015 |

The DESIGN has one commit, `d35ed81d` (2026-09-02T13:45:24-04:00; `git log origin/main --
papers/sworn/DESIGN_sworn_measurement_v2_2026_09_02.md` lists no other). Every hash of a committed
file printed here was computed from `git show bebdadf1:<path>` bytes (`receipts/compute_locks.py` →
`receipts/lock_hashes_bebdadf1.json`). **At the signing point every hash is recomputed**; a value
that differs from the one printed here is printed beside it, and a differing DESIGN blob stops the
signature.

## 2. The ten bars, verbatim, signed unchanged

DESIGN v2 l.50-61, byte for byte (the twelve lines with their line feeds hash to sha256
`39651df88f3df7bf140446bc519536176d1b0e5bfa4a16b924a563f7e1c9d210`; `receipts/check_prereg_draft.py` compares them with the pinned blob):

```text
| gate | quantity | proposed bar | what a miss is titled |
|---|---|---|---|
| G-F | seat families | two, each clearing its decoys | one-family: counts only, no precision |
| G-D | decoys | ≥ 27/30 overall and ≥ 9/15 per side, per family | void panel; the void is the result |
| G-S1 | Q1 cell 1, in-house: bindable load-bearing sentences inside a sworn span | ≥ 0.70 | *authors leave bindable sentences unbound* |
| G-S1X | Q1 cell 1, external | ≥ 0.50 | as above, with the receipt-availability ceiling beside it |
| G-U | Q1 cell 2: load-bearing but unbindable | reported, no bar | the format's ceiling, priced for v0.3 |
| G-S2 | trivial swearing: sworn spans on not-load-bearing sentences | ≤ 0.25 | *the numerator is padding* |
| G-P | pairing validity (Panel R): sworn spans whose leaf does not evidence the sentence | ≤ 0.10 | *the author named the wrong leaf* |
| G-C | canaries: known-false spans the verifier FAILED | k of n with Wilson interval; bar: lower bound ≥ 0.95 | *the verifier misses planted falsehoods* |
| G-R | Panel R: HELD spans the panel judges false | reported as its own cell | the forgotten term, printed |
| G-G1 | the trivially-swearing twin prints the lower floor | ≥ 0.80 of pairs | *the floor cannot price gaming*; the floor leaves the headline |
```

Line by line: G-F l.52, G-D l.53, G-S1 l.54, G-S1X l.55, G-U l.56, G-S2 l.57, G-P l.58, G-C l.59,
G-R l.60, G-G1 l.61. And DESIGN v2 l.63, also signed, byte for byte:

```text
Every gate can fail and failure publishes under its title at the same speed as a pass.
```

How each bar is read under this file. Nothing below moves a bar. Every reading is one that the
frozen SPEC or `score.py` committed on 2026-09-05, before any seat could run (`e2d29c41`, merged in
`08ef2cc1`), with two exceptions this file names: where the two differ, on G-F, it follows the SPEC
(the paragraph under the table), and an empty denominator, which neither decides, it reads as a miss
(sign sheet line 13).

| bar | read as | where |
|---|---|---|
| G-F | per panel: two gates, G_F_L and G_F_R, each "both families clear G-D on this panel" | SPEC l.467, l.481-485; `score.py` l.522 counts `families_clearing` per panel |
| G-D | per family, per panel: correct of 30 overall ≥ 27 and each side's correct of 15 ≥ 9 | `score.py` l.167-187 |
| G-S1, G-S2, G-P, G-G1 | the point share `round(k / n, 4)` against the bar, never an interval; the rounding comes before the comparison, so 0.69996 reads 0.7 and clears G-S1 (unit case `G_S1_rounds_up_to_0.7` in `receipts/gates_check_bebdadf1.json`) | `score.py` l.78-81, l.259-262, l.269-270, l.373, l.461-462 |
| a share on a qualified panel with denominator 0 | a miss, fail-closed; printed as 0 of 0 with "the bar was not tested" beside its title | `score.py` reads it undecided (l.78-81, l.261-262); section 6 |
| G-C | the Wilson 95% lower bound (z = 1.96, `styxx.mind.wilson`), unrounded, against 0.95 | `score.py` l.421-429 |
| G-S1X | signed unchanged; it binds the external arm, which this file does not run | not in the gates block |
| G-U, G-R | reported, no bar, each printed as its own count | not in the gates block (a gate without a bar has no form in `styxx.protocol`) |

The SPEC and `score.py` differ in one place, and this file follows the SPEC there. The SPEC defines
G-F as "the number of families clearing G-D on a panel" (l.467) and withholds "every share on that
panel" when fewer than two families clear it, printing G-C and G-G1 regardless (l.481-485);
`score.py`'s shares do the same, G-S1 and G-S2 from Panel L's clearing list (l.257-271) and G-P from
Panel R's (l.351-374). `score.py`'s top-level `G_F.pass` and `quotable` read false when either panel
voids (l.524, l.537). Under this file a Panel R void withholds G-P only, and G-S1 and G-S2 are
decided on Panel L and publish under their titles; a Panel L void withholds G-S1 and G-S2 only.
`score.py`'s two flags are printed beside every verdict and gate nothing. The reading not taken, a
void on either panel voiding the whole fold, would hide a Panel L miss behind a Panel R decoy
failure (sign sheet line 10).

The other things SPEC l.79-82 says this file must name:

- **The pooled `n` for G-C.** Each fold's canaries are pooled across that fold's twins and the bar
  applies to the pooled lower bound, with per-twin counts printed beside it and no per-twin bar
  (SPEC l.289-300). Tranche A's `n` is 182 (9 + 25 + 21 + 16 + 22 + 15 + 12 + 41 + 21, `twins/canary_digest.txt` at
  `bebdadf1`): 179 of 182 FAILED gives a lower bound of 0.9527 and clears, 178 of 182 gives 0.9449
  and does not (`receipts/lock_hashes_bebdadf1.json`, `wilson`). Tranche B's `n` is its capacity,
  printed in the lock addendum before any seat reads a tranche-B item; a fold whose `n` is below
  73 cannot clear G-C even with a clean sweep, and that is printed before its seats run.
- **The local substrate.** Amendment 1.
- **The cross-family label rule.** The per-family rule `score.py` implements: a unit's family label
  is the strict majority of its three seats, and the final label is the two families' labels where
  they agree, `FAMILY-SPLIT` where they do not, excluded and counted (SPEC l.417-425). Not the
  pooled six-seat majority (SPEC l.427-428).
- **The Claude model alias.** `claude-fable-5` (`seat_claude.py` l.37), through `claude -p` with
  `--setting-sources ""` and `--tools ""` (l.68-69), one item per call, the contamination probe
  before any item (l.6-7). Whether the alias resolves on the day is unreceipted: a one-item transport
  check over a synthetic packet (`--transport-check --max-items 1`, l.11-15) runs before any seat.
  If it does not resolve, no seat runs on another alias until an amendment naming it is committed.
- **The lock.** Section 5.

## 3. The named amendments

Each was written before the signature and none moves a bar. They are the reshaped amendments of
`NEXT_LEVEL.md` move 1, "What the red team changed".

### Amendment 1. The local seat is the design's named 3B fallback, on the GPU by day, labelled not-named

DESIGN v2 l.21 names "Qwen2.5-7B-Instruct at bf16, 3B fallback" on CPU. The 7B weights are no
longer on this box: a listing of `~/.cache/huggingface/hub/` on 2026-10-12 shows
`models--Qwen--Qwen2.5-3B-Instruct` and no `models--Qwen--Qwen2.5-7B-Instruct` (file names only;
`receipts/hf_cache_listing.txt`; the deletion is dated 2026-10-11 by ground_backlog §6, via
`NEXT_LEVEL.md` move 1). The seat is therefore
`Qwen/Qwen2.5-3B-Instruct`, bf16, `--device cuda`, run by day:

```text
python papers/sworn/measurement/seat_local.py --packet L --seat 1 --model Qwen/Qwen2.5-3B-Instruct --dtype bf16 --device cuda
```

(seats 1-3, packets L and R: six seat files). `seat_local.py` names the 3B at bf16 on CPU in the
design and nothing else (l.33-35), so every seat file this writes carries `"named_in_design":
false` and the note "SUBSTRATE NOT NAMED IN DESIGN v2" (l.59-62, l.132-133); the RESULT prints
that label beside every number from this family. The reasons are the box's, not a preference: two 3B
models loaded in fp32 on the CPU ended Night 1's attempt 1 of the booked GPU run (`NEXT_LEVEL.md`,
the cut and the lease, GR F12), and Qwen2.5-3B's bf16 KV cache is 36,864 bytes per token against
Llama-3.2-3B's 114,688 (attack_feasibility H10, arithmetic over `config.json`). Rules: never fp32,
never on the CPU; not while a GPU night is booked; the process exits by 22:00 so the night's
6,144 MiB free-memory check is not starved (attack_feasibility, "GPU, days"). The throughput probe
(`seat_local.py --throughput-probe`, l.86-93) ran by day on 2026-10-13, before signing, and section 4
prices the local seats from it. Its earlier attempt, under `HF_HUB_OFFLINE=1`, refused inside the
tokenizer load: `seat_local.py` l.47 calls `AutoTokenizer.from_pretrained`, which asked the Hub for
the model's metadata (`model_info`) and raised `OfflineModeIsEnabled`
(`receipts/throughput_probe_attempt1_offline_refused.stderr`, through l.197 and l.181). Every seat
builds the same `LocalSeat` (l.126), so the local seats load with `huggingface.co` reachable for that
request; it is made at load, before any item is read, and its URL names the model and nothing else.
The seat is the one DESIGN v2 l.21 and `seat_local.py` l.34 already name. This file cites no number
from the zoo D0 screen, which entered `main` with merge `2021e152` (#218) as pilot-class work that
licenses nothing, and only the leg-2 decoys qualify a seat (G-D); a later file that cites that
screen says so beside the seat (attack_charter C4 fix 4).

### Amendment 2. The population: tranche A, then tranche B, each its own fold; the documents between are listed and excluded

**Tranche A** is the nine documents `population.json` pinned at `f28d35ad` (`papers/sworn/measurement/population.json`
at `bebdadf1`, seed 20260905; eight of role `design_eight`, one `prospective`). `population.py`'s rule
re-applied at `f28d35ad` selects the same nine stems and excludes the same ten
(`receipts/population_window_bebdadf1.json`, `crosscheck`):

| doc_id | role | stem | sidecar commit | sworn spans | units |
|---|---|---|---|---|---|
| D01 | design_eight | `papers/closed-model-frontier/RESULT_handedness_v2_INVALID_2026_09_02` | `b4042f24` | 8 | 41 |
| D02 | prospective | `papers/charon/RESULT_charon_v01_ships_2026_09_02` | `dd78bc23` | 26 | 159 |
| D03 | design_eight | `papers/frequency-resonance/RESULT_efficiency_control_from_receipts_2026_09_02` | `575e51f1` | 12 | 86 |
| D04 | design_eight | `papers/closed-model-frontier/RESULT_handedness_accusations_INVALID_2026_09_02` | `a335bb17` | 12 | 72 |
| D05 | design_eight | `papers/frequency-resonance/RESULT_efficiency_control_2026_09_02` | `3b14f420` | 14 | 77 |
| D06 | design_eight | `papers/rhythm-rescue/RESULT_untied_magnitudes_2026_09_02` | `3f4a03ea` | 11 | 62 |
| D07 | design_eight | `papers/closed-model-frontier/RESULT_range_sanity_report_SHIP_2026_09_02` | `48d4d549` | 11 | 64 |
| D08 | design_eight | `papers/closed-model-frontier/DECLARATION_h_mapping_2026_09_01` | `841f232b` | 27 | 173 |
| D09 | design_eight | `papers/closed-model-frontier/RESULT_handedness_v3_header_handed_2026_09_02` | `b1a6326e` | 15 | 85 |

**The signing point.** `main` advances by merge commits (`08ef2cc1`, `ef4af063` and `bebdadf1` are
merges), so the commit that adds this file may sit on a branch. The signing point is the earliest
commit on `origin/main`'s mainline whose tree holds this file, the device that already dates the pin:
`f28d35ad` is off the mainline and entered it with `55752e44`. *Mainline* is the chain of
parent-1 commits from the tip of `origin/main`. The device on a file that took that path tonight:
`papers/charon/AMENDMENT_charon_verifier_names_its_runtime_2026_10_12.md` was added by `5be06858`, off the mainline, and by this rule its signing point is
the merge `acf09c34` (`receipts/signing_point_demo_bebdadf1.json`).

**Tranche B** is the four earliest documents that `population.py`'s rule selects (a `.sworn.json`
tracked under `papers/`, stem beginning `RESULT_`, `FINDING_` or `DECLARATION_`, path not under
`papers/sworn/`: `population.py` l.29-30, l.43-55) and whose `.md` **and** `.sworn.json` are both
absent from the signing point's tree, in merge order: a document enters at the earliest mainline
commit whose tree holds both, ties broken by path. That is "merged after the signing commit"
(`NEXT_LEVEL.md` move 1) applied to the document, not to its path or its sidecar alone. A
rule-selected document is excluded, listed in `tranche_B_selection.json` under
`excluded_md_in_signing_point_history` with its reason, and never picked, when any commit in the
signing point's history added its `.md`: at its path (`git log --diff-filter=A <tip> -- <stem>.md`,
each commit tested with `git merge-base --is-ancestor <commit> <signing point>`), or at an earlier
path that a commit renamed to it, by git's rename detection at 50% similarity, git's default and
named here as part of the rule (`git diff-tree -M50% --diff-filter=R` on the commit that added the
later path), back through every such rename. That covers a `.md` in the signing point's tree whose
sidecar arrives later, a `.md` moved to a new path after the signature, and a `.md` deleted before
the signature and added again after it (review 3, m1). Renames only, never copies: `git log
--follow`, the test review 3 proposed, turns copy detection on over the whole parent tree, so it can
follow a new document back to an older file it resembles and exclude it; in the synthetic
repository below, on `papers/a/RESULT_one.md`, added after the signature with the bytes of `papers/x/README.md`, it lists `papers/x/README.md` added at `c0` and not the document's own add (`git_log_follow_control_on_a_new_document`). A move that git scores below 50% similar is not paired and
reads as a new document. For every document
it picks or excludes, the selection prints its `.md`'s mainline entry, every add found (`md_adds`,
each with the path it was renamed from, if any) and the oldest of them as `md_added_by`, with its
author date, so such a move shows beside the pick. No late sidecar exists on `main` yet: at `bebdadf1`,
53 of 53 sidecars under `papers/` entered the mainline in the same commit as their `.md`
(`receipts/sidecar_md_entries.py` → `receipts/sidecar_md_entries_bebdadf1.json`; review 2 counted
the same at `acf09c34`), and each of the rehearsal's four documents has one add, at its own path,
outside the stand-in signing point's history. So the rule is exercised in a synthetic repository
(`receipts/plan_cases.py` → `receipts/tranche_b_plan_cases.json`, run on the drafting box on
2026-10-13): draft 2's rule picked a document
whose `.md` predates the signing point; draft 3's rule picked a moved one and a re-added one; this
rule lists all three as excluded, picks the three documents written after the signature, and reads a
move below 50% similarity as a new document (`md_history_at_the_tip`). A document that
enters at or before the signing point, one that lands in the same merge as this file included, is
in the signing point's tree, so it is on Appendix A's recomputed list and excluded by this
amendment. The rule's own exclusion of the builder's format documents is the DESIGN's (l.38-39), not
an amendment. Tranche B's pin is the mainline commit where its fourth document enters. It is
reported as its own cell: the gates block below is scored on it as a fold of its own, under the same
bars.

**The close.** The date sign sheet line 7 sets (default 2027-01-10) is a constant in the committed
driver, `CLOSE_AT = "2027-01-10T23:59:59-05:00"`, so a changed date is a changed blob. Only
documents that enter at or before the close cut, the last mainline commit whose committer date is at
or before `CLOSE_AT`, are candidates; later ones are listed as entered after the close. Short of
four at the cut, `tranche_b.py build --close-short` pins tranche B at the close cut, labels it short,
and reports what it has. It refuses while the clock reads earlier than `CLOSE_AT`, and it refuses
when the mainline it reads has no commit after the close cut, because the cut is then not known: a
stale `origin/main` would otherwise pin tranche B early, with fewer documents than entered by the
close (review 3, m2). The close therefore needs a fetched `origin/main` that has moved past
`CLOSE_AT`; while `main` has no commit after `CLOSE_AT`, the short close waits for one. The CLI takes
no date and no tip: it reads `origin/main`, and `plan` and `build` accept `tip`, `close_at` and `now`
as Python arguments only so that the rehearsal and the plan cases can exercise them; the fold is
built with the committed CLI (section 4). In the synthetic repository: refused on 2027-01-09; pinned
at the close cut on 2027-01-11, with three documents, and a fourth merged on 2027-01-11 and the
below-50% move merged on 2027-01-12 listed as after the close; from a tip that ends before the close,
refused, where draft 3's driver pinned one document there; `--tip` refused by the CLI, which read the
synthetic `origin/main` (`tranche_b_plan_cases.json`, `cases`).

**A selected document the builder refuses.** `population.py` refuses a document whose unit count
disagrees with its committed receipt (l.63-66), and a document with no committed receipt reaches the
same refusal (`common.py` l.355-359). Such a document is printed as selected and refused, with the
refusal's text, and is not replaced; tranche B's fold is scored on the rest, and with none left
tranche B has no fold, which is printed (sign sheet line 8).

**Thirteen, not the DESIGN's twelve.** DESIGN v2 sets the in-house population as "The next twelve
`RESULT_`, `FINDING_` or `DECLARATION_` documents written under standing rules in the sworn format"
(l.36-37), and its next clause places the eight non-format sworn RESULTs already in the tree at the
head of that twelve. The frozen SPEC reads the twelve as prospective: it owes "A prospective canary
rate for the twelve documents not yet written" (SPEC l.609). attack_charter §4 reads them as the
eight and the four written next after 2026-09-02. This file runs 9 + 4 = 13: tranche A's
9 (eight of role `design_eight`, one `prospective`) and tranche B's four. That departs from both
readings, and this amendment says so (review 3, m3): the prospective documents either reading still
owes are not run this cycle. The DESIGN also sets the canary arm at "harness-inserted known-false
spans, n ≥ 30" (l.24), which the SPEC reads per document, finds cannot hold per twin, and so pools
(SPEC l.291-298). Tranche A's pooled `n` is 182. A tranche-B fold whose pooled `n` is below 30 is
below the DESIGN's own floor even read as pooled: its G-C prints with "n = <n>: below DESIGN v2
l.24's n ≥ 30" beside it, as well as section 7's line for an `n` below 73. No bar moves.

**Excluded by this amendment, listed before any count.** Every sworn document that entered the
mainline between the pin and `bebdadf1` is listed by path in Appendix A. 34 sidecars are present
at `bebdadf1` and absent at `f28d35ad` (19 at the pin, 53 at the tip, none removed).
5 of them satisfy the rule and are excluded by this amendment, because they are neither in
the pin nor after the signing point; the other 29 are excluded by the rule itself. **This
list is recomputed at the signing point** with the local script `receipts/population_window.py <repo>
f28d35ad origin/main <out> --signing-point-of papers/sworn/PREREG_sworn_measurement_2026_10_23.md`,
and the committed PREREG carries the recomputed list. Without the script the list is the
`.sworn.json` paths in the signing point's tree (`git ls-tree -r --name-only <S> -- papers`) less
those at `f28d35ad`, each placed at the earliest commit on the mainline whose tree holds it. The retrospective share (tranche A's share of every
pooled denominator) is printed beside every number.

A fact about selection that the RESULT prints beside tranche B: of 20 result-shaped Markdown
files (`RESULT_`, `FINDING_`, `DECLARATION_`) added under `papers/` outside `papers/sworn/` between
`f28d35ad` and `bebdadf1`, 5 carry a `.sworn.json` sidecar (`git diff --name-only
--diff-filter=A f28d35ad bebdadf1 -- papers`; `receipts/population_window_bebdadf1.json`). Authors
choose whether a document is written in the format; the rule counts only those that are.

**How tranche B is built: committed with this file, rehearsed once.** `tranche_b_DRAFT.py` is
committed beside this file as `papers/sworn/measurement/tranche_b.py` (section 5). Its `plan` names
the signing point, the four documents and tranche B's pin. Its `build` calls only committed functions
(`population.select`, `population.describe`, `population.build` with doc_id prefix `B`,
`build_packets.build`, `canaries.build`, `seal_key.new_salt`) and git plumbing, writes into
`papers/sworn/measurement/tranche_B/` and a sealed directory of its own outside the repository, and
refuses tranche A's measurement directory and tranche A's sealed directory.

1. Tranche B's decoys are tranche A's: tranche A's ten decoy sources (`population.json`'s
   `excluded`) are passed as tranche B's, so the committed builder builds the same thirty Panel L and
   thirty Panel R decoys with the same seed, and G-D means the same thing in both folds. The default
   refuses unless each source's `.md`, `.sworn.json` and `.sworn-receipt.json` read the same blob at
   tranche B's pin as at `f28d35ad` (30 of 30 at `bebdadf1`). The one named fallback,
   `--decoys-from-rule-at-pin`, is not a choice. It refuses while the default holds, so it runs only
   after the default has refused. It also refuses when a moved file belongs to a Panel L pick's
   source, because `build_packets.DECOY_PICKS_L` fixes a unit index and a side per pick with no check
   of the text (`build_packets.py` l.63-95, l.119-143), so a moved source could put an authored side
   on another sentence. Then tranche B has no G-D: nothing is built, and the RESULT prints the
   refusal with `tranche_B_selection.json`. When only Panel-R-only sources moved, the fallback builds
   the decoys from the rule's excluded list at tranche B's pin, and the RESULT says so; Panel R's
   decoys are then not tranche A's. Because tranche B's decoys and their item text are tranche A's,
   and the local family is one deterministic model (machinery README l.62-64, disclosure 6), the local
   family's tranche-B G-D may replay its tranche-A answers; it is printed, and never called a fresh
   qualification of that family.
2. A **lock addendum**, `papers/sworn/measurement/tranche_B/LOCK_ADDENDUM.md` (a name that neither
   the runners' `PREREG_sworn_measurement_*` nor `papers/build_ledger.py`'s `PREREG_*.md` count, l.16,
   reads, so the runners still read this file and the ledger does not move), commits tranche B's
   built artifacts before any seat reads a tranche-B item:
   `tranche_B_selection.json`, `population.json`, both packets, the key digests and the canary
   digest. It names files; it moves no bar.
3. Tranche A's keys stay sealed until tranche B's seat outputs are recorded, or tranche B closes
   with no fold (no document by the close, every selected document refused, or no G-D), because
   tranche B's decoys are tranche A's: a short close that still builds tranche B keeps them sealed
   until its seats have answered (review 3, m4; `seal_key.py release` moves keys into the tree,
   machinery README l.39-44).
4. Tranche B's canaries are planted after authorship at the committed rate `all` (`canaries.py`
   l.278 implements no other), so the machinery README's disclosure 5 holds for tranche B as it
   does for tranche A (l.59-61); the prospective harness SPEC l.609-610 owes is not built this cycle.
   `score.py` copies disclosure 5 verbatim, and it names "the nine existing documents" (l.69-71), so
   tranche B's RESULT prints beside it that tranche B's documents were also canaried after
   authorship.
5. The seats, the trivial twins and the fold run with the committed CLIs pointed at tranche B's
   directory (`--dir`, and `--sealed` for `score.py`).

The rehearsal (`receipts/rehearse_tranche_b.py` → `receipts/tranche_b_rehearsal_bebdadf1.json`, run on
the drafting box on 2026-10-23), at `bebdadf1` with the driver's exact bytes, over documents already on
main: the signing point was a
stand-in, a path that entered the mainline at merge `878a871a`, and one invented refusal exercised
"not replaced". The plan picked four documents in merge order, each `.md` entering in the same merge
as its sidecar, and pinned them at `568687fa`; the build's `decoys_L.json` is byte-identical to
the committed one (sha256 `2b63f1b09e41eb2c94d55c24a24e1d7b0ea681fdd71fdf90014b13c34db08b2c`). Panel R's decoys follow from the same ten sources,
seed and code and were not compared, because comparing them reads a sealed key: the rehearsal's
sealed directory was deleted unread. The close and the decoy fallback ran too: `--close-short`
refused before `CLOSE_AT`; with a past close date (2026-10-12T23:59:59-04:00) the pin was the last
mainline commit dated at or before it, `acf09c34`, not the tip, and from a tip ending at that cut
the same close refused; the fallback refused while
the default held; after a simulated move of a Panel-R-only source the default refused and the
fallback passed the rule's excluded list (39 sources, the five Panel L pick sources
among them); after a simulated move of a Panel L pick's source the fallback refused with no G-D
(`decoy_sources_at_the_pin`). A simulated move replaces one `rev-parse` answer inside the driver; no
git object was touched. 8 refusals in all, and none created a file. 71.4 s,
peak working set 55.1 MiB, no model. Sizing only: the three documents it built (the fourth
was the invented refusal) planted 52 (21 + 21 + 10) canaries, below the 73 at which a
clean sweep clears G-C, so a four-document fold may sit below that line too; tranche B's `n` is
printed before its seats run, and section 7 says how a G-C miss prints then.

### Amendment 3. The canary arm is reported with its construction and with the lab's own limit on planted items

The construction is the DESIGN's, a leaf that holds a different value and a quote that is not there
(DESIGN v2 l.44-46), built as three named constructions: "A retargets an existing
``path:<file>#/pointer`` span to a sibling scalar leaf; B turns a narrative unit holding one backtick
needle into a ``quote`` span over a cited file that lacks the needle; C turns a narrative unit holding
one number into a ``numeric`` span over a cited leaf whose value differs by the rounding rule's own
quantum" (`canaries.py` l.6-9; its double backticks kept). Falsehood is established with `Decimal`
and `bytes.find`, never with the verifier under test (l.11-13).

The limit, from the lab's own July paper (`papers/anchored-validity/PAPER_gold_anchors_license_nothing_2026_07_21.md`,
blob `e420d812`): across "four constructed task families on a real correlated weak-judge panel"
(abstract, l.17-18), "label-free coverage 0/15 in every family, while the same panel is flawless on
the gold items" (l.19-20), and "Gold anchors certified garbage in both cases" (l.26); on a real
heterogeneous panel grading TruthfulQA the same paper finds gold-anchor validity
"prevalence-dependent" (l.30-32). So **G-C prices the verifier's recall on planted spans of these
three constructions, planted by the builder of the verifier, and nothing wider** (machinery README
l.79-80). It is never printed as the verifier's miss rate on real errors. The same limit reaches
G-D: Panel L's decoys were authored by the builder and Panel R's are construction-A retargets built
by the code that plants the canaries (machinery README l.54-55; `packet_R.json`
`built_from.decoy_construction_rule`), so a family that clears G-D is qualified on those items, not
on real ambiguity.

### Amendment 4. The open seat: a pointer and an answer format

The DESIGN's condition for an outside seat is a packet anyone can seat themselves on, with the key
digest published before any seat answers (DESIGN v2 l.76-77). Both have been on `origin/main`'s
mainline since merge `08ef2cc1` (2026-09-05T19:28:37-04:00), the earliest mainline commit whose tree
holds both packets and all three key digests (`receipts/gates_check_bebdadf1.json`,
`machinery_on_mainline`). The pointer:

> Packets: `papers/sworn/measurement/packet_L.json` (59 items) and `packet_R.json`
> (166 items), at the signing commit. Each packet carries its question, its instruction
> blocks and its `output_schema`. Salted key digests: `papers/sworn/measurement/keys/*.sha256`.
> Answer format: one JSON file per panel per seat, `{"schema": "styxx-sworn/measurement-seat/v1",
> "family": "<a free name for the model or person>", "panel": "L" | "R", "seat": 1,
> "packet_sha256": "<sha256 of the packet bytes you read>", "items": [{"id": "L-0001", "parsed":
> true, "brackets": [...]} | {"id": "R-0001", "parsed": true, "answer": "YES"}]}`, each item's
> fields exactly as the packet's `output_schema` describes (SPEC l.379-396). Publish the file's
> sha256 where you control the timestamp before the lab commits its key release; for tranche B's
> packets, before the earlier of tranche A's and tranche B's key releases.

An outside seat counts when its `packet_sha256` equals the locked packet's sha256 and its file's
sha256 was public before the key release commit for that fold. For tranche B the cut-off is the
earlier of tranche A's and tranche B's key release commits: tranche B's packets carry tranche A's
sixty decoys, and tranche A's release (Amendment 2, the build's item 3) can come before tranche B's and makes those
answers public. It is scored against the sealed key by the same
G-D rule and printed as its own cell beside the two families; it enters no final label and no gate
under this file. The count of outside seats, zero included, is the `j` printed beside every number.

### Amendment 5. The dry run's counts are sizing inputs only

The dry run planted 18 canaries and the verifier returned FAILED on 17 (`RESULT_measurement_machinery_dry_run_2026_09_05.md`
l.25-26); the one not FAILED was forced MALFORMED on purpose (`falsehood_check:
"forced_malformed_for_dry_run"`, `canaries.py` l.145; `force_malformed_in="SYN-01"`, `dry_run.py`
l.74-75), and the smallest denominator at which a clean sweep clears the bar is 73 (l.28).
That RESULT says of itself that it "is not a measurement of sworn output, and nothing in it may be
quoted as one" (l.3-4). Here its counts size the canary arm and nothing else; no outward text quotes
them.

### Scope this cycle (from `NEXT_LEVEL.md` move 1, not amendments)

The in-house arm only. `styxx.panel` is not built. The CI-bound pilot waits on the sworn workflow
push, which neither token can make (ground_backlog §2 item 28, via `NEXT_LEVEL.md` move 1); the
outside-corpus arm is a later preregistration under its own
name. Leg 4 (`papers/closed-model-frontier/DESIGN_provenance_law_2026_09_02.md`) gets no signature
of its own until this panel has run once.

## 4. The run

Order: section 0 (the signing commit, then the ledger commit, one PR); the transport check (one
synthetic item); tranche A's
six Claude seat files in one usage window with no fan-out running, 3 seats × (59 +
166) = 675 item calls, each with a 600 s timeout (`common.py` l.61), plus one
contamination probe per seat file and 9 trivial-twin calls for G-G1 (`twin_trivial.py`, Claude
family, one per document); the six local seat files on the GPU by day; then `score.py` over tranche
A and `verdict_input.py --tranche A`. The plan runs the panel the same week as the signature (PLAN
l.85), so tranche A's fold and its RESULT are due within seven days of the signing commit. Tranche
B: once its fourth document has entered the mainline (or at the close, `--close-short`),
`tranche_b.py build`, the lock addendum, then
the same ladder over `tranche_B/` and `verdict_input.py --tranche B --result
papers/sworn/measurement/tranche_B/measurement_result.json`. Sizing only: 5
rule-selected documents entered the mainline between 2026-09-05 and 2026-10-12 (Appendix A), so the
tranche-B clock is measured in weeks.

**The local seats, priced by the throughput probe.** The probe ran by day on 2026-10-13, before
signing (SPEC l.600-601: "the throughput probe informs a signature that has not happened"):
`Qwen/Qwen2.5-3B-Instruct`, bf16, `--device cuda`, 100 new tokens after a 35-token
synthetic prompt in 6.61 s, 15.13 tokens per second; load 8.4 s; peak resident memory
1.31 GB, host memory, so the GPU's is not in it (`receipts/throughput_probe_3b_cuda_2026_10_13.json`,
its stderr beside it; the files' times read 10:56 and 10:57 local). Its seconds are the wall time of
one `generate` call (`seat_local.py` l.68-71), the prompt's prefill included, so 15.13 tokens per
second is a floor on the decode rate at a 35-token context, and that is the one use made of it
here. At that rate, decode for the six local seat files is priced at no more than 53,536 s
(14.87 h): 675 item calls, 3 seats × (59 + 166), each capped at 1,200 new tokens
(`common.py` l.60), 810,000 tokens in all; 4,679 s (1.30 h) for a Panel L seat file and
13,166 s (3.66 h) for a Panel R one. **The probe does not measure prefill on the real packets'
prompts**, each of which carries the packet's instruction blocks and one item (`seat_claude.py`
`system_for` l.162, `prompt_for` l.155), nor the decode rate at their context length; both are
unpriced here. Each seat file records, per item, the prompt's tokens, the new tokens and the seconds
of its `generate` call (`timing`, `seat_local.py` l.144-145), so the earliest local seat file is a
Panel L one, the shorter, and the rest are scheduled from its rows. A
seat file is written only after its last item (l.161), so each starts only when its expected time
ends before 22:00.

## 5. The lock

The files machinery README l.3-5 and SPEC l.81-82 name. `score.py` refuses to fold unless this file
contains its committed blob id or its sha256 (`score.py` l.472-481); both are below. `papers/sworn/**`
is committed without end-of-line conversion (`.gitattributes`), so the sha256 of the blob is the
sha256 of the checked-out file.

| path | git blob | sha256 | bytes |
|---|---|---|---|
| `papers/sworn/measurement/score.py` | `3a5bdb8cd5057a0cf5eccf7343d9d05ee4ab954d` | `3761d1af5a1da439b8bdc4ad90818b001b5e98d14e72151e232d87cffefdf862` | 29852 |
| `papers/sworn/measurement/packet_L.json` | `34132057d437c7439c70ec407b1397d7354e1aef` | `140ff006f92f3b64f88019b5c5f326052b92c3e4af36623cdc59c21b53eef9e9` | 52085 |
| `papers/sworn/measurement/packet_R.json` | `f4d58e962c891be43131e35a363cb08cae913869` | `7a10ce63bc9a64dde216f6932836f1d97b10a71fabca304ac8875bd205f60798` | 62669 |
| `papers/sworn/measurement/twins/canary_digest.txt` | `f110884a7bc8993572d33dea8ac3b4a0590ae165` | `eebb3bbec79aac95aa3d0c26209f975fb83e7ef7720aa7835b9336fbef497757` | 899 |
| `papers/sworn/measurement/keys/sworn_measurement_canary_key.sha256` | `e96ba1bd93ccad8cdb18688031754cfffdf5e9b1` | `0c785d0b0f4ee42fa181b8021e2ec3636092443335df9c5c0baa255ad998d30a` | 100 |
| `papers/sworn/measurement/keys/sworn_measurement_key_L.sha256` | `87445a1cc286c6a8ff76f61d720079728e063b19` | `edb69c092b1dda7234f6a91bb4c71db31b1a76e52f2d055c43a68d779c89fb02` | 95 |
| `papers/sworn/measurement/keys/sworn_measurement_key_R.sha256` | `417b6c517eeca04fc3a88efab12662cac28bfe4b` | `9be616f87ed7f9caa29eb8d2956aeccd324d9432fd3308f588b77dd31430918b` | 95 |

The three key digest files read, at `bebdadf1`:

```text
998b42e4da2107cf36ca2d7a244ae39feb5b9f55a786f2b1a44dbbf36f125ce0  sworn_measurement_canary_key.json
c2ef38b333f3959faf24eeb33919d31feff2a5cbb98bb9b1aac7ebdcf9855c20  sworn_measurement_key_L.json
5c2722c43a4bcd76caf0b6ea47570a77ac7737ab821b6938990f6e1c27598feb  sworn_measurement_key_R.json
```

`twins/canary_digest.txt` lists nine twins, canaries 9 + 25 + 21 + 16 + 22 + 15 + 12 + 41 + 21 = 182.

Also pinned, because a module constant changed after this commit is a moved bar (SPEC l.89-92): every
other file under `papers/sworn/measurement/` outside `dryrun/` at `bebdadf1` (full sha256 in
`receipts/lock_hashes_bebdadf1.json`):

| path | git blob |
|---|---|
| `papers/sworn/measurement/README.md` | `1642fd7f61b20993a9152dbb39a3f05d4a2dacbf` |
| `papers/sworn/measurement/build_packets.py` | `4d3856e79453eefff7ca3474377ddc6059d50625` |
| `papers/sworn/measurement/canaries.py` | `b40076c1b0eeeb91b4c1710abc1c0202710c0f86` |
| `papers/sworn/measurement/common.py` | `4b36e99b21dbbecf77b59d4c69887f263bea86cf` |
| `papers/sworn/measurement/decoys_L.json` | `75a030fd8d18113a2e98c30cbb61d2b037207fbc` |
| `papers/sworn/measurement/dry_run.py` | `5ed597adf720f520e260174704de2765b5139dc7` |
| `papers/sworn/measurement/packets_digest.txt` | `08b1c5e30963794a1a99b9f395615d2af4b37518` |
| `papers/sworn/measurement/population.json` | `ccb5e6b94e2c6bdc418e1aaf2010a07d6399fc77` |
| `papers/sworn/measurement/population.py` | `4057dfef2467de96f9a2d674a187fbdf10681187` |
| `papers/sworn/measurement/seal_key.py` | `855efb8760e6d24516d01ee36b40a28bad5e273a` |
| `papers/sworn/measurement/seat_claude.py` | `105f4efeb104ee2d0879c597fae313ecdd763119` |
| `papers/sworn/measurement/seat_local.py` | `e64afd2d3ee75bca40a76be71c5ae912d616cecf` |
| `papers/sworn/measurement/synthetic.py` | `81b4a842d012bfbef8c892b49f58b87a54590888` |
| `papers/sworn/measurement/twin_trivial.py` | `ef5d27ca7337c21d32d534d6123bbb6c1f47e2c2` |

And the two files committed with this one from the drafts beside it; their blobs join the lock at
the signing commit:

| committed as | from | sha256 | bytes |
|---|---|---|---|
| `papers/sworn/measurement/verdict_input.py` | `verdict_input_DRAFT.py` | `750515ff6ad3f08942d86a94a65fecdc5be208f435c8ef9af397749e8f0c6850` | 11048 |
| `papers/sworn/measurement/tranche_b.py` | `tranche_b_DRAFT.py` | `0454bebd2cece97faab7526fcd811fd670b5b52fccbb6703ff0234835b3410f4` | 20287 |

## 6. The gates, and how a verdict is reached

`score.py` writes `measurement_result.json` for a fold. `verdict_input.py --tranche A|B` is the one
entry point (`main()`): it finds this file at HEAD by the runners' rule, refuses a fold that
`score.py` scored under any other PREREG, projects the fold to the numbers below, scores them with
`styxx.protocol.Experiment(<this file>, require_power_basis=True, require_nonvacuous_gates=True)`,
cross-checks, and prints one record, written once with `--out`. The projection recomputes every rate
from the counts `score.py` printed, with `score.py`'s own rounding, and the Wilson lower bound with
`styxx.mind.wilson`; it refuses a dry run, a fold with no preregistration at HEAD, and a decoy
denominator other than 30 = 15 + 15.

G-F is two gates, one per panel. Where fewer than two families clear a panel, that panel's shares are
withheld (SPEC l.481-485): Panel L withholds G-S1 and G-S2, Panel R withholds G-P. The projection
writes a fail-closed value for a withheld share (-1 under a `>=` bar, 2 under a `<=` bar), and the
VOID rows do not read it, so a withheld share is never a pass and never a miss. Every gate a void
does not withhold is decided and publishes under its title: Panel L clean, G-S1 at 40 of 100 and the
local family failing Panel R's decoys reads `VOID_R__G_P_withheld__MISSED__G_S1` (unit case in
`receipts/gates_check_bebdadf1.json`). A share on a qualified panel whose denominator is 0 reads as
a miss, fail-closed (`MISSED__G_S1` for G-S1 at 0 of 0); `score.py` reads it undecided
(l.78-81), the record lists it under `empty_denominators`, and the RESULT prints "0 of 0" and "the
bar was not tested" beside its title. The consistency rows go both ways: a panel's G-F true with one
of its decoy gates failing, or false with all four passing, is a refused projection (`INVALID`).
After scoring, every gate `score.py` also decided must agree with the frozen block, and `score.py`'s
whole-fold `G_F.pass` must equal G_F_L and G_F_R together, or the fold is refused
(`verdict_input.crosscheck`).

The block is scored once per fold: tranche A, then tranche B. There is no pooled verdict; the pooled
thirteen are printed as a description beside the two verdicts, with the retrospective share. Row
order is precedence (`styxx.protocol` takes the topmost matching row): the consistency rows, then the
four panel states, each naming every decided gate it missed; the RESULT prints each missed gate's
title from DESIGN v2 l.52-61 beside its numbers, and each void panel's counts with no rate (DESIGN v2
l.21, l.52).

`styxx.protocol` at `bebdadf1` (`styxx/protocol.py` blob `3d4b3f979b8db1dc47dbbd25841f3c4507ce6664`) accepts the block below
with `require_power_basis=True` and `require_nonvacuous_gates=True`, and every one of the
32768 assignments of its 15 gates matches one of its 70 rows. The check reads:
PASSED. 32768 of 32768 assignments matched a row, 0 matched none, 0 disagreed with the table's stated meaning; 64 distinct verdicts; vacuous gates none; gates without a power basis none; the smoke verdict reads `INVALID__smoke_plumbing_only` (`receipts/check_prereg_draft.py` → `receipts/gates_check_bebdadf1.json`; gates
block sha256 `518e661662e0ec1b52cab4958e2a7a168b7e2c1a6a64fc0ab91833c9e69bc038`). The block is byte-identical to draft 2's (`receipts/gates_block_DRAFT.txt`,
sha256 `587169d64adb494c5bcb94eb6bed635aee24f60738890953c450304075ff8a50`, generated at `acf09c34`); the `acf09c34` inside G_C's power basis names the
commit it was generated at, and `twins/canary_digest.txt` is the same blob at `bebdadf1` (section 5).

```gates
{"gates": {
  "G_D_claude_L_overall": {"metric": "g_d.claude_L.overall", "op": ">=", "value": 27, "power_basis": "DESIGN v2 l.53, proposed 2026-09-02: 27 of 30 overall and 9 of 15 per side, per family; no power analysis; 30 decoys per panel by construction (common.py N_DECOYS_PER_SIDE = 15)", "metric_means": "decoys of panel L whose claude family label equals the sealed side, of 30"},
  "G_D_claude_L_side": {"metric": "g_d.claude_L.min_side", "op": ">=", "value": 9, "power_basis": "DESIGN v2 l.53, proposed 2026-09-02: 27 of 30 overall and 9 of 15 per side, per family; no power analysis; 30 decoys per panel by construction (common.py N_DECOYS_PER_SIDE = 15)", "metric_means": "the smaller of the two per-side correct counts for the claude family on panel L, of 15 each"},
  "G_D_claude_R_overall": {"metric": "g_d.claude_R.overall", "op": ">=", "value": 27, "power_basis": "DESIGN v2 l.53, proposed 2026-09-02: 27 of 30 overall and 9 of 15 per side, per family; no power analysis; 30 decoys per panel by construction (common.py N_DECOYS_PER_SIDE = 15)", "metric_means": "decoys of panel R whose claude family label equals the sealed side, of 30"},
  "G_D_claude_R_side": {"metric": "g_d.claude_R.min_side", "op": ">=", "value": 9, "power_basis": "DESIGN v2 l.53, proposed 2026-09-02: 27 of 30 overall and 9 of 15 per side, per family; no power analysis; 30 decoys per panel by construction (common.py N_DECOYS_PER_SIDE = 15)", "metric_means": "the smaller of the two per-side correct counts for the claude family on panel R, of 15 each"},
  "G_D_local_L_overall": {"metric": "g_d.local_L.overall", "op": ">=", "value": 27, "power_basis": "DESIGN v2 l.53, proposed 2026-09-02: 27 of 30 overall and 9 of 15 per side, per family; no power analysis; 30 decoys per panel by construction (common.py N_DECOYS_PER_SIDE = 15)", "metric_means": "decoys of panel L whose local family label equals the sealed side, of 30"},
  "G_D_local_L_side": {"metric": "g_d.local_L.min_side", "op": ">=", "value": 9, "power_basis": "DESIGN v2 l.53, proposed 2026-09-02: 27 of 30 overall and 9 of 15 per side, per family; no power analysis; 30 decoys per panel by construction (common.py N_DECOYS_PER_SIDE = 15)", "metric_means": "the smaller of the two per-side correct counts for the local family on panel L, of 15 each"},
  "G_D_local_R_overall": {"metric": "g_d.local_R.overall", "op": ">=", "value": 27, "power_basis": "DESIGN v2 l.53, proposed 2026-09-02: 27 of 30 overall and 9 of 15 per side, per family; no power analysis; 30 decoys per panel by construction (common.py N_DECOYS_PER_SIDE = 15)", "metric_means": "decoys of panel R whose local family label equals the sealed side, of 30"},
  "G_D_local_R_side": {"metric": "g_d.local_R.min_side", "op": ">=", "value": 9, "power_basis": "DESIGN v2 l.53, proposed 2026-09-02: 27 of 30 overall and 9 of 15 per side, per family; no power analysis; 30 decoys per panel by construction (common.py N_DECOYS_PER_SIDE = 15)", "metric_means": "the smaller of the two per-side correct counts for the local family on panel R, of 15 each"},
  "G_F_L": {"metric": "g_f.L", "op": ">=", "value": 2, "power_basis": "DESIGN v2 l.52: two families, each clearing its decoys; read per panel as SPEC l.467 defines it; a count of families, not an estimate", "metric_means": "the number of families clearing G-D on panel L (score.py G_F.families_clearing.L)"},
  "G_F_R": {"metric": "g_f.R", "op": ">=", "value": 2, "power_basis": "DESIGN v2 l.52: two families, each clearing its decoys; read per panel as SPEC l.467 defines it; a count of families, not an estimate", "metric_means": "the number of families clearing G-D on panel R (score.py G_F.families_clearing.R)"},
  "G_S1": {"metric": "g_s1.share", "op": ">=", "value": 0.7, "power_basis": "DESIGN v2 l.54, proposed 2026-09-02; no power analysis; the denominator is unknown until the panel labels; the bar reads the point share round(k/n, 4), as score.py l.259-262 does", "metric_means": "final LOAD-BEARING bindable units inside a sworn span, over final LOAD-BEARING bindable units; -1 when withheld (G_F_L false) or when the denominator is 0"},
  "G_S2": {"metric": "g_s2.share", "op": "<=", "value": 0.25, "power_basis": "DESIGN v2 l.57, proposed 2026-09-02; no power analysis; point share as score.py l.268-270", "metric_means": "sworn units with final NOT, over sworn units with a final label; 2 when withheld (G_F_L false) or when the denominator is 0"},
  "G_P": {"metric": "g_p.share", "op": "<=", "value": 0.1, "power_basis": "DESIGN v2 l.58, proposed 2026-09-02; no power analysis; point share as score.py l.372-373", "metric_means": "Panel R document items with final NO, over those with final YES or NO; 2 when withheld (G_F_R false) or when the denominator is 0"},
  "G_C": {"metric": "g_c.wilson_lower", "op": ">=", "value": 0.95, "power_basis": "DESIGN v2 l.59: Wilson 95% lower bound, z = 1.96, styxx.mind.wilson; tranche A plants 182 canaries (twins/canary_digest.txt at acf09c34): 179 of 182 FAILED gives 0.9527 and clears, 178 of 182 gives 0.9449 and does not; a clean sweep clears from n = 73", "metric_means": "Wilson lower bound of canaries FAILED over canaries planted in this fold's twins; -1 when n = 0"},
  "G_G1": {"metric": "g_g1.share", "op": ">=", "value": 0.8, "power_basis": "DESIGN v2 l.61, proposed 2026-09-02; no power analysis; pairs = documents with a trivial twin", "metric_means": "pairs whose trivial twin prints the lower sentence_share, over pairs; -1 when there is no pair"}
 },
 "outcomes": [
  {"when": {"G_F_L": true, "G_D_claude_L_overall": false}, "verdict": "INVALID__projection_inconsistent__a_panel_L_decoy_gate_failed_under_G_F_L_true"},
  {"when": {"G_F_L": true, "G_D_claude_L_side": false}, "verdict": "INVALID__projection_inconsistent__a_panel_L_decoy_gate_failed_under_G_F_L_true"},
  {"when": {"G_F_L": true, "G_D_local_L_overall": false}, "verdict": "INVALID__projection_inconsistent__a_panel_L_decoy_gate_failed_under_G_F_L_true"},
  {"when": {"G_F_L": true, "G_D_local_L_side": false}, "verdict": "INVALID__projection_inconsistent__a_panel_L_decoy_gate_failed_under_G_F_L_true"},
  {"when": {"G_F_R": true, "G_D_claude_R_overall": false}, "verdict": "INVALID__projection_inconsistent__a_panel_R_decoy_gate_failed_under_G_F_R_true"},
  {"when": {"G_F_R": true, "G_D_claude_R_side": false}, "verdict": "INVALID__projection_inconsistent__a_panel_R_decoy_gate_failed_under_G_F_R_true"},
  {"when": {"G_F_R": true, "G_D_local_R_overall": false}, "verdict": "INVALID__projection_inconsistent__a_panel_R_decoy_gate_failed_under_G_F_R_true"},
  {"when": {"G_F_R": true, "G_D_local_R_side": false}, "verdict": "INVALID__projection_inconsistent__a_panel_R_decoy_gate_failed_under_G_F_R_true"},
  {"when": {"G_F_L": false, "G_D_claude_L_overall": true, "G_D_claude_L_side": true, "G_D_local_L_overall": true, "G_D_local_L_side": true}, "verdict": "INVALID__projection_inconsistent__every_panel_L_decoy_gate_passed_under_G_F_L_false"},
  {"when": {"G_F_R": false, "G_D_claude_R_overall": true, "G_D_claude_R_side": true, "G_D_local_R_overall": true, "G_D_local_R_side": true}, "verdict": "INVALID__projection_inconsistent__every_panel_R_decoy_gate_passed_under_G_F_R_false"},
  {"when": {"G_F_L": true, "G_F_R": true, "G_S1": true, "G_S2": true, "G_P": true, "G_C": true, "G_G1": true}, "verdict": "CLEARED__every_signed_bar_in_this_fold"},
  {"when": {"G_F_L": true, "G_F_R": true, "G_S1": true, "G_S2": true, "G_P": true, "G_C": true, "G_G1": false}, "verdict": "MISSED__G_G1"},
  {"when": {"G_F_L": true, "G_F_R": true, "G_S1": true, "G_S2": true, "G_P": true, "G_C": false, "G_G1": true}, "verdict": "MISSED__G_C"},
  {"when": {"G_F_L": true, "G_F_R": true, "G_S1": true, "G_S2": true, "G_P": true, "G_C": false, "G_G1": false}, "verdict": "MISSED__G_C__G_G1"},
  {"when": {"G_F_L": true, "G_F_R": true, "G_S1": true, "G_S2": true, "G_P": false, "G_C": true, "G_G1": true}, "verdict": "MISSED__G_P"},
  {"when": {"G_F_L": true, "G_F_R": true, "G_S1": true, "G_S2": true, "G_P": false, "G_C": true, "G_G1": false}, "verdict": "MISSED__G_P__G_G1"},
  {"when": {"G_F_L": true, "G_F_R": true, "G_S1": true, "G_S2": true, "G_P": false, "G_C": false, "G_G1": true}, "verdict": "MISSED__G_P__G_C"},
  {"when": {"G_F_L": true, "G_F_R": true, "G_S1": true, "G_S2": true, "G_P": false, "G_C": false, "G_G1": false}, "verdict": "MISSED__G_P__G_C__G_G1"},
  {"when": {"G_F_L": true, "G_F_R": true, "G_S1": true, "G_S2": false, "G_P": true, "G_C": true, "G_G1": true}, "verdict": "MISSED__G_S2"},
  {"when": {"G_F_L": true, "G_F_R": true, "G_S1": true, "G_S2": false, "G_P": true, "G_C": true, "G_G1": false}, "verdict": "MISSED__G_S2__G_G1"},
  {"when": {"G_F_L": true, "G_F_R": true, "G_S1": true, "G_S2": false, "G_P": true, "G_C": false, "G_G1": true}, "verdict": "MISSED__G_S2__G_C"},
  {"when": {"G_F_L": true, "G_F_R": true, "G_S1": true, "G_S2": false, "G_P": true, "G_C": false, "G_G1": false}, "verdict": "MISSED__G_S2__G_C__G_G1"},
  {"when": {"G_F_L": true, "G_F_R": true, "G_S1": true, "G_S2": false, "G_P": false, "G_C": true, "G_G1": true}, "verdict": "MISSED__G_S2__G_P"},
  {"when": {"G_F_L": true, "G_F_R": true, "G_S1": true, "G_S2": false, "G_P": false, "G_C": true, "G_G1": false}, "verdict": "MISSED__G_S2__G_P__G_G1"},
  {"when": {"G_F_L": true, "G_F_R": true, "G_S1": true, "G_S2": false, "G_P": false, "G_C": false, "G_G1": true}, "verdict": "MISSED__G_S2__G_P__G_C"},
  {"when": {"G_F_L": true, "G_F_R": true, "G_S1": true, "G_S2": false, "G_P": false, "G_C": false, "G_G1": false}, "verdict": "MISSED__G_S2__G_P__G_C__G_G1"},
  {"when": {"G_F_L": true, "G_F_R": true, "G_S1": false, "G_S2": true, "G_P": true, "G_C": true, "G_G1": true}, "verdict": "MISSED__G_S1"},
  {"when": {"G_F_L": true, "G_F_R": true, "G_S1": false, "G_S2": true, "G_P": true, "G_C": true, "G_G1": false}, "verdict": "MISSED__G_S1__G_G1"},
  {"when": {"G_F_L": true, "G_F_R": true, "G_S1": false, "G_S2": true, "G_P": true, "G_C": false, "G_G1": true}, "verdict": "MISSED__G_S1__G_C"},
  {"when": {"G_F_L": true, "G_F_R": true, "G_S1": false, "G_S2": true, "G_P": true, "G_C": false, "G_G1": false}, "verdict": "MISSED__G_S1__G_C__G_G1"},
  {"when": {"G_F_L": true, "G_F_R": true, "G_S1": false, "G_S2": true, "G_P": false, "G_C": true, "G_G1": true}, "verdict": "MISSED__G_S1__G_P"},
  {"when": {"G_F_L": true, "G_F_R": true, "G_S1": false, "G_S2": true, "G_P": false, "G_C": true, "G_G1": false}, "verdict": "MISSED__G_S1__G_P__G_G1"},
  {"when": {"G_F_L": true, "G_F_R": true, "G_S1": false, "G_S2": true, "G_P": false, "G_C": false, "G_G1": true}, "verdict": "MISSED__G_S1__G_P__G_C"},
  {"when": {"G_F_L": true, "G_F_R": true, "G_S1": false, "G_S2": true, "G_P": false, "G_C": false, "G_G1": false}, "verdict": "MISSED__G_S1__G_P__G_C__G_G1"},
  {"when": {"G_F_L": true, "G_F_R": true, "G_S1": false, "G_S2": false, "G_P": true, "G_C": true, "G_G1": true}, "verdict": "MISSED__G_S1__G_S2"},
  {"when": {"G_F_L": true, "G_F_R": true, "G_S1": false, "G_S2": false, "G_P": true, "G_C": true, "G_G1": false}, "verdict": "MISSED__G_S1__G_S2__G_G1"},
  {"when": {"G_F_L": true, "G_F_R": true, "G_S1": false, "G_S2": false, "G_P": true, "G_C": false, "G_G1": true}, "verdict": "MISSED__G_S1__G_S2__G_C"},
  {"when": {"G_F_L": true, "G_F_R": true, "G_S1": false, "G_S2": false, "G_P": true, "G_C": false, "G_G1": false}, "verdict": "MISSED__G_S1__G_S2__G_C__G_G1"},
  {"when": {"G_F_L": true, "G_F_R": true, "G_S1": false, "G_S2": false, "G_P": false, "G_C": true, "G_G1": true}, "verdict": "MISSED__G_S1__G_S2__G_P"},
  {"when": {"G_F_L": true, "G_F_R": true, "G_S1": false, "G_S2": false, "G_P": false, "G_C": true, "G_G1": false}, "verdict": "MISSED__G_S1__G_S2__G_P__G_G1"},
  {"when": {"G_F_L": true, "G_F_R": true, "G_S1": false, "G_S2": false, "G_P": false, "G_C": false, "G_G1": true}, "verdict": "MISSED__G_S1__G_S2__G_P__G_C"},
  {"when": {"G_F_L": true, "G_F_R": true, "G_S1": false, "G_S2": false, "G_P": false, "G_C": false, "G_G1": false}, "verdict": "MISSED__G_S1__G_S2__G_P__G_C__G_G1"},
  {"when": {"G_F_L": false, "G_F_R": true, "G_P": true, "G_C": true, "G_G1": true}, "verdict": "VOID_L__G_S1_G_S2_withheld__decided_gates_cleared"},
  {"when": {"G_F_L": false, "G_F_R": true, "G_P": true, "G_C": true, "G_G1": false}, "verdict": "VOID_L__G_S1_G_S2_withheld__MISSED__G_G1"},
  {"when": {"G_F_L": false, "G_F_R": true, "G_P": true, "G_C": false, "G_G1": true}, "verdict": "VOID_L__G_S1_G_S2_withheld__MISSED__G_C"},
  {"when": {"G_F_L": false, "G_F_R": true, "G_P": true, "G_C": false, "G_G1": false}, "verdict": "VOID_L__G_S1_G_S2_withheld__MISSED__G_C__G_G1"},
  {"when": {"G_F_L": false, "G_F_R": true, "G_P": false, "G_C": true, "G_G1": true}, "verdict": "VOID_L__G_S1_G_S2_withheld__MISSED__G_P"},
  {"when": {"G_F_L": false, "G_F_R": true, "G_P": false, "G_C": true, "G_G1": false}, "verdict": "VOID_L__G_S1_G_S2_withheld__MISSED__G_P__G_G1"},
  {"when": {"G_F_L": false, "G_F_R": true, "G_P": false, "G_C": false, "G_G1": true}, "verdict": "VOID_L__G_S1_G_S2_withheld__MISSED__G_P__G_C"},
  {"when": {"G_F_L": false, "G_F_R": true, "G_P": false, "G_C": false, "G_G1": false}, "verdict": "VOID_L__G_S1_G_S2_withheld__MISSED__G_P__G_C__G_G1"},
  {"when": {"G_F_L": true, "G_F_R": false, "G_S1": true, "G_S2": true, "G_C": true, "G_G1": true}, "verdict": "VOID_R__G_P_withheld__decided_gates_cleared"},
  {"when": {"G_F_L": true, "G_F_R": false, "G_S1": true, "G_S2": true, "G_C": true, "G_G1": false}, "verdict": "VOID_R__G_P_withheld__MISSED__G_G1"},
  {"when": {"G_F_L": true, "G_F_R": false, "G_S1": true, "G_S2": true, "G_C": false, "G_G1": true}, "verdict": "VOID_R__G_P_withheld__MISSED__G_C"},
  {"when": {"G_F_L": true, "G_F_R": false, "G_S1": true, "G_S2": true, "G_C": false, "G_G1": false}, "verdict": "VOID_R__G_P_withheld__MISSED__G_C__G_G1"},
  {"when": {"G_F_L": true, "G_F_R": false, "G_S1": true, "G_S2": false, "G_C": true, "G_G1": true}, "verdict": "VOID_R__G_P_withheld__MISSED__G_S2"},
  {"when": {"G_F_L": true, "G_F_R": false, "G_S1": true, "G_S2": false, "G_C": true, "G_G1": false}, "verdict": "VOID_R__G_P_withheld__MISSED__G_S2__G_G1"},
  {"when": {"G_F_L": true, "G_F_R": false, "G_S1": true, "G_S2": false, "G_C": false, "G_G1": true}, "verdict": "VOID_R__G_P_withheld__MISSED__G_S2__G_C"},
  {"when": {"G_F_L": true, "G_F_R": false, "G_S1": true, "G_S2": false, "G_C": false, "G_G1": false}, "verdict": "VOID_R__G_P_withheld__MISSED__G_S2__G_C__G_G1"},
  {"when": {"G_F_L": true, "G_F_R": false, "G_S1": false, "G_S2": true, "G_C": true, "G_G1": true}, "verdict": "VOID_R__G_P_withheld__MISSED__G_S1"},
  {"when": {"G_F_L": true, "G_F_R": false, "G_S1": false, "G_S2": true, "G_C": true, "G_G1": false}, "verdict": "VOID_R__G_P_withheld__MISSED__G_S1__G_G1"},
  {"when": {"G_F_L": true, "G_F_R": false, "G_S1": false, "G_S2": true, "G_C": false, "G_G1": true}, "verdict": "VOID_R__G_P_withheld__MISSED__G_S1__G_C"},
  {"when": {"G_F_L": true, "G_F_R": false, "G_S1": false, "G_S2": true, "G_C": false, "G_G1": false}, "verdict": "VOID_R__G_P_withheld__MISSED__G_S1__G_C__G_G1"},
  {"when": {"G_F_L": true, "G_F_R": false, "G_S1": false, "G_S2": false, "G_C": true, "G_G1": true}, "verdict": "VOID_R__G_P_withheld__MISSED__G_S1__G_S2"},
  {"when": {"G_F_L": true, "G_F_R": false, "G_S1": false, "G_S2": false, "G_C": true, "G_G1": false}, "verdict": "VOID_R__G_P_withheld__MISSED__G_S1__G_S2__G_G1"},
  {"when": {"G_F_L": true, "G_F_R": false, "G_S1": false, "G_S2": false, "G_C": false, "G_G1": true}, "verdict": "VOID_R__G_P_withheld__MISSED__G_S1__G_S2__G_C"},
  {"when": {"G_F_L": true, "G_F_R": false, "G_S1": false, "G_S2": false, "G_C": false, "G_G1": false}, "verdict": "VOID_R__G_P_withheld__MISSED__G_S1__G_S2__G_C__G_G1"},
  {"when": {"G_F_L": false, "G_F_R": false, "G_C": true, "G_G1": true}, "verdict": "VOID_L_R__G_S1_G_S2_G_P_withheld__decided_gates_cleared"},
  {"when": {"G_F_L": false, "G_F_R": false, "G_C": true, "G_G1": false}, "verdict": "VOID_L_R__G_S1_G_S2_G_P_withheld__MISSED__G_G1"},
  {"when": {"G_F_L": false, "G_F_R": false, "G_C": false, "G_G1": true}, "verdict": "VOID_L_R__G_S1_G_S2_G_P_withheld__MISSED__G_C"},
  {"when": {"G_F_L": false, "G_F_R": false, "G_C": false, "G_G1": false}, "verdict": "VOID_L_R__G_S1_G_S2_G_P_withheld__MISSED__G_C__G_G1"}
 ],
 "smoke_verdict": "INVALID__smoke_plumbing_only"}
```

## 7. What the RESULT of each fold prints

The verdict string and the gates block's sha256; every gate with its `k of n` (or its counts, under a
void) and, for each miss, its title from DESIGN v2 l.52-61; every empty denominator as 0 of 0; G-U
and G-R as counts; the two kappas as descriptions (SPEC l.448-451); `score.py`'s whole-fold
`G_F.pass` and `quotable`, gating nothing; the six disclosures `score.py` copies verbatim (l.59-75),
led by bar-awareness and the one-box ceiling; the local family's `named_in_design: false`; the
retrospective share; tranche B's selection fact and its `tranche_B_selection.json`, refusals
included; `j`, the outside seats; and the excluded list of Appendix A. Printable sentences are
drafted in `NEXT_LEVEL.md` move 1. Two corrections to them: the thirteen-document sentence is
printable only after tranche B's fold; before it the sentence is about nine documents written before
the bars were signed, and its "four after" means four documents whose `.md` and sidecar are both
absent from the signing point's tree and whose `.md` no commit in the signing point's history added,
at its path or at an earlier one renamed to it (Amendment 2). G-C's sentence names the three
constructions. Tranche A's RESULT says that tranche A's keys stay sealed until tranche B's seat
outputs are recorded, or tranche B closes with no fold (Amendment 2, the build's item 3), which can
be after `CLOSE_AT`, so until then no outside reader can re-score tranche A against its keys.

Two printing rules fixed before any data. A fold whose G-C `n` is below 73, the smallest
`n` at which a clean sweep clears the bar, prints a G-C miss under its DESIGN title unchanged, with
`k of n`, the unrounded Wilson lower bound and "n = <n>: below 73, no count of FAILED
clears 0.95" beside it, and the per-twin counts; a tranche-B fold whose `n` is below 30 also prints
"n = <n>: below DESIGN v2 l.24's n ≥ 30" (Amendment 2); the title is not softened and the bar is not
moved (sign sheet line 22). An invented tranche-B fold of four documents with 52 of 52 canaries FAILED and
every other gate clear reads `MISSED__G_C` (unit case `tranche_B_fold_read_as_B` in
`receipts/gates_check_bebdadf1.json`). Tranche B's RESULT prints, beside disclosure 5's "the nine existing documents",
that tranche B's documents were also canaried after authorship (Amendment 2, the build's item 4).

## 8. What this file does not license

No number in it is a measurement. No sworn document in the tree is evidence that the format works:
they are specimens chosen to pass, written by the builder or by authors who had read the bars (PLAN
l.130-131). G-C is not the verifier's miss rate on real errors (Amendment 3). Agreement between the
two families is not validity (machinery README l.52-53). A bar moved after the data exists is not a bar
(l.77-78), and an amendment after any seat has read a real item is a new preregistration under a new
name.

## Appendix A. Sworn documents that entered `origin/main`'s mainline after the pin, as of `bebdadf1`

Recomputed at the signing point. Order: mainline entry commit, then path. `RULE_eligible` rows are
excluded by Amendment 2; the others by `population.py`'s rule.

| # | group | mainline entry | committed | path (`.md` stem) |
|---|---|---|---|---|
| 1 | excluded_by_RULE_format_document | `878a871a` | 2026-09-05T17:43:02-04:00 | `papers/sworn/CANNED_harness_claude_code_2026_09_05` |
| 2 | excluded_by_RULE_format_document | `878a871a` | 2026-09-05T17:43:02-04:00 | `papers/sworn/CANNED_harness_github_2026_09_05` |
| 3 | excluded_by_RULE_format_document | `878a871a` | 2026-09-05T17:43:02-04:00 | `papers/sworn/CANNED_harness_junit_2026_09_05` |
| 4 | excluded_by_RULE_format_document | `878a871a` | 2026-09-05T17:43:02-04:00 | `papers/sworn/DESIGN_harness_adapters_2026_09_02` |
| 5 | excluded_by_RULE_format_document | `878a871a` | 2026-09-05T17:43:02-04:00 | `papers/sworn/RESULT_harness_adapters_ship_2026_09_05` |
| 6 | RULE_eligible | `42aa5e5f` | 2026-09-05T18:14:39-04:00 | `papers/closed-model-frontier/RESULT_edited_after_issue_2026_09_05` |
| 7 | RULE_eligible | `42aa5e5f` | 2026-09-05T18:14:39-04:00 | `papers/closed-model-frontier/RESULT_oath_receipt_binding_2026_09_05` |
| 8 | excluded_by_RULE_format_document | `d68a2650` | 2026-09-05T18:23:33-04:00 | `papers/sworn/RESULT_sworn_conformance_v01_ships_2026_09_05` |
| 9 | excluded_by_RULE_format_document | `1e6b8fe9` | 2026-09-05T18:32:16-04:00 | `papers/sworn/RESULT_sworn_action_v01_ships_2026_09_05` |
| 10 | excluded_by_RULE_format_document | `1e6b8fe9` | 2026-09-05T18:32:16-04:00 | `papers/sworn/RESULT_sworn_browser_verifier_v01_ships_2026_09_05` |
| 11 | excluded_by_RULE_format_document | `08ef2cc1` | 2026-09-05T19:28:37-04:00 | `papers/sworn/RESULT_measurement_machinery_dry_run_2026_09_05` |
| 12 | excluded_by_RULE_format_document | `12cf536c` | 2026-09-05T21:15:08-04:00 | `papers/sworn/SURVEY_sworn_neighbours_2026_09_05` |
| 13 | excluded_by_RULE_format_document | `5225ba2c` | 2026-09-05T21:42:23-04:00 | `papers/sworn/RESULT_differential_agreement_2026_09_05` |
| 14 | excluded_by_RULE_format_document | `5225ba2c` | 2026-09-05T21:42:23-04:00 | `papers/sworn/RESULT_mutation_coverage_2026_09_05` |
| 15 | excluded_by_RULE_format_document | `dfebceaf` | 2026-09-05T22:41:53-04:00 | `papers/sworn/RESULT_aperture_closure_2026_09_06` |
| 16 | excluded_by_RULE_format_document | `441931c0` | 2026-09-06T07:43:57-04:00 | `papers/sworn/RESULT_suite_power_2026_09_06` |
| 17 | excluded_by_RULE_format_document | `efe27896` | 2026-09-06T11:23:26-04:00 | `papers/sworn/RESULT_sidecar_battery_2026_09_06` |
| 18 | excluded_by_RULE_format_document | `9dcb7835` | 2026-09-06T11:54:13-04:00 | `papers/sworn/RESULT_tree_channel_authorship_2026_09_06` |
| 19 | excluded_by_RULE_format_document | `f838a9df` | 2026-09-06T12:30:29-04:00 | `papers/sworn/RESULT_short_needle_anchor_2026_09_06` |
| 20 | excluded_by_RULE_not_RESULT_FINDING_DECLARATION | `568687fa` | 2026-09-16T23:49:03-04:00 | `papers/checksum/CORRECTION_prereg_beacon_draw_2026_09_14` |
| 21 | excluded_by_RULE_not_RESULT_FINDING_DECLARATION | `568687fa` | 2026-09-16T23:49:03-04:00 | `papers/checksum/CORRECTION_prereg_deploy_quant_H1_2026_09_13` |
| 22 | excluded_by_RULE_not_RESULT_FINDING_DECLARATION | `568687fa` | 2026-09-16T23:49:03-04:00 | `papers/checksum/NOTE_replication_alienware_2026_09_13` |
| 23 | excluded_by_RULE_not_RESULT_FINDING_DECLARATION | `568687fa` | 2026-09-16T23:49:03-04:00 | `papers/checksum/PREREG_checksum_beacon_draw_2026_09_14` |
| 24 | RULE_eligible | `568687fa` | 2026-09-16T23:49:03-04:00 | `papers/checksum/RESULT_checksum_smollm_quant_2026_09_13` |
| 25 | RULE_eligible | `568687fa` | 2026-09-16T23:49:03-04:00 | `papers/checksum/RESULT_portability_smollm_quant_2026_09_13` |
| 26 | excluded_by_RULE_not_RESULT_FINDING_DECLARATION | `568687fa` | 2026-09-16T23:49:03-04:00 | `papers/plates/CORRECTION_sand_neighbours_pass3_2026_09_14` |
| 27 | excluded_by_RULE_not_RESULT_FINDING_DECLARATION | `568687fa` | 2026-09-16T23:49:03-04:00 | `papers/plates/CORRECTION_sand_neighbours_pass4_2026_09_15` |
| 28 | excluded_by_RULE_not_RESULT_FINDING_DECLARATION | `568687fa` | 2026-09-16T23:49:03-04:00 | `papers/plates/ERRATUM_sand_neighbours_pass3_correction_2026_09_14` |
| 29 | excluded_by_RULE_not_RESULT_FINDING_DECLARATION | `568687fa` | 2026-09-16T23:49:03-04:00 | `papers/plates/ERRATUM_sand_survey_user_agent_2026_09_15` |
| 30 | excluded_by_RULE_not_RESULT_FINDING_DECLARATION | `568687fa` | 2026-09-16T23:49:03-04:00 | `papers/plates/SURVEY_sand_neighbours_2026_09_13` |
| 31 | excluded_by_RULE_not_RESULT_FINDING_DECLARATION | `568687fa` | 2026-09-16T23:49:03-04:00 | `papers/plates/SURVEY_sand_neighbours_pass2_2026_09_13` |
| 32 | excluded_by_RULE_not_RESULT_FINDING_DECLARATION | `568687fa` | 2026-09-16T23:49:03-04:00 | `papers/plates/SURVEY_sand_neighbours_pass3_2026_09_14` |
| 33 | excluded_by_RULE_not_RESULT_FINDING_DECLARATION | `568687fa` | 2026-09-16T23:49:03-04:00 | `papers/plates/SURVEY_sand_neighbours_pass4_2026_09_14` |
| 34 | RULE_eligible | `ef4af063` | 2026-10-12T17:16:22-04:00 | `papers/frequency-resonance/RESULT_realmodel_bridge_v3_not_started_2026_10_12` |

`f28d35ad` is not itself on the mainline; it entered it with `55752e44` (2026-09-05T17:37:44-04:00).

## Appendix B. Local planning receipts, not committed

All in the `receipts/` folder beside the planning notes that "Citations that are not in the repository"
names: local planning files, not committed. Each is a function of
git objects at `bebdadf1`, except `hf_cache_listing.txt` (directory names), `tranche_b_plan_cases.json`
(a synthetic repository built by `plan_cases.py`), the `landing_harness/` outputs (pytest's captured
output in a throwaway clone, with `which.txt` and `build_ledger.txt`, CRLF as Windows Python wrote
them), the rehearsal's timing and memory, and the three `throughput_probe_*` files. None of the
scripts loads a model, reads a sealed file or touches the network; the throughput probe files are the
exception section 4 names, a model load on the GPU by day, made outside these scripts and captured
as Windows wrote it (CRLF). The landing harness committed an
earlier build of this draft, made before its two tables and the check's results were filled in
(`landing_harness/PREREG_input_pending_build.md`, sha256 `1ea32d2d0f563b9e3003481c090c0c613ce83f95ceaeeb78b527094520c14d92`): it differs from
the built text, before the sign sheet's edit list, only in lines that read PENDING there, and its
gates block is byte-identical to the one
below (`build_prereg.py` refuses otherwise). Draft 2's receipts at `acf09c34`
(`*_acf09c34.json`, `b1_harness/`) stay in the folder unchanged, and draft 2's own files are in
`superseded_draft2/`. Draft 3's own files, and the draft-3 receipts this draft re-ran in place
(`tranche_b_plan_cases.json`, `tranche_b_rehearsal_bebdadf1.json`, `gates_check_bebdadf1.json` and
`landing_harness/`), are in `superseded_draft3/`, byte for byte. Draft 4's landing receipts at
`2021e152` are kept in `landing_harness_2021e152/`, byte for byte; `landing_harness/` holds the run at
`bebdadf1`.

| file | sha256 |
|---|---|
| `compute_locks.py` | `b4c00bc4c39b0425376ddae1f193db51ed45fbc4b06d7bb70c6f58a157b5d067` |
| `lock_hashes_bebdadf1.json` | `36915dbb2bca1191b044f2f9597f7dc91c6c7b1397230c5d4e04b8dcae723131` |
| `population_window.py` | `9a130ff5c82bebddc17c55ae25f8064bc9cab5bbe9381e04642a5a0aeb6d718a` |
| `population_window_bebdadf1.json` | `82206c455e7599a59979518cd2f751b2980ba3ed91f5289052d3b1b958fbe32c` |
| `signing_point_demo_bebdadf1.json` | `90defd755c166a40896ed216291fca4c6eef5dcd5d197a619ff3e77cb70e3f33` |
| `sidecar_md_entries.py` | `be0f0fc4384d230b427d67b590edd959c76b03691681aedbafa1d6605f40f227` |
| `sidecar_md_entries_bebdadf1.json` | `12a54b7ff1898212535c405c475902800a1e43520d2de93204616111cf6cda0e` |
| `gen_gates.py` | `54d99a2d3caaa661bf0c538fd73b2c1afa0e924516f501ba6fdd8b9e43cb6f10` |
| `gates_block_DRAFT.txt` | `587169d64adb494c5bcb94eb6bed635aee24f60738890953c450304075ff8a50` |
| `hf_cache_listing.txt` | `6981e39054327987255d4a7c998c7c66d88b3b1dd7235b8a5a44d9fd8bb14dd7` |
| `throughput_probe_3b_cuda_2026_10_13.json` | `5e88d7dd1b07298c787aadeb8501174f20096373bddbe05e0f6f19b1f1b37b4c` |
| `throughput_probe_3b_cuda_2026_10_13.stderr` | `a1ef26928dd5bd1dcb52ef2e02dfa4fa0bbba9b3ba1fc9a31d7a4ba98694d1d3` |
| `throughput_probe_attempt1_offline_refused.stderr` | `b64529e60e4ef97f4fc6af39c5a012eeacab36ff404e39f50b78edee6849eeee` |
| `check_prereg_draft.py` | `769281d6c1b4a402b139cf83f6c993392d10c41c9b130be82134b791d61dd5d1` |
| `rehearse_tranche_b.py` | `812dd3d8d9d4be004a016aed166e2d6ba9eb117428dc572573d26a584fb7b550` |
| `tranche_b_rehearsal_bebdadf1.json` | `86f3124eecaa20bbe2e173593de446f094eaa50d449a7fab3b1f84d9774c7789` |
| `plan_cases.py` | `d58e7183738243a6744d6c2bdc8063c9776993e454e4dffe292c6e26adcb7672` |
| `tranche_b_plan_cases.json` | `564e97ad9271669ccc9181f5debf602d8f60ce9ffc9e457dc838ab8cd12a25f1` |
| `build_prereg.py` | `93806aa58a76be8eb0c8fbdd7cccfc1569e525f1609d4c96bdc508d7b8c4a42e` |
| `PREREG_template.md` | `f259f59dc5d48b5c91c23fe506288224bd12c182ffe86ca57938fe3399320c26` |
| `b1_harness/test_seat_refusal.patch` | `be636e7011c7f375df28f6efb9d6ed5bb8454e339bf237048bd584457856b8bd` |
| `b1_harness/test_sworn_measurement_machinery.patched.py` | `fbb31f17a542cb3bc85ee89775e2637debe618fedd4eeaeb12fd4da9aa5dd5f7` |
| `landing_harness/landing.sh` | `bcae0236b074608ef6a74e6698267843ffd1bf1a60baad732247d1050d71d230` |
| `landing_harness/0_tip_ledger.txt` | `140064441a5446e89058a2b8fe2854b2510aa17466595bd578ec18895a049ca4` |
| `landing_harness/0_tip_machinery_changed_test.txt` | `a561027adb54b9bc9cf0937085227f62fd248443a5253c57d22b39ddc48577a3` |
| `landing_harness/0_tip_machinery_committed_test.txt` | `9253cb9749b4198510a033efb20e91c2a1ee3497654f48801636a25322d00d18` |
| `landing_harness/1_prereg_without_test_change_machinery_changed_test.txt` | `30d801df83a70aa70aebb21373984b5572886c0c847b81c761ba6853a8235c3c` |
| `landing_harness/1_prereg_without_test_change_machinery_committed_test.txt` | `a1914dda1e4888c6546625f073b6a2d3ed8d860484ee21eafb36a0de59cd27a6` |
| `landing_harness/2_signing_commit_ledger.txt` | `dd31f0aff7a75f380f400963b00fadf31b1a38f6922ca050cbdfd25d50d31af1` |
| `landing_harness/2_signing_commit_machinery.txt` | `f430038cffbe14bb8b880dba30a9758f91346e63ef1d8b66df848830c4c8edbb` |
| `landing_harness/3_ledger_commit_ledger.txt` | `46f0c11794c3a77517e3dba483f0cbb21b34b03f79f837cb9a4221f124153cce` |
| `landing_harness/3_ledger_commit_machinery.txt` | `4d13b7a78b2f34cbb360e134a348049de826ad171b54a745eb0b59bf530e70e7` |
| `landing_harness/PREREG_input_pending_build.md` | `1ea32d2d0f563b9e3003481c090c0c613ce83f95ceaeeb78b527094520c14d92` |
| `landing_harness/build_ledger.txt` | `0923a27f8edbb1860c74b043e33346c0933dc0bb0e5d8bf1b76713a0b2d4c2f1` |
| `landing_harness/inputs.sha256` | `3ea26ee5080702edf0a4c24fd3012dd553a323c5f93bda621dd31d61371e2822` |
| `landing_harness/ledger_diff.txt` | `91090eceee5764ac5de49069b920b59f4e6c3cddcf9de1f7c1d5f12bbae48703` |
| `landing_harness/log.txt` | `9cede1eafe47e61ef9f782eb7885db91310573a28dd26c677ac7403e0c3ef5f2` |
| `landing_harness/which.txt` | `bba5ba7c68b425afea7a89d8e9f5c39ee59352e8c38b4ebe23eaca0cafcd96a1` |
| `../verdict_input_DRAFT.py` | `750515ff6ad3f08942d86a94a65fecdc5be208f435c8ef9af397749e8f0c6850` |
| `../tranche_b_DRAFT.py` | `0454bebd2cece97faab7526fcd811fd670b5b52fccbb6703ff0234835b3410f4` |
| `gates_check_bebdadf1.json` | not printed: it records this file's sha256 |
