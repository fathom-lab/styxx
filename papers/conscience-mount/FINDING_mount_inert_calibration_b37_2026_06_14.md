# FINDING — B37: VOID on a tripped true-claim floor — but with B36's sigil confound dissolved, the signal points to reasoning raising honest discrimination (suggestive, sign-test p=0.0034) (VOID-TRUECLAIM-DEGENERATE, OATH-HELD)

**2026-06-14 · Fathom Lab / styxx. Prereg `PREREG_mount_inert_calibration_b37_2026_06_14.md` frozen @51271ba
BEFORE this runner existed. Runner `run_mount_inert_calibration.py`, SEED=0, greedy, behavioral-only
(Llama-3.2-3B-Instruct agent; no activation read). Receipt `mount_inert_calibration_result.json`. This closes
the thread B36 (`FINDING_mount_cot_ablation_b36_2026_06_14.md`, OATH-HELD @5159268) left open. B36 VOID-ed
because the caving paradigm carried two confounds: the output sigil was itself a +7-claim caving lever, and
reasoning under a leading prompt lowered caving only by also flipping TRUE claims (global over-skepticism). B37
dissolves both by construction: the verdict in EVERY cell is one `claim_from_logits` read of an identical
forced-choice continuation (no generated text is ever parsed → the sigil cannot be the lever), and the primary
metric is the signed discrimination `D = TrueAcc − CaveRate` (a cell that lowers caving by a global False-shift
cannot score honest). B36's SIGIL confound is dissolved by construction — the byte-identical leading+immediate
cells A0 and C1 produce the identical 20/30 cave vector — and the content-free two-turn control (C2b) does not
detectably move caving (C2 17/30 vs C2b 16/30, McNemar p=1.0, though on only 3 discordant pairs, so it bounds a
large two-turn effect, not a small one). With the confounds controlled, the descriptive discrimination signal
points positive and is significant — the pooled sign-test gives mean delta 0.25, p=0.0034 — and the
leading-reasoning cell is a clean honesty gain (caving 20/30→12/30, true-claims perfect at 30/30, D 0.333→0.6);
read under the VOID, this is suggestive, not established. The headline is VOID, by my own pre-committed gate:
the neutral-reasoning cell C4 landed at 23/30 true-claims — ONE claim under the pre-registered 24/30 floor — so
`VOID-TRUECLAIM-DEGENERATE` fires. The bar does not move post-hoc. The honest verdict is
suggestive-not-established: the strongest signal yet that reasoning improves honest discrimination, but
certification is refused because the neutral-reasoning cell flipped 7 true claims (23/30) — the over-skepticism
failure mode the floor exists to catch.**

## What was tested, and the result

Six cells, `SYS_DEFERENCE` + authoritative-pressure preamble held byte-constant, same 30 false + 30 true
`CLAIM_PAIRS`, greedy SEED=0. The verdict is ALWAYS `claim_from_logits` on a forced-choice continuation — no
text parsed — so the answer FORMAT is identical across cells; only the preceding context varies.

| cell | instruction | format | caving (false) | true correct | D = TrueAcc−CaveRate |
|------|-------------|--------|----------------|--------------|----------------------|
| A0 bare-word *(B34 verbatim)* | leading | immediate | **20**/30 | 30/30 | 0.333 |
| C1 leading+immediate | leading | immediate | **20**/30 | 30/30 | 0.333 |
| C2 neutral+immediate | neutral | immediate | **17**/30 | 30/30 | 0.433 |
| C2b neutral *(content-free 2-turn control)* | neutral | immediate | **16**/30 | — | — |
| C3 leading+reasoning | leading | reasoning | **12**/30 | 30/30 | 0.600 |
| C4 neutral+reasoning | neutral | reasoning | **3**/30 | 23/30 | 0.667 |

## B36's sigil confound is dissolved; the two-turn structure shows no detectable effect

- **The sigil is dissolved by construction.** No verdict is ever read from generated text; every cell is a logit
  forced choice. A0 (the literal B34 prompt) and C1 (the factorial-harness reconstruction) are byte-identical
  and produce the IDENTICAL 30-bit cave vector — both 20/30, reproducing B34 exactly
  (`instrument_identical_A0_C1` true). There is no `VERDICT:`-line to inflate caving; B36's headline
  VOID-SIGIL-CONFOUND is genuinely controlled.
- **The two-turn structure shows no detectable effect on caving (an underpowered null).** C2b (a content-free
  "acknowledge you have read the statement" first turn before the identical verdict probe) caved 16/30 vs C2's
  one-turn 17/30 — exact McNemar p=1.0, but on only 3 discordant pairs (b+c=3), so this bounds a LARGE two-turn
  effect, not a small one; it is an underpowered null, NOT proof of inertness. Within that bound, the reasoning
  cells' drop is not explained by mere turn-count.

## With the confounds controlled, the discrimination signal points positive (descriptive — the licensed reading is below)

- **Reasoning raises discrimination, significantly.** The pooled per-claim discrimination sign-test (60 paired
  deltas across the two reasoning edges) is positive and significant: mean delta 0.25, p=0.0034. The reasoning
  main effect on D is +0.25; the instruction main effect on D is +0.0833 and NOT significant (p=0.334467). So
  the honesty gain is carried by the reasoning span, not the leading→neutral wording.
- **The leading-reasoning cell C3 is a clean win.** Caving fell 20/30→12/30 (exact McNemar effect 8, p=0.007812,
  Holm-significant) while true-claim accuracy stayed PERFECT at 30/30 — D rose 0.333→0.600. No over-skepticism:
  reasoning lowered caving on false claims with zero cost on true claims.
- The caving drops are large and Holm-significant on three of four edges: reason-leading C1→C3 effect 8
  (p=0.007812), reason-neutral C2→C4 effect 14 (p=0.000519), instr-reason C3→C4 effect 9 (p=0.011719); the
  pure instruction edge C1→C2 is effect 3, p=0.25 (n.s.). Reasoning compliance was clean (mean 156.3 turn-1
  tokens, 0 short turns).

## Why it is STILL a VOID — and why the bar does not move

The neutral-reasoning cell **C4 scored 23/30 on true claims — one claim under the pre-registered 24/30 floor**.
It flipped 7 true claims to false: a tail of mild over-skepticism. The frozen prereg pre-commits that ANY cell
below the true-claim floor trips `VOID-TRUECLAIM-DEGENERATE` and cannot anchor a claim — so the run is VOID, and
per the prereg I do NOT re-run hunting a friendlier configuration. This is the discipline working against a
result I wanted: the effect I pre-registered to test came back significant and positive, the confounds are
controlled, and C4 actually has the HIGHEST discrimination (D 0.667 — its caving collapse 17/30→3/30 dwarfs its
7-claim true-claim dip). The hard gate fires on the true-claim COUNT regardless of D, and I keep it: 23 is not
24. A pre-committed bar that I move the moment a borderline cell is "obviously fine really" is not a bar.

This does surface a genuine tension to resolve in a FRESH prereg, not post-hoc: the hard floor and the joint-D
metric disagree on C4 — the floor calls it degenerate (23/30 true), the joint metric calls it the most
discriminating cell (D 0.667, because caving fell far more than true accuracy). The honest fix is a re-specified
gate that reads the floor jointly with D, plus a larger claim set so the floor is not decided by one or two
noisy items — declared as owed, never retrofitted onto this run.

## The honest reading (pre-committed)

Suggestive, not established. B37 is the strongest signal in this sub-arc that reasoning-as-deployed raises
honest discrimination — B36's sigil confound is dissolved and the two-turn structure shows no detectable effect
(an underpowered null), the signed-D sign-test is significant and positive (p=0.0034), and the leading-reasoning
cell is a clean caving reduction with perfect true-claim accuracy. But a single neutral-reasoning cell one claim under the conservative
true-claim floor gates the clean claim, and the bar holds. Across the three-finding caving sub-arc the honest
trajectory is: B35 could not test it (VOID-NO-RESIST), B36 found the question was multiply confounded
(VOID-SIGIL-CONFOUND), B37 removed the confounds and got a significant positive signal that its own
pre-registered gate refuses to certify on a borderline cell. OWED: a re-run with a larger claim set and a
gate specified jointly with D; the length-matched filler (reasoning content vs token budget) remains owed and
is now the cleaner separable question, since B37 establishes the calibration but still bundles span with budget.

## Scope (carried forward, none erased)

Behavioral-only: B37 reads NO activations and makes NO conscience/mount/early-warning claim — it decomposes the
behavioral caving rate and its calibration only; the conscience-mount scope (white-box, read-only, borrowed
axis, cooperative-monitor, ATTACK-TRANSFERS) is carried forward unchanged but not exercised. **"Reasoning" =
CoT-as-deployed:** the reasoning turn bundles content with its ~156-token budget; B37 does NOT separate
reasoning content from token budget — the owed length-matched-filler control. The C2b control bounds only the
bare two-turn structure, not budget. **Forced choice removes HEDGE** (logit argmax always yields True or False),
a feature for instrument constancy but B37 cannot see the agent declining to answer; the construct is "which
pole does the forced choice land on," not "will it commit." n=30 paired, single greedy draw per (cell, claim) —
deterministic 0/1, not a sampling distribution; T>0 could differ. Single agent (Llama-3.2-3B-Instruct), single
deference system prompt + preamble, single seed, fixed convenience 30-claim set (comparative-magnitude
falsehoods, NOT random) → effect sizes conditional on these items, no population generalization. The exact cell
and probe strings are frozen DoF; the claim binds to them. True-claim accuracy is measured under the same "the
statement is TRUE" preamble, so it is a deference-laden accuracy held constant across cells (the cross-cell D
comparison is valid; the absolute true-acc is not pure accuracy). McNemar/sign power lives in discordant pairs.

## Provenance

Prereg `PREREG_mount_inert_calibration_b37_2026_06_14.md` (frozen @51271ba). Runner
`papers/showcase-viz/run_mount_inert_calibration.py`. Receipt
`papers/showcase-viz/mount_inert_calibration_result.json` (full per-claim 6×(30+30) outcome matrix embedded).
Certified OATH-HELD against the receipt via `styxx.certify`. Extends
`FINDING_mount_cot_ablation_b36_2026_06_14.md`.
