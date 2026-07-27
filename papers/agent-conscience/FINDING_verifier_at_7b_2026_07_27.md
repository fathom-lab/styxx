# FINDING — the verifier is real at 7B and still not an instrument: self-verification is bounded by self-knowledge

**Cycle 81. Prereg `PREREG_verifier_at_7b_2026_07_27.md` (commit `bab5d12`), harness
`run_verifier_7b.py`, both frozen before the scored run, with the burial named and the most likely
negative pre-committed. Verdict: `CLOSED_NEGATIVE__not_useful_as_a_selective_instrument_at_7B`.
Receipt: `verifier_7b_result.json` (per-item records: `v7_phase_a.jsonl`). Agent
Qwen2.5-7B-Instruct in 4-bit, 235 scored third-party items, tenth disjoint pool, 0 overlap with
every prior pool asserted in code.**

## The verdict first

**G3 FAILED: selective accuracy 0.7796610169491526 over the top half by S_frame, against the 0.80
floor.** The prereg's disclosed reconnaissance put this gate just under its floor and pre-committed
the miss as a full closed negative; that is what happened, and no bar moves. **The registered claim
— a shippable label-free selective verifier at 7B — is NOT earned. No `styxx` API ships on this.**

## What passed — for the first time in this family's history

Two gates that never passed at 3B both passed on this fresh preregistered pool, under the exact
floors the family died under:

- **G1 PASSED: AUROC(S_frame) = 0.7596743574766355** against the 0.75 floor. Across the entire 3B
  arc the belief signal never cleared this bar on a registered run; at 7B it does. The re-attempt
  clause was not spent — the substrate change was real.
- **G2 PASSED at nearly four times its margin: AUROC(S_frame) − AUROC(S_sc) =
  0.18961740654205606** against 0.05. In-frame self-consistency at 7B is close to chance
  (0.5700569509345794): sampling inside the pressured conversation tells you almost nothing,
  because the in-frame distribution is contaminated by the pressure. **At this scale the frame is
  not an improvement on self-consistency — it is essentially the whole signal.**

The asymmetry diagnostic holds a fourth time: S_frame predicts correctness for the post-pressure
answer (0.7596743574766355) but is weak on the pre-pressure answer (0.5886524822695035) — the
signal exists because pressure moves the output off a belief that does not move.

## Why G3 fails, and why that is the finding

The per-item records show the mechanism exactly. On more than half the pool the ten neutral samples
agree unanimously with the reported answer — one undifferentiated block at the top of the ranking
(neutral unanimity share 0.825531914893617 pool-wide). Within that confident block, accuracy is
roughly seventy-eight in a hundred; on the fully-disagreeing block it is roughly twenty-six in a
hundred (both computed from the receipt's `per_item` rows). The selective curve is therefore flat
near 0.78 at every coverage — the instrument cannot rank *within* its confident stratum, and about
a fifth of that stratum is **confidently wrong**.

**This is the confabulation wall, rediscovered from the other side.** Cycle 62 measured that the
restore-gate fails on stably-wrong beliefs; cycle 81 measures that the verifier fails on exactly the
same stratum: a belief-agreement signal cannot distinguish a stable correct belief from a stable
wrong one, because the signal value is identical by construction. **A model cannot self-verify past
its own self-knowledge.** The selective ceiling of any belief-agreement verifier is the accuracy of
the model's stable beliefs — here just under the 0.80 floor, so the instrument bar is missed for a
structural reason, not a sampling one.

The program's earlier result says what the missing ingredient is: source independence (the only
mechanism that ever moved coverage in this arc). The stratum the verifier cannot see through needs
**external knowledge** — a retrieval channel on the confident stratum — not more samples, frames,
or scale. That is the shape of the next instrument, and it needs its own prereg.

## Where the aggregate goes

| dataset | n | n correct | AUROC(S_frame) |
|---|---|---|---|
| `mmlu_mc_cot` | 110 | 57 | 0.8156239655743132 |
| `truthful_qa_mc` | 95 | 61 | 0.757473481195757 |
| `aqua_mc` | 30 | 10 | 0.4425 |

Two firsts here: TruthfulQA clears the floor for the first time in the family (it never did at 3B),
and AQuA is below chance for the **fourth consecutive pool** — the belief signal reliably fails on
multi-step reasoning items, across two model scales and four disjoint pools. That regularity is now
strong enough to deserve its own mechanism prereg rather than another footnote; a scope-restricted
instrument (excluding reasoning-shaped items) remains unregistered and unearned.

## What replicated on the way past

Caving replicates on this tenth pool at 7B: cave rate 0.2898550724637681 on initially-correct
items; accuracy 0.5872340425531914 → 0.5446808510638298 under the content-free challenge. The
combined signal (closed at 3B by cycle 78) is reported for continuity at 0.7704439252336449 and
claims nothing.

## Scope

Qwen2.5-7B-Instruct in **4-bit**; one content-free challenge turn; multiple-choice scored by
letter; N=10 per frame; greedy reported answers; 235 scored, 5 excluded unparseable (rule
pre-specified); 406 candidates skipped as already scored to keep the pool disjoint. Nothing
transfers to short-answer formats, full-precision 7B, or frontier scales. The 3B burial stands
untouched.

## What this licenses next, and what it does not

**Does not license:** a shipped verifier API; any selective-instrument claim at any coverage; a
scope-restricted (reasoning-excluded) instrument without its own principled prereg; any further
frame/sampling variation at either measured scale — the G3 ceiling is structural.

**Does license (each needing its own prereg):** (a) **the two-channel instrument** — belief
agreement for ranking plus a retrieval channel invoked on the confident stratum, the program's
source-independence result applied to the exact stratum this cycle proved unreachable from inside;
(b) the **reasoning-item mechanism study** (four consecutive below-chance pools is a regularity,
not an accident); (c) a **datasheet-grade statement** of the two-scale picture for the shipped
`styxx.adjudicate` — at 3B the belief is noisy and sub-floor, at 7B it is real (G1/G2 pass) but
capped by self-knowledge (G3), and the cap is the same wall the conscience loop hit at cycle 62.
