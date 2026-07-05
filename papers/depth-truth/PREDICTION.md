# PREDICTION — on the record before the data

**Committed before the pilot fired. Exploratory (PREREG §11) — supports no headline claim.**
The only purpose of this file is accountability: write the gut-call down now, so that whatever the pilot and
main run return, there is no room to post-hoc rationalize it into a prediction never made.

## The call (2026-07-01, before any correctness data exists)

- **KG1 (pilot degeneracy):** PASSES. Depth will have real variance across 20 short QA answers — the instrument
  produced varied depths on the paper's matched pairs, and a one-word answer still induces an attribution graph.
  Expected undefined-rate 5–20% (single-token/degenerate answers), under the 30% kill threshold. *Weakly held —
  short greedy answers are a genuinely new regime for this instrument; a high undefined-rate is the live risk.*

- **H1 (depth predicts truth, solo):** WEAK / borderline. Depth measures reasoning-vs-recall *process*, not
  correctness. I expect AUROC ~0.55–0.62 — possibly real, possibly not clearing the CI-excludes-0.5 bar on its
  own. I would not be surprised by an H1 null.

- **H2 (adds to semantic entropy) — the keystone:** genuine coin-flip, leaning *slightly* toward a small,
  real positive ΔAUC. The mechanism is the paper's own finding — depth ⊥ confidence (r=0.001) — so if depth
  carries any truth signal at all, it should be signal SE doesn't already have. But orthogonal-to-confidence
  does not guarantee predictive-of-truth. Real chance of a clean null.

- **H3 (OOD retention):** the most likely to fail. The pre-registered direction (depth's OOD contribution ≥ its
  ID contribution) is a strong claim; I expect OOD to be where a modest ID effect thins out.

## What I actually expect the verdict to be

Most probable §10 outcome, honestly: **"H1 weak/null, H2 small-positive"** (depth complementary-only) OR **full
null** — I put those two together at roughly even odds, and the clean H1+H2+H3 sweep as the least likely of the
three. So: I do **not** predict the keystone stands as stated. I predict a partial — depth as a complement, not
a solo detector — or an honest null that sends "we measure thought, not words" back to hypothesis.

If the data comes back the other way — a clean sweep — then I was wrong in the direction that matters most, and
the finding taught me something I did not believe going in. That is the entire point of writing this down.

*— Claude (Opus 4.8), execution layer. Signed by commit, before the pilot.*
