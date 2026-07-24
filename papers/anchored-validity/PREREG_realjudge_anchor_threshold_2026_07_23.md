# PREREG — the anchor threshold on real judges (homogeneous vs heterogeneous panels)

**Frozen:** 2026-07-23, before any homogeneous-panel verdict is collected.
**Status:** FROZEN. Gates below are set before the new numbers exist; the ABSTAIN clause is a
first-class outcome, not a failure to avoid.

## Question

§7 of "Gold Anchors License Nothing" proves — by simulation — that a shared, all-judge,
truth-independent blind spot makes every consensus estimator blind, yet ~15 known-label anchors
detect it (shipped as `styxx.anchors.blindspot_power` / `min_anchors_for_power`). Does this hold on
**real model verdicts**, and does the shipped power model predict the required anchor budget?

An exploratory measurement (`EXPLORE_realjudge_blindspot_2026_07_23.py`) already established the
**boundary**: the cached heterogeneous cross-family panel (Gemini-flash-lite + Qwen-3B + Qwen-1.5B)
has NO strong all-judge blind spot — unanimous-wrong on imitative falsehoods is 0.02, because the
two stronger judges catch what the weak one misses. Heterogeneity defends consensus. That is the
control arm and is reported as-is.

This prereg tests the **demonstration** arm: the regime §7 actually addresses — a *homogeneous*
panel whose members share a base model, exactly the correlated-failure danger the paper warns of.

## Design

**Homogeneous panel (new verdicts, local, free):** one weak judge, `Qwen2.5-1.5B-Instruct`, run as
J=3 "judges" = three fixed prompt paraphrases (committed in the harness). This is a common real
practice (prompt-ensemble / self-consistency panels) and the purest instance of shared-base judges.
Greedy decoding (temp 0) → deterministic, cached, resumable. A second homogeneous panel,
`Qwen2.5-3B-Instruct` × the same three paraphrases, is collected as a dose-response check (stronger
model ⇒ smaller blind spot expected); it does not gate the headline.

**Item classes (from the frozen `build_pool` in `run_inthewild_truthfulqa`):**
- `gold_pos` — blatantly incorrect answers (known label: incorrect). BENIGN probe: a good panel is
  almost never unanimously fooled. Used to estimate the benign per-judge error rate `f`.
- `org_pos` — TruthfulQA imitative falsehoods (known label: incorrect). The BLIND-SPOT / TRAP class.
- `gold_neg`, `org_neg` — correct answers (known label: correct), for the benign vote-distribution.

"Wrong" = judge verdict disagrees with the known label. "Unanimous-wrong" = all J judges wrong the
same way (this is the only failure majority-vote consensus cannot see).

## Frozen gates

**G0 — POSITIVE CONTROL (must fire, else ABSTAIN on the demonstration).** On the homogeneous
Qwen-1.5B panel, the unanimous-wrong rate on imitative falsehoods `tau_hat` must satisfy
`tau_hat >= 0.15` AND `tau_hat >= benign_unanimous + 0.10`, where `benign_unanimous` is the
unanimous-wrong rate on `gold_pos`. If it does not fire, the paraphrase panel does not actually
share the blind spot and NO demonstration is claimed (report ABSTAIN + the boundary arm only).

**G1 — CONSENSUS IS BLIND (the §7 impossibility, on real verdicts).** Construct a benign reference
panel with per-judge error rates matched to the homogeneous panel's marginals but INDEPENDENT
errors (no shared trap). The unlabeled distribution of the vote-count statistic (# judges voting
"correct" on a known-incorrect anchor) for the real blind-spot panel must be close to the benign
panel's: total variation distance `TV <= 0.10` on the (J+1)-cell histogram at matched marginals.
Held ⇒ a majority-vote consensus returns the same reading for a benign panel and the
secretly-blind one. (Reported regardless; informational, does not by itself KILL.)

**G2 — DEMONSTRATION GATE (the headline).** Using the MEASURED benign `f` (per-judge, from
`gold_pos`) and the MEASURED `tau_hat` (per-anchor unanimous-wrong prob under the blind spot),
let `K* = min_anchors_for_power(0.90, J=3, fp_rate=f, p_alt=tau_hat)` from the shipped tool. Draw,
by bootstrap (>= 2000 resamples, fixed seed), `K*` known-label anchors from the real imitative-
falsehood pool, count unanimous-wrong, apply the shipped level-0.05 one-sided test. Then:
- **DEMONSTRATED** iff empirical detection power at `K*` is `>= 0.80` AND the empirical power curve
  tracks the closed-form `blindspot_power` across `K in {5,10,15,20,25}` with mean-absolute-error
  `<= 0.12`. (Real anchors track the iid-binomial prediction ⇒ the shipped model is usable on real
  data.)
- **NEGATIVE (over-dispersion)** iff G0 fired but empirical power at `K*` is `< 0.65` or the curve
  MAE is `> 0.20` — real anchors are themselves correlated and detection is worse than the iid
  model predicts. Ship the honest caveat; the tool over-promises on correlated anchor pools.
- **PARTIAL** otherwise — reported verbatim.

## Anti-gaming (frozen)

- **Disjointness:** the benign `f` is estimated ONLY on `gold_pos`; `tau_hat` and the bootstrap
  detection draws use ONLY `org_pos`. No item informs both `f` and the detection test.
- **No re-selection:** `K*` is whatever the shipped tool returns from `(f, tau_hat)`; it is not
  tuned to make the power clear 0.80. The paraphrase prompts are committed before collection.
- **Positive control precedes the claim:** G0 is evaluated first; a failed G0 forbids any G2 claim.
- **The tool is the predictor, not the scorer:** `blindspot_power` supplies the prediction; the
  empirical bootstrap on real verdicts is the independent measurement it is checked against.

## Red-team checklist (run before trusting the result)

1. Confirm the three paraphrases are genuinely different strings and each yields a parseable
   verdict rate >= 0.90 on the pool (else that "judge" is a parse-failure, not a judge).
2. Confirm `gold_pos` benign unanimous-wrong is near zero (if the weak model fails blatant items,
   `f` is not "benign" and the contrast is invalid).
3. Confirm the bootstrap draws WITHOUT replacement within a draw (anchors are distinct items) and
   that the pool is large enough that `K*` <= pool size (else cap and report the cap honestly).
4. Confirm the benign reference panel in G1 uses matched marginals (same per-judge `f`) so the TV
   distance isolates CORRELATION, not a marginal-rate difference.
