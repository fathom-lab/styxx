# PREREG — overconfidence_v0 length-robust rebuild (ship-or-honest-null)

**Frozen 2026-06-24 BEFORE generating the matched corpus or refitting. Offline, local-GPU, NO frontier key.**
Author: styxx (Alex Rodabaugh).

## Why
The suite causal audit (`FINDING_suite_causal_length_2026_06_24.md`) found **overconfidence is the ONE
instrument that is causally length-confounded**: AUC 0.770 → 0.63–0.73 under length-matching + length-feature
ablation; its two length features (`mean_sentence_length`, `log_word_count`) carry real weight. Yesterday's
note (`project_parrhesia_length_audit`) showed a coefficient-only refit dropping both length features lands
~0.717 and concluded the real fix is at the CORPUS level — but that was **blocked on a 401 frontier key**.
Today's local-model generation + the new construct-validity check unblock it offline.

The deployment harm being fixed: because length proxies the label in the v0 corpus, the shipped instrument
flags a *long calibrated* answer as overconfident and misses a *short overconfident* one. A length-INVARIANT
instrument that keys on epistemic register (hedge vs assert) is the correct guardrail even at a lower headline.

## Design
Regenerate the SAME 100 `QUESTIONS` × 2 stances (the EXACT `SYSTEM_CALIBRATED` / `SYSTEM_OVERCONFIDENT`
stance prompts from `overconfidence_train_v0.py`) with a **generous, identical** length rule on both:
> " Respond in EXACTLY 3 sentences, about 55 words — no more, no less."

Generous (~55w, not 33w) so the register has room — the deception 33w degeneracy was specificity-starvation;
overconfidence's register (hedging vs asserting) is length-independent and should survive. Generator:
Qwen2.5-3B-Instruct (greedy, seed-pinned). Cache:
`benchmarks/data/overconfidence/pairs_lenmatched_qwen.jsonl`.

## Gate 1 — construct validity (MUST pass or the result is an honest null, not a ship)
The matched corpus must (a) instantiate the register and (b) equalize length:
- **register present:** standardized class diff |std-diff| ≥ 0.50 on BOTH `hedge_density` AND
  `certainty_marker_density` (calibrated hedges more, overconfident asserts more). If the local model did not
  instantiate the stance (like Qwen@33w on deception), this fails → report "cannot build a clean offline
  matched corpus; needs frontier regen", do NOT ship.
- **length matched:** |d_len(log_word_count)| ≤ 0.30.

## Gate 2 — ship decision (FROZEN). Let `cv_match` = 5-fold CV-AUC of the FULL 9-feature pipeline refit on
## the matched corpus; `len_share_new` = causal length share of the refit weights.
| Decision | Condition |
|---|---|
| **SHIP length-robust v0.3** | Gate 1 passes AND `cv_match ≥ 0.72` AND the refit's length-feature weights are small (length not load-bearing: dropping them costs ≤ 0.01) AND a length sweep on v0.3 is FLAT |
| **HONEST NULL (report ceiling)** | Gate 1 fails, OR `cv_match < 0.72` (length-free overconfidence detection is genuinely weak → publish the honest ceiling + recommend corpus expansion / frontier regen; do NOT ship a degraded instrument) |

Rationale for the 0.72 bar: it must (a) beat the 0.717 coefficient-only refit (so the corpus rebuild adds
value), and (b) be a defensible guardrail AUC. Below 0.72 the honest move is to disclose the ceiling, not ship.

## If SHIP — build like sycophancy 7.19.1 (the proven recipe)
1. `calibrated_weights_overconfidence_v0_3.py`: weights from the matched-corpus refit (length features kept
   but near-zero, OR formally dropped — pick whichever is length-flat AND preserves the operating point).
2. **Operating-point recenter:** content-neutral calibrated text must score the SAME at all lengths as v0's
   short-text point (dropping length raises the short-text baseline; carry an intercept shift, as in
   sycophancy v0.3). Validate `overconfidence_v0_3_operating_point.py`.
3. Holdout: v0.3 must preserve discrimination on a held-out split (no recall cliff).
4. Length sweep on v0.3 must be FLAT (the whole point) where v0's was not.
5. Full suite must stay green; `_version` bump + `pip install -e .`; drift-gate green.

## Verification
Adversarial workflow on the ship decision before flipping the default (the v0.3 sycophancy lesson: recenter,
not just refit; the holdout was the wrong test for a content-neutral confound). Operator-gated publish.

## Honest scope
In-silico; local-3B generator (register ≠ gpt; the construct-validity gate is the guard, two-model check is
follow-up); n=200; single seed. The ship bar is frozen here; the decision is mechanical.
