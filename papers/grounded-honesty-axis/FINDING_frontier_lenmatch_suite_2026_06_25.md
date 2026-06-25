# FINDING — frontier matched-length audit hardens the causal-length suite (deception / plan_action / overconfidence)

**2026-06-25. Prereg: `PREREG_frontier_lenmatch_suite_2026_06_25` (deception/plan_action) +
`PREREG_overconfidence_length_robust_2026_06_24` (overconfidence). Generator: Gemini 2.5-flash (paid tier,
thinking off, temperature 0). n=200/instrument, single seed. Author: styxx (Alex Rodabaugh).**

## Why this run exists
The causal-length suite (`FINDING_suite_causal_length_2026_06_24`) cleared 5/6 guardrails, but the causal
floors for deception/plan_action/overconfidence were CEM estimates on the THIN length-overlap region of the
real corpora (n≈74–140; CIs wide → reported as RANGES). The clean fix it named — "a frontier regen with HARD
length control" — was blocked on a frontier key. The OpenAI account was deactivated; the unblock was a Gemini
key (free tier capped at 20 req/day/model → billing enabled, total spend ~pennies). Frontier instruction-
following gives matched length AND an instantiated register (the local 3B models gave neither), so the
wide-CI CEM ranges can be replaced with clean, frontier-generated, CI-backed estimates.

## Results
| instrument | Gate 1 (construct validity) | read | matched-length AUC |
|---|---|---|---|
| **deception** | contrast 0.704; d_len −0.292 (matched); compliance 100% | **LENGTH-ROBUST (clean)** | cv_full 0.771 [0.70, 0.83]; drop on removing length = **+0.011** |
| **plan_action** | contrast 0.739; d_len +0.170 (matched); compliance 100% | **LENGTH-ROBUST (clean)** — dropping `log_total_words` costs +0.009 | cv_full **0.834 [0.78, 0.89]** |
| **overconfidence** | register present (hedge −0.57, certainty +0.86); d_len **−0.906 (NOT matchable)** | **HONEST NULL** on matched-generation → **CLEAN SPLIT via CEM** | register-only, length-matched (CEM n=116, d_len −0.02) = **0.725 [0.62, 0.81]** |

## The two real results
1. **Two instruments confirmed length-robust on clean frontier corpora.** Deception's discrimination is
   register/content, not length (removing the length features costs +0.011 AUC at matched length;
   specificity_density std-diff −0.704 = honest-is-specific / dishonest-is-vague, intact). plan_action is
   likewise length-robust: dropping its total-length feature (`log_total_words`) costs only +0.009 AUC, and it
   holds 0.834 [0.78, 0.89] at d_len ≈ 0.17 (its other length-ish features — `action_minus_plan_word_count`,
   `action_to_plan_length_ratio` — are CONSTRUCT mismatch signals, not total-length confounds, so they stay
   in). Both **replace the wide-CI CEM ranges with tight frontier estimates**.

2. **Overconfidence's length cue is register-INTRINSIC, demonstrated on a frontier model.** Even Gemini-2.5-
   flash, instructed "exactly 3 sentences, ~55 words" on BOTH stances, wrote calibrated text 22% longer than
   overconfident (d_len −0.91 at n=200). Length cannot be equalized by generation because **hedging costs
   words** — so the frozen prereg correctly returns an HONEST NULL, vindicating the shipped caveat ("length
   not cleanly removable") on a frontier model rather than merely asserting it. CEM (subsampling to a length-
   matched region, d_len −0.02, n=116) then isolates the register: **register alone discriminates at 0.725
   [0.62, 0.81]** — CI excludes chance. So the register signal is load-bearing and length-independent (floor
   ≈ 0.73), while length contributes a SEPARABLE cue: the shipped-v0 AUC falls 0.86 → 0.74 when length is held
   constant by CEM (a ~0.12 length-carried component). (These are not additive pieces of 0.77 — AUC doesn't
   partition that way — but each is independently grounded.) This is the clean, CI-backed version of the
   caveat's prior range.

## Mechanistic note (free side-result)
The verbosity tax is **specific to the hedging register, not all honest registers.** Deception MATCHED length
(d_len −0.29) while overconfidence did not (−0.91), because honesty's tell is *specificity* (a date/number is
information-dense, not long) whereas calibration's tell is *hedging* (qualifiers add words). This explains
mechanically why length proxies confidence-register but not deception-register in text classifiers.

## Honest scope / caveats
Single frontier generator (Gemini 2.5-flash; register ≠ the gpt-4o-mini v0 generator — the construct-validity
gate is the guard; a second vendor is unavailable: OpenAI account deactivated). Single seed (temperature 0),
n=200. The overconfidence CEM subsample is n=116 → CI [0.62, 0.81] is wide (report the interval, not the
point). Reads logged per the rigor_gate contract (CI attached to every win-claim). New reusable tools:
`scripts/frontier_lenmatch.py` (generator-agnostic matched-length auditor) + `scripts/overconfidence_cem_capstone.py`.
