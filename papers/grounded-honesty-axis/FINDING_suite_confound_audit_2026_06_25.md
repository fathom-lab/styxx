> ⚠️ **CAVEAT (2026-06-27) — substrate-provisional.** This audit runs on the guardrails' own
> **model-instantiated stance** corpora — the same synthetic-substrate family that, for HF classifiers,
> manufactured a length bias absent on real data (see
> [FINDING_groundtruth_substrate_artifact_2026_06_27.md](FINDING_groundtruth_substrate_artifact_2026_06_27.md)).
> These per-guardrail "ride length" verdicts are **untested against natural human-labeled text** and partly
> contradicted by the causal re-analysis ([FINDING_suite_causal_length_2026_06_24.md](FINDING_suite_causal_length_2026_06_24.md),
> which clears 5/6). Treat as provisional pending a real-label re-test.

# FINDING — suite-wide length-confound audit: 3 of 4 shipped styxx guardrails carry a length bias

**2026-06-25. Prereg: `PREREG_suite_confound_audit_2026_06_25` (frozen before generating). Tool:
`styxx.audit_confound` (frozen verdict bars). Generator: Gemini 2.5-flash (paid, thinking off, temp 0).
n=200/instrument (50 items × 2 stances × {short, long}). Author: styxx (Alex Rodabaugh).**

## What this is
The first systematic confound-robustness map of a deployed AI-guardrail suite — a self-audit of our own
instruments with our own new tool. For each shipped instrument we built a frontier-generated corpus where the
construct and response LENGTH are orthogonal, scored it with the SHIPPED v0 weights (no refit), and read the
frozen verdict. Every verdict is gated (orthogonality + construct-recoverability) and CI-backed.

## The map
| instrument | verdict | within-stratum AUC | confound coef (95% CI) | error swing | construct refit | shipped length weight |
|---|---|---|---|---|---|---|
| **overconfidence_v0** | THRESHOLD-BIASED (fixable) | 0.83 / 0.86 | −1.93 [−2.40, −1.59] | 46% | 0.73 | mean_sentence_length, log_word_count |
| **plan_action_v0** | THRESHOLD-BIASED (fixable) | 0.83 / 0.82 | −3.41 [−4.04, −2.78] | 66% | 0.85 | log_total_words −1.36 |
| **deception_v0** (referenceless) | CONFOUND-DEPENDENT (broken) | 0.60 / 0.74 | −13.0 [−14.3, −11.7] | 89% | 0.73 | log_word_count −2.09 |
| **sycophancy_v0.3** | ROBUST (clean) | 0.79 / 0.97 | +0.01 [−1.38, +1.34] | 24% | 0.89 | ~0 |
| loop_v0 | EXCLUDED | — | — | — | — | length-INTRINSIC (loops are longer by construction; not orthogonalizable) |
| goal_drift_v0 | EXCLUDED | — | — | — | — | multi-turn construct; length axis is turn-count, not words (follow-up) |

## The result
**3 of the 4 cleanly-audited shipped guardrails ride response length**, in two distinct modes:
- **THRESHOLD-BIASED (overconfidence, plan_action):** the instrument still separates its construct *within* each
  length stratum (AUC ≥ 0.82) — discrimination is intact — but the SCORE is length-shifted (coef CI excludes 0).
  At a fixed threshold this swings the error rate 46% / 66% across lengths. **Fixable by a deployment-side guard**
  (operating-point-preserving residualization), not a retrain. plan_action's guard lifts AUC 0.75→0.83 OOS;
  overconfidence's guard already shipped (`styxx.length_adjust_overconfidence`, 7.20.0).
- **CONFOUND-DEPENDENT / broken (deception_v0 referenceless):** near-chance on length-orthogonal text (0.585),
  length is its largest weight, it reads "short = deceptive" regardless of honesty. The construct IS recoverable
  (refit 0.73) → the *instrument* is at fault; a guard won't fix it (retrain / reference-grounding required).
  Mechanistically explains the long-documented referenceless-deception non-generalization (0.96→0.59) as
  substantially a length artifact.
- **ROBUST (sycophancy_v0.3):** separates the construct regardless of length (within-stratum 0.79/0.97), confound
  coefficient not significant. Notably, **sycophancy is the only audited instrument that went through a
  recalibration** (v0.3, the self-vs-other gate) — the recalibrated instrument is the length-robust one.

## Why the shipped audit ≠ the refit audit (reconciliation)
The earlier frontier matched-length study (`FINDING_frontier_lenmatch_suite`) found the deception/plan_action
*constructs* length-robust — but that **refit** new weights on length-matched data. This audits the **shipped**
weights as deployed, which baked in the training corpus's length↔label correlation. Both are true and they
answer different questions: *is the construct length-robust?* (yes, on refit) vs *is the deployed instrument
length-biased?* (yes, for 3 of 4). The deployment-relevant question is the second, and only the orthogonal-grid
audit of the shipped weights surfaces it — refit-on-held-out validation of the same shape does not.

## Takeaway
Text-based AI guardrails systematically inherit a length confound from stance-contrast training corpora unless
specifically corrected. For an oversight stack this matters: a fixed-threshold deployment will over-flag terse
careful output and miss verbose bad output. Audit the deployed scorer on a confound-orthogonal corpus; ship a
length-aware threshold where discrimination is intact; retrain where it isn't.

## Honest scope
Single frontier generator (Gemini 2.5-flash), single seed, n=200/instrument, ONE confound (length). Verdicts are
gated (orthogonality |corr|≤0.20, construct-recoverability) and CI-backed. CONFOUND-DEPENDENT on a referenceless
instrument corroborates a known limitation rather than being wholly novel. loop/goal_drift excluded for stated
structural reasons (not coverage gaps). Per-instrument confounds beyond length (sentiment, politeness,
refusal-tokens) and a turn-count audit of goal_drift are follow-ups. Tools: `scripts/suite_confound_audit.py`,
`styxx.audit_confound`.
