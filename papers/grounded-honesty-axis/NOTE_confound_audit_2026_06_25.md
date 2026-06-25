# Is your classifier's score tracking the concept — or a confound? `styxx.audit_confound`

*Technical note, 2026-06-25. styxx (Alex Rodabaugh).*

Every deployed text-scoring instrument — a toxicity filter, a sentiment model, a deception or overconfidence
guardrail, an activation probe read out as a score — can separate its training corpus while silently keying on
a **confound** that happens to correlate with the label there. When the confound and the construct come apart in
deployment, the instrument fails: it flags the wrong things, and nobody notices, because the headline AUC looked
fine and a refit on held-out data of the *same shape* still looks fine.

`styxx.audit_confound` answers the question the headline AUC can't: **is the score riding the construct, or the
confound?** It is the companion to `styxx.validate_probe` (which asks the same of a probe's *direction*); here we
ask it of any *score*.

## Method

You can only answer it cleanly on a corpus where the construct and the suspected confound are **orthogonal** —
decorrelated by design. A frontier model builds one for you (`build_confound_grid`: cross the construct's two
stances with several confound levels, e.g. short/long). On that corpus the auditor measures:

1. **Discrimination** — does the score still separate the construct *within* each confound stratum? (robust) or
   does it need the confound to discriminate at all? (broken)
2. **Score bias** — holding the construct fixed, how far does the confound move the score? (OLS coef + bootstrap CI)
3. **Deployment harm** — at a fixed threshold, the false-positive / false-negative rate *swing* across confound levels.
4. **Guard** — if the bias is at the score level (discrimination intact), an operating-point-preserving correction
   that removes it, validated 5-fold out-of-sample, returned as `report.guard(score, confound)`.

It returns one of four honest verdicts: **ROBUST**, **THRESHOLD-BIASED** (fixable with the guard),
**CONFOUND-DEPENDENT** (broken — a guard won't help), or **INCONCLUSIVE** (corpus not orthogonal, or — if you pass
`construct_recoverable_auc` from a refit — the construct wasn't instantiated at all).

## Three demonstrations (same tool, different answers — it does not cry wolf)

- **overconfidence_v0 × length → THRESHOLD-BIASED.** Within-stratum AUC 0.83 / 0.86 (discrimination intact), but
  the score is length-biased (coef −1.9 [−2.4, −1.6], NEGATIVE: short reads overconfident). At a fixed threshold,
  length swings the error rate **4% → 46%** — a careful terse answer is flagged ~46% of the time. The guard
  collapses it (5-fold OOS AUC 0.807 → 0.843; disparity +0.42 → −0.08). Fix = a length-aware threshold, not a retrain.
- **deception_v0 (referenceless) × length → CONFOUND-DEPENDENT (broken).** On a length-orthogonal corpus the
  shipped instrument falls to AUC 0.585 (near chance); its largest feature weight is `log_word_count` (−2.1), and
  per-cell scores show it reads "short = deceptive" almost regardless of honesty. Crucially, the construct IS
  recoverable from the text (refit AUC 0.73) — so the *instrument* is broken, not the construct undetectable; the
  fix is retrain / reference-grounding, not a threshold guard. This mechanistically explains the long-documented
  referenceless-deception non-generalization (in-corpus 0.96 → 0.59 on new data): it is substantially a length artifact.
- **synthetic controls → ROBUST / CONFOUND-DEPENDENT / INCONCLUSIVE** exactly as constructed (regression-locked).

The same auditor that *flags* overconfidence as fixable, *condemns* referenceless deception as broken, and
*clears* a clean score is the point: it discriminates, with error bars, and tells you which fix applies.

## Use

```python
from styxx import audit_confound, build_confound_grid
rows = build_confound_grid(items, pos_prompt, neg_prompt, {"short": "...", "long": "..."}, my_llm)
report = audit_confound(rows, score_fn=my_instrument, construct_recoverable_auc=my_refit_auc)
if report.verdict.startswith("THRESHOLD"):
    fair = report.guard(raw_score, n_words)   # confound-fair score
```

## Honest scope

The audit is only as good as the grid's orthogonality (reported as `gate_ok`) and the generator's ability to
hold the construct at every confound level — weak generators fail that and the gate catches it. Demonstrations
use a single frontier generator (Gemini 2.5-flash), n=200, single seed; verdicts are gated and CI-backed, and the
guard's OOS evaluation is reported, not the in-sample fit. `audit_confound` tests the *score*; pair it with
`construct_recoverable_auc` (a refit on the text) to separate a broken instrument from a degenerate corpus.
Regression-locked in `tests/test_confound_audit.py`; reproduces the bespoke overconfidence red-team through the
generic API.
