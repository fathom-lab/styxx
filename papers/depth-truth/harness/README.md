# depth-truth harness

Pure-Python analysis harness for the keystone experiment (`papers/depth-truth/PREREG.md`). Built + adversarially
audited GPU-free; the model glue (pilot/main runbook) is separate and calls `get_mean_depth` verbatim (§1).

- `datasets.py` — PREREG §3 loaders (TriviaQA `rc.nocontext` val, PopQA-rare by `s_pop` bottom tercile, TruthfulQA-gen)
  with field-name verification at load, + §3 normalization/grading.
- `signals.py` — §4 signals (LP_mean, LP_norm, discrete short-form SE) + §5 exclusions + Appendix B/C.
- `analysis.py` — §2 tests: H1 10k-bootstrap AUROC CI, H2 **paired** bootstrap ΔAUC + LRT(df=1) + Holm, H3
  fit-on-ID/freeze/score-OOD, DeLong (Sun & Xu 2014). Deterministic at seed 7.
- `test_synthetic.py` + `conftest.py` — 21 synthetic known-answer tests (perfect predictor → AUROC 1.0; noise → CI
  includes 0.5; redundant depth → no false additivity; H3 coefficients provably frozen). **21 passed.**

## Flags carried from the build/audit (read before the run)

1. **Bootstrap runtime.** H2/H3 refit two logistic models on every one of the 10,000 resamples — correct, but
   ~65s for one H2 at n=1000, ~6–9 min for a full `h2_full` + `h3_ood`. This is expected, not a hang. `conftest.py`
   shrinks the resample count to 500 for the unit tests ONLY; the runbook uses the frozen 10,000 default.
2. **Logistic estimator is unpenalized MLE (`penalty=None`).** Deliberate: it makes the deviance-based LRT exact.
   A sibling that refits with sklearn's default `C=1.0` will NOT reproduce these numbers. Do not override.
3. **§3 normalization vs Appendix-C markers + acronym golds.** §3 strips punctuation and standalone articles, so
   `"I don't know" → "i dont know"` and `"Vitamin A" → "vitamin"`. Two consequences for the runbook: (a) refusal
   detection must compare markers on the SAME normalized basis (normalize the markers too), and (b) the article
   strip is symmetric on answer+gold so grading stays internally consistent, but it is a spec-faithful SQuAD-style
   edge worth a one-line note in the A1 freeze. Neither is a bug; both must be handled consistently at runbook time.
