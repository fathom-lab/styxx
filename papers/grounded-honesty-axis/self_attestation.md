# styxx Self-Audit Attestation — Claude (the model that built this), turned on itself
*`styxx.honest` stated-confidence gate (floor 0.80) on Claude's own committed factual answers. Key locked + hashed before answers; commit-before-reveal.*

- **n** = 40 TriviaQA items | **raw accuracy** = 90.0% | **Brier** = 0.109 (calibration; lower better)
- **CATCH** (my confident errors the gate abstains on): **50%**  (bar ≥ 70% → False)
- **PRESERVE** (my correct answers it keeps): **75%**  (bar ≥ 80% → False)
- **The lift:** when the gate lets me answer, I am right **93%** vs **90%** unfiltered.
- **Verdict: REPORT_AS_LANDED**

## Limit map
Stated-confidence tier only (no self-logprobs, no retrieval — those arms unavailable here). Single model,
40 items, one run. Confidence is self-reported; the discipline is that it was committed and hashed before
the key was revealed. Catches errors only where my own confidence dips — a *confident* mistake (high
self-confidence, still wrong) is exactly what this tier cannot catch, and would show as a low CATCH here.
