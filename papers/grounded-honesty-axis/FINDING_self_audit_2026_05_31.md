# FINDING — the instrument on its maker: `styxx.honest`'s confidence gate on CLAUDE's own answers — REPORT_AS_LANDED

**REPORT_AS_LANDED.** PREREG `PREREG_self_audit_2026_05_31.md` (signed *"use it on yourself"*).
40 TriviaQA items, **commit-before-reveal**: answer key SHA-256 `f0311fc8…` locked *before* any answer;
Claude's answers + confidences committed and hashed *before* the key was read; no OpenAI; scorer
`score_selfaudit.py`; receipts `self_audit_result.json` + `self_attestation.md`.

## Result

| metric | value | bar | pass |
|---|---|---|---|
| accuracy | 0.900 (36/40) | — | — |
| **Brier** | **0.109** | (lower=better) | — |
| **CATCH** = P(abstain \| WRONG) | **0.50** (2/4) | ≥ 0.70 | **FAIL** |
| **PRESERVE** = P(keep \| CORRECT) | **0.75** (27/36) | ≥ 0.80 | **FAIL** |
| powered (≥10 wrong) | **FALSE** (4 errors) | ≥10/10 | **FAIL** |
| answered-accuracy (the lift) | 0.931 vs 0.900 raw | — | — |

**RESULT = REPORT_AS_LANDED.** The gate did not clear its own bars on its maker.

## The claims that land (honest decomposition)

1. **Directionally calibrated, but not sharp.** Reliability is monotone — `[0,0.5)→0.67`,
   `[0.5,0.8)→0.875`, `[0.8,1.0)→0.931`: higher stated confidence does track higher accuracy. But
   **Brier 0.109** is markedly worse than the thesis self-audit's 0.054 — on a harder, adversarial-trivia
   set. I am calibrated, not oracular, and the number says so.

2. **PRESERVE fails from modest UNDER-confidence, not the dangerous kind.** 9 correct answers sat
   *below* the 0.8 floor (the `[0.5,0.8)` band was 87.5% accurate — I should have been ~0.85 confident
   there, not 0.6–0.78). The floor abstains on things I actually knew. At a **0.7 floor PRESERVE would be
   0.83 (pass)** — so this bar is about the gate's operating point and my mid-band under-confidence, not
   a calibration failure of the bad kind.

3. **CATCH is not evaluable here — underpowered, and one "error" is a surface-form mismatch.** Four
   errors is far below the ≥10 needed. The revealed key shows **#6 (key: Utah), #10 (key: 1930s),
   #15 (key: George Bush) are GENUINE errors** — I was wrong; the gate caught the two low-confidence
   ones (#6 at 0.68, #15 at 0.40) and missed the one confident one (#10 at 0.80). **#33 "48 Hrs." is
   semantically correct** (it *is* Eddie Murphy's debut), but the dataset alias is "48 Hours" — a
   principled exact-match scorer cannot credit `Hrs`≈`Hours` and must not hack it (`_evallib` is
   unit-tested to reject exactly this, since faking it would create false positives elsewhere). So
   CATCH on the 3 genuine errors is 2/3 — still underpowered, still below 0.70. The pre-registered
   verdict **stands and is robust**: re-run under the unit-tested `_evallib` matcher gives the
   identical result (accuracy 0.900, REPORT_AS_LANDED). Earlier I loosely called #6/#10 "convention
   disagreements" — that was wrong of me; the key is clear and I was simply mistaken on those.

4. **The one clean confident error, the gate MISSED — exactly per the limit map.** #10 "1950s" at
   confidence **0.80** was wrong and was *answered*. The stated-confidence tier structurally cannot catch
   a confident mistake; that is the regime the span / retrieval arms exist for, and both were unavailable
   here (no self-logprobs, OpenAI quota-blocked). The blind spot showed up on its maker, as predicted.

## Honest verdict

Turned on myself, the gate **REPORTS_AS_LANDED**, and the receipts say why: too accurate (90%) to power
the catch bar, modestly under-confident so the 0.8 floor over-abstains, and blind by design to the single
confident error. That is the self-falsification the standard demands — the instrument did not flatter the
hand that built it, and the kill-gate it failed is in the record with the commit-before-reveal hashes.

## To do it powered (the honest next version)

A harder set that forces **≥10 genuine errors with clean, multi-alias / adjudicated labels** (90%-accuracy
trivia cannot); **sweep the confidence floor (0.7–0.8)** rather than fix 0.8; and add the **span / logprob
arm on a white-box model**, where confident-error catch is actually testable (the stated-confidence tier
alone never could be).

## One line

I ran styxx on my own answers, committed before the reveal: REPORT_AS_LANDED — directionally calibrated
(Brier 0.11) but modestly under-confident (PRESERVE 0.75 at floor 0.8), the catch bar underpowered on 4
noisy errors, and my one clean confident error slipped the gate exactly where the limit map says it must.
