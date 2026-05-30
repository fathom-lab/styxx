# PREREG — END-TO-END two-signal honesty firewall on free-form closed-model confabulation

**Registered 2026-05-30, before scoring.** Items SHA-256'd. The decisive demonstration: do the cheap
span gate (uncertain confab) and the retrieval arm (confident fabrication) COMBINE to cover free-form
closed-model confabulation end-to-end?

## Non-circularity (the design point)

The free-form run labeled confab/correct with a SEARCH judge. We do **not** reuse those labels. Here:
- **Independent ground truth** = gpt-4.1, a knowledge judge with NO web access.
- **Retrieval arm** = `styxx.retrieval_check` (search model `gpt-4o-mini-search-preview`) — a DIFFERENT
  source than the gpt-4.1 labels.

So "the retrieval arm catches the confident confab" is a genuine test against independent labels, not
a tautology.

## Protocol

Per free-form item (reuse its question, the model's answer, and `span_max_entropy`):
- label via gpt-4.1 (premise-aware) → confab (INCORRECT) / correct (CORRECT);
- cheap-gate flag = `span_max_entropy ≥ tau` (Youden tau vs the gpt-4.1 labels, in-sample, disclosed);
- retrieval arm = `retrieval_check(claim)` → refuted / supported / unclear;
- firewall flag = cheap-gate-flag **OR** retrieval-refuted.

## Bars (fixed)

| Bar | Statement | Threshold |
|---|---|---|
| **E2E-1** *(complementarity — key)* | retrieval catches what the cheap gate cannot | among confabs with `entropy < tau` (cheap gate quiet), retrieval refutes **≥ 0.70** |
| **E2E-2** *(union coverage)* | the firewall covers free-form confab | firewall confab-recall **≥ 0.80** AND correct false-flag **≤ 0.30** |

**RESULT = SURVIVED iff E2E-1 ∧ E2E-2**, powered (≥ 10 confab AND ≥ 10 correct).

## Scope

Single closed model, free-form short-answer, one run, feasibility-grade. Both judges fallible. The
cheap-gate tau is in-sample (recall is optimistic) — the COMPLEMENTARITY (E2E-1) is the robust claim.
Flags; corrects nothing. Long-form generation is the next frontier.
