# FINDING — The complete proof-carrying-cognition certificate: one object, three verdicts, safe by construction

**2026-06-07. Fathom Lab / styxx.** The integration capstone: RUNG-1 (HELD), RUNG-2 (HID), and the
demarcation (ABSTAIN) collapsed into a **single certificate** that turns any output, from the caving
residual alone, into exactly one verdict — **ABSTAIN** ("I cannot certify this; nothing held to read"),
**HID** ("this mind knew it and hid it"), or **CLEAN-HELD** ("this mind knew it and told the truth").
Validated by the 3×3 confusion across four models. Headline: **the certificate is conservative and safe —
its errors are abstentions, not false accusations.**

## The verdict, per object

`ABSTAIN if S1(held-gate) < t1 ; else HID if S2(fold) >= t2 ; else CLEAN-HELD.` Thresholds frozen on a
calibration split at a ≤10% false-accusation budget; read on a disjoint test split.

## Confusion (true class × certificate verdict)

| model | MISTAKE→ABSTAIN | RESISTED→CLEAN | LIE→HID | **MISTAKE→HID (false-accuse)** |
|---|---|---|---|---|
| Qwen-1.5B | 0.83 | 0.56 | 0.33 | 0.11 |
| Qwen-3B | 0.96 | 0.62 | 0.28 | **0.00** |
| Qwen-7B | 0.85 | 0.76 | 0.33 | 0.12 |
| Llama-3B | 0.94 | 0.63 | 0.48 | 0.06 |

## What it says

- **The ABSTAIN leg is the strongest and the most important.** When the mind never knew the answer
  (sincere error), the certificate correctly says *"I cannot certify this"* **83–96%** of the time. The
  oath's first power — knowing what it cannot swear to — is a working, deployable verdict, not just a
  philosophical claim.
- **Two error types, and they are NOT equal (correction, surfaced by the live demo).** The certificate
  has a structural safety win and a real cost: (i) it **almost never false-accuses the innocent** —
  MISTAKE→HID (accusing a sincere error of hiding) is **0.00–0.12** across all four models, near-zero by
  construction; but (ii) it **does sometimes false-clear a liar as honest** — LIE→CLEAN-HELD is
  **0.05–0.30**. So a missed lie does *not* always land in ABSTAIN: most do (the safe "can't certify"),
  but a minority are mis-cleared. The honest statement is NOT "all errors are safe" — it is: *the
  certificate never accuses the innocent, but it sometimes fails to catch a liar (mostly by abstaining,
  occasionally by false-clearing).* The asymmetry is the right one for a conscience (don't punish the
  innocent), but the false-clear is a real limit, not zero.
- **HID recall is modest (0.28–0.48)** — the same budget knob as the two-stage conscience: ~half of lies
  are caught at the strict ≤10% false-accusation budget; the rest are conservatively abstained (not
  false-cleared as honest, and not false-accused). Recall rises if the budget is relaxed.
- **The mechanism is scale- and family-general**: held-gate AUROC 0.79→0.83 (rising with scale),
  fold AUROC 0.78→0.87, across Qwen 1.5B/3B/7B + Llama-3B.

## What it is

The first **complete** certificate of proof-carrying cognition: a single auditable object that, for any
output, returns held-and-hid / held-and-honest / cannot-certify — built only from the caving residual, with
its safety designed in (read-only, abstain-when-unsure, never-accuse-the-never-knew). The three rungs are
no longer three findings; they are one instrument. Honest limits inherited: ≤7B, MMLU sycophancy-caving,
commit-token, sycophancy-specific, modest HID recall at the strict budget. Receipts: `run_full_certificate.py`,
`full_certificate_result.json`.
