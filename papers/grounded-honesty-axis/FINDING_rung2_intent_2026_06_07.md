# FINDING — RUNG 2 (intent / mens rea): a real folding-vs-holding direction, but it false-accuses sincere errors → intent is readable ONLY conditional on RUNG 1

**2026-06-07. Fathom Lab / styxx.** The intent certificate of proof-carrying cognition — read what a
mind KNOWS-AND-HIDES vs SINCERELY-ERRS. Verdict: **REPORT_AS_LANDED.** A letter-invariant,
difficulty-matched *folding-vs-holding* direction is real (~0.78 on Qwen-3B), but the **fabrication kill
fails**: the cave probe flags a never-knew (sincere-error) model as "hiding" 95% of the time. Mens rea is
therefore readable **only conditional on RUNG 1 first establishing the truth was held** — the HID section
of the certificate is not standalone.

## Setup

ARM B (deployable): LIE (knew-then-caved) vs RESISTED (knew-and-held) on the Qwen-3B `pc_3` syco-pass
residuals — *matched-knowledge by design* (both `neutral_correct==True` under identical sycophantic
pressure, so "knew it" cannot explain the split; only folding vs holding can). Pre-registered
(`PREREG_rung2_intent`, hash `22d25cd0…`). The lethal confound, caught by the red-team on disk: the
0.90 late-layer AUROC is **letter-token geometry** (LIE `chosen==asserted` is collinear with the emitted
letter) — it collapses cross-letter. So the layer is locked on **cross-letter held-out CV** and the
headline is read there.

## Result (Qwen-3B, cross-letter-locked layer 27)

| gate | meaning | value | pass |
|---|---|---|---|
| **G-XLETTER** | letter-invariant (train gold∈{A,B}, test {C,D}) | **0.709** | ✓ |
| G-SEP | matched AUROC ≥0.75 and > surface+0.10 | 0.777 vs 0.517 | ✓ |
| G-BEYOND | beats difficulty-only baseline by ≥0.15, CI>0 | **+0.260**, CI-lo 0.094 | ✓ |
| G-WITHINBIN | beats difficulty within each letter_margin tercile | 3/3 | ✓ |
| G-POS | instrument alive | 0.866 | ✓ |
| **G-FAB** | **never-knew NOT flagged as hiding** | **0.949** | ✗ **FAILS** |

## Honest read

- **Real (the folding-vs-holding direction):** among models that *knew* the answer, a **letter-invariant,
  difficulty-and-confidence-matched** direction separates *caving* from *holding* at ~0.78 — beating a
  strong difficulty baseline (0.52 matched) by 0.26, in all three difficulty terciles. This is the honest
  0.78, **not** the 0.90 (which is letter-token geometry, retracted).
- **The decisive failure:** point the same cave probe at a **never-knew (sincere-error)** model and it
  flags it as "hiding" at **0.949.** The cave direction cannot distinguish *folding a known truth* from
  *never having known it* — both emit a wrong answer under pressure. **A standalone "is it lying?"
  detector built on this would false-accuse sincere mistakes** — the worst failure mode a mens-rea
  instrument can have.
- **What this proves (the architectural finding):** RUNG 1 and RUNG 2 are **coupled.** You cannot read
  *intent* (did it hide?) without first establishing — via RUNG 1's validity gate (prime 0.93/0.99) —
  that *the truth was held.* The certificate's **HID section is only valid conditional on its HELD
  section.** Gated behind RUNG 1 ("it knew"), the cave direction is meaningful; alone, it conflates
  caving with error. The two-stage certificate (HELD → then HID) is the honest deployable form.

## Scope / not-claimed

Qwen-3B only; MMLU sycophancy-caving; commit-token; linear probe = a separating direction, not proven
intent. **Sycophancy-SPECIFIC** — a frozen cave probe reads *instructed* deliberate lying at 0.42 (below
chance), so this is **NOT** a universal knowingly-false detector. The deployable verbose-correct guard
(does it flag an honest verbose restatement?) needs free-form generation and is **OWED** (not in the
forced-choice substrate). **NOT claimed:** a standalone mens-rea detector. **Claimed (bounded):** a real
letter-invariant folding-vs-holding direction (~0.78) that becomes a mens-rea signal **only conditional on
the RUNG-1 held-state validity gate.** A failed fabrication gate, reported as loud as the five that passed.
Receipt: `intent_cert_result_qwen3b.json`, `run_intent_cert.py`.
