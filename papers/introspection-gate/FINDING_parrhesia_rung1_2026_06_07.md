# FINDING — PARRHESIA RUNG 1: the injected-concept read cannot certify a *thought* (apparatus bind)

**2026-06-07. Fathom Lab / styxx.** The first climb of RUNG 1 (the read-certificate of proof-carrying
cognition), at full pre-registered rigor. Verdict: **UNINFORMATIVE_APPARATUS / REPORT_AS_LANDED — the
injected-concept apparatus cannot validate the read-certificate, because it cannot separate a
behaviourally-live thought from the raw injected linear trace. This disciplines the program's flagship
"inaccessible thought" framing.** No cell was scored; the preconditions refused the apparatus first.

## Why this run

PARRHESIA's pre-registration (`PREREG_parrhesia_2026_06_07.md`, hardened by a 5-agent design +
adversarial red-team) mandates validating preconditions BEFORE scoring any cell. The red-team flagged
two confounds in the shipped flagship probe: (1) it is **leaky** (StratifiedKFold over all samples, no
holdout → partly re-detects the experimenter's own injected vector); (2) its clean-read == at-injection
read == 1.00 with `word_inject` (0.354) > `inject` (0.208) — the signature of reading the injected
vector's **lens shadow**, not a genuinely processed state. So the pre-reg added P5, a **divergence
gate**: the read is interpretable only if `acc(at-injection) − acc(clean-position) ≥ 0.10` (the direct
vector must add something beyond the propagated content) with the clean read still above chance.

## Result — the fundamental bind

Non-leaky read: train the probe on each concept's direction built from one template half, test on the
**other** half (mean cos 0.698 — genuinely different vectors), so it cannot memorize the exact injected
vector. Dose sweep on Qwen2.5-1.5B (the one steering-validated model):

| alpha | steering (full-tpl) | probe clean-pos | probe at-inj | divergence |
|---|---|---|---|---|
| 1 | **+0.007 (inert)** | **1.00** | 1.00 | 0.00 |
| 2 | +0.005 | 1.00 | 1.00 | 0.00 |
| 4 | +0.037 | 1.00 | 1.00 | 0.00 |
| 6 | +0.035 | 1.00 | 1.00 | 0.00 |
| 10 | +0.158 (live) | 1.00 | 1.00 | 0.00 |
| 16 | +0.167 (live) | 1.00 | 1.00 | 0.00 |

- **The probe reads the concept at 1.00 at EVERY dose — including α=1, where the injection is
  behaviourally dead (steering +0.007, the model ignores it entirely).** A linear probe detecting a
  linearly-injected direction succeeds at any dose; it is reading *"a concept-direction was injected,"*
  not *"a live thought is held."*
- **Divergence is exactly 0.00 at every dose.** The clean-position read never differs from the
  at-injection read, so the propagated content is the same linear trace, not a distinct re-representation.
  P5 fails by saturation at every dose; there is **no regime** where the read reflects a live thought
  separably from the raw trace.
- The held-out-direction generalization (1.00 at cos 0.698) rules out *exact-vector* memorization but
  not *trace*-reading: concept directions are template-robust, so the probe finds concept-c geometry
  whenever any c-ish vector is added — live or inert.

## Honest read — what this tempers, and what stands

- **TEMPERED (the flagship):** *"an external probe reads an inaccessible THOUGHT at 1.00 while the mind
  is blind to it."* The word **thought** is not supported by the injected-concept apparatus. At α=1 there
  is no thought — the model behaviourally ignores the injection — yet the probe still reads 1.00. The
  honest, narrower claim: *an injected linear concept-direction is externally decodable (whether or not
  the model entertains it) while the model cannot forced-choose it.* The probe side is partly tautological
  (linear probe ↔ linear injection); the read does not certify a held thought.
- **STANDS:** self-report is at chance for injected concepts (the model genuinely cannot forced-choose
  them — `P1` ARM-A 0.094, robust). The injected direction does propagate to a clean downstream position
  and is decodable. These are real; the *interpretation* as "an inaccessible thought" was the overclaim.
- **IMPLICATION for the north star:** RUNG 1 (the read-certificate) cannot be validated on an
  **injected** substrate, because injection conflates the trace with the thought. The certificate must be
  tested on **naturally-present** internal content — the suppressed-knowledge regime (knows-then-caves,
  DELTA 0.29–0.38), where the content is behaviourally real and not linearly planted by the experimenter.
  Proof-carrying cognition is not killed; its first apparatus was the wrong one, and the discipline
  caught it before a cell was scored.

## Discipline note

The pre-registration's order of operations (validate preconditions before scoring cells) is exactly what
prevented a fake win: the leaky-probe 1.00 would have produced a glamorous CELL-1 "trust beats
self-report" result that was reading the injected trace. The apparatus refused itself first. This is the
oath keeping itself — turned on the program's own flagship, and obeyed. Receipts:
`parrhesia_preconditions_qwen15.json`, `parrhesia_dose_sweep.json`, `run_parrhesia.py`,
`run_parrhesia_dose.py`.

## Scope / owed

Qwen-1.5B, injected linear concepts, one read layer (0.85 depth), 8-concept toy inventory, MiniLM
steering metric. The owed RUNG-1 climb: re-run the read-certificate on **naturally-present** content
(suppressed-knowledge / a task the model genuinely computes), where a behaviourally-live held state
exists independent of any injected trace, so "reads what the mind holds" can be tested without the
injection conflate. Until then, RUNG 1 is **UNINFORMATIVE on the injected apparatus** — honestly.
