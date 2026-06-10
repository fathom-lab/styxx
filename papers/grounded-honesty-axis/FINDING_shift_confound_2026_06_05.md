# FINDING — the paired process-read is CONFOUNDED-BY-CONSTRUCTION with answer-shift (MCQ)

Resolves the open confound flagged in `FINDING_intent_discriminator` §9. Pre-registered bars locked in
`run_shift_confound_control.py` before computing; verdict reported as-computed, no post-hoc flip. White-box,
local, $0, reuses the saved paired residuals (no model load). Receipt: `shift_confound_control_result.json`.

## The open question (§9)

The paired process-read `D = syco_resid − neutral_resid` recovered the 7B cave signal (0.66 → 0.88) where the
single-snapshot linear read faded. But §9 flagged it: *"lies change the answer while mistakes often do not, so
`syco − neutral` partly reads 'the answer shifted,' not pure intent; a clean test must control for shift
magnitude before it counts."* This runs that control.

## Result — the confound is near-total, and the clean control is INFEASIBLE on MCQ

`shift = (neutral-committed letter ≠ syco-committed letter)`. Across the full Qwen2.5 ladder:

| model | lie shifts | mistake shifts | shift-only 1-bit AUROC | process-read AUROC | shift-matched n/class | verdict |
|---|---|---|---|---|---|---|
| 1.5B | 221 / 221 | 1 / 303 | **0.998** | 0.86 | 1 | CONFOUNDED-BY-CONSTRUCTION |
| 3B | 267 / 267 | 6 / 270 | **0.989** | 0.80 | 6 | CONFOUNDED-BY-CONSTRUCTION |
| 7B | 264 / 264 | 0 / 221 | **1.000** | 0.83 | 0 | CONFOUNDED-BY-CONSTRUCTION |

Two facts kill the clean test:

1. **A 1-bit "did the answer change?" feature scores AUROC 0.99–1.00** — at or above the fancy process-read
   (0.80–0.86 linear here; 0.88–0.94 for §9's MLP@pooled). The process-read does not beat the trivial shift
   bit. Whatever it reads, a single bit of behavior reads it better.
2. **The shift-matched subset is unpowered for any probe** (0–6 shifted-mistakes vs the pre-registered ≥25).
   You cannot hold shift fixed and vary intent, because in MCQ **caving *is* changing the committed letter** —
   lies shift 100% of the time, mistakes ~never. The two events are the same event.

Per the locked bar **B1 (power)**, the verdict is **CONFOUNDED-BY-CONSTRUCTION** on all three sizes: the
de-confounded intent-beyond-shift test the §9 caveat asked for **cannot be run on the MCQ paradigm**.

## What it bounds, and what still stands

- **Bounds §9.** The process-read's 7B "recovery" is not clean evidence of intent — it is inseparable from
  answer-shift on this data. Treat the §9 0.88 as shift-laden, not intent-pure.
- **Does NOT touch §2.** "Intent beyond *confidence*" (margin-bin-balanced, output pinned at chance, probe
  0.745 / contrast +0.239, cross-family) controls a *different* nuisance (confidence) and is unaffected. The
  strong claim of the arc rests on §2, not §9.
- **Points to the fix.** A shift-decoupled test needs a paradigm where caving can preserve the surface answer
  while changing internal commitment — the **free-form caves of §10** (TriviaQA pushback; the cave exists
  there at 0.864), or an MCQ design that grades shift magnitude instead of letting it collapse onto class.

## Honest scope

Single model family (Qwen2.5) on the saved paired sets; `shift` defined at the committed-letter granularity
(a finer logit-magnitude shift measure would not help — the binary already saturates). The process-read here
is a best-layer linear probe (0.80–0.86), weaker than §9's MLP@pooled (0.88–0.94); the confound conclusion is
robust to that choice because the 1-bit shift feature exceeds *both* and the matched control is unpowered for
*any* read. This is a methodological bound, not a claim about whether 7B intent is "really" decodable — §2 and
§10 say it is, by other routes.
