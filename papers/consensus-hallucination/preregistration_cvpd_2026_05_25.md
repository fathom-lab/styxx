# Pre-Registration · Cross-Vendor Perturbation-Divergence (CVPD) — the sharper swing at the dark matter

**Committed BEFORE data.** The first dark-matter swing (single-model perturbation-fragility)
was a *marginal* pass: AUC 0.70, recall **40%**, missing the stubborn misconceptions. This
is the sharper mechanism, pre-registered, run once — not a re-roll.

## The idea (genuinely novel: cross-vendor × perturbation)

Single-model fragility only fires if *that* model flips. CVPD measures whether the
**cross-vendor consensus survives a neutral "reconsider" challenge.** A real fact stays
convergent under reflection; a shared misconception, challenged, **fractures** — vendors
flip to *different* places, re-introducing the divergence the static agreement hid. This
can catch the **stubborn** misconceptions the first swing missed (no single model needs to
fully flip — the council just has to crack).

## Design (run once, scaled)

- Council: gpt-4o-mini (OpenAI) + Qwen2.5-3B (Alibaba) + gemma-2-2b-it (Google).
- Process enough TruthfulQA candidates to get **n ≥ 25 consensus-misconceptions** (high
  baseline cross-vendor agreement ON a wrong answer) + **n ≥ 25 consensus-truths**.
- **Signal:** Δagreement = baseline cross-vendor agreement − post-challenge agreement
  (judge clustering). Truth → Δ≈0 (stays converged); misconception → Δ>0 (fractures).
- Baseline comparison: the first swing's single-model flip-rate (recall 0.40, AUC 0.70).

## Kill-gate (PASS iff C1 ∧ C2)

| ID | Bar |
|----|-----|
| **C1 (sharper)** | AUC(Δagreement → misconception) ≥ **0.75** — beats single-model fragility (0.70). |
| **C2 (higher recall, the novel win)** | fracture-detected recall on misconceptions > **0.40** (beats single-model flip) at low truth-fracture (truth mean Δagreement ≤ 0.15). |

**PASS** → CVPD is a genuinely better, novel reference-free consensus-hallucination
detector — cross-vendor fracture under perturbation catches what single-model fragility
and static agreement both miss. **FAIL** → single-model fragility ≈ CVPD (no lift), OR the
shared-stubborn misconceptions stay converged even under challenge → **a deeper, proven
floor**: the truly-shared-stable misconceptions are undetectable reference-free by any
divergence-based method. Either is a real, sharper result than the candle.

## Honest prior

Plausible CVPD lifts recall (partial fractures count, not just full flips) → the novel
win. Equally plausible the *most dangerous* misconceptions are shared AND stable across all
three vendors (they all hold, in unison, even when challenged) → no fracture → the deepest
floor. n still modest; report straight. No re-rolling: this gate runs once.
