# Pre-Registration · Justification-Divergence (JD) — agree-on-WHAT vs agree-on-WHY

**Committed BEFORE data.** Third swing at the dark matter, on a **different axis** from the
first two. Swing #1 (perturbation-fragility) was partial (recall 0.40/0.44, fragile shell
only). Swing #2 (CVPD, agreement-fracture) was a clean negative (AUC 0.52, worse than flip).
Both are *divergence-of-the-answer* methods, and both are blind to the stubborn myth core.
JD leaves the answer axis entirely. Pre-registered, run once — not a re-roll.

## The idea (genuinely novel: justification coherence, not answer divergence)

Every prior method asks whether the council **agrees on the answer**. On the dark core they
do — that's *why* it's dark. JD asks the orthogonal question: **in the regime where answer-
agreement is already pinned high (the floor), do the vendors agree on the WHY?**

> A true proposition is backed by **transferable structure** — a web of reasons the council
> re-derives the same way. A shared misconception is an **isolated memorized association**:
> everyone knows the answer, but there is no load-bearing mechanism, so when each vendor is
> asked *why*, they rationalize **differently** (or vacuously). Justification-divergence can
> therefore separate fact from misconception **even when the answer is identically agreed** —
> the regime where divergence, confidence, perturbation-fragility, and agreement-fracture all
> go blind. (Inspiration: analogical-reasoning work showing fact-knowledge carries
> transferable structure that memorized associations do not.)

## Design (run once, scaled)

- Council: gpt-4o-mini (OpenAI) + Qwen2.5-3B (Alibaba) + gemma-2-2b-it (Google).
- Process TruthfulQA candidates + clean truths to get **n ≥ 25 consensus-misconceptions**
  (answer-agreement ≥ 0.66 ON a wrong answer) + **n ≥ 25 consensus-truths**. Identical floor
  regime for both classes (both already agreed on the answer).
- **Primary signal — justification-divergence:** for each item, ask each vendor for the
  *reason* behind the consensus answer (one sentence, neutral phrasing). Cluster the three
  justifications by core-mechanism (the validated judge-clustering). JD = 1 − (largest
  justification-cluster / 3). Truth → low JD (shared mechanism); misconception → high JD.
- **Stubborn-subset labeling:** also run the neutral "reconsider" challenge (swing #1's
  perturbation) to tag each misconception flip vs **no-flip (stubborn)**. This lets J2 ask
  whether JD sees the *dangerous core* that fragility/fracture cannot.

## Kill-gate (PASS iff J1 ∧ J2)

| ID | Bar |
|----|-----|
| **J1 (the floor-regime signal)** | AUC(JD → misconception) ≥ **0.75** on the consensus pool — separates fact from misconception where the answer is equally agreed (beats answer-fragility 0.70). |
| **J2 (the novel win — sees the dark core)** | AUC(JD → misconception) on the **no-flip / stubborn** misconception subset (vs truths) ≥ **0.70** — i.e., JD detects the stubborn myth-core items that perturbation-fragility and agreement-fracture both miss. |

**PASS (J1 ∧ J2)** → JD is a genuinely new reference-free signal that reads into the dark
core: the council betrays a shared misconception by *failing to agree on why*, even while
agreeing on what. The truly novel result.
**FAIL via J2 only (J1 passes)** → JD is a (possibly better) detector of the *fragile shell*
but is **also** blind to the stubborn myth-core → a **third independent method confirms the
floor**: the dangerous core is dark to divergence of *any* kind (answer or justification).
**FAIL via J1** → the justification axis carries no reference-free signal.

## Honest prior

Most likely: **partial** — JD catches the factual misconceptions (the council's rationales
for a wrong fact diverge), but the myth class (astrology, folklore, "lucky charm") arrives
with a *stable cultural explanation* that also converges, so the stubborn core stays dark and
J2 fails. A clean PASS that reads the dark core is, honestly, **<30%**. The most valuable
non-PASS outcome is a *sharper* J1 on the shell (AUC > 0.70 where the answer-signal sat at
the bar) plus a third confirmation that the myth core is fundamentally divergence-dark. n is
modest; report straight. No re-rolling: this gate runs once.
