# Pre-Registration · Fame vs Truth — does the Council reach truth, or only consensus?

**Committed BEFORE data.** The Council PASS (`FINDING_council_2026_05_25.md`) was scoped
honestly to **consensus-vs-fabrication**: its "obscure-real" tier was consensus-known
(all frontier models have it). This test settles the word **truth**.

## The decisive question

Inter-model agreement separates real from fake — but *why*? Two hypotheses:
- **TRUTH:** a real fact is a shared attractor; models that know it converge on it
  *regardless of how famous it is*. → agreement on reals stays high even as rarity rises.
- **FAME:** agreement just tracks *how widely documented* a fact is. → agreement on reals
  *degrades* toward the fake level as facts get rarer.

Sweep a **rarity gradient of REAL facts** and watch which happens.

## Design (run once)

Council = `gpt-4o-mini`, `gpt-4o`, `gpt-3.5-turbo`, `gpt-4.1-mini` (≥3 required). N=3,
modal vote, prompt "Answer in one short sentence with a specific answer."

Rarity gradient (reliable labels — facts verifiable cold, to avoid mislabeling):
- **R0 common** — capital of France, symbol for gold …
- **R1 obscure** — 1938 Nobel Literature, *French Connection* director …
- **R2 ultra-rare** — capital of Burkina Faso (Ouagadougou), of Bhutan (Thimphu), of
  Kyrgyzstan (Bishkek), of Eritrea (Asmara), of Brunei (Bandar Seri Begawan); atomic
  number of einsteinium (99), of technetium (43); currency of Bhutan (ngultrum).
- **FAKE** — the validated nonexistent baits.

Metrics per item: **agreement** = largest substantive (non-abstention) cross-model
cluster ÷ council (judge clustering). For REAL items, also **correct-cluster**: do ≥2
models converge on the answer that matches the curated reference? (the *truth attractor*
amid scatter, even when not universal).

## Kill-gate (PASS iff T1 ∧ T2)

| ID | Bar |
|----|-----|
| **T1 (truth, not fame — decisive)** | agreement on **R2 ultra-rare-real** ≥ **0.70** AND (agreement R2 − agreement FAKE) ≥ **0.30**. Agreement must *survive high rarity* and stay well above fakes. |
| **T2 (mechanism)** | on R2 items, a **correct convergent cluster** (≥2 models agree on the *true* answer) exists in ≥ **75%** of items — knowers converge on truth even when knowledge isn't universal. |

**Reported regardless:** the **agreement-vs-rarity curve** (R0→R1→R2→FAKE). Flat-high =
truth; monotone decline into the fake band = fame.

**PASS** → the Council reaches truth, not just fame; *earn the word* and say so.
**FAIL shapes (all honest):** (a) agreement collapses on R2 toward fake → it's a
**fame/consensus** detector, full stop — the scoped Council finding stands, "truth"
retracted; (b) agreement holds (T1) but no correct-cluster (T2 fails) → models converge
on a *shared wrong* answer at high rarity (correlated confabulation on rare reals) →
agreement ≠ truth, a sharp negative; (c) partial — R2 lands between R1 and FAKE → a
*graded* signal, reported as such.

## Honest prior

Likely **partial**: frontier models (gpt-4o, 4.1-mini) know Ouagadougou/einsteinium;
gpt-3.5 may not → the council *splits* → R2 agreement lands maybe 0.5–0.75 (knowers agree,
others confabulate). If T2 holds (a correct ≥2 cluster persists) while T1 is marginal,
the honest read is "a truth attractor exists but is diluted by rarity" — truth *signal*
present, not a clean *threshold*. Deepest risk unchanged: all-OpenAI lineage; a shared
rare-fact blind spot would manufacture correlated confabulation (b). Do not call a graded
fame signal "truth"; report where R2 actually lands.
