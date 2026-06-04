# The readout principle: every honesty detector's blind spot is the dimension it doesn't read

**styxx / fathom-lab · 2026-06-04 · the synthesis of the consistency-robustness arc**

A day of attacking honesty detectors — and watching each ambitious thesis get refuted by its own
follow-up — converges on one principle that *did* survive every test:

> **A white-box honesty detector reads one projection of "is this output true." Its attack surface is
> exactly the orthogonal dimension it does not read. An adversary attacks the blind spot. No single
> readout is robust; only a set covering orthogonal projections is — down to an irreducible floor.**

## The detectors, their readouts, and the attack each one falls to (all from this arc)

| detector | reads | blind to | falls to | caught instead by |
|---|---|---|---|---|
| **activation probe** | the deception *representation* | a model that *believes* its lie (honest-looking activation) | heavy RL-obfuscation (literature) — **resisted every feasible attack I ran** (per-item, steering, LoRA) | grounded_honesty (belief intact) |
| **grounded_honesty** | the *stateless resampled belief* | belief manipulation | cheap question-framing — **cracked 18%** | canonical resampling + injection detection (→1.7%) |
| **detect_context_injection** | *cross-context* divergence (as-asked vs canonical) | a belief *stable across contexts* | a deep, context-independent false belief | the council (peers disagree) |
| **council_agreement** | *cross-model* agreement | a *shared* false belief (correlated confabulation) | a globally-shared misconception | nothing self/peer-based — needs ground truth |
| **derivation-consistency** | what the model *derives* (CoT) | *retrieved* knowledge (nothing to derive → collapses to retrieval); a belief stable across reasoning paths | a knowledge fact, or a robust mis-derivation | external ground truth |

Each row is a measured result from this session, not a hypothesis. The retracted thesis was
"consistency > signature"; the data says the *signature* was the more robust of the two against my
feasible attacks. The honest replacement is the table: **orthogonal surfaces, no winner.**

## Why defense-in-depth works (and the two measured cross-checks)
The attacks are orthogonal, so each defeats one readout while leaving another intact:
- **RL-obfuscation fools the probe → grounded_honesty still catches** (stateless belief 96%, AUROC 0.966
  after the attack). *Measured.*
- **Framing fools grounded(as-asked) → injection detection catches** (91% of the fooled items flagged as
  cross-context divergence). *Measured.*
To defeat the *stack*, an adversary must satisfy **every** readout at once: a lie that is stable under
resampling **and** across contexts **and** across models.

## The irreducible floor
That conjunction has a name: a **globally-consistent false belief** — one the model holds stably, in
every framing, and that its peers share. By construction, **no consistency- or agreement-based method
can distinguish it from genuine knowledge** — they all measure self/peer coherence, and it is coherent.
Only an *external* anchor (retrieval, verification, a ground-truth oracle) breaks it. The council arm
measures an **upper bound** on this floor (the shared-confabulation rate); it shrinks with model
capability and *vendor* diversity (memory: cross-vendor council 0.917 beat same-vendor by breaking
correlated confabulation) but does not vanish — it is the epistemological limit of consistency-based
honesty monitoring.

**Measured here (Qwen-3B + Gemma-2b + Qwen-1.5B, 100 MMLU items): the council came out WEAK** —
gold-vs-lie AUROC **0.63**, single-model-error coverage **26%**, shared-confabulation rate **55%**.
But only **9% of items had all three models correct**, so that 55% is almost entirely **shared
ignorance** of weak, partly-correlated members (two are same-vendor), not a clean false-belief floor —
an upper bound dominated by the members' weakness. It makes the framework's own claim sharply: **the
council layer is only as strong as its members are capable and vendor-diverse.** A frontier,
vendor-diverse council (memory: cross-vendor 0.917) would lift coverage and shrink the apparent floor;
I could not max that out on an 8 GB local cache. The floor *exists*; pinning its true value needs
better members or an external truth anchor — honestly, future work. (`council_floor_result.json`.)

**The floor decomposes (derivation arm).** A *derivation* readout (chain-of-thought) shows the floor is
not monolithic: re-derivation breaks the **derivable** sub-floor (math/logic — styxx's path-diverse moved
grounded AUC 0.69→0.96) but is **silent on retrieved knowledge** (n=100: direct 0.64 ≈ CoT 0.63,
disagreement at chance, reasoning fixes only 11% of errors). So the **knowledge sub-floor (~32% here) is
the genuinely irreducible residual** — no self / peer / derivation method touches it; only an external
truth anchor does. (`RESULT_derivation_consistency`.)

## Implication for the RSI era
When AI builds AI, the honesty monitor that matters is not a single unbreakable signal — there isn't
one — but a **layered readout** whose orthogonal surfaces force any deception into the open, plus the
explicit admission that the residual floor requires **grounding in external truth**, not more
self-consistency. styxx already ships the layers (probe, grounded_honesty, detect_context_injection,
council_agreement); this arc's contribution is the **map of what each one cannot see**, and the proof
that they cover each other — to a floor we should measure and name, not pretend away.

## Provenance (self-falsified)
Built by attacking our own tools and retracting what didn't survive: the "structural immunity" overclaim
(corrected), the cheap-probe-collapse claim (corrected — probe robust), "consistency > signature"
(retracted). Commits `70c2cbc → ` this. Owed: a vendor-diverse, frontier-capable council to tighten the
floor; in-vivo attack vectors (multi-turn, tool-output poisoning).
