# CORRECTIONS — due-diligence audit of the ancient-question program

**Date:** 2026-06-03. An independent adversarial peer review (red-team) of this program found
real overclaims and one reproducibility/discipline violation. **The review was largely correct.**
This document records each valid finding and the correction, and re-scopes the headline claims.
Per our own stated rule — *publish the losses next to the wins* — the corrections live here, in
the open, next to the results.

## The one that mattered most — a discipline violation (review C1/C7)

**Finding:** the geometry "scaling" writeup claimed unsupervised correspondence **recovery → up
to 1.00, "reproduces vec2vec"** for distinctive geometry — but (a) that number came from an
**inline run with no committed script**, and (b) it **overrode the committed, pre-registered
result of that experiment**, which had returned **PLATEAU (recovery max 0.10, failing the
frozen ≥0.50 gate)** on isotropic z. That is the *inverse* of our own discipline: a tool-limited
*null* was correctly discarded earlier (GW→RSA), but here a gate-**failing** result was discarded
in favor of an **unverifiable positive**. Inexcusable, and exactly the failure mode the program
claims to oppose.

**Correction:**
- The pre-registered geometry-scaling recovery result is **PLATEAU — recovery FAILED its gate.**
  That stands, reported plainly.
- The distinctive-geometry recovery follow-up is now a **committed, reproducible script**
  (`run_geometry_recovery.py`, 5 seeds, with a different-z control) and is labeled **EXPLORATORY
  / post-hoc, not pre-registered.** Its committed reading: **isotropic recovery fails entirely**
  (max-mean **0.12**, confirming the registered PLATEAU); **distinctive recovery is real and
  reproducible — mean 0.87 at faith 0.99** (4/5 seeds 0.89–1.0; control at chance ~0.01) — **but
  it is a sharp threshold** (collapses to 0.25 at faith 0.98, chance below 0.96). So recovery
  genuinely requires **distinctiveness + very high faithfulness**. The finding survives; the
  **"robustly 1.00 / reproduces vec2vec" gloss is retracted** in favor of this threshold-y,
  exploratory, properly-controlled version, and the pre-registered isotropic **failure is
  reported plainly** (discipline applied symmetrically).

## Geometry "universal forms" is overscoped (review C2/C3)

**Finding:** the experiment passes the **same latent z to both worlds** (`run_disjoint_worlds_rsa.py`),
then measures geometric similarity **using the known concept correspondence** (RSA). So the result
largely reduces to: *skip-gram is a consistent estimator of a shared generative metric.* Calling
this "the geometry of meaning is universal / Plato's forms, vindicated" is overreach on what is,
by construction, a consistency property — and "zero shared data → universal" conflates RSA
(known-correspondence similarity) with the strong unsupervised-translation claim (which **failed**).

**Correction — re-scoped claim:** *Two metric-learners trained on disjoint corpora sampled from a
**shared** latent metric recover **correlated** geometry (RSA, known correspondence), and the
correlation scales smoothly with faithfulness.* This is a controlled illustration of the
convergence **mechanism** the literature observes in real models (vec2vec/CKA) — **not** independent
evidence that meaning has substrate-independent universal form (the shared structure was built in),
and **not** unsupervised translation (that failed its gate). "Universal forms vindicated" is
**downgraded to "a controlled illustration of structure-driven convergence."**

## Rhythm over-generalizes (review C5/C6)

**Finding:** "rhythm is fundamental to *recurrent cognition*" rests on **one task (ordered copy),
one architecture (a linear LRU), n=3**, with visibly noisy/non-monotonic accuracy curves; the
larger-d CLAMPED control promised in the prereg was **not run**; the LRU↔theta-gamma link is
**analogized, not demonstrated**; and the 7±2 coincidence is disclaimed yet repeatedly invoked.

**Correction — re-scoped claim:** *In a complex-diagonal linear-recurrence unit on ordered copy,
eigenvalue rotation (oscillation) roughly doubles the 0.80-capacity at matched parameters.* That
is the defensible claim. The leap to "recurrent cognition" requires ≥3 architectures × ≥3 tasks +
more seeds + the larger-d control — **not yet run.** The **7±2 mention is cut** to a single labeled
"do-not-claim; would require the unrun D-and-threshold sweep." No brain claim is made.

## Framing overruns evidence (review C4/C8)

**Finding:** "made a 2,500-year question testable / universal forms finally falsifiable" is
grandiose over two synthetic toys; the Pythagoras→Berger lineage is intellectual history that adds
gravitas, not rigor; construct validity (do these toys measure "universal forms" / "is rhythm
necessary"?) is asserted, not argued.

**Correction:** the program is **re-titled in spirit to "in-silico mechanism probes of
representational convergence and oscillation-as-mechanism, motivated by the ancient question."**
The lineage is **motivation, not result.** Every philosophical term (universal forms, cognition)
is to be read at the **operational level** (distance-matrix correlation; ordered-copy capacity).

## Numerical fixes (review C9)
- "RSA ≈ 0.46" → the committed mean is **0.42** (the 0.46 was a single diagnostic seed).
- isotropic "recovery 0.15 at faith 0.99" → committed value is **0.10 at faith 0.99**.
- The geometry-scaling RESULT table mixed an isotropic committed sweep with an uncommitted
  distinctive run; the two are now **separated and labeled**, both reproducible.

## What survives the audit (the durable contributions)
- **The rhythm phase-clamp** (scoped to LRU/copy) — a clean, matched-confound ablation; honest gate.
- **The concept-dynamics structural backbone** — Mamba-1 all-real eigenvalues (verified on weights),
  transformer 0/20 oscillation, validated positive control. FACT/HYPOTHESIS tagged.
- **The geometry convergence MECHANISM** (scoped to "consistency of metric estimation, scaling with
  faithfulness") — modest and real.
- **The methodology** — pre-registration, positive controls, and — demonstrated by *this very
  document* — the willingness to let an independent audit retract our own headline.

**Verdict:** two scoped ML results + an exemplary method survive; the cosmic framing and the
"recovery reproduces vec2vec / universal forms vindicated" headlines do not, and are retracted/
downgraded here. The audit working *on us* is the point — not a footnote to it.
