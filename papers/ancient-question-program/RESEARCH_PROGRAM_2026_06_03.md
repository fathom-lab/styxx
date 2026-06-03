# The Ancient Question, Made Testable
## A research program: from Pythagoras to two decisive, runnable experiments

**2026-06-03 · fathom-lab / styxx.** Epistemic tags throughout: **[HIST]** history of
ideas · **[SCI]** established science · **[SPEC]** live/open. Built from three
citation-grounded deep-research passes + our own committed results.

---

## 0. What this document claims

The ~2,500-year question — *does a universal structure (number, frequency, geometry)
underlie mind and meaning?* — was never crackable because **you could not open a mind and
read it.** Now you can: an LLM is the first fully-readable mind. This program (1) traces
the question to its single legitimate modern form, (2) shows it splits into **two**
falsifiable sub-questions, each with a **decisive, pre-registerable, in-silico-runnable**
experiment, and (3) places our own results as the first brick in each. **This is not a
claim to have solved it — it is the conversion of a 2,500-year argument into an experiment.**

---

## 1. The lineage, and the one pattern through it [HIST]

Pythagoras (measured string-ratios — octave 2:1, fifth 3:2 — *true*) → "harmony of the
spheres" (*dead*). Plato (nature is mathematical — a *correct* methodological intuition) →
the World-Soul tuned to musical ratios (*dead*). **Aristoxenus** (the *ear/data* constrains
the ratio — the empirical turn). **Kepler** (chased the literal harmony of the spheres in
Tycho's data → found the **Third Law a³/T²**, real physics forever; the harmonics
evaporated). Galileo/Mersenne (pitch = measurable *frequency*; Mersenne's laws, exact).
Newton (split light into **seven** colours by *musical analogy* — indigo inserted to force
the fit: the harmonic prior **corrupting** an observation, a cautionary artifact). **Fourier**
(frequency decomposition becomes universal — but a **representation theorem**, a choice of
basis, *not an ontology*). **Helmholtz** (frequency → the **physiology of the ear**; first
real bridge to the nervous system). **Berger** (recorded the **EEG in 1924** — while hunting
*telepathy*; never found telepathy, found **brain rhythms**).

**The pattern, stated once:** at every step a *mystical motivation* fused to *disciplined
measurement* — the mysticism dies, the measurement endures. The fault line that separates
kernel from woo is always the same: a **measured ratio** (an empirical regularity) vs. an
**imposed ratio** (a pattern projected onto nature). "Everything is vibration" is the
category error of mistaking Fourier's representation theorem for an ontology; the famous
Tesla "energy / frequency / vibration" line is **misattributed** 1980s New Age; and a 2024
*Nature Communications* study (~4,000 listeners) **falsified** Pythagoras's universal-
consonance claim outright [SCI]. **This fault line is our methodology.**

**The convergent question** the whole arc lands on [SPEC, on established ground]: *is the
structure we find in minds — geometric and dynamical — **constitutive** of cognition and
meaning, or an **artifact/epiphenomenon** of substrate and data?* It splits cleanly in two.

---

## 2. Sub-question A — the GEOMETRY of meaning: universal, or shared-data artifact?

The modern form of Plato's "universal forms." Three nested claims, usually conflated:
**U1** different AI models converge in representational geometry · **U2** that geometry is
shared across substrates (artificial↔biological) · **U3** the shared structure is
*meaning/reality*, not an artifact of shared data/biases or the similarity metric.

**Status.**
- **U1 — established.** Model stitching: nets with *different objectives* stitch with ~0
  loss (Lenc & Vedaldi 2015; Bansal & Nanda, NeurIPS 2021). Linear CKA recovers cross-model
  correspondence (Kornblith 2019). Universal neurons ~1–5% across GPT-2 seeds (Gurnee 2024);
  Rosetta neurons across architectures/modalities (Dravid, ICCV 2023); universal features
  under superposition (Elhage 2022).
- **Geometry shared *up to transformation* (within modality) — established.** Relative
  representations (Moschella, ICLR 2023). **vec2vec (Jha/Morris/Shmatikov 2025):**
  unsupervised translation between **6 different embedding spaces with ZERO paired data**,
  cosine ≤ **0.92**, top-1 ≤ **100%** — the strongest universality result to date.
- **U3 — contested / the crux.** Is it *meaning*, or *shared training distribution + biases*?
  AI↔brain alignment is real-but-confounded: Schrimpf 2021 reports ~100% of a **0.32** noise
  ceiling, but Feghhi/Antonello (2024→Nat. Commun. 2026) show **56–82%** of that is a trivial
  temporal-autocorrelation model and **98–100%** of *untrained*-model predictivity is
  position+length. The Platonic Representation Hypothesis's own peak cross-modal alignment is
  **0.16/1.0** (authors' caveat).

**Our brick — and the caveat the frontier exposes.** Our convergence study: refusal geometry
agrees across **6 model families** (R_within 0.70; cross-concept null −0.17; Δ **+0.87**),
corrigibility does **not**. That's a clean **U1** brick. *But it used the same prompts across
models* — so it is subject to exactly the **shared-data confound**. It is not yet U3. The
frontier experiment is how we'd cross that line.

**DECISIVE EXPERIMENT A — "Disjoint-Worlds vec2vec."** Two synthetic worlds with *identical
latent causal structure* but *provably disjoint surfaces* (vocabulary, renderer, tokens —
**zero shared training data** by construction). Train ≥2 models per world (different
architectures/objectives). Attempt **unsupervised** cross-world translation (vec2vec /
relative-representations), **no paired examples**. Pre-register the kill-gate: cross-world
top-1 concept-matching beats the shuffled-anchor null by a fixed margin **and rises with
scale**. Causal arm: a steering direction mapped through the learned translation must steer
the *same* concept cross-world and **fail** to steer *separable* concepts (Park–Veitch
"causal inner product" orthogonality, ICML 2024). **Decision:** success → a substrate- and
**data-independent geometry of meaning** (universal forms, vindicated in the only testable
sense). Failure *despite same-world vec2vec succeeding* → the observed universality is a
**shared-data artifact** (the platonic reading falsified down to "shared-distribution
geometry"). Removes every confound that currently bounds the field at once; runnable today,
entirely inside readable artificial minds.

---

## 3. Sub-question B — the DYNAMICS (rhythm): necessary, or substrate-specific?

The modern, demythologized form of Berger's question.

**Status [SCI / contested].** "**Necessary-in-tissue, not necessary-in-principle.**" Every
function oscillation is credited with has a proven non-oscillatory implementation: trained
RNNs solve memory & routing with **fixed points and line attractors, not oscillation**
(Sussillo & Barak 2013); **transformers** do binding (slots/tensor-products), routing
(*attention = content-addressed dynamic routing — the exact job CTC ascribes to gamma
coherence*), and pointer-memory (Wu et al. 2025) **with no clock**; oscillatory SSMs (LinOSS,
ICLR 2025) are *competitive but not unique* vs. non-oscillatory S4/LRU/Mamba. Gamma's causal
status is contested (Ray & Maunsell: gamma frequency drifts with stimulus contrast → it
can't be the stable binding clock; may be an **E/I readout**). Binding-by-synchrony is
largely **falsified for vision** (Shadlen & Movshon 1999; Thiele & Stoner 2003). The 7±2
working-memory bound may be a **fingerprint of the oscillatory substrate**, not a law of
cognition — transformers don't have it.

**Our brick.** The rhythm program: transformer concept-signal **commits** (0/20 oscillation);
Mamba-1 **cannot oscillate** (all 3.1M state eigenvalues real-negative, verified on the
weights); complex-recurrence toy **oscillates at the true frequency** (positive control). ⇒
oscillation capacity *is* complex recurrence; **rhythm is substrate-specific mechanism, not
function.** A brick on the "achievable without rhythm" side.

**DECISIVE EXPERIMENT B — "Phase-clamp ablation with rescue."** Take a recurrent system doing
binding + routing + capacity-limited memory; **suppress its oscillation** (spectral
regularization killing the complex-eigenvalue/Hopf modes, or a theta–gamma band-power
penalty) **while holding mean rate and E/I ratio fixed** (the control the critics demand and
existing causal studies lack); then test whether non-oscillatory mechanisms
(attractors/attention) can **rescue** the function *at matched resources*. **Decision:** no
rescue possible → rhythm is **necessary**; rescue succeeds (perhaps losing the capacity
bound) → rhythm is a **substrate-specific mechanism** and 7±2 is its fingerprint. The
in-silico arm is **runnable now** — the spectral-suppression knob is essentially the analysis
we already built, and LinOSS/SSM tooling supplies the add-oscillation control.

---

## 4. The unifying frame

Both sub-questions are one: **are there universal, substrate-independent invariants of mind —
in what it represents (geometry) and how it processes (dynamics) — or are these
substrate-and-data-specific?** That is the ancient "universal forms" intuition, finally
falsifiable. Two decisive experiments, both pre-registerable, both runnable inside readable
artificial minds, **neither blocked** by the unavailable content-complex SSM. We have laid
the first brick in each.

---

## 5. Why this is not woo (the through-line)

The same discipline that killed the mysticism for 2,500 years is this program's method:
**measured, not imposed; pre-register the kill-gate before the data; validate the instrument
before trusting it; publish the losses next to the wins.** Berger chased telepathy and was
honest enough to keep only the EEG. We chased vibrations and kept only what survived the
gates. We do not claim to crack the ancient question — we convert it into two experiments
that can tell us *no*, and then we run them.

---

## 6. Status & next move

- **Experiment B (rhythm rescue), in-silico arm — RUN 2026-06-03.** Pre-registered (gate
  frozen before data). Reading: **ADVANTAGE** — phase-clamping the eigenvalues (removing only
  the oscillation) roughly **halves** ordered-memory capacity (kcap 6.0 → 2.67) and the free
  net *keeps* its oscillation (osc_use 0.62), yet the no-rhythm net still partially solves the
  task → **rhythm is a capacity-extending MECHANISM, not a hard requirement.** This refines
  "mechanism not function" into the honest middle the literature occupies
  ("necessary-in-tissue, not in-principle"), now shown in a clean ablation.
  `papers/rhythm-rescue/RESULT_rhythm_rescue_2026_06_03.md`. Next B-arms: binding/routing
  (transformers do these without rhythm) and capacity-vs-d scaling (is 7±2 an oscillatory
  fingerprint?).
- **Experiment A (Disjoint-Worlds vec2vec)** = the geometry-half decider; a larger but fully
  in-silico build (synthetic worlds + training), no blocked dependency. **The next major bet.**
- **One half of the 2,500-year question now has its first decisive, pre-registered result.**
  The geometry half is the next build.

---

### Selected sources (verified across the three passes)
Lineage: SEP *Philolaus*; Kepler *Harmonice Mundi*; Mersenne's laws; Fourier 1822; Helmholtz
*Sensations of Tone* 1863; Berger 1929; Marjieh et al. *Nat. Commun.* 2024 (consonance not
universal). Geometry: Huh/Cheung/Wang/Isola ICML 2024 (Platonic RH); Park/Choe/Veitch ICML
2024 (linear rep. / causal inner product); Moschella ICLR 2023; Jha/Morris/Shmatikov 2025
(vec2vec); Kornblith 2019; Bansal/Nanda NeurIPS 2021; Gurnee 2024; Dravid ICCV 2023; Elhage
2022; Feghhi/Antonello Nat. Commun. 2026; Schrimpf PNAS 2021; Caucheteux & King 2022. Rhythm:
Fries 2005/2015 (CTC); Lisman & Jensen 2013 (theta-gamma); Singer & Gray 1995; Shadlen &
Movshon 1999; Ray & Maunsell 2010/2015; Sussillo & Barak 2013; Pals/Macke/Barak 2024; Rusch &
Rus LinOSS ICLR 2025; Greff/van Steenkiste/Schmidhuber 2020; Wu et al. 2025.
