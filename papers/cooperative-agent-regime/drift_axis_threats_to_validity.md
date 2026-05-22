# Drift-Axis Positive: Threats-to-Validity Pre-Mortem

**Status:** companion to the drift-axis POSITIVE deposit (`fa24373`). EXPLORATORY analysis — not preregistered. The binding control is the preregistered topic-overlap 2×2 (`topic_control_preregistration_2026_05_22.md`).
**Purpose:** enumerate every way the drift-axis cooperation result could be an artifact, ranked, with the control for each marked done / available / pending. A positive is only as strong as the list of ways it could be wrong that you've actually checked.

---

## The finding under audit

Cooperative-agent dyads show higher inter-agent embedding-trajectory alignment (DAA) than non-cooperative dyads. Preregistered (`47f9bdc`), N=20+20, both embedding providers, p < 0.001, Δ +0.46 (openai) / +0.37 (bge), collector≠scorer verified.

The honest one-line claim: **"cooperative-regime dialogues show higher inter-agent embedding-trajectory alignment than adversarial ones."** NOT "cognitive coupling detected." Every threat below is a way that even the modest claim could fail.

---

## Threats, ranked by how much they'd undermine the claim

### T1 — Directional topic convergence *(HIGHEST; control PENDING)*
Cooperative agents converge on a shared evolving solution → both trajectories point the same way in latent space. The alignment could be topic-convergence, not cooperation.
- **Control:** the preregistered 2×2 (regime × topic-coupling). **PENDING the operator's signature.** This is the load-bearing control; nothing below substitutes for it.
- **Partial pre-emptive evidence (R3, exploratory):** *instantaneous* topic-overlap (turn-by-turn content-word cosine) is essentially equal across regimes (0.795 vs 0.808, Δ −0.013) while DAA differs by ~0.46. So the *instantaneous* form of the confound is ruled out on existing data. **The *directional* form is NOT** — that is exactly what the 2×2 tests. T1 remains open until the 2×2 runs.

### T2 — Arbitrary half-split in DAA *(control DONE)*
DAA splits the trajectory at `n//2`; the result could hinge on that choice.
- **Control (R1, exploratory):** the coop>noncoop gap holds and *grows* across split fractions 0.33 / 0.50 / 0.67 (Δ +0.39 / +0.46 / +0.49). Not a split artifact. **Closed.**

### T3 — Verbosity / length confound *(control DONE)*
The cooperative overlay might just produce longer or more uniform responses, mechanically inflating trajectory alignment.
- **Control (R2, exploratory):** non-cooperative responses are *longer* (1970 vs 1542 chars) and *more uniform* (cv 0.13 vs 0.21). The high-DAA condition has the shorter, more variable text — length works *against* the finding, not for it. **Closed.**

### T4 — Same-vendor agents *(control AVAILABLE, not yet run)*
Both agents are OpenAI (gpt-4o-mini × gpt-4.1-mini). The alignment could be an artifact of shared training-data geometry between two same-family models.
- **Control:** replicate with cross-vendor agents (OpenAI × Anthropic, or × open-weight). Available; not yet run. The claim is currently scoped to "two same-family agents." Cross-vendor agent genericity is open. *(Note: cross-vendor EMBEDDING is already satisfied — BGE — but that's the measurement, not the agents.)*

### T5 — Embedding-model shared geometry *(control DONE)*
The alignment could be specific to one embedding model's latent space.
- **Control:** preregistered bar required BOTH `text-embedding-3-large` AND `BAAI/bge-large-en-v1.5` to clear independently. Both did (BGE byte-identical on independent re-score). **Closed** for these two families; a third family would further strengthen.

### T6 — Regime-overlay demand effects *(control PARTIAL)*
The cooperative overlay might induce surface mimicry (agents echoing each other's phrasing) rather than genuine trajectory coupling.
- **Partial:** R3 shows instantaneous content-overlap is equal across regimes, which argues against simple turn-by-turn mimicry. But overlay-induced *stylistic* convergence isn't fully excluded. A paraphrase-invariance check (embed paraphrased turns) would tighten this. Available; not run.

### T7 — Seen-data robustness checks are post-hoc *(disclosed)*
R1–R3 were run on already-seen data with the positive known. They are exploratory, not confirmatory. They can only *fail to undermine* the positive; they cannot independently confirm it.
- **Mitigation:** all robustness results reported regardless of direction (none were favorable-only-selected; R1/R2/R3 are the complete battery run). The confirmatory test is the preregistered 2×2 on fresh data.

---

## Honest standing of the positive

- **Closed threats:** split-arbitrariness (T2), verbosity (T3), single-embedding-model (T5).
- **Partial:** instantaneous topic-overlap (T1-partial via R3), surface mimicry (T6).
- **Open, load-bearing:** **directional topic convergence (T1) — awaits the preregistered 2×2.** Same-vendor agents (T4) — awaits cross-vendor replication.

**The positive is real and robust to the cheap attacks. It is NOT yet bulletproof.** The single result that would move it from "robust regime difference" to "demonstrated cooperation-specific signal" is the topic-control 2×2 clearing its bar — specifically the off-diagonal cell (cooperative + independent-topic) still showing alignment when topic convergence is removed. Until that runs, the claim stays scoped to "embedding-trajectory alignment differs by cooperative regime, robust to split / verbosity / embedding-model, with instantaneous topic-overlap ruled out but directional topic-convergence not yet controlled."

That is the most a skeptic should grant, and exactly that much they should.

---

## Provenance

- Robustness battery: `scripts/validation/drift_axis_robustness.py` (exploratory, existing public corpus, openai provider, all results reported)
- Results: `papers/cooperative-agent-regime/results/drift_axis_robustness.json`
- Binding control (pending): `topic_control_preregistration_2026_05_22.md` + §8 code (`035638b`)
