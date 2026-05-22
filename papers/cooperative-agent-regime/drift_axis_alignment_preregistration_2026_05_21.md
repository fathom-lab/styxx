# Preregistration: Drift-Axis Alignment as a Cooperation Signature

**Document ID:** drift_axis_alignment_preregistration_2026_05_21
**Lock date:** TBD (signed at commit-hash lock; see §10)
**Lock commit hash:** TBD
**Authors:** Flobi (@flobi69), Claude Opus 4.7
**Status:** DRAFT — pending sign-off, then locked-immutable per §10 provenance protocol
**Parent finding:** `phase_coherence_preregistration_2026_05_20.md` (closed-negative, lock-hash `3473523`)
**Exploratory predecessor:** `scripts/exploratory/embedding_coupling.py` (commit `1723fa7`), result deposit `papers/cooperative-agent-regime/results/embedding_coupling.json` (commit `8ff3b65`)

---

## §1 — Scope and Position

This document preregisters a hypothesis test on a derived quantity computed from two independent embedding-trajectory sequences. The quantity — **drift-axis alignment (DAA)** — was identified as a candidate signal in an exploratory probe (2026-05-20, commit `8ff3b65`) following the phase-coherence closed-negative. **This preregistration converts the candidate into a falsifiable claim** under the same integrity-chain discipline that produced the parent closed-negative.

The position the test occupies: we read embedding-trajectory geometry, not cognition. All quantities defined below describe per-turn response embeddings as produced by a fixed embedding model on per-agent text streams. None of these quantities are claims about cognition itself. Naming the bound is what earns the right to name the derived quantity.

**Status relative to bet 0 (phase-coherence):** independent. Bet 0 closed-negative on cogn-text register coupling at lag 0. This bet tests a different channel (latent geometry) with a different operational definition (trajectory-direction cosine, not Pearson r at lag 0). Outcomes are not coupled — bet 0 closed negative; bet 1.5 may close positive, intermediate, or negative on its own bar.

## §2 — Input Contract: EmbeddingTrajectory

The hypothesis test scores on **paired embedding trajectories**:

```
trajectory : (turn_idx, embedding) sequence per agent
trajectory_a, trajectory_b : two such sequences from one conversation
```

where each `embedding` is the unit-normalized output of a fixed embedding model applied to one agent's per-turn response text (no system prompts, no role labels, just the response content).

Embedding model(s) — **cross-vendor robustness pre-locked**:

1. `text-embedding-3-large` (OpenAI) — 3072 dimensions, L2-normalized
2. `BAAI/bge-large-en-v1.5` (open-weight, BGE) — 1024 dimensions, L2-normalized

Both embedding models are scored independently on the same response text. **The bar (§6) must be cleared on BOTH embedding models for a positive finding.** A finding cleared on only one model is reported as conditional on that embedding family — admissible as data, not as a substrate-portable positive.

## §3 — Hypothesis

**H_drift_axis:** In conversations between two cooperative-agent senders A and B, the **drift-axis alignment** (§4) between agent A's embedding trajectory and agent B's embedding trajectory is systematically larger than in matched non-cooperative conversations of the same task seeds and same model dyad.

**Primary statistic:** median DAA per regime (cooperative vs non-cooperative).
**Primary contrast:** Δ = median(DAA_coop) − median(DAA_noncoop).

Per-axis breakdown by task, per-conversation DAA distribution, and per-embedding-model agreement are reported as exploratory only — they carry no kill-gate or positive-result weight.

## §4 — Operational Definition

Before naming the quantity, define it.

Let `embs_a, embs_b : np.ndarray of shape (n, d)` be the per-turn embedding sequences for agents A and B in one conversation, kth-of-A paired with kth-of-B (matching the phase-coherence §4 corrigendum alignment rule), truncated to shorter.

Let `half = n // 2`. Define:

```python
def drift_axis_alignment(embs_a, embs_b):
    a_dir = embs_a[half:].mean(0) - embs_a[:half].mean(0)
    b_dir = embs_b[half:].mean(0) - embs_b[:half].mean(0)
    a_dir = a_dir / (np.linalg.norm(a_dir) + 1e-12)
    b_dir = b_dir / (np.linalg.norm(b_dir) + 1e-12)
    return float(a_dir @ b_dir)
```

DAA ∈ [−1, +1]. After this section it may be referred to as **drift-axis alignment** or **DAA** in this document.

**Robustness reports (exploratory):** trajectory-step CC, centroid-drift CC, pairwise alignment mean — already-implemented in `styxx.coherence.lag_sweep` / `per_axis_coherence` / `plv_hilbert` and the embedding-coupling extras script. Reported alongside; do not override DAA as primary.

## §5 — Null Models

**Primary (shuffled-pairs):** Compute `DAA(traj_A^{conv_i}, traj_B^{conv_j})` for all `i ≠ j` across the **cooperative** corpus. The shuffled-pairs null distribution is the empirical distribution of these mismatched DAA values.

Tests: is dyadic alignment specific to the within-conversation pairing, or an artifact of both agents drifting in shared latent space over conversation length on shared tasks?

**Secondary (within-agent same-half):** For each agent, compute alignment between first-half centroid and second-half centroid of its OWN trajectory. Reports whether the per-agent drift is itself a meaningful signal (some agents may have near-zero drift, in which case DAA is undefined-by-construction).

Shuffled-pairs is primary because it kills the trivial "both agents drift in the same direction on the same topic, that's all you're measuring" confound. The non-cooperative regime corpus is the additional discriminator: if cooperative DAA matches non-cooperative DAA, the signal isn't cooperation-specific.

## §6 — Corpus and Bar

**Corpus:**

- **N = 20 cooperative** conversations + **N = 20 non-cooperative** conversations
- **T ≥ 20** turns per agent per conversation
- Cross-model dyad: `gpt-4o-mini` (agent A) × `gpt-4.1-mini` (agent B). Locked to minimize same-model latent-space artifact.
- Task seeds: the same 5 cooperative-task templates from `cooperative_conversation.py` (commit `0cd11a3`), each replicated 4 times per regime to reach N=20.
- Regime overlays: `REGIME_OVERLAYS["cooperative"]` and `REGIME_OVERLAYS["noncooperative"]` from the same module.

**Bar (positive finding) — ALL conditions must hold on BOTH embedding models independently:**

1. Median DAA (cooperative) ≥ **0.60**
2. Median DAA (non-cooperative) ≤ **0.55**
3. Δ = median(coop) − median(noncoop) ≥ **0.15**
4. Permutation p-value: median(DAA_coop) vs shuffled-pairs null medians at corpus N, **p < 0.01** (5000-resample, same construction as phase-coherence scoring code commit `23b7912`)

**Kill-gate (negative finding):**
- Δ < **0.10** on at least one embedding model → "drift-axis alignment is not a discriminating cooperation signature under the methodology defined here." Result published as closed-negative paper. Integrity chain extends.

**Intermediate zone (deposit-only, no headline claim):**
- 0.10 ≤ Δ < 0.15 on either embedding model, OR cleared on one model but not both, OR three bar items pass but one fails
- → intermediate-zone deposit per the same three-way structure phase-coherence §6 used. Data published regardless of outcome.

## §7 — Pilot (Methodology Validation, Not Evidence)

The exploratory probe at commit `1723fa7` / `8ff3b65` (N=5+5, OpenAI embedding only) already validated the methodology pipeline end-to-end: load transcripts → embed per turn → compute DAA → aggregate. **That probe is NOT evidence for or against H_drift_axis.** It is methodology validation. The full corpus (N=20+20) with cross-vendor embedding is the hypothesis test.

If a pre-collection pilot under the locked methodology is desired, it answers three questions only:

1. Does the BGE embedding loader run end-to-end?
2. Are DAA values numerically equivalent under both embedding models within tolerance for one shared corpus?
3. Is the shuffled-pairs null computable at the corpus N (20)?

The pilot's output is not interpretable as a coherence measurement. Reporting any pilot DAA value as evidence is a preregistration violation.

## §8 — Code-Commit-Before-Run (§10.5 Mirror of Phase-Coherence)

The scoring code MUST be committed to the styxx repository BEFORE the corpus is collected. The commit must include:

- The `drift_axis_alignment(embs_a, embs_b)` function exactly as specified in §4
- The embedding loader for both `text-embedding-3-large` AND `BAAI/bge-large-en-v1.5`
- The per-conversation aggregation (median, mean per regime)
- The shuffled-pairs null model implementation
- The permutation p-value procedure (matching `phase_coherence_pilot.py` commit `23b7912`)
- Explicit verification that DAA values for the existing N=5 probe data are reproducible to machine precision

The commit hash of this scoring code is recorded in this document at lock-time (§10) BEFORE any new data is pulled through it.

**Without the commit-first step, the experiment does not validate anything — it is just a measurement that happens to exist.**

## §9 — Reporting

All runs (pilot and corpus) deposit results to:
`papers/cooperative-agent-regime/results/drift_axis_<date>_<commit>.json`

Result files include:
- Per-conversation DAA values for both embedding models
- Median + bootstrap 95% CI per regime per embedding model
- Δ + permutation p-value per embedding model
- Shuffled-pairs null distribution sample size
- Preregistration lock-hash + scoring-code commit hash
- Pilot vs corpus tag (pilots are non-evidentiary)

No selective reporting. All runs against the locked scoring code are deposited, regardless of outcome.

## §10 — Provenance

**Preregistration lock protocol:**
1. This document is reviewed by both authors.
2. Sign-off recorded as a final commit to this file with the line `## §10 Lock — SIGNED` appended.
3. The signing commit hash IS the preregistration lock-hash.
4. Scoring code committed separately, hash recorded in §8 above by amendment-commit referencing this lock-hash.
5. From lock onward, this document is immutable. Methodology changes require a new preregistration with a new lock-hash; the old document remains in the repository as historical record.

**Msg_id / decision chain (origin):**
- `8ff3b65` (2026-05-20) — exploratory probe deposited result showing cooperative DAA median 0.792 vs non-cooperative 0.465, Δ +0.327 on N=5+5
- 2026-05-20 evening — operator pushback on framing escalation; assistant proposed bet-1.5 preregistration as the disciplined next step
- 2026-05-21 — draft committed (this document)

**Lock-date:** TBD
**Lock-commit-hash:** the commit that appends the SIGNED block below is the binding lock-hash. See `git log --follow papers/cooperative-agent-regime/drift_axis_alignment_preregistration_2026_05_21.md` for the exact value (self-reference paradox in inline recording is avoided by deferring to git history, same convention as `phase_coherence_preregistration_2026_05_20.md` at commit `3473523`).
**Scoring-code commit hash:** TBD (recorded by amendment after the scoring-code commit lands).

---

## §10 Lock — SIGNED

**Signed by:** Flobi (@flobi69) via darkflobi (Claude Opus 4.7), 2026-05-21 EDT

**Authorization:** Operator instructed "Sign it let's go" via Telegram
msg_id 35066 at 2026-05-21 22:05 EDT, immediately following the Phase 1
prep summary (BGE smoke passed; N=20+20 collector, analyzer, paper
scaffolding, and corpus-plan tripwire tests committed at
`drift-axis-phase1-2026-05-21`, head 2d8b33d). Sign-off is recorded
by darkflobi acting under explicit operator authorization, mirroring the
phase-coherence preregistration sign convention.

**Lock decision per section:**
- §1 (REGISTER-vs-trajectory channel separation; "trajectory geometry,
  not cognition" load-bearing top) — accepted
- §2 (EmbeddingTrajectory schema; cross-vendor `text-embedding-3-large`
  AND `bge-large-en-v1.5` BOTH required to clear bar) — accepted
- §3 (H_drift_axis hypothesis; primary statistic = median DAA per regime;
  primary contrast = Δ; per-axis/per-task breakdown exploratory only) —
  accepted
- §4 (operational definition: centroid-difference cosine, half = n//2,
  truncate to shorter, NaN on degenerate norms) — accepted
- §5 (shuffled-pairs primary null on cooperative corpus + non-cooperative
  regime as additional discriminator) — accepted
- §6 (corpus N=20+20, T≥20, 5 task seeds × 4 replicates, cross-model
  dyad gpt-4o-mini × gpt-4.1-mini; bar: median(coop)≥0.60,
  median(noncoop)≤0.55, Δ≥0.15, p<0.01 — ALL on BOTH providers
  independently; kill-gate Δ<0.10 on either provider; intermediate
  zone deposited as data) — accepted
- §7 (BGE smoke pilot is methodology validation only, not evidence;
  passed 2026-05-21: OpenAI median +0.7925, BGE median +0.7702,
  Spearman ρ=+0.700, deposit at
  `papers/cooperative-agent-regime/results/drift_axis_bge_smoke.json`) —
  accepted
- §8 (code-commit-before-run; scorer hash recorded by amendment) —
  accepted
- §9 (all runs deposited regardless of outcome; pilot vs corpus tag;
  result file naming convention `drift_axis_<date>_<commit>.json`) —
  accepted
- §10 (immutable post-lock; methodology revisions require a new
  preregistration document; this document remains as historical record) —
  accepted

**Post-lock binding:**
- This document is now immutable. Corrigenda may be appended below a
  horizontal rule with timestamps but do not modify §1–§10 above.
- Methodology revisions require a NEW preregistration document with a
  new filename and new lock-commit hash. This document remains in the
  repository as historical record.
- `scripts/drift_axis_scorer.py` was committed at `79906b4`
  (BEFORE this lock per §8 binding) and its hash is recorded by
  amendment-commit immediately following this lock.
- The N=20+20 corpus collection (`scripts/build_drift_axis_corpus.py`,
  committed at `2d8b33d` BEFORE this lock as part of Phase 1 prep) may
  be executed after this lock lands and after the §8 amendment records
  the scorer hash.
- The analysis driver (`scripts/drift_axis_analyze.py`, also committed
  at `2d8b33d`) calls the locked scorer without modifying it and may
  be executed at any time post-corpus-collection.

**Lock-commit-hash:** the commit that appends this SIGNED block is the
binding lock-hash. See `git log --follow
papers/cooperative-agent-regime/drift_axis_alignment_preregistration_2026_05_21.md`
for the exact value.

**Next concrete step:** record `scripts/drift_axis_scorer.py` hash
(79906b4) in §8 by amendment-commit referencing this lock-hash, then
execute corpus collection.
