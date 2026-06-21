# POSITIONING — cross-model WRITE channel vs the literature (BEFORE the writeup)

**2026-06-20. Fathom Lab / styxx.** Lit-check workflow (`wf_6fd196f4`, 3 scouts + a positioning critic,
58 web tool-uses) run BEFORE Stage 1 produces a number — the same prereg→lit-check→writeup discipline that
caught the obfuscation novelty-overclaim. Purpose: fix the honest novelty boundary and the redlines now, so
no verdict gets dressed up later. **arXiv IDs below are scout-reported; several are future-dated (2026) and
MUST be re-verified before any formal/public citation.**

## The honest novel core (IF Stage 1 lands a behavioral write)
The prior art splits cleanly and leaves exactly one gap:
- **Zero-paired / unsupervised methods all READ only:** vec2vec (translate/invert *encoder embeddings*,
  no steering, no generative internals), relative representations (anchor-based stitching), Gromov-Wasserstein
  (OT alignment of word-embedding spaces), Platonic Rep Hypothesis (measures convergence, no method).
- **Cross-model WRITE methods all use PAIRED/anchor/projection correspondence:** L-Cross, LRTH (parallel
  activations), Master Key, Cross-Model Safety Steering 2606.05290 (transport fit on **paired benign anchors**),
  ExpertSteer (dimension-only projection, no learned correspondence).
- **The unclaimed triple intersection:** *unsupervised correspondence-recovery* (the map is handed two
  unordered concept clouds and recovers the matching itself, via Sinkhorn/GW — no label fed) **+ cross-model
  behavioral CONTROL** (the transferred direction changes the *target's output*, tested on HELD-OUT concepts).
  That is the only thing nothing above claims.

## ⚠ The load-bearing caveat for OUR setup — are we even "zero-paired"?
**Our rig probes BOTH models on the SAME concept battery** (the same `CONCEPT_TEMPLATES` prompts). The map
fitting (`_fit_Q`) does **not** use the per-concept correspondence (Sinkhorn + GW warm-start recover the
permutation from geometry alone; no identity init) — so it is **label-free / correspondence-unsupervised**.
**But it is NOT "zero-shared-data" in vec2vec's strict sense** (vec2vec uses *unpaired corpora* — different
texts per space). Per redline (and 2606.05290's standard that "even benign-prompt alignment counts as paired"),
**we MUST NOT call this "zero-paired" / "zero-shared-data."** The defensible claim is precisely:
*"label-free cross-model control transfer — the correspondence is recovered from the geometry of a shared
concept inventory without using the concept labels, and steers genuinely held-out concepts."* The
disjoint-worlds line (truly disjoint tokens/corpora) is the zero-shared-data result; THIS is not that.

## Framing by verdict (locked now)
- **`TRANSFER WORKS`** (G0 cleared AND behavioral output-change with native-ceiling + random-Q + wrong-concept
  controls, as a RATE over the 70 held-out concepts): claim *"label-free cross-model steering transfer with a
  validated causal behavioral positive control — correspondence recovered unsupervised from shared-concept
  geometry, not from paired anchors."* Lead with the **correspondence regime** (unsupervised recovery), NOT
  "cross-model steering" (taken by L-Cross/ExpertSteer/2606.05290) and NOT "universal geometry" (contested).
  Headline number and the negative control in the same sentence.
- **`PARTIAL` / `REPORT_AS_LANDED` (read works, write doesn't):** *"the label-free map is read-faithful yet
  control-inert"* — an honest READ≠WRITE dissociation, consistent with our own prior `ALIGNER_LIMITED` finding
  and with every external cross-model write needing correspondence data. Credit vec2vec for the read lineage;
  this EXTENDS zero-paired *reading* from encoder embeddings to generative hidden states (incremental, not a
  write result). A null here is **prior-consistent, not a surprise** — do not dress it as a "wall we hit."
- **`INSTRUMENT-CEILING` (G0 never clears):** report the pre-registered gate failed even at 4× concepts;
  converge with our prior closed-negative on zero-paired transport + the external frontier. *(Note: Stage-0's
  first grid point already read pc_cos 0.87 > 0.80, so this branch looks unlikely — pending the sealed FINAL.)*

## Redlines (do NOT write)
1. NOT "first cross-model steering / first to change model B from model A / first weak-to-strong" — owned by
   L-Cross (2501.02009), LRTH (2506.00653), Master Key (2604.06377), 2606.05290, ExpertSteer (2505.12313).
2. NOT "first unsupervised/zero-paired translation" — vec2vec (2505.12540) owns zero-paired *reading*.
3. NOT "zero-paired" / "zero-shared-data" for THIS rig — both models see the same concept battery (see caveat).
4. NOT a WRITE/control claim on the strength of cosine / read-out / CKA — translation quality ≠ behavioral
   control (CKA inflates: Davari 2022). Requires a measured OUTPUT change in Llama-1B or it doesn't exist.
5. NOT a single successful transfer as a capability — bar is a RATE + FPR over many trials WITH random-Q /
   random-pairing negative control (the orthonormal-pairing paper: random pairing → 0.00).
6. NOT "models share a universal geometry of meaning" as fact — Platonic Rep Hypothesis is CONTESTED post
   null-calibration (Aristotelian-view 2602.14486; Nature MI s42256-025-01139-y). Hypothesis-under-challenge only.
7. NOT Anthropic introspection (~20%) as cross-model evidence — it is INTRA-model write-then-read; cite only as
   the reliability-reporting (rate + ~0% FPR) discipline standard.
8. NOT "the ancient question of a universal structure of mind" as an empirical result — register/motivation only.

## Verified credits (scout-confirmed; re-verify IDs before formal citation)
- **Zero-paired READING:** vec2vec (2505.12540), mini-vec2vec (2510.02348).
- **Cross-model WRITE (paired/anchor/projection — the write result we are NOT first at):** L-Cross (2501.02009,
  ACL 2025), LRTH (2506.00653), Master Key (2604.06377 — future-dated, verify), Cross-Model Safety Steering
  (2606.05290 — future-dated, verify), ExpertSteer (2505.12313).
- **Alignment lineage (our Wasserstein-Procrustes is here, not new):** Relative Representations (2209.15430,
  ICLR 2023), Gromov-Wasserstein (1809.00013, EMNLP 2018).
- **Universal-geometry (motivation, contested):** Platonic Rep Hypothesis (2405.07987, ICML 2024); critiques
  Aristotelian-view (2602.14486, verify), Nature MI (s42256-025-01139-y).
- **Same-model steering baselines:** RepE (Zou 2023), ActAdd (2308.10248), CAA (2312.06681).
- **Reliability discipline:** Anthropic/Lindsey introspection (transformer-circuits.pub, Oct 2025; arXiv mirror
  2601.01828 UNCONFIRMED — cite the circuits page).
- **Cross-model reading generally:** Patchscopes (2401.06102), Model Stitching (2106.07682), Universal Neurons
  (2401.12181). **Metric caution:** CKA critique (Davari 2022; 2405.01012) — pair any similarity with a behavioral test.
- **DO NOT CITE without check:** 'Back into Plato's Cave' (2604.18572), Universal SAEs (no stable id).
