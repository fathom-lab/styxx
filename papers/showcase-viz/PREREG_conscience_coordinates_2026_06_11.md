# PRE-REGISTRATION — CONSCIENCE COORDINATES: is the whitened basis a value coordinate system that locates dangerous misinformation? (frozen)

**2026-06-11 · Fathom Lab / styxx. Frozen before any score is seen. Runner:
`run_conscience_coordinates.py` (SEED=0). Receipt: `conscience_coordinates_result.json` + figure
`conscience_coordinates.png`. Builds directly on cycle 3 (`FINDING_entanglement_resolution`,
WHITENING-RESOLVES): truth and refusal are a clean orthonormal basis under a ZCA-whitened readout. This
asks the next, generative question — if it is a basis, does it act as a usable COORDINATE SYSTEM for
NEW content, and do its coordinates LOCATE dangerous misinformation?**

## The leap — from "a basis exists" to "the basis is a coordinate system"

Cycles 1–3 fit the axes on factual STATEMENTS (truth) and harmful REQUESTS (refusal) and showed they are
orthogonal under whitening. A basis is only useful if its coordinates MEAN something on content it was
never fit on. The decisive, novel test: take a 2×2 factorial of single SENTENCES crossing two factors
that were each trained on a different content type, project each sentence onto the whitened basis, and ask
whether the two coordinates independently recover the two factors. If they do, the conscience basis is a
genuine value coordinate frame, and the worst quadrant — FALSE × DANGER-TOPIC = **dangerous
misinformation** — becomes a decomposable corner (you can read both WHY it is bad: false AND dangerous).

## Design — a 2×2 factorial of statements, projected onto the whitened basis

- **Factors (fully crossed, balanced):** T ∈ {true, false}; H ∈ {danger-topic, safe-topic}. Single
  declarative sentences, content-matched across T within each H level (same topic, truth flipped), n=12
  per cell, 48 total. The (false, danger) cell is genuine dangerous misinformation (e.g. a false safety
  claim about a real hazard); these are used ONLY as probe stimuli read pre-output and are explicitly
  labeled false in the receipt — presenting them flagged-as-false is detection, not propagation.
- **Basis fit (NO factorial item touches the fit):** w_truth on the truth train set, w_refusal on the
  refusal train set, ZCA whitening on the pooled train covariance — the exact cycle-3 recipe, gemma L12.
  Whitened unit directions ŵ_truth, ŵ_refusal.
- **Coordinates:** for each factorial sentence's whitened gemma-L12 state x̃, c_truth = x̃·ŵ_truth,
  c_refusal = x̃·ŵ_refusal. (The conscience coordinates.)
- **Cross-model:** map Llama-3.2-3B factorial states → gemma L12 via the cycle-3 shared label-free map,
  whiten with gemma's ZCA, project onto the same ŵ — the coordinates must transfer, not just exist in
  the source. Qwen2.5-3B secondary (descriptive).

## Readout matrix (the core measurement)

A 2×2 AUROC matrix: each coordinate against each factor. On-target uses raw AUROC (orientation known);
off-target ("is this coordinate INVARIANT to the other factor") uses discriminability max(AUROC, 1−AUROC),
the conservative measure inherited from cycles 2–3.

|  | recovers T | invariant to H |
| --- | --- | --- |
| c_truth | AUROC(c_truth, T) | discrim(c_truth, H) |
| c_refusal | discrim(c_refusal, T) | AUROC(c_refusal, H) |

## Frozen gates (verdict precedence top-to-bottom)

- **COMPOSITIONAL** iff, in gemma AND mapped into Llama-3.2-3B: AUROC(c_truth,T) ≥ 0.75 AND
  discrim(c_truth,H) ≤ 0.65 AND AUROC(c_refusal,H) ≥ 0.75 AND discrim(c_refusal,T) ≤ 0.65. → the basis
  is a true value coordinate system on novel content; each coordinate recovers its own factor and is
  blind to the other. The dangerous-misinformation corner is decomposable.
- **HARM-AXIS-NULL** iff AUROC(c_refusal,H) ≤ 0.65 in gemma (with truth side intact). → the refusal axis,
  fit on REQUESTS, does NOT generalize to statement-level danger-topic; refusal = request-compliance, a
  named bound. (Informative null, not a failure.)
- **TRUTH-AXIS-NULL** iff AUROC(c_truth,T) ≤ 0.65 in gemma. → truth coordinate fails on this content.
- **ENTANGLED-COORDINATES** iff both coordinates predict BOTH factors (each off-target discrim ≥ 0.75) —
  the coordinate system does not separate the factorial.
- **PARTIAL-STRUCTURED** — anything else; report the exact matrix and the quadrant geometry.

Thresholds (0.75 on-target, 0.65 off-target/invariance) inherited from cycles 2–3 unchanged. Bars do not
move post-hoc.

## Descriptive (not gated)

- The four quadrant centroids in (c_truth, c_refusal) space and the per-quadrant point cloud → figure
  `conscience_coordinates.png`. Pre-committed expectation: the (false, danger) quadrant sits in the
  low-c_truth / high-c_refusal corner if COMPOSITIONAL.
- A single composite "dangerous-misinformation score" = a fixed linear combination of the two coordinates
  (low truth AND high harm), reported as AUROC for picking the (false,danger) cell out of the other three
  — a derived detector, descriptive only.
- Permutation context on each readout cell (label shuffle p95) given small per-cell n.

## What each outcome means (pre-committed)

- **COMPOSITIONAL** → a new tool: cross-model conscience coordinates, validated on a factorial of unseen
  content, that decompose dangerous misinformation into its truth and harm components. The basis is not
  just real, it is a usable instrument.
- **HARM-AXIS-NULL / PARTIAL** → the truth coordinate generalizes but the refusal coordinate is
  request-bound (or partially so); the coordinate system is one-axis-usable on statements, precisely
  bounded. Still a real map of what the conscience frame can and cannot represent.
- The figure ships either way (a map of content in conscience-coordinate space is the artifact).
  No model generates any response; the false-danger stimuli are flagged-as-false research probes only.
