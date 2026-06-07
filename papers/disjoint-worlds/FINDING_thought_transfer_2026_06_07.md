# FINDING — Zero-anchor cross-model CONTROL transfer: INCONCLUSIVE (instrument-limited)

**2026-06-07. Fathom Lab / styxx.** PREREG_thought_transfer_2026_06_06. Companion to the
cross-model LEGIBILITY result (reading partially works) — this is the WRITING direction.
**Bottom line: writing was NOT demonstrated and the result is ALIGNER_LIMITED per the frozen prereg
(positive control 0.74 < required 0.80; faint signal not significant). "Reading ≠ writing" is a
hypothesis here, not a landed finding.**

## Question

Can a steering/control direction computed only in model A be installed in model B through the
unsupervised zero-anchor map (`styxx_transfer.TransferMap`) and actually STEER B — the "compute a
control once, mount it on any model" idea?

## Instrument validation (the honest diagnosis that saved the result)

- The random-direction positive control read 0.138 — but that is a **measurement artifact**: a
  random vector in d≈3072 has only ≈√(k/d)≈0.14 of its norm inside the top-k concept subspace the
  map operates on. Not a broken map.
- **Fair** positive control (transfer the actual CONCEPT directions on a KNOWN A→A rotation) =
  **0.742** — i.e. the instrument transfers concept directions at their ceiling (concept-vec
  subspace fraction is also 0.742; the 26% out-of-subspace is the only loss). In-subspace random
  dirs transfer at **1.000** (the map/Q is exact on a known rotation).
- First cross-model run was **underpowered** (hardcoded dose → native steering floored at 0.013);
  re-run **locked the dose** (α=16 → native steering mean 0.085, individual concepts to +0.33).

## Result (Llama-3.2-3B → Llama-3.2-1B, RSA 0.965, 21 held-out concepts, α=16)

| quantity | value |
|---|---|
| mean **transfer** steering gain | **0.019** |
| mean **native** steering gain (ceiling) | 0.085 |
| mean random-Q gain (null) | −0.004 |
| mean wrong-concept gain (specificity null) | 0.003 |
| transfer / native (NTE) | **0.22** |
| transfer beats random-Q | 14 / 21 |

**Verdict per the frozen prereg: ALIGNER_LIMITED / INCONCLUSIVE — not a landed falsification.**
Two reasons, stated honestly:
- **The instrument fell below its own pre-registered gate.** PREREG G0 required the A→A-rotation
  concept-direction positive control ≥ 0.80; it reached only **0.74** (capped by the 0.742
  concept-vector subspace fraction at k=60). Under a *failing* positive control, a cross-model null
  is **not interpretable** — it could be the map's imprecision, not a real boundary. (The program's
  own rule: don't read a null from an un-validated instrument.)
- **The faint signal is not significant.** transfer-beats-random 14/21 → sign-test **p=0.19**,
  Wilcoxon **p=0.055**; only paired-t reaches p=0.03. NTE 0.22 and "beats random by ≈0.02" do not
  survive the prereg's own test.

## Reading vs writing — a HYPOTHESIS, not a landed result

The *suggestive* picture: reading partially transfers zero-anchor (legibility top1 0.15–0.48), while
a transferred control direction did **not** measurably steer (NTE 0.22, n.s.). The *leading
hypothesis* is that **steering is more sensitive to map imprecision than reading is** — a 48%-accurate
correspondence is enough to partly read but scrambles a specific control direction. **But this is not
established here:** the writing test is instrument-limited (G0 failed) and underpowered (n.s.). It is
a hypothesis to confirm by **raising the instrument past G0** (higher PCA dim / a read layer where
concept directions sit more in-subspace) and re-running prereg-clean — owed, not done.

We do **not** claim "reading ≠ writing across minds" as a result, nor that it "bounds a class of
methods." What we have: cross-model *reading* works (separate finding); cross-model *writing* was
**not demonstrated**, with an instrument that fell short of its own gate.

## Honest scope

One near-isometric same-lineage pair (Llama-3B↔1B); MiniLM steering-gain metric; 8-concept-family
held-out set; shared training data not controlled. Cross-family (lower RSA) would be ≤ this. A
faint transfer signal exists; the claim that is falsified is *functional* zero-anchor control
transfer, not the existence of any shared directional structure.
