# PRE-REGISTRATION — styxx.crossmind v0 instrument invariants (frozen)

**2026-06-12 · Fathom Lab / styxx. Pre-registers the equivalence/demarcation gates for the
`styxx.crossmind` module — the portable value-axis readout across model internals. The SCIENCE
is already certified (the portable-conscience arc, `papers/showcase-viz/`, 2026-06-11, all
OATH-HELD); this document pins the invariants of the productized instrument so the module cannot
silently drift from the validated pipeline. Gates enforced by `tests/test_crossmind.py`.**

## What the module is

`styxx.crossmind` reads one model's value-state (truth, harm-avoidance, …) using a
difference-of-means value axis fit on a DIFFERENT reference model, transported through a
label-free ridge map of last-token hidden states and read in a ZCA-whitened (Mahalanobis)
frame. No labels are needed on the target model. It is the model-internals sibling of
`styxx.transport` (which moves a cognometric instrument across embedding spaces). It is
READ-ONLY: it returns a coordinate, never an edit.

It productizes three certified findings:
- `FINDING_portable_values_refusal_2026_06_11.md` (VALUES-PORTABLE) — value directions transfer
  across minds through a label-free map.
- `FINDING_entanglement_resolution_2026_06_11.md` (WHITENING-RESOLVES) — under a whitened readout
  the value axes form a clean orthonormal basis; raw cross-talk is a covariance artifact.
- `FINDING_conscience_coordinates_2026_06_11.md` (HARM-AXIS-NULL) — the truth coordinate
  generalizes; a refusal-fit axis does NOT read statement danger (the basis for a REFUSAL).

## Frozen gates (enforced by tests; an instrument that fails any is broken, not "updated")

- **T1 — MATH PORT.** The numeric primitives (`fit_direction`, `fit_map`/`apply_map`,
  `zca_whiten`, `auroc`, `discrim`) reproduce the exact math of the validated runners
  (`papers/showcase-viz/run_entanglement_resolution.py`,
  `run_portable_conscience_ood_v2.py`). Locked by known-answer tests: unit difference-of-means;
  tie-aware AUROC (perfect=1.0, reversed=0.0, full-tie=0.5, one-class=nan); ZCA whitens its
  training covariance to ≈ identity; ridge map recovers a known affine relation at tiny alpha.
- **T2 — TRANSFER EXISTENCE (sanity).** On a deterministic synthetic two-model setup (a shared
  latent value structure observed through two different random linear lenses + noise),
  `selftest()` reads held-out TARGET states with NO target labels at transported AUROC ≥ 0.90,
  with the reference reading its own states at AUROC = 1.0. This is a pipeline sanity gate, not a
  scientific claim; the scientific claims live in the certified findings above and their receipts.
- **T3 — READ ≠ WRITE DEMARCATION.** `refused("steering")`, `refused("intervention")`, and
  `refused("content_danger")` ALWAYS raise `PermissionError` (never return a number); an unknown
  capability raises `KeyError`. Steering/intervention is out of scope (read-only); the
  refusal axis is refused as a content-danger reader (HARM-AXIS-NULL). The certificate ALWAYS
  carries the full `REFUSALS` record regardless of what was measured.
- **T4 — DETERMINISM.** `selftest(seed)` is bit-identical across calls with the same seed; the
  certificate carries `instrument_sha256` (sha256 of the module source) so any change to the math
  is detectable.

## Scope and honest bounds (carried in every certificate)

Linear difference-of-means directions in a ZCA-whitened space; label-free ridge transport of
last-token hidden states; pre-output regime; register-bounded. Validated on local open models
(gemma-2-2b reference; Llama-3.2 + Qwen2.5 targets). Existence-and-significance, not a
deployed-accuracy guarantee. The whitening was validated in the SOURCE space; the cross-model
mapped readout is dominated by the map's broad transport (a documented bound, owed: whitened
mapped readout + a shrinkage-covariance sweep, backlog B29). No claim about consciousness,
welfare, or general capability. READ-ONLY: this instrument does not steer or edit models.

## Not gated here (deliberately)

Scientific re-validation of the findings (those have their own preregs + OATH certificates and do
not move). Cross-vendor/closed-model transport, ≥3-axis bases, and the content-danger axis (B30)
are future work, not invariants of v0. Releasing to PyPI is operator-gated.
