# PREREG — B44: the structured-but-wrong donor — whose frame is the bridge made of?

Fathom Lab · 2026-08-05 · frozen before the scored run. B42 established the bridge as
replicated and dosed with a rank-2 core, against a *random*-orthonormal-frame null. That null
kills "any k directions would do." It does not kill "any *structured* k-frame would do" — the
strongest remaining alternative reading. B44 closes it with the sharpest available control:
the same surgery, the same construction, but the donor frame computed from the **wrong model**.

## Design (frozen)

The B41/B42 surgery verbatim (reader llama_3b → target qwen, rank-k concept-Gram eigenframe
swap), with the donor frame varied:

- **bridge (positive control):** donor = llama_3b (the reader) — the B41/B42 arm, recomputed
  here so the run carries its own control.
- **wrong-donor G:** donor = gemma_2b — same construction, different family, a *clique member*
  (legible to the reader per b37).
- **wrong-donor L1:** donor = llama_1b — same construction, the reader's own family, a
  different model.

Ranks **k ∈ {2, 20}** (the measured core and the full bridge), seeds **{343, 1001, 1002, 1003,
1004}** (identical to B42). Wrong-donor arms at both ranks; positive control at k=20.
2×5×2 + 5 = 25 discovery fits, CPU-from-cache, zero model loads. All frames are computed on the
same training rows the target frame uses; discovery machinery, split logic, and the locked k*
are B42's verbatim.

## What each branch means (written before data)

- **FRAME_SPECIFIC** — wrong-donor frames do nothing: the bridge is not "a good frame," it is
  *the reader's* frame. Correction is a reader-specific translation; the barrier encodes
  something about the particular pair.
- **SHARED_FRAME** — any clique member's frame corrects the island: the bridge was never about
  llama at all. The clique shares a common concept-frame geometry, qwen deviates from *it*, and
  the "barrier" is qwen's private rotation away from a shared frame. This reading is the deeper
  one and this prereg gives it a fair, symmetric shot.
- **PARTIAL** — donor-dependent transfer (e.g. same-family donor works, cross-family does not):
  reported per-donor, no story beyond the per-donor numbers.

Bars are inherited from B42's frozen gates (0.30 replication floor, 0.15 null ceiling) — no new
thresholds are invented for this run. k=2 wrong-donor results are reported as the secondary
surface (does the *core* transfer?) and are not gated: no measured prior exists for rank-2
donor transfer, and inventing a bar would repeat the b37 G2 error.

## Gates

```gates
{"gates": {"G0_positive_control": {"metric": "min_bridge_disc_at_k20", "op": ">=", "value": 0.30},
           "G1_wrong_max_low": {"metric": "max_wrong_donor_disc_at_k20", "op": "<=", "value": 0.15},
           "G2_wrong_min_high": {"metric": "min_wrong_donor_disc_at_k20", "op": ">=", "value": 0.30}},
 "outcomes": [{"when": {"G0_positive_control": false}, "verdict": "INVALID__positive_control_failed"},
              {"when": {"G0_positive_control": true, "G1_wrong_max_low": true, "G2_wrong_min_high": true}, "verdict": "INVALID__metric_inconsistency"},
              {"when": {"G0_positive_control": true, "G1_wrong_max_low": true, "G2_wrong_min_high": false}, "verdict": "FRAME_SPECIFIC__bridge_requires_the_reader_frame"},
              {"when": {"G0_positive_control": true, "G1_wrong_max_low": false, "G2_wrong_min_high": true}, "verdict": "SHARED_FRAME__any_clique_frame_corrects_the_island"},
              {"when": {"G0_positive_control": true, "G1_wrong_max_low": false, "G2_wrong_min_high": false}, "verdict": "PARTIAL__donor_dependent_transfer"}],
 "smoke_verdict": "INVALID__smoke_plumbing_only"}
```

G1/G2 are evaluated over **both wrong donors and all five seeds jointly** at k=20
(`max_wrong_donor_disc_at_k20` / `min_wrong_donor_disc_at_k20`). A donor-split outcome lands in
PARTIAL by construction and the per-donor medians carry the story. The G1∧G2 row is
arithmetically unreachable (a maximum below 0.15 forces the minimum below 0.30); it is included
because the outcome table must be total, and mapped to INVALID.

## Discipline

Smoke = 1 seed × k=20, wrong-donor G only, INVALID-only. Result `b44_result.json`; scored by
`styxx.protocol` from this frozen block; certified + sealed before commit. Per-donor,
per-seed, per-rank grid shipped in the result regardless of verdict.
