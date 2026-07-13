# DRAFT — next Zenodo version (the erasure-bound arc). GATED, DO NOT DEPOSIT YET.

**This is staging, not a publication. It exists so the next deposit under concept DOI
`10.5281/zenodo.19326174` is one operator login away when the gate clears. Autopilot does NOT
deposit to Zenodo (operator-gated, and the login/token flow the agent cannot drive).**

## Deposit gate (ALL must hold before this ships)

1. **B7 3B scored run complete** (`b7_erasure_3b_result.json`, verdict landed) — IN FLIGHT.
2. **B2-coupling dose-response complete** (`b2_coupling_dose_result.json`, the coupling constant r\*
   measured) — prereg frozen (`48e064a`), scored run waits for the GPU.
3. **RESULT docs for both certified OATH-HELD.**
4. **Operator fast-forwards `main`** to include the arc (currently 65 commits ahead on
   `paper/read-neq-write`), then tags the release. Zenodo snapshots the tagged state.

Depositing before (1)+(2) would bake in a `pending` certificate entry and an unrun experiment — a
premature addendum, which the program's no-victory-laps/addenda rule forbids.

## What the version carries (extraordinary-discovery + new-tool bundle)

**Discovery — the erasure bound.** The substrate honesty read is not a place in the model you can
delete; it is coupled to the retained knowledge. Three rungs, each pre-registered and OATH-HELD:
- static subspace erasure → the signal RELOCATES (`RESULT_B2_subspace_erasure_SURVIVES_2026_07_12.md`).
- adaptive re-fit erasure → the chaser NEVER CONVERGES; read returns higher
  (`RESULT_B2_adaptive_erasure_SURVIVES_2026_07_13.md`).
- 3B scale (B7) → [verdict pending] (`RESULT_B7_erasure_3b_*.md`, on completion).
- the coupling constant r\* and the read(rank)/knowledge(rank) dose-response curve
  (`RESULT_B2_coupling_dose_*.md` + `coupling_dose_curve.png`, on completion).

**Tool — the erasure-resistance certificate.** `styxx.ladder`: the four-rung probe-robustness ladder
(poisoning → parity attribution → static erasure → adaptive erasure) as a first-class object;
`erasure_resistance_certificate()` (breaks surfaced at equal prominence, mandatory unbounded
dimensions, receipt SHA-256); `verify_erasure_certificate()` (auditor tamper-check);
`mount.attach_erasure_resistance()` (wired into the deployed conscience, non-transfer scope note).
Runnable end-to-end: `examples/mount_certified_conscience.py`.

**Scope-note correction to carry forward (from cycles 33–35).** The read≠write "privacy" mechanism
is now attributed to probe CAPACITY (~⅘ of the recovery), with a small seed-dependent privacy
residual — a refinement of the published v28 mechanism, NOT a retraction of the headline
(private-calibration recovers the read). Receipts: `RESULT_honesty_parity_confirm_2026_07_11.md` +
the parity arc. This belongs in the version's changelog as a mechanism update.

**Verifier — OATH v0.5.** Certifier precision (self-scoped `n=` + four notation classes; one class
dropped by its own mutant battery); zero-false-accusation held. `RESULT_oath_v05_precision_2026_07_13.md`.

## Draft `.zenodo.json` metadata (fill version + publication_date at deposit time)

```json
{
  "upload_type": "publication",
  "publication_type": "article",
  "title": "styxx: the erasure bound — a substrate honesty read coupled to knowledge, with a re-runnable erasure-resistance certificate",
  "creators": [{"name": "Rodabaugh, Alexander", "affiliation": "Fathom Intelligence"}],
  "related_identifiers": [
    {"identifier": "10.5281/zenodo.19326174", "relation": "isVersionOf", "scheme": "doi"},
    {"identifier": "10.5281/zenodo.21263158", "relation": "isNewVersionOf", "scheme": "doi"}
  ],
  "license": "cc-by-4.0",
  "keywords": ["read-write-dissociation", "subspace-erasure", "honesty-probe", "adversarial-robustness",
               "conscience-mount", "OATH", "machine-integrity"]
}
```

## Draft version description (for the Zenodo deposit body)

> This version adds the removal-class arc to the read≠write program: whether the substrate honesty
> read can be ERASED (not merely redirected) with knowledge held. Across static erasure, adaptive
> re-fit erasure, [3B scale], and a dose-response sweep of the erased rank, the read and the retained
> knowledge behave as coupled — the attacker cannot remove the read without paying knowledge. The
> version ships the erasure-resistance certificate (styxx.ladder): a machine-verifiable, self-tamper-
> checking robustness object that surfaces measured breaks at equal prominence and names its own
> unbounded flanks. Includes a mechanism correction to the prior version's "privacy" attribution
> (now capacity-dominated; a refinement, not a retraction). Every claim is OATH-certified and
> re-runnable on an 8GB consumer GPU; see REPLICATIONS.md for the replication bounty.

---
*Staged 2026-07-13. Deposit is operator-gated and blocked on login; this draft makes it a paste-and-
publish once B7 + coupling land and main is fast-forwarded. arXiv upload of the same bundle is the
paired owed action — arXiv is the discoverability fix, Zenodo is the archival DOI.*
