# styxx replications — run the receipts, get named

**Every flagship claim in this repo ships with receipts that re-run on an 8GB consumer GPU. This
file converts that from a statement into a ledger. The first independent re-run of each target
earns a named credit line in the next Zenodo version of the corresponding paper and a permanent row
here. Zero replications after six months is also an answer, and we will publish that too.**

## Why this exists

A moat nobody has tried to cross is indistinguishable from a field. The claims below are only as
strong as the number of people who have independently re-run them. We are asking you to try to
break them — a replication that *fails* to match is MORE valuable than one that matches, and earns
the same credit.

## Targets (in order of leverage)

| target | claim | frozen code | canonical receipt | replication bar |
|---|---|---|---|---|
| **B2-adaptive** | the chasing eraser never converges; private-13 read survives (`SURVIVES__vs_adaptive_erasure_1p5B`) | `papers/calib-poison-general/b2_adaptive_erasure.py` | `papers/calib-poison-general/b2_adaptive_erasure_result.json` | verdict string matches; per-cell `private13_auroc` within tolerance |
| **B2 static** | subspace verifiably zeroed, signal relocates (`SURVIVES__vs_subspace_erasure_1p5B`) | `papers/calib-poison-general/b2_subspace_erasure.py` | `papers/calib-poison-general/b2_subspace_erasure_result.json` | same |
| **parity control** | the flagship private>naive recovery is probe-capacity-dominated (`PRIVACY_PARTIAL__between_bars`) | `papers/calib-poison-general/honesty_parity_control.py` | `papers/calib-poison-general/honesty_parity_control_result.json` | verdict matches; `parity_gap` sign per cell matches |
| **E1** | stage-2 "evasion" was calibration poisoning; private-calibration recovers the read | `papers/read-neq-write/e1_clean_auditor.py` | `papers/read-neq-write/e1_result.json` | verdict matches; recovery reads within tolerance |
| **OATH corpus audit** (CPU-only) | every published claim doc certifies against its receipts | `python -m styxx.corpus_audit papers/` | committed `*.certificate.json` files | certificate verdicts match exactly (CPU-deterministic) |

**Tolerance for GPU targets:** bf16 CUDA training is non-deterministic across hardware; the honest
replication bar is the frozen VERDICT STRING plus per-cell decisive reads within ±0.05 AUROC, with
your per-cell deltas disclosed in the PR. The CPU target (corpus audit) must match exactly.

## How to replicate

1. Clone at the commit recorded in the target's RESULT doc (each RESULT names its frozen commit).
2. Run the frozen script verbatim — no flags beyond those in the RESULT's Reproducibility section.
   GPU targets need ~8GB VRAM and a few hours; the corpus audit needs only CPU.
3. Open a PR titled `[replication] <target> — <your name/handle>` adding:
   - your result JSON at `replications/<target>__<handle>.json` (verbatim, unedited);
   - one row to the ledger below (date, hardware, verdict match yes/no, max per-cell delta).
4. CI runs `python scripts/verify_replication.py <target> replications/<your file>` — it checks the
   verdict string and tolerances against the canonical receipt and posts the comparison. Honest
   mismatches are merged too, labeled `replication-divergent` — divergence is data.

## Credit

- First matching replication per target: named acknowledgment line in the next Zenodo version of
  the paper that claim belongs to, plus your row here, permanently.
- First DIVERGENT replication per target (verdict flip that survives our re-run of your exact
  setup): co-credit on the correction note itself. Breaking our claim earns more than confirming it.

## Ledger

| date | target | replicator | hardware | verdict match | max per-cell delta | PR |
|---|---|---|---|---|---|---|
| — | — | *none yet — be first* | — | — | — | — |

---
*Published 2026-07-13. If this table is still empty in 2027-01, that emptiness will be reported
as-is in the next paper's limitations section: unreplicated is a property of a claim, and we do not
hide properties of claims.*
