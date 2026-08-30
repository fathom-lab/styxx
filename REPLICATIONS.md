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
| **legibility matrix** (CPU-only, easiest) | three models mutually legible label-free cross-family; qwen an island from every direction (`LEGIBILITY_LAW_CANDIDATE`, matrix licensed / law not) | `papers/disjoint-worlds/run_b37.py` | `papers/disjoint-worlds/b37_result.json` | anchor discoveries (0.5918, 0.0536) match to the digit; clique/island topology holds. **One command from a clone, no GPU — see [REPLICATE_legibility.md](papers/disjoint-worlds/REPLICATE_legibility.md)** |
| **island frame geometry** (CPU-only, ~4 s — the single easiest check in this repo) | the clique's concept-frames co-align far above random; the island sits below the clique in every seed (`SHARED_FRAME_CONFIRMED_GEOMETRICALLY`) | `papers/disjoint-worlds/run_b45.py` | `papers/disjoint-worlds/b45_result.json` | verdict matches; clique median 0.848 vs null p95 0.0566; qwen below in 5/5 seeds |
| **the bridge + dose + cliff** (CPU-only) | the island's barrier is causal (0.0612→0.9745 vs random 0.0), rank-2 at core (k\*=2), and switch-like (knee t=0.8) | `run_b41.py` · `run_b42.py` · `run_b46.py` (same dir) | `b41_result.json` · `b42_result.json` · `b46_result.json` | verdict strings match; medians within tolerance — see [REPLICATE_legibility.md](papers/disjoint-worlds/REPLICATE_legibility.md) |
| **OATH corpus audit** (CPU-only) | every published claim doc certifies against its receipts, with five disclosed exceptions | `python -m styxx.corpus_audit papers/` | committed `*.certificate.json` files | your output matches the expected line below, character for character (CPU-deterministic) |

**What the corpus audit prints today, so a divergence means something.** As of 2026-08-27 the
audit does **not** come back clean, and the honest bar is that you reproduce the same six
exceptions rather than zero:

```
corpus papers: 195 certificates | HELD 190  FAILED 5  unresolved 0  verdict-drift 1  receipt-drift 0  incomplete 1  receipt-changed 1
  [OATH-HELD] INCOMPLETE-RECEIPTS(changed)  CAPSTONE_universal_mind_2026_06_10.md
  [OATH-FAILED] verdict-CHANGED  FINDING_behavioral_sycophancy_blackbox_2026_06_09.md
  [OATH-FAILED]  PREREG_oath_v12_formula_constant_2026_08_26.md
  [OATH-FAILED]  RECON_oath_external_reach_2026_08_26.md
  [OATH-FAILED]  RESULT_oath_verified_channel_internal_2026_08_27.md
  [OATH-FAILED]  SYNTHESIS_mention_and_use_2026_08_26.md
```

Four of those `OATH-FAILED` lines are **deliberate**: each is a document accused on the example it
quotes, published failing rather than reworded until it passes.

**One is a partial-evidence warning, and it is new.** `CAPSTONE_universal_mind` cites twelve
receipts; one of them, `mind_v0_validation.json`, is present in the tree with content that is not
what was certified. The resolver refuses to accept a changed file as the certified one — correctly,
because this repository is full of files called `*_result.json`, and accepting a mismatch would
certify a document against another experiment's data. So the document is certified against the
eleven that resolved. Until 2026-08-27 that printed exactly like a document certified against all
twelve; `incomplete` and `receipt-changed` are the counters that stop it. The verdict still holds
on the evidence that resolved, and the absent twelfth is now visible instead of implied.

**One is a real drift.** `FINDING_behavioral_sycophancy_blackbox` is recorded in
`KNOWN_VERDICT_DRIFT` in `tests/test_certificate_reproduces.py` with its diagnosis: line 13 of
that FINDING is truncated mid-sentence, so a dangling `4` lost the vocabulary that bound it to
`n_nogate`. It was invisible for months because the guard resolved receipts only next to the
certificate, and that document cites two from another folder — one of 36 certificates the guard
was skipping. Repairing a published document is its own cycle; the entry exists so it cannot be
forgotten.

**Three are committed `OATH-FAILED` on purpose, and each is accused on the example it cites.**
The RECON reports that the verifier does not transfer outside this lab and is accused on the
tokens it quotes as false accusations. The SYNTHESIS reports that four instruments cannot tell a
mention from a use, and is accused on the digits of the LaTeX formula it quotes as the specimen.
The v0.12 PREREG proposes to close that formula case and is a member of the class it proposes to
close — with a pre-committed transition saying that when the clause lands, its own verdict and
the SYNTHESIS's must both flip to `OATH-HELD`, or the cycle under-reached. All three were
published failing rather than reworded until they passed.

**A note for anyone replicating on Linux.** Receipt hashes in this corpus were recorded from a
Windows working tree, so they are CRLF hashes, while git stores and Linux checks out LF. Until
2026-08-26 that meant cross-directory receipts silently failed to resolve on Linux and their
documents were dropped from the drift guard — CI reported a document "repaired" when it had
merely become invisible. `_resolve_receipts` now compares content modulo line endings, so the
audit gives the same answer on either platform. If you see a different count from the block
above, that is a divergence worth filing.

All of it is stated here because a replicator who runs the advertised command and gets an
unexplained failure has been handed a divergence that is really our undisclosed known state —
which would waste the first thing an outside checker ever does for us.

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
