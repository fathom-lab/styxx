# styxx replications — run the receipts, get named

**The claims below ship with receipts: the GPU targets re-run on an 8GB consumer GPU, the rest on
a CPU, and one needs only reading. This
file converts that from a statement into a ledger. The first independent re-run of each target
earns a named credit line in the next Zenodo version of the corresponding paper and a permanent row
here. Zero replications after six months is also an answer, and we will publish that too.**

## Why this exists

A moat nobody has tried to cross is indistinguishable from a field. The claims below are only as
strong as the number of people who have independently re-run them. We are asking you to try to
break them — a replication that *fails* to match is MORE valuable than one that matches. It gets its
row here like any other, and a verdict flip that survives our re-run of your exact setup earns
co-credit on the correction (see Credit).

## Targets (in order of leverage)

| target | claim | frozen code | canonical receipt | replication bar |
|---|---|---|---|---|
| **EXTERNAL-1 re-adjudication** (reading only, no GPU, no code; [#43](https://github.com/fathom-lab/styxx/issues/43)) | the diff gate's accusations on external agent pull requests: a blind panel upheld 23 of 100, precision 0.23 against a preregistered 0.95 floor, with 30 of 30 decoys called correctly | `papers/closed-model-frontier/external1_packet.json` (130 items, no verdicts) and `papers/closed-model-frontier/external1_key_digest.txt` (the sealed key's salted SHA-256), rules in `papers/closed-model-frontier/PREREG_external1_aidev_2026_08_31.md`; you post SUPPORTED or CONTRADICTED per item, and the lab scores them against the sealed key as `papers/closed-model-frontier/external1_packet.py score` scored ours | `papers/closed-model-frontier/external1_adjudication.json`; our panel's answers `papers/closed-model-frontier/external1_answers.json` | 27 of 30 decoys or the adjudication voids, for you as for us; agreement with our answers published item by item, disagreements named. The published packet's item ids carry the arm, and its 15 synthetic decoys carry a visible `zz_` path perturbation ([#125](https://github.com/fathom-lab/styxx/issues/125)): judge the claims without reading the ids, or the decoy gate measures nothing; the `zz_` paths weaken its synthetic half even so |
| **collateral census** (CPU-only, deterministic; the corpus is downloaded; [#44](https://github.com/fathom-lab/styxx/issues/44)) | of 1,386,104 files changed by 69,654 AIDev agent pull requests, 10.96% are collateral (lockfile, generated, snapshot, migration, whitespace-equivalent), each file in exactly one class | `pip install -e . datasets`, then `python papers/closed-model-frontier/external1_harness.py --stage shelf`, then `python papers/closed-model-frontier/collateral_census.py` (it rewrites `collateral_census.json` in place; `git diff` it) | `papers/closed-model-frontier/collateral_census.json` | every count matches; name the dataset revision you read. `whitespace-equivalent` is a lower bound by design, and an AST comparison that moves the headline is the more valuable result (#44) |
| **certify silent-pass census** (CPU-only, a few minutes) | mutate one significant digit of 3951 claims the verifier certified VERIFIED: 2696 are not accused (0.6824), 604 of them VERIFIED against an unrelated leaf (README, `certify` row) | `git checkout e463824f`, then `python papers/closed-model-frontier/oath_v07_silentpass_census.py` (seed 1; it rewrites its JSON in place) | `papers/closed-model-frontier/oath_v07_silentpass_census.json` | every field matches. `verifier_sha256` (`2bd7299f…`) is the sha256 of `styxx/certify.py` as a CRLF checkout reads it, so it matches only on a CRLF checkout (an LF checkout reads `d84bb373…`). That commit predates the 2026-08-26 line-ending fix described below, so a Linux run may diverge for that reason |
| **BENCH-2 rows** (CPU-only; the diffs are downloaded) | 604 claims from 568 public agent pull requests, each with the sha256 of its diff and no verdict of ours | `python papers/closed-model-frontier/bench_reproduce.py --fetch`, then `--oracle`, then `--score styxx` or `--score yourmodule:check` | the `diff_sha256` column of `papers/closed-model-frontier/bench2_dataset.jsonl` | every re-fetched diff matches its published sha256. Both benchmarks voided themselves on their own audit gates (`papers/closed-model-frontier/RESULT_bench2_INVALID_2026_09_17.md`), so the rows are data, not a validated benchmark |
| **B2-adaptive** | the chasing eraser never converges; private-13 read survives (`SURVIVES__vs_adaptive_erasure_1p5B`) | `papers/calib-poison-general/b2_adaptive_erasure.py` | `papers/calib-poison-general/b2_adaptive_erasure_result.json` | verdict string matches; per-cell `private13_auroc` within tolerance |
| **B2 static** | subspace verifiably zeroed, signal relocates (`SURVIVES__vs_subspace_erasure_1p5B`) | `papers/calib-poison-general/b2_subspace_erasure.py` | `papers/calib-poison-general/b2_subspace_erasure_result.json` | same |
| **parity control** | the flagship private>naive recovery is probe-capacity-dominated (`PRIVACY_PARTIAL__between_bars`) | `papers/calib-poison-general/honesty_parity_control.py` | `papers/calib-poison-general/honesty_parity_control_result.json` | verdict matches; `parity_gap` sign per cell matches |
| **E1** | stage-2 "evasion" was calibration poisoning; private-calibration recovers the read | `papers/read-neq-write/e1_clean_auditor.py` | `papers/read-neq-write/e1_result.json` | verdict matches; recovery reads within tolerance |
| **legibility matrix** (CPU-only, easiest) | three models mutually legible label-free cross-family; qwen an island from every direction (`LEGIBILITY_LAW_CANDIDATE`, matrix licensed / law not) | `papers/disjoint-worlds/run_b37.py` | `papers/disjoint-worlds/b37_result.json` | anchor discoveries (0.5918, 0.0536) match to the digit; clique/island topology holds. **One command from a clone, no GPU — see [REPLICATE_legibility.md](papers/disjoint-worlds/REPLICATE_legibility.md)** |
| **island frame geometry** (CPU-only, ~4 s — the single easiest check in this repo) | the clique's concept-frames co-align far above random; the island sits below the clique in every seed (`SHARED_FRAME_CONFIRMED_GEOMETRICALLY`) | `papers/disjoint-worlds/run_b45.py` | `papers/disjoint-worlds/b45_result.json` | verdict matches; clique median 0.848 vs null p95 0.0566; qwen below in 5/5 seeds |
| **the bridge + dose + cliff** (CPU-only) | the island's barrier is causal (0.0612→0.9745 vs random 0.0), rank-2 at core (k\*=2), and switch-like (knee t=0.8) | `run_b41.py` · `run_b42.py` · `run_b46.py` (same dir) | `b41_result.json` · `b42_result.json` · `b46_result.json` | verdict strings match; medians within tolerance — see [REPLICATE_legibility.md](papers/disjoint-worlds/REPLICATE_legibility.md) |
| **OATH corpus audit** (CPU-only) | every published claim doc certifies against its receipts, with five disclosed exceptions | `python -m styxx.corpus_audit papers/` | committed `*.certificate.json` files | your output matches the expected line below, character for character (CPU-deterministic) |
| **islands demo** (CPU-only, seconds) | from frame geometry alone, with no labels, the instrument separates a planted island from a seeded synthetic cohort (README, MEASURE) | `python -m styxx.islands --demo` | the printed block in README.md, MEASURE | `ISLANDS_PRESENT`, the planted member the only island listed, mind_0 to mind_6 between 0.4407 and 0.4529, ISLAND 0.1636; the same cohort has differed by up to about 0.005 between machines, cause not established ([#93](https://github.com/fathom-lab/styxx/issues/93)) |
| **sense demo** (CPU-only, seconds) | a channel sharing the agent's latent state reads `COUPLED_BEYOND_CONFOUND__attribution_pending`, an unrelated one `NO_DETECTABLE_COUPLING__above_measured_floor`, and no stronger verdict exists (README, SENSE) | `python -m styxx.sense --demo` | none committed: SENSE has no measurement claim to replicate yet, and the demo is the check | both verdict strings match |

**What the corpus audit prints today, so a divergence means something.** As of 2026-09-01 the
audit does **not** come back clean, and the honest bar is that you reproduce the same nine
exceptions rather than zero:

```
corpus papers: 208 certificates | HELD 200  FAILED 8  unresolved 0  verdict-drift 1  receipt-drift 0  incomplete 1  receipt-changed 1
  [OATH-HELD] INCOMPLETE-RECEIPTS(changed)  CAPSTONE_universal_mind_2026_06_10.md
  [OATH-FAILED]  ANALYSIS_base_rate_ceiling_2026_09_01.md
  [OATH-FAILED] verdict-CHANGED  FINDING_behavioral_sycophancy_blackbox_2026_06_09.md
  [OATH-FAILED]  PREREG_oath_v12_formula_constant_2026_08_26.md
  [OATH-FAILED]  RECON_oath_external_reach_2026_08_26.md
  [OATH-FAILED]  RESULT_oath_verified_channel_internal_2026_08_27.md
  [OATH-FAILED]  RESULT_obligate1_does_not_ship_2026_08_31.md
  [OATH-FAILED]  RESULT_struct1_beats_the_null_2026_08_31.md
  [OATH-FAILED]  SYNTHESIS_mention_and_use_2026_08_26.md
```

This block had itself gone stale, which is worth recording rather than quietly fixing. It
listed six exceptions while the audit printed nine: `RESULT_obligate1_does_not_ship` and
`RESULT_struct1_beats_the_null` were published OATH-FAILED on 2026-08-31 and never added here,
and `ANALYSIS_base_rate_ceiling` joined them on 2026-09-01. The pinned counts were tested;
the pinned **list** was not, so only the numbers were kept honest.

`receipt-drift` reads 0 again after a repair on 2026-09-01. It had silently been 1:
`external1_summary.json` was regenerated by the CORRECTION commit of 2026-08-31 under a
certificate that had already sworn to the older bytes. Re-certifying against the corrected
receipt returns the identical verdict and identical counts — the correction moved the bytes but
no claim the paper swears to — so the certificate was re-issued rather than excused. This is
the third document hit by the same defect, after `CORPUS_STATE` earlier the same day: **a
receipt is history too**, and regenerating one in place silently invalidates every certificate
citing it.

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

**The `binding:` line, and what it is for.** Since `SPEC_oath_receipt_binding_2026_09_04.md`
the audit prints a third line beneath the verdict line and the uncovered line. On a full clone
(`--history auto`, the default, or `on`) it reads
`binding: over N/M certificates — citations same … at-issue … elsewhere … unbacked …
unrecoverable … | documents same … at-issue … moved … | certificates: …` — for every receipt a
certificate cites, where the bytes it swore to are: in the working tree (anywhere in it; a
receipt outside the audit root is `same` with a note), at the certificate's issuing commit,
elsewhere in history, nowhere in history, or unanswerable; the same for the document; and for
every certificate whether the current verifier reproduces the recorded verdict class over the
document and receipt bytes at the issuing commit (`stands-over-sworn-bytes`, null with a reason
when those bytes cannot all be fetched). On a shallow clone, or with `--history off`, it reads
`binding: history unavailable (…)` and no cell is filled; `--history on` additionally exits 2.
CI never runs this audit, so nothing in CI prints either line; the shallow reason is pinned by
`tests/test_certify_by_digest.py` on a clone the test builds. **The first line is byte-identical
in every mode; it is the only line this file pins.** The corpus census over all 213 tracked
certificates is `papers/closed-model-frontier/receipt_binding_census_result_2026_09_05.json` (the third run; the
second, `receipt_binding_census_result.json`, stays as committed); the census
refuses to overwrite a tracked result, so a new census is a new dated file; its reading is
`RESULT_oath_receipt_binding_2026_09_05.md`, sworn to its leaves. One thing it settles for the
block above: `CAPSTONE_universal_mind`'s changed twelfth receipt has its sworn bytes at the
issuing commit, and the certificate stands over the full set — the receipt moved seventeen
minutes after issue on 2026-06-10; the certificate stands over the bytes it swore to. Whether
those bytes were true is not a question this audit answers.

All of it is stated here because a replicator who runs the advertised command and gets an
unexplained failure has been handed a divergence that is really our undisclosed known state —
which would waste the first thing an outside checker ever does for us.

**Tolerance for GPU targets:** bf16 CUDA training is non-deterministic across hardware; the honest
replication bar is the frozen VERDICT STRING plus per-cell decisive reads within ±0.05 AUROC, with
your per-cell deltas disclosed in the PR. A CPU target with a command must match exactly, except
where its bar states a tolerance (the islands demo); EXTERNAL-1 is a judgement, and its bar is the
decoy gate.

## How to replicate

1. Clone at the commit recorded in the target's RESULT doc (each RESULT names its frozen commit).
2. Run the frozen script verbatim — no flags beyond those in the RESULT's Reproducibility section.
   GPU targets need ~8GB VRAM and a few hours; the CPU targets need no GPU, and EXTERNAL-1 needs
   no code.
3. Open a PR titled `[replication] <target> — <your name/handle>` adding:
   - your result JSON at `replications/<target>__<handle>.json` (verbatim, unedited);
   - one row to the ledger below (date, hardware, verdict match yes/no, max per-cell delta).
4. CI runs `python scripts/verify_replication.py <target> replications/<your file>` — it checks the
   verdict string and tolerances against the canonical receipt and posts the comparison. Honest
   mismatches are merged too, labeled `replication-divergent` — divergence is data.
   `verify_replication.py` knows the four GPU targets (B2-adaptive, B2 static, parity control,
   E1). For the others the comparison is the `git diff` of the receipt the command rewrote, or,
   for EXTERNAL-1, the lab's item-by-item comparison, posted in the issue.

## Credit

- First matching replication per target: named acknowledgment line in the next Zenodo version of
  the paper that claim belongs to, plus your row here, permanently.
- First DIVERGENT replication per target (verdict flip that survives our re-run of your exact
  setup): co-credit on the correction note itself. Breaking our claim earns more than confirming it.

## Ledger

| date | target | replicator | hardware | verdict match | max per-cell delta | PR |
|---|---|---|---|---|---|---|
| — | — | *none yet — be first* | — | — | — | — |

## Sworn-document replications (added 2026-09-13)

A sworn document (`papers/**/*.md` with a `.sworn-receipt.json` beside it) is replicated by
`python -m styxx.challenge`, which re-derives the receipt at the commit it names and writes a
record plus your own receipt — see `papers/plates/SAND_CHECK.md` §3. The record is a self-report;
the lab re-runs it before a row lands here. Rows are added by the lab from the issue you open.

| date | document | commit | replicator | machine | agree | record_sha256 |
|---|---|---|---|---|---|---|
| 2026-09-13 | `RESULT_checksum_smollm_quant_2026_09_13.md` | `60fad45c` | the lab (second machine; not independent, not credited) | Windows 11, cpu | yes (receipt) — the recipe's magnitudes did not, see `papers/checksum/NOTE_replication_alienware_2026_09_13.md` | in the NOTE |

---
*Published 2026-07-13. If this table is still empty in 2027-01, that emptiness will be reported
as-is in the next paper's limitations section: unreplicated is a property of a claim, and we do not
hide properties of claims.*
