# DEPOSIT PACKAGE — Frame-Locality (ready for operator to deposit)

**The agent will NOT fire the deposit: it is permanent, under a real name, credentialed, and the paper
has not yet had human review. This package makes the operator's part copy-paste. Read the paper once,
then deposit.**

## Pre-deposit checklist (operator, ~10 min)

1. **Read `PAPER_frame_locality_2026_07_28.md` end to end.** It is AI-drafted and receipt-certified but
   human-unreviewed. Confirm every claim is one you're willing to attach your name to.
2. Confirm the honest caveats survived (they must): recovery rate "about half," near the floor; weight
   channel is 1.5B / one attack class; coupling is behavioral not probe-level.
3. Confirm scope-note decisions (below) are reflected or explicitly deferred.
4. Only then run your existing Zenodo deposit flow (token in `secrets/arxiv-creds.txt`, per prior
   Fathom deposits v26–v30).

## Zenodo metadata (paste into the deposit form)

- **Title:** Frame-Locality: Where Corruption Captures a Language Model's Report, and Where It Reaches the Belief
- **Creators:** Rodabaugh, Alexander (Fathom Lab)
- **Resource type:** Publication → Preprint
- **Description:** *(use the paper's Abstract verbatim — it is receipt-grounded and OATH-held)*
- **Keywords:** machine honesty; sycophancy; belief elicitation; knowledge editing; LoRA;
  specificity control; preregistration; AI evaluation; interpretability; integrity instruments
- **License:** CC-BY-4.0 (matches the open-core stance) — operator confirms
- **Related identifiers** (this program's prior deposits, as "isPartOf" / "continues"):
  - Fathom v30 — 10.5281/zenodo.21522035 (anchored validity)
  - Fathom v28 — 10.5281/zenodo.21263158 (read≠write)
  - Fathom v26 — 10.5281/zenodo.21241185 (calibration poisoning)
- **Version:** next Fathom sequence number (operator assigns; do NOT let the agent bump)

## Files to upload

- `papers/PAPER_frame_locality_2026_07_28.md` (the paper)
- `papers/PAPER_frame_locality_2026_07_28.certificate.json` (its OATH certificate)
- `papers/SYNTHESIS_frame_locality_2026_07_28.md` (the cross-channel map + full receipt table)
- Optionally bundle the receipts below so the deposit is self-verifying.

## Receipt manifest (every number in the paper binds to one of these)

- agent-conscience/frame_recovery_result.json — social pressure, 3B recovery/specificity
- agent-conscience/scale_test_result.json — 7B cave
- agent-conscience/frontier_knowsay_result.json — frontier cave
- agent-conscience/frontier_recovery_result.json — frontier recovery (powered)
- agent-conscience/adjudicated_loop_result.json — frame-vs-parameter (0.2742 / 0.8226)
- grounded-honesty-axis/injection_gap_closure_result.json — context injection
- closed-model-frontier/behavioral_sycophancy_b22_result.json — silent sycophancy
- read-neq-write/e1_result.json — probe-level knowledge survival
- agent-conscience/poisoned_recovery_result.json — unregularized weight attack
- agent-conscience/kp_recovery_result.json — knowledge-preserving attack
- agent-conscience/kp_replication_result.json — replication (ARC, 2nd seed)
- agent-conscience/coupling_battery_result.json — capability coupling (22.7 vs 0.0)

## Reproduction line (goes in the description or a README)

> Every quantity is from a preregistered run with frozen numeric gates and a machine-checkable
> certificate. From the repository: `python -m styxx.certify papers/PAPER_frame_locality_2026_07_28.md
> <receipts…>` re-derives the verdict (OATH-HELD). Open-model results run on a single 8 GB consumer GPU.

## The other three "all of them" items — status for the operator

- **PyPI release with `knowsay`** — HARD RAIL in AUTOPILOT.md (`never: PyPI publish, version bump,
  tag`). Agent will not do it. When you're ready: bump `styxx/_version.py`, tag `v*.*.*`, the CI ships
  it (~90s). knowsay already has tests (suite 1811/8) and its datasheet.
- **External replication invite** — drafted and staged in `OUTREACH_external_replication_2026_07_28.md`.
  You pick the recipient and send; the agent won't cold-contact a real org on your behalf.
- **Scope-note sign-off** — the two DRAFT notes
  (`calib-poison-general/SCOPE_NOTE_privacy_vs_capacity_2026_07_09.md`,
  `read-neq-write/SCOPE_NOTE_probe_survival_is_not_behavioral_survival_2026_07_28.md`) propose wording
  corrections to already-deposited papers and to `styxx.mount.relock` docs. Applying them touches
  shipped surfaces (a new Zenodo version + a code docstring change), so they wait for your explicit
  go on the specific wording — say the word and the agent will apply the in-repo edits and run the
  suite, leaving only any re-deposit to you.

## Status

DEPOSIT PACKAGE READY — agent-prepared, operator-fired. The paper, certificate, synthesis, and all
receipts are already public on the `paper/anchored-validity` branch of fathom-lab/styxx (tip
`195762b`); depositing assigns the permanent citable DOI.

---

## STATUS 2026-07-28 — DRAFT CREATED IN THE CORRECT LINEAGE (agent), files + publish remain

Operator asked for the deposit as **a new version of the existing series, not an orphan**. Done:

- **Concept record: `19326174`** — the Fathom series (v30 = `21522035` sits under it). Confirmed by
  API, not assumed.
- **New version draft created: deposition `21659191`**, `conceptdoi 10.5281/zenodo.19326174`
  inherited — so it cites as the next version of the same record, exactly as asked.
- **Reserved version DOI: `10.5281/zenodo.21659191`** (activates on publish).
- **Metadata SET** (mirrors v30's schema): title "Fathom v31 / styxx: Frame-Locality — Where
  Corruption Captures a Language Model's Report, and Where It Reaches the Belief"; creator
  "Rodabaugh, Alexander / Fathom Lab"; preprint; CC-BY-4.0; version `v31`; date 2026-07-28;
  keywords; related identifiers (GitHub + PyPI). Description carries the abstract **including the
  stated limits** (recovery rate near one-half; 1.5B / one attack class; coupling is behavioral).
- **Edit link:** https://zenodo.org/deposit/21659191

### What is NOT done, and why

**File upload is blocked from this environment.** Both Zenodo upload paths fail here: the bucket
`PUT` returns 400 "file upload transfer failed" and even a 4-byte probe returns **504 Gateway
Time-out**; the multipart `POST` times out. JSON calls (new-version, metadata) work fine — the
sandbox's egress does not carry request bodies to this host. This is an environment limit, not a
Zenodo or credential problem. **The inherited v30 files were removed first, so the draft currently
has zero files — it therefore cannot be published accidentally.**

### Operator: two steps, ~2 minutes

1. Open https://zenodo.org/deposit/21659191 and drag in:
   - `papers/PAPER_frame_locality_2026_07_28.md` (upload as `source.md` to match series convention)
   - `papers/PAPER_frame_locality_2026_07_28.certificate.json` (as `source.certificate.json`)
   - `papers/SYNTHESIS_frame_locality_2026_07_28.md`
   - `papers/REPRODUCTION_frame_locality_2026_07_28.md`
   - optionally the six receipts listed in the manifest above, so the record is self-verifying
2. **Read the paper once**, then press Publish. The pre-deposit checklist at the top of this file
   still stands: it is receipt-certified but human-unreviewed, and publishing mints a permanent DOI
   under a real name.

If the draft is ever abandoned, discard it in the Zenodo UI — nothing is public until Publish.
