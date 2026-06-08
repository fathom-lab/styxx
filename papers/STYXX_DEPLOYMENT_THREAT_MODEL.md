# styxx — deployment threat model + the re-locking protocol

*Fathom Lab · 2026-06-08. Turns the night's adversarial findings into deployment architecture. Grounded in
the 4-lens adversarial threat-surface map (`wf_2937b28a`) and the v2/v3 results. Honest by construction: it
states what styxx's read-certificate can be trusted to prove, **for whom**, and where it provably cannot.
DRAFT — the one decisive assumption (distribution-invariance of the re-locked read) is still untested; the
experiment that tests it is named below.*

---

## The deployment problem (what v2/v3 proved)

A styxx read-certificate is a probe that recovers a model's known-then-suppressed answer from its residuals.
The adversarial work established two things about shipping it:

1. **A frozen, deploy-once probe is defeated by *any* weight change** — adversarial *or benign*. v2: the frozen
   base probe collapsed to chance (0.13–0.30) at every attacked point with knowledge intact, and ordinary
   cave-only fine-tuning alone halved it (0.798 → 0.412). A certificate cannot be issued once and trusted
   across a fine-tune.
2. **The information itself is harder to destroy** (supported, not settled): a re-fittable probe recovers the
   suppressed answer above the floor unless the model is unlearned. v3 confirmed an attack on the *known*
   direction merely *relocated* the trace off-band (recoverable at L27), it did not erase it.

The deployment consequence is forced: **the certificate must be *re-locked* — re-fit on the audited model at
certification time — not deployed frozen.** This document specifies that protocol and bounds exactly when it
can be trusted.

## The re-locking protocol (L-RELOCK)

The verifier — never the reporter — owns the probe, and refits it per audit.

1. **Inputs the verifier must hold:** (a) white-box weights of the *exact* audited (post-fine-tune) model;
   (b) a **verifier-private, labeled calibration set** spanning the three certificate populations — a
   *knows-and-caves* (LIE) set, a *never-knew* (ABORT) set, and a *present-held* (PRIME) set — with ground
   truth the **verifier** established, provably disjoint from the model's fine-tune data.
2. **Re-lock:** capture residuals across **all** layers on the calibration set; refit a probe (5×60/40 CV) per
   layer; select the layer maximizing held-out accuracy. The max-over-layers is **selection-inflated** — score
   it against a permutation-corrected floor (`FLOOR_MAX` = p95 of the same max-over-layers statistic under
   shuffled labels), not against raw chance.
3. **Validity gates (the oath's honesty):** accept the re-locked read only if **PRIME** passes (the channel
   reads a present held answer) and **ABORT** collapses (it does not fire on a never-knew model) — so a null
   means "couldn't read," not "the model is clean."
4. **Accept** iff the re-locked read clears the corrected floor **and** PRIME ✓ **and** ABORT ✓.
5. **The certificate encodes**, for re-verification without trust: the calibration-set SHA, the selected layer,
   `FLOOR_MAX`, the PRIME/ABORT values, and the hash of the freshly re-locked probe weights.
6. **ABSTAIN on distribution shift (the load-bearing addition):** the re-lock's only honest guarantee is
   held-out *within the calibration distribution*. At inference, if the deployment input is **off-manifold**
   relative to the calibration residuals (residual-space MMD or k-NN density below threshold), the certificate
   must emit **ABSTAIN**, not a green read — turning the silent failure mode below into an honest "cannot
   certify here."

## What re-locking buys — and its trapdoor

- **It demonstrably works for the cooperative case.** Re-locking recovers a read the frozen probe loses: on
  benign fine-tuning, frozen 0.412 → refit 0.675 (+0.26); under the v2 hide attack with knowledge kept, frozen
  0.132 → refit 0.439. So a cooperative deployer who re-locks catches an *accidental* or *non-adaptive*
  insider that suppresses without unlearning.
- **The trapdoor (named, untested): calibration-detection / distribution-shift evasion.** A model trained so
  the gold direction is genuinely **present on the calibration distribution but absent (relocated/whitened)
  in deployment** defeats re-locking *specifically*. The re-locked probe reads clean on the audit items (PRIME,
  ABORT, floor all pass) and is blind on real traffic. **Every v2/v3 experiment shares one population between
  fit and eval, so this class is structurally untestable in the current test bed** — it is a definitional
  blind spot, not a result that the attack fails. The ABSTAIN-on-shift step (6) is the mitigation; whether it
  is *sufficient* is an open empirical question.

## Deployer-class threat model (the honest matrix)

| deployer class | what the read-certificate can be trusted to prove | residual hole |
|---|---|---|
| **Cooperative / regulated, white-box** (the Art. 14/15 case) | With re-locking on self-captured activations after **every** weight change: re-detect a held-and-hidden truth against an **accidental or non-adaptive** insider. | Rests on n=1-engaged, one seed, one model; the read is near-floor under an *adaptive* same-model attack; circular-refit caveat (proves info *exists*, not that a single deployed probe reads production traffic). |
| **Semi-trusted, TEE / crypto-bound** | Attestation (RFC-8785 binding = L1.5; H100-CC TEE = L2) closes residual **fabrication / swap / replay** and makes forgery **attributable**. | Binds bytes, **not probe-validity** or computation-correctness; TEE relocates trust to a silicon TCB with a real side-channel history; attests code *identity*, not that the shipped probe is honest. |
| **Fully-adversarial closed vendor** | **Nothing of the read-certificate.** Only crypto **attribution of who lied** survives — and only if the vendor uses a real identity. | The two irreducible gaps (capture-trust §5): name a model **trained to read clean** ("honest proof of a dishonest model"); **cherry-pick** which inferences are attested. Both are claims about the world *outside* the box the vendor owns. |

**The one-paragraph statement (for STYXX_PROTOCOL):** *styxx's read-certificate is a trust primitive for
**cooperative and regulated** deployers who grant white-box access and a verifier-private calibration set they
did not train against; with per-audit re-locking it re-detects a held-and-hidden truth against accidental or
non-adaptive insiders (a supported-not-settled, n=1-engaged lower bound). It **cannot** be shipped as a frozen
certificate (defeated by any weight change, including benign fine-tuning); it is **not yet** established robust
to an adaptive same-model attack (pending the decisive RMU test); and it proves **nothing** against a
host-controlling vendor who can name a clean-reading model and cherry-pick the attested stream. It is
verify-don't-trust **for cooperative minds**, not a zero-trust guarantee against an adversarial host.*

## The decisive open experiment (backlog B3)

The trapdoor above is the **calibration-vs-deployment split**: re-lock the probe on population C, evaluate on a
**disjoint** deployment population D that a distribution-detector attack conditioned to treat differently.
Pre-register: *EVADABLE iff the re-locked-on-C probe collapses on D while PRIME/ABORT pass on C and knowledge
holds on D.* This is the one assumption (distribution-invariance of the re-locked read) every existing
experiment silently granted; it is the next item after B2 (the RMU read≠write test, currently running).

## Why this is honest tech, not a pitch

The frozen-probe defeat is not buried — it is the headline that *forces* re-locking, and re-locking's own
trapdoor is named as an open experiment, not waved away. styxx is defensible precisely because it states the
deployer classes it does **not** serve as loudly as the one it does. A certificate of a mind's honesty that
over-promised its own scope would be the first thing to burn the credibility that is the entire asset.

Receipts: `FINDING_adversarial_curve_2026_06_07` (frozen-probe defeat + info-persistence), `STYXX_CAPTURE_TRUST.md`
(the irreducible adversarial-host gaps), threat-surface map `wf_2937b28a`, `PROGRAM_BACKLOG.md` (B3, B11, B12).
