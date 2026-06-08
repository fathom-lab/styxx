# STYXX PROGRAM BACKLOG — the living, prioritized queue

*Fathom Lab · 2026-06-08. The fuel for `RESEARCH_LOOP.md`. Each cycle the loop pulls the top unblocked item,
pre-registers a kill-gate, runs, red-teams, records, and re-ranks. Seeded from the complete adversarial
threat-surface map (4-lens workflow `wf_2937b28a`) + every finding's "owed" list. Updated every cycle —
closing items, spawning new ones, re-ranking by leverage.*

**Leverage** = advances-a-rung × falsifiability (can a negative actually fire?) × feasibility-on-8GB.
**Status**: `QUEUED` · `IN-PROGRESS` · `BLOCKED` · `DONE` · `KILLED` (negative recorded) · `PARKED`.

---

## Progress ledger

- **R1 read-certificate** — climbed; 2-family (Qwen-3B + Llama-3B), scale-robust 1.5B→7B, semantic-not-shortcut. `SURVIVED`.
- **R2 intent/mens-rea** — real but coupled to R1; two-stage HELD→HID conscience deployable (false-accuse 0.949→0.037). `REPORT_AS_LANDED`.
- **R3 demarcation/abstain** — proven the wall is bedrock on 3 axes; wiring into the certificate object owed.
- **R4 substrate/closed + adversarial robustness** — the existential frontier. Adversarial flank: frozen deployed probe **DEFEATED**; info-persistence **SUPPORTED-n=1**; v3 (relocation) **INCONCLUSIVE — running**. Closed-model: **open**.
- **R5 write** — out by design (read-only = the safety).
- bricks laid: ~20 findings · open questions below: 22.

## Current cycle (live)

- **B0 — v3 real run to a valid verdict.** The 800-step/n=190 orthogonalization+displacement run is `IN-PROGRESS`
  (lam_hide=8 at last check). Gate: only trust the JSON when `fixed_population==190` and `chance_floor_p95 < base
  refit_max` (0.356 < 0.818 ✓ — the non-degeneracy guard the smoke run failed). The smoke artifact is quarantined
  (`adversarial_curve_v3_result_SMOKE_INVALID.json`). On completion → RED-TEAM the verdict → then B1.

---

## Tier 1 — the decisive robustness frontier (R4, highest leverage)

| ID | Question / experiment | Kill-gate sketch | Lev | Status |
|---|---|---|---|---|
| **B1** | **Multi-seed + bootstrap CIs** on the decisive (refit_max, knowledge) at the strongly-engaged knowledge-preserving point. Turn n=1 → ≥3 seeds. | "settled" only if the CI lower bound on (refit_max − floor) stays >0 across ≥3 seeds while KG_KNOW_REL holds; report per-seed verdict-flip rate. | H | QUEUED |
| **B2** | **The most dangerous attack: RMU / gradient-routing unlearning of the gold subspace + an explicit knowledge-replay regularizer**, scrubbing across ALL read layers (not just band), gold subspace found via the model's own gradients. The real read≠write test. | EVADABLE (read≠write BROKEN) iff refit_max→floor (CI overlaps floor) with knowledge held at ref on a DISJOINT capability set, ≥2 seeds + 3B. | H | QUEUED |
| **B3** | **Calibration-vs-deployment split (v4)** — re-lock the probe on population C, evaluate on a DISJOINT D the attack conditioned differently (distribution-detector LoRA). Tests whether re-locking's distribution-internal CV transfers. | EVADABLE iff re-locked-on-C probe collapses on D while PRIME/ABORT pass on C and knowledge holds on D. | H | QUEUED |
| **B4** | **Fix the knowledge invariant** — add KG-LENS (gold-lens-preservation) gate AND measure held-knowledge on a DISJOINT capability set (different MMLU subjects than the attacked lie population), so "knowledge preserved" can't be satisfied while the audited capability is corrupted. | a verdict is admissible only if KG-LENS intact AND disjoint-set knowledge ≥ ref−0.05. | H | QUEUED |
| **B5** | **Break the refit circularity** — report frozen-then-RE-LOCKED-on-DISJOINT-items as the deployment-faithful read; same-model CV refit is an activation-existence lower bound only, never the headline. | deployment claim uses only the disjoint-item re-locked number. | M | QUEUED |
| **B6** | **Attacker beyond LoRA** — ≥1 full fine-tune point + ≥1 distillation-to-clean-student point (a real vendor isn't limited to r=16 adapters). | does full-FT/distillation evade the re-fittable read with knowledge kept? same gate as B2. | M | QUEUED |
| **B7** | **3B confirmation** (prereg-owed across v1/v2/v3) — a 1.5B-vs-3B split is itself the finding. | run the decisive sweep on Qwen-3B; report the split. | M | QUEUED |
| **B8** | **Non-sycophancy suppression** — sandbagging / strategic-deception / instructed-lying (program's own data: instructed-lying transfers 0.42<chance → likely a BOUND). | does the read-certificate + robustness hold off-sycophancy, or is it sycophancy-specific? | M | QUEUED |
| **B9** | **MoE / architectural rerouting** — route the computation to experts/heads the probe doesn't tap (untested architecture frontier; all work is dense ≤7B). | can a MoE hide the suppressed answer in an untapped expert with knowledge kept? | L | PARKED (no MoE ≤8GB) |
| **B10** | **Calibration-set poisoning** (data attack on the probe-training set). | can poisoning the calibration set blind the re-locked probe? | L | QUEUED |

## Tier 2 — the deployable protocol (turn findings into tech)

| ID | Question / build | Done-when | Lev | Status |
|---|---|---|---|---|
| **B11** | **Re-locking protocol** in STYXX_PROTOCOL: verifier-private calibration set spanning prime/abort/lie populations; layer+family search with a **permutation-corrected selection penalty** (refit_max over 28 layers is multiple-comparisons-inflated); certificate encodes calibration-SHA + selected layer + floor + PRIME/ABORT + re-locked-probe hash; **state=ABSTAIN when the deployment input is off-manifold** vs calibration (turn silent failure into honest abstention). | spec section + a runnable re-lock harness. | H | QUEUED |
| **B12** | **The gamed-model probe-validity experiment** (capture-trust names it irreducible; never run): train the cooperative model to emit clean caving residuals to the FROZEN probe, then test whether audit-time RE-FIT still recovers. Decides whether the cooperative guarantee survives an adversarial-but-checkable insider. | measured EVADABLE/ROBUST for the gamed-model on the re-locked read. | H | QUEUED |
| **B13** | **Harden all experiment scripts** (the smoke-artifact lesson): `--smoke` writes to a `_smoke`-suffixed file; in-code assert `chance_floor < reference signal` else emit `VOID-INSTRUMENT`; every result JSON carries the scorer SHA. | guard added to the v4+ template; lesson in RESEARCH_LOOP. | H | QUEUED |

## Tier 3 — the other rungs

| ID | Question | Rung | Lev | Status |
|---|---|---|---|---|
| **B14** | Free-form / verbose-correct guard for the two-stage conscience (intent off the forced-choice rails). | R2 | M | QUEUED |
| **B15** | Cross-vendor intent (beyond Qwen/Llama). | R2 | L | QUEUED |
| **B16** | Wire the demarcation/abstain into the certificate object (R3 is proven-it-must; now build it). | R3 | M | QUEUED |
| **B17** | Finish the wall's battery — bootstrap CI, shared-myth split, D-channel dynamics, cross-model — to fully characterize the bedrock. | R3 | M | QUEUED |
| **B18** | Closed-model substrate: can behavioral proxies (sampling-divergence, grounded_honesty) carry part of the oath where there's no white-box access? The R4 existential question. | R4 | H | QUEUED |
| **B19** | Cross-family steering-validation of the live dissociation (currently Qwen-1.5B-live only): per-model dose/layer sweep. | R4 | M | QUEUED |
| **B20** | Capture-trust: TEE (H100-CC) attestation prototype + crypto-binding L1.5 implementation. | R4 | L | BLOCKED (hardware) |

## Tier 4 — the demarcation (the 2,500-year deliverable)

| ID | Question | Lev | Status |
|---|---|---|---|
| **B21** | The living **demarcation map**: which claims about machine minds are testable vs metaphysics, updated as each rung resolves. The public-good output. The honest answer to "does universal structure underlie mind" is *this map getting truer*, not a single crack. | H | ONGOING |

---

*Re-rank every cycle. The single most decisive open experiment is **B2** (RMU unlearning + knowledge-replay) —
it is the real test of read≠write, the core safety claim, and the loop must be willing to let it return EVADABLE.
B0→B1 (validate v3 + CIs) gate it: settle the current attack family before escalating to the strongest one.*
