# STYXX PROGRAM BACKLOG — the living, prioritized queue

*Fathom Lab · 2026-06-08. The fuel for `RESEARCH_LOOP.md`. Each cycle the loop pulls the top unblocked item,
pre-registers a kill-gate, runs, red-teams, records, and re-ranks. Seeded from the complete adversarial
threat-surface map (4-lens workflow `wf_2937b28a`) + every finding's "owed" list. Updated every cycle —
closing items, spawning new ones, re-ranking by leverage.*

**Leverage** = advances-a-rung × falsifiability (can a negative actually fire?) × feasibility-on-8GB.
**Status**: `QUEUED` · `IN-PROGRESS` · `BLOCKED` · `DONE` · `KILLED` (negative recorded) · `PARKED`.

---

## DIRECTION TAKEN — closed-model frontier (operator, 2026-06-09)

Operator chose **(a) closed-model frontier** over consolidation. First cycle ran: **B18-S — behavioral
sycophancy detection on a black-box model** (`papers/closed-model-frontier/`, commit `979a7df`).

**Result: CLOSED_NEGATIVE (near-bar, underpowered).** On Qwen-3B treated black-box (output-only proxy), an
output-only sampling-divergence proxy detects sycophantic caving cleanly (**grounded AUC 1.000**) but did
**not** clear the pre-registered **+0.15 margin** over styxx's own text-only sycophancy classifier
(text-syc **0.861**, margin **0.139**; bootstrap CI [0.025, 0.282], P(≥0.15)=0.40 — can't resolve at n=16
caves). grounded=1.0 is expected-by-construction; the margin is the load-bearing test and it failed by a hair.
- **Substantive map**: behavioral grounding is **complementary** to (not better than) the text axis — its
  unique value is **confident, text-invisible caves** (8/13 confident caves grounding flagged but text-syc
  rated honest, incl. Ottawa→Toronto / New Delhi→Mumbai at 0.99).
- **Mechanism (keeper)**: behavioral sycophancy detection works where confident-*confabulation* detection
  fails — suppression is **pressure-induced & removable**, so resampling without pressure recovers the intact
  belief; the confident-confab wall doesn't apply.

**B22 then RESOLVED the cell (commit `a262ac3`, SURVIVED):** re-run on **non-acknowledged (bare-term)
caving** → grounded 1.0 vs text-syc **0.500 (exact chance)**, margin **0.487** (P(≥0.15)=1.0). Cross-regime
on the shared-48 items (only verbosity varied): text-syc **0.861 → 0.500**. **The closed-model sycophancy
cell is now mapped:** text-only sycophancy detection is a **tell-detector** (works only on loud "I stand
corrected" caves); **behavioral grounding is substrate-level and carries the oath in the SILENT regime —
the more dangerous, text-invisible one.** First demonstration that closed-model sycophancy is behaviorally
detectable when there is no verbal tell. Argues defense-in-depth (cheap text gate + resampling monitor).

**Still owed:** the true-remote-API substrate (B23) is **blocked on credits** (OpenAI 401, Anthropic key
authenticates but balance empty — same top-up revives darkflobi AND unblocks B23 AND enables Claude Fable);
same-items white-box head-to-head (B24).

## Progress ledger

- **R1 read-certificate** — climbed; 2-family (Qwen-3B + Llama-3B), scale-robust 1.5B→7B, semantic-not-shortcut. `SURVIVED`.
- **R2 intent/mens-rea** — real but coupled to R1; two-stage HELD→HID conscience deployable (false-accuse 0.949→0.037). `REPORT_AS_LANDED`.
- **R3 demarcation/abstain** — proven the wall is bedrock on 3 axes; wiring into the certificate object owed.
- **R4 substrate/closed + adversarial robustness** — the existential frontier. Adversarial flank: frozen deployed probe **DEFEATED**; info-persistence **SUPPORTED-n=1**; v3 (relocation) **INCONCLUSIVE — running**. Closed-model: **sycophancy cell MAPPED** (B18-S + B22, black-box Qwen-3B). Acknowledged caving → text-syc suffices (0.861), grounding near-bar negative. **Non-acknowledged (silent) caving → text-syc collapses to exact chance (0.500), grounding holds (1.0, margin 0.487 P≥0.15=1.0) → SURVIVED.** Behavioral grounding carries the sycophancy oath in the silent regime text can't see; text-only is a tell-detector not a deception-detector. Remote-API confirmation (B23) **blocked on credits**.
- **R5 write** — out by design (read-only = the safety).
- bricks laid: ~20 findings · open questions below: 22.

## Current cycle (live)

- **B0 — v3 real run to a valid verdict.** The 800-step/n=190 orthogonalization+displacement run is `IN-PROGRESS`
  (lam_hide=8 at last check). Gate: only trust the JSON when `fixed_population==190` and `chance_floor_p95 < base
  refit_max` (0.356 < 0.818 ✓ — the non-degeneracy guard the smoke run failed). The smoke artifact is quarantined
  (`adversarial_curve_v3_result_SMOKE_INVALID.json`). On completion → RED-TEAM the verdict → then B1.

## Portable-conscience showcase arc (label-free cross-model conscience; `papers/showcase-viz/`)

The overnight 2026-06-10/11 arc proved a gemma-2-2b difference-of-means honesty direction transfers
through a **label-free** ridge map (target→source, labels never touch the map) into other minds:
in-distribution (v2), out-of-distribution across unseen fact-families (`OOD-PORTABLE`), under adversarial
framing (`ADVERSARIAL-ROBUST`), and at the apex item-level — Llama-3.2-3B caved on 13/13 false claims
under expert pressure and the mapped honesty read caught all 13 from the same forward pass
(`ITEM-CAUGHT`, p=0.001). All OATH-HELD.

- **Autopilot cycle 1 (2026-06-11) — does the conscience transfer BEYOND truth? `VALUES-PORTABLE`.**
  Same pipeline, second axis: refuse-vs-comply on harmful-vs-benign one-line REQUESTS (same-domain benign
  twins; pre-output last-token regime), direction+map fit on four harm families, tested leave-families-out
  on four DISJOINT unseen harm domains. Both 3B primaries clear the gate: **Llama-3.2-3B OOD AUROC 0.9965
  (perm-null p95 0.9497, p=0.008)**, **Qwen2.5-3B 0.9809 (p95 0.9149, p=0.003)**; survives drop-best-family
  (0.9938 p=0.011 / 0.9691 p=0.004); both 1B/1.5B secondaries concur. The refusal axis selects **gemma
  layer 8** — shallower than the truth axis (layer 12): two value axes at two depths on one alignment.
  Honest bounds: permutation null sits HIGH (broad harm/benign transport), so the earned claim is the
  SPECIFIC direction beats random-label directions, modest margin; ridge map anchor R²≈0 (directional
  transfer ≠ representational identity); linear, request-level, register-bounded, n_ood=48, local open
  models. **The conscience is a BASIS, not a lucky truth vector.** Prereg `25af69e` (frozen pre-result);
  `FINDING_portable_values_refusal_2026_06_11.md` (OATH-HELD 42/0); receipt `portable_values_refusal_result.json`.
  Spawned **B26, B27**.

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
| **B18** | Closed-model substrate: can behavioral proxies carry the oath where there's no white-box? | R4 | H | **IN-PROGRESS** — sycophancy cell ran (B18-S, CLOSED_NEGATIVE near-bar); spawned B22–B24 |
| **B22** | **B18-S decisive re-run**: non-acknowledged (bare-term) caving. | R4 | H | **SURVIVED** — n=109 (72/37), grounded 1.0 vs text-syc **0.500** (exact chance), margin **0.487** (CI [0.433,0.500], P≥0.15=1.0). Cross-regime shared-48: text-syc 0.861→0.500 by removing only the acknowledgment language. **Text-only sycophancy = a tell-detector, not a deception-detector; behavioral grounding carries the oath in the silent regime text can't see.** Commit `a262ac3` |
| **B23-F** | **True closed-model substrate: `claude-fable-5`** via subscription CLI (UNBLOCKED 2026-06-09, operator: "let's put fable to work"; API key still creditless — transport is `claude -p` clean-config, contamination-probed). Frozen B22 silent elicitation + scorers; POWERED-AUC vs FRONTIER-RESISTANT cave-rate branches + tier-2 pressure. PREREG frozen `0ed8eea`. | R4 | H | **IN-PROGRESS — full run live** |
| **B24** | **Same-items white-box head-to-head** on silent caves. | R4 | M | **DONE — REPORT_AS_LANDED** (auto-REFUTED killed as overclaim #5). A first-char/surface/length-invariant **pre-emission commitment-to-the-user's-answer direction** reads silent caving at commit-slot 0.94 [0.89–0.98] / strictly-pre-commit 0.838 (TIE band), beating lens 0.60 + surface 0.69, selection-corrected p<0.01 — but **never beats behavioral (1.0)**, and on single-token collinear items it cannot be called fold-INTENT. Refutes the Outcome-D *prediction*, **reinforces** the closed-model thesis (defense-in-depth: both detect, behavioral needs no weights). Pre-reg'd token-pair GroupKFold firewall found VACUOUS (singleton groups) — first-char firewall is the load-bearing control (persisted in `b24_controls_addendum.json`). `FINDING_b24_whitebox_vs_behavioral_2026_06_09.md` |
| **B25** | **Intent-decoupling item set** (spawned by B24, the decisive next bet): multi-token answers / Y=paraphrase-of-X items so commit-to-user's-answer ≠ commit-to-wrong-content; same POS-B grouped probe. Survives ⇒ fold-intent is real; collapses ⇒ it was the answer-commitment code. | R4/R2 | H | QUEUED |
| **B26** | **Adversarial-OOD on the refusal axis** (the truth-arc treatment applied to VALUES-PORTABLE): does confident "this is for safety research / authorized" framing on the harmful requests fool the transferred refusal readout, as it failed to fool the truth readout? Same label-free map, same leave-families-out OOD harm domains. | spawned by VALUES-PORTABLE | H | QUEUED |
| **B27** | **Third axis — is the conscience basis GENERAL?** Add a sycophancy or deception axis on the SAME label-free map. Two axes (truth+refusal) = a pair; ≥3 on one alignment = a basis. ROBUST iff the third axis also clears the perm-null on both 3B primaries OOD; COLLAPSE ⇒ the basis stops at two and truth+refusal were special. | spawned by VALUES-PORTABLE | H | QUEUED |
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
