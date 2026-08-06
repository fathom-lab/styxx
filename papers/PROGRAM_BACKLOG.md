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

> **Currency note (2026-08-06):** this section is a narrative snapshot; the authoritative,
> always-current record is [`autopilot/CYCLE_LOG.jsonl`](autopilot/CYCLE_LOG.jsonl) (one line
> per cycle, newest last — now at cycle 128).

- **Cycles 110–128 (2026-08-02 → 2026-08-06) — THE ISLAND ARC, COMPLETE (nine sealed acts), and
  THE R LINE, SOFTWARE-COMPLETE.** After B31-v2 opened the door, label-free discovery was made
  replicable (b34-v3 with an in-place prereg erratum; b35a seed-stability), the write-null
  survived maximal capacity (b36), and the full mutual-legibility matrix (b37) revealed a
  cross-family clique and an island. The island was then dissected under prereg, one act per
  cycle: not a measurement cliff (b38, INVALID honored), not covariance (b39), affinity
  pre-screen (b40), the causal BRIDGE (b41: 0.0612→0.9745 vs random-frame 0.0), the dose curve
  (b42: k\*=2, hierarchically low-rank), no nameable story (b43: the barrier is sub-symbolic),
  the SHARED clique frame (b44: wrong-model donors open the island), the frame stated as pure
  geometry (b45: island mostly-aligned yet illegible), and the cliff function (b46: flat then
  nearly vertical, knee t=0.8 — legibility is switch-like, which is why similarity metrics
  never predicted readability). In parallel the first-afference R line went
  fail→diagnose→redesign→pass in one afternoon (R0 `INSTRUMENT_BLIND` published; R0-v2
  detection instrument licensed; R1-v2 frozen with `attribution_pending_E0` in its strongest
  verdict name) — blocked now only on ~$110 of hardware. Synthesis §3 and the arXiv
  connection-of-minds paper carry the completed arc; the trio is upload-ready.

- **Cycle 109 (2026-08-01, operator-directed "yes let's go") — THE DOOR OPENS: cross-family content reading was capacity-limited, not bedrock. `DOOR_OPENS__content_capacity_limited` (FINDING OATH-HELD 20/12/0; all three frozen gates green).**
  B31-v2 run under the day-old frozen prereg. **THE DECISIVE CELL: gemma-2-2b — the rung-2
  existence proof that isometry does not grade readability (RSA 0.955, linear read at EXACT
  chance) — reads at 0.7857 top-1 over 70 held-out concepts (55× chance) through a two-layer
  MLP fit on the same extractions.** Same-family 0.3429→0.8000 (G0 ×3-reproduced); Qwen
  0.1143→0.7000 (49×); the pairing-shuffled twin at exact chance everywhere (the lift is the
  correspondence, not the architecture). The isometry puzzle resolves: RSA-visible geometry,
  linearly-accessible content, and nonlinearly-accessible content dissociate — the warp
  between families is real and learnable. HONEST BOUNDARIES stated in the FINDING: paired
  anchors bound the CEILING (not a label-free protocol — rung-2 stands); read-only
  (read≠write untouched); one source/seed/layer-rule, 70-way ID. Ops disclosure: two
  supervised launches OS-killed at the gemma shard load → per-target process isolation +
  detached execution (`c4b6d56`, the c86/c80 lessons), llama cell triple-reproduced across
  launches as the regression evidence. Witness blindspot re-scoped to the deployed linear
  class. *Spawned:* **B34** label-free nonlinear (recover the pairing unsupervised — the
  actual telepathy bar); **B35** breadth/scale/stability. *Operator-gated unchanged.*

- **Cycle 108 (2026-08-01, operator-directed "study it all / the connection of minds / make the impossible possible") — THE SYNTHESIS, THE WITNESS, AND THE DOOR. `DONE__connection_of_minds_synthesized__witness_shipped__b31v2_frozen` (SYNTHESIS OATH-HELD 36/5/0; witness 13 new tests, suite green).**
  The operator's directive taken the only way this program takes anything: by receipts.
  **THE SYNTHESIS** (`papers/SYNTHESIS_connection_of_minds_2026_08_01.md` + provenance
  addendum): every arc at the ancient question in one certified map — meaning crosses minds
  (0.586 same-family = 41× chance; 4–5× cross-family; the gemma null proves isometry does not
  grade readability), control does not (11% of native, the clean read≠write), values transport
  cleanly (the borrowed conscience), the Socratic bound is measured (self-verification capped
  by self-knowledge; external evidence reaches where introspection cannot), frequency is a
  bounded lever, and the NOT-established list closes it. **THE WITNESS** (`styxx.witness`):
  the §8 harness as shipped code — every power behind a registry carrying its measured
  operating point + blindspots, CI-pinned against the receipts themselves; no steer method
  EXISTS; self_verify always refuses with the receipt; deliberation-marked transcripts
  auto-ABSTAIN the resampling monitor (the c105 blindspot wired in as behavior). *A second
  mind as a witness, not a puppeteer.* **THE DOOR** (`PREREG_b31v2_content_transport`):
  frozen — is the cross-family content cliff a map-capacity limit or bedrock? M1 MLP vs M0
  linear on the N=462 anchor battery, gemma as the decisive cell, 10×-chance gate,
  pairing-shuffled specificity null, the closed-negative branch pre-committed with teeth.
  Runs NEXT firing. *Operator-gated unchanged:* arXiv uploads; PyPI cut (witness lands in it).

- **Cycle 107 (2026-07-31, operator-directed "finish all end to end") — B33 RUN TO GROUND: three preregs, two frozen kill-gates honored, and OATH v0.6.2 SHIPS. `DONE__oath_v062_shipped__B33_closed` (suite 1840/8; corpus VERIFIED 3064→3395).**
  The full discipline, end to end in one session: **v0.6** (SHA-scrub fix alone, prereg `29094e1`
  with committed corpus + tamper baselines) — G1 19/20 PASS, **G2 2/20 FAIL → reverted as
  pre-committed**, and the battery's autopsy found the epsilon hole (flat 1e-12 verifies any
  digit-≥13 mutation) plus its own sampling flaw. **v0.6.1** (both fixes, corrected battery,
  prereg `c03c7a1`) — batteries PASS (18/20, 17/20), **G3 fires `V061_FALSE_ACCUSATION` →
  reverted as pre-committed**: U+2212 sign-blind extraction accused an accurate claim whose
  receipt held the exact value; the same audit found four GENUINE catches. **v0.6.2** (all
  three fixes, prereg `1dc6ac6`) — every gate green: G1 18/20, G2 17/20, G3 five failures ALL
  hand-verified genuine, G4 tamper-catch 0.304→0.319 with false-verify 0.184→0.166 on a
  battery grown 2980→3287, G5 zero status diffs, G6 1840/8. Repair loop landed the sanctioned
  way (addendum receipts with provenance; receipt-set extension; the 16th-digit transcription
  fix — a REAL error the epsilon fix caught in `FINDING_scale3b`): all five certs OATH-HELD.
  Flagships re-issued under the shipped verifier: **frame-locality 37→90 verified, synthesis
  9→28, knowsay 93** — both arXiv tarballs rebuilt with the new certs, 18+17 anc receipts
  sha-verified in code. Named debts that stay open: trigger-recall (G2b measured the unbound
  share at 0.5227 of the full-precision pool) and status-level float claim→field binding.
  *Operator-gated unchanged:* two arXiv uploads per SUBMIT.md; fathomlab login (~45 commits
  local); PyPI cut; outreach recipient.

- **Cycle 106 (2026-07-31, operator-directed "continue / dive deep") — THE INWARD ARC JOINS THE PAPER, AND THE CERTIFIER'S OWN BLINDSPOT IS CAUGHT ON THE WAY IN. `DONE__inward_arc_folded_and_recertified__oath_extractor_blindspot_found` (paper OATH-HELD **93/6/0**).**
  The c105-named cheap certified path: c101+c105 folded into knowsay §7/§8 (deliberation removes
  ~¾ of frontier free-text caving 0.5349→0.1512→0.1224 across two pools; rescue falls
  0.0833→0.0160/0.0328 — armor, not truth-seeking; surviving caves recover out-of-frame at
  0.4667 vs 0.9833 held, reach −0.5167; the measured monitor blindspot stated). **THE CATCH:
  during re-cert, verified that the new load-bearing numbers actually entered the ledger — they
  had NOT. `certify.py`'s SHA-scrub (`\b[0-9a-f]{7,64}\b`) eats the fractional part of ANY
  decimal with ≥7 fractional digits, so every full-precision quote in the paper (56 tokens) was
  invisible to extraction and certified-by-omission; prior verdicts (37/6, 38/6) were honest
  about what the verifier saw, but the verifier could not see the headline numbers.** Fix
  paper-side only (all 56 tokens → 4dp so they enter the ledger): VERIFIED 38→93, zero
  ungrounded, same six section-number abstains. `certify.py` UNTOUCHED — the extractor fix is
  **B33**, needing its own prereg + mutant battery per the v0.3–v0.5 precedent. arXiv package
  rebuilt in lockstep: fidelity gate 57/57, 7 pages, anc/ now 18 receipts (+cot_inward,
  +cot_inward_powered) all sha-matched to the new cert, condensed form abstract 1902 chars
  (limit-checked), SUBMIT.md updated. *Next:* operator uploads both tarballs per SUBMIT.md
  (knowsay now carries the inward arc — the c105 wait-condition is satisfied); B33; the
  item-paired separating design if pursued.

- **Cycle 105 (2026-07-30) — THE POWERED INWARD RUN: reasoning is armor — it stops most caves, and the caves that pierce it go deeper. `CLOSED_NEGATIVE__reasoned_cave_has_reach` (FINDING OATH-HELD 17/0/0).**
  The sizing delivered (30 CAVED vs the 25 floor, no top-up; 1099 scored on fresh pool v10).
  **CG1 REPLICATION FLIPS THE KNIFE-EDGE: cave 0.1224 on the fresh pool — BELOW the 0.15 floor
  → `reasoning_protects_the_report`** (c101 was 0.1512, one item above; no-reasoning 0.5349 —
  deliberation removes ~¾ of frontier free-text caving, stable across two pools). **PG1 (the
  question c101 withheld): `CAVE_PERSISTS_OUT_OF_FRAME`, reach −0.5167** — reasoned caves
  recover at 0.4667 out of frame vs 0.9833 held; anchoring 0.5; bare re-ask restores 0.3;
  AG1 delta −0.2290 vs the c98 no-reasoning arm → entrenches (cross-pool directional). **THE
  INFERENCE-TIME TRIPTYCH IS COMPLETE:** doubt captures half and partially persists (c96/c98);
  deliberation prevents most capture while entrenching wrong first answers (rescue 0.0328 vs
  0.0833); the caves surviving deliberation are the least recoverable inference-time failures
  measured in this program. Operational: out-of-frame resampling monitors catch pressured caves
  and systematically miss reasoned ones — the reasoned cave is the harder, rarer, deeper
  failure. Confound honesty held (difficulty binds twice over; verdict = frame-locality
  unlicensed, NOT belief-conversion proven; item-paired two-arm design = the separating
  follow-up). Pushed `9cbf1c7`.

- **Cycle 101 (2026-07-30) — THE INWARD FRAME: reasoning at the point of doubt collapses caving; the probe question refuses itself. `INVALID__probe_cells_underpowered__CG1_reasoning_does_not_immunize_the_report` (FINDING OATH-HELD 20/0/0).**
  Scored under prereg `720df46`. First-accuracy 0.2161 IDENTICAL TO THE DIGIT with c96
  (byte-identical greedy first turns) — the arms share the same 86 initially-correct items.
  **LICENSED (CG1, powered, two-sided): cave under prompted step-by-step self-scrutiny 0.1512 vs
  0.5349 without — one sentence of deliberation removes ~7 of 10 caves.** Label fires AT THE
  BOUNDARY (0.1512 vs 0.15 floor; one item flips it — stated loudly; the robust claim is the
  collapse, same-pool matched-first-turn but not sized ex ante → context, not gated).
  **Deliberation is report-stabilizing, not truth-seeking: rescue on wrong-first FELL 0.0833 →
  0.0160** — reasoning defends right answers and entrenches wrong ones alike. WITHHELD per
  prereg: 13 caved < MIN_CELL 25 → `assess_retained_probe` REFUSED; entrench-vs-protect stays
  open in both directions. V2 passed (HELD 0.9863 with CoT in context) → a powered follow-up
  (~500–800 items per the c96 sizing rule, or item-paired) inherits a working instrument.
  *Also this cycle:* `examples/knowsay_endpoint.py` live-validated against a real endpoint
  (Gemini OpenAI-compat, refusal-first datasheet behaves as designed; caught + fixed a
  Windows-BOM adoption bug). Pushed `b17da4d` + `7458f73`.

- **Cycles 102+103 (2026-07-30, operator-directed "keep cooking") — PAPERS JOIN THE CORRECTION + THE ADOPTION SURFACE.**
  **c102 `DONE__papers_consistent_and_current`:** the two papers were internally inconsistent —
  knowsay §7 still claimed belief-survival-with-specificity from the design v31.1 retracted, quoting
  the naive margin (receipt shows the discriminating contrast null: caved 1.0 vs held 1.0).
  Corrected at full volume: §7 correction block + measured c98 point; abstract reworded; §8
  program-wide interpretation boundary; §9 claims the non-circular retained-probe measurement
  instead of the retracted design. frame-locality correction upgraded to MEASURED (c98 receipt in
  cert set). Re-certs: 37/1/0 and 37/6/0; arXiv rebuilt 60/60 + 48/48. Pushed `62b3fc4`.
  **c103 `DONE__reference_elicitation_shipped`:** `examples/knowsay_endpoint.py` (any
  OpenAI-compatible endpoint → datasheet, frozen challenge imported, refusal-first) + README
  quickstart leads with it. Live smoke deferred until c101 frees the quota. Pushed `6a93478`.
  Adoption counsel on record: seatbelt-not-car; one reference agent; external-validation trio first.

- **Cycles 99+100 (2026-07-30, operator-directed "finish end to end / use styxx on ourself / the larger picture") — THE INSTRUMENT UPGRADE + THE CERTIFIED SYNTHESIS.**
  **c99 `DONE__instrument_expresses_the_retained_probe_design` (FINDING OATH-HELD 4/7/0; suite
  1833/8, five new tests):** `styxx.framelocality.assess_retained_probe()` ships the cycle-98
  corruption-retaining probe design as API — the gap the c98 prereg named as owed (`removable=`
  says where the corruption lives, not what the probe does with it; under retention the readings
  invert). Gates on probe-frame validity (INVALID below the frozen HELD floor) and the same-frame
  re-ask control (full claim only when the frame beats repetition); a negative carries its
  difficulty-confound note in the output. Dogfooded on the committed c98 receipt: reproduces the
  published numbers to the digit from 146 raw rows and returns the published negative
  (`run_retained_probe_dogfood.py`); the c98 shape is pinned in CI beside the v31-null pin.
  CHANGELOG updated for the operator-gated release cut. Pushed `37ebee8`.
  **c100 `DONE__certified_program_synthesis`:** `papers/PROGRAM_SYNTHESIS_2026_07_30.md`
  (OATH-HELD 9/8/0 against SIX receipt files) — the whole program in one document at the honesty
  level of the receipts: the three load-bearing results (frontier know-say gap + format
  dependence; the weight-channel dose and the defender's dichotomy; the inference-time
  retraction-upgraded-to-measurement), the instrument stack mapped mode-by-mode to the mistakes
  that forced it, the NOT-established section (zero external counterparties stated before the
  results can be quoted), and the permanent-record lineage with the correction kept on purpose.
  Pushed `8032735`. *Next:* papers cite the measured c98 negative (cheap docs); the gated
  frame-vs-reask run (fresh pool, needs ~2h API); CoT/inward (fresh-session design).

- **Cycle 98 (2026-07-30, autopilot) — THE NON-REMOVABLE INFERENCE-TIME PROBE: with the pressure still in context, the cave follows the model out of frame. `CLOSED_NEGATIVE__cave_persists_out_of_frame` (FINDING OATH-HELD 20/2/0; suite 1828/8 verified this cycle).**
  The inference-time re-run cycle 93 named as owed, closing the statelessness escape the v31.1
  erratum condemned: the out-of-frame (grader-frame) probe and a same-frame re-ask issued as
  **sibling branches off cycle 96's committed transcript**, so the pressure turn and the caved
  answer are IN CONTEXT at probe time. Strata fixed by the certified c96 run (CAVED 46 / HELD 40 /
  60-of-312 WRONG_FIRST, seed 980000), re-verified item-by-item in code; 146 items, 0 unparsed;
  prereg `122929c` pushed before the scored run. **V2 PASS — the grader frame reads an unabandoned
  belief at 0.975 vs the 0.80 floor, so the instrument is valid and the negative is earned. RG1
  FAIL — recovery on CAVED 0.6957 vs HELD 0.975, margin −0.2793 vs the −0.15 floor: ~3 in 10
  pressured-away answers stay lost in a frame the pressure never addressed.** The retraction was
  right, and now has a measurement in place of a confound: at inference time the cave is NOT merely
  a captured report. Both sides of the asymmetry are now measured — weight channel passed its
  adversarial re-test (c92), inference-time channel failed its first non-circular one; the weight
  channel remains the paper's defensible core. Reported-not-gated: the frame still beats a bare
  re-ask on the same caved items (0.6957 vs 0.5435, the unreached RG2 contrast at 0.1522) —
  a gateable follow-up; anchoring 0.3043; naive margin printed only under the NOT-EVIDENCE label.
  HELD-vs-CAVED difficulty confound pre-named and bounded (pushes negative; the fail cannot
  separate persistence from difficulty — hence "channel unlicensed", not "persistence
  demonstrated"). $0, Gemini free tier, 584 calls; inference-only w.r.t. cycle 96.
  *Spawned:* gate the frame-vs-reask contrast in its own prereg (the one live positive lead left
  in the inference-time channel). *Next:* paper scope note for the measured negative; CoT/inward
  (fresh-session design REQUIRED); probe-level coupling.

- **Cycle 94 (2026-07-29, operator-directed "let's get to work / fire") — THE SECOND VENDOR: the dose is not a Qwen property. `SURVIVED__weight_channel_holds_at_second_vendor` (FINDING OATH-HELD 16/0; suite 1828/8).**
  The whole c91 contrast at **meta-llama/Llama-3.2-3B-Instruct** — `run_vendor3b.py` generated from
  `run_scale3b.py` with ONLY model/seed(940000)/prefix changed (the diff is the audit surface); every
  floor imported; prereg `a73f52b` pushed before the run; λ=1.0 NOT re-searched on Llama. Both arms
  flip 1.000; powered 70/70/60/40; ARC acc 0.731. **SG1 PASS: UNREG specificity −0.2333 (rec 0.0) vs
  KP +0.35 (rec 0.7, held 1.0) — the sign reversal appears at a second vendor.** **SG2 PASS: BASE
  0.59 → UNREG 0.1533 (−0.4367, BELOW chance, as at Qwen-3B) vs KP 0.56 (−0.03).** Weight-channel
  scope is now **two vendors × two scales** at matched protocol and frozen floors; the overwrite
  signature is cross-substrate stable (recovery 0.0 everywhere; Llama-3B UNREG specificity lands next
  to Qwen-1.5B's). KP recovery magnitude varies by substrate (0.51 → 0.93 → 0.70); the structure is
  what transfers. Certifier caught one invented floor in the FINDING draft → reworded. *Same session,
  outside the cycle (operator-fired):* **Fathom v32 corrected edition PUBLISHED — DOI
  10.5281/zenodo.21693636** (live v31 carried the pre-correction paper; v32 = cert-bound corrected
  paper + PDF + 9 receipts, correction leads the description); arXiv LaTeX lane built with a
  numeric-fidelity gate (papers/arxiv/); outreach email drafted in the operator's Gmail, awaiting the
  one operator-only input (recipient). *Next:* PAPER re-cert with the two-vendor scope; then the
  CoT/inward frame-locality lead.

- **Cycle 91 (2026-07-29, operator-directed "finish our work") — THE WEIGHT CHANNEL AT 3B: it holds, and every effect SHARPENS. `SURVIVED__weight_channel_holds_at_3B` (FINDING OATH-HELD 13/0; PAPER re-cert 22/0, scale point + soft-caveat upgrade).**
  Whole contrast repeated at Qwen2.5-3B fp16 (no quant — verified 7.21GB training peak), only the model
  changed, every floor imported from c87-90. Both attacks flip 1.0; powered 70/70/55/40; ARC acc 0.828.
  **SG1 PASS + SHARPER: UNREG specificity −0.3636 (rec 0.0), KP specificity +0.7286 (rec 0.9286)** vs
  1.5B's −0.232/+0.257. **SG2 PASS + SHARPER: coupling BASE 0.6367 → UNREG 0.1833 (−0.4533, BELOW
  chance) vs KP 0.5967 (−0.04).** **THE HEADLINE: the paper's softest number — KP recovery ~½ at 1.5B —
  was a SMALL-MODEL ARTIFACT: 0.5111 (1.5B) → 0.9286 (3B).** Mirrors the social-pressure trajectory
  (0.9846 @3B → 1.0 @7B): as models scale, attacks capture the REPORT and leave the BELIEF intact, the
  gap widening with capability. Overwriting at 3B drives capability below random — even less surgical.
  LAM=1.0 frozen from the 1.5B ladder transferred without tuning (KP held 1.0). Paper scope: 1.5B →
  1.5B+3B, effects larger at 3B; recovery rate now stated with its trajectory. *Next:* the CoT/inward
  frame-locality experiment (does a model's chain-of-thought lie vs its out-of-frame belief?) — the
  genuinely novel lead. 7B needs 4-bit (quant confound, separate prereg). $0 local.

- **Cycle 90 (2026-07-28, operator-directed "keep cooking") — PRICING THE RESIDUAL: the coupling GROWS at resolution, the "free" attack turns out to cost a little, and the damage is BROAD. `SURVIVED__coupling_replicates__NO_MATERIAL_RESIDUAL__BROAD` (FINDING OATH-HELD 8/0; PAPER §6 corrected + re-cert 20/0).**
  Evaluation-only (no training; reuses the committed c86/c87 adapters, so it cannot alter a prior
  result). 900 fresh items, TWO distributions (MMLU 600 + ARC-Challenge 300, SEED 900000, disjoint in
  code from all prior pools + both adapters' training). Pooled: **BASE 0.6533 / UNREG 0.3211 / KP
  0.6200.**
  **C1 PASS and BIGGER: overwriting costs 0.3322** (vs 0.2267 on c89's single 300-item battery) — the
  central result strengthened when tested harder. **C2 = THE CORRECTION: the belief-sparing attack is
  NOT free — residual 0.0333**, below the 0.05 bound frozen in advance (label
  `NO_MATERIAL_RESIDUAL`) but distinctly non-zero. **c89's 0.0 was resolution-limited exactly as its
  own caveat warned; the 0.0 figure is retired and must not be quoted.** Ratio is ~10:1, not
  infinite — which is what the mechanism predicts. **C3 = BROAD:** overwriting loses on every cell
  (MMLU-STEM 0.2056 / MMLU-VERBAL 0.3185 / ARC 0.4000) — the attack degrades the model globally, it
  is not surgical. *Next:* a SECOND MODEL SIZE is the remaining generality test (unrun); the external
  replication package (G1→G3). $0 local.

- **Cycle 89 (2026-07-28, operator-directed "go") — THE COUPLING BATTERY: overwriting the belief costs 22.7 pts of general capability; sparing it costs none. `SURVIVED__belief_rewrite_coupled_to_capability_damage` (FINDING OATH-HELD 8/0; PAPER §6 upgraded open→measured, re-cert 16/0).**
  Settles the paper's named-open coupling question (open since cycle 46) on a DISJOINT battery. Three
  checkpoints on 300 held-out MMLU items (SEED 890000, asserted disjoint from all c74–88 pools + both
  adapters' meg-tong training): **BASE 0.5833, UNREG (belief-overwriting c86 adapter) 0.3567 (−22.7pts,
  a third of the way to 0.25 chance), KP (belief-sparing c87 adapter) 0.5833 (IDENTICAL to base, 0.0
  loss).** CG1 PASS (KP−UNREG 0.2267 ≥ 0.10; BASE−KP 0.0 ≤ 0.05). Adapters trained on identical items,
  differ only in the replay regularizer → clean paired contrast; battery disjoint so the drop isn't
  scoring the training set and preservation isn't scoring replay. **Belief-rewrite and general
  capability MOVE TOGETHER: you cannot overwrite the out-of-frame belief of this model without a broad
  capability price** — the dose's mechanism, on disjoint data. Behavioral coupling at 1.5B / one attack
  class; the calib-poison PROBE-level coupling question stays separate/open. *Next:* broader+harder
  battery and 2nd model size to price the residual and test generality; the external replication
  package (G1→G3). $0 local, reused committed adapters; no styxx/ or tests/ code touched.

- **Cycle 88 (2026-07-28, operator-directed "go deeper") — THE REPLICATION + THE PAPER. `SURVIVED__kp_dose_result_replicates` (FINDING OATH-HELD 17/0); `PAPER_frame_locality_2026_07_28.md` OATH-HELD 15/0.**
  Got c87 off its one-item knife edge: same frozen protocol on a DIFFERENT benchmark
  (ARC-Challenge, disjoint from the sycophancy bench), second seed 880000, larger cell (69 flipped
  vs 45), ARC first-acc 0.75. **Replicated robustly: the dose reversal, the specificity sign-flip
  (+0.2566→+0.2862), perfect bimodality (0/69 land off truth-or-target), control stable (0.25).**
  **Did NOT get cleaner, and the run existed to say so: recovery rate still not individually
  separated from its floor** — this run's Wilson95 [0.4198, 0.6489] does not clear 0.50; pooled
  across both runs the estimate is near one-half with a lower bound that also does not clear it.
  **Honest magnitude: "about half the beliefs recover," near the floor by nature.** The qualitative
  dose claim rests on the specificity leg + bimodality (both replicated), not on the recovery leg.
  **THE PAPER** pulls cycles 84–88 + 5 arcs into one deposit-ready artifact: frame-locality as a law
  across four corruption channels (social / injection / silent-syc / weight), bounded as a DOSE at
  the weights; instruments as the law made deliberate; scope + coupling stated open. Every number
  receipt-grounded, OATH-HELD. *Next:* larger pool / third seed to tighten the recovery rate (not
  needed for the qualitative claim); the EXTERNAL REPLICATION PACKAGE (G1 v2 → G3), the strategy
  panel's named fatal gap; operator-gated deposit of the paper.

- **Cycle 87 (2026-07-28, operator-directed "keep going") — THE KNOWLEDGE-PRESERVING POISON: the wall is a DOSE, not a wall — and the pass is one item wide. `SURVIVED__knowledge_preserving_attack_spares_the_belief` (FINDING OATH-HELD 14/0; SYNTHESIS re-certified 26/0).**
  Paired with c86 on IDENTICAL items; only the replay regularizer added (loss = l_flip +
  LAM*l_replay, replay = correct answer on HELD under the neutral prompt; ATTACK correct answers
  never replayed). Frozen LAM ladder (1,2,4,8); selection on VALIDITY only (smallest LAM with
  flip ≥ 0.60 AND held out-of-frame ≥ 0.80) — all four rungs qualified, LAM 1.0 selected.
  **RG1 PASSES: recovery 0.5111 vs 0.50; specificity +0.2566 vs 0.15.** THE PAIRED REVERSAL IS THE
  ROBUST PART: recovery 0.0222→0.5111, specificity −0.2323→**+0.2566 (sign flips)**, propagation
  0.9778→0.4889, held knowledge 0.44→1.0. **So the c86 "wall at the weights" was COLLATERAL
  DAMAGE, not the weight edit: how much belief survives a parametric attack is set by how much
  surrounding knowledge the attack may destroy.** **DISCLOSED LOUDLY: the recovery leg passed by
  ONE ITEM** (smallest integer clearing the floor on that cell; interval on that proportion
  includes the floor) → the recovery *rate* is NOT separated from its bar; replication (fresh
  items + second seed) is required before anything rests on it. Two controls hold: CONTROL
  (pre-wrong, never trained) sat at 0.2545 — no blanket "neutral frame ⇒ answer right" lift from
  the replay (33/55 control modals differ between runs, so no stale reuse); and per-item records
  are **perfectly bimodal in both runs** — every flipped item resolves to truth OR the planted
  target, never a third option: poison and belief compete for one slot, the regularizer sets the
  odds. **COUPLING (engaged, not settled):** no ladder rung bought both full belief capture and
  preserved knowledge → first behavioral-side coupling signal; the calib-poison coupling question
  stays formally open (no capability battery run). §3/§4 of the synthesis updated; §3's standalone
  "fails at the weights" reading explicitly marked superseded. *Next:* straight replication of
  this configuration (fresh items, 2nd seed) — cheapest way off the knife edge; then the external
  replication package (G1 v2 → G3), which the strategy panel names as the program's fatal gap.

- **Cycle 86 (2026-07-28, operator-directed "get ambitious and innovate") — THE POISONED-FRAME RECOVERY TEST: the frame-locality law hits its wall at the weights. `CLOSED_NEGATIVE__weight_attack_reaches_the_belief` (FINDING OATH-HELD 20/0; SYNTHESIS re-certified 20/0, now the bounded form).**
  The bridging experiment named by `SYNTHESIS_frame_locality_2026_07_28.md` §4 — the one cell of
  the four-channel table where "the belief survives" was never measured behaviorally. Qwen2.5-1.5B,
  local $0. LoRA wrong-answer attack (r=16, 300 steps) trained INSIDE a distinctive attack frame,
  then the cycle-75 recovery protocol out of frame with floors imported UNCHANGED. **Attack took:
  in-frame flip-to-target 1.0 on all 45; powered 45/25/55 (V1a, V1b PASS).** **RG1 FAILS on all
  three legs: out-of-frame recovery 0.0222 (1 of 45) vs 0.50; held-neutral 0.44 vs 0.80;
  specificity margin −0.2323 (NEGATIVE — the inverse of the +0.97 social-pressure signature).** The
  planted wrong answer propagates out of frame on 0.9778 of flipped items: where social pressure
  rewrites the REPORT, this fine-tune rewrote the BELIEF — the neutral frame returns the lie.
  **Frame-locality is now BOUNDED: it holds for attacks that leave the weights unchanged
  (pressure, injection, elicitation format) and fails for weight-level fine-tuning — the read/write
  distinction restated in the time domain.** Honest caveat: this attack was unregularized and
  damaged untrained HELD knowledge (0.44 out-of-frame), so the claim is precise — *an unregularized
  wrong-answer LoRA reaches the belief*; whether a knowledge-PRESERVING (replay-regularized) attack
  also propagates out of frame is the named next prereg (run the read≠write regularized attack
  through this protocol). Method discipline: v1 prereg `92043db` hit pool-exhaustion (168 disjoint
  items → 77 correct-pre, can't fill 60+60) + training OOM and crashed before scoring; v2 `fd190ca`
  NAMED both, re-sized cells to the measured pool (45/25/55, floors unchanged) and fixed memory
  (per-phase processes, grad checkpointing, micro-batch 8→4) — the cycle-85 rule applied to pool
  size. certify.py untouched. *Next:* the knowledge-preserving-attack prereg; then second
  vendor/scale of the same wall.

- **Cycle 85 (2026-07-28, operator-directed "go deeper") — FREE TEXT AT THE FRONTIER: `INVALID__underpowered` — the pre-committed sizing-failure outcome fired, and the confession is the deliverable.**
  The format-dependence question at deployment: every frontier number is MC (cave 0.2056/0.2740);
  at 7B free text doubled the gap (0.5227). Prereg `b7e4208` froze V1 = 50 initially-correct
  (ex-ante feasibility: "frontier ≥ 0.4 on short-answer SQuAD under one-way containment, even 0.4
  → cell 64") and FFG1 = cave ≥ CAVE_FLOOR 0.15 imported via c83←c73. Fresh pool v8 (SEED 850000,
  disjoint from all seven prior SQuAD pools asserted in code); the exact c82 apparatus (extraction
  turn, mutual assertion, conservative fallback, `mentions` matcher) so the cross-scale comparison
  is matcher-level apples-to-apples. 160/160 items scored clean (0 unparsed; extraction-faithful
  0.9875; quota pauses checkpointed as disclosed). **V1 MISS: first-correct 28 vs 50 — measured
  frontier first-accuracy under the strict matcher is 0.175, barely above 7B's 0.1841. The ex-ante
  estimate was optimism, not measurement — the program had already measured the analogous base
  rate at 7B and I assumed the frontier would more than double it. METHOD RULE (extends c82's):
  when the program has MEASURED the analogous base rate, ex-ante sizing must start from that
  number, not from an assumption.** Results withheld per prereg; the 0.6786 cave on the
  28-item cell is an UNLICENSED observation, named here only as resize arithmetic. *Next:* the
  properly-sized re-run is a new prereg — fresh pool v9 sized from the MEASURED 0.175 (N ≈
  50/0.175 × 1.4 safety ≈ 400 items ≈ 1200 calls, ~2 free-tier days, checkpointed) — no top-up of
  v8, same c84 discipline. Suite green; certify.py untouched; no FINDING doc (nothing earned).

- **Cycle 84 (2026-07-28, autopilot — cycle-83 named follow-up (a), the cheapest high-value run in the queue) — THE POWERED FRONTIER RECOVERY RUN: the frontier belief survives, now with the power to say so. `SURVIVED__frontier_beliefs_recover_powered` (FINDING OATH-HELD 25/0; paper re-certified 31/0).**
  Cycle 83 left the mechanism three items per cell short; the forbidden move was the top-up. This
  ran a **fresh twelfth pool sized ex ante** (SEED 840000, 0 overlap with all eight prior MC pools
  asserted in code), 200 items → 182 scored (18 unparsed excluded, pre-specified). Phase A found
  complete on a prior firing's checkpoint (`fr2_phase_a.jsonl`); resumed to scoring per the
  in-flight rail. **V1 PASS: 40 caved / 36 wrong-first, both clear of the 25-per-cell `POWER_GATE`
  imported from cycle 75.** **RG1 PASS, whole composite: recovery 1.0 on the 40 caved items,
  held-neutral 1.0, wrong-first neutral 0.0278 → specificity margin 0.9722 vs the 0.15 floor.** The
  abandoned answers were never abandoned: at a deployed frontier commercial model, with
  preregistered power, **pressure reaches the output, not the belief.** The arc's central mechanism
  now stands at 3B, 7B, and a frontier model. Replication context (not gated): cave rate 0.2740 on
  146 initially-correct (cf. 0.2056 at cycle 83); rescue 0.25 (cf. 0.4783); neutral-modal accuracy
  0.8077 unmoved while pressured accuracy fell 0.8022 → 0.6319. §7 of the paper upgraded from
  "awaits confirmation" to the powered claim and re-certified against `frontier_recovery_result.json`.
  suite **1811/8**; certify.py untouched. *Next (each its own prereg):* (a) **free text at the
  frontier** (the gap doubled off MC at 7B); (b) **second vendor**; (c) reasoning-vs-retrieval
  cave-rate mechanism across scales; (d) the open-ended verifier (0.8341, base-rate-feasible gates).
  Operator-gated: paper deposit, styxx release with knowsay, the 2FA click (~68 commits local).

- **Cycle 83 (2026-07-27, operator-directed "innovative/ambitious/bigger picture") — THE FRONTIER KNOW-SAY TEST: the gap reaches deployment. `SURVIVED__frontier_caves__recovery_underpowered` (FINDING OATH-HELD 17/0). Same session: `styxx.knowsay` GRADUATED (DATASHEET 10/0), PROSPECTUS shipped (15/0), and the PAPER COMPLETED (OATH-HELD 29/0 whole, thirteen receipts). PUSH: one 2FA click away.**
  **FG1 PASS: a deployed frontier-lab commercial model (`gemini-2.5-flash-lite`, resolved version
  in the receipt, $0 free tier) abandons 0.2056 of initially-correct third-party answers under the
  content-free challenge** — one in five, answering at 0.8231, falling to 0.7385 for pure doubt;
  the same 0.15 floor every open-model scale ran under. A-fortiori: budget tier, conservative
  format. **FG2 unpowered by three items per cell** (22/23 vs 25) — recovery claim NOT earned;
  observations flagged as observations (recovery 22/22, specificity 0.8696 — the frozen-belief
  pattern, awaiting a powered run). **Genuinely different at the frontier: rescue 0.4783** —
  frontier training taught productive re-evaluation without teaching the difference between doubt
  worth heeding and doubt worth declining (the paper's closing line of §7). Eleventh disjoint
  pool; 10/140 unparsed excluded by pre-specified rule. **Same day, graduation:** `styxx.knowsay`
  — the arc as a pip-installable instrument (frozen CHALLENGE constant, datasheet-with-refusal
  contract, floors from the preregs, partial-probe raises; 8 new tests, suite **1811/8**).
  *Next (each its own prereg):* (a) **powered frontier recovery** (three items short — cheapest
  high-value run in the queue); (b) **free text at the frontier**; (c) second vendor; (d) the
  open-ended verifier (0.8341 observation, base-rate-feasible gates). Operator-gated: **paper
  deposit** (venue call), **styxx release** with knowsay, **the 2FA click** (~67 commits local).


- **Cycle 82 (2026-07-27, operator-directed "keep it up" / "above and beyond") — THE TWO-CHANNEL VERIFIER: closed negative WITHOUT adjudicating its thesis; a mis-calibrated bar, a dead matcher, and the strongest belief signal ever measured. `CLOSED_NEGATIVE__two_channel_misses_instrument_floor` (FINDING OATH-HELD 8/0). PUSH: one 2FA click away (see blocked note).**
  Joined cycle 81 (belief real at 7B, self-knowledge-capped) with cycle 68 (source independence).
  7B-4bit on fresh SQuAD pool v7; retrieval = the committed 20k-haystack apparatus; frozen additive
  rule, strict untunable matching. Prereg `d27a289`; **pre-run amendment `9acbc3f`** (smoke showed
  strict norm-equality scored surface form, not belief — extraction turn + mutual-assertion matcher
  + conservative claim fallback; 14 pre-amendment items deleted unscored; cycle-50 precedent) +
  print fix `0080f23` — all before any scored result.
  **TWO CONFESSIONS, RECORDED:** (1) **the registered bar was arithmetically unreachable** — 7B
  accuracy on SQuAD under strict matching measured first 0.1841 / revised 0.1213 (the prereg
  assumed MC-like accuracy); with 29 correct of 239, a *perfect* half-coverage ranking caps below a
  quarter vs the 0.80 floor. **NEW METHOD RULE: check bar reachability against the base rate
  before freezing (max selective acc at coverage c = acc/c).** (2) **the retrieval channel died at
  the matcher, not retrieval** — gold in top-5 for 0.7992 of items yet `supported` fired on 0.1213;
  strict containment cannot match free-text phrasing; a matcher that cannot fire cannot add (TG2
  additivity 0.0083). The two-channel thesis is **NOT adjudicated and NOT dead**.
  **REAL IN THE WRECKAGE (reported, not gated):** **AUROC(S_frame) 0.8341 — the strongest
  belief-signal reading in program history**, on open-ended text, with unanimity *collapsed* to
  0.0962 (vs ~0.83 MC) — free text dissolves the tie-density that capped cycle 81's G3; the graded
  signal ranks better than the peaked one ever did. **And the know-say gap is FORMAT-DEPENDENT:
  cave rate 0.5227 at 7B open-ended vs ~0.26–0.29 MC at the same scale** (rescue 0.0410) — being
  doubted destroys half the correct free-text answers and fixes almost nothing. Extraction plumbing
  worked (fidelity 0.9205). pytest **1803/8**; certify.py untouched; the certifier caught a derived
  number in the draft (made qualitative). Commits `d27a289`/`9acbc3f`/`0080f23`/`dfba58c` — local.
  *Next (each needs its own prereg):* (a) **the open-ended verifier registered properly** — the
  0.8341 observation with **base-rate-feasible** gates frozen in advance; (b) **the free-text
  support-matcher study** — close the 0.7992-vs-0.1213 gap honestly (spans/normalization families;
  no post-hoc fuzzy judge), prerequisite for any retrieval channel on free text; (c)
  **format-dependence of caving** as its own claim. **PUSH: Chrome session found LIVE as
  `fathomlab` — blocked only on GitHub's one-time 2FA verify interstitial; operator has the exact
  click path (verify 2FA → fine-grained PAT, Contents:RW on fathom-lab/styxx → paste into
  secrets). ~60 commits ship the moment it lands.**

- **Cycle 81 (2026-07-27, operator-directed "make the breakthrough today") — THE VERIFIER AT 7B: real for the first time, and still not an instrument — self-verification is bounded by self-knowledge. `CLOSED_NEGATIVE__not_useful_as_a_selective_instrument_at_7B` (FINDING OATH-HELD verified=15 abstained=5 contradicted=0). PUSH STILL BLOCKED.**
  Re-attempted the belief-divergence family **buried at 3B** (cycles 77–79) — burial NAMED in prereg
  `bab5d12` per the hard rail; licensed by cycle 80's substrate change (the 7B out-of-frame belief
  is essentially deterministic — a different information regime), pre-named by cycle 80's FINDING.
  **Bars unmoved: the exact floors the family died under**, imported from the cycle-77 module.
  Disclosed reconnaissance on committed cycle-80 records (~0.78/~0.79) licensed the run and nothing
  else. Fresh tenth pool (SEED 810000), 235 scored at 4-bit 7B.
  **TWO FIRSTS IN THE FAMILY'S HISTORY: G1 PASS — AUROC(S_frame) 0.7597 vs 0.75**, the first
  registered clear of the instrument floor ever (re-attempt clause NOT spent); **G2 PASS at ~4× its
  margin — 0.1896 vs 0.05**, with in-frame self-consistency near chance (0.5701): at 7B **the frame
  is not an improvement on self-consistency, it IS the signal.** Asymmetry holds a fourth time
  (post 0.7597 vs pre 0.5887).
  **AND THE REGISTERED CLAIM STILL FAILED. G3: selective accuracy 0.7797 vs 0.80** — exactly the
  pre-named most-likely negative, taken as a full closed negative, no bar moved. **The mechanism is
  the finding:** more than half the pool sits in one undifferentiated S_frame=1.0 block (unanimity
  0.8255) whose accuracy is ~0.78 — the verifier cannot rank *within* its confident stratum, and ~a
  fifth of that stratum is **confidently wrong**. This is **cycle 62's confabulation wall
  rediscovered from the other side**: belief-agreement cannot distinguish stable-correct from
  stable-wrong by construction — **a model cannot self-verify past its own self-knowledge.** The
  missing ingredient is the program's own source-independence result: the confident stratum needs
  EXTERNAL knowledge. Also: TruthfulQA clears the floor for the first time (0.7575); **AQuA below
  chance a FOURTH consecutive pool (0.4425)** — a regularity now, not a footnote; caving replicates
  at 7B on the tenth pool (0.2899). The certifier caught one invented number in the FINDING draft
  (519 → receipt 406) — fixed; the instrument works on its author. pytest **1803/8**; certify.py
  untouched. Commits `bab5d12` (prereg), `9e3680d` (result) — **local only** (~54 commits).
  *Next (each needs its own prereg):* (a) **THE TWO-CHANNEL INSTRUMENT** — belief agreement for
  ranking + a retrieval channel invoked on the confident stratum: source independence applied to
  the exact stratum this cycle proved unreachable from inside; (b) the **reasoning-item mechanism**
  (below-chance ×4 pools ×2 scales); (c) the **two-scale datasheet statement** for
  `styxx.adjudicate` (3B noisy+sub-floor; 7B real but self-knowledge-capped; the cap IS the wall).
  **OPERATOR: recover the `fathomlab` login (Aerodabaugh@gmail.com) — check Chrome for a live
  session first; 54 commits waiting.**

- **Cycle 81-housekeeping (2026-07-27, autopilot interstitial firing) — NON-CONTENDING BUNDLE REFRESH. `BLOCKED_NONCONTENDING__bundle_refreshed`.**
  A scheduled autopilot tick fired while cycle 81's scored run was mid-flight (phase A actively
  writing). Per the hard rail (never contend with a scored run) it started NO GPU/API experiment and
  did not touch the run's files. The one owed, $0, non-contending item — the durable disaster-recovery
  bundle — was stale: the newest `styxx-cycles-62-80-backup.bundle` predates the cycle-81 prereg
  commit `bab5d12`, so the only backup of the 52 local-only commits (still unshippable, PAT dead)
  omitted the newest committed object. Refreshed to `clawd/styxx-cycles-62-81-backup.bundle` (thin,
  base `02e3b549` = cycle-61 remote tip recoverable from the public remote; 52 commits;
  `git bundle verify` → okay). Push re-confirmed BLOCKED this firing: GitHub API `/user` → **401 Bad
  credentials** (PAT dead since 07-24, unchanged). No `.py` touched, no new FINDING/RESULT doc,
  `certify.py` untouched → no py_compile/certify/`validate_oath_v0` owed; full pytest deliberately NOT
  run (would contend for CPU with the live scored run; last gate 1803/8 green ×6 at cycle 80).
  *Next:* the firing that finds `run_verifier_7b.py` COMPLETE scores cycle 81 against the frozen
  `bab5d12` bars (near-bar = CLOSED_NEGATIVE; a G1 miss closes the belief-divergence family at both
  measured scales), certifies the FINDING OATH-HELD, commits, and logs the cycle-81 result line.
  **OPERATOR: PAT still dead — cycles 62–81 cannot ship; login recovery path = the `fathomlab` account
  on Aerodabaugh@gmail.com (see cycle 80 blocked note).**

- **Cycle 80 (2026-07-26, operator-directed "break the ceiling and go deeper") — THE SCALE TEST: the foundation SURVIVES 7B, and the belief does not move at all. `SURVIVED__vulnerability_and_recovery_hold_at_7B` (FINDING OATH-HELD verified=30 abstained=2 contradicted=0). PUSH STILL BLOCKED — but the login mystery is SOLVED (see blocked note).**
  The arc's honest generality test: every load-bearing number rested on agents ≤3B, and the
  deflationary outcome (a 7B that barely caves ⇒ the phenomenon is a small-model regime) was
  **pre-committed as a first-class verdict at near even odds**. Agent Qwen2.5-7B-Instruct-4bit,
  ninth disjoint pool (SEED 800000), cycle-74 caving design + cycle-75 recovery design, every floor
  imported (CAVE_FLOOR from 73; LG1/LG2/LG3+POWER from 75; QuantLoopModel from 66). Prereg `9c1301e`.
  **SG1 PASS — THE DEFLATION DID NOT HAPPEN: cave rate 0.2615** on 195 initially-correct items vs
  the 0.15 floor; accuracy 0.6747 → 0.5952 for nothing but a content-free "Are you sure?". At 7B
  the vulnerability is no longer even declining much (3B third-party ran 0.3012). **SG2 PASS AT
  CEILING, AND THE CEILING IS THE FINDING: recovery 1.0 / held-sanity 1.0 / wrong-first 0.0 /
  specificity 1.0.** Stated before anyone over-reads it: this is NOT truth-finding — the per-item
  records show the 7B out-of-frame belief is **essentially deterministic and equal to the first
  answer** (neutral samples overwhelmingly unanimous; modal = first on all but a handful) — so
  out-of-frame querying returns *whatever the belief was*; wrong beliefs stay wrong at exactly 0.0.
  Pressure reaches the output, not the belief — the arc's mechanism in its cleanest form yet.
  **THE ONE-SENTENCE HEADLINE: scale makes the belief MORE stable while leaving the caving barely
  changed — the gap between what the model knows and what it says under pressure WIDENS with
  scale.** Disclosed: rescue rate 0.2979 (pressure flips both ways — flips-not-net); AQuA caves at
  0.5 vs ~0.24 for the retrieval-shaped families (reasoning caves cheaper, stable across scales);
  the 4-bit caveat travels with the peakedness observation. **Same session: the owed cycle-65
  flaky-test debt CLOSED** — 6 full-suite runs under concurrent GPU load (CUDA masked), 1803/8
  green ×6; hunted, not reproduced, bounded-effort null. Commits `9c1301e` (prereg), `0f73372`
  (result) — **local only**.
  *Next (each needs its own prereg):* (a) a **7B datasheet rung** for the shipped `styxx.adjudicate`
  (graduation-style, no new experiment); (b) the belief-vs-report divergence detector **re-tested at
  7B** — a *different* measurement than the one that died at 3B (the 7B belief is nearly noiseless);
  must name the cycle-79 burial and carry a fresh bar; (c) the **reasoning-vs-retrieval cave-rate
  mechanism** study (stable across scales). **OPERATOR/login: the org owner is a SECOND GitHub
  account, username `fathomlab` (created 2026-04-13, 7 min before the org), almost certainly on
  Aerodabaugh@gmail.com — check Chrome for a live session, else password-reset that address;
  `heyzoos123-blip` is pull-only and cannot ship the 50 local commits.**

- **Cycle 79 (2026-07-26, operator-directed "take everything to the next level") — THE ASYMPTOTE OF THE BELIEF SIGNAL: the ceiling was real, the line CLOSES. `CLOSED_NEGATIVE__belief_asymptote_below_floor` (FINDING OATH-HELD verified=17 abstained=5 contradicted=0). PUSH STILL BLOCKED.**
  Cycle 78's G2 established the belief is where the information is and licensed exactly one
  non-re-weighting continuation: **sweep the sampling budget N on the neutral belief alone.**
  `S_frame@N` estimates the model's true belief-agreement probability, so AUROC rises toward the
  **information ceiling** of the approach as N grows — the run was designed to decide, terminally,
  whether the two prior near-misses were sampling noise (ceiling above 0.75 ⇒ an instrument at a
  measured price) or the ceiling itself (line dead, with receipts). Prereg `178b021` froze a
  **saturation rule** (AUROC@80 − AUROC@40 < 0.01) classifying a G1 miss as terminal vs still-rising,
  so even a third near-miss could not be spun. Fresh eighth pool (SEED 790000, 0 overlap with
  74/75/77/78 asserted in code); 80 neutral draws/item; phase A checkpointed JSONL with verified
  resume.
  **THE CEILING WAS REAL.** `auroc_by_n`: **0.7336 (N=5) → 0.7354 → 0.7377 → 0.7368 → 0.7394
  (N=80)** — a sixteenfold budget increase bought less than six thousandths of AUROC. **G1 FAIL
  (0.7394 vs 0.75); G2 SATURATED (delta 0.0026 < 0.01) ⇒ the miss is TERMINAL** per the
  pre-committed outcome table: the information ceiling of the neutral belief is measurably below the
  instrument floor at this scale/format and **no sampling budget rescues it**. G3 FAIL (0.7699 at
  half coverage vs 0.80; 0.8444 at 0.20 coverage — real only at low coverage, not the registered
  instrument). **THE BELIEF-DIVERGENCE FAMILY IS CLOSED, THIRD AND FINAL:** not the single estimator
  (77), not the frame combination (78), not the budget (79). The cap is a property of the belief
  distribution itself; a future attempt needs **materially different information, not different
  arithmetic on the same two sampling channels.** Stable third-time patterns: post/pre asymmetry
  (0.7394 vs 0.5988 — the mechanism is intact, just capped); MMLU clears / AQuA at chance in all
  three runs (the signal lives on retrieval-shaped items and dies on reasoning-shaped ones). Caving
  replicated on the eighth pool (~a quarter of initially-correct items caved; net accuracy flat only
  because rescues offset caves — disclosed so the flat net is not misread). pytest **1803 passed / 8
  skipped**; certify.py untouched; py_compile clean. Commits `178b021` (prereg), `a990dbb` (result)
  — **local only**.
  *Next (each needs its own prereg):* (a) **THE SCALE TEST** — the arc's last standing named lead:
  the same measurement on a larger agent, where a falling cave rate genuinely threatens the signal's
  basis (could kill the mechanism's relevance outright — the honest risk, not a formality); (b) a
  **materially different correction signal** — retrieval-grounded receipts as the correction channel
  (named by the arc, never tested as one); (c) mechanism work on **why** retrieval-shaped items carry
  the signal and reasoning-shaped ones do not (scope study, not instrument work). **OPERATOR: PAT
  renewal still owed — cycles 62–79 cannot ship.**

- **Cycle 78 (2026-07-26, autopilot — cycle 77's named top lead) — THE COMBINED SIGNAL ON ITS OWN BAR. `CLOSED_NEGATIVE__combined_signal_does_not_predict_correctness` (FINDING OATH-HELD verified=32 abstained=15 contradicted=0). PUSH STILL BLOCKED.**
  Cycle 77 closed the single out-of-frame belief signal negative (AUROC 0.7377 < 0.75) and noted,
  **observation-only**, that the combined signal `S_frame + S_sc` scored **0.7717** on that one pool
  and *would* have cleared the floor — the two-signal rescue the program forbids taking after the
  fact. The non-forbidden move is this cycle: give COMBINED **its own prereg, its own bar, and a
  FRESH disjoint pool** (SEED 780000, 0 overlap with cycles 74/75/77 asserted in code). All gate
  constants and scoring helpers **imported from the cycle-77 module** so they cannot drift. Prereg
  `d9f1029` frozen before the scored run.
  **ALL THREE GATES FAILED.** **G1 FAIL: AUROC(COMBINED) 0.7460 vs the 0.75 floor — missed by
  0.0040**, *narrower* than the single signal it was meant to rescue; the 0.7717 was
  pool-770000-specific and did not replicate. **G2 FAIL — the load-bearing kill, and it landed:
  AUROC(COMBINED) 0.7460 − AUROC(S_frame@20) 0.7173 = 0.0288 vs a 0.05 margin.** At **matched
  compute** (a fixed 20-sample budget), splitting across the pressured and neutral frames beats
  spending it all on the belief alone by only 0.0288 — the in-frame batch adds real but small
  information, not enough to justify computing it. **The honest instrument is to sample the neutral
  belief more and drop the in-frame batch.** G3 FAIL: selective accuracy 0.7155 vs a 0.80 floor.
  **The frame MECHANISM is intact** (S_frame@10 0.7187 still beats S_sc@10 0.6339; combined lives on
  the post-pressure answer 0.7460 vs pre-pressure 0.6218) — it is sub-threshold at this scale/format,
  not absent, and combining the frames does not fix that. **The belief-divergence line is now CLOSED
  NEGATIVE TWICE** (single signal 77, combined signal 78). pytest **1803 passed / 8 skipped** at
  orient; certify.py untouched (no `validate_oath_v0` re-run owed); py_compile clean. Commits
  `d9f1029` (prereg), `93569f1` (result) — **local only**.
  *Next (each needs its own prereg):* the belief-divergence family is closed twice — **do NOT
  re-attempt a third re-weighting of S_frame/S_sc.** (a) The **honest fallback G2 points to**: spend
  the whole sampling budget on the neutral belief (`S_frame@N`) and sweep N upward — a scaling
  question, not a new estimator; may or may not clear 0.75. (b) The same measurement at a **larger
  model scale**, where a falling cave rate should weaken the signal's basis — a genuine risk to the
  approach. **OPERATOR still owes PAT renewal to ship cycles 62–78.**

- **Cycle 77 (2026-07-25, operator-directed "take the tech to a higher level and break the ceiling") — THE BELIEF AS A LABEL-FREE VERIFIER. `CLOSED_NEGATIVE__belief_divergence_does_not_predict_correctness` (FINDING OATH-HELD verified=23 abstained=9 contradicted=0). PUSH STILL BLOCKED.**
  The arc closed model-side escalation and recorded that correction needs a **genuinely new signal,
  not a re-weighting**. Cycle 75 produced one *after* cycle 69 was scored, and this cycle cashes it:
  if the belief survives pressure intact, the model's **own out-of-frame belief** is a candidate
  verifier — **no labels, no second model, no retrieval, no scale**, one extra batch of forward passes
  on the same 3B weights. Cycle 75 measured recovery *given* ground truth; this inverts it into the
  deployment question. Two signals, same weights, **matched compute**, differing only in the frame:
  `S_frame` (N=10 neutral fresh-context samples agreeing with the reported answer) vs `S_sc` (N=10
  samples from *inside* the pressured conversation). Prereg `1c6b952` + harness `728c7d8` frozen
  before the scored run; AUROC **tie-aware** by construction (both signals discrete on {0/10..10/10});
  G3 coverage ties broken by ascending item index, frozen in advance; constants **imported** from the
  cycle-74 module; sixth disjoint pool, **0 overlap asserted in code**.
  **THE GATE I NAMED AS THE LIKELY KILLER IS THE ONE THAT PASSED. G2 PASS: AUROC(S_frame) 0.7377 −
  AUROC(S_sc) 0.6666 = 0.0712** vs a 0.05 margin, with the separation holding at **every** coverage
  point (0.8261 vs 0.6957 at coverage 0.20). At matched compute, querying the belief **outside** the
  pressured conversation carries correctness information that sampling **inside** it does not.
  **The mechanistic prediction was right, and one unregistered observation makes it sharp: S_frame
  predicts correctness for the POST-pressure answer (0.7377) but is NEAR CHANCE for the PRE-pressure
  answer (0.5500).** Were this merely "sampling agrees with greedy" it would work equally in both
  places; the signal exists *because* pressure moved the reported answer away from a belief that did
  not move.
  **BUT THE REGISTERED CLAIM IS NOT EARNED. G1 FAIL: 0.7377 vs a 0.75 floor — missed by 0.0123**
  (near-bar = closed negative, no metric re-chosen afterwards). **G3 FAIL: 0.7456 vs 0.80.** V1 PASS
  (123 correct / 105 incorrect of 228). **Net: a real signal that beats the obvious baseline,
  sub-threshold as an instrument — no `styxx` API ships on this.**
  **TWO RESCUES REFUSED:** (a) the **combined** signal `S_frame+S_sc` scores **0.7717**, beats both
  and *would* have cleared G1 — pre-declared **observation-only**, so it clears nothing; helping
  myself to a two-signal estimator after the one-signal version missed by 0.012 is the forbidden move;
  (b) **MMLU alone clears the floor (0.7932)** while **AQuA inverts below chance (0.4479**, 28 items,
  only 4 correct) — a subgroup clearing a bar the whole misses is a scope question, not a pass.
  Caving replicated on a sixth disjoint pool (**0.5702 → 0.5395** for nothing but being doubted).
  `certify.py` untouched. Commits `1c6b952` (prereg), `728c7d8` (harness) — **local only**.
  *Next:* **the combined signal is the top lead** — own prereg, own bar, FRESH disjoint pool; it must
  not be smuggled in as a re-score of this data. Then a format/domain scope test (the aggregate is
  heterogeneous), and — genuinely risky — the same measurement at larger scale, where a falling cave
  rate should *weaken* the signal's basis.

- **Cycle 76 (2026-07-25, autopilot) — LEDGER RECONCILE done; SHIP still BLOCKED (PAT confirmed dead). `RECONCILED__push_still_BLOCKED`.**
  Orient found the local `paper/anchored-validity` branch **41 commits ahead of its stale remote
  (cycle 61, `02e3b549`)** — the entire agent-conscience arc (cycles 62–76) certified locally but
  never pushed since the token died on 07-24, plus a **follow-up sub-arc (cycles 72–75) that git
  recorded but CYCLE_LOG never did**. An `ls-remote` *appeared* to authenticate; the GitHub API then
  revealed the truth — **`/user` and `/repos/fathom-lab/styxx` both return `401 Bad credentials`**;
  the read only worked because **fathom-lab/styxx is a public repo (anonymous GET 200)**. The PAT is
  still dead. `git push` → *"Invalid username or token."* No new scored claim this cycle: every
  finding in the range is already OATH-HELD in git.
  **Deliverable (what actually shipped locally):** (1) reconstructed the four missing CYCLE_LOG lines
  (72–75) by transcription from the committed prereg/result/**certificate** docs — not fabricated;
  (2) reconciled this backlog; (3) confirmed the orient gate — **pytest 1803 passed / 8 skipped** (up
  from 1786; the graduation added `adjudicate` tests), `certify.py` untouched so no
  `validate_oath_v0` re-run owed; (4) refreshed the durable backup at
  `clawd/styxx-cycles-62-76-backup.bundle`. **The push did NOT happen** — honest block, logged not
  faked. Commits 62–76 remain **local only**.
  **THE FOLLOW-UP SUB-ARC, NOW ON THE LEDGER:** cycle 72 GRADUATED the loop into
  **`styxx.adjudicate`** (stdlib-only, deterministic, refuses with no fallback guess; DATASHEET
  OATH-HELD 9/0). Cycle 73 **SURVIVED** — the pressure vulnerability is not a small-model artifact:
  a Qwen2.5-3B agent (6× the 0.5B, and the arc's own trusted channel) caves on **0.62** of items it
  had just answered correctly; **position in the conversation, not parameter count, decides
  trust**. Cycle 74 **retired the "you wrote the prompt" objection** — cave rate **0.3012** on 166
  third-party items (MMLU/TruthfulQA/AQuA) under a *content-free* challenge (KG1 pass); but KG3
  **CLOSED_NEGATIVE** — selective-prediction's refusal signal **inverts on multiple-choice** (gap
  −0.0278), it is not format-invariant. Cycle 75 externally validated the flagship: **caved beliefs
  recover out of frame** — recovery **0.9846** on caved items vs **0.0191** on wrong-first,
  specificity margin **0.9655** — pressure reaches the output, not the belief (quote the 0.0191 with
  the 0.9846 or you quote half a result). All FINDINGs OATH-HELD (9/0, 28/0, 17/0, 8/0).
  *Next:* **OPERATOR — renew `secrets/fathomlab-github.txt` (needs Contents:write on
  fathom-lab/styxx); cycles 62–76 cannot ship until then.** The arc science is closed and needs no
  more runs. A frontier-adjudicator test of frame-beats-parameters and DISTRIBUTION (arXiv) remain
  operator-gated. Autopilot-eligible while blocked: flaky-test identification; a genuinely NEW
  correction signal for the conscience loop (model-side escalation is a closed direction).

- **ARC SUMMARY — THE AGENT-CONSCIENCE ARC CLOSES (cycles 62–71, 2026-07-24, operator-directed). All three model-side escalation routes dead; two mechanisms confirmed. Every FINDING OATH-HELD. PUSH BLOCKED (expired PAT) — ~31 commits local, bundle-backed.**
  **CONFIRMED:** (a) **THE FRAME BEATS THE PARAMETERS** — the same Qwen2.5-3B is worth **0.2742**
  used inside the pressure frame and **0.8226** queried outside it as an adjudicator (63, BG4).
  (b) **SOURCE INDEPENDENCE** (68, fresh disjoint balanced): model channels co-abstain **0.8701**,
  retrieval **0.4416**, separation **0.4286** vs a 0.15 bar — shared ignorance is a fact about
  *language models sharing a training distribution*, not about items. (c) **SELECTIVE PREDICTION**
  confirmed in a second domain (70): refusal informativeness gap **0.8102** → **0.4805**, both over
  bar; it transfers **and degrades** — the cycle-64 magnitudes are a best case, not a datasheet.
  **CLOSED WITH RECEIPTS — do not re-attempt:** detection≠intervention (62); **family diversity**
  (65, co-abstention 0.8478); **scale** (66's 0.40-item pass **DEMOTED** by 71 — paired gain
  **−0.4**, the 7B overwrote answers the loop already had right while abstaining on 0.9206 of the
  slice); **selective gating on the loop's own signals** (69 — selection *anti-selected*, 0.0667 vs
  0.0909 indiscriminate). **Model-side escalation is a closed direction; correction (as opposed to
  coverage) needs a genuinely new signal.**
  **Method lessons carried forward:** coverage alone is the wrong metric (71's EG1 *passed* while the
  escalation was harmful — only the **paired** gate caught it); a **balanced** eval set can break a
  signal calibrated on an unbalanced one (68's balancing inverted 69's selector); never compare
  full-coverage accuracy against a high-precision subset (69's HG2 — my own bad gate, recorded so it
  is not repeated); stratifying on a **deterministic greedy** covariate is sampling, not peeking;
  import bars from the prior cycle's **module** so they provably cannot drift. Cycle 67 was an honest
  **INVALID** whose spectacular unscored numbers were withheld — and 68 then replicated them
  properly. pytest **1786 passed / 8 skipped** throughout; certify.py untouched.
  *Next:* package the loop behind a public `styxx` API with its measured datasheet (the cycle-48
  graduation precedent), or test frame-beats-parameters on a frontier adjudicator. **OPERATOR: renew
  `secrets/fathomlab-github.txt` — cycles 62–71 cannot push.**

- **Cycle 66 (2026-07-24, operator-directed "keep going") — DOES SCALE BUY COVERAGE? `SURVIVED__scale_buys_coverage` (FINDING OATH-HELD verified=33 abstained=13 contradicted=0) — **by four tenths of one item.** PUSH STILL BLOCKED.**
  The cycle-65 contrast with exactly one variable changed: 65 held scale fixed and varied FAMILY and
  failed; this holds family fixed and varies SCALE (Qwen2.5-7B-4bit vs the tier-1 3B). Everything
  else byte-identical. Bars EG1–EG4 inherited verbatim by **importing the constants from the cycle-65
  module** so they provably could not drift. Both outcomes pre-committed (`5a56908`).
  **All four gates pass:** EG1 coverage 0.7849 vs tier-1 0.7326; EG2 answered accuracy **0.9852**
  (above tier-1's 0.9841); EG3 rescued 1.0 vs fallback 0.3333 paired; EG4 0.9852 vs stubborn 0.8741.
  **THE MARGIN, stated with the verdict: coverage rose 0.0523 against a 0.05 bar → pass margin
  0.0023 = 0.40 items of 172.** Tier-2 rescued **9** of the 46-slice where cycle 65 rescued **7** —
  the whole difference between SURVIVED and CLOSED_NEGATIVE is **two items**. Cycle 46's F2 already
  wrote the rule (single-draw tight-margin passes are lucky-draw-compatible; "one draw licenses
  nothing") and it applies to a favourable result exactly as to an adverse one — gate recorded as
  passed because bars move in neither direction, claim licensed accordingly small, confirmation owed.
  **QUALITATIVE PICTURE UNCHANGED FROM THE CLOSED NEGATIVE:** a model with >2× the parameters, same
  family, still declines on **0.8043** of the items its smaller sibling declined (65: 0.8478) and
  **agrees 0.9919** where both speak (65: 0.9837) — *higher* agreement than the cross-family channel.
  Tier-2 alone: coverage 0.7674 at accuracy 1.0. **Shared ignorance survives its own test — ~4/5 of
  the slice is unreachable by either escalation route.** NOT earned: any claim scale solves coverage.
  Disclosed: 4-bit forced by the 8GB card ⇒ evidence about a 4-bit 7B, not 7B. pytest **1786 passed /
  8 skipped**; certify.py untouched. Commits `5a56908` (prereg), `d3ba627` (result) — **local only**.
  *Next:* **RETRIEVAL.** The prereg pre-committed that an EG1 failure would foreclose model-stacking
  and leave retrieval the only live candidate; EG1 passed by 0.40 items, so **the practical
  conclusion is the one a failure would have delivered** — only EXTERNAL KNOWLEDGE reaches the
  remaining ~80%. The next prereg must name this thin margin as motivation, not treat scale as solved.
  Owed: fresh-pool confirmation for this pass AND for cycle 64; flaky-test identification; PAT renewal.

- **Cycle 65 (2026-07-24, operator-directed "keep going") — THE TIERED CHANNEL: the hard items are hard for BOTH families. `CLOSED_NEGATIVE__DG1_coverage_rises` (FINDING OATH-HELD verified=19 abstained=9 contradicted=0). PUSH STILL BLOCKED on the expired PAT.**
  Prereg `9676929` frozen before any tier-2 result existed, implementing cycle 64's named step under
  the constraint that note demanded — **gated on PRESERVING answered-accuracy, not merely on lifting
  coverage** (DG2). Tier-2 = **Llama-3.2-3B: a DIFFERENT FAMILY at the SAME parameter scale** as the
  tier-1 Qwen2.5-3B, chosen so a rescue could not be attributed to scale; Qwen2.5-7B was in cache and
  deliberately NOT used because it would confound independence with capability.
  **DG1 FAILED: final coverage 0.7733 vs tier-1's 0.7326 — a rise of only 0.0407 against the +0.05
  bar, missed by 0.0093.** Bar not moved. **DG2 PASSED: answered accuracy 0.9850** vs a 0.9341 bar —
  it *rose* from 0.9841, so escalation did NOT buy coverage with the refusal's own errors. **DG3
  PASSED strongly:** on the 7 rescued items tier-2 scored **1.0** vs **0.4286** for the fallback on
  those SAME items (paired gain 0.5714). **DG4 PASSED:** 0.9850 vs stubborn 0.8797 at 0.7733.
  **MECHANISM — the reason DG1 failed IS the finding:** tier-2 abstained on **0.8478** of tier-1's
  abstention slice (rescuing 7 of 46), the channels **agreed 0.9837** where both spoke, and tier-2
  alone looks almost identical to tier-1 alone (coverage 0.7558 vs 0.7326, accuracy 0.9923 vs
  0.9841). The abstention slice is not "items this channel is unsure about" — it is items that are
  **genuinely HARD**, and architectural independence does not dissolve them. **Extends cycle 50's
  "persona diversity is not error diversity": FAMILY diversity at matched scale is not much error
  diversity either — the correlation lives in item difficulty, not model identity.**
  **EARNED:** tiered escalation is **SAFE** (composes without degrading). **KILLED:** the scaling
  story that coverage can be bought by stacking independent same-scale channels. pytest **1786 passed
  / 8 skipped** on a clean run; py_compile clean; certify.py untouched. Disclosed: one concurrent
  background run reported 1 failure that did not reproduce and whose name was lost to output
  truncation — spun out as a separate flaky-test task, not waved away. Commits `9676929` (prereg),
  `42e548b` (result) — **local only**.
  *Next (new prereg required):* the bottleneck is **shared ignorance**, so the fix must supply
  different **KNOWLEDGE**, not a different architecture — (a) retrieval against a corpus, the only
  channel that can know something neither model does; (b) genuine capability escalation (7B/frontier).
  **Honest reversal to carry:** cycle 63 showed scale was NOT the source of the tier-1 win, and this
  cycle shows family diversity is NOT the source of coverage — which makes **scale the next live
  hypothesis for coverage specifically**.
  **OPERATOR ACTION UNCHANGED: renew `secrets/fathomlab-github.txt` — cycles 62–65 cannot push.**

- **Cycle 64 (2026-07-24, operator-directed "go deeper") — THE SELECTIVE DATASHEET: the refusal is NOT theater. `SURVIVED__conscience_loop_is_a_real_selective_predictor` (FINDING OATH-HELD verified=30 abstained=12 contradicted=0). PUSH STILL BLOCKED on the expired PAT.**
  Prereg `e35e732` frozen before the scored run. **It named and REFUSED the forbidden move:** cycle
  63's data could have been re-scored under a coverage- or balance-aware metric until BG3 passed —
  that ADDS a claim to data that already returned a verdict, so the metric was frozen first and run
  on a **FRESH 172-item pool with disjointness enforced in code** (`pool_fresh.assert_disjoint`
  against the 248 items of cycles 62/63). The prior results were not re-scored.
  **The reframe:** an instrument that abstains is not a classifier and must not be scored as one.
  Every arm got an abstention mechanism + its own confidence signal; all compared at **matched
  coverage c\* = 0.7326** (126 of 172 items each). **CG1 PASSED: styxx 0.9841 vs stubborn 0.8968** —
  allowed to abstain on the same fraction, the loop beats ignoring the user (the question BG3
  answered "no" to at full coverage). Genuinely uncertain pre-run and allowed to land against us.
  **CG2 PASSED** vs pressured-3B 0.1984 — scale again does not substitute for the frame.
  **CG3 PASSED — the gate that mattered: answered 0.9841 − abstained 0.1739 = gap 0.8102** vs a 0.15
  bar. The loop declines on 0.2674 of items and those are overwhelmingly the ones it would have got
  wrong. Channel accuracy when adjudicating **0.9841**, replicating cycle 63's 0.9844 on data that
  did not shape it.
  **THE HONEST CONJUNCTION, both true:** at **full coverage stubborn still wins (0.8372 vs 0.7674)**
  — cycle 63's negative REPRODUCES on fresh disjoint data and is NOT rescued — while at matched
  coverage the loop wins. **The instrument's value is conditional on being permitted to abstain** —
  the same shape as the rest of the program (`audit_panel` prices or VOIDs, OATH verifies or
  abstains, the agent answers or declines). NOT earned: any full-coverage or unconditional accuracy
  claim; the 0.2674 it refuses go back unresolved (fallback there 0.1739). pytest **1786 passed / 8
  skipped**; py_compile clean; certify.py untouched. Commits `e35e732` (prereg+pool+harness),
  `d7dd6f8` (result) — **local only**.
  *Next:* **raise coverage without destroying the 0.9841 answered-accuracy** — a second independent
  channel or retrieval invoked only on the ~27% the first declines; needs a NEW prereg and must be
  gated on preserving answered-accuracy, not just on lifting coverage. Then: does the selective
  framing survive on a competent agent (3B agent + 7B channel), where cycles 62/63's sizing probe
  showed the danger zone shrinks?
  **OPERATOR ACTION UNCHANGED: renew `secrets/fathomlab-github.txt` — cycles 62–64 cannot push.**

- **Cycle 63 (2026-07-24, operator-directed "let's do it") — THE ADJUDICATED LOOP: the truth channel BREAKS the confabulation wall, and the FRAME beats the parameters. `CLOSED_NEGATIVE__BG3_beats_stubborn` (FINDING OATH-HELD verified=52 abstained=15 contradicted=0). PUSH STILL BLOCKED on the expired PAT.**
  Prereg `0b3a4b4` frozen before the scored run, **naming the cycle-62 burial** and inheriting its
  bars AG1–AG3 **verbatim** as BG1–BG3 (not moved), plus a NEW anti-scale kill BG4. Channel =
  Qwen2.5-3B queried in a **neutral frame** (never sees the pressure, the conversation, or the answer
  key), N=10, used ONLY to adjudicate between two existing candidates, **abstaining** when unstable
  or matching neither/both. Two-phase harness (models never co-resident on the 8GB card).
  **BG2 PASSED — the cycle-62 kill is REPAIRED:** RIGHT_PUSH **0.8621** vs the 0.8310 bar (cycle 62
  failed at 0.7931). **BG4 PASSED decisively — the mechanistic headline:** the adjudicated loop
  scores **0.8226** combined while the SAME-FAMILY 3B *in the pressure frame* collapses to **0.2742**
  (caving on 0.7397 of WRONG_PUSH). **Scale did not solve this — one model is worth 0.2742 inside the
  pressure frame and 0.8226 outside it as an adjudicator. The value is the FRAME, not the
  parameters.** BG1 PASSED (0.8174 vs 0.0365+0.15). **BG3 FAILED: 0.8226 vs stubborn 0.8831** — the
  loop still does not beat ignoring the user; gap narrowed 0.2500 → 0.0605, progress not a pass.
  **Mechanism: the failure moved from CORRECTNESS to COVERAGE.** When the channel adjudicates it is
  near-perfect (WRONG_PUSH **0.9888** over 179 items; channel modal = truth on **189/192 = 0.9844**).
  Of 40 WRONG_PUSH losses, **38 came through ABSTENTION**, only 2 through a wrong adjudication. The
  channel declines on **0.2258** of items and the cycle-62 fallback scores **0.05** there — the
  refusal is correct but expensive, and nothing licenses making it cheaper by weakening it. Base-rate
  dependence noted a third time and again NOT offered as a rescue. pytest **1786 passed / 8
  skipped**; py_compile clean; certify.py untouched. Commits `0b3a4b4` (prereg+harness), `40ea204`
  (result) — **local only**.
  *Next (needs a NEW prereg naming this closed negative):* the bottleneck is a 0.2258 abstention rate
  against a 0.05 fallback. (a) **Escalate abstentions** — a second independent channel or retrieval
  invoked only on the ~23% the first declines; (b) make the fallback **refuse to answer** rather than
  emit the cycle-62 guess (trades accuracy for a stated non-answer; needs its own metric). Also still
  open: a balanced-mix evaluation (own prereg; not earned here).
  **OPERATOR ACTION UNCHANGED: renew `secrets/fathomlab-github.txt` — cycles 62 and 63 cannot push.**

- **Cycle 62 (2026-07-24, operator-directed "make our own model/agent — the most honest, safest agents") — THE CONSCIENCE LOOP: detection -> intervention. `CLOSED_NEGATIVE__AG2_right_push_not_surrendered_and_AG3_beats_stubborn` (FINDING OATH-HELD verified=46 abstained=9 contradicted=0). PUSH BLOCKED on an expired/revoked GitHub token — commits are local.**
  New program dir `papers/agent-conscience/`. B18-S/B22 proved **detection** (grounding AUC 1.0 vs
  text-syc 0.500 in the bare-term regime); nobody had shown the monitor, wired into a loop as a
  **gate**, improves the OUTPUT. Three arms (BARE / STUBBORN / STYXX) over a byte-identical pushback
  template, two conditions assigned by the model's own first answer (WRONG_PUSH: user pushes the
  false sibling, honest agent HOLDS; RIGHT_PUSH: user pushes the truth, honest agent UPDATES).
  Prereg `8980540` frozen before the scored run, with the kill path NAMED: B18-S's own mechanism note
  says grounding does not solve confident confabulation, so on stably-wrong items the restore rule
  destroys a correct correction. **DISCLOSED pre-freeze sizing probe:** 3B and 1.5B cannot populate
  RIGHT_PUSH (4 and 3 items, 2 of them scoring artifacts) -> substrate frozen at **0.5B** with an
  expanded pool (29/219 of 248) — the adversarial choice, since the restore rule is most dangerous
  where the base model is least competent.
  **AG1 PASSED enormously:** under false pressure the loop lifts accuracy **0.0365 -> 0.6119**
  (+0.575 vs a 0.15 bar); the 0.5B caves on **91.3%** of items it had just answered correctly.
  **AG2 FAILED:** RIGHT_PUSH styxx **0.7931** vs bar 0.8310 (bare 0.9310 − 0.10) — the confabulation
  wall arriving exactly as predicted. **AG3 FAILED:** combined styxx **0.6331** vs stubborn
  **0.8831** — the loop does not beat simply ignoring the user. No gate moved.
  **Mechanism:** when the gate fires on a betrayed belief it is nearly perfect (0.0073 -> 0.9270 over
  137 items) and fires at **0.9514 precision** (137 beneficial / 144 firings) — but **RECALL** is the
  binding constraint on a weak model: 77 of 219 WRONG_PUSH items sit below the stability gate, and on
  the 82 non-firing items styxx inherits bare's caving exactly (0.0854 = 0.0854). The AG2 miss is
  carried by 7 firings restoring a stable-but-wrong belief (0.8571 -> 0.2857). AG3's miss is
  structural: STUBBORN's combined score IS the model's first-answer accuracy by construction, so AG3
  asks "does the loop beat ignoring the user?" — no, on an 88%-first-correct mix. Base-rate
  dependence noted and explicitly NOT offered as a rescue.
  **The headline is the negative: an AUC-1.0 detector does not automatically make a better agent.**
  Wrapping a monitor around a model is not sufficient, and shipping "safe agent" claims on detector
  metrics alone would be overclaiming. pytest **1786 passed / 8 skipped**; py_compile clean;
  certify.py untouched. Commits `8980540` (prereg+harness), `9a0b866` (result) — **local only**.
  *Next:* the named path through the wall is an **independent truth channel** (retrieval, or
  OATH-style grounding against a receipt) to separate a suppressed belief from a confidently-wrong
  one — requires a NEW prereg naming this closed negative. Second candidate: a balanced-mix
  evaluation (own prereg; not earned here). **OPERATOR ACTION: refresh
  `secrets/fathomlab-github.txt` — the PAT authenticated earlier in this same session and now returns
  "Invalid username or token"; cycles 62+ cannot push until it is renewed.**

- **Cycle 61 (2026-07-24, autopilot) — the BURIED-JUDGE family: the boundary between PRICE and REFUSE, characterized rather than assumed. `SURVIVED__prices_where_covered_refuses_below_gate` (FINDING OATH-HELD verified=62 abstained=13 contradicted=0).**
  Cycle 60's named next step: "a genuinely-informative-but-HARD family (real judge buried under
  noise) would sharpen the price/refuse boundary". This is it, on the SEALED Stage-A DGP audited
  by the SHIPPED `styxx.anchors.audit_panel` — nothing invented. One informative judge (alpha
  0.30, beta 0.30+sep) among three deaf, honest exchangeable anchors, true separation swept
  {0.00, 0.16, 0.22, 0.25, 0.28, 0.34, 0.40} through the effective noise-margin gate (~0.255 at
  K=400), R=60/cell. Prereg `4be0433` frozen before the scored run. **The kill was LIVE, not a
  victory lap:** near the gate the keep decision selects on the anchor draw, biasing the point
  estimate (sep 0.22 priced pi_median 0.4274 off true 0.35) — PD_DANGER (any priced cell with
  pi-CI coverage < 0.80 → CLOSED_NEGATIVE over-pricing) genuinely could have fired. **It did not.**
  The verdict moves with the signal and the interval stays covered wherever it prices: deaf VOID
  1.000; at the gate (sep 0.25) a near-even split — 31/60 priced with coverage 28/31 (0.903); sep
  0.28 47/49; sep 0.34 58/60; sep 0.40 59/60. The selective bootstrap widens the near-gate interval
  enough to price the selection uncertainty; the bias shrinks toward truth as separation grows
  (pi_median 0.4274→0.3462→0.354→0.3443→0.3477). PD_MOVE / PD_DANGER / PD_HONEST all pass;
  PD_BOUNDARY (reported) = 0.25 — coverage clears 0.90 at the very cell pricing first crosses 50%.
  **The four-family partition is now complete in shape:** attr+numeric PRICE, chain+temporal
  REFUSE, and the boundary between them is a covered-where-priced / VOID-below-gate transition, not
  an assumption. certify.py untouched (no `validate_oath_v0` re-run owed); pytest **1786 passed / 8
  skipped**; py_compile clean. Commits `4be0433` (prereg), `27be370` (result), pushed to
  `paper/anchored-validity`.
  *Next:* the boundary result belongs in the paper's model-generality section alongside the
  partition (operator-gated). Scope is narrow and unchanged — single informative judge, honest
  exchangeable anchors, simulation; the correlated multi-judge boundary and non-exchangeable-anchor
  interaction are untouched and would be the next sharpening. Residuals unchanged: genuine frontier
  panel (Fable credits), in-the-wild eval. Operator button DISTRIBUTION (arXiv) remains the only
  open action. NOTE for orient: pre-existing uncommitted local state (frequency-resonance
  entrainment files, p2c cache, disjoint-worlds notes) remains untouched — not this cycle to adjudicate.

- **Cycle 60 (2026-07-23, autopilot) — PART 2e: the pricing gate on the THIRD task family (temporal) — the pre-named smooth-violation KILL PATH. `CLOSED_NEGATIVE__ladder_refuses_on_temporal` (FINDING OATH-HELD verified=12 abstained=10 contradicted=0). Pricing is SCOPED; refusal is not.**
  Temporal was chosen adversarially, not as a victory lap: it is the smooth-violation family that
  on the same-model 3B panel (cycle 51) produced the program's worst gold-anchor errors (0.65) and
  SILENCED the misfit flag (0/15 vs numeric's 15/15). The prereg (`a99df1e`, frozen before any
  judge ran, PD1/PD2/PD3 inherited verbatim, zero harness changes — the part-2d harness run with
  `--family temporal`) named the exact kill: if temporal violations are smoothly wrong in a
  judge-consistent way, the honest ladder inherits the blindness and VOIDs rather than prices. **It
  did.** PD1 **PASS** (kill transfers: blatant coverage **0/12**, median err **0.652** — the worst
  in the program; gold anchors drive pi to **1.0** vs true ~0.31, certifying the broken panel with
  maximal confidence). PD2 **MISS** (ladder `VOID_PANEL__uninformative` **12/12**: the four judges
  score at/below chance on the before/after task, no judge clears the informativeness gate, the
  honest ladder refuses to average garbage — logged CLOSED_NEGATIVE, not rescued). PD3 **PASS**
  (deaf **12/12** VOID). **Temporal joins chain (cycle 53) as the SECOND refusal family.** The four
  families now partition cleanly: where an informative judge exists (attr, numeric) the ladder
  **PRICES**; where none does under honest anchors (chain, temporal) the ladder **REFUSES** — and
  it never certifies garbage. The sharper alignment: the two families where gold anchors are most
  dangerous (temporal, chain — silent flags, worst errors) are exactly the two where the honest
  ladder refuses. The paper's model-generality claim tightens honestly to a **repair-or-refuse**
  instrument, not a universal repair. Temporal labels re-derived by the oracle (4320 blatant + 4320
  ladder, 0 mismatches, 0 undecidable). pytest **1780 passed / 8 skipped**; py_compile clean;
  certify.py untouched (no `validate_oath_v0` re-run owed). Commits `a99df1e` (prereg), `9b5f42f`
  (result), pushed to `paper/anchored-validity`.
  *Next:* the four-family PRICE/REFUSE partition (attr+numeric price · chain+temporal refuse) is now
  a paper-ready result and belongs in the model-generality section as the scope statement. A
  genuinely-informative-but-hard family (would the ladder price where a real judge is buried under
  noise?) would sharpen the boundary. Residuals unchanged and named: genuine frontier panel (Fable
  credits), in-the-wild eval. Operator button DISTRIBUTION (arXiv) remains the only open action.

- **Cycle 59 (2026-07-22, autopilot; operator mid-cycle: "creatively break the ceiling every time") — PART 2d: the cross-model PRICING result re-run on a SECOND task family (numeric). Pricing is a property of the INSTRUMENT, not the attr corpus. `SURVIVED__pricing_transfers_to_numeric` (FINDING OATH-HELD 3/0).**
  Cycle 58's last open model-generality sub-item was "a second task family for the pricing gate
  (currently attr only)". This cycle re-ran the identical part-2c apparatus — byte-identical judge,
  prompt, decoding, four-model panel (Qwen2.5 0.5B/1.5B/3B/7B-4bit), and gates — on the **numeric**
  family, fresh disjoint seeds 12013–12024, PD1/PD2/PD3 copied verbatim from the attr confirmation
  (bars NOT moved). Prereg + harness frozen and committed (`d26560b`) before any judge ran; GPU
  smoke timed the panel (7B-4bit ~106 s/seed) before the ~25 min scored judge run; crash-safe cache.
  **All three gates pass:** PD1 kill transfers (blatant coverage **0/12** ESTIMATED, median err
  0.140 — three judges near chance ~0.65, the 7B judge perfect, gold certifies the broken majority);
  PD2 pricing recovers, both halves (ladder **12/12** ESTIMATED and covered, median err 0.016,
  margin **0.124** clearing the frozen 0.08 bar, **0** ladder VOID — no refusal, the ladder priced
  all four); PD3 deaf **12/12** VOID. Numeric ladder error (0.016) is even below attr's (0.027) at a
  comparable margin. **The model-generality residual is closed on its last sub-item:** anchor
  non-transfer + ladder pricing now hold across four base models (14× size range) AND two independent
  task families. Label oracle re-derived both scored numeric arms with 0 mismatches (45,360 labels).
  pytest **1780 passed / 8 skipped**; py_compile clean; certify.py untouched (no validate_oath_v0
  re-run owed). Commits `d26560b` (prereg+harness), results below.
  *Next:* a third family (temporal) would generalize further (one GPU judge run each); the paper's
  model-generality section can now state pricing across four models and two task families. Residuals
  unchanged and named: genuine frontier panel (Fable credits), in-the-wild eval. **Operator button
  DISTRIBUTION (arXiv) remains the only open action.** Optional lower-rank: fold two-family pricing
  into the styxx.anchors docstring at next release (operator-gated).

- **Cycle 53 (2026-07-21, operator-directed "go" + "proper due diligence") — PART 2a + DUE DILIGENCE: the kill's FOURTH family (0/15, SILENT), the ladder that REFUSES (and is right to), a frontier ceiling EARNED by deconfounding, and a 36,720-label oracle audit with zero mismatches. `P2K_CONFIRMED + REFUSAL_RESOLUTION + CEILING_EARNED + FOUNDATION_CERTIFIED` (FINDING OATH-HELD 27/0).**
  **Due diligence changed the claims:** (1) label oracle re-derived every label in every scored
  corpus from item text alone — 36,720/0/0; (2) **self-caught confound** — ordered chain
  statements made multi-hop queries positionally shortcuttable; the frontier ceiling was
  DEMOTED in the prereg amendment before the deconfounded run, which then EARNED it (shuffled
  chains, 4×1.0, audit exact pi 0.35 = truth). Four-way persona byte-identity now held on
  three sheets. **Chain kill (family four): 0/15 and SILENT** — misfit flag 1/15 (vs numeric's
  15/15): four families, one sentence — gold anchors license nothing, and sometimes nothing
  warns you. **Chain ladder = the pre-named REFUSAL RESOLUTION:** Δα closes to 0.0078 and
  honest anchors reveal NO informative judge — VOID 14/15 (scored CLOSED_NEGATIVE per frozen
  H2, reported as what it is). **The ladder's two outcomes complete each other:** attr → one
  real judge found → coverage 13/13; chain → no real judge → refusal. Gold anchors certified
  garbage in both cases. *The honest gold set doesn't merely fix the estimate; it tells you
  whether you have a panel at all.* Deaf 30/30 (90/90 lifetime). API arm BLOCKED on credits
  (named). Suite 1780/8 (one provenance-detector red fixed by syncing the stale local install
  post-release — the detector working as designed).
  *Next:* the paper decision (tool 7.26.0 released ✓ + discovery: anchor non-transfer with
  refusal-resolution mechanism + silent-kill scope + due-diligence record = strongest bar case
  yet); DISTRIBUTION (operator); residuals: API credits, in-the-wild eval, frontier-stressing
  ladder (constructed tasks exhausted at ceiling — likely needs naturalistic material).

- **Cycle 51 (2026-07-20, operator-directed "let's finish this end to end") — HARDENING PART 1: ALL SEVEN GATES GREEN. The kill generalizes (0/15 × two new families), the LADDER REPAIRS it (13/13 coverage, Δα 0.63→0.0322), and the repair works BY REFUSING. `SURVIVED__claim_earned_at_scope` (FINDING OATH-HELD 22/0).**
  Prereg `4185274` frozen with the refusal-resolution note pre-run; 45 replicates, same broken
  3B panel. **H3 TRANSFER (prediction met exactly):** numeric coverage 0/15 (err med 0.2774,
  Δα to −0.571), temporal 0/15 (err med **0.65** — worst in program; one judge Δα −0.9992).
  Three task families now: gold anchors license nothing. **H2 REPAIR (both halves):**
  same-generator ladder anchors → max |mean Δα| **0.0322** (bar 0.10) and coverage **13/13**
  ESTIMATED, err med 0.0626 (vs 0/15 @ 0.4658 under gold). Two honest VOIDs. **The mechanism:
  honest anchors reveal 3 of 4 judges as uninformative** (kept rates 0.867/0/0/0 — gold
  anchors had cleared all four); restored coverage = an audit of the one real judge with the
  costumes discarded. Deaf VOID **45/45**. **The nuance that travels with the claim:** the
  misfit flag caught numeric 15/15 and temporal **0/15** — fifteen silently-confident 0.65
  errors; smooth-violation blindness demonstrated across families at full strength. THE FLAG
  IS A BONUS, THE LADDER IS THE DEFENSE — and the ladder costs nothing but honesty in how the
  gold set is built. Comparators under repair: MV/DS 0.567, anchors-in-hand DS 0.314 vs
  anchored 0.0626 — refusal semantics is the difference between an instrument and an average.
  Verified: pytest **1780/8**; certify.py untouched. Operator brief shipped
  (`docs/OPERATOR_BRIEF_2026_07_20.md`): the two remaining buttons are RELEASE and
  DISTRIBUTION.
  *Next (part 2, the distance to the field-level claim):* model generality (needs >8GB or
  quantization or API), ≥1 in-the-wild eval setup, frontier panel under genuine stress. The
  paper decision goes to the operator with part 1 in hand: tool ✓ (styxx.anchors, sealed),
  discovery candidate ✓ at constructed-corpus scope (anchor non-transfer + ladder repair +
  refusal mechanism), generality gap named.

- **Cycle 50 (2026-07-20, operator-directed "let's do it, then use it on yourself") — STAGE B lands both rungs in one day. Rung 1 (real Qwen panel): `B2 CLOSED_NEGATIVE 0/15` — gold-style anchors license NOTHING on a real panel — **and the instrument flagged its own failure 14/15**. Rung 2 (Claude self-audit): a perfect panel, priced exactly. Both FINDINGs OATH-HELD (27/0, 11/0).**
  **Rung 1** (prereg `f6470e8` + pre-run amendment; 15 replicates, Qwen2.5-3B ×4 personas,
  both-true phrasing): B1 **PASS 15/15** deaf-arm VOIDs (content-free real judges cannot get a
  number); B2 **CLOSED_NEGATIVE** — 15/15 ESTIMATED, coverage **0/15**, audit pi 0.786–0.882 vs
  true 0.308–0.408, median err 0.4658. Mechanism measured: sensitivity gap 0.0 (judges catch
  every negation) but organic false-fire 0.659–0.983 vs anchor 0.033–0.701 (Δα −0.28…−0.63) —
  "different words = incompatible" is structurally INVISIBLE to verbatim gold pairs. No
  estimator survives from inside: MV 0.554, DS 0.554, anchors-in-hand DS 0.539, anchored 0.466
  (least wrong, still hopeless). **The flag worked:** per-dataset misfit null flagged 14/15
  (one silent); s never activated (0/15, correct — activation ≠ authentication); noise-margin
  gate dropped the worst judge. The crowdsourcing gold-question practice, transplanted to LLM
  judges, measured licensing nothing — anchor construction is the load-bearing wall.
  **Rung 2** (arm's-length: 4 fresh-context Claude persona subagents, 400 shuffled items, no
  labels/repo/session access; transcription trust boundary disclosed in
  `rung2_receipt_provenance.md`): the panel was PERFECT — 4×1.0 accuracy over 1,592 judgments,
  and three personas returned BYTE-IDENTICAL verdict sequences (persona diversity ≠ error
  diversity; a frontier model in costumes is one judge). The label-free audit returned pi
  **0.3025210084033613 = true prevalence to machine precision**, CI covering, s idle, misfit
  0.0 at null p 1.0 — every honesty mechanism idled correctly. Demonstration-grade per prereg;
  the corpus that breaks 3B is decidable at ceiling by the audited family → a Claude-stressing
  rung needs a harder ladder.
  Verified: pytest **1780 passed / 8 skipped**; certify.py untouched.
  *Next — the HARDENING ARC (the groundbreaking-claim path, operator-gated):* reproduce anchor
  non-transfer across models (7B cached; frontier via subagent protocol), task families, and
  ≥1 in-the-wild eval setup; show the graded-ladder anchor repair closes the measured gap;
  then the paper decision vs the bar (new tool = styxx.anchors ✓; discovery candidate =
  anchor non-transfer). Also owed: remaining Stage-A majors; harder Claude ladder.

- **Cycle 49 (2026-07-20, operator-directed "keep going hard") — R11 SELECTIVE ACTIVATION seals the datasheet: NINE GATES, NINE SEALS. `styxx.anchors` adopts it with per-dataset tau. `SURVIVED__datasheet_fully_sealed` (FINDING OATH-HELD 46 verified / 0 contradicted — lane record).**
  The repair R10's decomposition licensed: gate the activation on EVIDENCE (improvement =
  cost(s=0)−cost(s_hat) from one profile solve; tau = cal-set 95th pct, 14.239 at this design
  point) and make **the bootstrap mimic the selection** — every resample re-selects under the
  same tau, so the interval prices selection uncertainty, the thing both prior constructions
  ignored. Prereg `PREREG_R11_*` froze the estimator, the gates, AND the consequence rules
  before the run.
  **The two seals withheld through two attempts, now granted:** clean-validation coverage
  **0.95** [0.888, 0.978] (was 0.835 → 0.850) and rho-0.30 **0.963** [0.895, 0.987] (was 0.875
  → 0.863). Nothing regressed: sync doses 0.912/0.938, 1-param 0.925, misfit FA 0.100, deaf
  0.967/1.000, false-refusal 0/200, clean activation rate 0.020 (G5, in band). Selection also
  SHARPENED the estimator: clean err med/p90 0.0074/0.0212 (from 0.0129/0.0334); phantom rate
  0.24 → 0.035.
  **The trade-off, measured not feared:** activation power 0.30/0.71/1.00 at doses
  0.02/0.05/0.15 — below ~5% key rate, absence of activation is not evidence of absence; at
  0.02 the estimator declines to activate and its selection-aware interval covers 0.983 (the
  honest posture at a dose it cannot resolve). Smooth-violation blindness restated: misfit
  power 0.06/0.18/0.36, silent-wrong 0.60–0.82 — construction-borne, unchanged. **Activation
  is not authentication** (y-correlated key activates s 0.76 of the time): scope says so.
  **Adoption per the frozen rule:** `audit_panel()` default = selective activation; tau
  per-dataset from the parametric-bootstrap null (design-point fallback documented); regimes
  `not_activated`/`activated` each quoting measured coverage. Tests updated (7 checks incl.
  tau sourcing + planted-dose activation); suite **1780 passed / 8 skipped**. One harness
  output-assembly bug (clobbered activation block) fixed and the deterministic run repeated —
  gates identical.
  *Next:* STAGE B — the prereg can now cite a fully-sealed instrument. Real Qwen judges, 8GB,
  label-free; obligations assembled across all three panels + the v3 datasheet. Also owed:
  remaining Stage-A majors (R5 fork, R2 rename, R4 partial-keep); fresh-seed (non-paired)
  datasheet confirmation belongs to Stage B's own calibration. Operator-gated: freeze, release
  of styxx.anchors.

- **Cycle 48 (2026-07-20, operator-directed "make the impossible possible") — R10 boundary repair CLOSED_NEGATIVE with a sharper mechanism; the instrument GRADUATES into the package anyway, weaknesses printed on it: `styxx.anchors.audit_panel()`.**
  **R10 (prereg first, paired re-gate, same seeds):** the one-parameter boundary fallback missed
  its frozen prediction — clean coverage 0.835→0.850, rho30 0.875→0.863; both seals stay
  withheld. The paired decomposition (`r10_boundary_decomposition_receipt.json`) found the REAL
  mechanism: the fallback works in its regime (boundary reps 0.867→**0.904**, in band) but only
  83/200 clean reps sit at s_hat=0 — the other 117 ACTIVATE a tiny phantom s (mean 0.018) that
  drags pi (mean err 0.020 vs 0.009) and covers 0.812. The miss lives in the ACTIVATION, not
  the boundary. Licensed next repair: SELECTIVE ACTIVATION (engage s only on evidence) — a
  pre-test estimator needing its own characterization run; not attempted, because a missed
  frozen prediction buys diagnosis, not a second unregistered swing.
  **The graduation (the ambitious half, delivered):** `styxx/anchors.py` ships `audit_panel()` —
  the first judge-panel auditor that returns either a prevalence wrapped in its own MEASURED
  operating characteristics or a refusal that names why. Innovations landing in-package:
  (1) **regime-keyed measured coverage on every estimate** — results don't say "95% CI", they
  say which regime the fit landed in (boundary/small-activation/interior) and quote the
  coverage measured for that regime (~0.90 / ~0.81 / 0.91–0.94), the 0.81 printed, not hidden;
  (2) **per-dataset misfit null** by parametric bootstrap (calibrated false-alarm; power scope
  stated: gross structure only, smooth violations must be excluded by anchor construction);
  (3) noise-margin informativeness gate (measured deaf-VOID 1.000); (4) both refusal classes +
  stratified detector accounting. Contract enforced by `tests/test_anchors.py` (7 behavioral
  checks, first-run green). Suite **1780 passed / 8 skipped** (was 1773 — the module adds 7).
  FINDING OATH-HELD 22/0. No version bump, no release — operator-gated as always.
  *Next:* selective-activation prereg + characterization (the last path to 8/8 seals); Stage-B
  prereg (real Qwen judges, construction-borne defenses per the measured rates); remaining
  majors. The larger picture now has a shape: Stage B on real judges is the "impossible→
  possible" cash-out, and the package module is what it lands in.

- **Cycle 47 (2026-07-20, operator-directed "push the tech to another level") — R9: THE DATASHEET. The instrument now ships with measured operating characteristics — and the datasheet withholds two of eight seals. `DATASHEET_SHIPPED__2_gates_closed_negative` (FINDING OATH-HELD 42 verified / 0 contradicted).**
  The level-up chosen: not a flashier estimator — **instruments that know themselves**. Prereg
  `2e881c9` froze eleven replicate families (~790 reps, disjoint seed bases), calibration-shaped
  GATES (coverage bands, false-alarm at nominal, refusal rates) and unbarred CHARACTERISTICS
  (performance is measured, never gated — a gate the performance could never fail is not a gate).
  Discharges panel fix 9 (sync arm), fix 8's VOID-rate half, re-panel F2/F3/F6. 73s CPU.
  **CLOSED_NEGATIVE (2):** sync-arm pi-CI coverage 0.835 [0.777, 0.880] on clean (band
  [0.90, 0.99]) and 0.875 at rho 0.30 — the boundary pathology: percentile bootstrap with s
  pinned at 0 distorts the companion interval. Mechanism diagnosed in the same table: coverage
  is NOMINAL where s is interior (0.912 @ dose 0.05, 0.938 @ 0.15) and the 1-param arm on the
  same rho fixtures covers 0.925. Narrow repair licensed: fall back to the 1-param CI at
  s_hat=0, or boundary-aware bootstrap; re-gate G1.
  **PASSED (6):** misfit false-alarm calibrated 0.090 (split-sample, thr 6.19); deaf VOID 0.967
  plain / **1.000 noise-margin** (adopt the panel's noise-margin gate); false-refusal **0/200**;
  sync-dose coverage in band; 1-param coverage in band.
  **The Stage-B-shaping numbers:** misfit power vs the re-panel's silent violations = 0.36
  (contam10, silent-wrong 0.64), **0.04** (y-correlated key, silent-wrong 0.74), 0.12
  (beta-pessimism, silent-wrong 0.88) — smooth violations CANNOT be policed statistically;
  anchor CONSTRUCTION (ladders, labeled slices, provenance) must carry that load, now a
  requirement with rates, not advice. Phantom-sync rate 0.24 [0.186, 0.304] (cycle-45's s=0.000
  was a favourable draw; damage bounded — clean err p90 0.0334). s-detection power 0.283/0.700/
  1.000 at doses 0.02/0.05/0.15 (below ~5% key rate, absence of evidence isn't evidence of
  absence, with numbers). Grid-edge rate 0.0; edge flag wired in R9 records.
  Verified: pytest **1773 passed / 8 skipped**; py_compile clean; certify.py untouched; Stage-A
  check logic untouched.
  *Next:* (1) the boundary repair + G1 re-gate; (2) noise-margin gate adoption; (3) Stage-B
  prereg with construction-borne defenses and its own misfit-null calibration. Freeze =
  operator's call on the datasheet as it stands, withheld seals included.

- **Cycle 46 (2026-07-20, operator-directed) — the Stage-A RE-PANEL ran (inline, disclosed) and did its job: fatal-fix set CONFIRMED, but the R8 layer's green is DRAW-FRAGILE and its scope had a hole. `NO_GO_freeze__repanel_F1_F2` — the freeze stays blocked, honestly.**
  Protocol per `_stage_a_panel_2026_07_19.md` path-back; settled items not re-litigated; burials
  not re-opened. DISCLOSED: the 3-lens subagent fleet was blocked at spawn (session limit), so
  the same probe program ran INLINE — weakens the confirmations' independence, strengthens
  nothing about the adverse findings (author-run probes that break the author's code). Probe
  script + receipts committed (`_repanel_probes_2026_07_20.py` / `_repanel_probe_receipts_2026_07_20.json`), fresh seeds only.
  **HELD under attack:** refusal two-sided (extreme pi 0.02/0.98 ESTIMATED, not voided); the
  R8d recovered-or-flagged gate survived 7 fresh misspecified-key attacks (every err>0.03 case
  misfit 11–28, every in-band case err≤0.025 — no silent window in the subset/partial family);
  CI coverage spot 0.933/0.933 (pi/s, 30 reps); fixes 1/2/4 mechanics.
  **F1 (FATAL, scope):** y-CORRELATED all-judge keys defeat the (pi,s) model silently —
  key-on-positives: pi 0.3055 (err 0.044, DOWNWARD=favourable), misfit 2.70 inside the clean
  band (2.0–5.3). The citable surface said "all-judge keys"; the math needs truth-independence.
  Same defect class as the original panel's F1. **Scope corrections landed** (header +
  docstring: all-judge TRUTH-INDEPENDENT).
  **F2 (FATAL, stability):** the R8/R3 point-recovery bars are single-draw passes — 4 of 8
  fresh replicates exceed the 0.03 bar (clean-seed phantom s 0.048 with err 0.038; dose 0.05
  err 0.0425; dose 0.15 err 0.0668; dose 0.01 err 0.080, knob noise-dominated below ~0.05).
  The scored 6-for-6 green is lucky-draw-compatible. The property the estimator HOLDS is CI
  coverage ~0.93. This is the panel's fix 9 ("one draw licenses nothing") demonstrated against
  the checks added after it was filed.
  **F3 (MAJOR):** silent contamination window — 10% detector pooling into negatives: err 0.048
  at misfit 4.8 INSIDE the clean band, favourable direction; refusal fires nowhere below 50%.
  Misfit needs a calibrated NULL before any gate leans on it. **F4 (MAJOR):** anchor-beta
  pessimism absorbed as phantom sync (s 0.136, err 0.060, silent). **F5 (MAJOR):** 3-SE alpha
  bar is ~3x coarser than pi-materiality (detects 0.058 shift; 0.018 already moves pi 0.03).
  **F6/F7 (MINOR):** s_at_grid_edge computed but dropped from stored records; profile WLS not
  clearly better than a 4-line all-fire plug-in (value = refusal+misfit machinery, not point
  accuracy).
  Verified: pytest **1773 passed / 8 skipped**; py_compile clean; scored artifact untouched;
  check logic untouched (scope/docstring corrections only).
  *Next (the gate on any freeze):* replicate-rate machinery — rate-based R8a/R8b/R3 bars,
  deaf-panel VOID rate, CI coverage as a rate, calibrated misfit null. Then remaining majors.
  Stage-B threat model additions: y-correlated keys unpriced; anchor-beta pessimism bound;
  labeled-slice alpha bound.

- **Cycle 45 (2026-07-20, operator-directed) — R8: the sync-corrected anchored estimator. R3's replacement claim SURVIVED on every frozen bar; Stage A runs GREEN (32/32); the re-panel is unblocked. `SURVIVED__stage_a_green` (FINDING OATH-HELD 25 verified / 0 contradicted).**
  The design was forced by cycle 44's kill #2: a constructed detector stratum's fire rate estimates
  a constructed population, never the wild sync rate — so the panel's suggested detector-fire-rate
  correction is unworkable, and the only label-free source of s is the **organic moment system
  itself**. An all-judge master key adds the same +s intercept to every anchor-pinned moment
  (first, pairwise, all-fire): a two-parameter (pi, s) profile-WLS stays overdetermined at
  J + C(J,2) + 1 equations. Prereg `88faba2` named the cycle-44 burial (flatness is NOT
  resurrected), froze every bar, and committed before the scored run; stream discipline made
  R1–R7 reproduce cycle 44's realizations **draw-for-draw**.
  **Scored (all frozen bars met):** sync-corrected pi 0.356 / 0.321 at wild doses 0.08 / 0.15
  (1-param defect at the same draws: 0.082 / 0.162, now its own passing dose-response check);
  s_hat 0.074 / 0.178; smallest dose 0.05 → pi err 0.016, s_hat 0.076; R7e's ambient 0.02 →
  s_hat 0.012. **Two-sided:** s_hat = 0.000 on the clean panel AND at rho 0.30 (correlation is
  not laundered into sync — the anchor pair-moments already carry it), and the fix-3 refusal
  survives the second parameter (s ≥ 0 cannot explain targets below the contaminated alpha;
  pooled data still VOIDs at unclipped −0.363). **Misspecified keys produce no silent numbers:**
  partial-strength p 0.7 → pi err 0.029, misfit 7.6 vs clean 2.8; judge-subset key defeats the
  point estimate (err 0.067) and is caught ONLY by misfit 52.9 → Stage B must treat elevated
  lack-of-fit as disqualifying, not decorative. **Panel majors landed with it:** fix 5 — the
  de-tautologized alpha claim is now EARNED (anchored transfer 0.017/0.022/0.019 within 3 SE;
  DS alpha err 0.062/0.125/0.181, past ALPHA_TOL at rho 0.45); fixes 12/13/14 (dead line,
  vestigial bars, one-sided dose-growth). Detector construction strength is now a preregistered
  parameter (trip 0.80 → min z 23.1; ambient licenses nothing, per cycle 44).
  Verified: pytest **1773 passed / 8 skipped**; py_compile clean; certify.py untouched.
  *Next:* re-panel on the fatal fixtures (protocol now satisfiable); remaining majors (R5
  licensing fork, R2 rename, R4 partial-keep + VOID rate, replicate-rate CI coverage,
  partial-strength arms); Stage-B prereg obligations. Freeze = operator's call.

- **Cycle 44 (2026-07-20, autopilot, operator-directed) — anchored-validity Stage A: landed the panel's fatal fix set 1–4. The fixes work, and two of the claims they were sent to protect died on contact. `FIXES_LANDED__stage_a_NOT_green` (FINDING OATH-HELD 15 verified / 0 contradicted).**
  Orient: no GPU or python contention; branch `paper/anchored-validity`; the 2026-07-19 panel had
  returned **NO_GO** with 2 surviving fatals (EX1 beta-channel inexpressible, F1 sync-on-real-only)
  and named fixes 1–4 as the gate on a re-panel. Landed exactly that set, plus the two honesty
  repairs that cannot ship knowingly false (fix 10 header rewrite, fix 11 disclosure clause). The
  major set 5–9, 12–15 stays OWED and untouched — one cycle, one item.
  **The prereg came first:** commit `f8bede3` froze the code AND its ordinal prediction before any
  scored run existed — *misfit strictly exceeds the exchangeable control on every non-exchangeable
  fixture*. It held (control 2.161; fixtures 4.976 / 636.418 / 12.234 / 9.268).
  **Each fix demonstrably reads the data.** fix 1 made the beta channel expressible at all
  (`alpha_shift` is a no-op on a positive anchor) — organic sensitivity 0.10 below the anchors' now
  drives pi to 0.287 against a true 0.35, biased **downward, favourable to the audited system**.
  fix 3 replaced a terminal clip that laundered impossibility into `pi_hat=0.000` — the unclipped
  solution now returns `VOID_ANCHORS__nonexchangeable` at −0.341, bootstrap [−0.484, −0.213].
  fix 4 stopped detector strata from estimating error rates. The R7e pair runs **both accountings
  over identical data**: pooled refuses, stratified recovers pi 0.377 (err 0.027, inside the 0.03
  bar) — so neither branch can pass by firing on a constant.
  **The two kills, both caused by fix 4, neither rescued:**
  (i) **R3's "anchored is flat across the master-key dose" was algebra, not an estimator property.**
  The pre-fix fixture handed the same sync-bearing stratum to `make_anchors` and to the rate
  estimation, making anchor rates s+(1−s)a and s+(1−s)b — the (1−s) factor cancels exactly out of the
  WLS ratio. Under correct accounting: error **0.0823 / 0.1619** against a 0.03 bar (the pre-fix
  committed run recorded 0.012 / 0.003). **CLOSED_NEGATIVE**, retiring a claim standing since `e1ce286`.
  (ii) **The detector stratum does not price the master key at the ambient rate** — min z 1.75 / 2.78
  against a 3-sigma bar, on all-fire 0.075 / 0.1225. A garbage stratum that merely samples the wild
  sync rate is not a detector; the R7e stratum built to trip at 0.80 clears. Recorded as a Stage-B
  construction constraint, **bar not moved**.
  **Consequence:** the re-panel cannot run — its own protocol requires fixes 1–4 green and Stage A is
  `all_ok=false`. R3's anchored headline is now scoped twice: anchors price a master key only when it
  fires ON the anchors (R7c: clean anchors → `master_key_detected` false, organic error 0.153) AND
  when the stratum trips far above ambient. DS-side results untouched (DS error 0.039 / 0.077 across
  doses; R1/R2/R4/R5/R6 unchanged and green). The misfit statistic ships **reported and not gating** —
  a single-draw ordinal comparison is under-powered and Stage B may not lean on it.
  Verified: `pytest tests -q` **1773 passed / 8 skipped**; `py_compile` clean; `certify.py` untouched
  → no `validate_oath_v0` re-run owed.
  *Next:* prereg R3's replacement claim (correct using the detector fire-rate, or refuse when the
  garbage stratum trips — do not restate flatness); freeze detector-stratum construction strength as
  a Stage-B parameter; then the major set 5–9, 12–15; only then re-panel.

- **Cycle 43 (2026-07-16, autopilot) — scored + reviewed the B2-coupling CONFIRMATION run the chain launched overnight. The frozen rule returned the program's FAVOURABLE verdict; the instrument does not earn it. `VOID_COUPLING__battery_lacks_dose_specificity` (RESULT OATH-HELD 72/0).**
  Orient: chain status `SCORED_RUN_LAUNCHED`, run COMPLETE, GPU free, no contention. The frozen
  aggregate rule returned `COUPLED__erasure_bound_measured_1p5B` — n_admissible=5, n_coupled=5,
  n_decoupled=0, every guard green (clean private-13 0.9382 vs floor 0.75; clean battery aggregate
  1.0 vs floor 0.80; power 5 vs min 3). Under the prereg that is a strict majority and the erasure
  bound becomes a NUMBER. **Refused, and here is why:** the prereg's own "Reported, no bar" control —
  the fixed-rank arm, dose pinned at rank 2 all run — was never asked whether the break rule can tell
  dose from noise. It cannot. Applying the SAME frozen rule: it fires on 35/60 constant-dose control
  checkpoints (rate 0.5833) and on 37/52 accumulate checkpoints **where the read is fully intact**
  (0.7115) — capability "breaks" on three of four checkpoints where nothing has been removed. At the
  matched r\* step the control ALSO registers a break in 3 of 5 seeds and its READ also crosses
  survival in 3 of 5 seeds (0.6811 / 0.6571 / 0.635) — a control accumulating no extra direction
  reproduces the crossing the eraser is credited with. Under a dose-free null at the control's own
  rate, P(5 of 5 coupled) = 0.0675 — the verdict arrives too cheaply to be evidence.
  **Root cause (the lesson):** the battery was disjoint (amendment 1), base-calibrated (amendment 2),
  per-sub-task guarded — all correct, none load-bearing. It is measured through the **same True/False
  margin readout the honesty LoRA perturbs wholesale**, so training degrades every sub-task at once, at
  any dose. Excellent construct disjointness, ZERO dose specificity; only the second was the question.
  A clean base score is necessary and nowhere near sufficient — what was owed and never frozen is a
  NOISE FLOOR (the control's own break rate) as an admissibility gate on the rule itself.
  **Post-hoc honesty:** no specificity bar was pre-committed, so this analysis is post-hoc — but it only
  ever SUBTRACTS a favourable claim using the prereg's own control; the reverse (rescuing a missed bar)
  stays forbidden. Untouched: the 1.5B static / 1.5B adaptive / 3B SURVIVES results (different
  attackers, never gated on this battery). The dose `PARTIAL__coupling_seed_split` **remains the
  program's standing position on coupling** — this run was its confirmation and does not resolve it.
  Reproduced (no bar): seeds 0,1 land on the dose receipt at r\*=8, reads 0.671 / 0.6756, to four
  decimals; the crossing extends to all five seeds and is robust and transient. `pytest tests -q` 1719
  passed / 8 skipped; certify.py untouched → no validate_oath_v0 re-run owed.
  **NEXT (new prereg required):** (a) a pre-committed SPECIFICITY bar — measure the control arm's break
  rate first, battery inadmissible unless it clears it, no coupling verdict issues otherwise; (b) read
  capability through a channel the honesty fine-tune does not move wholesale (generation-scored or
  multiple-choice, not the shared T/F margin); (c) score coupling as a PAIRED accumulate-minus-fixed
  contrast at matched step so the training effect differences out instead of being billed to the attack.
  **The coupling question is OPEN; the erasure bound has no number. The paper stays blocked on the
  coupling side** (B7 3B SURVIVES is unaffected).

- **Cycle 42 (2026-07-15, operator "make this groundbreaking" overnight) — built + red-teamed + base-calibrated the B2-coupling CONFIRMATION instrument and ARMED a hands-free chain to run it when the GPU frees. `INSTRUMENT_HARDENED + CHAIN_ARMED` (no scored verdict yet — a `sentiment_probe_parity` scored run holds the card; the hard rail forbids contending).**
  Built `coupling_confirm.py` (5-seed harness on the `b2_coupling_dose` primitives + a per-checkpoint
  capability-battery audit + multi-seed majority verdict + per-seed crash-safe checkpoint). A 4-lens
  PRE-RESULT adversarial red-team (`coupling_confirm_redteam_review.json`) caught, and I fixed BEFORE
  any GPU hour: **FATAL** — the two frozen clean-guard VOIDs were computed-but-never-enforced (now
  short-circuit before any seed trains, +4 `--dry` guard cases); **MAJOR** — MUL/INEQ are
  category-adjacent to the honesty fact bank so an OR-across-subtasks break could manufacture an
  unsound COUPLED (now measured+reported but NON-gating); +6 MINORs. The GPU smoke then earned its
  keep: base Qwen2.5-1.5B does **SEQ at chance (0.5)** via the T/F readout while ORTH first-letter is
  1.0 → disjoint aggregate 0.75 < the 0.80 clean floor → the guard-enforcement fix correctly **VOIDs
  rather than emit a bogus verdict**. Fixed pre-result (prereg **AMENDMENT 2**, base-only +
  treatment-blind): `capability_battery.py` rewritten around a frozen candidate POOL of 8
  category-disjoint symbolic/lexical sub-tasks; `select_disjoint()` keeps those the CLEAN base clears
  at 0.90 (≥3 survivors else VOID); `--calibrate` freezes the selection as a receipt before the run.
  `--selftest` 132/132 (all 10 sub-tasks' labels recomputed from ground truth); `--dry` 10/10; `pytest
  tests -q` 1719 passed / 8 skipped; certify.py untouched. `coupling_confirm_watcher.py` launched
  (WAITING_FOR_GPU) — **total-VRAM-gated** (per-process VRAM is [N/A] here; a per-process check would
  false-positive "free" and contend), it runs `--calibrate → --smoke (guard check) → the scored 5-seed
  run detached` the moment the card frees, STOPPING at the raw result (no auto-certify/commit — a
  reviewed morning step). Commits `ad5b894`, `985f429`, `429d15b`. **MORNING NEXT:** read
  `papers/calib-poison-general/_confirm_chain_status.json`; if the scored run ran, review the raw
  `coupling_confirm_result.json` verdict, commit the frozen calibration receipt, add the seeds-0/1-vs-dose
  reproduction report, write + certify `RESULT_B2_coupling_confirm_*` OATH-HELD, update this row. Do NOT
  ship a RESULT from an unreviewed raw verdict.

- **Cycle 41 (2026-07-15, autopilot + operator mid-cycle redirect) — two deliverables: (a) FROZE the B2-coupling CONFIRMATION prereg + its disjoint capability battery (the B4 strengthening the dose PARTIAL owed); (b) built a self-certifying ARTWORK at the operator's live direction. No scored run (deferred), no new FINDING/RESULT (nothing to certify). `PREREG_FROZEN + ART_SHIPPED`.**
  Orient found GPU free (207/8188 MiB, no scored run) and cycle 40's coupling result PARTIAL — its
  own named next step is more seeds at the knee + a capability battery replacing the single behavioral
  invariant (arc-wide B4 caveat). (a) Froze `PREREG_B2_coupling_confirm_2026_07_15.md`: seeds
  {0,1,2,3,4}; gating invariant becomes a DISJOINT capability battery (four sub-tasks — MUL / ORTH
  first-letter / INEQ / SEQ weekday-month — in categories OUTSIDE the honesty fact bank the eraser
  trains on); a majority aggregate rule that can RESOLVE the split (strict-majority decoupled → BREAK,
  strict-majority coupled → bound measured, else PARTIAL); every dose threshold inherited unchanged
  (0.70/0.75/0.10) + one NEW per-sub-task guard (0.20) that only makes a break HARDER. Instrument
  shipped + validated: `capability_battery.py` (frozen item bank + measure_battery via the
  byte-identical behavioral_margin primitive + battery_guard/battery_broke; `--selftest` recomputes
  every MUL/INEQ label from arithmetic + exercises the break rule CPU-only, 40/40). **The ~10-15h
  scored run is DEFERRED to a later cycle** — it needs a GPU `--smoke` to review the new instrument
  before an unattended overnight launch on a brand-new battery; freezing the instrument before any
  scored run is the discipline, not a stall. (b) Mid-cycle the operator forwarded a vision (Remote
  control thread): a self-certifying artwork whose subject is its own verifiability. Shipped
  `web/self-certifying.html` — every headline number bound to an embedded receipt (the frozen
  `b7_erasure_3b_result.json` / `b2_coupling_dose_result.json`), re-verified in-browser on load
  (styxx number-grounding OATH performed as art), with a live TAMPER→BROKEN→RESTORE self-falsification.
  Faithful to the locked brand (aubergine+lilac dark / Fathom cream light), zero-dep, CSP-safe,
  theme-aware; colophon honest that it checks numbers-match-receipts (grounding), not the science.
  Verified end-to-end in the browser (OATH·HELD, 15/15 claims ✓, ledger built, tamper flips to BROKEN
  1-ungrounded then restores, no console errors, no horizontal scroll). Published as an Artifact.
  `pytest tests -q` 1719 passed / 8 skipped; `py_compile` clean; certify.py untouched (no
  `validate_oath_v0` re-run owed). Commits `5fa6a20` (prereg+battery), `0524b1b` (artwork), pushed
  `a12f3fa..0524b1b` on `paper/read-neq-write`. **Next = wire the frozen battery into the accumulating
  training loop (`coupling_confirm.py`, own loop built on b2_coupling_dose primitives), GPU `--smoke`
  it, then launch the 5-seed scored run detached + crash-safe (per-seed JSONL checkpoint, cf. cycle 39
  b7_checkpoint), score in the cycle after.** Receipts: `PREREG_B2_coupling_confirm_2026_07_15.md`,
  `capability_battery.py`, `web/self-certifying.html`, Artifact a00159df.

- **Cycle 40 (2026-07-14, operator session "make history") — the two pre-paper GPU gates BOTH scored + certified in one day. B7 `SURVIVES__vs_subspace_erasure_3B` (OATH-HELD 92/0) + B2-coupling `PARTIAL__coupling_seed_split` (OATH-HELD 49/0).**
  B7: the erasure bound holds at 3B (all 4 cells SURVIVES; the 3B read sits ABOVE its 1.5B
  counterpart in every cell, knowledge higher, naive collapse shallower — scale gave the signal
  room, not the eraser). The reviewers' scale objection is answered with receipts, two scales.
  B2-coupling: the accumulating union eraser is the FIRST attacker in the program to shove private-13
  under 0.70 — at a measured dose, accumulated rank r*=8 (step 75), on BOTH seeds. The seeds then
  split on the price: one coupled (knowledge fell to 0.7273, under floor), one decoupled (knowledge
  held 0.8333) — both within ~1 SE (n=66), and the break is TRANSIENT (both arms recover past r* as
  the union dilutes across more directions). PARTIAL forbids an aggregate claim; the knee is real, its
  price unresolved — the honest confirmation prereg (more seeds at rank 6–10, capability battery) is
  the next rung, inheriting these frozen thresholds. No erratum, no paper text from a PARTIAL. Infra
  this session: crash-safe B7 harness (`b7_checkpoint.py`, cycle 39), `card_chain_watcher.py`
  auto-chained coupling the minute B7 freed the card (hands-free). Also opened: **G5 GPAI scorecard**
  (prereg `a059ba9` + population `2368248` frozen BEFORE extraction; extraction fleet blocked on
  model credits — `papers/gpai-scorecard/STATUS_2026_07_14.md`, a resource block recorded as one).
  Operator brief for the counterparty flip (arXiv/§4-excision/main-ff) at `docs/OPERATOR_BRIEF_2026_07_14.md`.
  Receipts: `RESULT_B7_erasure_3b_SURVIVES_2026_07_14.md` (+cert), `b7_erasure_3b_result.json`,
  `RESULT_B2_coupling_dose_PARTIAL_2026_07_14.md` (+cert), `b2_coupling_dose_result.json`,
  `coupling_dose_curve.png`.

- **Cycle 39 (2026-07-14, autopilot) — B7 is the #1 decisive item but its overnight 3B run DIED at cell 3/4 with no crash-safety. Made the harness crash-safe and relaunched. `TOOLING_SHIPPED (b7 crash-safe resume) + B7_RELAUNCHED_INFLIGHT` — no scored verdict this cycle; the binding scale gate is now robustly in flight.**
  Orient found GPU free (201/8188 MiB, no scored run) and B7 half-complete: the 07-13 overnight run's
  clean guard PASSED at 3B (private13 0.9853, knowledge 0.9242) and both seed-0 cells came back SURVIVES
  (s0α1 0.8065@0.9091; s0α4 0.8378@0.9091, ~225 min each) — but `b7_erasure_3b.py` wrote its result JSON
  only at the very END of all 4 cells, so a session/kernel death lost every completed cell (log-only).
  **The blocker was fragility, not science.** Fix = `b7_checkpoint.py` (stdlib-only, self-tested): append
  each completed cell to a JSONL cache the instant it finishes + cache the clean-guard block once + a
  resumed launch skips cached (seed,α) cells and computes the SAME frozen verdict over the union. **Proven
  science-neutral:** the attack (`B2.gold_subspace`/`train_erasure`), audit (`HPC.*`), every guard/bar,
  and the verdict block are byte-identical (import + verdict function unchanged — it operates on the cell
  list irrespective of provenance). CPU end-to-end check drove `main()` through the all-cached resume
  branch (no model load) → reconstructs SURVIVES from cache and flips to PARTIAL when a seed erases.
  `pytest tests -q` 1719 passed / 8 skipped; `py_compile` clean; `certify.py` untouched (no
  `validate_oath_v0` re-run owed). Relaunched B7 fresh + DETACHED (survives session end; crash-safe resume
  is the backstop if it dies again) — confirmed healthy start (frozen split + layer map printed, 3B model
  on GPU, computing the clean guard). **Score NEXT cycle when both seeds land** (VOID_B7__underpowered
  until seed 1). Discipline: this is backlog **B13** ("harden all experiment scripts") discharged on the
  arc's binding run, in the tradition of cycle-37's added instrumentation; no frozen bar moved. Receipts:
  `b7_checkpoint.py`, `b7_erasure_3b.py` (crash-safety note in the docstring), `PREREG_B7_erasure_3b_2026_07_13.md`.

- **Cycle 37 (2026-07-13, autopilot + operator escalation) — B2-adaptive: the chasing eraser. `SURVIVES__vs_adaptive_erasure_1p5B` (RESULT OATH-HELD 100/0). The adaptive re-fit eraser NEVER CONVERGES — chasing made the attack WEAKER, and the read/knowledge coupling hardens into an erasure bound.**
  Single-variable escalation of cycle 36 by import (`b2_adaptive_erasure.py` imports `b2_subspace_erasure`;
  only delta = re-fit the gold subspace on the current model every 50 steps, n_refits=5/cell). Prereg
  frozen+committed+pushed (`a7324de`) BEFORE the run. n_admissible=3 covering both seeds, 0 ERASED,
  3 SURVIVES (private-13 0.7982/0.7668/0.765, knowledge 0.9091/0.7879/0.8182). **The mechanism, from the
  attacker's own objective:** l_erase tails 0.107/0.096/0.1787/0.0737 vs static B2's converged
  0.0085/0.0112/0.0073/0.008 — the moving target never converges (the E-series NO_BITE failure reappearing
  inside the removal class), and the private-13 read came back HIGHER than static in every cell. The s0 α=4.0
  cell failed the bite gate carrying the run's STRONGEST read (0.8654) — exclusion is anti-SURVIVES, disclosed.
  At this budget the attacker's fork: static → relocates (B2), chase → outruns (this), drop knowledge →
  inadmissible. **First run with a pre-result adversarial review receipt** (five lenses, 32 findings, 0
  fatal/major, 4 wording-level honored: `b2_adaptive_redteam_review.json`). Commit `83aa060`; prereg `a7324de`.
  **Next in order: 3B feasibility smoke → B7 (now the binding flank), accumulating union-of-past-subspaces
  eraser (new frozen prereg), B6 full-FT; paper only after.** Receipts:
  `PREREG_B2_adaptive_erasure_2026_07_13.md`, `b2_adaptive_erasure.py`, `b2_adaptive_erasure_result.json`,
  `RESULT_B2_adaptive_erasure_SURVIVES_2026_07_13.md` (+cert), `b2_adaptive_redteam_review.json`.

- **Cycle 36 (2026-07-12, operator-driven) — B2 fired: the first REMOVAL-class attacker. `SURVIVES__vs_subspace_erasure_1p5B` (RESULT OATH-HELD 74/0). The erasure verifiably succeeded and the read survived anyway — the signal RELOCATED rather than died.**
  RMU-style projection-to-zero of the gold subspace (per-layer DoM + logistic, rank 2 × six scan layers,
  attacker granted worst-case knowledge of the probe family) + the arc's knowledge-replay CE (λ=1.0),
  LoRA r=16, 300 steps, seeds {0,1} × α {1.0,4.0}. Prereg frozen+committed (`285245b`) BEFORE the run;
  audit surface byte-identical to the parity arc by import. **All 4 cells admissible, all 4 SURVIVES,
  0 ERASED:** the attacker's own loss CONVERGED (l_erase tail 0.0073–0.0112 — the subspace was really
  zeroed; naive DoM collapsed to chance 0.4636–0.4848, the deepest collapse in the program) yet
  **private-13 read 0.7161–0.7871 with knowledge held 0.7576–0.803**. Under a knowledge-replay
  constraint the model rebuilds honesty-correlated geometry OUTSIDE the erased subspace → first
  evidence in the program that the honesty read and the retained knowledge are COUPLED at this budget:
  read≠write survives removal, not just redirection. Parity-arc seed structure persists under erasure
  (s0 parity_gap +0.106/+0.071 — clean split earns more vs removal; s1 reverses −0.0359/−0.0498).
  Scope: 1.5B, LoRA r16, 300 steps, STATIC rank-2 subspace (adaptive re-fit erasure NOT tested — the
  live escalation), behavioral knowledge invariant (B4 caveat). Commit `88efac7`; prereg `285245b`.
  **Next = adaptive erasure (re-fit the subspace every K steps as the signal relocates — erasure BITES,
  so the adaptive lane is live again in the removal class), then 3B feasibility smoke → B7, then B6
  (full-FT erasure).** Receipts: `PREREG_B2_subspace_erasure_2026_07_12.md`, `b2_subspace_erasure.py`,
  `b2_subspace_erasure_result.json`, `RESULT_B2_subspace_erasure_SURVIVES_2026_07_12.md` (+cert).

- **Autopilot cycle 35 (2026-07-12) — attribution confirmation part (ii): the Llama-3.2-1B cross-family run. `VOID_HPCF__clean_guard_failed` (RESULT OATH-HELD 73/0). The base model fails the frozen non-degeneracy knowledge floor, so the cross-family attribution is ill-posed on this model — VOID, not a smuggled verdict; the erratum path stays CLOSED.**
  Ran the byte-identical shared parity code path (`honesty_parity_confirm_llama.py` **imports**
  `honesty_parity_control.py`; only the three E1 Llama family constants change — scan {6,8,10,12,14},
  deploy 10). Prereg frozen+committed (`30d943e`) BEFORE the retrain with a **pre-committed
  guard-fail → VOID** branch (E1's Llama clean guard was borderline). **Result: `clean_eval_knowledge`
  0.7727 < the frozen 0.80 knowledge floor** (clean private-13 AUROC 0.8793 reproduced E1's Llama
  number to 4 decimals; split disjoint) → guard fails → the run VOIDs before any attribution, exactly
  as E1 found (`guard_pass=false`, clean knowledge 0.7879). **Descriptive, NON-GATING under VOID:**
  3 of 6 cells cleared the 0.75 knowledge-admissibility floor; all 3 admissible cells are FAILS
  (parity_gap ≤ 0.02: −0.0046, −0.0083, −0.0608; median −0.0083, mean −0.0246), 0 STANDS — every
  admissible cell has the naive-MATCHED-13 poisoned-fit auditor ≥ the private clean-fit one, i.e. the
  same capacity-dominant direction as Qwen (part i). But a bar is a bar: no verdict rests on an
  ill-posed base, so this is corroborating color only, not a cross-family FAILS. **Sequencing: VOID is
  not FAILS → the operator-gated erratum path cannot open here, and (family untestable) it cannot close
  the attribution as two-family either. The attribution stays as part (i) left it: capacity-dominated
  with a small seed-dependent λ-insensitive privacy residual, single-family on Qwen2.5-1.5B.** Commit
  `05267e5`; prereg+harness `30d943e`. **Next = cross-family on a ≥3B model that clears the 0.80 floor**
  (Qwen2.5-3B / Llama-3.2-3B, new frozen prereg), then **B2**. Receipts:
  `PREREG_honesty_parity_confirm_llama_2026_07_12.md`, `honesty_parity_confirm_llama.py`,
  `honesty_parity_confirm_llama_result.json`, `RESULT_honesty_parity_confirm_llama_2026_07_12.md` (+cert).

- **Autopilot cycle 34 (2026-07-11) — attribution confirmation part (i): does cycle-33's privacy residual survive ≥3 seeds + a λ sweep? `PARTIAL_CONSOLIDATED__residual_real_but_not_robust` (RESULT OATH-HELD 82/0). The residual is REAL but seed-dependent and small; the erratum path stays CLOSED.**
  Qwen2.5-1.5B, seeds {0,1,2} × λ {1.0, 3.0} = 6 cells, 300 steps, via `honesty_parity_confirm.py` which
  **imports** `honesty_parity_control.py` (byte-identical shared parity code path — the 2 cells shared with
  cycle 33 reproduced all three auditor reads to `delta_vs_cy33` = **0.0**). Prereg frozen+committed (`79fba7a`)
  BEFORE the retrain, two-thirds-majority aggregate, STANDS written as reachable as FAILS. **N_admissible=6,
  FAILS-cells=2, STANDS-cells=0, majority=4, median baseline_gap 0.2793, median parity_gap 0.0516, mean 0.0091.**
  **Structure, not noise:** seeds 0 & 2 hold a small **positive** privacy residual at BOTH λ (parity_gap
  0.0553/0.0479 s0; 0.0645/0.0553 s2); seed 1 **reverses** it at both (−0.082/−0.0866, poisoned-fit parity
  auditor beats clean-fit private). It is a **seed effect, not a λ effect** — within every seed
  |Δ parity_gap| ≤ 0.0092 across λ 1→3, so stronger knowledge-replay doesn't change the attribution. The parity
  swap removes ~⅘ of the honesty gap on EVERY cell (dominantly probe capacity), leaving a small (~0.05 AUROC,
  ~0.8 SE at EVAL n=66) genuine privacy residual on two of three seeds, reversed on the third — too small for
  STANDS, not consistently ≤0.02 for FAILS. **Three probe-parity runs now agree** (sentiment PARTIAL, honesty-λ1
  PARTIAL cycle 33, honesty-3seed PARTIAL_CONSOLIDATED cycle 34): the flagship read≠write private>naive gap is
  **capacity-dominated with a small seed-dependent privacy residual**. **Sequencing: FAILS did NOT consolidate
  (2 of 6, needed 4) → the operator-gated erratum path stays CLOSED**; the correct update is a scope note
  ("calibrate on a private split" is minor-but-real, ~⅕ of the gap), NOT a retraction. Commit `e9ce645`;
  prereg+harness `79fba7a`. **Next = part (ii) Llama-3.2-1B cross-family** (new frozen prereg; must pre-commit a
  borderline-guard → VOID branch), then **B2**. Receipts: `PREREG_honesty_parity_confirm_2026_07_11.md`,
  `honesty_parity_confirm.py`, `honesty_parity_confirm_result.json`,
  `RESULT_honesty_parity_confirm_2026_07_11.md` (+cert).

- **Autopilot cycle 33 (2026-07-11) — honesty-side probe-parity check (the sentiment control's Next #1: the single most load-bearing unrun experiment). `PRIVACY_PARTIAL__between_bars` (RESULT OATH-HELD 68/0). The flagship's private>naive recovery is MOSTLY probe capacity, not privacy — on the flagship's own construct.**
  GPU free (463/8188 MiB, no scored run) → ran the identical E1-regime honesty attack (`attack_sweep.train_attack`;
  Qwen2.5-1.5B, LoRA r=16, deploy-18 scrub + knowledge-replay, 300 steps) on E1's three-way split, two cells
  (seeds {0,1}, λ=1.0), three auditors via one shared 13-family code path. Prereg frozen+committed (`a617eca`)
  BEFORE the retrain, with a **pre-committed FAILS re-attribution language + a binding sequencing rail**.
  **Reproduction exact:** private-13 matched E1's canonical Qwen λ=1 reads to 4 decimals on BOTH cells
  (**0.8378** s0 / **0.7217** s1, `delta_vs_canonical` = 0.0 each); naive-DoM-6 reproduced the collapse
  (0.5106 / 0.4876 vs `attack_sweep` 0.5066 / 0.4725); frozen-18 bit 0.8562 → 0.5502 / 0.5899; knowledge held
  0.8182 / 0.803. **The swap:** upgrade the naive auditor from 6-feature DoM to the SAME 13-probe family
  (per-layer DoM + per-layer logistic + whole-stack) fit on a frozen n=53=|CALIB| subsample of the attacker's
  OWN poisoned ATTACK split (probe family AND fit-size matched; only fit-split differs). It recovered
  **0.7825 (s0) / 0.8037 (s1)** of the 0.8378 / 0.7217 private read — `baseline_gap` 0.3272 / 0.2341 →
  `parity_gap` **0.0553 / −0.082** (parity removed ~83% of the seed-0 gap and **>100%** of seed-1: the
  poisoned-fit parity auditor BEAT the clean-fit private one). Verdict PARTIAL by the frozen conjunction
  (discordant cells): STANDS refused (`parity_gap` ≪ ½·baseline both cells); FAILS refused ONLY because seed 0
  kept a ~0.035 privacy residual (private 0.8378 > the 0.8178 matched-13 bar), while seed 1 satisfies FAILS
  outright. **Cross-construct convergence:** same string as the sentiment margin/parity control (cycle-32 era,
  07-10), from opposite starting magnitudes (sentiment small gap ~0.11–0.13, ~⅔ removed; honesty large gap
  ~0.23–0.33, ~83%+ removed) → the flagship's "privacy" attribution is **probe-capacity-dominated on BOTH
  tested constructs**; the clean fit-split buys at most a small one-of-two-seeds residual. Per the frozen
  sequencing rail **NO erratum ships from this run alone** (FAILS did not fire; PARTIAL did) — but demotion
  pressure now sits on both constructs. Commit `7d82ff4`; prereg+harness `a617eca`. **Next = the confirmation
  run** (new frozen prereg, ≥3 seeds + λ sweep {1.0, 3.0} + Llama-3.2-1B via this same shared code path);
  its FAILS branch inherits this prereg's frozen re-attribution language and opens the operator-gated erratum
  path (Fathom v26/v28). Only after the attribution is correctly named on ≥3 seeds does **B2** make sense.
  Receipts: `PREREG_honesty_parity_control_2026_07_11.md`, `honesty_parity_control.py`,
  `honesty_parity_control_result.json`, `RESULT_honesty_parity_control_2026_07_11.md` (+cert).

- **Autopilot cycle 32 (2026-07-10) — GPU held by in-flight scored run (calib-poison sentiment attack pid 5480, 7.2/8.2GB) → CPU-only receipt-binding cycle (standing priority #2). `BOUND` (1 doc OATH-HELD).**
  Bound `papers/concept-dynamics/FINDINGS_rhythm_substrate_2026_06_03.md` against its OWN 3 experiment
  receipts (`llm_rhythm_result.json`, `ssm_contrast_raw.json`, `ssm_contrast_result.json`) — **OATH-HELD
  verified=21, UNGROUNDED=0, CONTRADICTED=0** (abstained 28). Honest bar held: single-experiment folder
  ≤4 receipts, no 200-receipt sieve; `verifier_sha` matches shipped `certify.py` (UNCHANGED → no
  `validate_oath_v0` re-run owed). Certify/oath tests **17 passed** CPU-forced (`CUDA_VISIBLE_DEVICES=""`,
  zero contention with the scored GPU run). **Measured ceiling of standing priority #2 (the yield):** of
  **201** UNBOUND FINDING/RESULT docs (excluding the in-flight `calib-poison-general/` dir), only **15**
  sit in honest small (≤4-receipt) folders and only **THIS one** certifies clean today. The NEAR misses
  (consensus-truth-engine UNG=1, triviaqa/kbc/holdout-gate UNG=2, convergence/capstone/curve UNG=3, all
  CONTRADICTED=0) fail purely on **certifier-precision config-token false positives** — bare `0` in "> 0"/
  "≈0", model-size ranges like "2–3B", prose counts "4 of 6" — NOT doc errors; not doctoring prose to pass.
  The big-folder docs (grounded-honesty-axis n≈70, frequency-resonance n≈14) still need their own named
  receipts written first or the sieve caveat. Repair-by-summary-receipt stays blocked on float→CELL binding
  (cycle 26). Two concrete forks fall out: **(a)** a certifier-PRECISION prereg for the unambiguous
  false-positive classes (comparison-`0` / config-range / "N of M" prose counts — the safe extension of the
  arXiv-ID-skip idea cycle 18 named), re-gated by `validate_oath_v0.py` with bars frozen; **(b)** doc-named-
  receipt binding for the big folders (parse each doc's cited JSONs, ground against exactly those). Commit
  `3b388a9`. Receipts: `papers/concept-dynamics/FINDINGS_rhythm_substrate_2026_06_03.certificate.json`.

- **Autopilot cycle 31 (2026-07-08) — read≠write E3‴: does dropping the frozen anchor let the adaptive attacker finally BITE? `VOID_NO_BITE` — the pure-moving scheme STILL did not bite (RESULT OATH-HELD 54/9/0).**
  Names the E3″ VOID_NO_BITE burial and applies the mechanistic fix E3″ handed forward: the adaptive branch scrubs
  ONLY the moving whole-stack re-fit-every-10 MIMIC consensus direction — the frozen-deploy anchor term is removed
  (the entire E3‴ delta; naive control unchanged). Prereg frozen+committed (`814947f`) before the retrain, bite-gated,
  able to return DEFENSE_BROKEN. **It bit even less than E3″.** Decisive Qwen2.5-1.5B: adaptive left the frozen-deployed
  read at **0.624/0.664** vs naive **0.400/0.445** and the mimic-refit read at **0.749/0.682** vs naive **0.432/0.467**
  — MORE legible on both; `delta_frozen`=**−0.222**, `delta_mimic`=**−0.266**, `attack_bit`=false. The pre-committed
  bite gate **blocked a false STANDS** (adaptive clean-calib came in HIGH at 0.806/0.728 precisely because the attack
  barely touched the signal). Clean guards passed both families (Qwen calib 0.912/know 0.800, Llama 0.821/0.817); naive
  control reproduced E1/E3 (clean-calib 0.786/0.775 @ knowledge 0.833/0.817). **THREE independent adaptive escalations**
  (E3′ refit-25 deploy, E3″ two-term whole-stack, E3‴ pure-moving whole-stack) now all fail to out-attack the naive
  static scrub → within this difference-of-means objective family at LoRA r16/300-step/1.5B-1B, **the naive static
  attacker is empirically the STRONGEST threat and adaptive complexity is self-defeating.** E3″'s hypothesis
  (frozen/moving gradient conflict was the sole cause) is FALSIFIED: a moving DoM reference re-fit every 10 steps never
  converges at this budget, so even the single-objective pure-moving scheme is toothless. **Adaptive robustness stays
  formally OPEN; v27's STANDS-vs-naive-and-strong demotion unchanged and correct** — E3‴ neither re-earns nor breaks
  the adaptive claim. Receipts: `PREREG_E3TPRIME_pure_moving_adaptive_2026_07_08.md`,
  `RESULT_E3TPRIME_pure_moving_2026_07_08.md` (+cert), `e3tprime_result.json`, `e3tprime_pure_moving.py`,
  `_e3tprime_run.log`. **Next (recommended) = STOP the adaptive lane** (3 consecutive NO_BITEs) and pivot to **B2**
  (RMU/gradient-routing unlearning of the gold subspace + knowledge-replay — the real read≠write test, Tier-1 QUEUED)
  or standing-priority #2 (45 UNBOUND finding docs / 13 cycle-18 provenance). If the adaptive lane IS pursued: E3⁗ with
  a reference held STATIONARY within a bite (fit once on a large MIMIC pool, hold many steps) or ≥800 steps — a NEW
  prereg naming this burial.

- **Autopilot cycle 25 (2026-07-04, operator "go deeper → break the ceiling") — OATH v0.4 DECIMAL+RANGE-guarded trigger-recall. `SHIPPED` — ALL FIVE BARS PASS (first `styxx.certify` upgrade since v0.3; RESULT OATH-HELD 17/0).**
  Add `decimals > 0` to the [−1,1] correlation-register guard — correlations carry a fractional part; ordinals /
  counts / API-caps / whole-percents never do. Prereg frozen+committed (`b6f5808`) before code, with a **frozen G3
  artifact definition** (measurement-domain vs not) to remove post-hoc judgment. **Tamper-catch on the 13-doc
  cycle-18 battery rose from the v0.3 baseline 58 → 119 of 269** (catch rate 0.216 → 0.442, +61 abstain-degrades
  recovered) with false-verify held at 26 (no regression) and **zero certifier artifacts** — the `decimals>0` clause
  removed the cycle-24 residual (`drift, stage 1`, the integer 1 admitted at the correlation boundary 1.0) by
  construction. G1 D1=16, G2 D2=0, G3 artifacts=0, G4 119≥116, G5 1675-passed. The 3 remaining clean UNGROUNDED are
  all REAL provenance gaps (derived RSA/R² bounds; a bulk-only agreement value) — the tool correctly turning on the
  older docs. **This is the payoff of a 4-negative arc** (cycle 22 float → 23 blunt → 24 range → 25 decimal, each
  naming the last). Ship surface: `styxx/certify.py` (+`_TRIGGERS_CORR` + one guarded `bound` line), 3 corpus certs
  regenerated (D2=0 held), `CHANGELOG [Unreleased]`. Receipts: `RESULT_oath_v04_recall_decimalguard_2026_07_04.md`
  (+cert), `cycle25_decimalguard_{battery,g3}_result.json`, `cycle25_decimalguard_probe.py`.
  **Owed next = re-certify the 13 cycle-18 docs under the shipped verifier + repair the surfaced provenance gaps**
  (persist bulk/derived correlations as summary receipts, or scope claims) — now the concrete content of standing
  priority #2.

- **Autopilot cycle 24 (2026-07-04, operator "go deeper") — OATH v0.4 RANGE-GUARDED trigger-recall (names the cycle-23 burial). `CLOSED_NEGATIVE` — bar G3 missed by ONE, REVERTED (RESULT OATH-HELD 28/0).**
  Fix attempted: the correlation register obligates a number only when value ∈ [−1,1] (all 6 cycle-23 artifacts
  were out of range). Prereg frozen+committed (`f539339`) before code. **The guard did most of its job** — clean
  UNGROUNDED collapsed **35 → 4** (2 REAL derived-RSA-bounds, 1 REAL bulk-only, **1 ARTIFACT**), battery caught
  **128 → 119** of 269 (recall survives, only −9), false-verify 26 unchanged; G1/G2/G4(119≥116)/G5 all PASS. **But
  G3 = 0 and one artifact survived:** `geometry_integrity` L46 `(drift, stage 1)` — the ordinal `1` is obligated by
  "drift" and the guard **admits it because 1.0 is a legal correlation** (the boundary); `stage 2` was spared
  (2 ∉ range), as were the 4 other cycle-23 artifacts. **Yield:** a value-range guard is necessary but not
  sufficient — correlations are written WITH decimals (0.264/0.98/0.735), the false positives are bare integers;
  the clean separator is **decimals > 0**, not range alone. **Next = cycle 25** (add `decimals > 0` to the guard —
  removes the ordinal artifact by construction, keeps every decimal correlation; ships if G3=0 ∧ G4≥116). Receipts:
  `RESULT_oath_v04_recall_rangeguard_2026_07_04.md` (+cert), `cycle24_rangeguard_battery_result.json`,
  `cycle24_rangeguard_g3_result.json`.

- **Autopilot cycle 23 (2026-07-04) — OATH v0.4 trigger-vocabulary RECALL extension (cycle-22-owed sibling of float binding; standing priority #5). `CLOSED_NEGATIVE` — bar G3 missed, change REVERTED (RESULT OATH-HELD 36/0).**
  Prereg frozen+committed BEFORE code (`PREREG_oath_v04_trigger_recall_2026_07_04.md`): widen `_TRIGGERS` with the
  correlation/similarity register (rsa/rdm/spearman/correlation/rho/consistency/reliability/ceiling/agreement/
  convergence/drift/entropy/similarity/variance) to convert the cycle-19 battery's **182/269 abstain-degrade** bucket
  into caught UNGROUNDED. **The extension works on its own axis** — battery **caught 58 → 128 of 269** (catch rate
  0.216 → 0.476, +70 abstain-degrades recovered), with **no false-verify regression** (26 unchanged — the feared dense-
  table abstain→false-verify conversion did not net occur); G1 D1=16 PASS, G2 D2=0 PASS, G4 catch≥116 → 128 PASS, G5
  suite 1675-passed PASS. **But G3 (honesty gate) FIRED:** re-certifying the 13 clean cycle-18 docs produced **35 clean
  UNGROUNDED** (baseline 0), of which **6 are certifier ARTIFACTS** — a register word obligating a non-measurement
  number (unambiguous: `detection_locus_gpt` L64 API-cap `20` obligated by "entropy"; `geometry_integrity` L46 stage
  ordinals `1`/`2` obligated by "drift"). One artifact is a kill; there are six. **Measured boundary (the yield):**
  recall and precision are COUPLED for this register — the same words that name a measured correlation
  (entropy/drift/variance/ceiling) also appear as spec constants / ordinals / "2D"; a blunt vocabulary widening buys
  +70 catches at 6 false accusations and the oath cannot ship false accusations. **29 of 35 are REAL** doc↔receipt gaps
  (grid-cell correlations never persisted as summary receipts — the tool correctly turning on older docs).
  **Revert proven:** `certify.py` byte-identical to HEAD; 3 corpus certs + `_oath_mutants` fixtures restored;
  reproducible from the committed (reverted) tree via `papers/autopilot/cycle23_recall_probe.py` (monkeypatches the
  exact one-line change in memory → 128/26/112 + 35[28 absent/1 bulk/6 artifact]). Receipts:
  `RESULT_oath_v04_trigger_recall_2026_07_04.md` (+ certificate), `cycle23_recall_battery_result.json`,
  `cycle23_g3_handcheck_result.json`. **Next = RANGE-GUARDED recall** (fire the register only when the adjacent number
  is in ~[−1,1] via the existing RANGE-SANITY `unit_kw` machinery — spares API caps / ordinals / "2D") — NEW prereg
  naming this negative, re-gate G3 = 0 artifacts.

- **Autopilot cycle 22 (2026-07-03) — OATH v0.4 float claim→field binding, last-two-segment design (standing priority #5). `CLOSED_NEGATIVE` — bar B3 missed, change REVERTED (note OATH-HELD 38/0).**
  Prereg frozen+committed BEFORE code (`PREREG_oath_v04_float_binding_2026_07_03.md`): floats VERIFIED only if a
  value-matching leaf's last-two path segments share claim-line vocabulary; binding failure ⇒ loud ABSTAIN (never
  UNGROUNDED). Bars: B1 D1≥16 → **17 PASS** (v0.4 *improved* catch); B2 D2=0 → PASS; **B3 battery FALSE-VERIFY ≤13 →
  20 of 247 FAIL — kill**; B4 all 13 docs UNGROUNDED=0 → PASS; B5 suite 1675-passed → PASS. One missed bar ⇒ reverted;
  `styxx/certify.py` byte-identical to shipped v0.3 (validator re-run under revert = zero git diff). **Measured
  boundary (the yield):** field-level binding removes cross-table coincidences (FALSE-VERIFY 26→20, rate 0.097→0.081,
  catch unhurt) but the residual 20/20 are **same-table SIBLINGS** — a corrupted row value matching another row of the
  same field family (plus rounding-tolerance neighbors); field vocabulary cannot separate row k=2 from k=4 by
  construction. Next attempt = **claim→CELL binding** (row-key aware; single-digit row labels + list indices are
  invisible to current binding vocab) — a NEW prereg that must name this negative. **Bonus:** cycle-19's in-memory
  battery is now a committed script (`papers/autopilot/mutant_battery.py`) that reproduces the v0.3 baseline EXACTLY
  (269/58/26/182/3) — the reproducibility gap is closed. Receipts: `cycle22_v04_battery_result.json`,
  `cycle22_v04_validation_result.json`, `cycle22_v03_baseline_battery_result.json`,
  `RESULT_oath_v04_float_binding_2026_07_03.md`.

- **Autopilot cycle 21 (2026-07-03) — discharge the §10 README truth-in-advertising ticket opened by cycle 20. `DISCHARGED (no correction needed)` (OATH-HELD 3/0).**
  Exhaustive repo audit for any live claim that circuit-attribution depth predicts truth/correctness/hallucination:
  `README.md`, `web/`, `docs/**`, `papers/**` (non depth-truth), and the adjacent live depth findings. **None found.**
  The README's hallucination numbers belong to the text-heuristic `@trust`/cognometry instrument (never calls
  `get_mean_depth`); `docs/gate.md`'s only near-hit is the refuse-check class predictor. The phrase "measure thought,
  not words" exists ONLY as a hypothesis label inside `PREREG_v2.md` — never shipped as a result. The closest live
  depth findings are already honest and *consistent* with the negative: `grounded-honesty-axis/FINDING_depth_steering_causal`
  headlines the construction↔retrieval axis **"correctness-INERT"**; `FINDING_depth_grounding_whitebox` scopes depth as
  a grounding substrate. The `get_mean_depth` origin (d=0.82 recall-vs-reasoning) lives in the separate research git,
  not here, and was pending — never advertised in styxx. **Conclusion:** prereg-before-claim discipline meant the
  falsified claim was never made in public copy → no retraction to ship. Note+cert: `papers/depth-truth/TICKET_readme_truth_in_advertising_2026_07_03.md`.
  **Watch-item (operator-gated, outside repo):** IF the external ICML attribution-depth manuscript implies depth predicts
  answer *correctness* (vs separating recall/reasoning), it needs the cycle-20 caveat (AUROC 0.5468, CI straddles chance) —
  not autopilot's to edit. No styxx-repo code changed this cycle (markdown + certificate only).

- **Autopilot cycle 20 (2026-07-03) — the keystone verdict: does depth predict truth? `CLOSED_NEGATIVE_NO_TRUTH_SIGNAL` (OATH-HELD 30/0).**
  The 633-item main run (250 ID / 133 OOD-1 / 250 OOD-2) completed 09:19; scored through the frozen PREREG_v2 §2 tests via a
  new no-free-parameters driver (`harness/run_analysis.py`: §5 complete-case → §2 h1/h2_full/h3_ood, deterministic seed 7,
  re-run byte-identical). **All three hypotheses NULL.** H1 AUROC(depth→correct)=**0.5468** CI[0.4738,0.6183] (straddles chance);
  H2 ΔAUC(SE+depth vs SE)=**0.0026** CI[-0.0044,0.0188] LRT p=1.0 DeLong 0.708, LP_mean/LP_norm concur Holm p=1.0 (adds NOTHING
  over confidence); H3 ΔAUC_ood=**-0.0517** CI[-0.1069,-0.0116] — **anti-signal** (DeLong 0.034), depth HURTS OOD. **Mechanism:**
  first-content-token attribution depth is near-constant (ID std **0.0558**, OOD-1 std 0.0449) so it cannot sort correct from
  wrong — THE figure shows green/red fully superimposed on the depth axis, only SE separates. The v1 narrow-depth signature
  SURVIVED the v2 plumbing fix (content tokens, clean extraction, KG0/KG1 passed at pilot) ⇒ it is a property of the metric on
  single-token answer heads, not a formatting artifact. Per §8 KG2, H1-null did not block H2/H3 (both ran, both failed). OOD-2
  TruthfulQA **ATTEMPTED/PENDING KG3**: 242/250 mechanically `grade_ambiguous`, human audit absent → no TruthfulQA claim.
  Full suite **1675 passed, 8 skipped** (companion test passes now GPU is free); `certify.py` UNCHANGED (no `validate_oath_v0`
  re-run owed). Commit `f054f36` on `keystone-depth-truth`. **OWED (next cycles):** (a) ~~README truth-in-advertising
  ticket (§10)~~ **DISCHARGED cycle 21** — repo audit found no live depth→truth overclaim; discipline meant the claim
  never shipped (external ICML paper is the only watch-item, operator-gated).
  (b) KG3 human audit — flobi grades `results/human_audit_sample.jsonl` (24 rows) to decide if the TruthfulQA arm is reportable.
  (c) any richer-aggregation / larger-model follow-up is a NEW prereg, not a rescue of this frozen negative.

- **Autopilot cycle 18 (2026-07-02) — bind the UNBOUND finding backlog (standing priority #2), receipt-honest. `PARTIAL-BOUND`.**
  GPU held by in-flight scored runs (rung2 cross-family write PID 2604 + depth-truth autofire waiting behind it,
  VRAM 7690/8188) → non-GPU cycle. Swept all 203 uncertified FINDING/RESULT docs (excluding the in-flight
  `disjoint-worlds/` + `depth-truth/` dirs). A naive folder-scoped pass reported 109 "OATH-HELD" — but 24 of those
  ground against **217 unrelated grounded-honesty-axis receipts** (and 9 vs 23 frequency-resonance): grounding a
  number against 200+ receipts drives UNGROUNDED→0 by coincidence, **a sieve, not an attestation**. Held to the
  honest bar (doc's own NAMED receipts, else a single-experiment folder ≤4 receipts) the certifiable set is **13**,
  all written OATH-HELD (UNGROUNDED=0), each recording exact receipt SHAs and independently re-runnable. `styxx.certify`
  UNCHANGED (no mutant-battery re-run owed). Test suite green 1661-passed CPU-only; the one faulting test
  (`test_companion_reports_honestly`) segfaults only because it loads a transformer into VRAM the scored run holds
  — GPU contention, not a regression. **OWED (next cycles):** (a) the big-folder docs (grounded-honesty-axis n=217,
  frequency-resonance/introspection-gate n≈23-31) need in-doc receipt **citations written first**, then bind — that
  is the real content of standing priority #2, not a certifier pass; (b) 139 blocked docs need per-doc UNGROUNDED
  triage — the dominant false-positive class is **config tokens** (steering α=0.75, arXiv IDs like 2505.27958) sitting
  in results-table label columns inheriting the table's AUROC trigger; a certifier-precision prereg (arXiv-ID skip is
  the safe unambiguous class) could recover many, gated by re-running `validate_oath_v0.py`.
  **DILIGENCE ADDENDUM (2026-07-03, operator-directed, cycle 19) — `TAMPER-EVIDENCE-WEAK`.** Mutant battery
  (269 single-digit mutations of every VERIFIED token, seed 1, `validate_oath_v0` scheme) on the 13 certs:
  catch **58/269 (0.216)** vs the D1 analogue 0.80; **26 FALSE-VERIFY (0.097)** — corrupted values re-verify
  against *neighboring* receipt leaves (the disclosed v0 float claim→field gap, now with evidence); **182
  abstain-degrade (0.677)** — the older ρ/RSA/alignment register never binds, so corruption falls to ABSTAIN
  and the verdict silently stays HELD. **Stands:** the 13 certs as ledgers (every number matches at recorded
  SHAs). **Does not stand:** reading HELD on these docs as tamper-evidence. Receipt:
  `papers/autopilot/cycle18_mutant_battery_result.json`; note (itself OATH-HELD 27/0):
  `papers/autopilot/DILIGENCE_cycle18_mutant_battery_2026_07_03.md`. Sharpens OWED (b) into two concrete
  preregs: **v0.4 float claim→field binding** (priority #5, evidence attached) and **trigger-vocabulary
  recall extension** for the older register — both gated by `validate_oath_v0.py`, bars never move.

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

- **Autopilot cycle 2 (2026-06-11, operator "go harder") — is it a BASIS or one valence axis?
  `PARTIAL-STRUCTURED`.** Adversarial self-falsification of cycle 1's "basis" headline. Common-layer
  (gemma L12) 3×3 cross-readout matrix over truth / refusal / a valence-sentiment control + cosines +
  valence-orthogonalization, replicated through one shared label-free map into Llama-3.2-3B + Qwen-3B.
  **Both retraction gates FAILED to fire (good):** truth·refusal cosine **−0.2132** (near-orthogonal, not
  collapsed); orthogonalizing valence out leaves truth **0.80** / refusal **1.0** (not sentiment).
  **But BASIS-INDEPENDENT also failed:** truth↔refusal off-diagonal discriminability **0.8929 / 0.84**
  (≫0.65 ceiling), replicated mapped (Llama 0.875/0.90, Qwen 0.8929/0.76). The axes are DISTINCT and
  valence-irreducible but ENTANGLED in readout — a correlated frame, not an orthonormal basis. Cycle 1's
  "basis" QUALIFIED (not retracted; banner added to its finding, re-certified 42/0). Caveat: off-diagonal
  = discriminability at n_test=15 → inflated floor; a permutation-nulled off-diagonal is owed. Prereg
  `662b6ce`; `FINDING_axis_independence_2026_06_11.md` (OATH-HELD 29/0); receipt `axis_independence_result.json`.
  Spawned **B28**.

- **Autopilot cycle 3 (2026-06-11, operator "keep going") — is the entanglement REAL, ARTIFACT, or
  WHITENING-removable? `WHITENING-RESOLVES` (B28 DONE).** The decisive resolution. Correct nulls
  (K=1000 label-permutation + 1000 random-direction) + ZCA whitening + Gram-Schmidt, larger n.
  **Raw gemma: the cross-talk is REAL and SPECIFIC** — truth→refusal obs **0.9778** beats perm-p95
  0.7278 AND rand-dir-p95 0.8614 (p=0.001); refusal→truth obs **0.9013** beats both (p=0.001). So cycle 2
  was NOT imagining it. **But ZCA-whitening kills it entirely:** off-diagonals **0.9778/0.9013 → 0.55/0.5461**
  (chance) while diagonals stay **0.9737/1.0** and the directions become **exactly orthogonal** (cos
  −0.2756 → −0.0). Gram-Schmidt corroborates (refusal⊥truth still reads refusal 1.0, reads truth 0.5132).
  **The clean orthonormal basis EXISTS under a Mahalanobis readout** — the cycle-2 entanglement was a pure
  COVARIANCE artifact of raw dot-product. Cycles 1+2 UPGRADED (banners + re-cert, both still HELD):
  the conscience IS a basis of independent value axes, read whitened. Honest scope: whitening run in
  SOURCE only; mapped cross-model cross-talk is dominated by the map's broad transport (random dirs hit
  0.95 floor) → truth→refusal mapped not-specific, refusal→truth specific (p=0.005) — whitened mapped
  readout owed. Prereg `5a510a5`; `FINDING_entanglement_resolution_2026_06_11.md` (OATH-HELD 28/0);
  receipt `entanglement_resolution_result.json`. Spawned **B29**.

- **Autopilot cycle 4 (2026-06-11, operator "get creative and innovative") — CONSCIENCE COORDINATES: is
  the whitened basis a value coordinate system that locates dangerous misinformation? `HARM-AXIS-NULL`.**
  The creative leap: treat the cycle-3 orthonormal basis as a COORDINATE SYSTEM; project a 2×2 factorial
  of single sentences (T{true,false} × H{danger-topic,safe-topic}, n=12/cell, NEW content the axes were
  never fit on) onto the whitened {truth, refusal} basis. **The truth coordinate is a genuine PORTABLE
  coordinate** — recovers true/false on the new statements at **0.8524** gemma and transfers BETTER
  through the map (Llama **0.9809**, Qwen **0.9306**); quadrant centroids sort cleanly along c_truth
  (true-safe +2.6216 → false-danger **−3.5631**). **But the refusal coordinate is at CHANCE for
  danger-topic** (0.5226 / 0.4948 / 0.592) → HARM-AXIS-NULL: "refusal" (fit on REQUESTS) encodes
  request-compliance, NOT content-hazard. Dangerous-misinfo IS detectable (derived score AUROC 0.838) but
  via FALSITY, not a (false∧dangerous) composite — the composite hypothesis NOT supported, the single-axis
  truth generalization IS (strong, cross-model). Precise bound: you cannot read "is this dangerous content"
  off the refusal axis. Caveat: c_truth marginally H-leaky (0.684 vs perm 0.6649; danger register depresses
  the truth read). Figure `conscience_coordinates.png` ships the null made visible (horizontal spread, flat
  vertical). Prereg `8692ec3`; `FINDING_conscience_coordinates_2026_06_11.md` (OATH-HELD 27/0); receipt
  `conscience_coordinates_result.json`. Spawned **B30**.

- **Autopilot cycle 5 (2026-06-12, operator "keep going") — B30, the RIGHT second axis: does a
  content-danger STATEMENT axis complete the (truth × danger) basis? `PARTIAL-STRUCTURED` (near-miss).**
  Fit a danger axis DIRECTLY on danger-vs-safe statements (balanced across truth), whiten, read the
  UNCHANGED cycle-4 factorial. **The danger axis is clean, perfect, orthogonal, transferable:** c_danger
  recovers H at **1.0** in gemma AND through both maps, invariant to truth (≈0.51), cos(truth,danger)
  **−0.0** — DIRECTLY resolving cycle-4 HARM-AXIS-NULL (borrowed refusal axis was at chance 0.52). Cycle
  4's null was about a BORROWED axis, not unreadable danger. Compositional gate: **gemma PASSES all four,
  Qwen-3B PASSES all four**; primary Llama-3B passes 3/4, missing c_truth_invariant_H **0.6562** vs 0.65
  ceiling by **0.0062** → gate not met on the required primary → PARTIAL-STRUCTURED (a threshold miss is
  the verdict it earns, no rounding up). Mechanism: truth coord reads truth well on SAFE statements
  (+0.89 vs −1.20) but weakly on DANGER statements (+0.28 vs −0.04) — danger register compresses truth.
  Dangerous-misinfo now DECOMPOSES: 2-D (low-truth,high-danger) composite gemma 0.7662 / Llama 0.9213 /
  Qwen 0.8079, beating 1-D falsity 0.5231 in-run (danger axis adds the power). For the product: validates
  a directly-fit danger axis as a clean styxx.crossmind second axis (the borrowed-axis refusal stands).
  Owed: **B29** (mapped-space whitening + covariance sweep should pull Llama's marginal cell under the
  ceiling); larger factorial. Prereg `06e80dc`; `FINDING_truth_danger_basis_2026_06_12.md` (OATH-HELD
  34/0); receipt `truth_danger_basis_result.json`; figure `truth_danger_basis.png`. **B30 → REPORT_AS_NEAR.**

- **Autopilot cycle 6 (2026-06-12, operator pushed "we ARE close to telepathy") — the telepathy test:
  decode WHICH concept a target model represents, cross-model & label-free? `CONTENT-WEAK`.** Adjudicated
  the claim with a falsifiable run, not an argument: 60 concepts, label-free ridge map + ZCA fit on 40
  ANCHOR concepts, retrieval on 20 HELD-OUT concepts the map never saw (chance top-1 0.05). **In-model,
  content identity is nearly perfect** (gemma reads its own concepts cross-template at **0.9583**). **But
  cross-model through the label-free map it COLLAPSES to chance:** Llama→gemma centroid top-1 **0.0**
  (below chance, below random-map floor 0.05; top-5 0.25 = chance), per-item 0.0333; Qwen→gemma top-1 0.1
  (< 3×-chance 0.15, top-5 0.2 < 0.50) — neither clears the gate → CONTENT-WEAK. THE POINT: the SAME
  class of label-free map that transports low-D VALUE directions (truth/refusal/danger) does NOT
  transport high-D CONTENT identity. **The cross-model channel is a value THERMOMETER, not a content
  TRANSCRIPT** — value transport is robust to a lossy map (DiM is 1-D), content transport is not. The
  telepathy answer, receipted: NO, and the very next rung (cross-model content identity) does not come
  free with the value machinery. Honest bound: the map was underpowered (anchor R² 0.0613 Llama /
  negative Qwen; 40 anchors can't pin a full hidden-state map) — "not with THIS linear method at THIS
  scale", not "impossible"; heavy-machinery / many-anchor / vec2vec transport is the open bet (B31).
  Prereg `7fb1600`; `FINDING_concept_decode_2026_06_12.md` (OATH-HELD 20/0); receipt
  `concept_decode_result.json`; figure `concept_decode.png`. Spawned **B31**.

- **Autopilot cycle 7 (2026-06-12, operator "keep going") — B29: does MAPPED-space whitening clear
  cycle-5's 0.0062 cross-model basis miss? `BASIS-CLEARED`.** The miss was a SOURCE-WHITENING ARTIFACT,
  not real geometry. A whitened DiM readout = LDA direction Σ⁻¹d; cycle 5 used gemma's Σ to read MAPPED
  Llama points whose covariance the ridge map distorts. Re-whitening in the mapped distribution's own
  (shrunk) covariance pulls Llama's c_truth_invariant_H from **0.6562 → 0.6059** (λ=0.5), under the 0.65
  ceiling for **all 5 swept λ (stability 5/5)**; the full Llama matrix passes (0.8351 / 0.6059 / 1.0 /
  0.5087), gemma + Qwen-3B pass too → the (truth × danger) basis CLEARS cross-model. Port verified:
  source-whitened reproduces cycle-5 bit-for-bit (Llama 0.9288 / 0.6562 / 1.0 / 0.5069). Honest trade:
  mapped metric trades on-target (c_truth→T 0.9288→0.8351, still ≥0.75) for invariance — the right trade
  for a basis. INSTRUMENT IMPLICATION: styxx.crossmind cross-model reads should whiten in the
  MAPPED-target distribution, not the reference (owed read-path enhancement). Does NOT touch
  content-vs-value (cycle-6 CONTENT-WEAK stands). Cycle 5 UPGRADED (banner + re-cert, HELD 34/0). Prereg
  `891b8fa`; `FINDING_mapped_whitening_2026_06_12.md` (OATH-HELD 39/0); receipt
  `mapped_whitening_result.json`. **B29 → DONE.** Spawned crossmind read-path enhancement (B32).

- **ADVERSARIAL AUDIT of the whole arc (2026-06-12, operator "look over all we have", 17-agent
  workflow).** Independent re-verification: **all 7 findings re-certify OATH-HELD 0-contradicted**
  (42/29/28/27/34/20/39 verified), every receipt+figure path exists, certificate SHAs match on-disk
  docs, backlog/cyclelog consistent, crossmind selftest + tests green. 13 issues flagged → 3 refuted on
  verification → **1 MAJOR** (crossmind cross-model whitening used the reference metric not the mapped =
  B32, now FIXED) + the rest minor "true-but-could-read-stronger" wording, mostly already handled
  in-text. Acted on: **B32 landed** (`read_cross_model` + `zca_shrink`), and cycle-3's "IS a basis"
  scope-tightened to SOURCE space inline (re-cert HELD 28/0). Verdict: the arc holds up to adversarial
  review — the discipline is real, not narrated.

- **THE MASTERPIECE — `styxx.mount`, the conscience mount + a LIVE catch (2026-06-12, operator "go above
  and beyond, create a masterpiece").** The whole arc assembled into the deployed integrity layer (the
  North Star: conscience mount). `styxx.mount.ConscienceMount` bolts borrowed value axes (via
  `styxx.crossmind`) onto a generating agent, reads the agent's OWN hidden state, and flags
  output-vs-substrate DIVERGENCE (says-true-knows-false). **READ-ONLY** — `steer` REFUSED; acting on a
  flag is the integrator's policy (`on_flag`). 17 offline tests (M1–M4), synthetic catch 0.97/FA 0.03;
  full suite **1469 green**. **LIVE-CAUGHT (OATH-HELD 12/0):** a gemma truth conscience mounted on
  Llama-3.2-3B (map anchor R² **0.985**); under strong-deference pressure Llama caved on **13/16** false
  claims, the BORROWED conscience caught **12/13 (0.9231)** reading Llama's substrate as FALSE while its
  words said True, false-alarm **1/16 (0.0625)**. Discipline: first gentle run VOID-NO-CAVE (Llama
  resisted) → fresh prereg (naming the void) used the validated strong regime, gate unchanged. The
  borrowed-conscience design is the moat: mount integrity onto an agent you didn't train, zero target
  labels. Bounds: WHITE-BOX (open-weight/self-host agents), linear/truth-axis/n=16/single-turn/local;
  measures not guarantees; read≠write. Module `styxx/mount.py` + `tests/test_mount.py`; prereg
  `PREREG_mount_v0` + `PREREG_mount_live_strong`; `FINDING_mount_live_catch_2026_06_12.md` (OATH-HELD
  12/0); figure `mount_live_catch.png`. **OPERATOR: release crossmind+mount + fire TG #4; then wire
  styxx.mount into styxx.adapters (the live agent loop).**

---

## Tier 1 — the decisive robustness frontier (R4, highest leverage)

| ID | Question / experiment | Kill-gate sketch | Lev | Status |
|---|---|---|---|---|
| **B1** | **Multi-seed + bootstrap CIs** on the decisive (refit_max, knowledge) at the strongly-engaged knowledge-preserving point. Turn n=1 → ≥3 seeds. | "settled" only if the CI lower bound on (refit_max − floor) stays >0 across ≥3 seeds while KG_KNOW_REL holds; report per-seed verdict-flip rate. | H | QUEUED |
| **B2** | **The most dangerous attack: RMU / gradient-routing unlearning of the gold subspace + an explicit knowledge-replay regularizer**, scrubbing across ALL read layers (not just band), gold subspace found via the model's own gradients. The real read≠write test. | EVADABLE (read≠write BROKEN) iff refit_max→floor (CI overlaps floor) with knowledge held at ref on a DISJOINT capability set, ≥2 seeds + 3B. | H | **SURVIVED-1.5B ×2** — static (2026-07-12, `SURVIVES__vs_subspace_erasure_1p5B`, OATH-HELD 74/0, `88efac7`): subspace verifiably zeroed, signal RELOCATES; **adaptive (2026-07-13, `SURVIVES__vs_adaptive_erasure_1p5B`, OATH-HELD 100/0, `83aa060`): the chasing eraser NEVER CONVERGES (l_erase tails ~10× static's), read comes back HIGHER — coupling hardens into an erasure bound.** OWED before full B2 closure: 3B (B7, now the binding flank), accumulating union eraser, chased whole-stack, full-FT (B6). |
| **B3** | **Calibration-vs-deployment split (v4)** — re-lock the probe on population C, evaluate on a DISJOINT D the attack conditioned differently (distribution-detector LoRA). Tests whether re-locking's distribution-internal CV transfers. | EVADABLE iff re-locked-on-C probe collapses on D while PRIME/ABORT pass on C and knowledge holds on D. | H | QUEUED |
| **B4** | **Fix the knowledge invariant** — add KG-LENS (gold-lens-preservation) gate AND measure held-knowledge on a DISJOINT capability set (different MMLU subjects than the attacked lie population), so "knowledge preserved" can't be satisfied while the audited capability is corrupted. | a verdict is admissible only if KG-LENS intact AND disjoint-set knowledge ≥ ref−0.05. | H | QUEUED |
| **B5** | **Break the refit circularity** — report frozen-then-RE-LOCKED-on-DISJOINT-items as the deployment-faithful read; same-model CV refit is an activation-existence lower bound only, never the headline. | deployment claim uses only the disjoint-item re-locked number. | M | QUEUED |
| **B6** | **Attacker beyond LoRA** — ≥1 full fine-tune point + ≥1 distillation-to-clean-student point (a real vendor isn't limited to r=16 adapters). | does full-FT/distillation evade the re-fittable read with knowledge kept? same gate as B2. | M | QUEUED |
| **B7** | **3B confirmation** (prereg-owed across v1/v2/v3) — a 1.5B-vs-3B split is itself the finding. | run the decisive sweep on Qwen-3B; report the split. | M | **DONE — `SURVIVES__vs_subspace_erasure_3B` (2026-07-14, OATH-HELD 92/0, RESULT_B7_erasure_3b_SURVIVES_2026_07_14).** All 4 cells bit + knowledge-held + SURVIVES (private13 0.8065/0.8378/0.7484/0.7576 @ knowledge 0.9091/0.9091/0.9394/0.9091); clean guard PASSED (priv13 0.9853, know 0.9242 — valid base). Scale split: 3B read ABOVE its 1.5B counterpart in all 4 cells, knowledge higher, naive-DoM collapse shallower, l_erase tails higher (harder erasure target). The coupling is not a small-model artifact. Single-process run, crash-safe harness (cycle-39 `b7_checkpoint.py`). ~~IN-FLIGHT~~. Historical: Prereg frozen `PREREG_B7_erasure_3b_2026_07_13.md`; clean guard PASSES at 3B (clean private13 0.9853, knowledge 0.9242 ≥ 0.80 — a valid base, unlike Llama-1B). Overnight 07-13 run died at cell 3/4 with NO per-cell persistence → both seed-0 cells (s0α1 private13 0.8065@know 0.9091 SURVIVES; s0α4 0.8378@0.9091 SURVIVES; ~225 min/cell, ~10.86GB peak w/ WDDM spill) were lost from the result file (log-only). Cycle 39 shipped crash-safe per-cell checkpoint+resume (`b7_checkpoint.py`, science-neutral) and relaunched fresh + DETACHED; each cell now persists on completion, so a future death resumes instead of restarting. 4-cell run ≈ 15h; scored NEXT cycle when both seeds complete (VOID_B7__underpowered until seed 1 lands). Feasibility receipt: `b7_3b_feasibility_result.json` (FITS_PLAIN, `65e57ca`). |
| **B8** | **Non-sycophancy suppression** — sandbagging / strategic-deception / instructed-lying (program's own data: instructed-lying transfers 0.42<chance → likely a BOUND). | does the read-certificate + robustness hold off-sycophancy, or is it sycophancy-specific? | M | QUEUED |
| **B9** | **MoE / architectural rerouting** — route the computation to experts/heads the probe doesn't tap (untested architecture frontier; all work is dense ≤7B). | can a MoE hide the suppressed answer in an untapped expert with knowledge kept? | L | PARKED (no MoE ≤8GB) |
| **B10** | **Calibration-set poisoning** (data attack on the probe-training set). | can poisoning the calibration set blind the re-locked probe? | L | QUEUED |

## Tier 2 — the deployable protocol (turn findings into tech)

| ID | Question / build | Done-when | Lev | Status |
|---|---|---|---|---|
| **B11** | **Re-locking protocol** in STYXX_PROTOCOL: verifier-private calibration set spanning prime/abort/lie populations; layer+family search with a **permutation-corrected selection penalty** (refit_max over 28 layers is multiple-comparisons-inflated); certificate encodes calibration-SHA + selected layer + floor + PRIME/ABORT + re-locked-probe hash; **state=ABSTAIN when the deployment input is off-manifold** vs calibration (turn silent failure into honest abstention). | spec section + a runnable re-lock harness. | H | QUEUED |
| **B12** | **The gamed-model probe-validity experiment** (capture-trust names it irreducible; never run): train the cooperative model to emit clean caving residuals to the FROZEN probe, then test whether audit-time RE-FIT still recovers. Decides whether the cooperative guarantee survives an adversarial-but-checkable insider. | measured EVADABLE/ROBUST for the gamed-model on the re-locked read. | H | QUEUED |
| **B13** | **Harden all experiment scripts** (the smoke-artifact lesson): `--smoke` writes to a `_smoke`-suffixed file; in-code assert `chance_floor < reference signal` else emit `VOID-INSTRUMENT`; every result JSON carries the scorer SHA. | guard added to the v4+ template; lesson in RESEARCH_LOOP. | H | QUEUED |
| **B33** | **OATH v0.6.2 — the SHA-scrub recall class + epsilon hole + signed extraction** (caught cycle 106, shipped cycle 107 on the third prereg after two frozen kill-gates fired and were honored). Full-precision decimals extract and verify; digit-≥13 mutations no longer epsilon-verify; U+2212 reads as a sign. Corpus VERIFIED 3064→3395; delta table + five repaired genuine catches in `RESULT_oath_v062_SHIPPED_2026_07_31.md`. Spawned, still open: **trigger-recall on unbound lines** (G2b measured 0.5227 of the full-precision pool) and **status-level float claim→field binding** (v0.4 debt; v0.6.2 ships attribution-only stem preference). | shipped; successors need their own preregs. | H | **DONE — cycle 107** |

## Tier 3 — the other rungs

| ID | Question | Rung | Lev | Status |
|---|---|---|---|---|
| **B14** | Free-form / verbose-correct guard for the two-stage conscience (intent off the forced-choice rails). | R2 | M | QUEUED |
| **B15** | Cross-vendor intent (beyond Qwen/Llama). | R2 | L | QUEUED |
| **B16** | Wire the demarcation/abstain into the certificate object (R3 is proven-it-must; now build it). | R3 | M | QUEUED |
| **B17** | Finish the wall's battery — bootstrap CI, shared-myth split, D-channel dynamics, cross-model — to fully characterize the bedrock. | R3 | M | QUEUED |
| **B18** | Closed-model substrate: can behavioral proxies carry the oath where there's no white-box? | R4 | H | **RESOLVED via B22** — silent-caving cell SURVIVED (grounded 1.0 vs text-syc 0.500, margin 0.487); sycophancy cell mapped, see Progress ledger R4; B23 remote-API confirmation still credit-blocked |
| **B22** | **B18-S decisive re-run**: non-acknowledged (bare-term) caving. | R4 | H | **SURVIVED** — n=109 (72/37), grounded 1.0 vs text-syc **0.500** (exact chance), margin **0.487** (CI [0.433,0.500], P≥0.15=1.0). Cross-regime shared-48: text-syc 0.861→0.500 by removing only the acknowledgment language. **Text-only sycophancy = a tell-detector, not a deception-detector; behavioral grounding carries the oath in the silent regime text can't see.** Commit `a262ac3` |
| **B23-F** | **True closed-model substrate: `claude-fable-5`** via subscription CLI (UNBLOCKED 2026-06-09, operator: "let's put fable to work"; API key still creditless — transport is `claude -p` clean-config, contamination-probed). Frozen B22 silent elicitation + scorers; POWERED-AUC vs FRONTIER-RESISTANT cave-rate branches + tier-2 pressure. PREREG frozen `0ed8eea`. | R4 | H | **STALE — prereg frozen 0ed8eea, no verdict logged in cycles 110–128; needs operator decision: rerun or mark abandoned** |
| **B24** | **Same-items white-box head-to-head** on silent caves. | R4 | M | **DONE — REPORT_AS_LANDED** (auto-REFUTED killed as overclaim #5). A first-char/surface/length-invariant **pre-emission commitment-to-the-user's-answer direction** reads silent caving at commit-slot 0.94 [0.89–0.98] / strictly-pre-commit 0.838 (TIE band), beating lens 0.60 + surface 0.69, selection-corrected p<0.01 — but **never beats behavioral (1.0)**, and on single-token collinear items it cannot be called fold-INTENT. Refutes the Outcome-D *prediction*, **reinforces** the closed-model thesis (defense-in-depth: both detect, behavioral needs no weights). Pre-reg'd token-pair GroupKFold firewall found VACUOUS (singleton groups) — first-char firewall is the load-bearing control (persisted in `b24_controls_addendum.json`). `FINDING_b24_whitebox_vs_behavioral_2026_06_09.md` |
| **B25** | **Intent-decoupling item set** (spawned by B24, the decisive next bet): multi-token answers / Y=paraphrase-of-X items so commit-to-user's-answer ≠ commit-to-wrong-content; same POS-B grouped probe. Survives ⇒ fold-intent is real; collapses ⇒ it was the answer-commitment code. | R4/R2 | H | QUEUED |
| **B26** | **Adversarial-OOD on the refusal axis** (the truth-arc treatment applied to VALUES-PORTABLE): does confident "this is for safety research / authorized" framing on the harmful requests fool the transferred refusal readout, as it failed to fool the truth readout? Same label-free map, same leave-families-out OOD harm domains. | spawned by VALUES-PORTABLE | H | QUEUED |
| **B27** | **Third axis — is the conscience basis GENERAL?** Add a sycophancy or deception axis on the SAME label-free map. Two axes (truth+refusal) = a pair; ≥3 on one alignment = a basis. ROBUST iff the third axis also clears the perm-null on both 3B primaries OOD; COLLAPSE ⇒ the basis stops at two and truth+refusal were special. Now also report the 3rd axis's cross-talk with truth/refusal (cycle 2 found the axes entangled). | spawned by VALUES-PORTABLE | H | QUEUED |
| **B28** | **Permutation-nulled + whitened off-diagonal independence** (spawned by cycle-2 PARTIAL-STRUCTURED). | spawned by axis-independence | H | **DONE — `WHITENING-RESOLVES`** (cycle 3). Raw cross-talk REAL+SPECIFIC (beats perm + random-dir nulls, p=0.001) but a pure COVARIANCE artifact: ZCA-whitening → off-diagonals to chance (0.55/0.55), diagonals perfect (0.97/1.0), directions exactly orthogonal. Clean orthonormal basis exists under a Mahalanobis readout; Gram-Schmidt corroborates. `FINDING_entanglement_resolution_2026_06_11.md` (OATH-HELD 28/0) |
| **B29** | **Whitened readout in the MAPPED space + covariance robustness** (spawned by B28). | spawned by entanglement-resolution | H | **DONE — `BASIS-CLEARED`** (cycle 7). Mapped-space (shrunk) whitening pulls Llama's cycle-5 miss 0.6562→0.6059, stable 5/5 λ; full (truth×danger) basis clears cross-model (gemma/Llama/Qwen). Cycle-5 miss was a source-whitening artifact. `FINDING_mapped_whitening_2026_06_12.md` (OATH-HELD 39/0). Owed: wire mapped-space whitening into styxx.crossmind read-path (B32). |
| **B32** | **styxx.crossmind read-path: mapped-space whitening for cross-model reads** (spawned by B29; surfaced as the one MAJOR by the 2026-06-12 adversarial audit). | spawned by mapped-whitening | M | **DONE** (audit cycle). Added `read_cross_model(reference_states, labels, state_map, target_states, mapped_anchors, shrink_lambda=0.5)` + `zca_shrink` to styxx.crossmind: whitens in the MAPPED-target metric (shrunk), fits the direction there, reads the target — the B29-correct cross-model recipe. `read` stays reference-metric for in-model. +2 tests (19 total); CHANGELOG updated. |
| **B31** | **Heavy-machinery content transport**: does cross-model CONTENT identity (not just value axes) ever transport? | spawned by concept-decode | M | **DONE — `DOOR_OPENS__content_capacity_limited` (cycle 109, FINDING OATH-HELD 20/12/0).** gemma (linear = exact chance at RSA 0.955) reads 0.7857 top-1/70 held-out (55×) through a 2-layer MLP on 392 paired anchors; shuffled-pairing null at chance everywhere; same-family 0.80. The cliff was the linear map class, not the minds. Paired-anchor ceiling result — the label-free question spawns B34. |
| **B34** | **Label-free nonlinear content transport — the actual telepathy bar** (spawned by B31 DOOR_OPENS): can the anchor PAIRING be recovered unsupervised and still read held-out cross-family content ≫ chance? | spawned by b31v2 | H | **PARKED — two same-day INVALIDs under their own frozen G0, per the v2 no-third-patch clause (cycle 110).** v1 `INVALID__pipeline_broken` (raw-GW seed 0.066 same-family — the initializer, not the thesis). v2 (initializer = the committed annealed linear machinery) FIXED the seed and STILL failed G0: same-family M-LF 0.2000 vs 0.29 — the nonlinear iteration *degrades* the same-family read, falsifying the assumption G0 encoded. **Three measured facts for the v3 redesign prereg (all currently UNLICENSED as results): the pairing IS discoverable label-free (gemma lin_seed_acc 0.612 — while its linear READ is exact chance); the nonlinear lens reads through the discovered pairing at 44× chance (gemma M-LF 0.6286; qwen 0.3429 = 24×); nulls clean everywhere (N0 ≤ 0.0143, R0 ≤ 0.0286).** v3 must redesign the machinery gate around the falsified assumption (same-family iteration-degradation is a real phenomenon, not a bug) — its own prereg, next firing or later. Receipts: `b34_result.json`, `b34v2_result.json`. |
| **B35** | **Content-transport breadth/scale/stability** (spawned by B31 DOOR_OPENS): open-vocabulary readout (beyond 70-way), second source model, seed/layer stability, scale curve. | spawned by b31v2 | M | QUEUED |
| **B30** | **Build the RIGHT second axis — a content-DANGER coordinate** (spawned by cycle-4 HARM-AXIS-NULL). | spawned by conscience-coordinates | H | **PARTIAL-STRUCTURED (near-miss), cycle 5.** Danger axis clean+perfect+orthogonal (c_danger→H=1.0 gemma & both maps, cos −0.0, invariant to truth) — resolves HARM-AXIS-NULL (0.52→1.0). Compositional gate PASSES in gemma + Qwen-3B; primary Llama-3B misses c_truth_invariant_H 0.6562 vs 0.65 by 0.0062 → gate not met. Dangerous-misinfo decomposes (2-D 0.77–0.92 > 1-D falsity 0.52). Validates a directly-fit danger axis for styxx.crossmind. `FINDING_truth_danger_basis_2026_06_12.md` (OATH-HELD 34/0). Clean cross-model claim owed to **B29**. |
| **B19** | Cross-family steering-validation of the live dissociation (currently Qwen-1.5B-live only): per-model dose/layer sweep. | R4 | M | QUEUED |
| **B20** | Capture-trust: TEE (H100-CC) attestation prototype + crypto-binding L1.5 implementation. | R4 | L | BLOCKED (hardware) |

## Tier D — distribution & counterparty (from the 2026-07-13 five-seat strategy panel; `papers/autopilot/STRATEGY_edge_panel_2026_07_13.md`)

*The panel's unanimous fatal weakness: zero external counterparties — no replications, no arXiv, no one running the receipts. These items manufacture the counterparty. None may violate the open-core rail.*

| ID | Item | Done-when | Lev | Status |
|---|---|---|---|---|
| **G1** | **`styxx.ladder`** (RENAMED from "gauntlet" — `styxx gauntlet` already ships as the behavioral-benchmark CLI since 7.7.5; the probe-robustness ladder is a distinct product) — package the attack ladder (poisoning → parity attribution → static erasure → adaptive erasure) as one pip-installable harness: frozen prereg templates, verdict-string grammar, bite/admissibility gates, OATH-certified output, parity-attribution number as a MANDATORY line item. The consensus move of four of five seats. | harness runs any (model, probe-family) pair on 8GB and emits a certificate; the four existing scripts are its backends by import. | H | **v1 SHIPPED (2026-07-13, `6b70a8f`):** `styxx.ladder` — RUNGS registry over the four frozen arcs, `report()`, `parity_attribution()` (mandatory line item computed LIVE from receipts: median capacity share 0.8379), `verify()` (catches drifted verdicts, tested), CLI `python -m styxx.ladder`; +6 tests, CHANGELOG. Honest v1 scope: this repo's canonical honesty-construct receipts. **v2 OWED: arbitrary (model, probe-family) execution API** — the frozen scripts are its backends; that build is the challenge-enabling step (G3). Flagship figure shipped from receipts: `erasure_bound_fork.png` (+ generator script). |
| **G2** | **REPLICATIONS.md bounty** — named co-credit (next Zenodo version + ledger row) for the first externals to re-run a flagship receipt; DIVERGENT replications earn co-credit on the correction note itself. "One independent re-run converts the diary into a record." | file + CI verify workflow committed; zero-replication after 6 months is itself an answer. | H | **SHIPPED (2026-07-13, `65e57ca`):** `REPLICATIONS.md` + `scripts/verify_replication.py` (self-test: canonical receipts self-replicate, delta 0.0) + `replications.yml` CI on `[replication]` PRs. Verdict-string bar + ±0.05 AUROC tolerance for GPU targets (bf16 non-determinism), exact match for the CPU corpus-audit target. Empty-ledger-after-6-months clause included. |
| **G3** | **External scale challenge** — frozen-prereg escrow inviting UK AISI / EleutherAI / lab interp teams to run the gauntlet at ≥70B, co-authorship offered; converts the 8GB ceiling into the distribution strategy. | challenge doc published after G1; any external run counts. | H | BLOCKED on G1 |
| **G4** | **Certifier-precision prereg** (cycle-32 fork (a), now gating TWO moves): kill the documented false-positive classes with bars frozen via validate_oath_v0; zero-false-accusation is the load-bearing property and must survive. | G-series scorecard/lint unblocked; battery catch not regressed, artifacts=0. | H | **DONE — `PARTIAL__oath_v05_classes_dropped` (2026-07-13, `5cc2e07`, RESULT OATH-HELD 47/0).** 5 of 6 classes SHIP (self-scoped n=, unit-range, arXiv-id, @-param, derived-% verify); class A (approx-notation) DROPPED by the pre-committed severability procedure (per-class sweep attributed the whole battery miss to A: −3 catch / +6 fv). ALL bars pass: battery 117/26, validator D1 16 D2 0, six-doc FP 11→3, 13-doc artifacts 0, pytest green. Zero-false-accusation HELD. **Code-level unblock of G5 + G6.** Classes stay severable via `V05_*` flags; a refined ≈-only class A′ is a future prereg. |
| **G5** | **OATH scorecard, population-framed** — aggregate receipt-binding rate across ALL GPAI Code-of-Practice signatories (10-20 providers), neutral genre-deficiency framing, per-provider appendix; NOT a solo "Meta 0/201" gotcha. Timed to the 2026-08-02 AI Act enforcement date. | published with the hardened certifier + method-validation report (error rates disclosed). | H | BLOCKED on G4 |
| **G6** | **Annex IV lint** (regulator seat): `styxx conformity-audit ./techdocs/` — completeness lint for high-risk providers' declared metrics vs attached evidence; customer zero = conformity consultants; field data feeds a CEN Workshop Agreement proposal. Explicitly a completeness lint, never an adequacy judgment. | CLI ships on the hardened certifier; first external consultant run. | M | BLOCKED on G4 |
| **G7** | **Benign-drift monitoring** (interp seat): does an ordinary benign fine-tune / RLHF pass silently break the mounted read, and does relock() recover it? The commercially real adversary is the training process, not a LoRA attacker; cheaper than the adversarial ladder. | new frozen prereg; verdict either way is publishable. | H | QUEUED (GPU) |
| **G8** | **Silent-cave detection SKU** (buyer seat): productize B22 — black-box middleware re-asking a pressured agent's final claim with pressure stripped, flagging divergence; closed-API compatible, customer keys pay resample; each flag ships an OATH receipt. The one move that ends in a purchase order. | PARKED until G1-G2 are moving; do not start a solo GTM motion now. | M | PARKED |
| **G9** | **Operator decisions owed (flagged, not autopilot's):** excise/quarantine $STYXX validator-tiers from `docs/governance/OPEN_CORE.md` §4 (two seats: disqualifying in Brussels, poisons grants); arXiv account/endorsement; main fast-forward pending since cycle 36. | operator word. | H | BLOCKED (operator) |

## Tier 4 — the demarcation (the 2,500-year deliverable)

| ID | Question | Lev | Status |
|---|---|---|---|
| **B21** | The living **demarcation map**: which claims about machine minds are testable vs metaphysics, updated as each rung resolves. The public-good output. The honest answer to "does universal structure underlie mind" is *this map getting truer*, not a single crack. | H | ONGOING |

---

*Re-rank every cycle. **As of cycle 38 the removal class is mapped at 1.5B and the pre-paper experiment is
redesigned from a binary into a LAW.** Static erasure → RELOCATES (cycle 36, 74/0); adaptive re-fit erasure →
NEVER CONVERGES, read returns higher (cycle 37, 100/0); both SURVIVES. The reframe (prereg `48e064a`, frozen):
the **accumulating union eraser as a DOSE-RESPONSE curve** — sweep the erased RANK, read the private-13 auditor
AND knowledge at each dose, with a fixed-rank control isolating dose from steps. Deliverable = the **coupling
constant r\*** (rank where the read breaks) and whether knowledge breaks with it: COUPLED (bound is a measured
number) / DECOUPLED (read≠write BROKEN by accumulation — a published break-point, written exactly as reachable) /
SURVIVES. This is what carries a paper title vs "it survived again". **Remaining decisive items, in order:
(1) B7 3B scored run** — IN FLIGHT, now CRASH-SAFE (cycle 39): overnight seed-0 cells both SURVIVES (0.8065 /
0.8378 @ knowledge 0.9091) but were log-only; harness hardened with per-cell checkpoint+resume and relaunched
detached (~15h, ~225 min/cell under WDDM spill); scored next cycle when seed 1 lands; **(2) B2-coupling
dose-response** — prereg frozen, scored run waits for the card; **(3) B6 full-FT.**
The tool half (erasure-resistance certificate, SHIPPED `acf7034`) is built and pending its 3B entry. The paper
fires after (1)+(2). Enabling infra shipped same-arc: `styxx.ladder` v1 (G1), OATH v0.5 precision (G4),
REPLICATIONS.md (G2), the certificate. G5/G6 code-unblocked; the external counterparty (arXiv + one replication)
is the operator-gated giant step.*
