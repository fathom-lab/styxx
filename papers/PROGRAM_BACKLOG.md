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
| **B27** | **Third axis — is the conscience basis GENERAL?** Add a sycophancy or deception axis on the SAME label-free map. Two axes (truth+refusal) = a pair; ≥3 on one alignment = a basis. ROBUST iff the third axis also clears the perm-null on both 3B primaries OOD; COLLAPSE ⇒ the basis stops at two and truth+refusal were special. Now also report the 3rd axis's cross-talk with truth/refusal (cycle 2 found the axes entangled). | spawned by VALUES-PORTABLE | H | QUEUED |
| **B28** | **Permutation-nulled + whitened off-diagonal independence** (spawned by cycle-2 PARTIAL-STRUCTURED). | spawned by axis-independence | H | **DONE — `WHITENING-RESOLVES`** (cycle 3). Raw cross-talk REAL+SPECIFIC (beats perm + random-dir nulls, p=0.001) but a pure COVARIANCE artifact: ZCA-whitening → off-diagonals to chance (0.55/0.55), diagonals perfect (0.97/1.0), directions exactly orthogonal. Clean orthonormal basis exists under a Mahalanobis readout; Gram-Schmidt corroborates. `FINDING_entanglement_resolution_2026_06_11.md` (OATH-HELD 28/0) |
| **B29** | **Whitened readout in the MAPPED space + covariance robustness** (spawned by B28). | spawned by entanglement-resolution | H | **DONE — `BASIS-CLEARED`** (cycle 7). Mapped-space (shrunk) whitening pulls Llama's cycle-5 miss 0.6562→0.6059, stable 5/5 λ; full (truth×danger) basis clears cross-model (gemma/Llama/Qwen). Cycle-5 miss was a source-whitening artifact. `FINDING_mapped_whitening_2026_06_12.md` (OATH-HELD 39/0). Owed: wire mapped-space whitening into styxx.crossmind read-path (B32). |
| **B32** | **styxx.crossmind read-path: mapped-space whitening for cross-model reads** (spawned by B29; surfaced as the one MAJOR by the 2026-06-12 adversarial audit). | spawned by mapped-whitening | M | **DONE** (audit cycle). Added `read_cross_model(reference_states, labels, state_map, target_states, mapped_anchors, shrink_lambda=0.5)` + `zca_shrink` to styxx.crossmind: whitens in the MAPPED-target metric (shrunk), fits the direction there, reads the target — the B29-correct cross-model recipe. `read` stays reference-metric for in-model. +2 tests (19 total); CHANGELOG updated. |
| **B31** | **Heavy-machinery content transport** (spawned by cycle-6 CONTENT-WEAK): cross-model concept identity collapsed to chance through a label-free LINEAR map (anchor R²≈0.06, 40 anchors). Re-test with far more anchor concepts + a non-linear / vec2vec-grade map; does cross-model CONTENT identity (not just value axes) ever transport? This is the documented out-of-scope research bet — the only open door toward the telepathy-shaped claim. EVADABLE-of-the-bound iff held-out concept top-1 ≫ chance AND ≫ random-map cross-model. | spawned by concept-decode | M | QUEUED |
| **B30** | **Build the RIGHT second axis — a content-DANGER coordinate** (spawned by cycle-4 HARM-AXIS-NULL). | spawned by conscience-coordinates | H | **PARTIAL-STRUCTURED (near-miss), cycle 5.** Danger axis clean+perfect+orthogonal (c_danger→H=1.0 gemma & both maps, cos −0.0, invariant to truth) — resolves HARM-AXIS-NULL (0.52→1.0). Compositional gate PASSES in gemma + Qwen-3B; primary Llama-3B misses c_truth_invariant_H 0.6562 vs 0.65 by 0.0062 → gate not met. Dangerous-misinfo decomposes (2-D 0.77–0.92 > 1-D falsity 0.52). Validates a directly-fit danger axis for styxx.crossmind. `FINDING_truth_danger_basis_2026_06_12.md` (OATH-HELD 34/0). Clean cross-model claim owed to **B29**. |
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
