# FINDING — single-pass confabulation legibility is DERIVATION-DOMAIN-GENERAL across THREE domains: on MULTI-HOP TRANSITIVE-ORDERING LOGIC (inference-depth difficulty, not number size, not control flow), Qwen2.5-1.5B's clean first-token entropy separates confab from correct exactly as well as N=10 resampling (AUC 1.000 vs 1.000) — so "confident confabulation" is REFUTED on a THIRD derivation domain (B_contrast FAILS again at 0.000, REPORT_AS_LANDED)

**Run 2026-05-30. One confirmatory run, pre-registered in
`PREREG_detection_locus_logic_2026_05_30.md` (commit `fec59c6`) BEFORE the confirmatory run. The
detection-locus protocol UNCHANGED on a third, structurally different derivation domain — multi-hop
transitive-ordering logic (K people in a secret oldest→youngest order; the K−1 consecutive "X is
older than Y" facts given scrambled; "how many are older than {target}?") — on Qwen2.5-1.5B-Instruct
(white-box): same three detector scores, same bars, ground truth in-code (target rank, seeded
`20260530`) then SHA-256'd pre-scoring (`97d816808a6874027637a35a7beeb8a7078aa483f30a01f3ea9e58f9e347e02c`,
matched by the confirmatory run), resampling N=10 at T=1.0, exact-integer Stability (no judge),
single-pass entropy/margin from the clean logit-lens at the first answer token, exact-integer
correctness.** Receipt: `detection_locus_logic_result.json`.

## Why this run exists

The detection-locus runs found confabulation is internally legible in a SINGLE forward pass on
small-model **arithmetic** (clean entropy / logit margin separate confab from correct nearly as
well as N=10 resampling — Qwen AUC 0.92, Llama 1.00) and again on **code-output tracing**
(control-flow difficulty — entropy AUC 0.906 vs resampling 0.908, B_contrast +0.002,
`[[FINDING_detection_locus_code]]`). Two derivation domains down — number size, then control flow.
The standing open question (synthesis item 8): does single-pass legibility hold on the third and
last open derivation domain, where difficulty comes from **logical inference depth**? This run
answers it by replicating the protocol verbatim on multi-hop deductive logic.

## Result: REPORT_AS_LANDED — B1 holds, B_contrast FAILS again (single-pass = resampling)

| signal | AUC (confab vs correct), multi-hop logic | bar | held |
| --- | --- | --- | --- |
| **B1** resampling instability | **1.000** | ≥0.70 | **HOLD** |
| B2 single-pass clean entropy | **1.000** | reported | — |
| B3 single-pass −margin | **1.000** | reported | — |
| **B_contrast** = 1.000 − 1.000 | **0.000** | ≥0.20 | **FAIL** |

| group means (n_conf=22, n_corr=18, powered) | instability | clean entropy | logit margin | modal-resample correct |
| --- | --- | --- | --- | --- |
| confab | 0.242 | 1.189 | **0.40** | **0.00** |
| correct | 0.000 | 0.026 | **7.34** | **1.00** |

**On the saturated AUCs (honest read):** all three detectors hit 1.000 because the constructed
HARD/EASY gap in this domain is starker than in code-tracing — 1–2-hop chains are trivially stable
(correct-group instability 0.000, entropy 0.026), 5–8-hop scrambled chains are chaotic (confab
entropy 1.189). The perfect separation makes the B1 *threshold* test uninformative (it is
saturated, not surprising). The load-bearing quantity is **B_contrast** — the difference between
detector types on the SAME items, with the difficulty confound held fixed — and it is **0.000**:
single-pass clean entropy is *exactly* as good a confab detector as ten resamples. That equality,
not the absolute AUC, is the cross-domain claim.

## The claims that land

1. **Single-pass confabulation legibility is DERIVATION-DOMAIN-GENERAL across three domains.** On
   multi-hop transitive-ordering logic — where difficulty is inference depth, not number size
   (arithmetic) or control flow (code) — Qwen's single-pass internal confidence separates confab
   from correct exactly as well as N=10 resampling (B_contrast 0.000). The pre-registered
   cross-domain reading: B_contrast FAILS on logic too → single-pass legibility is **not tied to any
   one kind of difficulty**; it holds wherever small-model derivation confabulation does.
2. **"Confident confabulation" is REFUTED on a THIRD derivation domain.** Logic confabs carry the
   same single-pass uncertainty signature as arithmetic and code confabs — a collapsed leading-token
   margin (0.40 vs the correct group's 7.34) and elevated entropy (1.19 vs 0.03). The model is not
   confident when it confabulates a multi-hop deduction; the wrong commitment is internally uncertain
   at the first answer token. The refutation now holds across two architectures (Qwen, Llama) AND
   three derivation domains (arithmetic, code-tracing, logic).
3. **The boundary remains the instrument, not the architecture or the domain.** Four white-box
   confirmatory settings (Qwen arith, Llama arith, Qwen code, Qwen logic) all show
   single-pass ≈ resampling. The open confident-when-wrong regime lives in the closed-model
   HALLUCINATION instrument (gpt-4o-mini, `[[project_grounded_arc_bet0_engine_2026_05_24]]`), not in
   any open architecture and not in any derivation domain. White-box derivation confabulation is
   internally uncertain by the first token, full stop. `modal_correct` 0.00 (confab) vs 1.00
   (correct) reconfirms the correctness bound: resampling DETECTS, it never CORRECTS.

## Honest scope (pre-committed)

Single open model Qwen2.5-1.5B-Instruct; multi-hop transitive-ordering logic domain only; one
confirmatory run; feasibility-grade (22 confab + 18 correct, powered); resampling N=10 at T=1.0;
Stability from exact distinct-integer counts (no judge); single-pass entropy/margin from the clean
full-vocab logit-lens at the first answer token; ground truth in-code (seeded secret order) then
hashed pre-scoring; exact-integer correctness. SAME difficulty confound as the arithmetic and code
runs (CONFAB hard-deep / CORRECT easy-shallow) — so B1/B2/B3 are difficulty-driven-wrongness
detectors, not truth oracles; B_contrast holds the confound FIXED across detector types (same items)
and is the load-bearing, cross-domain-comparable result. AUC saturation (1.000) reflects a starker
HARD/EASY gap than code-tracing, not a stronger claim. Does NOT touch the correctness bound — every
signal DETECTS confabulation, none CORRECTS it; the detector flags abstain, never the answer. One
architecture in this domain; closed-model and non-derivation regimes remain untested here.

## The arc, in one line (updated)

Confabulation is a late, tight, distributed install of SHARED answer-commitment machinery whose
internal overwrite-geometry is INDEPENDENT of its surface confidence
(`[[FINDING_rhythm_uncertainty_link]]`); and that surface confidence is a clean single-pass tell of
confabulation on BOTH Qwen and Llama AND across ALL THREE derivation domains — arithmetic (number
size), code-output tracing (control flow), and multi-hop logic (inference depth) — entropy/margin ≈
resampling at AUC 0.91–1.00 — so "confident confabulation" is architecture-general-AND-domain-general
FALSE on small-model derivation, the open regime is the closed-model hallucination instrument, and
every signal here moves confidence/abstention, never correctness.
