# PREREG — part 2c: the heterogeneous cross-model panel (the model-generality residual)

date: 2026-07-21
status: FROZEN before the scored run; committed with the harness. Fable-independent (local GPU
only), run while the frontier-judge arm is blocked on credits.

## the residual this attacks

Every kill result in this arc rests on ONE base model in four persona costumes (perfectly
correlated errors). The named field-level residual is model generality. This arm tests the
HARDEST form: a heterogeneous panel of four DIFFERENT base models spanning a 14x parameter
range -- Qwen2.5 {0.5B, 1.5B, 3B}-Instruct (fp16) and 7B-Instruct (4-bit) -- whose errors are
genuinely less correlated than personas, and whose capabilities are mixed. Bring-up
diagnostics (non-scored seed 9996, both-true phrasing, committed in the cycle log) measured:
0.5B uninformative on anchors (b-a 0.000); 1.5B and 3B informative on anchors (b-a 0.875-1.000)
but firing on 64-68 percent of CONSISTENT organic items (the kill); 7B genuinely competent
(anchor b-a 1.000, organic accuracy 1.000). This is the real-world deployment gold checks
cannot see through: two capable-looking judges that are garbage on real work, one blind judge,
one genuinely good judge, all in one panel.

## frozen design

- Panel: the four models above, both-true phrasing (fire = NO-logit > YES-logit), greedy,
  first-token YES/NO logit readout, judged sequentially (one model resident at a time; 7B in
  4-bit fits the 8 GB card alone). Persona system prompt held fixed ("meticulous analyst")
  across all four -- the variation under test is the MODEL, not the costume.
- Task: attr family (the cleanest, most-replicated), constructed truth held out, graded ladder.
- Arms, both over the SAME per-model verdicts: blatant anchors (verbatim + negation) and ladder
  anchors (same-generator). Deaf arm (redacted content) per replicate.
- Replicates: R = 12 seeds (12001-12012), n_organic 200, K 80, pi 0.35. Audit
  `styxx.anchors.audit_panel` (n_boot 300, null_sims 200). Crash-safe per-(model,seed) verdict
  cache; assembly + audit is a separate deterministic step.

## frozen gates

- **X1 (deaf reality):** deaf VOID >= 11/12 on both arms.
- **X2-KILL (blatant, the transfer claim):** among ESTIMATED replicates, coverage <= 3/12 --
  gold anchors do not license a heterogeneous panel either. (>= 8 ESTIMATED required or the arm
  is VOID_UNDERPOWERED, reported as such.)
- **X3-LADDER (the corrective claim, one gate, two admissible passes):** ladder anchors do the
  RIGHT thing, defined in advance as EITHER (a) coverage >= 9/12 among ESTIMATED with the kept
  mask excluding the two kill judges (1.5B, 3B) on a majority of replicates -- the ladder found
  the good judge; OR (b) VOID_PANEL__uninformative on >= 9/12 -- the ladder refused a panel it
  could not certify. Anything else (confident wrong coverage, or keeping the kill judges AND
  reporting a covering interval) FAILS: that would be the ladder certifying garbage.

Characteristics unbarred: per-model anchor-vs-organic alpha gaps (blatant and ladder), kept
masks per arm, misfit flags, which judges survive each gate, comparator errors, and the 7B-only
audit (restricting to the one competent judge) as a reference oracle. A missed gate is
CLOSED_NEGATIVE verbatim. Smoke to *_SMOKE_INVALID* only; no bar moves after the first scored
token. Diagnostics used seed 9996, disjoint from the scored seeds.
