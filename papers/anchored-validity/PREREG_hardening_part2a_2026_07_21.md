# PREREG — hardening part 2a: the chain family, and a frontier panel under load

date: 2026-07-21
status: FROZEN before any scored inference; committed with the corpus extension.
operator authorization: this session ("go").

## resource block, recorded first

The deployed-API-judge arm (cross-vendor panels) is BLOCKED ON CREDITS: the OpenAI key
answers 401 (the standing top-up unblocks it, per the ledger), and the Gemini free tier's
rate limits cannot carry a 15-replicate scored arm and were not attempted for one. The
model-generality claim therefore remains OPEN at the API tier; this prereg runs what is
executable tonight and says so.

## the chain family (task family four, with a difficulty dial)

Transitivity chains: L people in a strict height order stated pairwise with mixed phrasing
direction. Contradictions and consistents at the hard end use the chain ENDPOINTS (L-1 hops
of transitive inference); medium uses an adjacent pair (1 hop); blatant anchors are verbatim
pairs and 1-hop reversals. `chain` = L in 3-5; `chain_long` = L in 6-8. Bring-up diagnostics
(seed 9997, non-scored): the 3B panel fires on 0.886-0.971 of consistent organics and 0.45 of
VERBATIM pairs (logician), anchor informativeness 0.55 for the logician, near-zero for
casual — the kill configuration exists and the panel is even sicker here than on attr.

## frozen arms and gates

- **(a) chain_blatant_3B** — seeds 8001-8015, R=15, n_organic 240, K 80, pi 0.35, identical
  panel/phrasing to rung 1. P2-K (prediction): coverage <= 3/15. Gate: deaf VOID >= 14/15.
- **(b) chain_ladder_3B** — seeds 8501-8515, same design, ladder anchors. H2-style gates
  carried verbatim from part 1 (coverage >= 12/15 among ESTIMATED with >= 8 ESTIMATED, AND
  max |mean delta_alpha| <= 0.10), with the refusal-resolution note carried verbatim: honest
  anchors may VOID a panel this sick, which scores VOID_UNDERPOWERED and is reported as the
  instrument declining to certify — on the diagnostics that outcome is LIKELY here, and it is
  recorded as such before the run.
- **(c) chain_long_claude** — the frontier-stress arm. Seed 9001, single run
  (demonstration-grade, token budget; one draw is not a property), n_organic 200, K 60 per
  stratum, hard_frac 1.0 (endpoint queries only, 5-7 hops), BLATANT anchors, four
  fresh-context Claude persona subagents, arm's-length exactly per the rung-2 protocol
  including the transcription trust boundary. Purpose: does a frontier panel under genuine
  inferential load exhibit organic error that blatant anchors cannot see? Every outcome is
  reported verbatim: a perfect panel (ceiling again), a kill-on-frontier (organic error with
  clean anchors, the strongest possible form of the claim), or a refusal.

Characteristics unbarred everywhere: delta_alpha/beta, misfit flag rates, comparators, kept
patterns, per-hop error rates on (c) if errors exist. Smoke only to *_SMOKE_INVALID*; missed
bars CLOSED_NEGATIVE verbatim; no bar moves after the first scored token.
