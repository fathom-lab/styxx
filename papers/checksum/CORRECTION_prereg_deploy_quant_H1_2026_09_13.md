# CORRECTION — PREREG_checksum_deploy_quant_2026_09_13, hypothesis H1: what "A vs A′ reads SAME" can read, stated before the run

**status: a reading rule for a frozen document, written before any data and to be sealed beside
it (anchor 3b in `SEALS_2026_09_13.md`). The PREREG is not edited. The rule is sworn to a
committed, deterministic probe; nothing here touches a model.**

## the construction

H1 as frozen: "The null floor on GPU is > 0 and ≤ 1e-3 nats/token. A vs A′ reads SAME." The
design freezes the floor as the worst pairwise distance among three loads of the same weights,
A, A′ and A″. `styxx.checksum.distance` grades SAME only when the bootstrap upper bound of the
graded pair's mean |Δ log-prob| lies below the effective floor, which is the measured floor or
1e-4, whichever is larger. The graded pair, A–A′, is one of the three pairs the floor is the
maximum of. When A–A′ is itself the worst pair, its mean *is* the floor and its upper bound
exceeds it, so SAME is impossible; when it is not, SAME needs the pair's whole interval to sit
under another pair's mean. The clause was written as if the floor were held out from the pair it
grades. It is not, and the PREREG's first clause — a floor above zero — is exactly the condition
under which the second clause is starved.

## the probe

`probe_h1_by_construction.py` draws three synthetic same-weights fingerprints with a fixed jitter
at three scales, <sworn r="path:papers/checksum/probe_h1_by_construction_result.json#/n_trials_per_jitter" k="numeric">30</sworn> trials per scale, fixed seeds, and grades A vs A′ against the worst
pairwise floor exactly as the runner does. At the scale where the floor sits inside H1's band,
A vs A′ read SAME in <sworn r="path:papers/checksum/probe_h1_by_construction_result.json#/jitters/0.0003/verdicts_A_vs_Aprime/SAME" k="numeric">6</sworn> trials and INCONCLUSIVE in <sworn r="path:papers/checksum/probe_h1_by_construction_result.json#/jitters/0.0003/verdicts_A_vs_Aprime/INCONCLUSIVE" k="numeric">24</sworn>, and the graded pair was the worst pair in <sworn r="path:papers/checksum/probe_h1_by_construction_result.json#/jitters/0.0003/trials_where_the_graded_pair_is_the_worst" k="numeric">10</sworn>. At the next scale, SAME in <sworn r="path:papers/checksum/probe_h1_by_construction_result.json#/jitters/0.001/verdicts_A_vs_Aprime/SAME" k="numeric">3</sworn> and INCONCLUSIVE in <sworn r="path:papers/checksum/probe_h1_by_construction_result.json#/jitters/0.001/verdicts_A_vs_Aprime/INCONCLUSIVE" k="numeric">27</sworn>; at the largest, SAME in <sworn r="path:papers/checksum/probe_h1_by_construction_result.json#/jitters/0.005/verdicts_A_vs_Aprime/SAME" k="numeric">3</sworn> and INCONCLUSIVE in <sworn r="path:papers/checksum/probe_h1_by_construction_result.json#/jitters/0.005/verdicts_A_vs_Aprime/INCONCLUSIVE" k="numeric">27</sworn>. DRIFT never: the weights are the same. An instrument that reads its own null pair as INCONCLUSIVE four times in five is not broken; the hypothesis asked it a question it cannot answer in the affirmative.

## the reading rule, fixed before the run

1. H1's first clause — the measured floor is above zero and at most 1e-3 nats/token — is
   evaluated as frozen, from `sanity.null_floor_nats` in the certs.
2. H1's second clause — A vs A′ reads SAME — is reported exactly as `distance` computes it
   against the worst-pairwise floor, as frozen. If it reads INCONCLUSIVE while the first clause
   holds, the RESULT records **H1: HELD on the floor, INCONCLUSIVE on the pair by construction**,
   citing this correction. That outcome is neither HELD nor FAILED for H1 as written, and the
   RESULT says so in those words rather than choosing one.
3. Beside it, the runner writes `h1_held_out`: A vs A′ graded against a floor taken from the two
   pairs that do not contain A′'s partner — the larger of A–A″ and A′–A″ — which is what the
   clause meant. It is reported as an unpreregistered reading, labelled as such, and decides
   nothing in this run.
4. K1 (floor above 1e-2, the run INCONCLUSIVE) is unchanged and is now evaluated in code.
5. The next PREREG on this instrument freezes the held-out form and does not carry this clause.

## what this does not change

The predicted bands for H2, H3 and H4 and the gates K2, K3 and K4 stand as frozen. Nothing in this
correction reads a number from the run, and the run has not started; the seal of this file is
what makes that checkable.
