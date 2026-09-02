# RESULT — OATH v0.14: range-sanity should report, not accuse — SHIP — 2026-09-02

Fathom Lab · 2026-09-02 · Frozen by `PREREG_range_sanity_report_2026_09_02.md`, committed with
the flag before the A/B ran. Runner: `range_sanity_report_ab.py`. Receipt:
`range_sanity_report_ab_result.json`, scored through `styxx.protocol`. Every number below is
sworn to the receipt at commit `48d4d5494bc8`. **The flag ships OFF; this RESULT recommends, it does
not flip.** <sworn r="path:papers/closed-model-frontier/range_sanity_report_ab_result.json#/verdict" k="quote">The frozen verdict reads `SHIP__range_sanity_reports`.</sworn>

## The gates

**Internal regression.** <sworn r="path:papers/closed-model-frontier/range_sanity_report_ab_result.json#/internal/documents" k="numeric">207 committed documents</sworn> and
<sworn r="path:papers/closed-model-frontier/range_sanity_report_ab_result.json#/internal/tokens" k="numeric">8583 ledger tokens</sworn> were certified with the flag off and on:
<sworn r="path:papers/closed-model-frontier/range_sanity_report_ab_result.json#/internal/tokens_moved" k="numeric">0 tokens moved</sworn>, and
<sworn r="path:papers/closed-model-frontier/range_sanity_report_ab_result.json#/metrics/internal_held_to_failed" k="numeric">0 documents moved from HELD to FAILED</sworn>. The
range-sanity rule never fires on this lab's own prose: every out-of-range token here was already
obligated by vocabulary and already carried a receipt, or was never a token at all. The rule's
whole effect lives outside the corpus it was written for.

**What the flag removes.** On the rebuilt external corpus,
<sworn r="path:papers/closed-model-frontier/range_sanity_report_ab_result.json#/external/accusations_removed" k="numeric">13 accusations were removed</sworn>, every one to ABSTAIN with
its out-of-range flag on the ledger, and <sworn r="path:papers/closed-model-frontier/range_sanity_report_ab_result.json#/metrics/removed_genuine_share" k="numeric">the share of them the panel had judged genuine claims was 0.0</sworn>.
The bar was a quarter.

**The gain.** <sworn r="path:papers/closed-model-frontier/range_sanity_report_ab_result.json#/external/accusations_remaining" k="numeric">353 accusations remain</sworn>, of which
<sworn r="path:papers/closed-model-frontier/range_sanity_report_ab_result.json#/external/remaining_false_by_panel" k="numeric">82 the panel judged not claims</sworn>:
<sworn r="path:papers/closed-model-frontier/range_sanity_report_ab_result.json#/false_accusation_rate_after" k="numeric">a false-accusation rate of 0.2323</sworn> against the
2026-08-27 rate of 0.2596, <sworn r="path:papers/closed-model-frontier/range_sanity_report_ab_result.json#/metrics/false_accusation_rate_gain" k="numeric">a gain of 0.0273</sworn> on
a bar of a point.

## Reading

The v0.3 rule was right about this lab and wrong about the world: *precision 4.0* is a typo in a
RESULT and a version string, a count or a configuration value in a README. As an accuser outside
the lab's idiom it produced thirteen false accusations and not one catch; as a reporter it costs
nothing anywhere the corpus tests can see, because on the corpus that trained it the rule never
decides anything. This is the header-vs-line finding again, one clause down: a bounded word is a
handed target, and a handed target from free prose is a coin.

## What this does not say

The flag is OFF and stays OFF in this cycle. Flipping it is a corpus re-issue — every certificate
gains a field only under the flag, and the M7 addendum to the program audit says what a new field
does to a corpus — and that is the operator's release decision, with `OATH_CONTRACT.md` owed a
sentence when it lands. The gain is measured on one panel of one model family, on 82 READMEs
drawn by seven queries with one repository holding half the accusations; the receipt prints its
weight. The prior was strong and contaminated, and the internal gate was the one that could have
killed it; it did not.

---

*A rule that never fires at home and only fires wrongly abroad is not a verifier's rule. It is a
report, and now it can be one.*
