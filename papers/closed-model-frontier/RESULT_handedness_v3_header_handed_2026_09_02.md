# RESULT — token-level h: an accusation handed by a table header is true; one handed by the line is a coin with a thumb on it — 2026-09-02

Fathom Lab · 2026-09-02 · Frozen by `PREREG_handedness_v3_join_rule_2026_09_02.md`, committed
before the join was computed. Receipt: `handedness_v3_result.json`, scored through
`styxx.protocol`. Panel: the 2026-08-27 external adjudication, three seats, decoy-gated, 366
accusations; **no token was re-judged**. Corpus: rebuilt from pinned shas and re-certified at the
current verifier (`oath_external_recertify_summary.json`), so every token's obligation source is
the verifier's own. Every number below is sworn to a receipt at commit `b1a6326ef35b`.
<sworn r="path:papers/closed-model-frontier/handedness_v3_result.json#/verdict" k="quote">The frozen verdict reads `HEADER_HANDED_ACCUSES_TRUER`.</sworn>

## The gates

<sworn r="path:papers/closed-model-frontier/handedness_v3_result.json#/metrics/unresolved_share" k="numeric">The unresolved share was 0.0464</sworn> under the frozen 0.05
(<sworn r="path:papers/closed-model-frontier/handedness_v3_result.json#/unresolved" k="numeric">17 accusations</sworn> joined to no row or to several accused rows).
<sworn r="path:papers/closed-model-frontier/handedness_v3_result.json#/metrics/min_cell_n" k="numeric">The smaller cell held 165 accusations</sworn>, over the floor of 20.
<sworn r="path:papers/closed-model-frontier/handedness_v3_result.json#/metrics/delta_header_minus_line" k="numeric">The header cell's genuine share exceeded the line cell's by 0.3124</sworn>,
twice the frozen 0.15.

## The finding

<sworn r="path:papers/closed-model-frontier/handedness_v3_result.json#/cells/header/genuine_share" k="numeric">Of the accusations the verifier was handed by a table header, 0.9515 were genuine claims</sworn>
(<sworn r="path:papers/closed-model-frontier/handedness_v3_result.json#/cells/header/n" k="numeric">n=165</sworn>). <sworn r="path:papers/closed-model-frontier/handedness_v3_result.json#/cells/line/genuine_share" k="numeric">Of those it was handed by a trigger word in the line's own prose, 0.6391 were</sworn>
(<sworn r="path:papers/closed-model-frontier/handedness_v3_result.json#/cells/line/n" k="numeric">n=169</sworn>). Both are object_text under the handedness declaration; the
mapping is right that neither is found. But they are not the same handing. A column header is a
label the author chose for a column of numbers — *Accuracy*, *Score* — and a number under it is
a reported quantity almost by construction. A trigger word on a prose line is a co-occurrence:
the word *rate* two clauses away from a number that is a version, a count, a date. That is
mention-versus-use (M1) living inside handed-target (M2): the verifier's false accusations on
foreign text are, to first order, the line-handed ones. The join RESULT of 2026-08-30 found the
volunteered oaths were the weak half of verification; this finds the line-handed accusations are
the weak half of accusation, on the same panel.

Reported, not gated: <sworn r="path:papers/closed-model-frontier/handedness_v3_result.json#/cells/range-sanity/genuine_share" k="numeric">range-sanity accusations were genuine at 0.0</sworn>
(<sworn r="path:papers/closed-model-frontier/handedness_v3_result.json#/cells/range-sanity/n" k="numeric">n=13</sworn>; Wilson upper bound 0.228). The v0.3 out-of-range rule
— *an AUC of 4.0 cannot be* — accused thirteen tokens on foreign READMEs and the panel called
every one not a claim. On this corpus that rule is a false-accusation generator, and it is the
smallest obligation source in the tree; a preregistration that reads it as a report rather than
an accusation outside the lab's own idiom is owed.

## The corpus, as rebuilt

<sworn r="path:papers/closed-model-frontier/oath_external_recertify_summary.json#/repos_recertified" k="numeric">82 repositories re-certified</sworn> from
<sworn r="path:papers/closed-model-frontier/oath_external_recertify_summary.json#/files/fetched" k="numeric">244 pinned files</sworn>, every byte hash-verified against the manifest,
<sworn r="path:papers/closed-model-frontier/oath_external_recertify_summary.json#/tokens" k="numeric">5179 tokens</sworn> — the 2026-08-27 count to the digit — and
<sworn r="path:papers/closed-model-frontier/oath_external_recertify_summary.json#/header_bound_obligations" k="numeric">270 obligations that came through a table header</sworn>. The
2026-08-27 harness ledger recorded trigger words from the row alone, which is why the first
attempt at this study diverged on half the accusations: the corpus did not move; the ledger had
never seen the headers.

## What this does not say

One panel of three seats of one model family, with correlated error as the ceiling both source
documents disclose. One repository holds 194 of the 366 accusations, and its weight in each cell
is printed in the receipt rather than hidden. The hypothesis was frozen after two INVALID runs
whose exploratory numbers pointed here; the prior was contaminated and declared so in the
preregistration, and the bar did not move. No claim about the object_form class: no accused
token on this corpus printed seven fractional digits, so the question the brief asked — form
against text — has no cell here, and the question this corpus can answer is the one it
answered. Nothing about verification: this is accusation only.

---

*Handed is not one thing. A label the author chose hands the verifier a claim; a word that
happened to be nearby hands it a coin. The mapping said object_text; the panel says the text has
a grain.*
