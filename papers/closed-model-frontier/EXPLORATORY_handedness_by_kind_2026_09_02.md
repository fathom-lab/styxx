# EXPLORATORY — the token-kind split of the handedness v3 rows: a receipt for a number that circulated without one — 2026-09-02

Fathom Lab · 2026-09-02 · **Exploratory, not a result.** Not preregistered; computed after the v3
verdict was read, on the rows that produced it, by the lab that produced them. It may never be
quoted as a finding. The only sentence it licenses is that the split now exists as a committed
table, so the next preregistration in this lane can declare it as a contaminated prior by digest
rather than from memory. Script: `handedness_v3_by_kind.py`. Receipt:
`handedness_v3_by_kind_result.json`. Every number below is sworn to it at the commit this
document names.

## Why this exists

`RESULT_handedness_v3_header_handed_2026_09_02.md` reports header-handed accusations genuine at
0.9515 against line-handed at 0.6391 and reads the gap as structure. A same-day objection said the
gap is mostly token *kind* — the header cell is mostly decimals, and a decimal is a claim whoever
hands it — and that objection circulated with a number attached, a kind-adjusted gap of 0.117,
and **no receipt**: no stratified file exists anywhere in this repository at `320b303`, on any
branch, and neither the RESULT, the grain synthesis nor the changelog carries the number. The
objection was right in shape and wrong in the digit, and both facts needed a receipt.

## The table

<sworn r="path:papers/closed-model-frontier/handedness_v3_by_kind_result.json#/source_verdict" k="quote">The source receipt's frozen verdict is `HEADER_HANDED_ACCUSES_TRUER`.</sworn>
<sworn r="path:papers/closed-model-frontier/handedness_v3_by_kind_result.json#/rows_considered" k="numeric">The header and line cells together hold 334 panel-judged rows.</sworn>
A token is DECIMAL when it prints a point and INTEGER otherwise; genuine means the panel's label
was not in the not-a-claim family.

| cell | n | genuine share |
|---|---|---|
| header / decimal | <sworn r="path:papers/closed-model-frontier/handedness_v3_by_kind_result.json#/cells/header~1decimal/n" k="numeric">142</sworn> | <sworn r="path:papers/closed-model-frontier/handedness_v3_by_kind_result.json#/cells/header~1decimal/genuine_share" k="numeric">1.0</sworn> |
| header / integer | <sworn r="path:papers/closed-model-frontier/handedness_v3_by_kind_result.json#/cells/header~1integer/n" k="numeric">23</sworn> | <sworn r="path:papers/closed-model-frontier/handedness_v3_by_kind_result.json#/cells/header~1integer/genuine_share" k="numeric">0.6522</sworn> |
| line / decimal | <sworn r="path:papers/closed-model-frontier/handedness_v3_by_kind_result.json#/cells/line~1decimal/n" k="numeric">76</sworn> | <sworn r="path:papers/closed-model-frontier/handedness_v3_by_kind_result.json#/cells/line~1decimal/genuine_share" k="numeric">0.9605</sworn> |
| line / integer | <sworn r="path:papers/closed-model-frontier/handedness_v3_by_kind_result.json#/cells/line~1integer/n" k="numeric">93</sworn> | <sworn r="path:papers/closed-model-frontier/handedness_v3_by_kind_result.json#/cells/line~1integer/genuine_share" k="numeric">0.3763</sworn> |

<sworn r="path:papers/closed-model-frontier/handedness_v3_by_kind_result.json#/raw_gap_header_minus_line" k="numeric">The raw gap recomputed from the rows is 0.3124.</sworn>
<sworn r="path:papers/closed-model-frontier/handedness_v3_by_kind_result.json#/header_decimal_mix" k="numeric">The header cell is decimals at a share of 0.8606</sworn>
<sworn r="path:papers/closed-model-frontier/handedness_v3_by_kind_result.json#/line_decimal_mix" k="numeric">and the line cell at 0.4497.</sworn>
<sworn r="path:papers/closed-model-frontier/handedness_v3_by_kind_result.json#/kind_adjusted_header_share" k="numeric">Reweighting the header cell to the line cell's kind mix gives a header share of 0.8086</sworn>,
<sworn r="path:papers/closed-model-frontier/handedness_v3_by_kind_result.json#/kind_adjusted_gap" k="numeric">a kind-adjusted gap of 0.1695</sworn> — not the 0.117 that circulated.
<sworn r="path:papers/closed-model-frontier/handedness_v3_by_kind_result.json#/odds_ratio_header_over_line/integer_stratum" k="numeric">Within integers the odds ratio of header over line is 3.1071</sworn>;
<sworn r="path:papers/closed-model-frontier/handedness_v3_by_kind_result.json#/odds_ratio_header_over_line/decimal_stratum" k="numeric">within decimals it is 13.5714</sworn>,
a Haldane-corrected figure over a cell with no misses, which is why it is printed with the
interval in the receipt and not read here.
<sworn r="path:papers/closed-model-frontier/handedness_v3_by_kind_result.json#/top_repository/repo" k="quote">One repository, `hopit-ai/Moda`, dominates</sworn>:
<sworn r="path:papers/closed-model-frontier/handedness_v3_by_kind_result.json#/top_repository/rows" k="numeric">it supplies 184 of the rows</sworn>,
<sworn r="path:papers/closed-model-frontier/handedness_v3_by_kind_result.json#/top_repository/share_of_rows" k="numeric">a share of 0.5509.</sworn>

## What this table is and is not

It is a description of one panel's labels on one corpus, re-cut by a rule chosen after the labels
were read. It says the kind confound is real and large — decimals are genuine claims in both
cells — and that a header residue remains inside the integer stratum, wide-intervalled, on
twenty-three rows. It does not say the residue is structure: a header cell is also a table cell
in benchmark idiom, and the panel saw table rows as rows. It does not say anything measured
against a second model family or a second corpus. A preregistration that wants to test the
provenance reading must cite this file by digest as its contaminated prior, size its integer
cells above thirty, seat two families, and publish with and without the dominant repository.

---

*A number that circulates without a receipt is a rumour with decimals. This one was off by
five hundredths, and the receipt says so.*
