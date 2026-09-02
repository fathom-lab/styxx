# RESULT — token-level h, v2: INVALID, the join was incomplete — 2026-09-02

Fathom Lab · 2026-09-02 · Frozen by `PREREG_handedness_v2_header_bound_2026_09_02.md`. Receipt:
`handedness_v2_result.json`, scored through `styxx.protocol`. Corpus rebuilt exactly from pinned
shas by `oath_external_recertify.py` (`oath_external_recertify_summary.json`: every file
hash-verified, zero verdict-class drift). <sworn r="path:papers/closed-model-frontier/handedness_v2_result.json#/verdict" k="quote">The frozen verdict reads `INVALID__join_incomplete`.</sworn>

## What tripped

<sworn r="path:papers/closed-model-frontier/handedness_v2_result.json#/metrics/unresolved_share" k="numeric">The unresolved share was 0.0683</sworn>:
<sworn r="path:papers/closed-model-frontier/handedness_v2_result.json#/unresolved" k="numeric">25 accusations</sworn> join to zero or several ledger rows on the key the
panel receipt carries, (repository, line, token), and the preregistration allowed 0.05. The
allowance was frozen without accounting for what the verifier's own v0.10 note records — the same
token string occurs more than once on roughly a tenth of lines — and the panel receipt carries no
column to break the tie. A rule that could not have been met on this corpus is a defect in the
preregistration, not in the corpus, and it is shipped as INVALID.

## Exploratory, under an INVALID verdict, and labelled so

<sworn r="path:papers/closed-model-frontier/handedness_v2_result.json#/cells/header/genuine_share" k="numeric">Among header-handed accusations the genuine-claim share was 0.9515</sworn>
(<sworn r="path:papers/closed-model-frontier/handedness_v2_result.json#/cells/header/n" k="numeric">n=165</sworn>);
<sworn r="path:papers/closed-model-frontier/handedness_v2_result.json#/cells/line/genuine_share" k="numeric">among line-handed accusations it was 0.6391</sworn>
(<sworn r="path:papers/closed-model-frontier/handedness_v2_result.json#/cells/line/n" k="numeric">n=169</sworn>); <sworn r="path:papers/closed-model-frontier/handedness_v2_result.json#/metrics/delta_header_minus_line" k="numeric">a difference of 0.3124</sworn>,
twice the frozen bar, on the cells the join did resolve. Nothing here is a finding; the gate says
so, and the next document is the one that decides.

## What follows

v3 freezes the one rule v2 lacked: a token joining several rows takes the single row the verifier
accuses, because the accused token is the one the panel judged. Bars unchanged. The prior is now
contaminated twice and declared twice.

---

*Two plumbing gates in a row, each written to fail first, each failing first. The data have been
saying one thing throughout; the rule that lets them say it is the next one frozen.*
