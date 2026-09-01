# RESULT — obligation predicts claimhood, and the verified-channel defect lives in the volunteer channel

Fathom Lab · 2026-08-30 · Receipt: `oath_obligation_claimhood_join.json`. Joins
`RESULT_unobligated_oath_2026_08_28.md` with the 2026-08-27 blind panels
(`oath_internal_result.json`, `oath_adjudication_result.json`). **No new adjudication was
performed** — this is a recombination of committed evidence, which is the only reason it could be
produced in an afternoon.

## The question

The unobligated-oath census counted the split: `0.5811` of this corpus's verifications were
volunteered, not obligated. It could not say whether that matters for **quality**. The blind
panels had already adjudicated claimhood for `225` verified tokens — `150` internal, `75`
external — three seats each, decoy-blinded, majority verdicts. The epistemics annotation makes
each of those tokens taggable as `obligated` or `volunteered` by live re-certification at the
pinned verifier. Join them:

> Among panel-adjudicated VERIFIED tokens, is the claim-share of volunteered oaths lower than
> that of obligated oaths?

## The answer

| arm | obligated | volunteered |
|---|---|---|
| **internal** | `0.8472` claim-share (61 of 72) | `0.7403` (57 of 77) |
| **external** | `0.7826` (18 of 23) | **`0.3654` (19 of 52)** |

Unresolved: 1 internal token (document could not be re-certified; reported, not dropped),
0 external. Cross-checks are exact: the external cells recompose the arm's published overall
claim-share to the digit — `(18+19)/75 = 0.4933` — and the internal cells recompose `118/149`
against the published `119/150` with the single unresolved token accounting for the difference.

**Obligated oaths travel. Volunteered oaths do not.** On foreign text an oath the obligation
predicate required is right about claimhood at nearly the internal rate — `0.78` against `0.85`.
An oath the value-match volunteered collapses to `0.37`: **most volunteered oaths on external
text are sworn to things that are not claims** — the `gpu_memory_fraction` class, measured rather
than exemplified.

## What this reframes

**The 2026-08-27 sanity-gate failure was mis-attributed.** The internal RESULT read the external
verified channel's `0.4933` as "the panel and the instrument disagree about what a claim is."
Decomposed, the panel agrees with the *obligated* instrument just fine (`0.78`). What it rejects
is the channel where the instrument's judgement was never consulted. The defect is not in the
obligation predicate's taste — it is that **the affirmative attestation ignores the predicate
entirely**, and the further the text is from this lab's idiom, the worse the volunteers behave.

**The repair direction inverts, again.** Three days ago the top repair was "obligate more" (killed:
added triggers only manufacture accusations). The join says the leverage is the opposite move:
**the verified count should be scoped or split by obligation.** Gating verification on obligation
would cost real bindings — `74%` of internal volunteers are genuine claims — so an outright gate
fails the same test the v0.13 gates did. But *reporting* the split costs nothing and converts the
weakest 34% of the count from hidden to labelled. That is exactly the certificate-schema change
already in design.

## Limits, stated before anyone else states them

* **Small cells.** External obligated is `n=23`; no interval or significance is asserted anywhere
  in this document, and the direction — not the digits — is the finding.
* **The panel is three seats of one model family.** Correlated error is the ceiling, as disclosed
  in both source panels. A verdict this consequential deserves eventual human re-adjudication of
  the 52-token external volunteer cell above all.
* **The internal sample was drawn uniformly from verified tokens**, so its 77/149 volunteer mix is
  an independent estimate of the census's `0.5811` (sampling, not contradiction).
* **The external join depends on the frozen fetch cache** at the pinned shas; every blob was
  hash-verified before use, and a missing or corrupt blob reports the token unresolved rather
  than substituting a live fetch.

## What is owed

The certificate-level split (in design, red-teamed before freezing). The contract's coverage
section gains one sentence when that lands: the verified count's volunteered share is not merely
unscoped — externally, it is **majority non-claims**.

---

*Two measurements, both already paid for, one join. The instrument's judgement was fine. The
problem was every oath it never got to judge.*
