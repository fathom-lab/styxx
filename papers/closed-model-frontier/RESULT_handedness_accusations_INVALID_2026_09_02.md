# RESULT — token-level h on the blind panel: INVALID, the re-derivation diverged — 2026-09-02

Fathom Lab · 2026-09-02 · Frozen by `PREREG_handedness_accusations_2026_09_02.md`. Runner:
`handedness_accusations.py`. Receipt: `handedness_accusations_result.json`, scored through
`styxx.protocol`. **The verdict is the plumbing gate's, and it ships as what it is.**
<sworn r="path:papers/closed-model-frontier/handedness_accusations_result.json#/verdict" k="quote">The frozen verdict reads `INVALID__rederivation_diverged`.</sworn>

## What tripped

The preregistration re-derived each accused token's obligation source from the harness ledger —
the recorded trigger words, the 200-character context, the column, the value — in the verifier's
clause order, and required that 95% of accused tokens reach a named clause, because an accusation
cannot exist without an obligation. <sworn r="path:papers/closed-model-frontier/handedness_accusations_result.json#/metrics/unknown_or_ambiguous_share" k="numeric">The share that reached none was 0.5027</sworn>:
<sworn r="path:papers/closed-model-frontier/handedness_accusations_result.json#/by_source/unknown/n" k="numeric">167 accused tokens</sworn> reached no clause and
<sworn r="path:papers/closed-model-frontier/handedness_accusations_result.json#/ambiguous" k="numeric">17 more</sworn> could not be joined to a single ledger row —
<sworn r="path:papers/closed-model-frontier/handedness_accusations_result.json#/unknown" k="numeric">184 in all</sworn>.
<sworn r="path:papers/closed-model-frontier/handedness_accusations_result.json#/accusations_joined" k="numeric">All 366 accusations were joined</sworn>; half of them are invisible
to the re-derivation.

## Why, as a hypothesis this document does not test

The verifier obligates a markdown table cell through its column *header* — v0.3's
`binding_context`, "table rows bind via their header too" — while the harness recorded
`obligating_words` from the row's own line. README benchmark tables are exactly where an
external corpus accuses: a cell under an *Accuracy* or *Score* header, on a line that carries no
trigger word of its own. The re-derivation read the line; the verifier read the header. That is
the divergence the gate was written to catch, and it caught it. The hypothesis is untested here
because the documents are not in the tree: the harness's fetch cache lived in a temporary
directory, and a re-derivation from a ledger is not a re-certification.

## Exploratory, under an INVALID verdict, and labelled so

Nothing below is a finding. It is what the receipt shows, printed because hiding it would be
the silent omission this lab names as a defect.
<sworn r="path:papers/closed-model-frontier/handedness_accusations_result.json#/by_source/vocabulary/genuine_share" k="numeric">Among tokens the line's own trigger words obligated, the genuine-claim share was 0.6391</sworn>
(<sworn r="path:papers/closed-model-frontier/handedness_accusations_result.json#/by_source/vocabulary/n" k="numeric">n=169</sworn>). <sworn r="path:papers/closed-model-frontier/handedness_accusations_result.json#/by_source/range-sanity/genuine_share" k="numeric">Among range-sanity accusations the genuine-claim share was 0.0</sworn>
(<sworn r="path:papers/closed-model-frontier/handedness_accusations_result.json#/by_source/range-sanity/n" k="numeric">n=13</sworn>): every out-of-range accusation the panel saw was
on something that was not a claim. <sworn r="path:papers/closed-model-frontier/handedness_accusations_result.json#/by_source/unknown/genuine_share" k="numeric">Among the tokens no clause reached, the genuine-claim share was 0.9521</sworn>
— the half the re-derivation could not see is the half the panel most often agreed with. The
object_form cell <sworn r="path:papers/closed-model-frontier/handedness_accusations_result.json#/by_class/object_form/n" k="numeric">holds 0 tokens</sworn>: no accused token in this
corpus printed seven fractional digits, so the hypothesis as formulated could not have been
tested on this panel even with perfect plumbing, and the power gate would have been the next to
fail. That is a finding about the preregistration's design, and it is recorded rather than
repaired in place.

## What follows

The correct v2 is not a looser re-derivation. It is the documents: the corpus manifest pins
every file to a commit and a sha256, raw GitHub serves them, and a hash-verified rebuild
re-certified at the current verifier gives every token its exact `obligation_source` from the
verifier itself, with no re-derivation to diverge. And the hypothesis needs a cell to live in:
with `precision` empty on accusations, token-level h on this corpus is a question about the
object_text sources among themselves — vocabulary against range-sanity against n-glued — which
the exploratory numbers above suggest is where the separation actually is.

---

*The gate that failed was the one written to fail first. The panel's answer is still there,
under half the tokens the ledger could not name.*
