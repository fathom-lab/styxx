# NOTE — the core's summaries do not diverge from its spans (negative result)

*2026-09-06. Records a hypothesis that was tested and failed. Nothing here is a defect report.*

## The hypothesis

Several findings in the sidecar-battery leg looked like one shape: **a summary field and a detail
field that can disagree, with nothing checking the summary is faithful to its details.** If that
shape were general, the verdict core would be the place to find more of it — it carries five
summaries over its span table (`counts`, `sworn_total`, `unresolved`, `document_verdict`, `rungs`),
and every reader acts on those rather than on the spans underneath.

Stated as a check: for each summary field, recompute it from `core["spans"]` alone and compare.

## What was run

The differential harness's seeded grammar, seed `20260906`, indices 0–3999. Each generated document
was verified; documents the verifier refuses have no summaries and were skipped. Every summary field
of every resulting core was recomputed from its spans and compared.

    documents checked: 4000 of 4000 generated
    every summary field follows from its spans, on all 4000.

## The result

**Negative. No divergence in any field, in any of 4000 cases.** The hypothesis is wrong in the form
it was stated: the class of defect it predicted does not exist inside the verdict core.

## What that corrects

The generalisation was too broad, and the corrected reading is narrower. Re-examined, the leg's
actual findings were not summaries diverging from details:

| finding | what it actually was |
| --- | --- |
| rounding: `receipt=0.4211`, `receipt_rounded=0`, verdict HELD | the verdict is *faithful* to its comparison — `detail` honestly records that it compared against 0. The question is whether that comparison should be made, which is a semantics decision, still operator-gated. |
| `check` printing VERIFIED on a SWORN-FAILED document | CLI presentation. The receipt re-derives; the document did not hold. Two different questions, one word. |
| the `load_sidecar` / `render` gap | across two artifacts — one validated, a different one consumed. |

The shape those share is **artifact A validated, artifact B consumed**, not a broken summary
function. That is a narrower and more useful description, and it is the one to carry forward.

## What was kept

`tests/test_sworn_core_is_faithful_to_its_spans.py`, which pins the property as a standing check
over 1500 generated documents. It is kept because nothing else states the property, **not** because
it caught anything — it has never caught a real defect and may never.

Its claim is deliberately modest, and the test says so in its own docstring: it checks that the
summaries agree with the span table shipped in the same core. It does not independently verify the
adjudication rules — `_recompute` restates the rules in `verify`, so if both are wrong in the same
way it passes.

Watched to fail before it was kept, against two mutations of `styxx/sworn.py`:

| mutation | result |
| --- | --- |
| `"spans": verdicts[1:]` — summaries tallied from a list the reader is not given | 5 of 5 fields fail |
| ladder drops `counts["MALFORMED"] == 0`, so a malformed span no longer fails the document | `document_verdict` alone fails; the other 4 pass |

The second is the more useful demonstration: the guard fires on exactly the field that broke, so it
localises the defect rather than only reporting that something moved.

## Standing

Negative results are cheap to omit and expensive to lose — the omission is what makes a corpus look
like it only ever confirms its author. This one is recorded at the same prominence as a finding.
