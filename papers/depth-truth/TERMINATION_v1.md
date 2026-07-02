# TERMINATION — keystone depth-vs-truth, PREREG v1

**Fathom Lab · 2026-07-01 · papers/depth-truth/ · this is an asset, not an embarrassment.**

The v1 pilot ran to completion and was **terminated at the §9 checkpoint by design.** No main run fired. No
frozen section was touched. The instrument was not the problem — the plumbing was — and the discipline caught
it in **28 minutes** for the price of 20 quarantined items.

## What the pilot revealed

The frozen 5-shot prompt (Appendix A) did not constrain the *format* of base gemma-2-2b's answers. The model
knew the answers and then wrapped them — as a numbered list (`"1. …"`) or in HTML (`"<strong>…</strong>"`).
That single format drift broke the experiment in two places at once:

1. **It stole the A1 target.** The metric attributes the *first token* of the answer. When the answer begins
   `"1. "` or `"<strong>"`, the first token is `'1'` or `'<strong>'` — so depth measured the reasoning behind a
   *list bullet or an HTML tag*, not behind the answer. Across all 20 items the A1 target was a formatting token;
   zero were content. The near-constant depth (mean 8.76, std 0.068) is the fingerprint of measuring the same
   formatting operation twenty times.
2. **It defeated the grader.** Exact-alias match cannot see through the wrapper, so **correct answers scored
   `False`.** The receipts:

```
Q: who wrote "The Wonderful Wizard of Oz"?     model: "1. L. Frank Baum"          gold: "L Frank Baum"     -> correct=False, A1 target='1'
Q: which sitcom star ... Object of My Affection? model: "<strong>Jennifer Aniston</strong>" gold: "Jennifer Aniston" -> correct=False, A1 target='<strong>'
Q: Aconcagua stands in which country?           model: "1. Argentina"              gold: "ARGENTINA"        -> correct=False, A1 target='1'
```

Three right answers, three `False` grades, three junk targets. `correct = 0/20` was not the model failing to
know — it was the pipeline failing to *read*.

## What the rules did

- **KG1 did not fire.** Depth was defined on 20/20 and varied — the instrument produces a signal on short
  answers. That is the load-bearing positive: the metric survives this regime; only the plumbing was wrong.
- **The §9 halt held.** The auto-fire monitor ran the pilot **once** and stopped. Nothing chained to the main
  run. The pilot's job is to be terminal until a human reads it — and it was.
- **No amendment laundering.** The flaw was tempting to patch as an "A1 adaptation." It was refused: the failure
  spans three *frozen* sections at once (the Appendix-A prompt, the §3 grading, the §1 target), and the amendment
  door (A0 sizes, A1 span) is not a patch channel. A fix that reaches into frozen text is a **new prereg**, not
  an amendment.
- **v1 stays frozen forever.** `PREREG.md` (v1) is untouched and remains the record. This file is the honest
  epitaph; `PREREG_v2.md` supersedes, it does not edit.

## Commit trail (re-runnable)

`c2099f8` design freeze · `295075f` harness (adversarially audited, 21 synthetic tests) · `fa15ed9` pilot runner
+ auto-fire monitor · `66195da` pilot receipts (`pilot/pilot_results.jsonl`, `pilot/pilot_timing_report.json`).
(Between fire and receipts, three untested-path bugs were debugged live and committed: the `datasets.py`→
`qa_data.py` HF-shadow, the first-word→first-*token* A1 target, and freeing the card from a hung sibling sweep.)

## Why v2 exists instead of a rescue

Because the honest thing to do with a design that cannot test its hypothesis is to say so, publish the reason,
and rebuild the part that failed — not to reach through a frozen wall and call it an adjustment. The pilot did
exactly what a pilot is for. The instrument has a pulse. v2 fixes the plumbing.

*Nothing crosses unseen. The bar structure outranks the dream.*
