# DRAFT ONLY — do not send until: (1) flobi seals KG_HUMAN_seal_table.md (18 rows), (2) flobi reads this once.
# channel: telegram first ("results come here first"), then X thread. fathom voice, numbers-first.

---

we spent a week pointing our instruments at the things nobody audits. receipts for every number below
are committed at github.com/fathom-lab/styxx.

**1 — model cards.** we took a frontier model card (llama-3.2-1b-instruct) and asked one question: do its
201 published benchmark numbers bind to re-runnable receipts? answer: 0 of 201. then the twist — meta's raw
eval receipts EXIST as a public dataset the card never links. so we re-bound the card ourselves, third-party,
no vendor help: **18 of 22 in-scope claims reproduce to the decimal.** the industry's provenance gap is
mostly a missing hyperlink. the residue is why the link should be mandatory: one benchmark value matches
nothing in the vendor's own receipts, and the card disagrees with its receipts about the eval config.

base-model cards fully re-bind (14/14 across two models). gemma's card disagrees with its own linked tech
report in 15 of 17 rows — direction: the card publishes the LOWER number. not cherry-picking; two snapshots
published as one truth. and cards publish ~1.8% of what vendors actually measure (22 of 1,246 metric rows).

**2 — the ground truth itself.** benchmark grading is an auditor too, so we audited it: 382 answers from a
frozen run, blind re-judged (judges saw only question + answer), every disagreement confirmed by a second
independent blind judge. result: **mechanical grading falsely fails 12.6% of triviaqa answers it marks wrong**
(wilson 95% [7.5, 20.4]) — answers like "apollo 11", "austerlitz", "tumbrils". a schoolteacher would mark them
right; the alias list can't see them. gold lists also credit wrong answers ("cursor" for the mouse patent).
popqa's problem is different: its kb-derived questions under-determine their referent ("who composed symphony
no. 3?" — the gold means nielsen; everyone hears beethoven). and truthfulqa-gen? **96.8% of items were
mechanically ungradeable by its own matching path** on real model output.

every mechanically-scored qa number you've ever cited is a deflated lower bound with a phrasing-dependent
bias term — often larger than the model-vs-model gap being claimed.

**3 — ourselves.** same week, same instruments, pointed inward: our binder caught its own author citing an
unsourced number. our rigor gate flagged our own verdict wording. and our study's confirmation stage was
voided and re-run because the operator typed judge inputs from memory instead of reading the file — caught by
a row-level autopsy before publication, disclosed in full, and the one "surprising" result it had manufactured
was retracted. the corrected finding is smaller and true.

we also killed our own biggest hypothesis this week: circuit-attribution depth does not predict answer
correctness (pre-registered, 633 items, all three hypotheses null, published at full size).

nothing crosses unseen — including our claims, their claims, the benchmarks, and us.

reproduce: papers/oath-economy/ and papers/auditor-ceiling/ in the repo. every prereg was committed before
its data existed.
