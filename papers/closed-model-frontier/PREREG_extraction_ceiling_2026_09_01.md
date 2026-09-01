# PREREG — the extraction ceiling: is V14's 0.16 an adjudication failure or an extraction one?

Fathom Lab · 2026-09-01 · Frozen before the packet is re-opened, before a single item is
re-read, and before any panel is convened. Successor question to
`RESULT_v14_naming_the_defects_did_not_save_it_2026_09_01.md`. Corpus and split governed by
`SPLIT_external_corpus_2026_08_31.md`. Population and receipts: `v14_gates.json`,
`v14_adjudication.json`.

**The standing commitment this document is written under:** do not ship an accusing verdict
whose precision has not been measured by a blind panel. Absence of evidence is never a
contradiction. Never "first". Never "nobody". Always "we know of no other."

---

## The question, and why it has never been asked

`RESULT_v14` is this lab's most expensive published number. After two repair cycles that removed
3,083 of 4,427 path accusations corpus-wide, held-out precision came back at **0.16 against a
floor of 0.95** (`v14_adjudication.json`: 100 scored, 16 upheld, 30/30 decoys). The report's own
title is *the named false accusations were removed, and precision did not follow*, and its
closing position is that the shortfall is **not explained**. The template account was neither
displaced nor confirmed. `ANALYSIS_base_rate_ceiling_2026_09_01.md` later killed the base-rate
excuse — ceiling 1.000, not binding — which removed a defence without supplying an explanation.

That document is 84 lines long and the word **extraction** does not appear in it once.

Every precision this lab has published answers one question: *given that a claim was extracted,
was the verdict on it right?* Not one answers the question before it: **given that the regex
fired, was a claim being made at all?** A false extraction produces a false accusation exactly
as surely as a false adjudication does, and it is invisible to every panel we have run, because
every panel we have run was shown the extracted claim and asked about the verdict.

`extraction_census.json` asked the prior question on the `tests_pass` template only, and we
know of no other measurement of it, ours or anyone's. Its licensed, mechanical half — a pure
function of the summary bytes, no panel required — found that **179 of 5,514 matches sit in an
unticked task-list box** (`extraction_census.json`, `headline`). That is a CONTAINMENT figure:
it records where a match SITS, not whether the extraction was WRONG. Whether an unticked box or
a fenced line makes an extraction wrong is precisely the question this document preregisters and
it has never been asked, so no share of matches may be described here as "not assertions". Its
judgment half is explicitly unvalidated and its own warning forbids quoting it as a rate. So the
prior question is now known to be non-empty, and unanswered for the class that actually shipped
an accusation.

**This preregistration does not measure a new corpus. It re-asks a different question of a
packet that is already sealed and whose answers are already committed.**

---

## The decomposition, stated before any number exists

An accusation survives to be upheld only if two independent things went right. Write it:

    P  =  E  x  A

  * **P** — precision of the shipped accusation. Already measured, already published, already
    committed: **0.16** on the held-out packet.
  * **E** — the extraction term. Of the accusations, the share in which the author was making a
    path claim at all, as opposed to naming a path, referring to one, quoting one, listing one
    inside a fence, or writing one into an unticked template line. **Never measured.**
  * **A** — the adjudication term. Given that a claim was genuinely being made, the share in
    which the gate's verdict on it was right. **Never isolated** — every published number of
    ours is P, and has silently been read as if it were A.

Three repair cycles have been spent on A. If E is small, they were spent on the wrong term, and
the retired class was retired for a reason we know of no measurement of, ours or anyone's.

---

## The measurement

**No new sample is drawn.** The V14 held-out packet is 130 sealed items — 100 sampled
accusations plus 30 decoys (15 verified, 15 not) — with the key digest committed before judging
and the per-item answers already in `v14_adjudication.json`. Issue #43 publishes it as a
standing re-adjudication invitation.

The same 130 items are put to a blind panel with **one question substituted**:

> Ignoring entirely whether the gate's verdict was correct, and reading only the author's
> summary: **is the author making a claim about this path** — asserting that this change
> created, deleted or touched it? Answer CLAIM / NOT-A-CLAIM / UNREADABLE.

Panellists see the summary sentence and the path. They are **not** shown the gate's verdict, the
gate's reason, the diff, or any V14 material. Multi-seat, majority-scored, same protocol as G-S3.

Because the packet and our answers are both already committed, E and A are then computed
**exactly, item by item** — not estimated by dividing one aggregate by another:

  * `E` = CLAIM items / scored items
  * `A` = upheld items / CLAIM items   (upheld read from the committed V14 key)
  * Reconciliation obligation: `upheld / scored` **must** re-derive 0.16. If it does not, the
    join is wrong and the run is VOID, not adjusted.

---

## Gates — thresholds fixed now, before the packet is re-opened

**G-E1 — reliability.** The 30 sealed decoys are re-purposed as extraction decoys: 15 in which a
claim is unambiguously being made, 15 in which it unambiguously is not (a path inside a code
fence, an unticked template line, a comparative reference). Panel accuracy on decoys **>= 27/30**
or the panel is VOID and no E is reported. Non-negotiable, and a void panel is published as a
void panel.

**G-E2 — reconciliation.** `upheld / scored` re-derives 0.16 (tolerance: the packet's own
rounding). FAIL -> VOID.

**G-E3 — the hypothesis, pre-committed in both directions.**

| observed E | verdict, fixed now |
|---|---|
| **E <= 0.23** (implies A >= 0.70) | **SUPPORTED.** The adjudicator is sound and extraction is the binding constraint. V14's shortfall is explained, and the explanation is that three repair cycles worked the wrong layer. |
| **E >= 0.40** (implies A <= 0.40) | **REFUTED.** The claims were real and the gate misjudged them. Extraction does not exculpate the adjudicator, and V14's 0.16 remains unexplained. Published as a failed hypothesis of ours. |
| 0.23 < E < 0.40 | **INDETERMINATE.** Published as indeterminate. No narrative is built on it. |

We commit to publishing the REFUTED cell with the same prominence as the SUPPORTED one. This lab
has published two failed instruments and one OATH-FAILED paper; a third failure is the expected
cost of asking.

**G-E4 — no receipt is regenerated.** `v14_adjudication.json`, `v14_gates.json` and
`RESULT_v14...md` are history and are not edited, re-run, or "corrected" by this work. A receipt
is history too. Findings land in a new RESULT beside them.

**G-E5 — the two carried defects are fixed before anything is classified.** Both were found in
`extraction_census.py` on 2026-09-01 by adversarial review and both would corrupt this run:

  1. `_PREV_WORD` requires a leading `[A-Za-z]` and cannot cross digits, so `907/913 tests pass`,
     `19 of 21 tests pass` and `3 tests pass` are labelled bare unqualified assertions — a
     numeric scope is read as no scope.
  2. `negation_cue_near` is a regex approximating a human reading and is published inside the
     MECHANICAL group, whose entire selling point is that it needs no panel. It is 0-for-4 on the
     file's own emitted examples.

Repairs ship with the classifier, and `extraction_census.json` is **re-emitted as a new dated
receipt**, never regenerated in place.

---

## What would make us not ship

If the panel splits (no majority on more than 10% of items), E is not reported as a number at
all. If the decoy gate fails, nothing is reported but the failure. If the reconciliation gate
fails, the join is disclosed as broken and the packet is left alone.

---

## Honest limits — the price list

**This cannot revive the retired class.** A low E would explain the 0.16; it would not license
re-enabling a path accusation. Reviving requires repairing extraction and then measuring
precision again, blind, under its own preregistration. Nothing here is a permission slip.

**The panel is ours.** Convened, instructed and sealed by us, on our sample — the same limitation
named in `PREREG_third_party_precision_2026_09_01.md`. It is disinterested in the verdict but not
independent of the lab. A reader who believes we fool ourselves has only our receipts to check us
with, and that remains true here.

**E is a judgment quantity and is labelled one.** Unlike the mechanical half of
`extraction_census.json`, "was a claim being made" cannot be decided from bytes. It is exactly
the class of instrument this lab measured at 0.16, which is why it gets a panel and a decoy gate
rather than a regex.

**The classes differ, and the transfer is not free.** Path claims already reach the `_REFERENTIAL`
and `_CONTAINMENT` guards; `tests_pass` reaches neither. Path extraction may therefore be markedly
cleaner than the `tests_pass` census was, and E may come back high. That asymmetry is the honest
reason this question is worth asking rather than assuming.

**Disclosed before the fact:** in preparing this document we read held-out `tests_pass` example
sentences emitted by `extraction_census.json`. No rule in this preregistration is derived from
them, and they concern a different template set. Recorded here because
`SPLIT_external_corpus_2026_08_31.md` rule 3 requires that any contact with held-out prose be
declared, not because we believe it contaminated anything.

---

## What this can and cannot support

**Can:** a decomposition of one published precision into an extraction term and an adjudication
term, measured on a sealed packet whose answers were committed before the question was asked.

**Cannot:** a claim about any other instrument, any other corpus, or any other lab's tool. Not a
prevalence. Not a statement about how often agents lie. Not a novelty claim of any kind.

**Prior art and credit.** Decomposing a pipeline metric into per-stage terms is ordinary practice
in information extraction and information retrieval, and we claim nothing about the method. What
we know of no other instance of is a claim-checking gate publishing an extraction term for its own
shipped accusation against free-form developer prose. Predecessors that bind natural language to
executable checks — Cucumber/Gherkin (2008), Doc Detective, Jdoctor/Toradocu (ISSTA 2018) —
largely avoid this term by construction, because they require the claim to be written in a form
built to be machine-read. That is a reasonable design, and it is the one this measurement exists
to price.
