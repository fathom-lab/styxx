# four instruments, one defect: this lab cannot tell a mention from a use

*2026-08-26. A synthesis, written after three unrelated investigations in one day each ended at
the same sentence, and a fourth turned up while checking the third.*

Receipts: `closed-model-frontier/oath_ext_recon.json`, `ledger_classifier_audit.json`,
`closed-model-frontier/oath_v11_battery_result.json`, and — for the addendum's eighth instance —
`closed-model-frontier/formula_span_census.json`. The diffgate figures are re-derived on
demand by `python scripts/diffgate_validation_sweep.py` and are disclosed as command-reproducible
rather than pinned to a committed receipt.

---

## the thing that was hiding in plain sight

Today began by shipping OATH v0.11, a preregistered cycle whose entire content was the discovery
that four accusations this verifier had been making were not wrong — they were **not claims at
all**. A markdown table's row ordinal has no truth condition. Saying it failed to ground is a
category error, and the cycle retracted it under a nine-gate battery and a blind adjudication.

By the end of the day the same sentence had been written three more times, about three other
instruments, none of which share a line of code with the first:

**1. The OATH obligation predicate, outside this corpus.** Pointed at twelve public repositories
it had never seen, the verifier abstained on 0.9408 of 507 tokens and made 13 accusations, **not
one of which is a catch**. Every one traces to vocabulary read out of context: `rate` firing on
*"the learning rate of the low-rank matrices"* in a limitations paragraph; `mean` firing on the
left margin of a pasted `describe()` table; `score` on a line of console output. The purest is
`delta` firing on `\Delta` inside `\left(1 \pm \frac{\Delta \sigma^2}{\sigma^2}\right)` — a
mathematical constant, accused of being a claim whose truth condition was never met.

**2. The OATH verifier on quotation.** The document reporting that finding is itself OATH-FAILED,
accused on the very tokens it quotes as examples of false accusations. There is one narrow escape
for quotation — disclosure phrasing like *originally printed* — and it reaches nothing else. It
was published failing rather than reworded until it passed.

**3. The ledger's own refusal classifier.** `papers/LEDGER.md` is the document every other claim
here is collateralised against. Under a heading reading *"cycles where a preregistered gate
returned `INVALID__*`"* it listed `SHIPPED`, `PRODUCT`, `DO`, `REWRITTEN` and `BUILT`. The
selection test was a substring match over a free-prose verdict field, so any cycle whose
commentary discussed an earlier invalid was counted as a machinery refusal. Sixteen became nine.
The specimen that should have caught this months ago: **cycle 156 is the cycle that built the
ledger, and it was counted as a loss because its verdict text quotes the ledger's own negatives
count.**

**4. The diffgate claim extractor.** Re-run at 7.46.0, the gate reports 13 claims and 4
contradictions across its validation corpus. Hand-adjudicated, all four are false accusations of
one shape: a filename *mentioned* in a commit message treated as a file the diff must contain.
One commit's prose says a candidate file is "nothing alike". One describes a fix in somebody
else's repository. One is a commit made today, whose message discusses the document it is
reporting a defect in. The README's present-tense claim of zero false accusations has been
withdrawn.

## the common cause, stated plainly

Every one of these instruments infers **claimhood from co-occurrence**. They look at what appears
near a token — a word on its line, a filename in its paragraph, a substring anywhere in a blob —
and conclude what the token asserts.

Co-occurrence is a serviceable proxy inside a corpus written by people who share an idiom. This
lab writes lines that report measurements, so a line containing `recall` usually does contain a
claim about recall. The proxy holds right up until the idiom changes — and then it does not
degrade gracefully. It produces confident accusations at a rate its own authors would not
tolerate from anyone else's tool.

## the part that should be uncomfortable

The proxy does not only break on strangers' documents. **It breaks hardest on writing that is
ABOUT claims rather than making them.**

A recon quoting false accusations. A ledger describing its own negatives. A commit message
reporting a defect in a file it does not touch. A limitations paragraph discussing the
hyperparameters a method needs. In every case the text mentions the furniture of measurement
without asserting any measurement, and in every case the instrument fires.

That is a genre. It is, specifically, **the genre this laboratory produces most of.** A programme
whose stated method is publishing its negatives loudly, dogfooding its own instruments, and
writing openly about what it got wrong has built four separate tools that systematically misread
exactly that kind of writing. The instruments are best calibrated for the papers we would write
if we were not doing the thing we say makes us different.

## this document is OATH-FAILED, on the example

The certificate beside this file says `OATH-FAILED`, and every accused token is a digit inside the
LaTeX formula quoted above as the canonical specimen of the defect. The line carries the word
`delta`, so the verifier obligates every number on it — including numbers that exist only because
the paragraph is *describing* an accusation rather than making a claim.

An earlier draft of this section quoted the formula a second time, in order to explain why quoting
it gets accused — and that doubled the accusation count. Removing the second quotation halved it
again. The exchange is in this file's history and is the cleanest part of the demonstration: under
this verifier the cost of explaining a defect scales with how carefully you explain it. No count
is printed here for the same reason.

A synthesis about the inability to distinguish mention from use cannot be written inside this
system without being accused of the errors it reports. That is the shortest available proof that
the four instances above were not bad luck.

## what v0.11 already knew

One place in this repository solved this, and its solution is the template.

The v0.11 cycle did not add a cleverer regex. It asked a different question — *is this token a
claim at all?* — put it to a hand panel with ties resolved **against** the clause, re-checked it
with a second blind adjudicator, and only then shipped a structural, value-blind predicate over
one narrow, enumerated class. It measured what the rejected alternatives would have cost: the
broad positional rule destroys a mean of 28.1 reader-visible catches per seed; the value-reading
detector misses the mutant it exists to catch at every seed.

The lesson generalises even though the clause does not: **claimhood needs its own predicate,
adjudicated rather than inferred from neighbours.** Every instrument above is currently deciding
claimhood as a side effect of deciding something else.

## addendum, 2026-08-27: it is not only the instruments

This synthesis found the defect in four instruments and concluded that claimhood needs its own
predicate. Over the following day it turned up three more times, and the three were not
instruments. They were the **measurements built to audit the instruments** — including the
measurements written specifically to stop the previous instance recurring.

| where | the proxy it matched | the class it named | what it cost |
|---|---|---|---|
| OATH obligation predicate | line vocabulary | this token is a CLAIM | every accusation false, outside this lab |
| OATH on quotation | a numeral appears | the author ASSERTS it | three certificates OATH-FAILED on quoted examples |
| the ledger's classifier | the substring `INVALID__` | the gate RETURNED one | `SHIPPED` listed as a machinery refusal |
| diffgate's claim extractor | a filename is mentioned | the diff CONTAINS it | every contradiction false; a README claim withdrawn |
| v0.12's frozen bar | a LINE-level marker | the SPAN the clause reads | reached six where the bar said eleven |
| the vacuous-pass census | the current tree | the PRE-FIX code | three repaired defects scored as zero reach |
| the formula-span census | a backslash | a LaTeX command | 31 of 40 corpus tokens are regexes, not formulas |
| the same census's deciding column | `destroys_nominal` is 0 | this rule COSTS nothing | it reads 0 for the worst possible rule too |

Eight, and the last four are the uncomfortable ones. The v0.12 bar was frozen by someone who had
just read this synthesis. The vacuous-pass census made v0.12's exact error *inside the census
written to stop it*. The formula-span census was built to be the careful, population-first
measurement the previous failure demanded, and its marker matched `\b` and `\d` — regex escapes —
while claiming to identify rendered mathematics.

The eighth is the sharpest, and it is a proxy for a **cost** rather than for a class.
`destroys_nominal` stood in for *would this rule destroy a real verification*, was named "the
column that decides", and cannot decide: a control run afterwards with no span test at all — the
worst rule available — scores the same zero. The column was blind because its frame holds only 184
documents, and every genuine measurement these rules would silence lives in one of the roughly 935
markdown files outside it. It was published the same day as the RECON that catalogues vacuous
gates, by that RECON's author, and an adversary found it. See
`closed-model-frontier/RECON_v13_not_frozen_2026_08_27.md`.

So the generalisation in the section above is too narrow. It is not that instruments decide
claimhood as a side effect. It is:

> **A marker that co-occurs with a class is not the class, and this holds for the measurement
> exactly as it holds for the instrument.** A census defines a population by matching something
> observable; if what it matches is a correlate rather than the thing, every number downstream is
> about the correlate. Freezing a bar against it does not make it true, it makes it frozen.

There is no reason to think measurement is safer than instrumentation here, and three instances
in one day say it is not. The countermeasure is not care — care was applied and failed three
times. It is the same one v0.11 used: **look at what the marker actually matched**, case by case,
before quoting any number it produced. Every one of the three was caught that way and none was
caught by reasoning about the definition in advance.

## what follows, and what does not

**Does not.** This synthesis licenses no fix. Nothing here is preregistered, and the correct
response to noticing a defect in four instruments on one afternoon is emphatically not to patch
four instruments that afternoon. Three of the four are unrepaired on purpose and their failing
cases are recorded where they can be re-run.

**Does.** Three things are now owed, and named so they cannot quietly lapse:

1. **A mention/use predicate, with its own preregistration and its own adjudication panel.** The
   v0.10 panel already brushed against this and logged it as its only MEDIUM-confidence cases: a
   reader treating quotation as mention-not-use would have moved its split. It was disclosed and
   not pursued. It is now the largest known defect in the OATH lane.
2. **A machine-readable verdict token per cycle.** The flagship negatives ratio is produced by
   keyword-matching prose, so it is not a measurement; head-scoping the same keywords gives a
   very different number and neither is right. Until a cycle's verdict has a parseable head, no
   count over that field is a count of anything.
3. **A standing rule for new predicates.** Any predicate in this repository that decides
   something about a token by reading text near it should be assumed to have this defect until
   an adjudication shows it does not. Four for four is not a run of bad luck.

---

*The instrument that certifies claims cannot reliably tell a claim from a sentence about one. We
found that out four times in a day, and the fourth time it was a commit written that morning,
reporting the third.*
