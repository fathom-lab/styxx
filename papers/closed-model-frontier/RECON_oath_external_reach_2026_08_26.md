# RECON — OATH-EXT: what the verifier does on documents this lab did not write

> **Back-pointer, added 2026-09-01.** The false-accusation claim this document makes was withdrawn the following day; the measured rate on foreign text is published there as an upper bound — see `RESULT_oath_external_corpus_2026_08_27.md`. Inserted per the staleness table of `AUDIT_the_whole_program_2026_09_01.md`; the text below is unchanged.

Fathom Lab · 2026-08-26 · **RECON. This document licenses no claim.** It sizes a class and
reports a boundary. Its numbers are inputs to a future preregistration's bars, not results.
Receipt: `oath_ext_recon.json`. Harness: `oath_ext_recon.py`, deterministic, re-runnable.

---

## The unexamined assumption

Every one of the 178 committed OATH certificates in this repository is on a document written by
this lab. Twelve cycles of instrument work — v0.1 through v0.11, each with its own frozen
preregistration, its own battery, its own adversarial pass — have been scored entirely against a
corpus the instrument's authors also wrote.

That is the largest unexamined assumption in the OATH lane, and it is the first thing a hostile
reader finds. An instrument tuned on its authors' own prose may be measuring their idiom rather
than anything about claims. Another lane of this lab already went outside and found real defects
in third-party code (`RESULT_sp_ext_2026_08_21.md`); the OATH lane never has.

This RECON points the shipped verifier at README claim documents in public repositories, against
those repositories' own committed summary-metric receipts.

## The standard, inherited verbatim from SP-EXT

> styxx does not adjudicate whether the external authors are right. The only thing measured is
> whether a number in a repository's own claim document grounds in that repository's own
> committed receipt.

Nothing below is a criticism of anyone's research. Every repository named here is doing what
essentially all of the field does; that is the point.

## What is being tested, stated fairly before the numbers

**This tests OATH outside its own stated contract, deliberately.** `styxx/certify.py` scopes
itself in its docstring: *receipts are the explicit set passed in (the doc's own cited result
JSONs), not discovered.* External READMEs do not cite receipts. The harness supplied them by a
filename heuristic — the summary metric files the HuggingFace Trainer writes.

So this is not a demonstration that OATH fails at something it promised. It is a measurement of
whether it TRANSFERS to documents that were not written to be certified. That transfer is not
something the instrument claims — but it is what "trust becomes a measurement" would require, so
it is worth knowing, and it had never been measured.

## Selection, pre-committed before any repository was read

GitHub code search for `filename:all_results.json` and `filename:eval_results.json`; the first
distinct repositories in the API's own returned order; no inspection, no substitution. Disclosed
as it stands: GitHub's default ordering is "best match", not random, so this is a convenience
sample of a mechanically-defined population. It supports no inference about base rates anywhere.
It is reported because a mechanically-selected sample is auditable and a hand-picked one is not.

Of the repositories selected, 12 yielded a document-and-receipt pair and were certified. External
repositories were cloned and READ; no external code was imported or executed.

## What happened

| | |
|---|---|
| repositories certified | 12 |
| numeric tokens examined | 507 |
| VERIFIED | 17 |
| ABSTAIN | 477 |
| UNGROUNDED | 13 |

**The instrument abstains on 0.9408 of what it sees.** On documents written without any
intention of being certified, OATH overwhelmingly has nothing to say. That is the honest headline
and it is not a surprise — abstention is the designed behaviour where nothing grounds. What
matters is the two thin columns on either side of it.

## The accusation column: 13 for 13 traceable to a trigger read out of context

Every UNGROUNDED token is an assertion that *this token is a claim whose truth condition was
never met*. Each one below is recorded with the trigger vocabulary that obligated it.

| document | token | obligated by | what the token actually is |
|---|---|---|---|
| DePT README L93 | 100,000 / 300,000 / 40 / 60 | `rate` | hyperparameter ranges in a prose paragraph about the work's limitations — "the learning rate of the low-rank matrices and training steps" |
| CoReVLA README L16 | 72.18 / 50 / 7.96 / 15 | `rate`, `score` | figures cited in a paper abstract, whose home is the paper's own tables and not the eval receipt supplied |
| BetterMixture README L116, L128 | 1.554562 / 1.598064 | `score` | lines of pasted console output reading `score:1.554562` |
| BetterMixture README L180, L192 | 637.879562 / 167.052877 | `mean` | rows of a pasted `describe()` table reading `mean       637.879562` |
| GradRetentionNet README L67 | 1 | `delta` | the literal `1` inside the LaTeX formula `\left(1 \pm \frac{\Delta \sigma^2}{\sigma^2}\right)` |

**Not one is a catch.** The last row is the purest specimen this lab has produced: a mathematical
constant inside a rendered formula, accused because the formula contains a Greek delta. It has no
truth condition at all. It is exactly the class the v0.11 cycle spent a full preregistration
retracting four instances of — a token that is not a claim — and it was found in minutes, on a
document nobody here wrote, by an instrument that had just been hardened against that very class.

The mechanism is visible and identical in every row: the obligation predicate reads a LINE, not a
CLAIM. `rate` fires on "learning rate". `mean` fires on the left margin of a pandas table.
`delta` fires on `\Delta`. Inside this lab's corpus that vocabulary is a decent proxy for "this
line reports a measurement," because this lab writes lines that report measurements. Outside it,
the same words are configuration, notation, and console noise.

## The verification column: thin, concentrated, and partly coincidence

Of the 17 VERIFIED tokens, 16 come from a single repository. Graded by the structural definition
frozen in this lab's dogfood instrument — a binding is coincident when its receipt path ends in a
bare array subscript or an index-like name — 5 of 17 are structurally coincident, grounding at
things like `config.seeds[0]`, `metric_history.step`, and `results[0].metric_history.loss[…]`.

That 5 is a floor and hand inspection immediately exceeds it: a bare `3` on two separate lines is
sworn to `rope.aggregated.standard.final_loss`, and a bare `1` is sworn to `[0].level`. A count of
layers matching a loss value is not a verification; it is arithmetic coincidence wearing a
receipt.

## What this shows, and what it does not

**Shows.** Pointed at documents not written to be certified, the shipped OATH verifier abstains on
almost everything, verifies a little — some of that coincidentally — and every accusation it
makes is false. Its trigger vocabulary is tuned to this lab's writing idiom. That is now measured
rather than suspected, and it is measured by this lab rather than by a reviewer.

**Does not show.** That OATH is broken: it is doing what a demarcation instrument should do when
it cannot ground anything, which is abstain, and the accusation surface here sits outside its
stated contract. Nor does it show any base rate — the sample is a convenience sample of one
file-naming idiom, twelve repositories deep. Nor does it license a false-accusation *rate*: the
adjudication above is one reader's, published so that it can be disputed, and a hostile
adjudication of the same 13 tokens is invited.

## This document is OATH-FAILED, and that is the second finding

The certificate beside this file says `OATH-FAILED`. It is published that way rather than
reworded until it passes, because of what it is accused of.

Two of the accusations are the tokens in the table above — the hyperparameter ranges from the
DePT README — accused *here*, in a row whose entire purpose is to report that accusing them was
wrong. The rest fall on the sentence describing which verifications were coincidences, where
the digits being discussed are read as digits being asserted.

**OATH cannot tell a mention from a use.** A document that quotes a number is treated as claiming
it. The shipped verifier has one narrow escape for this — the v0.1 quoted-historical rule, which
fires only on disclosure phrasing like *originally printed* or *superseded* — and it does not
reach quotation in general. The v0.10 hand panel already brushed against this and recorded it as
its only MEDIUM-confidence cases: a reader treating quotation as mention-not-use would have moved
its split. It was disclosed there and not pursued.

It is worth being exact about what this does and does not mean. The instrument is not
malfunctioning; it is doing precisely what its predicate says, and the predicate has no notion of
quotation. But it means an error report cannot be written inside this system without being
accused of the errors it reports, which is a real limit on a corpus whose whole method is
publishing its own negatives loudly. The retraction cycle that just shipped was licensed by a
panel asking *is this token a claim at all?* — and mention-versus-use is the same question,
unanswered, on a much larger class.

Filed here as a named residual rather than fixed: fixing it inside a RECON, with no frozen
preregistration and no adjudication, is exactly the move this lab does not make.

## The boundary, and what it clarifies

The north star reaches for proof-carrying code as its analogy, and the analogy is more exact than
it has been given credit for. **Proof-carrying code does not verify arbitrary binaries.** It
requires a compiler that emits the proof alongside the program. Retrofitting it onto software
that was not built to carry proofs is not a thing anyone claims to do.

This RECON says the same is true one level up. OATH is not a lie detector that can be aimed at
arbitrary prose; on arbitrary prose it is nearly silent, and where it speaks it is wrong. What it
is, and what the certified corpus demonstrates it can be, is **a contract**: documents written to
carry receipts can be mechanically held to them, by anyone, without trusting the author. That is a
narrower claim than "trust is a measurement" and a considerably more defensible one, and it is the
claim the evidence in this repository actually supports today.

The gap between those two statements is the honest size of the remaining work, and naming it is
worth more than another cycle spent tuning the instrument against its own authors.

## What this RECON exists to inform

A preregistration, not a result. The bars a successor should freeze, in leverage order:

1. **A false-accusation bar on external documents.** The measurement above is the pilot; the
   cycle is a frozen adjudication protocol over a larger mechanically-selected sample, with ties
   resolved AGAINST the instrument, and a pre-committed ceiling above which the accusation
   surface is declared undeployable outside the contract.
2. **The obligation predicate reads a line, not a claim.** Every failure above is that one
   defect. Whether a predicate exists that reads claimhood rather than line vocabulary is the
   real open question, and it is the same question v0.11 answered structurally for one narrow
   class. A negative here is as valuable as a positive.
3. **The contract, written down.** If OATH's honest scope is documents authored to carry
   receipts, then the deliverable is a specification an outside author could adopt, plus the
   check that tells them whether they have. That is testable, it is useful to someone other than
   this lab, and nothing in this repository currently offers it.

---

*The instrument that certifies claims was pointed at the outside world, and it failed. Publishing
that on the same day is the only reason the rest of the record is worth anything.*
