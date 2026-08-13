# NOTE — is "self-audit converges slowly and flatteringly" a publishable result? Not yet, and here is the shape of the study that would make it one

Fathom Lab · 2026-08-13 · written in response to a direct question from the red-team operator ·
this document deliberately argues against its own attractive conclusion first.

## The observation

One day, one module, one author:

- three self-audit passes, three real defects, **each pass refuting the previous pass's "honest"
  number** (headline uninterpretable → false positive inside it → the data both ran on was 62%
  missing);
- an adversary then found two more defects **in the two places already corrected twice**, plus a
  judgment error the author had consciously recorded and walked past;
- the author's attempt to contest one finding produced a broken instrument returning
  0/54,701 — the self-serving answer — until the fixture was corrected to 178/54,695.

Every error, without exception, was in the flattering direction. That is nine or so events with
no counterexample, and it feels like a law.

## Why I am not writing it up as one

**1. n=1 on every axis that matters.** One author, one codebase, one adversary, one day. The
"pattern" has no variance to speak of because there is nothing to vary against. Nine events drawn
from one session are not nine independent samples; they are one session.

**2. The direction claim is nearly unfalsifiable as stated.** "Errors trend flattering" needs a
defined null. What fraction of *arbitrary* bugs in an audit tool would happen to be flattering?
Plausibly most of them: a tool that under-reports its own limitations is a tool that reports a
higher score, and there are simply more ways to be silently permissive than silently strict. If
the base rate is 0.8, observing 9/9 is unremarkable (p≈0.13). **Without that base rate the
headline is a vibe.**

**3. Selection effect on which defects got found.** Defects are discovered by the very audits
whose bias is under study. Flattering defects survive longer, so they are *available* to be found
later — which manufactures the temporal pattern ("self-audit converges slowly") independent of
any property of self-audit. This is the serious confound and it is not fixable by more anecdotes.

**4. The adversary was not blind.** Fable was handed my prioritised list of suspicions and found
defects where I said to look. That is evidence adversarial audit is *efficient given good
priors*; it is much weaker evidence that it finds what self-audit *structurally cannot*. The
strong claim needs an adversary working without the author's hints.

**5. I am the interested party.** The conclusion flatters the practice I have spent the day
performing and would like to be known for. Per the standing rule, a favourable number about
myself gets a qualified judge first — and here the "judge" would be my own narrative of my own
day, which is the least qualified instrument available.

## What would make it publishable

**Claim, narrowed to something falsifiable:**

> On a fixed defect corpus, the expected number of audit passes to detect a defect is greater
> when the auditor is the artefact's author than when it is an independent adversary, and the
> gap is larger for defects whose presence inflates a reported metric than for defects that
> deflate it.

Two measurable quantities: **passes-to-detection** (author vs adversary) and an **interaction
with defect polarity**. The polarity interaction is the load-bearing part — it is what
distinguishes "authors are slower" (uninteresting, obviously true) from "authors are
*selectively* slower where the error flatters them" (the actual claim).

**Design.** Mutation testing, which sidesteps the selection effect entirely:

- Take audit/measurement tools with test suites. Inject synthetic defects by mutation, each
  pre-labelled **flattering** (inflates a reported score / hides a limitation) or **deflating**
  (the mirror image), matched for code-location and severity so polarity is the only difference.
- Have the author and an independent adversary each audit under identical time and information
  budgets, blind to which mutants exist.
- Outcome: detection rate and passes-to-detection, per arm, per polarity.
- **Pre-register the polarity interaction as the primary endpoint**, and pre-register the null:
  that detection depends on defect *salience* alone, with polarity contributing nothing.

**What falsifies it.** If flattering and deflating mutants are detected at the same rate by
authors — the interaction term is null — the interesting claim is dead, and what remains is the
banal "fresh eyes help." That outcome must be publishable in advance or the study is decoration.

**N.** For a within-subject design detecting a medium interaction (d≈0.5) at 80% power, roughly
**30–35 author/adversary pairs**, each auditing ~20 mutants (10 per polarity) — order 600–700
mutant-audit events. Under the C6 discipline the bar comes off a measured power curve, not this
paragraph: the honest procedure is a Monte-Carlo detection simulation on synthetic mutants
*before* any real auditor is recruited, exactly as `power_c6.py` was run before the C6 bar was
frozen. **A study of audit bias that asserts its own sample size without a power basis would be
self-refuting.**

**Cheaper intermediate that is worth doing regardless.** The `resolution_probe.py` suite
generalises today's defect class into three black-box tests. Running it across a corpus of
published audit/eval tools yields a real, reportable number — *what fraction of shipped
measurement instruments report a rate without its chance floor* — with no claims about human or
agent psychology at all. That is a finding I can stand behind on evidence I can actually collect,
and it is the honest version of today.

## Verdict

**Not publishable as a result. Publishable as a pre-registration**, and the pre-registration is
worth more than the anecdote because it can lose. The corpus study of shipped instruments is the
near-term paper; the author-vs-adversary polarity study is a real experiment that needs a power
basis before it needs a hypothesis.

The pattern is real enough to design an experiment around and nowhere near strong enough to
announce. Saying so is cheaper today than retracting later — and today already contains one
retraction of a number I wanted to be true.
