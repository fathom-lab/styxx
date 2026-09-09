# What a self-written record can establish, and what it cannot

Fathom Lab · 2026-09-09 · **An argument with receipts, not a result.** It proves nothing. It reports
a sequence of five repairs and five defeats that happened in one day on one machine, and states the
invariant they share. The attacks are all real, all reproducible from scripts under
`papers/v8/challenge_and_attack_2026_09_09/` and the session scratchpad, and every one was found by
an adversary asked to break the lab's own newest work. Not sworn.

## The sequence

Each row is a defence this lab added, and the attack that walked through it the same day. Read the
right-hand column as one move repeated.

| round | the check we added | how it was defeated |
|---|---|---|
| 1 | the noise plan must be logged **before** the runs it governs | the plan was logged, and the runs ignored it — every run at batch size 1 while the plan declared 1, 8 and 32. All ten pairwise distances zero. Perfect paperwork, empty number. |
| 2 | the runs must **record** assignments that differ | the assignments differed as *labels*. Two orderings that are one order, a batch size the runner coerces, a padding side that does not apply. Eight constructions, eight accepted, six with a floor of exactly zero. |
| 3 | the labels must map to distinct **executions** | an execution is derived from the cert's own item list chunked by the cert's own batch-size field. Both halves are written by the same party. One computation, relabelled, becomes five runs over four executions. |
| 4 | the log must **recompute** the floor from the run certs it names | the recomputation is exact and confirms the arithmetic to the last digit — of whichever bytes the issuer chose to name. Hand-pick a favourable subset (twelve of fifteen appended). Or put a float16 body into a bfloat16 floor, and every channel reads `same`, because the floor *is* the drift and the recomputation certifies it. |
| 5 | a floor must rest on exactly the runs the plan fixed | the plan's `runs` key was not required by the schema. One omitted key and the issuer picks its sample again, with the floor's own consistency check reporting no disagreement. |

Two further results belong in the same column. The verifier never checked that it had loaded the
model the cert names, so a run of different weights produced a signed drift accusation against
another party's certificate — until it was repaired, by making the runner *report what it loaded*.
And the environment is still asserted rather than observed: one CPU forward pass, ninety seconds,
the shipped command line, and the signed record names a GPU it never touched.

## The invariant

Every defence in that table asks a certificate a question about itself, and the certificate answers.
The repairs got steadily better at asking — from *did you declare it* to *did you record it* to
*did it change an execution* to *do the numbers follow from the bytes* — and each one moved the
trust boundary inward without crossing it. The adversary's own summary, after round four, is the
cleanest statement of it we produced:

> the anchoring repair answers the question "is this floor the Appendix B function of the bytes it
> names?" when the question the vacuous floor was always asking is "do those bytes measure
> anything?", and the issuer still writes the bytes.

The terminal case is worth naming because it bounds the whole exercise. Two certificates written by
one party can always be made to agree with each other. No check performed on bytes an issuer wrote
can distinguish a computation that happened from a record of one, because both are the same kind of
object: a claim, signed.

## The partition, which is sharper than the argument above

A third adversarial pass, asked directly whether any remaining attack was reachable by *any* check
the log could perform on bytes an issuer wrote, answered by partitioning them. The partition is a
better statement of this document's thesis than the narrative that produced it.

**Class one — reachable, and therefore owed as work.** Seven remaining defects are predicates over
bytes already on disk: the floor census is unsigned metadata outside the tree while the function
that re-derives it exists and nothing calls it; `covers` and `not_covered` are never compared
against the plan the floor names by id; a missing issuer roster fails open instead of closed; no
cert names the log it belongs to, so "this cert is in *the* log" is uncheckable; the plan's index is
never compared against its runs'; the gap between two baselines is computable from two logged certs
and does not exist; and `subject.environment` is not required by any schema. None of these is deep.
All of them are work.

**Class two — the label class, reachable by nothing.** Every other surviving defect is a certificate
field that describes a computation nobody but the issuer witnessed, and whose forged value is
**byte-indistinguishable** from its honest value. The adversary's demonstration is the cleanest
statement of the boundary this session produced:

> five copies of one forward pass with five batch labels give `executions = 5`,
> `pairs_same_execution = 0` and a floor of 0.0 — and five real runs that happened to agree exactly
> give the same bytes.

The two artifacts are identical. No predicate over them can differ. What follows is the general
form: *a check on bytes an issuer wrote can only ask whether that party contradicted itself, and a
party that does not contradict itself is not caught by asking.*

**CORRECTION, same day, and it is a correction to this document's central claim.** The roster
published here first had four members. One of them was wrong. It read: *"the environment the whole
coverage vocabulary resolves against, which is copied out of a specification file by a command that
builds no runner at all."* That defect was **reachable**, and a fourth adversarial pass — asked
specifically to hunt for a member misclassified in this direction — found it by noticing that the
specification's own amendment A-52 describes the same defect as "a predicate over bytes already on
disk." One document's class two was the other's class one. It has since been closed in both halves:
the plan now observes its environment through a runner at mint, and the log compares every
undeclared environment leaf of the plan against every floor cert's subject, which is a genuine
predicate over logged bytes and was confirmed refusing.

This is the most damaging error this analysis could have contained — declaring something impossible
that was merely unaddressed — and it survived the writing of the document, the writing of the
specification amendment that contradicted it, and a public summary of both. It was found only
because the adversary was asked to disbelieve the conclusion, which nobody does by default.

**SECOND CORRECTION, same day, same error.** A fifth pass was asked to attack the corrected roster
on the grounds that an error present once is likely present twice. It was. **Member 3 — the
repository and revision read off a directory name — is also reachable**, by exactly the test that
reclassified the environment member: another logged cert already carries the same information and
nothing compares them. Beside `hf_repo` and `revision` in the same subject sit the Appendix A.2
hashes over the snapshot's actual bytes. The predicate *"across the entries of one log, one
(hf_repo, revision) names one set of content hashes, and one set of content hashes names one
(hf_repo, revision)"* is a predicate over bytes already on disk, is not implemented, and both halves
were demonstrated: the same `weights_sha256` under two revisions appends, and the same revision
under two weights hashes appends.

The adversary's own caveat is the honest part, and it is why this reclassification is forced rather
than optional: that predicate does not *fully* close member 3, because renaming the directory once
makes every cert tell the same lie consistently. But that is exactly as true of the environment
member, and this lab counted that one as class one. By the standard the lab itself applied, member 3
belongs in the backlog.

**The roster is now two members.** It has shrunk at every pass that attacked it: four, then three,
then two. That trend is the finding, and it should be read as a warning about this document rather
than a record of progress. Each member was placed in class two by an argument that felt conclusive
when written, and two of the four did not survive an adversary told to disbelieve it. The correct
inference is that **the burden of proof for "no check can reach this" is much higher than the burden
this document originally applied**, and that the remaining two should be assumed reachable until
someone has tried and failed to reach them, rather than the reverse.

A separate correction to this document, of a different kind. It credited member 2 with a mitigation:
*"still unreachable, but no longer silent — the `baseline_gap` quantity now reports it."* That
sentence is false. The previous-comparable lookup never excludes the appending cert's own floor runs:
verified directly on this lab's published verdict, where the lookup for the canonical fingerprint
returns entry 5, which is that canonical's own run 4. A fresh append would therefore measure the
canonical against one of its own floor runs and call the result a baseline gap. (The stored entries
carry no `baseline_gap` at all, because the log predates the field, so nothing in the published
bytes announces anything — the defect is in what the code would now compute, not in what that log
says.) the honest and dishonest cases produce the same shape; and the announcement is
pairwise, so a staged move publishes nothing cumulative. The member's classification stands. The
claim that it is disclosed does not, and a quantity that announces an event that did not occur is a
false accusation of the same species as the tamper report repaired earlier today.

The roster, corrected **twice**, and to be read as provisional:

1. **The batch labels written alongside the recipe meant to corroborate them**, both halves signed
   by one party. Re-confirmed at every pass: five byte-identical bodies wearing five batch labels
   produce the same bytes and the same census as five real runs that agreed exactly.
2. **The choice of which run becomes the baseline.** Classification stands; the mitigation this
   document credited it with does not, and is withdrawn above.

The environment member and the repository-and-revision member were both moved to class one after
being placed here, each by an adversary asked to disbelieve the claim.

The general form survives both corrections: a check on bytes an issuer wrote can only ask whether
that party contradicted itself. What did not survive is this document's confidence about *which*
defects have that shape. Twice, a defect was called unreachable when another logged cert already
carried the information that would have caught it and nothing compared them. That is not a subtle
failure mode; it is the first thing to check, and it was missed twice by the same author in one
day. A reader should treat the two members above as claims awaiting a serious attempt to refute
them, not as established results.

The practical difference between the classes is what a reader should do about them. Class one is a
backlog. Class two is a disclosure: it belongs in the output beside the verdict, not in a limits
section, because no future release closes it.

## What such a record does establish

This is not an argument that the machinery is worthless, and the distinction matters more than the
complaint. A self-written, content-addressed, append-only record establishes a real list of things,
all of them checkable by a stranger with no trust in the issuer at all:

- **Integrity.** These exact bytes, unaltered. One flipped byte in one entry of a real log was
  caught and named three ways.
- **Attribution.** This key signed them, and a challenge cannot be re-attributed to the party it
  challenges.
- **Order.** This entry preceded that one, relative to a tree head — and this is the property that
  survives a broken signature scheme, which is why it is the one worth anchoring outside.
- **Internal consistency.** The arithmetic follows from the bytes; a floor is the function of the
  runs it names; a claim's numerals resolve to leaves that exist.
- **Commitment.** A hypothesis was fixed before data existed, *if* the log's order is trustworthy.

That list is not small, and most published computational evidence establishes none of it. What the
list does not contain is any statement that a computation described in the record actually occurred,
or that it measured what the record says it measured.

## What follows, and it is the design consequence

The challenge mechanism is not a feature of this system. It is the only part of it that introduces a
byte the issuer did not write.

Everything else — the canonical bytes, the Merkle tree, the signatures, the floor recomputation, the
plan-before-runs rule, the anchor — is bookkeeping over one party's assertions. Good bookkeeping,
and worth having: it makes a lie *durable and attributable*, which raises its cost and lets a later
reader find it. But the transition from "this record is consistent" to "this measurement happened"
requires a second party to run the same battery and put their own bytes in the log. There is no
substitute, and every check we wrote today was an attempt to find one.

Three practical consequences, stated so they can be argued with:

1. **A verdict on a subject nobody else has run is a claim about a record, not about a model.** The
   spec should say that where it prints verdicts, not only in a limits section.
2. **The reproduction count is the number that matters** — how many independent keys ran this
   battery while running it was possible — and it is cheap to compute and currently computed
   nowhere. It is also the one number that decays: once a model is unobtainable it can never rise.
3. **A design that cannot attract a second party has not failed at engineering.** It has failed at
   the only thing that would have made the engineering mean anything.

## What this does not say

It does not say the attacks are unfixable — five of the twelve named across three passes are now
refused by rules that decide rather than report. It does not say self-written records are worthless;
the list above is the argument against that. It does not claim novelty: this is the ordinary
distinction between an audit and an attestation, and every mature evidence discipline has some
version of it. And it is not a proof of anything — it is five rounds on one machine in one day,
where the same adversary attacked its own previous conclusions, which is a method with obvious
limits and one virtue: nobody here was trying to make the system look good.

## Limits

One lab, one operator, one day, one model, one battery of 64 prompts. Every attack and every repair
was produced by the same session that is now reporting on them, and none has been reviewed by anyone
outside it. The invariant is an observation over five cases, not a theorem; a sixth repair might
cross the boundary rather than move it, and this document would be wrong. That would be the best
outcome available and nothing here should discourage the attempt.
