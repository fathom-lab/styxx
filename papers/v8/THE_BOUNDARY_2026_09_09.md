# What a self-written record can establish, and what it cannot

Fathom Lab · 2026-09-09 · **An argument with receipts, not a result.** It proves nothing. It reports
a sequence of five repairs and five defeats that happened in one day on one machine, and states the
invariant they share. It has been corrected four times since, each time in the same direction: a
defect it called unreachable turned out to be reachable. Its record on impossibility claims is 0
for 5 and is reported in the fourth correction. The attacks are all real, all reproducible from scripts under
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

**Class one — reachable, and therefore owed as work.** *The count below is seven as of the third adversarial pass and is stale: the class took the four reclassified class-two members, took everything the fifth pass added, and gave back what each later round closed. It has grown and shrunk repeatedly, and no total in this document tracks it. The current state lives in the amendment ledger and the retirement records, not here.* Seven remaining defects are predicates over
bytes already on disk: the floor census is unsigned metadata outside the tree while the function
that re-derives it exists and nothing calls it; `covers` and `not_covered` are never compared
against the plan the floor names by id; a missing issuer roster fails open instead of closed; no
cert names the log it belongs to, so "this cert is in *the* log" is uncheckable; the plan's index is
never compared against its runs'; the gap between two baselines is computable from two logged certs
and does not exist; and `subject.environment` is not required by any schema. None of these is deep.
All of them are work.

*Status as of the end of the day, since the list above is a snapshot. Several are closed. A cert may
now name its log, and the log answers the more useful reverse question — does this log hold these
bytes — comparing whole certs rather than ids, since two certs can share an id and be two entries.
The environment is observed through a runner at mint and compared against every floor cert. The
retirement ledger has moved inside the digest that protects it, so deleting a row moves
`set_sha256`. What is newly stated rather than newly fixed: an issuer who re-signs without the
log-naming field gets a cert that appends anywhere, and no predicate over these bytes prevents it,
because the field lives inside the signed core and the issuer is exactly whom it constrains.*

**Class two — the label class, reachable by nothing.** *All four members this class listed were
later reclassified as reachable (third correction), and the class was then declared empty — which
the fifth correction withdraws, because a fifth member was found that no roster here ever listed.
The class is not empty; its roster was incomplete. The paragraph is kept because the argument that
built it is the error worth reading.* Every other surviving defect is a certificate
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
(hf_repo, revision)"* is a predicate over bytes already on disk, and both halves were demonstrated:
the same `weights_sha256` under two revisions appends, and the same revision under two weights
hashes appends.

*This paragraph said "is not implemented" when it was written, and that is no longer true: the
predicate is now in the append path. Only one of its halves refuses, though. One name over two
snapshots is a contradiction in the log's own bytes and is refused; one snapshot under two names has
an innocent generator — a commit touching only files outside the hashed set — that this design
cannot tell from the guilty one, so it is reported as a disclosure and still appends. Which half
refuses is an open operator decision.*

The adversary's own caveat is the honest part, and it is why this reclassification is forced rather
than optional: that predicate does not *fully* close member 3, because renaming the directory once
makes every cert tell the same lie consistently. But that is exactly as true of the environment
member, and this lab counted that one as class one. By the standard the lab itself applied, member 3
belongs in the backlog.

**The roster fell to two members here, and to zero in the third correction — before the fifth found a member no roster had listed.** It has shrunk at
every pass that attacked it: four, then three, then two, then none. That trend is the finding, and
it should be read as a warning about this document rather than a record of progress. Each member was placed in class two by an argument that felt conclusive
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

**THIRD CORRECTION, same day, and it empties the class.** The paragraph above says the last two
members should be assumed reachable until someone has tried and failed to reach them. Someone
tried. Both are reachable, and this time the check was run against the lab's own published verdict
log rather than argued about. Receipts: `member1_structure.py`, `member1_demo.py` and
`member2_demo.py` under `class_two_empty_2026_09_09/`, all reading stored entry bytes and
importing no styxx. They run against the published log and print their own controls.

**Member 1, the batch labels, is reachable.** The roster's own demonstration is the thing that
falls. Its claim was that five relabelled copies of one forward pass and five real runs that
happened to agree exactly produce identical bytes. True of one certificate. False of a certificate
in a log that has already measured the same subject on the same battery — and the published log
has. Entry 6 records ten pairwise distances over five runs at three batch sizes, on three channels,
and the structure is not loose:

| what entry 6 records | on all three channels |
|---|---|
| pairs sharing a batch size | all 3 exactly zero |
| pairs at different batch sizes | all 7 strictly positive |
| distance as a function of the unordered batch pair alone | exact: 10 distances collapse to 4 values |
| three different item orders at batch 1 | distance 0, and byte-identical channel values |

So this subject's batch sensitivity is a logged measurement, not an assumption. A later floor
claiming that batch 1 against batch 8 separates nothing contradicts a certificate already in the
log. Built as the roster describes and checked against entry 6, the forgery is contradicted on **9 of 9
factor-level pairs across all three channels**, while the honest floor checked against itself
agrees on all 12 and a fresh subject with no logged history is correctly reported *unconstrained*
rather than accused.

*This paragraph said "refused" and that word was wrong, in the way that matters most. The predicate
lives in the independent JavaScript verifier and in the receipt script beside this document. No
verification walk calls it, no append path calls it, and there is no Python implementation at all.
The corroborating byte is real and the predicate works; the wiring does not exist. **A receipt
refuses the forgery. The system accepts it**, appends it, verifies it, and stores an empty
disagreement list beside it.* Two further constraints fall out of the same bytes: same-batch runs produced
byte-identical channel values, so determinism at a fixed batch size is logged and a fabricated run
at a batch level the log has already seen must reproduce it exactly.

The predicate was then written a second time, in the independent JavaScript verifier whose author
was forbidden to read the Python. It agrees on all four controls and reaches further than the
Python did, comparing 57 cells rather than 12 — 27 factor-level pairs and 30 assignment-level pairs
— and refusing the forgery on 45 of them, of which it marks 36 clean and 9 confounded. It adds a
fourth control the Python lacked: a floor on a *different* subject is not permitted to accuse this
one, and returns `unconstrained` rather than a verdict.

**They diverged, and the second implementation was right, which is the argument for writing one.**
The Python receipt this correction cites defined a subject-comparability gate and then never called
it, so it would have accepted a prior floor from *any* subject in the log and accused on it — a
false accusation across subjects, the defect class this lab already holds a receipt for at 0.23
precision. The JavaScript gated comparability first, through the section 2.3 predicate it already
had for challenges, and its extra control returns `unconstrained` with the differing field named.
The Python has been repaired and now reports `differs on subject.precision`; the repair took two
attempts, because the first one called the gate but keyed it on four fields not including
`precision` — which is exactly the comparison the published verdict treats as a drift candidate.
Neither error would have been found by the author who made them.

What that predicate does **not** reach is a forger who writes plausible non-zero distances instead
of zeros. So member 1 is not closed; it is *constrained by logged history*, which is precisely the
condition every class-one member is in. Member 3 was moved on a weaker showing than this.

**Member 2, the choice of baseline, is reachable.** This system already owns the mechanism for
taking a choice away from the party who benefits from it, and this document lists it among the five
things a self-written record establishes: commitment, fixed before the data existed. The published
plan fixes the factors, their levels and the count. It does not fix the schedule. **57,600 run
schedules satisfy that plan and it names none of them**, so the reference run — the one whose values
become the subject's fingerprint — is chosen from nine available cells after the plan is logged, by
the party the fingerprint is about. Carry the schedule in the plan and *"the runs' assignments are
the plan's schedule, in order"* is a comparison between two logged certificates, which is the same
shape as the predicate that reclassified member 3. The residue is identical too: a party who commits
to a favourable schedule in advance is not contradicted by anything. That is suppression, and the
lab filed member 3 under class one carrying exactly that caveat.

**So every member of class two was reachable, and the partition this document is built on does not survive.** *(This paragraph then concluded the class was empty. The fifth correction withdraws that: the roster was incomplete, and `precision` is a member nobody had listed.)* Four
members, four reclassifications, and the same mistake every time: *the argument examined one
certificate instead of the log the certificate sits in*. A log is not a pile of certificates. It is
a set of mutually constraining claims, and the constraint grows with its length. A party who lies
must be consistent not with one document but with everything they have ever signed, and they must
have been consistent before they knew which check would be written.

That does not rescue the self-written record, and the general invariant is untouched — every
predicate above asks only whether one party contradicted itself, which is the boundary this document
states. What changes is where the boundary sits. The residue is not a list of unreachable fields. It
is a single unreachable act: **the first claim about anything.** A subject nobody has measured, a
battery nobody has run, a factor no prior certificate exercised — there is nothing to contradict, and
every check is vacuous by construction. That case is control B above, and the honest predicate
declines to answer it rather than passing it.

Consistency accrues; it cannot be bootstrapped. This is the same conclusion the document reaches
about challenges, arrived at from the opposite direction, and it sharpens what the challenge is for:
a second party's bytes are the only thing that can constrain a first claim. It also promotes the
reproduction count from a nice-to-have to the quantity that says how much of a verdict was
constrained by anything at all.

A reader should extend no credit to this correction that the previous two earned back. Four
consecutive impossibility claims by this author were wrong. The prior on the fifth is poor, and the
right response to *"no check can reach this"* — including as written above — is to go and look at
the rest of the log.

**FOURTH CORRECTION, same day, and the fifth claim fell within the hour.** The residue above —
*the first claim about anything* — is too large, and the error has the familiar shape: it was
reasoned about instead of tested. Receipt: `class_two_empty_2026_09_09/residue_test.py`, run
against the published floor.

A floor's distance matrix is not free, and the constraints need no prior certificate:

| condition on one certificate | the published `exact` channel |
|---|---|
| metric axioms, including the triangle inequality | hold; **18 of 30** distinct identities are exactly tight |
| every distance is k/64, the battery being 64 items | holds; counts 2, 1, 0, 0, 3, 2, 2, 1, 1, 0 |
| realizable as item disagreements over 64 items | holds; a witness uses 3 items, 61 agreed by all five runs |

`d(run 1, run 2) = d(run 1, run 0) + d(run 0, run 2)` holds to the digit because the disagreeing
item sets are disjoint and 2 + 1 = 3. A forger who inflates one pair by editing a **cell** breaks
the triangle inequality against the rest of the matrix, and both control forgeries in the receipt
fail on that without needing a prior log.

*The published count here first read 21, which was an enumeration artifact: the receipt walked
three orientations per triple, and on a symmetric matrix two of them test the same condition, so
every apex-k identity was double-counted and the apex-i identity was never tested at all. The
distinct counts are 18, 15 and 15 of 30. The axioms held throughout; the count did not, and it was
found by an adversary rather than by its author. The receipt is repaired. And the general claim in
this paragraph does not survive the fifth correction below — a forger who substitutes a whole run
rather than editing a cell keeps every metric axiom by construction.*

**The deflation, which belongs in the same breath.** Those two controls were already refused. Round
4 of the table above made the log recompute the floor from the run certificates it names, so
distances are derived and not writable: a hand-edited matrix dies of arithmetic before these
conditions are consulted, and a forger who instead fabricates five plausible *output sets* gets a
metric, granular, realizable matrix for free, because a real computation over real sets cannot
produce anything else. What survives is the category rather than this instance, and the category was
then tested rather than asserted — `class_two_empty_2026_09_09/first_claim_battery.py`, eight
predicates over one certificate, run on all five published fingerprints. **All eight hold on all 320
items.** The two with teeth tie the emitted tokens, the per-position distributions and the sequence
score to each other by arithmetic, so none of the three can be written freely by a party who does
not recompute.

**The control forgeries were then re-run properly and the result reverses.** This paragraph first
reported four of four caught — a different token claimed as emitted, a flattering sequence score, a
confident-looking distribution, altered answer text. Every one of those four left a derived field
stale, so what was measured was carelessness. Re-run in careful form, with each derived field
recomputed the way an honest mint recomputes it, the battery catches **0 of 4**. By this lab's own
standing rule — an agreement number without its detection power is not a number — the earlier
sentence was exactly what the rule forbids, and it stood in this document for several hours. One
further predicate would catch the fourth: `output_text` must be the detokenization of `token_ids`
under the tokenizer the subject names, which holds on all 320 published items and is implemented
nowhere.

Testing it also found the hole in it. The certificate records at most eight per-position
distributions while items generate up to sixteen tokens, so for the longer items the sequence score
has nothing to be the sum of. **185 of 320 items are fully tied down; 135 are bounded on one side
only, and 965 emitted tokens carry no recorded distribution at all.** A score can be understated
there and nothing objects. That is a defect in the certificate format, it is owed as work, and it
was found by writing the checker rather than by describing it.

**And it was weaponised within the hour, which is the useful part.** A calibration of the sampled
challenge (`sampled_challenge_2026_09_09/calibration/`) generated forgery classes against the
published certificate and scored each against the free internal battery and against a re-run of *k*
items. Two classes sit at a residual miss rate of **1.000 against everything free**, and both live
in exactly this gap: `seqlp_partial_under`, a score understated in the unrecorded tail, and
`seqlp_partial_to_prefix`, *a score set to the largest value the one-sided bound permits*. The
second one is the forger reading the bound and taking every inch of slack it leaves. A third,
`topk_reshape_coordinated` — the recorded distributions rewritten and the score moved to agree with
them — also misses completely, and at full corruption only a re-run of the whole battery catches it.

The loop is worth naming because it is what this kind of work looks like when it goes right: a
checker was written, the checker's own limit was measured rather than assumed, and the limit was
then attacked and found to be a real place for a forger to stand. None of the three is caught by
any predicate over the certificate alone. All three are caught by a second party re-running the
items, which is the same conclusion the rest of this document keeps arriving at.

So the corrected residue is **a first claim fabricated carefully enough to be internally
realizable**, which is smaller than "the first claim about anything". The gap between the two is the
gap between forbidding carelessness and forbidding fabrication. Only the second would matter, and no
predicate over one certificate reaches it.

**The record, which is the finding this document should be read for.** Five impossibility claims
were made here today. Five were narrowed or withdrawn the same day, four of them after an adversary
was told to disbelieve them and the fifth after its own author tested it. That is 0 for 5, and it is
a calibration measurement about the author, not a run of bad luck. This lab already holds that an
agreement number without its detection power is not a number. The same rule applies to a claim of
impossibility: **an impossibility claim that has not survived a serious attempt to refute it is not
a finding, it is a hypothesis with an author's confidence attached.**

Every remaining sentence of the form *no check can reach this* should therefore be read as a
conjecture carrying a measured base rate of 0 for 5, including the corrected residue two paragraphs
up, which has now been attacked exactly once.

**And the residue was then measured on this lab's own flagship artifact, which turns out to be the
case it describes.** `styxx/v8/constraint.py` computes, for a certificate, how many prior logged
entries each cross-certificate predicate could have compared it against — a census, never a verdict.
Run on the published verdict log (`constraint_census_2026_09_09/`), the counts by entry are:

| entry | type | prior entries | its own material | left to constrain it |
|---:|---|---:|---:|---:|
| 0 | battery | 0 | 0 | **0** |
| 1 | prereg | 1 | 1 | **0** |
| 2 | fingerprint | 2 | 2 | **0** |
| 3 | fingerprint | 3 | 2 | 1 |
| 4 | fingerprint | 4 | 2 | 2 |
| 5 | fingerprint | 5 | 2 | 3 |
| 6 | **the verdict** | 6 | 6 | **0** |

Entry 6 is the canonical fingerprint and the noise floor — the thing the whole artifact exists to
publish. Its `refs` name the battery, the plan and all four floor runs, so **every one of the six
prior entries is its own input, and nothing is left for any predicate to compare it against. The
census is 0 on all five.** The predicates that emptied class two this morning, run against the
verdict published this morning, reach it on nothing.

The shape is the part worth reading. Entries 3, 4 and 5 do not name one another, so constraint
accrued 1, 2, 3 across the floor runs — and then the certificate that aggregates them consumed all
of them, and the count returned to zero. Consistency accrued, and the claim built on top of it
started over. A separate column counts entries signed by a *different* key, and it is 0 at every
index, because this log has one issuer: on these bytes a party is only ever compared with itself.

So the residue is not an abstraction about hypothetical first claims. It is the published verdict
this document has been reasoning about all day, and the honest disclosure beside that verdict is
that no consistency check reached it.

**Three separate lines of work reached it independently, which is the only corroboration this
document has for it.** The constraint census arrived from counting prior entries. The realizability
test arrived from asking what constrains one certificate alone. And the truncation work — a
different attack entirely, on whether a log can be shortened without anyone noticing — arrived at
the same sentence from the other end: a mirror that has never seen an earlier tree head cannot
distinguish a log that was truncated from one that was always that size, so *a first observation
establishes nothing about the past*. None of the three was looking for the others' answer. That is
weak evidence and it is the kind this document is short of, so it is recorded as weak. The general invariant is not in that category and is
not weakened by any of the four corrections: every predicate found today, including the ones that
emptied class two, asks only whether one party contradicted itself. That much has held through five
rounds and four corrections. What has never held is any statement about *which* defects have no
contradiction available.

---

## FIFTH CORRECTION — the predicate reaches the careless forger and not the profitable one

A sixth pass was asked to refute the third and fourth corrections on the published bytes. It began
by reproducing all thirty distances and all three floors from the stored entries, then broke the
central claim. Its own summary is the sentence this document has to carry:

> the document proves that a careless forger is caught and presents it as proof that a defect is
> reachable; on the published bytes the careful forger is not caught, and gets the entire benefit
> the defect was worth.

**The attack.** Build a *later* floor on the same subject under a new plan declaring batch levels 1,
8 and **64**. Four of the five runs are this lab's published bytes verbatim. The fifth, at batch 64,
is spliced from three genuine runs — the fp16 canonical, and two bf16 runs. Nothing is invented.

| | published | forged |
|---|---|---|
| exact floor | 0.046875 | **0.078125** against a distance of 0.0625 |
| seqlp floor | 0.036070694 | **0.076447918** against 0.058564664 |
| topk floor | 2.1402339 | **2.207838883** against 1.407087824 |
| the verdict on this lab's own published comparison | `exceeds_floor`, exit 2 | **`same`, exit 0** |

Every inequality strict. And every check passes: the member-1 predicate returns `agrees`; the
determinism predicate reproduces at each batch level the log has recorded and has nothing to say
about batch 64; snapshot agreement is satisfied because the subject is byte-identical; symmetry, the
triangle inequality, k/64 granularity and realizability hold on all three channels; same-batch pairs
are exactly zero, different-batch pairs strictly positive, and the distance is a function of the
unordered batch pair alone — *every structural regularity the third correction cites as the
signature of a real measurement.*

The relabel it costs is **one string.** The published bf16 and fp16 subjects differ in `precision`
and in nothing else: same weights, config, tokenizer and generation-config hashes, same revision,
same environment.

**Two things follow, and the second is the more important.**

**First, `precision` is a class-two-shaped field that appears on no roster in this document.** The
snapshot-agreement predicate correctly excludes it — a rule putting precision on the hash side would
refuse an honest bf16/fp16 pair — so nothing anywhere constrains it, and it is the pivot of every
forgery above. The roster of four was never complete, which is a worse error than any of the four
misclassifications: those were wrong entries, this is a missing one, and nothing in the method used
here would have found it.

**Second, the escape hatch is a new factor level, and that unmakes the third correction's thesis.**
A log constrains a claim only at the factor levels it has already seen. Declare a level nobody has
measured — batch 64 — and the claim is back in the first-claim case, with nothing to contradict it.
So the residue is not *the first claim about a subject*. It is **the first claim in any (subject,
factor-level) cell**, and the space of cells is unbounded because the issuer declares the levels.

That is the damaging version, and it is stated here rather than softened: **consistency never
finishes accruing, because a liar can always open a fresh cell.** The third correction's picture — a
growing record that closes in on a liar — is wrong in the direction that flattered it. The record
grows, and so does the space in which nothing has been recorded, and the issuer chooses where to
stand.

**What survives, and this document has to say it against itself: class two is not empty after all.**
The two paragraphs above cannot both stand. One says the class was emptied; the other names
`precision` as a field whose forged value is byte-indistinguishable from its honest one, with no
corroborating byte anywhere — which is the definition of a class-two member. So the class has at
least one member, and the emptying was never a result about the world. It was a result about a
roster of four, and the roster was incomplete.

That is a harder failure than the four reclassifications. Those were wrong entries on a list, found
by testing entries. This is a missing entry, and no amount of testing the entries finds it: the
method used here can only interrogate defects someone already thought to write down. Whatever else
is missing from the roster is missing for the same reason and is not discoverable by continuing to
do this.

What does survive is narrower. The *reclassification standard* holds: under this document's own
definition of class one — which already counts predicates that exist and are uncalled — moving those
four was legitimate and not a goalpost move. What is smaller than the document's tone is the consequence — **emptying class two converted
four disclosures into four TODOs**, and the sixth pass shows one of those TODOs is insufficient for
the defect it was assigned. Member 3's predicate is implemented in the append path; member 1's is
implemented nowhere, and this document's claim that *"member 3 was moved on a weaker showing than
this"* is false on the axis a reader cares about.

The fourth correction's residue survives, tested rather than asserted: a wholly fabricated answer,
tokenized and re-derived end to end, passes all eight internal predicates. One route closes — there
is no duration field anywhere, and all five runs are stamped inside a one-second window, so any
throughput predicate would accuse the honest log.

**Four further findings from the same pass, all verified.**

- **The prior the predicate leans on is itself a first claim.** Running this lab's own constraint
  census on entry 6 returns `first_claim: true`, so this document's *"this subject's batch
  sensitivity is a logged measurement, not an assumption"* overstates. It is an earlier free claim
  by the same key.
- **The honest published log already contains the member-1 shape, three times in five.** Runs 0, 3
  and 4 carry byte-identical channel values and identical item records under three different labels.
  The predicate cannot separate those from three relabellings of one forward pass, and does not
  claim to.
- **57,600 is criterion-dependent.** It is the count under the receipt's rule that every covered
  factor take at least two values; the stricter reading that every declared level be exercised gives
  22,500, and all nine cells give 59,049. The argument survives every reading and the number should
  have carried its criterion.
- **Two implementations of `item_order_sha256` disagree** inside this repository — one hashes
  newline-joined ids, the other the canonical JSON. The published certificates match the second.
  Member 2's proposed repair would false-refuse on its first day if a plan minted by one were
  compared against a fingerprint minted by the other.

**The record, updated.** Six impossibility-shaped claims, six narrowed or withdrawn. The fourth
correction said every remaining *no check can reach this* should be read as a conjecture with a base
rate of 0 for 5. It is 0 for 6, and the sixth fell in under an hour to the first adversary who was
told to disbelieve it and given the bytes.

---

## SIXTH CORRECTION — the constraint accrues in whichever direction the issuer publishes

A seventh pass ran against the repaired tree. It reports that of twenty-five attack constructions
named across the whole sequence, sixteen are now refused by rules that decide and **nine still
work**, and it adds seven more. Two of the seven land on this document.

**The predicate is not wired to anything.** Covered above: the word *refused* has been corrected to
*contradicted*, because only a receipt refuses. The same is true of the determinism comparison this
document leans on: nothing in the verification path implements it, and a log can therefore hold both
answers about one subject at one batch level at once. Four runs at batch size 8 append without
objection giving one pair at distance 0 and another at 0.5, so the log states that this machine is
deterministic and that it is not.

**And the direction of accusation is a choice the issuer makes.** This is the finding that should be
read as the end of the argument rather than another entry in it. The floor predicate accuses in one
direction only: a candidate reporting zero where a prior reported positive. That asymmetry is
deliberate and correct on its own terms, because a candidate louder than its prior is honest
re-measurement and refusing it would accuse honest labs. But *prior* means *earlier in the log*, and
the issuer writes both certificates and appends both. Demonstrated on one pair:

| comparison | verdict |
|---|---|
| honest floor, against a forged prior | **agrees**, 18 cells compared, none contradicted |
| forged floor, against an honest prior | contradicts, naming the level and both values |

**Publish the vacuous floor first and no later honest measurement of that cell ever contradicts
it.** The same holds for the constraint census, which counts only prior entries. Both quantities are
functions of publication order, and publication order is the issuer's.

So consistency does not merely fail to close in on a liar because the space of cells is unbounded,
which is the fifth correction. It fails for a second and simpler reason: **the liar publishes
first.** Every mechanism this document proposes rewards whoever writes the earliest entry about a
cell, and nothing in the design makes the earliest entry the true one. That is the same shape as
trust-on-first-use, which the prior-art check already named, arrived at from inside.

**Member 2 is reachable by a second route this document did not find**, and it is the same test a
third time. The check that a floor names as many runs as its plan declares counts them; nothing
enumerates the fingerprints in the log that name the same plan and appear in no floor. So one issuer
under one preregistration appends eight cells, reads the numbers, and builds the floor from the four
it prefers. Sixty-eight admissible four-of-eight subsets, and the floor they yield ranges **0.125 to
0.500 on the exact channel — a factor of four in the yardstick every later verdict is measured
against.** Both extremes appended, each with an empty disagreement list, while the discarded cells
sit in the same log with their own signatures. The corroborating bytes are there. Nothing reads
them.

**Three more, recorded without ceremony.** The baseline-gap repair excludes a cert's own floor runs
from the baseline search, and that exclusion list is written by the appending party: name the
baseline you are replacing among your own runs and the announcement vanishes and the log positively
asserts `first-canonical`, which is a false statement rather than a silence. A mirror's report does
not record whether an out-of-band head was supplied, so `verified: true` means either *consistent
with a head the operator does not control* or *consistent with itself*, and no reader can tell
which. And the retirement ledger's move inside the conformance digest was undone by the reader's own
compatibility path: move the ledger back under `provenance`, recompute the digest over the smaller
core, and the index is self-consistent while the reader still finds it.

*That one is closed. The fallback existed to migrate sets written before the ledger moved, and the
migration is finished: the only set in this tree carries the ledger at the top level and none under
`provenance`, so the branch had no honest consumer left at the moment it was shut. The reader now
refuses a ledger it finds only under `provenance`, and refuses an index carrying one under both keys,
since two ledgers can disagree and a reader cannot tell which is the record. Demonstrated refusing at
exit 1, and the committed set still checks clean at the same digest. What stays open, and is now
stated in the generator's own source rather than only here: an attacker who edits the ledger in place
and recomputes the digest leaves a self-consistent index, and nothing inside the file catches that.
What catches it is the digest pinned in the coverage receipt and the previous bytes in git, both
outside the artifact a stranger receives.*

**What this leaves.** Every predicate this document produced is in one of two states. Unwired, so a
receipt refuses what the system accepts. Or wired and order-dependent, so it constrains only a party
who published in the unfavourable order. Neither state is a defect in a particular function; both
follow from the same fact the document opened with, which is that the issuer writes the bytes and
now also chooses when they appear. The correct closing sentence is therefore weaker than any this
document has offered so far: **a self-written log raises the cost of lying, and the whole of that
cost is borne by a liar who was careless, or slow.**

### One thing that worked, and it is a method rather than a result

Every roster in this document was assembled by an author deciding, by inspection, which fields have
no corroborating byte. Four entries were wrong and a fifth was missing, and the missing one is the
worse failure: testing the entries on a list never finds an entry that is not on it.

There is a mechanical test that needs no list. **A field has a corroborating byte when forging it
would force some other byte in the same certificate to change. So look for two real certificates
that differ in exactly one field.** If such a pair exists, that field moved and nothing else did,
which is a demonstration on real bytes rather than an opinion. Run over the thirteen certificates on
disk in this arc (`class_two_empty_2026_09_09/lone_difference_census.py`):

| result | fields |
|---|---|
| **demonstrated lone difference** | `recipe.decoding.batch_size`, witnessed by 32 pairs; `subject.precision`, witnessed by 7 |
| varied only alongside other fields | 17, including the battery, the template hash and the seed |
| never varied at all | 14, including `weights_sha256`, `revision` and the whole environment block |

The two it names are exactly roster member 1 and the member no roster listed. It recovers the one
entry this document had right about its own shape, and the one it never thought of, from the same
bytes, in one pass, without being told what to look for.

Three things about how to use it, since this document's habit is to overclaim. It is a **lower
bound**: the fourteen fields that never varied here are unknown, not cleared, and `weights_sha256`
sitting in that column is why member 3 needed a cross-certificate predicate rather than an internal
one. It says nothing about whether a field matters, only about whether the bytes constrain it. And a
field it names may still be constrained from **outside**: `precision` is checkable by anyone who
re-runs the battery, which is the challenge, which remains the only part of this design that
introduces a byte the issuer did not write.

The practical difference between the classes was to have been what a reader does about them: class
one a backlog, class two a permanent disclosure beside every verdict. With class two empty, the
disclosure that belongs beside a verdict is a different and more useful one — **how much prior
logged material constrained this claim**, which is zero for a first claim about a subject and grows
with the subject's history. A verdict carrying no such history is a first claim, and every
consistency check on it is vacuous whatever it reports.

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
**Measured, and it deflates the obvious next move.** Reading the three consequences above, the
natural inference is that a challenge is expensive and the work owed is to make it cheap. A sampled
design was commissioned on that premise: re-run *k* of the 64 items rather than all of them, with
the sample fixed by the tree head that already commits to the certificate, so neither party picks
it. Then the battery was actually re-run and timed, for the first time in this project's history
(`reproduction_2026_09_09/`).

| | |
|---|---|
| the 64 items, one runner call at batch 1 | **49.9 seconds**, 0.78 per item |
| the same 64 items as 64 separate calls | 51.7 seconds |
| end to end including model load | about 75 seconds |
| GPU memory, peak allocated | about 5.0 GiB |
| items reproducing bitwise against the published cert | **64 of 64**, on output digests, token ids, `n_generated`, `seq_logprob` and `topk` |

The re-run also produces the Appendix A.3 exact-channel hash the certificate carries, digit for
digit. It is the same box, the same session and the same operator, so it is a re-run and not a
second party, and it establishes that the mechanism runs end to end rather than that anyone else
agrees.

**A challenge to this verdict costs about a minute of compute on a laptop GPU. The reproduction
count is still zero.** So the barrier to a second party was never the arithmetic, and a design that
reduces a one-minute cost to a ten-second one addresses a problem nobody had. The sampled design is
kept, because a battery is not always 64 items and the selection rule is the interesting part of it,
but it is not the repair this document's conclusion calls for.

What the barrier actually is, stated as the uncomfortable version: nobody else has the harness
installed, the weights pinned at that revision, or a reason to spend the minute. Those are
packaging problems and social problems, and this session produced no evidence about either. The
honest form of consequence 3 is therefore sharper than it was written — the engineering was never
what stood in the way, which removes the last excuse available to it.

3. **A design that cannot attract a second party has not failed at engineering.** It has failed at
   the only thing that would have made the engineering mean anything.

## What this does not say

It does not say the attacks are unfixable — five of the twelve named across three passes are now
refused by rules that decide rather than report. It does not say self-written records are worthless;
the list above is the argument against that. It does not claim novelty, and that is now checked
rather than asserted — see `PRIOR_ART_constraint_accrual_2026_09_09.md`, whose verdict is that
**nothing in this document's finding is new as an idea**:

- *A record constrains its author, and the constraint grows with the record.* This is Barton and
  Simko's thesis in accounting (2002), where every earnings-increasing choice also lands on the
  balance sheet and caps the next one — published twenty-four years earlier, with an estimator and a
  sample, which we do not have. It is CONIKS' stated design goal in 2015: equivocation must be
  maintained forever or be detected. It is the premise of analytical review in auditing, and the
  mechanism is Crosby and Wallach's history trees (2009) and Schneier and Kelsey before them.
- *The residue is the first claim.* This is trust-on-first-use, deployed in SSH since the 1990s. The
  closer structural analogue is weak subjectivity in proof-of-stake: a node with no prior state
  cannot distinguish the real history from a costlessly simulated one, because both are internally
  consistent. Our only difference is a reporting choice — we return `unconstrained` where TOFU
  accepts and binds.
- *Using a prior measurement of a nuisance factor to refuse a fabricated later one.* Fisher did this
  to Mendel in 1936, and his version is the superset: he caught dispersion that was implausibly
  small, where ours only catches dispersion of exactly zero. Standardised since as ISO 13528
  proficiency testing, and as Westgard rules on control charts, which flag a run whose dispersion is
  anomalously small. Carlisle does it to clinical trial baseline tables.
- Forensic science already names our threat model. *Dry labbing* is reporting a result for a
  procedure never run, and its literature makes our point in one sentence: a falsified result can
  sometimes be caught by re-examining the underlying materials, and a dry-labbed result has no
  underlying materials to re-examine.
- The ceiling is old too. A consistent set of sentences is satisfiable, not true. That is the
  circular horn of Agrippa's trilemma, and it is the formal reason this document's invariant can
  never be upgraded from *raises the cost of lying* to *detects lying*.

**And one citation cuts against us**, which is the most useful thing the search returned. In the
deception-detection literature, consistency across statements is the most-used cue to deceit and one
of the worst-performing; the finding there is that *checkability* — can a third party verify this
detail — discriminates better, and that observers who substitute consistency for verifiability do
worse. This document argues its way to the same place in the section below, and should say plainly
that a measured literature got there first. Forensic statistics states the resulting doctrine
directly: internal consistency is what you assess **in the absence of an external source**.

So the contribution here is an instance, not an idea: this particular predicate, on these
certificates, with its receipts. The four impossibility claims that fell were four of our own
errors, which is a report and not a result. And it is not a proof of anything — it is five rounds on one machine in one day,
where the same adversary attacked its own previous conclusions, which is a method with obvious
limits and one virtue: nobody here was trying to make the system look good.

## Limits

One lab, one operator, one day, one model, one battery of 64 prompts. Every attack and every repair
was produced by the same session that is now reporting on them, and none has been reviewed by anyone
outside it. That session's impossibility claims ran 0 for 5, so the parts of this document a reader
should trust least are exactly the parts that sound most conclusive. The invariant is an observation over five cases, not a theorem; a sixth repair might
cross the boundary rather than move it, and this document would be wrong. That would be the best
outcome available and nothing here should discourage the attempt.
