# PREREG — precision against labels we did not produce

Fathom Lab · 2026-09-01 · Frozen before any join is computed, before the overlap is counted,
and before a single one of their labels has been read against a single one of our
accusations. Successor to `RESULT_v14_naming_the_defects_did_not_save_it_2026_09_01.md`,
which measured the repaired path-claim accuser at **0.16 precision against a 0.95 floor** on
held-out prose (`v14_adjudication.json`: 100 scored, 16 upheld, 30/30 decoys) and retired the
class rather than repair it a fourth time. Corpus and split governed by
`SPLIT_external_corpus_2026_08_31.md`.

**The standing commitment this document is written under:** do not ship an accusing verdict
whose precision has not been measured by a blind panel. Absence of evidence is never a
contradiction. UNCHECKABLE is a first-class verdict here and is printed loudly, not hidden.
Never "first". Never "nobody". Always "we know of no other."

---

## Why this design is stronger than every precision this lab has published

Every precision figure this lab has ever reported — 0.23 in EXTERNAL-1
(`external1_adjudication.json`), 0.16 in V14 (`v14_adjudication.json`), and every OATH battery
before them — was produced by a panel **we convened, instructed, seeded, and sealed**. The
protocol is good: blind items, sealed decoys, a reliability gate, a key digest committed
before judging. It is still our protocol, run on our sample, against our notion of what
"upheld" means. A reader who believes we are fooling ourselves has no lever to check it with
except our own receipts.

The PR-MCI team published **974 human-annotated pull requests** with consensus three-way
labels — aligned / partial / misaligned — over the same AIDev corpus this lab already has on
disk (`external_citations.json`, transcribed and recomputed from the authors' published files
on 2026-09-01). Those labellers have **no stake in our instrument**. They had never heard of
it. They were not shown our verdicts, our sentences, or our reasons, and they could not have
been, because the labels predate our interest in them.

That removes us from the adjudication entirely, which is the one thing our own protocol can
never do. **It is a strictly stronger design for the adjudication leg, and it is strictly no
help at all for the extraction leg** — a distinction this preregistration takes seriously
enough to carry a gate for it, below.

**Stronger in the one dimension named, and not sufficient.** `ANALYSIS_base_rate_ceiling_2026_09_01.md`
§9 already published the countervailing fact about these same labels, and this document does not
get to quietly drop it: the annotators are **the authors of the instrument the annotations were
built to validate**, so on that paper's own words, *any precision figure derived from them does
not satisfy this lab's standing commitment* to a blind panel. Both sentences are true at once.
What this design buys is disinterest **relative to us**; what it does not buy is a blind
adjudication in the sense our standing commitment means. **A number out of this design is
therefore never on its own a satisfied standing commitment**, which is the substantive content
of G-J8 and is stated here in plain words rather than left to be inferred from a gate.

It is not, and must never be written as, ground truth. Section *Honest limits* is the price
list.

---

## The two corpora, and the licence boundary

**Ours.** The AIDev shelf EXTERNAL-1 built: `external1_shelf.sqlite`, table `pr`
(`id`, `agent`, `title`, `body`, `html_url`, `state`, `merged_at`) and table `f`
(`pr_id`, `filename`, `status`, `patch`). Structure read before this freeze, not outcomes:
**71,677 shelved pull requests** across **6,673 distinct repository URLs**, of which
**71,016 are eligible** under the EXTERNAL-1 eligibility rule and are the population every
published number of ours is computed over (`v14_gates.json` `prs_scored`, reconciled by
`flag_rate.json`).

**Theirs.** Gong, Pinna, Bian, Zhang, *Analyzing Message-Code Inconsistency in AI Coding
Agent-Authored Pull Requests*, MSR'26 Mining Challenge Track, DOI 10.1145/3793302.3793583,
arXiv:2601.04886v2, paper CC BY 4.0. Replication repository `gjz78910/PR-MCI`. Their analysed
set is **23,247 pull requests**, derived from an AIDev-pop subset (33,596 PRs across 2,807
repositories with more than 100 stars) filtered to closed PRs under a permissive licence.
Their annotated set is **974 pull requests**: a 600-PR validation sample plus **374 additional
high-MCI PRs** selected by their own detector. All figures in this paragraph:
`external_citations.json`, second-hand.

**The licence boundary, which is a gate and not a footnote.** The GitHub API reports
`license: null` for `gjz78910/PR-MCI` and there is no LICENSE file in the tree. The paper is
CC BY 4.0; the replication repository carries no explicit grant. **Their files may be read and
cited. They may not be copied into this repository, mirrored, embedded in a capsule, quoted at
length in a receipt, or redistributed in any artifact of ours.** Their files are read from a
scratch path outside this repository. What enters our receipts is derived counts, our own item
identifiers, and citations.

---

## THE JOIN

### The key

Both sides identify a pull request by its GitHub URL. The join key is the normalised triple
derived from it:

```
key(pr) = (owner.lower(), repo.lower(), int(number))
```

parsed from `https://github.com/<owner>/<repo>/pull/<number>`, scheme and host discarded, any
trailing slash or query string discarded, `owner` and `repo` lowercased, `number` parsed as an
integer with no leading-zero tolerance. Our side supplies it from `pr.html_url`
(`external1_shelf.sqlite`). Their side supplies it from whichever published column resolves to
a pull request URL or to a repository-plus-number pair.

**A confirmatory second key.** Our shelf carries the GitHub pull request id integer as
`pr.id` (e.g. `2756921963`). Where their published files carry a comparable identifier, the
two keys **must agree on every joined pair**. A pair whose keys disagree is voided, not
resolved by preference, and **the count of voided pairs publishes**.

**If their published files carry no resolvable pull request identifier at all, the join is
UNCHECKABLE and publishes as UNCHECKABLE.** We do not reconstruct identity from titles, bodies,
diffs, or timestamps. This lab has measured twice what happens when it infers structure from
strings a stranger wrote.

### The direction, and the three populations

- **Population A — precision (primary).** Our accused pull requests ∩ their 974 annotated
  pull requests. This is the only population that yields a precision.
- **Population B — instrument agreement (report-only).** Their 406 detector flags ∩ our accused
  set. Two independent instruments over one corpus; a count, never a precision, and never an
  argument that either is right.
- **Population C — the recall side (report-only).** Their annotated pull requests labelled
  positive that we did **not** accuse. A count of what we missed. Recall is not gated here and
  no recall figure from this design may be reported as the instrument's recall, because their
  annotation is not exhaustive over our claim class.

### Diagnostics that publish before any precision

The join can fail for two very different reasons and they must not be confused:

1. **Corpus mismatch** — their pull request is not in our shelf at all, because the two
   projects read different AIDev snapshots. **The PR-MCI paper names no AIDev version, tag, or
   snapshot date anywhere in its text** (`external_citations.json`, `unverified`), so this
   failure mode is live and cannot be ruled out in advance.
2. **Eligibility mismatch** — their pull request is in our shelf but outside our eligible
   71,016, most often because it has no reconstructable changed-file record.

Published before any precision, and published whatever the precision turns out to be:
|974 ∩ shelf|, |974 ∩ eligible|, the count failing for reason 1 split by *repository absent*
versus *repository present but PR absent*, and the count failing for reason 2 with the
eligibility reason attached.

### What the overlap is expected to be, and why that expectation is grim

**Derived here from published receipts; no new measurement, and nothing below is a result.**

Our post-repair accuser fires on **1,197 of 71,016 eligible pull requests** — a per-PR flag
rate of **0.01686** — issuing **1,344 path accusations**, i.e. 1.123 accusations per accused
pull request (`flag_rate.json`). Applying that rate to their annotated set gives the
orientation figure that governs this whole design:

| their annotated subset | expected joined accused PRs, post-repair | of which HELD-OUT |
|---|---|---|
| all 974 | ≈ 16.4 | ≈ 12.8 |
| 974 less their ≥230 zero-diff flags | ≈ 12.5 | ≈ 9.8 |
| the 600-PR validation batch alone | ≈ 10.1 | ≈ 7.9 |

The held-out column applies the corpus's own held-out share, 55,399 / 71,016 = 0.7801
(`flag_rate.json`, `SPLIT_external_corpus_2026_08_31.md`). The second row exists because
**230 of their 406 flags are zero-diff pull requests** (`external_citations.json`,
`the_two_arms`), and a pull request with no reconstructable changed-file record is outside our
eligible set by EXTERNAL-1's own rule. Every one of these figures assumes their annotated set
is present in our shelf at the corpus-wide rate, which is exactly what the diagnostics above
exist to check and what section *Honest limits* says we cannot assume.

**This is an expected overlap in the low tens at best, and plausibly under ten on the primary
arm.** That is stated now, before counting, so that a thin overlap cannot later be presented as
an unlucky surprise.

The unrepaired configuration is the one arm with any prospect of power. The unrepaired gate
issues **4,427 accusations** corpus-wide against the post-repair 1,344 (`v14_gates.json`).
*Assuming* the same 1.123 accusations per accused pull request — an assumption, not a
measurement, and marked as one wherever it is used — that is ≈3,943 accused pull requests, a
flag rate of ≈0.0555, and an expected overlap of ≈54 annotated pull requests (≈42 held-out;
≈32 held-out after removing their zero-diff flags).

**Both arms are preregistered now, before we know which one has the items.** Choosing the arm
after seeing which one reaches a power threshold is the same degree of freedom as choosing the
label mapping after seeing the numbers, and it is closed here for the same reason.

### The rule when the overlap is too small

**If the overlap is too small to measure, the finding is that it is too small.** We publish the
overlap, the diagnostics, and the sentence *the strongest measurement available to this lab
does not reach this instrument*. We do **not** convene a panel of ours and report its number
under this preregistration's name. We do not widen the join, relax the key, drop the held-out
restriction, pool the arms, or reach for the development bucket to make a number appear. A
smaller panel quietly substituted for a failed third-party join would be the exact move this
document exists to make impossible.

---

## THE LABEL MAPPING, decided now

Their positive class is **misaligned**. **partial** is a genuine third state carrying real
mass: of their 432 partial-or-misaligned pull requests, **167 are partial and 265 are
misaligned** (`external_citations.json`). There is no principled mapping from their *partial*
to our panel's upheld / not-upheld, and there will not be one after we have seen the numbers
either.

**Committed now, and both are reported always:**

- **STRICT** — an accusation is upheld iff the consensus label is `misaligned`. `partial` and
  `aligned` are not upheld.
- **LOOSE** — an accusation is upheld iff the consensus label is `misaligned` **or** `partial`.

**Why both, and why neither may ever appear alone.** On their own instrument the choice moves
the answer by a factor of four: their non-trivial arm scores **0.136 strict and 0.591 loose**
over the same 176 pull requests. **Citation defect, recorded rather than smoothed over:**
`external_citations.json`, `the_two_arms` carries the strict figure
(`arm_b_non_trivial_strict_precision`) and **does not carry the loose figure at all**. The
0.591 is published only in the §9 table of `ANALYSIS_base_rate_ceiling_2026_09_01.md`, a
document that states in its own opening that it "creates no new receipt of its own". So this
number is currently asserted in prose with no receipt behind it, which is the exact defect
that paper's own certificate confesses to. **`external_citations.json` gains an
`arm_b_non_trivial_loose_precision` field before this preregistration's result publishes**,
and until it does, the loose arm figure is cited here as UNRECEIPTED. A mapping chosen
after the numbers are visible is not a mapping, it is a result. **Every sentence, table,
abstract, and headline that carries a precision from this design carries the strict figure and
the loose figure together, in that order.** A document of ours containing one without the other
has failed G-J3 and does not publish.

**The construct-matched reading is DEFERRED, not omitted.** Their taxonomy contains a category
that is much closer to our construct than generic misalignment — *Incorrect Claims* /
`misleading_specifics`, **13 of 432** — alongside `phantom_changes` (196 of 432, of which
194 are literally empty diffs, `external_citations.json`) and `understated_scope`. **Same
citation defect as above:** `external_citations.json` carries the `phantom_changes` block
but has **no taxonomy field for `misleading_specifics` and none for `understated_scope`**;
the 13/432 is published only in §6 of `ANALYSIS_base_rate_ceiling_2026_09_01.md` and the
38.8% `understated_scope` figure survives only inside a prose caveat string in the
`phantom_changes` block. Both gain their own receipt fields before any result cites them. A
category-restricted mapping would be the sharpest reading of all, and we are not preregistering
it today because we have not read their full taxonomy vocabulary and will not name a category
set we have not seen. It may be added **only** by a dated amendment that (a) names the admitted
categories exhaustively, (b) is frozen before the join is run, and (c) is published whether or
not it helps. Absent that amendment, this reading does not appear in the result at all.
**On the counts already known it would be underpowered by an order of magnitude — 13 pull
requests corpus-wide — and it may be reported as a count, never as a precision.**

---

## THE UNIT MISMATCH, handled rather than hidden

**Their label is per pull request. Our accusation is per path claim.** These are different
units and no join makes them the same unit.

**The conversion, fixed now.** Our accusations are lifted to the pull request level: a pull
request is *accused* if it carries at least one path accusation, and it is scored once. Their
label attaches to the pull request already. The primary quantity is therefore

```
precision_PR = |accused ∧ annotated ∧ label positive| / |accused ∧ annotated|
```

We do not lower their labels to the claim level, because that would require re-annotation by
us, which reinstates our own panel and destroys the only property this design was chosen for.

**What the conversion costs, stated in the direction it costs it:**

1. **False credit, and it runs in our favour.** A pull request may be labelled misaligned for a
   reason entirely unrelated to the path our sentence named. The join scores our accusation as
   upheld anyway. This inflates our number.
2. **Resolution loss.** Corpus-wide, 1,344 accusations collapse onto 1,197 pull requests
   (`flag_rate.json`); held-out, 951 onto 858. Roughly one accused pull request in nine carries
   more than one accusation and they are scored as one item.
3. **A deflating direction exists and is weaker.** A pull request whose description is
   misleading about one file but sound overall may have been labelled aligned, scoring a
   correct accusation of ours as wrong. Their annotation instrument judges the whole
   description, so this is real; we judge it the smaller of the two effects and we do not know
   that, and saying we do not know it is the point.

**The discipline this buys, committed now: every precision produced by this design is a
per-pull-request figure and is an UPPER BOUND on our per-claim precision, and it is printed
with the words "per-PR, upper bound" attached to it.** Not in a footnote. In the sentence. It
is not comparable to the 0.16 of `v14_adjudication.json`, which is per claim, and no document
of ours may chart the two together or describe either as a movement from the other.

**One report-only concordance measure.** Their per-PR packets carry each annotator's free-text
reason. For each joined item we record, mechanically, whether the accused path string appears
in the consensus reason text. This is a **count**, published as a count, and it is **not** a
claim-level precision, **not** a correction factor for effect 1 above, and may not be used to
adjust any figure. It exists so a reader can see how often the human was looking at the same
file we were.

---

## THE ARM SPLIT — their decomposition applied to us

Their headline blends two instruments with wildly different difficulty: accusing a pull request
that changed no files at all scores **0.965** over 230 items; accusing one that contains a real
diff scores **0.136 strict** over 176 (`external_citations.json`, recomputed from their
published files, not a number their paper reports).

We will not repeat that blend. **Joined items are split by whether the pull request has a
non-empty reconstructed diff on our side, and the two arms are reported separately and never
pooled without both being visible.** The headline is the non-trivial arm.

An expectation, recorded before counting: our eligibility rule requires at least one
reconstructable changed-file record, so the trivial arm should be close to empty. **If it is
not empty, that is a finding about our eligibility filter and is published as one** — it would
mean our corpus admits pull requests whose diff is effectively absent, which would matter to
every number EXTERNAL-1 and V14 published.

---

## The reporting grid, fixed now

Three binary dimensions, all preregistered, all published in one table in the result:

- arm: **post-repair (V13+V14)** × **unrepaired**
- mapping: **strict** × **loose**
- split: **HELD-OUT** × **DEVELOPMENT**

and within each, the trivial and non-trivial diff arms shown separately. **Every cell publishes,
including the empty ones and the underpowered ones, with its N.**

**The headline cell is named now: post-repair × strict × HELD-OUT × non-trivial diff.** That is
the cell that corresponds to the instrument RESULT_v14 measured, on the split
`SPLIT_external_corpus_2026_08_31.md` rule 2 requires, under the conservative mapping, on the
arm that is not trivially easy. Pooled figures may appear only beside the grid, never instead
of it.

---

## THE EXTRACTION GATE — the leg this lab has never measured

Every precision this lab has published measures **adjudication**: given that we called this
sentence a claim about this path, was our verdict right? Not one of them measures
**extraction**: was the sentence a claim about that path at all?

The two compose, and the composition is the whole of what a reader means by precision:

```
P(accusation correct) = P(the sentence asserted this path changed) x P(verdict right | it did)
```

A sentence that asserted nothing yields a wrong accusation no matter how good the adjudicator
is, so **precision is bounded above by extraction validity**. We have been reporting the second
factor and letting it be read as the product. `PREREG_evidence_leg_2026_09_01.md` already
commits this lab to two panels rather than one before any future accusing verdict; this
document carries the extraction panel itself, and we know of no earlier preregistration of ours
that did.

**Third-party labels cannot measure this leg.** Their annotation is about a pull request, not
about our sentence; there is no claim-level or sentence-level annotation anywhere in their
corpus. So the extraction leg is measured by a panel of ours, and the honest reading of this
document is that **its adjudication leg is third-party and its extraction leg is not** — which
is the weakest joint in the design and is named here rather than discovered later.

**The protocol, frozen.**

- **Items.** Every joined accusation in the grid, at claim level rather than PR level — the one
  place the per-claim unit survives. If the joined set exceeds 100 claims, a uniform random
  subsample of 100 with **seed 20260901**, committed in this document.
- **The diff is WITHHELD.** The seat sees the sentence and the claimed path and nothing else.
  A seat that can see the diff will answer the adjudication question instead of the extraction
  question, and the two legs would stop being separable.
- **The question, exactly one per item.** *Does this sentence assert that this path was changed
  by this pull request in the way named?* Answers: **ASSERTS** / **DOES NOT ASSERT** /
  **CANNOT TELL FROM THE SENTENCE ALONE**. CANNOT TELL counts as not-extracted — the
  conservative direction, against us.
- **Three seats per item, majority.** Any item returning fewer than three seats is disclosed
  with its item id and its seat count, in the result, by name. V14 shipped three items
  adjudicated by two seats and disclosed it after the fact; here the disclosure obligation is
  written before the batches run.
- **30 sealed decoys**, shuffled in: 15 constructed sentences that plainly assert a path
  changed, 15 that plainly mention a path without asserting any change to it — the
  mention-versus-use boundary this lab has now found in four other instruments and in its own
  OATH verifier. **Fewer than 27 of 30 correct voids the extraction measurement with no headline
  number**, exactly as in EXTERNAL-1 and V14.
- **Key digest committed publicly before any item is judged**, the `v14_key_digest.txt`
  pattern.

**Threshold, fixed now: extraction validity ≥ 0.90.** Below that, more than one accused
sentence in ten was not making the claim we accused it of, and the instrument is defective in a
way no adjudication figure can repair. Below that, the adjudication figure still publishes but
**publishes conditioned** — as a precision over extracted claims, with the extraction validity
printed immediately beside it and the product printed after both.

---

## Power, and an asymmetry recorded before the counting

Exact binomial arithmetic against the 0.95 floor, derived in this document, computed before any
item exists:

- **Refuting the floor is cheap.** At N = 12 joined items, any observation at or below 9 upheld
  (0.75) rejects `p ≥ 0.95` at one-sided α = 0.05 (P(K ≤ 9 | n=12, p=0.95) = 0.0196). At N = 20
  the rejection region reaches 16/20; at N = 30 it reaches 25/30.
- **Establishing the floor is expensive.** A one-sided 95% Clopper–Pearson lower bound reaches
  0.95 only on a **perfect record of at least 59 items** — 0.05^(1/59) = 0.9505, while
  0.05^(1/58) = 0.9497. Fifty-eight consecutive upheld accusations are not enough.

**This design can fail our instrument on a dozen items and cannot vindicate it on fewer than
fifty-nine.** That asymmetry is a property of the arithmetic, not of anyone's intentions, and it
is written down now so that a thin overlap producing a refutation is not read as a rigged
result, and a thin overlap producing a clean record is not read as a vindication.

---

## Gates — thresholds committed now

- **G-J0 (provenance and licence), blocking.** The join runs against a **pinned commit SHA** of
  `gjz78910/PR-MCI`, recorded in the result before the run. Their files are read from a scratch
  path outside this repository. **No file of theirs, and no row, packet, or annotation text of
  theirs, is committed to this repository, embedded in a capsule, or redistributed in any
  artifact of ours.** Receipts carry derived counts, our item ids, and citations. A violation
  means the measurement does not publish at all.
- **G-J1 (join integrity), blocking.** The key is the normalised `(owner, repo, number)` triple
  above. Where both sides carry a pull request id, the two keys must agree on every joined pair;
  disagreements void the pair and the voided count publishes. **If their files carry no
  resolvable identifier, the join publishes UNCHECKABLE** and no precision appears.
- **G-J2 (overlap tiers), blocking on what may be printed.** Let N be the joined item count in
  the headline cell. **N ≥ 30**: a precision point estimate publishes, with an exact
  Clopper–Pearson interval. **12 ≤ N < 30**: the counts and the interval publish; **no point
  estimate appears in any headline, abstract, or summary line**. **N < 12**: no precision of any
  kind, in any cell; the overlap is the finding. No panel of ours is substituted under this
  document's name at any tier.
- **G-J3 (mapping discipline), blocking on publication.** Strict and loose appear together
  everywhere a precision appears. A draft carrying one without the other does not ship.
- **G-J4 (unit discipline), blocking on publication.** Every precision from this design is
  labelled **per-PR, upper bound on per-claim precision**, in the sentence carrying it. No
  document of ours charts it against the per-claim 0.16 of `v14_adjudication.json` or describes
  a change between them.
- **G-J5 (arm split).** Trivial-diff and non-trivial-diff arms reported separately with their
  Ns. Pooled figures may not stand alone.
- **G-J6 (extraction), blocking on any revival.** Extraction validity **≥ 0.90**, measured by
  the panel above, with **≥ 27 of 30 decoys correct** or the extraction measurement voids with
  no headline. Below 0.90, the adjudication figure publishes conditioned and the class stays
  retired regardless of what the adjudication figure says.
- **G-J7 (floor claims).** A statement that the 0.95 floor is **refuted** must print the exact
  one-sided p-value beside it. A statement that the floor is **met** requires a one-sided 95%
  Clopper–Pearson lower bound at or above 0.95 — at minimum a perfect record over 59 items.
  Nothing in between licenses either sentence.
- **G-J8 (this measurement cannot revive the class on its own).** RESULT_v14 retired the
  path-claim accuser and committed this lab to not repairing it again. **A favourable result
  here does not re-enable it.** Re-enabling requires all of: the headline cell clearing G-J7's
  *met* condition, G-J6 passing at ≥ 0.90, and a **new preregistration** written after this
  result is published and frozen before that code is written. The four `xfail(strict=True)`
  markers guarding the catches EXTERNAL-1 gave up are revisited in the same commit as any
  re-enabling, so that it cannot happen silently.

---

## What result means we do not ship

- **The join is UNCHECKABLE** (no resolvable identifier on their side): we publish that, and
  the strongest measurement available to this lab is recorded as unavailable.
- **N < 12 in the headline cell**: no precision publishes. The overlap is the finding, with the
  diagnostics attached, and this preregistration closes without a number.
- **The headline cell refutes the floor** (which the expected overlap makes the most likely
  outcome that produces a number at all): the class stays retired, and this is **not** a fourth
  repair cycle. There is no repair proposal in this document and none may be attached to its
  result.
- **Extraction validity below 0.90**: the class stays retired regardless of the adjudication
  figure, and the finding is that our sentences were not the claims we treated them as.
- **The licence question is not settled in our favour**: nothing publishes that reproduces any
  part of their data, whatever it would have shown.
- **Their labels turn out unresolvable to a consensus per pull request** (missing, conflicted,
  or absent for part of the 974): the affected items are excluded, counted, and published as
  excluded, and an excluded item is never scored as a pass.

---

## The commitment to publish the failure

In the words the preregistrations before this one used, unchanged:

**Both outcomes publish under the same seal as a success, per the charter, and the failure
capsule ships next to the others.** A failure publishes **under its own name, at the same speed
as the failures before it**, with the counts attached. The result of this preregistration is
written and published whether the third party's labels uphold this instrument, refute it, or
never reach it at all — and the most likely of those three, on the arithmetic above, is the
third.

---

## Honest limits

Every one of these is load-bearing and none of them is discovered later.

- **Their labels are not ground truth.** They are two annotators' consensus, produced by the
  authors of the instrument the annotations were built to validate. Third-party relative to
  *us* is the property we are buying; disinterested in absolute terms is not a property anyone
  has here.
- **Their annotators saw a packet, not the repository.** Each per-PR packet carries the
  description and the unified diff. An annotator could not check whether a path exists
  elsewhere in the tree, could not run anything, and could not consult history. Their view is
  strictly narrower than the view our accusation implicitly claims to have, and on some items
  that narrowness will cut against us and on others for us.
- **`partial` is doing real work.** 167 of their 432 positives. The strict/loose spread is not a
  robustness check we are performing out of caution; it is the honest width of the answer.
- **Their annotated set is enriched by their own detector.** 374 of the 974 are that detector's
  flags, and the enrichment is severe: their headline phantom-changes rate of 45.4% falls to
  9.7% inside the unenriched 600-PR validation batch (`external_citations.json`). **Joined
  items are therefore reported split by source batch — validation-600 versus enriched-374 —
  and never pooled without the split visible.** A precision computed over an enriched population
  is not a corpus precision and will not be called one.
- **Their repository carries NO LICENSE.** Read and cite; never redistribute. This constrains
  what our result can show a reader, and the constraint is real: a reader wanting to check our
  join must fetch their files themselves.
- **A discrepancy in their Table 1 that we noticed and have not resolved with them.** The paper
  states an operating threshold of 0.61 while printing precision 0.742 / recall 0.548 /
  F1 0.630, which their own `validation/validation_metrics.json` labels as the 0.60 operating
  point; at the deployed 0.61 our sweep reproduces 0.719 / 0.548 / 0.622
  (`external_citations.json`, `THRESHOLD_DISCREPANCY`). We cite 0.742 **as reported**, never as
  the deployed operating point. We have not contacted the authors, we do not assert which they
  intended, and this observation is not evidence about the quality of their annotations, which
  is the only part of their work this measurement depends on.
- **Their AIDev version is unnamed.** The paper names no version, tag, or snapshot date. A join
  failure caused by a snapshot mismatch is indistinguishable from a genuine non-overlap except
  through the diagnostics above, and even those cannot fully separate the two.
- **Whether their annotators were blind to their detector's score is unverified.** Their
  sampling script says scores were used for stratification only, and three packets our sweep
  read carried no score — three of 974. The 374 batch consists entirely of flagged pull
  requests, which is itself a selection.
- **Our accused set is a post-repair residual.** Two accusation-removing repairs, designed on
  the development bucket, ran before the accusations this join will score existed. If those
  repairs generalised at all, the survivors are enriched for the false-accusation shapes we
  could not name (`ANALYSIS_base_rate_ceiling_2026_09_01.md`, §7). That is a reason this design's
  number could be low that has nothing to do with their labels.
- **The extraction leg is ours.** The design's central virtue does not extend to it, and G-J6 is
  measured by a panel with exactly the stake this document was written to escape.
- **This measures one snapshot, of five agents, in one window, on one corpus**, and it says so
  wherever the numbers appear. It says nothing about whether the changed code is correct.
  Coverage is not correctness; correctness was never the claim.

---

## Receipts this document stands on

Every threshold above is fixed by the figures below and by nothing else. No figure below is a
result of this preregistration, which has none.

**Ours, published and certified**

- `v14_gates.json` — 71,016 PRs scored; 4,427 unrepaired accusations, 1,344 after V13+V14;
  development bucket 15,617 PRs, 1,299 → 393.
- `v14_adjudication.json` — held-out, 100 scored, 16 upheld, precision 0.16, 30/30 decoys,
  `G_S3_pass` false.
- `external1_adjudication.json` — 100 scored, 23 upheld, precision 0.23, 30/30 decoys.
- `RESULT_v14_naming_the_defects_did_not_save_it_2026_09_01.md`,
  `PREREG_v14_repair_2026_08_31.md`, `PREREG_external1_aidev_2026_08_31.md`,
  `SPLIT_external_corpus_2026_08_31.md`, `PREREG_evidence_leg_2026_09_01.md`.

**Ours, uncertified, reconciling with the above**

- `flag_rate.json` — per-PR flag rates by split; corpus-wide 1,197 accused of 71,016 (0.01686)
  and 1,344 accusations; held-out 858 of 55,399 (0.015488) and 951 accusations; development 339
  of 15,617 and 393 accusations.
- `external1_shelf.sqlite` — 71,677 shelved pull requests, 6,673 distinct repository URLs, join
  fields `pr.html_url` and `pr.id`. **Structure, read before this freeze; not an outcome.**

**External, second-hand, transcribed or recomputed by our sweep from the authors' published
files on 2026-09-01, and recorded in `external_citations.json`**

- 23,247 analysed PRs; 974 annotated (600 validation + 374 enriched); 432 partial-or-misaligned
  (167 partial, 265 misaligned); 406 detector flags, all annotated.
- Arm decomposition: zero-diff arm 230 items at 0.965; real-diff arm 176 items at 0.136 strict.
- Taxonomy: 196/432 phantom_changes with 194/196 literally empty diffs.

**External, second-hand, NOT PRESENT IN `external_citations.json` — asserted in prose only,
and each must gain a receipt field before this preregistration's result publishes**

- The real-diff arm's LOOSE precision, 0.591 over 176 items.
- The taxonomy counts 13/432 `misleading_specifics` and the 38.8% `understated_scope` share.

These are load-bearing here — the first fixes the width of the strict/loose spread that G-J3
exists to enforce, the second sizes the deferred construct-matched reading — and a
load-bearing number whose only home is prose is the defect this lab keeps finding in itself.
- Threshold/metric discrepancy: 0.61 stated, 0.742/0.548/0.630 printed, 0.719/0.548/0.622
  reproduced at 0.61.
- Repository licence: none found.

**Derived in this document from the receipts above, and marked as derived wherever used**

- Expected joined-accusation counts (≈16.4 / ≈12.8 held-out post-repair over all 974; ≈54 /
  ≈42 held-out unrepaired under the stated accusations-per-PR assumption).
- The exact binomial thresholds: refutation region 9/12 at N=12, 16/20 at N=20, 25/30 at N=30;
  Clopper–Pearson one-sided lower bound 0.05^(1/59) = 0.9505 versus 0.05^(1/58) = 0.9497.

**Not measured by anyone, and not resolved by this design**

- Our per-claim precision against third-party labels. This design cannot produce one without
  re-annotation, and re-annotation would return the adjudication to us.
- Extraction validity for any instrument this lab has published, prior to G-J6.
- Whether their 974 are present in our shelf at all.

---

*Every precision this lab has published was scored by a panel this lab convened. This one will
not be, and the design is therefore the strongest available to us — which is why its label
mapping, its arm, its split, its unit conversion, and its headline cell are all nailed down in
this document rather than chosen once the overlap is counted. The arithmetic says the overlap is
likely to be small. If it is too small, that is the result we publish, and we know of no reading
of our own standards under which substituting our own panel for it would be honest.*
