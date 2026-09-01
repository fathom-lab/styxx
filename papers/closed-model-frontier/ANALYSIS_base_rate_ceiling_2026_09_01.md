# ANALYSIS — does a low base rate explain the 0.16?

Fathom Lab · 2026-09-01

**This is an ANALYSIS, not a RESULT. No preregistration covers it, and it therefore may not
carry a headline finding.** Nothing below is certified, nothing below is a gate outcome, and
nothing below amends `RESULT_v14_naming_the_defects_did_not_save_it_2026_09_01.md`, which is
published and stands as written. What this file contains is arithmetic run over receipts that
already exist, plus a reading of one external paper. It creates no new receipt of its own. The
one measurement produced today — `flag_rate.json` — was produced by a separate agent against a
population that reconciles with the published one, and is cited here rather than claimed here.

---

## The challenge

A reader of RESULT_v14 offered a competing explanation for our held-out precision of 0.16
(`v14_adjudication.json`: `accusations_scored` 100, `upheld` 16, `precision` 0.16, held-out
split, 30/30 decoys correct). The explanation: genuinely misleading claims are rare, so any
accuser must have poor precision, and our published explanation — that our template set is too
small to catch the classes that dominate the error — is unnecessary. The number offered as the
base rate was 1.7%, from the MSR'26 mining-challenge paper on message–code inconsistency in
agent-authored pull requests.

The question this file answers: **does that explain the 0.16 better than the explanation we
published?**

The answer is no, and the reasons are not the reasons we expected going in. Two of the three
findings below are unfavourable to us. One of them is unfavourable in a way that costs us a
defence we would have reached for.

---

## 1. The defence we would have reached for is wrong

The intuitive rebuttal is that we sampled inside the accused population — 100 accusations, not
100 pull requests — so a statement about the prevalence of defects across all pull requests
cannot bound what we measured.

That rebuttal is false, and it should be recorded as false before anything else, because it is
the kind of plausible-sounding move this lab exists to stop making.

Precision is by definition a property of the accused population. `v14_packet.py:89` draws a
uniform simple random sample from exactly that population (`sample_acc = rng.sample(acc,
N_ACC)`); the same design appears at `external1_packet.py:63`. A uniform draw from the accused
set is an unbiased estimator of P(genuinely contradicted | accused), which is precisely the
quantity a base-rate ceiling bounds. Sampling changes the variance of the estimate; it does not
change what is being estimated. Had we sampled non-uniformly — stratified by sentence shape, or
drawn from the residual the repairs were designed against — the objection would have had force.
We did not.

**So our sampling design buys us no exemption. The bound applies to our estimand, and the
challenge has to be defeated on its merits.**

---

## 2. A base rate cannot explain an observed precision. It can only bound one.

Fix a population, one unit of analysis, and one definition of the positive class. Let
pi = P(defective), r = P(flagged | defective), f = P(flagged). Bayes gives an identity, not an
inequality:

```
precision = P(defective | flagged) = pi * r / f
```

Since r <= 1:

```
ceiling = min(1, pi / f)
```

Two consequences follow, and the second one settles the shape of this whole argument.

**The bound is pi/f, not pi.** A low base rate on its own bounds nothing. It bounds only
relative to how often the instrument fires. RESULT_v14 reported no flag rate, so as published
the base-rate account was not yet a claim — it was an expression with an unevaluated term. That
omission is ours, and it is the one thing in this analysis that warrants a change to what we
publish: *"precision 0.16" without a denominator is an incomplete receipt by this lab's own
standards.*

**The regime in which the ceiling would "explain" 0.16 is the regime in which observing 0.16
was impossible.** For the ceiling to account for the level, we need pi/f <= 0.16. But pi/f is an
upper bound on precision, so pi/f <= 0.16 says the true precision is at or below the number we
measured — and the identity then yields an implied recall r = 0.16 * f / pi > 1, which cannot
happen. A ceiling below the observed point estimate is not an explanation of that estimate; it
is a refutation of the base rate that produced it.

The only strictly non-contradictory role left is the narrow band where the ceiling sits between
the Wilson lower bound and the point estimate — a ceiling in [0.101, 0.16] is compatible with
observing 16/100 as sampling noise on the high side. That is a razor-thin explanatory role, and
it is not the role the challenge was offered to play.

What remains is the honest middle case: a ceiling in (0.16, 0.95) would mean **the 0.95 floor
was unattainable by arithmetic *and* the gate still sits far below its own ceiling**. Both true
at once. Our published explanation survives for the gap; the preregistration's choice of a 0.95
floor becomes separately indefensible. That case is live, and it is decided by f.

---

## 3. f is now measured, and on the split that produced the 0.16 the ceiling does not bind

Receipt: `flag_rate.json` / `flag_rate.py`, produced 2026-09-01 by the corpus agent on the
post-repair code path (`WITHHOLD_PATH_ACCUSATION` false, V13 and V14 repairs on, `run` null) —
the configuration `v14_packet.py` scored. Population reconciles exactly with the published one:
71,016 eligible pull requests, matching `v14_gates.json` `prs_scored`; development bucket 15,617
matching `development_bucket.prs`; 393 development accusations matching
`development_bucket.after_v13_v14`; 1,344 corpus-wide accusations matching
`corpus_wide.accusations_after_v13_v14`. Zero gate errors, one round-trip skip.

Per-pull-request flag rates, and the ceiling each implies when paired with the MSR figure
(406/23,247 = 0.01746 — see section 4 for why that pairing is questionable regardless):

| split | accused PRs / scored | f | ceiling = min(1, pi/f) | regime |
|---|---|---|---|---|
| development | 339 / 15,617 | 0.02171 | **0.804** | binding, non-explanatory |
| **held-out** | **858 / 55,399** | **0.01549** | **1.000** | **not binding** |
| corpus-wide | 1,197 / 71,016 | 0.01686 | 1.000 | not binding |

*Arithmetic in this file, from `flag_rate.json` and the external count; no new receipt.*

The 0.16 was measured held-out. Held-out, **the ceiling is 1.000 — the base rate imposes no
constraint at all**, and a precision of 0.95 was arithmetically reachable. The derivation that
preceded this analysis fixed its decision thresholds in advance: f <= 0.0179 refutes the
base-rate account outright. Held-out f is 0.01549. Even on development, where the gate fires
40% more often, the ceiling is 0.804 — five times the observed precision.

Three supporting figures, same source, same caveat about the pairing:

- The recall our own numbers imply, at held-out f: r = 0.16 * 0.01549 / 0.01746 = **0.142**.
  That is under 1 and therefore internally consistent — the identity reconciles. It simply does
  not reconcile in the direction the challenge needs.
- The recall required to have hit the 0.95 floor at this flag rate: r >= 0.95 * 0.01549 /
  0.01746 = **0.843**. Demanding, but not forbidden. The distinction between "hard" and
  "impossible" is the whole of what a ceiling argument can contribute, and it lands on "hard."
- The Wilson 95% interval on 16/100 is [0.101, 0.244] (the derivation's figure; note section 7
  on clustering). For the ceiling to reach even the top of that interval, f would have to be
  >= 0.0716 — more than four times what was measured.

---

## 4. And 1.7% is not our base rate anyway

The external sweep read the paper in full and recomputed its published numbers from the authors'
own files. Source: Gong, Pinna, Bian, Zhang, *Analyzing Message-Code Inconsistency in AI Coding
Agent-Authored Pull Requests*, MSR'26 Mining Challenge Track, DOI 10.1145/3793302.3793583,
arXiv:2601.04886v2, CC BY 4.0. Replication repo `gjz78910/PR-MCI`.

**The 1.7% is a detector's flag rate, not a validated prevalence.** Verified: the denominator is
23,247 pull requests, the numerator is 406, and the flag is a threshold on a hand-built
heuristic — s = 0.3*s_scope + 0.4*s_filetype + 0.3*s_tasktype, high-MCI iff s < 0.61. The sweep
confirmed that sim < 0.61 reproduces exactly 406 flags and agrees with the shipped
`high_mci_heuristic` column on all 23,247 rows. Treating a detector's firing rate as the
prevalence of the thing it detects silently imports that detector's own precision as an unstated
multiplier. The authors' own annotators uphold 246 of the 406 strictly (0.606) and 326 loosely
(0.803), so the human-confirmed strict prevalence is at least 246/23,247 = 1.06% — a lower bound
only, since recall is well under 1.

**The heuristic never reads a diff.** Verified from the authors' `construct_validity_analysis.md`:
it reads file-type counts, churn, and description wording, and the authors state it cannot detect
semantic mismatches when file types and scope appear consistent. Their embedding baseline is
explicitly run on patch summaries rather than raw diffs. Our gate reads the diff. These are not
instruments measuring the same construct.

**The units do not match the number we published.** The 1.7% is per pull request. Our 0.16 is
per *claim* — `v14_packet.py` iterates `g.claims` and appends one item per `file_*` claim. The
per-PR flag rates in section 3 are unit-matched against the 1.7%, but the precision they bound
is a per-PR precision *we never measured*. Pairing a per-PR pi with a per-claim precision is
invalid, and the error does not cancel.

**Conditioning moves the base rate the wrong way for the challenge.** Our gate can only accuse
where prose asserts a path. The quantity that actually bounds us is
pi_claim = P(genuinely contradicted | a path claim was made) — and conditioning on "the author
wrote a checkable assertion about a file" plausibly enriches for false assertions. The ceiling is
a ratio, so the challenge needs f to rise without pi rising. Nothing supports that.

The receipt shows how much that conditioning matters. Held-out, 858 of the 6,657 pull requests
that make any path claim are accused: a conditional rate of **0.1289** (development 0.1269 —
essentially flat across splits, which is mild evidence the accuser is not overfit to the bucket
its repairs were designed on). Pair that conditional flag rate with the unconditioned 1.7% and
the identity returns an implied recall of **1.18 — greater than 1, arithmetically impossible**.
That pairing is refuted on its own arithmetic, and the refutation falls on the pairing, not on
our measurement.

---

## 5. What is genuinely unsettled, stated without softening

The per-claim flag rate — the one figure that would close this — was **not** measured. The
receipt reports accusations (951 held-out) and pull requests making a claim (6,657 held-out),
but not the total count of path claims examined. So f_claim is bounded rather than known.

Lower bound on the denominator: accused PRs contribute at least their 951 accusations, and the
5,799 claiming-but-unaccused PRs contribute at least one each, giving >= 6,750 claims and
therefore **f_claim <= 0.141**.

The impossibility threshold is f_claim >= pi/0.16 = 0.109, which sits *inside* that interval. It
would be reached if the mean number of path claims per claiming pull request were below
approximately **1.31**. The accusation-side ratio in the receipt is 951/858 = 1.108 held-out and
1,344/1,197 = 1.123 corpus-wide. If claims track accusations, the true value plausibly sits below
1.31.

We are not going to dress that up. **On the per-claim units that correspond to the number we
actually published, the ceiling is UNCHECKABLE, and the plausible value lands in the region where
the pairing is arithmetically impossible rather than the region where it exonerates the gate.**
An impossible pairing tells us one of pi, r, f is measured on the wrong population — it does not
tell us the base rate explains 0.16. But it does mean the tidy version of section 3 rests on a
per-PR pairing whose precision leg was never adjudicated. One inexpensive measurement closes
this: the claims-per-PR distribution over the held-out eligible set.

---

## 6. The strongest form of the challenge, and why it also fails its own check

The challenge is stronger if it uses a construct-matched number instead of 1.7%.

The MSR authors' taxonomy has a category for exactly our construct: *Incorrect Claims* —
descriptions making specific factual claims about implementation that do not match the actual
code changes — published as `misleading_specifics`, **13 of 432 = 3.0%** of the pull requests
their annotators judged partial or misaligned. Their headline category, Phantom Changes at
196/432 = 45.4%, is operationally something else entirely: the checkbox the annotators ticked
defines it as no files actually modified, zero files changed, and 194 of the 196 carry a patch
representation of `Scope: 0 files, +0/-0 lines` — 99.0% literally empty diffs. All verified by
the sweep against the authors' own files, and the authors' README carries the 45.4% check line
itself.

So the external evidence on our construct is 13 pull requests, not 196. As a share of their
corpus that is 13/23,247 = 0.056%. Run the identity on it: r = 0.16 * 0.01549 / 0.000559 =
**4.43**, greater than 1 and impossible. That refutes the pairing too — but it refutes it by
showing the 13 is a severe undercount from an enriched, non-exhaustive annotation with unmeasured
recall, not by vindicating anything of ours. This is the reconciliation failure the derivation
anticipated, and the failure is itself the finding: **the construct-matched base rate for our
class does not exist, in their work or in ours.**

**One variant survives and we should say so.** pi is heterogeneous across repositories, agents
and prose styles. If our gate fires disproportionately in strata where pi is far below the global
rate, effective precision can fall below any global ceiling. That is a real mechanism. But it is
an argument about *where the instrument looks* — a property of the instrument — so it sharpens
RESULT_v14's account rather than displacing it, and it is not the argument the 1.7% was offered
to make.

---

## 7. What this analysis does not establish

The base-rate account failing does not establish the template account. Those are separate
propositions and only one of them has been tested here.

RESULT_v14's published explanation is **not displaced** and **not confirmed**. Its status is
unchanged: it remains the explanation that paper offered, resting on the argument made there,
which this file does not touch.

Two further limits on what is above:

- **Clustering.** The Wilson interval [0.101, 0.244] assumes independence that the sampling frame
  does not have — sampled claims cluster within pull requests, repositories and agents, so the
  effective sample size is below 100 and the true interval is wider. This does not threaten "far
  below the 0.95 floor." It does blur the boundaries between the regimes in section 2.
- **Panel error.** The 30/30 decoy result (`v14_adjudication.json`) demonstrates competence on
  constructed easy items. It does not directly bound panel error on the hard residual the
  accusations actually are.

And an independent conditioning effect worth recording, which is RESULT_v14's own thesis reached
from the sampling frame rather than from the theory: the held-out accusation set the panel sampled
is a **post-repair residual**. Two accusation-removing repairs, designed on the development
bucket, ran before that sample was drawn. If those repairs generalised at all, the surviving accusations are enriched for
false accusations of the shapes we could not name — which is a sufficient reason for the number to
be 0.16 that owes nothing to base rates.

---

## 8. What this would change about RESULT_v14 — conditionally, and what would settle it

**No correction to the published paper is warranted on this evidence.** What is warranted is an
addendum reporting the flag rate, which the paper should have carried from the start.

If the per-claim flag rate is later measured and lands in the binding-but-non-explanatory band,
one sentence of RESULT_v14's *interpretation* would need qualifying — not its finding. The paper
says the 0.95 floor was missed. It would then also be true that the floor was unattainable at the
flag rate the gate actually ran at, which makes the preregistration's choice of 0.95 indefensible
in retrospect and makes "0.16 against a floor of 0.95" a comparison against a bar that could not
have been cleared. The finding — that removing the nameable false accusations did not move
precision — is untouched either way, because it is a statement about a *change* under repair, and
a fixed ceiling cannot explain the absence of a change.

Measurements that would settle it, in order of value per unit of cost:

1. **Claims-per-PR distribution** over the held-out eligible set — mean, variance, and the share
   of pull requests with zero path claims. Cheap. Converts section 3 into per-claim units and
   closes section 5.
2. **Per-claim flag rate** on the post-repair code path, with the denominator stated explicitly.
   Thresholds fixed in advance by the derivation: f <= 0.0179 refutes the base-rate account;
   0.0179 < f < 0.109 leaves it binding but non-explanatory; f >= 0.109 makes the pairing
   impossible and indicts one of the three quantities.
3. **pi_claim** — P(a path claim in agentic PR prose is genuinely contradicted by the diff),
   by blind panel over a random sample of *all* path claims the gate examined, flagged and
   unflagged alike. The expensive one, and the only route from UNCHECKABLE to decided. Sizing
   warning to carry into any preregistration: at a prevalence near 0.017 a 200-item panel expects
   about 3 positives, which cannot bound pi usefully. Stratify with known weights and reweight, or
   budget for n in the high hundreds. Under-powering it and reporting a point estimate would be
   worse than not running it.
4. **Recall on held-out**, from the same all-claims panel, so that pi * r / f can be checked
   against the measured 0.16. If the three do not reconcile, the reconciliation failure is the
   finding.
5. **Residual shape sensitivity** — compare the shape distribution of held-out accusations before
   and after V13+V14. A measured enrichment for unnameable shapes would support RESULT_v14's
   account directly, and is worth more than further argument about base rates.

---

## 9. Someone else's labels are public, and that is a better design than ours

**The PR-MCI human annotations are publicly downloadable in full, with no login and no request
form.** Repository `gjz78910/PR-MCI`, verified by the sweep by direct download over
`raw.githubusercontent.com`.

This matters more than the base-rate question it arrived attached to. **It would let us measure
precision against labels we did not produce, which is a stronger design than convening our own
panel**, because it removes us from the adjudication entirely.

What is there:

- `results/all_annotations_with_taxonomy.csv` — all 974 annotated pull requests with consensus
  three-way labels (aligned / partial / misaligned).
- `data/manual_annotations/*.md` — 974 individual packets, each carrying the **full unified diff**
  plus both annotators' independent labels, confidences and free-text reasons plus the consensus.
  This is the set you would re-adjudicate at claim level.
- `data/manual_annotations_validation.csv` — the 600-PR validation set with per-annotator labels,
  giving disagreement rates the consolidated file does not.
- `data/mci_scores_heuristic.csv` — per-PR score and flag for all 23,247.

The sweep established that all 406 detector flags are individually annotated with none missing
(32 fall inside the 600-PR validation sample; the 374 additional high-MCI PRs are exactly the
remaining flags). That is what makes precision measurable over a complete accusation set rather
than a sample.

Four limits, and they are load-bearing:

- **Whole-PR, three-way labels.** There is no claim-level or sentence-level annotation anywhere in
  the corpus, so it cannot score a path-claim accusation without re-annotation.
- **Author-adjudicated, not blind.** The annotators are the tool's own authors. Any precision
  figure derived from them does not satisfy this lab's standing commitment. Whether the annotators
  were blind to the detector score is **unverified** — the sampling script says scores were used
  for stratification only and the three packets the sweep read carry no score, but the 374 batch
  consists entirely of flagged PRs, which is itself a selection.
- **Licensing is unverified.** The GitHub API reports `license: null` for the repository and there
  is no LICENSE file in the tree. The paper is CC BY 4.0 and the AIDev dataset is CC-BY-4.0, but
  publicly readable is not licensed for redistribution. **Settle this before republishing any of
  their annotations in an artifact of ours.**
- **Corpus-wide `files_changed` is absent**, gitignored as too large, so an unbiased corpus base
  rate cannot be reconstructed from the repository alone (the zero-files stratum weight is not
  recoverable). Marked **unverified** by the sweep, which did not attempt regeneration from AIDev.

### The parallel, with its caveat attached

Splitting their 406 flags by whether the pull request had any diff at all (sweep's computation
from their published files, reproducible, not a number their paper reports):

| arm | n | strict precision | loose precision |
|---|---|---|---|
| flagged AND zero-diff | 230 | 0.965 | 0.965 |
| flagged AND real diff | 176 | **0.136** | 0.591 |
| all flags | 406 | 0.606 | 0.803 |

Their headline is a blend of a near-perfect trivial check — is the diff empty — with a weak prose
check. Their non-trivial arm scores 0.136 strict. Our V14 path-claim accusation scored 0.16
against a blind panel.

**This is a suggestive parallel and must not be written as a replication or a like-for-like
comparison.** Different instruments, different corpora, different adjudication procedures, and
their labels are author-adjudicated rather than blind. Their 0.136 becomes 0.591 if "partial"
counts as upheld, and there is no principled mapping from their "partial" to our panel's
upheld/not-upheld, so the convention must be stated wherever the number is used. What the parallel
does support, weakly, is that a prose-reading accuser scoring in the teens is not peculiar to our
instrument.

One citation caution the sweep flagged: the paper states an operating threshold of 0.61 but prints
precision 0.742 / recall 0.548 / F1 0.630, which are the 0.60 operating point. At the deployed
0.61 the sweep reproduces P 0.719 / R 0.548 / F1 0.622, matching the authors' own
`construct_validity_analysis.md`. Cite 0.742 as *reported*, not as the deployed operating point.

---

## 10. A finding from today's receipt that is not about base rates

`flag_rate.json` also measures the evidence leg's hole. Held-out, 3,402 pull requests make 4,038
`tests_pass` claims; corpus-wide, 4,638 pull requests — 6.5% of the corpus — make 5,514. The
accusation count is **zero**, and the verdict distribution is `UNCHECKABLE` for every single one.
With `run` null the gate refuses to take the agent's word (`diffgate.py:454-457`).

That is not a low flag rate. It is the absence of a leg, and it is the correct behaviour:
UNCHECKABLE is the honest verdict when there is no execution evidence. What the receipt adds is
the size of the hole — 5,514 claims parked there. Nothing here licenses any statement about how
well that branch *would* accuse with a `--run` supplied, because it has never accused.

---

## Answer to the question asked

**No.** On the one pairing where units match, the base rate imposes no ceiling on the split that
produced the 0.16 (held-out ceiling 1.000, against a measured f of 0.0155). On the pairing that
uses our claim-conditional flag rate, the implied recall exceeds 1 and the pairing is
arithmetically impossible. On the units that correspond to what we actually published, the
ceiling is **UNCHECKABLE** and stays that way until the claims-per-PR distribution is measured.
And the number offered as the base rate is a heuristic detector's firing rate over a different
unit, a different construct, and a corpus filtered differently from ours.

The base-rate account is not a better explanation than the one RESULT_v14 published. It is also
not a worse one in the sense of having lost a contest — it never became evaluable as stated, and
where it is evaluable it fails. **The correct verdict on the challenge is refuted under the
unit-matched reading and UNCHECKABLE-as-stated under the reading that matters, with the plausible
evaluation running against it.**

What this analysis costs us is not the explanation. It is the discovery that we published a
precision figure without the denominator that makes it interpretable, and that the defence we
would have offered — that our sampling exempts us — was wrong.

---

## Provenance

Every figure above traces to one of these. Arithmetic performed in this file is derived from
them and constitutes no new receipt.

**Measured by us, published and certified**

- `v14_adjudication.json` — held-out, 100 accusations scored, 16 upheld, precision 0.16, 30/30
  decoys, `G_S3_pass` false.
- `v14_gates.json` — 71,016 PRs scored; 4,427 to 1,344 accusations; development 15,617 PRs,
  1,299 to 393; G-S1 and G-S2 pass.
- `RESULT_v14_naming_the_defects_did_not_save_it_2026_09_01.md`, prereg
  `PREREG_v14_repair_2026_08_31.md`.
- `v14_packet.py` (sampling design, per-claim item construction), `external1_packet.py`.

**Measured by us today, uncertified, reconciles with the above**

- `papers/closed-model-frontier/flag_rate.json` and `flag_rate.py` — per-PR flag rates by split;
  path-claim and `tests_pass` counts; population 71,016 matching `v14_gates.json`.

**External, verified by the sweep against the authors' published files**

- arXiv:2601.04886v2 / DOI 10.1145/3793302.3793583 (CC BY 4.0); repo `gjz78910/PR-MCI`.
- 406/23,247 = 1.746% flag rate; high-MCI = heuristic score < 0.61, reproduced on all 23,247 rows.
- 974 annotated PRs; 432 partial-or-misaligned; 196/432 phantom changes, 194/196 zero-diff;
  13/432 misleading_specifics.
- Precision over the complete flagged set 246/406 = 0.606 strict, 326/406 = 0.803 loose; arm
  decomposition 0.965 / 0.136.
- Deployed-threshold metrics P 0.719 / R 0.548 / F1 0.622 versus reported 0.742 / 0.548 / 0.630.

**External, UNVERIFIED — marked as such wherever used**

- Which AIDev snapshot the paper used. Not named anywhere in the text. Treat as unpinned.
- Whether the PR-MCI annotations are licensed for reuse. No LICENSE file in the tree.
- Whether the two annotators were blind to the detector score.
- Whether a corpus-wide human base rate can be reconstructed; the zero-files stratum weight is
  not recoverable from the repository alone.
- Provenance of 127 templated taxonomy evidence strings in the validation batch — templated
  evidence is observable; whether the labels were human-assigned is not.

**Not measured by anyone, and required to decide the question**

- pi_claim: P(a path claim in agentic PR prose is genuinely contradicted by the diff).
- The per-claim flag rate on the post-repair code path.
- The claims-per-PR distribution over the held-out eligible set.
- Our own per-PR precision, which is the quantity the unit-matched ceiling in section 3 actually
  bounds.

## This paper's own certificate, disclosed

**OATH-FAILED**, and published under that seal rather than repaired into a pass.

The first certification refused fifty-nine numerals. Most were real defects in this paper and
were fixed the way this lab fixes them — not by removing the accusation but by producing the
receipt the number should always have had. Figures cited from outside now live in
`external_citations.json`, each tagged as printed or recomputed and carrying its provenance;
every quantity this paper derives is computed by `base_rate_ceiling.py` into
`base_rate_ceiling.json` rather than asserted in prose. That took the refusals to twenty-one.

The twenty-one that remain are **not claims**. They are section cross-references, a DOI, a
source citation of the form `file.py:89`, and the year inside a venue name. The verifier reads
them as unbound quantities because it cannot tell a numeral that asserts something from a
numeral that points somewhere.

That is worth stating plainly, because it is our own instrument exhibiting the defect we have
now found in four others: **mention versus use.** A pointer is not an assertion, and a verifier
that cannot separate them will refuse prose that is doing nothing wrong. We are not repairing it
here. The standing rule after V14 is that a class is not repaired again on the strength of a
hunch, and this one has never had its precision measured either. It is recorded as a defect with
a receipt attached — this certificate.

The verdict costs nothing that matters. Every load-bearing quantity in this paper is VERIFIED
against a receipt; what failed is the apparatus of citation around them.

---

*A tempting explanation arrived, it would have taken the failure off our instrument, and it does
not survive its own arithmetic. Publishing that is the point. The reader who raised it also
exposed a real defect: we shipped a precision without a denominator, and we know of no reading of
our own standards under which that was complete.*
