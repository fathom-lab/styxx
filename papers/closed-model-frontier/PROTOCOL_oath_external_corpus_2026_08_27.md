# PROTOCOL — a frozen external corpus: the population this lab did not write

Fathom Lab · 2026-08-27 · **Frozen before collection runs.** Queries, caps, inclusion rules, what
gets persisted, the control arm and the pre-committed outcome table below do not move. A corpus
whose selection rule is chosen after seeing what it returns is a measurement of the collector's
priors.

## Why this exists

Three cycles died the same way in twenty-four hours, and the post-mortem is the same sentence each
time.

* **v0.12** froze a bar against a marker measured in this lab's own corpus and specified a clause
  that read something else. `V12_UNDERREACH`. Dead.
* **v0.13** was never frozen: its census's deciding column scored `0` for every candidate *and for
  the rule that does nothing*, because the column could only see the certified sixth of this lab's
  documents and every specimen that could have made it non-zero lives outside that frame.
* The **corpus audit** was found enumerating 365 certificates of which 178 were phantom copies
  inside an agent scratch worktree — a population defined by what a glob reached.

`SYNTHESIS_mention_and_use_2026_08_26.md` catalogues eight instances of one defect: *a marker that
co-occurs with a class is not the class, and this holds for the measurement exactly as it holds for
the instrument.* Every one of the eight was measured against text this laboratory wrote.

`RECON_v13_not_frozen_2026_08_27.md` states the consequence plainly: closing the open defect
"needs a population this lab did not write — which the external recon suggests exists and which
nobody has collected." **This protocol collects it.** It freezes no clause and licenses no fix.

## Disclosed contamination, stated first

**A pilot already happened and this protocol is therefore not blind.**
`RECON_oath_external_reach_2026_08_26.md` selected 14 repositories, certified 12, and read 507
tokens. Its headline: the verifier abstained on `0.9408` of tokens and made 13 accusations, **not
one of which was a catch.**

Two consequences, both binding:

1. **The pilot's repositories cannot count as reach.** Any pilot repository the frozen queries
   happen to re-reach is flagged `is_pilot_repo` and reported separately, and is excluded from
   every claim that this corpus is new. The rule does **not** seek them out — an earlier draft of
   this line promised they were "collected again", which the collector never implemented; it only
   ever flagged them. So `pilot_repos_redrawn` may be zero, and if it is, **no replication is
   claimed.** Seeding all fourteen as a labelled arm outside the cap is defensible and is not done
   here, because it is a larger edit to a frozen rule on the morning of collection.
2. **The pilot's queries were a monoculture and this is the pilot's largest defect.** Both
   (`filename:all_results.json`, `filename:eval_results.json`) are HuggingFace `Trainer` output
   conventions. A "sample of public repositories" that only reaches projects using one training
   harness is a sample of that harness. The finding *"not one accusation is a catch"* may be a
   property of HF Trainer's `all_results.json` schema rather than of external writing, and nothing
   in the pilot can distinguish those. Breaking the monoculture is the primary purpose here.

## The question

When the OATH verifier reads documents written by people who never heard of it, what does it do —
and is the pilot's *"not one accusation is a catch"* a fact about external prose or an artifact of
fourteen repositories drawn from one ecosystem?

## Selection, frozen

**Query families.** Executed verbatim against the GitHub code-search API. The first two replicate
the pilot; the remaining five exist to break its monoculture and were chosen for naming
conventions belonging to *different* ecosystems, before any of them was run:

| # | query | ecosystem it reaches |
|---|---|---|
| 1 | `filename:all_results.json` | HuggingFace `Trainer` (pilot) |
| 2 | `filename:eval_results.json` | HuggingFace `Trainer` (pilot) |
| 3 | `filename:metrics.json` | MLflow, DVC, generic |
| 4 | `filename:results.json` | generic / hand-rolled |
| 5 | `filename:scores.json` | generic / leaderboard |
| 6 | `filename:benchmark_results.json` | benchmarking harnesses |
| 7 | `filename:evaluation_results.json` | eval harnesses (lm-eval, OpenCompass) |

**Rule.** For each query in the order listed: take repositories in the code-search API's own
returned order, no inspection, no substitution, no skipping. A repository already taken by an
earlier query is skipped as a duplicate and attributed to the first query that reached it. Cap
**20 distinct repositories per query** and **140 in total** (= 20 × 7: the total cap must never be
smaller than the per-query cap times the number of queries, or it silently deletes trailing arms).

*Amended before the first request, after an adversarial pass.* The total was frozen at 120, and
20 × 7 = 140 > 120. Because a query's take counts only repositories new to the global seen set,
cross-query duplicates never reduce a take — they only force more paging — so queries 1–6 each
fill 20, the total cap fires, and **query 7 is never issued a single request.** Deterministic
under every overlap regime tested, including one where only 200 repositories exist across all
seven queries. The deleted arm was `filename:evaluation_results.json`, the non-HuggingFace
eval-harness family this protocol exists to reach. Raising the total is the only repair that
leaves every already-frozen quantity untouched: reordering the queries would be inspection-driven
selection, and lowering the per-query cap to 17 would silently re-freeze a smaller sample in all
six other arms. The collector now asserts the arithmetic so it cannot recur if a query is added.

**The collection run is issued with no arguments.** Any `--max-repos` override is a different
selection rule and voids this freeze; the manifest records the value actually used beside the
frozen one so a reader can check.

**Pinning.** Each repository is pinned to its default branch's HEAD commit sha at fetch time. The
sha is recorded. Every fetched file's `sha256` is recorded. A later reader can re-fetch at that
sha and verify byte-identity without trusting this record.

**Disclosure, carried into the receipt.** GitHub's default ordering is *best match*, not random.
This is a **convenience sample of a mechanically-defined population** and supports no inference
about base rates anywhere. It is reported because a mechanically-selected sample is auditable and
a hand-picked one is not.

## Inclusion, exclusion and caps, frozen

* **Document:** the repository-root file whose basename case-folds to `readme.md`, matching the
  pilot so the two are comparable. Documents over `400_000` bytes are `DOC_TOO_LARGE` and counted.
  The first draft matched four exact spellings, so `ReadMe.md` was `NO_DOC` — a string match
  standing in for "has a README", which is the mention/use shape again. Any other document —
  `README.rst`, `docs/README.md`, none at all — is still `NO_DOC`, and **for every such repository
  the collector records the readme-like basenames it did see**, so a reader can tell how much of
  `NO_PAIR` is *no claim document* and how much is merely *not this filename*.
* **Receipts:** files anywhere in the tree named `all_results.json`, `eval_results.json`,
  `test_results.json`, `train_results.json`, `metrics.json`, `results.json`, `scores.json`,
  `benchmark_results.json`, `evaluation_results.json`. Cap **12 receipts per document** (in tree
  order) and `2_000_000` bytes each. The cap is the pilot's and exists so `receipt_values` does
  not flatten a whole model zoo into a coincidence surface.
* **A repository with no document, or no receipt, is `NO_PAIR`** and is counted, not silently
  dropped. Silent dropping is how a population becomes what survived rather than what was chosen.
* **No inspection-based exclusion of any kind.** Not for language, quality, topic or size.

## What is persisted, and what deliberately is not

The corpus is frozen as the **measurement surface**, not as a copy of other people's work:

* per repository — `repo`, `sha`, default branch, fetch timestamp, status;
* per file — path, byte length, `sha256`;
* per token — line, column, the token, its value, `status`, `receipt_ref`, the obligating
  vocabulary if any, and a context excerpt capped at **200 characters**.

**Full document and receipt bodies are NOT vendored into this repository.** These are third-party
works under their own licences; wholesale redistribution is not necessary for the measurement and
is not done. Short excerpts are retained for verification and criticism. The recorded `sha256` per
file is what makes the corpus checkable: re-fetch at the pinned sha and the hashes either match or
the corpus is stale, and either answer is informative.

## The control arm, frozen

Yesterday's lesson, applied before it can be repeated: **a number with no control is a number
about the frame.**

**This laboratory's own corpus is RE-CERTIFIED live under the pinned verifier** in the same run and
reported beside the external numbers. That arm is the null: what the instrument does on text
written to keep its contract. Every external number is reported as a pair with it.

*Amended before the first request.* The first draft of this section said the control was scored by
"the same verifier, at the same `verifier_sha256`" — and the collector implemented it by summing
the `counts` stored inside committed certificates. That made the sentence false. The 186
certificates under `papers/` carry **ten distinct `verifier_sha256` values, and exactly four were
produced by the current `styxx/certify.py`.** Summing them while stamping today's verifier sha on
the manifest would have published a mixture across ten instrument versions as a matched control,
in a protocol whose thesis is that a number without a matched control is a number about the frame.
Re-certification costs about ninety seconds and makes the sentence true; the drift it exposes is
recorded rather than smoothed.

**The deciding column is the false-accusation rate.** `styxx-discriminates` is deliberately **not**
run on it, and the refusal is part of the protocol. That check compares candidate *rules* scored
on a shared frame and asks whether any beats doing nothing; here there is one rule — the shipped
verifier — and two *populations*. Passing a population where the tool expects a rule would produce
a verdict-shaped object that means nothing, which is a fair description of the defect this lane
exists to catch. The discrimination check belongs to whatever later cycle proposes candidate
clauses over this corpus. The column can still fail in both directions: external accusations may
be real catches, which would retire the pilot's headline, or false, which would confirm it at
scale, and the internal arm is pinned to neither.

**Three things about the control that no edit can fix, disclosed here rather than discovered
later.**

1. **Its accusation column is eleven events in three documents.** All eleven internal `UNGROUNDED`
   tokens are dated 2026-08-26 and sit in `SYNTHESIS_mention_and_use` (4),
   `RECON_oath_external_reach` (4) and `PREREG_oath_v12_formula_constant` (3). Two of those three
   are documents *about the external corpus*, and their accused tokens are external numbers
   **quoted inside internal prose** — so the null arm's deciding column is partly made of the
   treatment arm, quoted. It carries no interval and licenses no ratio.
2. **The control frame grows underneath the measurement.** It is whatever is committed under
   `papers/` at run time — 178 when the pilot was written, 186 now — and this protocol's own
   RESULT will enter it. `RECON_v13_not_frozen_2026_08_27.md` recorded this exact failure. The
   certificate roster, with each `document_sha256`, is written into the manifest so the arm is
   reproducible even though it is not stable.
3. **The two arms are not constituted alike, and that difference is the contract itself.** Internal
   documents cite receipts their authors chose for them; external documents are certified against
   whatever receipt-named files happen to sit in the tree. The pair therefore measures
   contract-keeping and instrument behaviour together and cannot separate them. No causal claim is
   made from it.

## Ground truth, frozen

Every external accusation is hand-adjudicated. Not sampled — **every one.**

* Each token is judged independently by **three adjudicators** who see the token, its line and
  surrounding context, and the cited receipts, and answer one question: *is this token a claim
  whose truth a reader could check against those receipts?*
* Majority of three decides. **Ties and unanimity failures resolve AGAINST the instrument** — a
  token nobody can confidently call a real claim is counted as a false accusation. This is the
  direction v0.11 used when it resolved panel ties against its own clause, and it is the direction
  that makes the instrument look worse.

**Blinding is structural, not promised.** Telling an adjudicator "you are not told what the
verifier decided" is worthless when the packet handed to them is 100% accused tokens: membership
alone leaks the verdict. So each packet is **salted with decoys** — tokens the verifier ABSTAINED
on and tokens it VERIFIED, drawn from the same corpus, shuffled, and indistinguishable in
presentation from the accused ones. An adjudicator cannot infer status from inclusion because
inclusion no longer carries it.

The decoys are not padding; they buy a second measurement the pilot never had. Judgements on
**abstained** tokens give the instrument's **miss rate** — how many real claims it declined to
check — and on an external corpus where the verifier abstains on roughly nineteen tokens in
twenty, that is the more important number. An instrument that accuses nothing because it checks
nothing is not accurate, it is inert, and only the decoy arm can tell those apart.

**Who the adjudicators are, stated plainly.** They are LLM agents of the same family as the author
of this protocol and of the verifier under test. That is a **correlated-error problem and it is
not solved here**: three judges sharing a training distribution are not three independent readers,
and a defect in how this model family reads a claim will be invisible to all three at once. Their
majority is evidence, not ground truth, and every number derived from it inherits that ceiling.
Disagreement rates within the panel are reported so a reader can see how much independence there
actually was, and the packets are retained so a human can re-adjudicate any of it.

## Pre-committed outcome table

| what the corpus shows | what it means | what it costs us |
|---|---|---|
| Accusations ≈ 0 and abstain share ≈ pilot's `0.9408` | the instrument is **inert** outside its contract, at scale and across ecosystems | OATH-as-detector stays retired; the contract framing is confirmed, and "abstains on almost everything" becomes the honest headline |
| The decoy arm shows a **high miss rate** — adjudicators call many ABSTAINED tokens real checkable claims | the abstentions are not caution, they are blindness; the instrument declines to check things it should check | the pilot's flattering `0.9408` abstain share is re-read as a defect rather than as restraint, and the obligation predicate becomes the priority repair |
| The decoy arm shows a **low miss rate** — abstained tokens really are unclaimable | abstention is well-calibrated outside the contract | the strongest result available for the contract framing: the instrument is quiet because there is nothing to check, not because it cannot see |
| Accusations > 0, **all** adjudicated false | the pilot's finding replicates and generalises past one harness | the obligation predicate is confirmed to over-fire on foreign prose; the mention/use defect is the largest open item and is now measured, not asserted |
| Accusations > 0, **some adjudicated TRUE catches** | the pilot's *"not one is a catch"* was a small-sample artifact | a published claim is **withdrawn and corrected**; the instrument transfers further than we said, and we said it too confidently |
| Pooled queries 3–7 vs pooled 1–2: `abstain_share` and `accusation_share` per arm, each printed with its token count and its **obligation-surface line count** | **DESCRIPTIVE, not gated.** This protocol freezes no bar, so no threshold is claimed and no significance asserted | an arm under 200 tokens supports no comparison at all and is reported as such |
| Those per-arm shares differ, **and the differing arms have a non-trivial obligation surface** | the pilot measured a harness, not a population | the pilot's headline is re-scoped to HF-Trainer repositories and the corpus supersedes it |
| Those per-arm shares differ **only because the new arms have almost no obligation surface** | the new queries reached repositories whose READMEs carry no checkable numbers at all | nothing is learned about the monoculture; the query set is the wrong instrument for the question and must be redesigned |
| `NO_PAIR` dominates **and** `readme_like_paths_seen` is near zero for those repositories | almost nobody publishes a claim document beside machine-readable results | that is the strongest available evidence for the contract framing, and the weakest possible position for OATH-as-a-product |

**A leg that cannot fail must not gate.** Every row above is reachable from the collection rule as
amended; none is guaranteed by it.

That sentence was false when this protocol was first frozen, and it is worth leaving the scar
visible: it asserted reachability while the cap arithmetic deterministically deleted the seventh
arm, so two rows quantified over "the five new queries" when only four could ever run. A
self-certifying claim about reachability, in a document about vacuous gates, that had not itself
been checked. It was found by an adversary reading two constants — not by its author.

## What this protocol does not license

It does not license a clause, a bar, a repair, or a version bump. It produces a corpus and a
measurement over it. Any future cycle that freezes a bar against this corpus must do so in its own
preregistration, and must run `styxx-discriminates` against a null rule before calling any column
decisive.

It also cannot show base rates: this is a convenience sample from one host's best-match ordering,
and README files are not the genre most scientific claims live in.

---

*Frozen 2026-08-27, before the first request was issued.*
