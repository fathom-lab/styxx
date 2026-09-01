# PREREG — the extraction ceiling for the cross-model read: what was the reader pointed at?

Fathom Lab · 2026-09-01 · Frozen before a single new extraction exists, before any candidate
ladder is scored, and before any panel is convened. Sibling of
`../closed-model-frontier/PREREG_extraction_ceiling_2026_09_01.md`, which asks the same prior
question of diffgate's path accusations. Apparatus and receipts governed by
`PREREG_b31v2_content_transport_2026_08_01.md`, `PREREG_b34v3_labelfree_read_2026_08_03.md`,
`PREREG_b35c_open_vocab_2026_08_03.md`. Committed receipts referenced here:
`b31v2_result.json`, `b34v3_result.json`, `b34v3_fresh_split_addendum.json`,
`b35c_result.json`, `b35c_null_replication.json`, `g0clear_result_llama3b.json`,
`b50_result.json`.

**The standing commitment this document is written under:** do not ship a number whose
conditioning has not been stated. Absence of evidence is never a contradiction. UNCHECKABLE is
a first-class verdict. Never "first". Never "nobody". Always "we know of no other."

---

## The question, and why it has never been asked

Every read number this arc has published answers one question: *given a vector that names one
of seventy candidates, was the right candidate picked?* Not one answers the question before it:
**was the thing the reader was handed a single nameable thing at all?**

The apparatus makes the prior question invisible by construction. `read_top1` in
`run_b31v2.py:90` and `run_b34v3.py:46` is an index-matched `argmin` over the held-out target
centroids. The candidate array is index-aligned with the query array, so the truth is present
with probability 1; there is no score, no threshold and no way to answer "none of these". Every
query vector was manufactured by the protocol from the twelve sentences in
`../introspection-gate/introspection_gate.py:25-38`, each of which contains the target word,
mean-pooled at the last token, differenced against a fixed `"object"` baseline, at the layer
locked by the G0 sweep (layer 11, k 150, `g0clear_result_llama3b.json`).

So the pointing has never once been wrong, on any trial this arc has run. That is not a
strength. It means the pointing term has never been measured, and it means the published
numbers describe a regime the apparatus itself created.

**This preregistration prices that term. It does not re-open any published verdict.**

---

## The decomposition, stated before any number exists

A deployed identification survives only if several independent things go right. Write it, and
note immediately that this is *not* offered as an exact chain rule:

    P_deploy  ~  E  ·  V  ·  A

  * **A** — the adjudication term. Given a correctly-pointed vector and a candidate set
    containing the truth, the share where the verdict is right. **This is what every published
    read number is**: gemma 0.7857 70-way (`b31v2_result.json`), label-free gemma 0.5714
    (`b34v3_result.json`; 0.5263 recomputed on the 57 genuinely-unseen concepts,
    `b34v3_fresh_split_addendum.json`). Structurally these are the same species of number as a
    benchmark that hands the instrument its span.
  * **E** — the extraction term. Of the moments at which the reader emits an identification,
    the share in which the state it was handed was in fact a single-concept-dominated state at
    the committed site. **Never measured, and by this apparatus it cannot have been**, because
    the apparatus has only ever been pointed at states it built itself.
  * **V** — the candidate-set term. What survives when the truth is no longer guaranteed to be
    in the array. **Partially measured and currently unlicensed**: `b35c_result.json` widened
    70 → 462 with everything else held favourable and returned gemma 0.2000, llama 0.3143 under
    verdict `INVALID__null_artifact`.

**We commit now to publishing no product of these terms.** `~` is not `=`. b35c already shows
that A is not invariant to the conditioning, so a multiplicative haircut would understate the
gap rather than express it, and a published `E × A` would be an overestimate by a factor that
is itself unmeasured. Three numbers, three denominators, three scopes. If a reader wants one
number, the honest answer is that this apparatus does not produce one.

A fourth quantity already exists and is **not** re-measured here: pair coverage, the share of
(reader, target) model pairs legible at all — 45 directed pairs, max pair legibility 0.2396,
per-member means 0.0370–0.0914, "a ramp, not a cliff" (`b50_result.json`,
`FINDING_b50_no_legibility_islands_2026_08_08.md`). It is cited wherever a headline read number
is cited. It is a property of minds, not of pointing, and folding it into E would double-count.

---

## The estimand, stated so it cannot be widened afterwards

    E_read = P( the activation handed to the reader was a single-concept-dominated state,
                at the committed extraction site, at the moment it was read
              | the reader emitted an identification )

Every clause is pinned now:

  * **"the committed extraction site"** — the last-token residual stream at the G0-locked layer
    of the source, differenced against the fixed `"object"` baseline, exactly as
    `run_g0clear.py:extract_multi` computes it. Not a nearby layer, not a pooled window, not a
    re-swept operating point. Changing the site voids the run rather than producing a different
    E.
  * **"single-concept-dominated"** — a *judgment* about the text, not about the vector: reading
    only the generated passage up to and including the marked token, is there exactly one
    battery concept that the passage is about at that point? Answer ONE-CONCEPT (name it) /
    NOT-ONE-CONCEPT / UNREADABLE. Deciding this from the vector would be circular, since the
    vector is the thing under test.
  * **"at the moment it was read"** — per-event, not per-concept and not per-corpus. E is a
    conditional probability over firings, in the same sense diffgate's E is.

Three questions that are **not** this estimand and may not be substituted for it after the
fact: "was this concept represented in both models" (that is inside A), "would anyone have
wanted to read this thought" (that is external validity, and no measurement here touches it),
"is this concept in the shared vocabulary" (that is V, and it is output-side).

---

## Can this re-use committed extractions? Partly. Said plainly.

**V can. E cannot.** We prefer re-use and we are not able to have it for the term that matters
most.

**V — CPU-from-cache, zero model loads.** The candidate ladder, the reject option and the
truth-absent cell all run off `_b31v2_ptsA.npz` and `_b31v2_pts_*.npz`, the same caches b35c
used. This half costs nothing but CPU and is run **first**, because it is free.

**E — a new run, and it is unavoidable.** Measuring E requires pointing the reader at states
the protocol did not construct: free-running continuations from the source model, no concept
template, no mean-pooling over twelve sentences, a token position the experimenter did not
choose to be the moment a word appears. No such run exists in this arc, and no re-analysis of
committed caches can synthesise one, because every cached vector is a template mean.

The cost, stated without an estimate we have not earned: unlike b35c, b35a and b34v3, this run
**loads models**. It needs generation from `meta-llama/Llama-3.2-3B-Instruct` and a fresh
extraction pass at the committed site, on GPU. It is the most expensive run this arc has
proposed. We do not put a runtime number here because we have not measured one, and a guess in
a frozen document is the defect this lab keeps catching in itself.

**Ordering commitment, fixed now:** V runs first and its result does **not** license skipping E.
Cancelling E after seeing V would be a post-hoc degree of freedom, and it is forbidden by this
document.

---

## The closed-set problem, preregistered rather than discovered

Top-1 over 70 candidates is identification, not recall. Deciding after seeing results how an
open-set condition should be scored is the single degree of freedom that would invalidate all
of this, so it is decided here.

**(a) The ladder is fixed now and may not be extended, trimmed or reordered.** Three conditions,
scored on the same held-out queries under the same map: **70-way** (the committed condition),
**462-way** (the full battery, trained anchors included as distractors), **truth-absent** (see
(d)). No fourth condition may be added after results; a fourth condition may be *proposed* for a
successor preregistration.

**(b) The 462-way null is rank-based, not hit-rate-based.** This is the remedy b35c's own
finding already named. The pairing-shuffled null mode-concentrates — 22 of 70 distinct
predictions, ~30% modal share, gemma scoring exactly one hit on five of five independent seeds
(`b35c_null_replication.json`) — so a floor derived from Poisson independence is not a valid
gate for it. Fixed now: five null seeds `[11, 22, 33, 44, 55]`, scored by **mean reciprocal
rank**. The read's MRR must exceed **every one** of the five null MRRs, and must be at least
**2.0×** the largest of them. The factor 2.0 is a choice, not a derivation; it is deliberately
coarse because no measured base rate exists to size it — the same sentence b35c's prereg wrote,
and it is still true.

**(c) The reject option, calibrated on train anchors only.** A ranking instrument with no
abstention has no deployment precision at all. Fixed now: τ = the **90th percentile** of
mapped-query-to-nearest-centroid distance over the **392 train anchors** under the same map, in
the same target space, computed before any held-out query is scored. A read whose nearest
distance exceeds τ is **ABSTAIN** — not a hit, not a miss, and in the denominator. The
percentile is 90, no sweep is permitted, and τ is never re-fit on held-out data. The abstention
rate is reported for every condition.

**(d) The truth-absent cell — the one that turns identification into detection.** Half the
held-out queries, selected by a seed fixed now (`seed=907`), are re-scored with the *true*
target centroid removed from the candidate array. A usable instrument must abstain there.
Define **R** = share of truth-absent queries on which the instrument abstains. If R falls below
its gate, then **no deployment precision may be quoted for this instrument in any document this
lab publishes**, and in particular the string "deployment precision = E × A" is forbidden.

---

## Gates — thresholds fixed now

The V half is machine-scorable and carries a gates block. The E half is panel-adjudicated and
cannot be scored from a JSON; its gates are stated in prose below and are equally binding.

### V — the candidate-set half

```gates
{"gates": {"G0_reconcile70": {"metric": "ladder.gemma_2b.read70", "op": "==", "value": 0.5714},
           "G1_ladder": {"metric": "ladder.conditions_scored", "op": ">=", "value": 3},
           "G2_null_rank": {"metric": "ladder.gemma_2b.mrr462_over_max_null_mrr462", "op": ">=", "value": 2.0},
           "G3_reject_option": {"metric": "truth_absent.gemma_2b.abstain_rate", "op": ">=", "value": 0.50}},
 "outcomes": [{"when": {"G0_reconcile70": false}, "verdict": "VOID__reconciliation_failed"},
              {"when": {"G0_reconcile70": true, "G1_ladder": false}, "verdict": "VOID__ladder_incomplete"},
              {"when": {"G0_reconcile70": true, "G1_ladder": true, "G2_null_rank": false}, "verdict": "VOID__null_model_unrepaired"},
              {"when": {"G0_reconcile70": true, "G1_ladder": true, "G2_null_rank": true, "G3_reject_option": false}, "verdict": "READ_HAS_NO_REJECT_OPTION__no_deployment_precision_may_be_quoted"},
              {"when": {"G0_reconcile70": true, "G1_ladder": true, "G2_null_rank": true, "G3_reject_option": true}, "verdict": "CANDIDATE_PENALTY_PRICED__reject_option_exists"}],
 "smoke_verdict": "INVALID__smoke_plumbing_only"}
```

**G0** is a reconciliation obligation, not a result: re-running the committed 70-way cell off
the same cache must re-derive `b34v3_result.json`'s gemma `read_top1` **exactly**. Any deviation
means the join is wrong and the run is VOID, not adjusted. **G3**'s bar of 0.50 is a choice — a
coin's worth of abstention is the weakest bar under which the word *detection* is not simply
false — and it is coarse for the same reason as (b): there is no base rate to size it against.

### E — the extraction half

**G-E1 — panel reliability.** A sealed decoy set of **40** windows, 20 in which exactly one
battery concept unambiguously dominates and 20 in which none does, drawn from a held-aside
generation pool under a separate prompt seed, labelled and digest-committed before any panel
seat opens. Panel accuracy **≥ 36/40**, or the panel is VOID and no E is reported. Non-negotiable.

**G-E2 — packet size, fixed by precision not by convenience.** **N = 300** extraction events,
one per continuation, token position sampled uniformly from positions ≥ 32 tokens into the
continuation, seeds fixed in the run script before the first generation. 300 is chosen so the
95% Wilson half-width at E = 0.5 is ≤ 0.06 (which requires 267); it is rounded up, and it is not
adjusted after any interim look, because there are no interim looks.

**G-E3 — the hypothesis, pre-committed in both directions.** There is no published deployment
precision for the read to divide into, so — unlike the sibling document — these bars **cannot**
be derived by inverting a known P. They are set on interpretability grounds and that is stated
rather than disguised.

| observed E_read | verdict, fixed now |
|---|---|
| **E ≤ 0.30** | **SUPPORTED.** Fewer than a third of untemplated moments satisfy the pointing precondition. Extraction is a binding constraint for the read, the published numbers are A-terms describing a regime untemplated cognition mostly does not enter, and the sibling's decomposition extends. |
| **E ≥ 0.60** | **REFUTED for the read.** Most untemplated moments do satisfy it. Extraction is *not* the binding constraint here, the analogy to diffgate's path claims fails at the term it was built on, and the deployment gap must be explained by V and A instead. Published as a failed hypothesis of ours, at the same prominence as the other cell. |
| 0.30 < E < 0.60 | **INDETERMINATE.** Published as indeterminate. No narrative is built on it. |

**G-E4 — no receipt is regenerated.** `b31v2_result.json`, `b34v3_result.json`,
`b35c_result.json` and `g0clear_result_llama3b.json` are history and are not edited, re-run or
"corrected" by this work. A receipt is history too. New findings land in new dated receipts
beside them.

---

## The doubt this document carries as a live possibility

An adversarial review of the proposed extension, run before this document was written, returned
HOLDS WITH MODIFICATION and killed half of it. Recorded here so it cannot be quietly dropped:

  1. **The originally-proposed estimator does not exist.** "E for the read is a selection ratio
     computable from committed code and receipts" is **withdrawn**. The battery is a hand-typed
     triple-quoted string (`run_g0clear.py:_BANK`), deduped to 462 (`g0clear_result_llama3b.json`,
     `n_concepts`). No concept was ever rejected by any measurement; a grep of the three run
     scripts for exclude/drop/filter/reject/discard returns nothing. The only selection ratio the
     committed code supports measures **deduplication**. The candidate pool was never written
     down, so any denominator supplied now is chosen post hoc — the exact defect this house bans.
  2. **The doubt is not resolved by this document.** It is entirely possible that E comes back
     high and the extension is REFUTED. That cell is in the table above with its own prominence
     commitment, and this document was written expecting it might fire.
  3. **The gap may be larger than E × A can express.** b35c is the lab's own lower bound on the
     candidate-set penalty, and it is unlicensed. That is why V is measured separately and why
     no product is published.

---

## What would make us not ship

If the panel splits — no majority on more than 10% of items — E is not reported as a number at
all. If the decoy gate fails, nothing is reported but the failure. If UNREADABLE exceeds 25% of
scored items, the stimulus stream is disclosed as unjudgeable and the run is **VOID for E**;
UNREADABLE items otherwise sit in the denominator and never in the numerator. If G0
reconciliation fails, the join is disclosed as broken and the caches are left alone. If the
extraction site cannot be reproduced from the new code path to within 1e-5 max absolute
deviation on the committed source centroids, the run is VOID before a single event is judged.

**A void run is published as a void run**, under its own dated finding, with the void named in
the title. This arc has published two INVALIDs in a single session
(`FINDING_b35bc_generality_invalids_2026_08_03.md`) and an INVALID whose retraction was about
our arithmetic rather than our data (`FINDING_b48_invalid_null_bar_2026_08_06.md`). A third is
the expected cost of asking, and we commit to it in advance.

We further commit to publishing the REFUTED cell of G-E3 with the same prominence as the
SUPPORTED one.

---

## Honest limits — the price list

**No model is run by this document.** It freezes thresholds. Nothing here is a result and
nothing here licenses a claim.

**The battery is a stimulus set, not a claim set.** It was enlarged from N~110 to N~480 under
`PREREG_thought_transfer_g0clear_2026_06_20.md` for one stated reason — raising PCA subspace
coverage — in service of the cross-model **write** channel. The read arc inherited it wholesale.
It has no sampling frame. Reinterpreting it as a sample of "thoughts a model might have" is a
post-hoc reframing of an object built to be a PCA cloud, and this document does not do it.

**"This concept was readable" and "someone wanted to read it" are different questions, and
nothing proposed here resolves the second.** The continuation prompts are ours. External
validity of the stimulus stream is not measured, is not gated, and may not be inferred from any
value of E.

**E is a judgment quantity and is labelled one.** It cannot be decided from bytes, which is why
it gets a panel and a decoy gate rather than a regex.

**The panel is ours.** Convened, instructed and sealed by us, on our sample. It is disinterested
in the read's verdict but not independent of the lab. A reader who believes we fool ourselves
has only our receipts to check us with.

**If the run is void, E stays UNCHECKABLE.** That is a verdict, not a gap. Absence of evidence
is never a contradiction, and a low E would not mean the read is worthless any more than a high
one would mean it is deployable.

**Two preconditions must be named wherever a headline is quoted.** b31v2's 0.7857 additionally
requires being handed 392 true cross-model concept pairs — a supervision precondition, not an
extraction one. b34v3's 0.5714 is the figure that survives without it, and its 57-genuinely-unseen
recompute is 0.5263; both are quoted together, per the erratum the b34v3 prereg forced.

---

## What this can and cannot support

**Can:** a price for the pointing term of one cross-model read, and a separately-scoped price
for the candidate-set term, each with its own denominator, each gated before the data existed.

**Cannot:** a claim about any other instrument, any other model pair, any other lab's tool. Not
a prevalence. Not a statement about minds. We measure behaviour and representation, never minds;
the boundary is the product.

**Prior art and credit.** Decomposing a pipeline metric into per-stage terms is ordinary practice
in information extraction and information retrieval, and we claim nothing about the method.
Open-set and open-vocabulary evaluation, abstention thresholds calibrated on training data, and
rank-based nulls for many-way retrieval are all standard, and we claim nothing about those
either. What we know of no other instance of is a cross-model representational readout
publishing an extraction term for its own identifications against states it did not itself
construct. That is the measurement this document exists to price, and it is the one the
apparatus, as it stands, cannot yet perform.
