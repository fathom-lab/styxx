# PREREG — the open-set read: does the reader know when it was handed nothing?

Fathom Lab · 2026-09-01 · Frozen before any mapper is fit for this study, before any reject
statistic is scored, and before any value of the open-set quantity exists. Sibling of
`PREREG_read_extraction_ceiling_2026_09_01.md` and of
`../closed-model-frontier/PREREG_extraction_ceiling_2026_09_01.md`. Apparatus and committed
receipts: `PREREG_b31v2_content_transport_2026_08_01.md`,
`PREREG_b34v3_labelfree_read_2026_08_03.md`, `b31v2_result.json`, `b34v3_result.json`,
`_b31v2_ptsA.npz`, `_b31v2_pts_gemma_2b.npz`, `_b31v2_pts_llama_1b.npz`,
`_b31v2_pts_qwen_1p5b.npz`.

**The standing commitment this document is written under:** do not ship a number whose
conditioning has not been stated. Absence of evidence is never a contradiction. UNCHECKABLE is a
first-class verdict. Never "first". Never "nobody". Always "we know of no other."

---

## The question, and why the apparatus has made it unaskable

`read_extraction_census.json` established, from source rather than by inference, that the read is
closed by construction. Both implementations are the same three lines
(`run_b31v2.py:90-93`, `run_b34v3.py:46-49`):

```python
def read_top1(fin, ptsA, fin_ptsB, mapper):
    hits = sum(1 for i, c in enumerate(fin)
               if int(np.argmin(np.linalg.norm(fin_ptsB - mapper(ptsA[c]), axis=1))) == i)
    return hits / len(fin)
```

`fin_ptsB` is index-aligned with `fin`, so the correct target is in the candidate array on every
trial. `argmin` always returns something. There is no threshold, no score, and no way to answer
*none of these*. **E = 1 by construction on every trial these scripts have ever run**, and every
read figure this arc has published is therefore an A-term: *given that the reader was handed a
concept that is definitely in its vocabulary, did it pick the right one?*

The question underneath has never been asked here, and **we know of no other treatment of it in
the cross-model alignment-transfer literature, ours or anyone's** — closed-set top-k over a
candidate set containing the truth is the near-universal protocol. That is a reasonable protocol
and it is not being called wrong. It simply cannot answer this:

> **Handed a state whose target is not in the candidate set at all, does the reader say so?**

A reader that cannot decline has not read a concept. It has ranked a list.

---

## The design

**Nothing is collected and no model is loaded.** Everything below runs on the four committed
`.npz` banks, CPU-only: 462 concepts per model, index-aligned across
`ptsA` (3072-d), `gemma_2b` (2304-d), `llama_1b` (2048-d), `qwen_1p5b` (1536-d).

**Two trial types.** Partition the 462 concept indices under a committed seed into a candidate
set `C` and a disjoint out-of-vocabulary probe set `O`. The candidate array shown to the reader
is `C` **only**, for every trial.

  * **IN trials** — query `q` drawn from `C`. The target is present. The correct behaviour is to
    accept and to pick `q`.
  * **OOV trials** — query `q` drawn from `O`. The target is **absent from the candidate array
    entirely**. The correct behaviour is to **abstain**. There is no right answer to pick.

**The reject statistic**, fixed now: the top-1/top-2 **margin** in the mapped space,
`m = d2 - d1`, where `d1 <= d2` are the two smallest distances from the mapped query to the
candidate array. A confident read is one where the nearest candidate is decisively nearer than
the runner-up. The margin is chosen over the raw `d1` because `d1` scales with the query's norm
and would measure the query rather than the match; both are recorded, only `m` decides.

**The quantity.** `AUROC(m)` separating IN trials from OOV trials — the probability that a
randomly chosen IN trial carries a larger margin than a randomly chosen OOV trial. This is
threshold-free, so no operating point has to be chosen to report it, and it is exactly the
capability the closed-set protocol cannot express.

---

## Gates — thresholds fixed now, before any mapper is fit

**G-O1 — reconciliation. The new apparatus must be the old one plus a reject option, and this is
proven, not asserted.** With the reject option disabled and the candidate set set to the same
concepts the committed runs used, the procedure must reproduce `b34v3_result.json`
`.targets.*.read_top1` **exactly**: gemma `0.5714`, llama `0.6857` at the full-70 candidate set.
Any deviation means this study is measuring a different apparatus than the one whose numbers it
is decomposing. FAIL -> **VOID**, and the divergence is published rather than tuned away.

**G-O2 — the null must fail.** A random-orthogonal mapper, matched in shape and fit on shuffled
targets, is run through the identical pipeline. It must score `AUROC(m) <= 0.55`. If a random map
separates IN from OOV, the margin is reading something about the geometry of the banks rather
than about transported content, and the run is **VOID**. This gate can fail. It is here because
`b48` already died on a mis-specified null in this same arc and that failure is committed.

**G-O3 — the hypothesis, pre-committed in both directions.**

| observed `AUROC(m)` on held-out | verdict, fixed now |
|---|---|
| **>= 0.75** | **OPEN-SET SIGNAL.** The reader can decline. The published read figures become the accept-branch of a capability that is real in both branches, and "reads a concept" survives contact with an absent target. |
| **<= 0.55** | **CLOSED-SET ONLY.** The reader cannot tell a present target from an absent one. Every read number in this arc is then licensed *only* under the assumption that the answer is in the list, and every document of ours that describes the read as one model reading another's concept must be corrected. Published with equal prominence, and the corrections made. |
| 0.55 < AUROC < 0.75 | **INDETERMINATE.** Published as indeterminate. No narrative is built on it. |

**G-O4 — the split is not consulted twice.** `C`/`O` is drawn once under seed `20260901`,
committed to a hashed file before any mapper is fit. Any threshold or operating point, if one is
ever reported, is chosen on development concepts only. The held-out AUROC is computed once.

**G-O5 — no receipt is regenerated.** `b31v2_result.json`, `b34v3_result.json`,
`b35c_result.json` and the banks are history. Nothing here edits them. This study emits a new
dated receipt beside them.

---

## Constructibility, checked before freezing — and why that clause is here

`ADDENDUM_extraction_ceiling_gate_unsatisfiable_2026_09_01.md`, written earlier today, records a
preregistration of ours frozen with a reliability gate its own packet could not supply: the
decoys it assumed were 15 claim / 15 not-a-claim were in fact 30 claims and zero non-claims. The
panel was voided and no number was reported. **That failure is the reason this section exists.**

Before freezing this document, each gate was checked against the actual arrays for
*constructibility only*:

  * banks load and are index-aligned at 462 rows across all four models — **verified**;
  * `C` and `O` are drawn disjoint (70 / 57 in the check) — **verified**;
  * the candidate distance matrix and the top-1/top-2 margin are well-defined and finite —
    **verified**;
  * the reconciliation target exists as a committed value — **verified**.

**No accuracy, no AUROC, and no read score of any kind was computed in that check**, and none
exists at the time of freezing. What was established is that each gate *can be evaluated*, not
what it will say.

---

## Honest limits — the price list

**This does not widen the vocabulary.** `C` and `O` are both drawn from the same hand-typed
462-concept literal, built from 12 experimenter-written templates. An OOV probe here is
out-of-*candidate-set*, not out-of-distribution and not open-vocabulary. A high AUROC would show
the reader can reject a concept **of the same manufactured kind** that was withheld from its
list. It would say nothing about arbitrary natural language, and no result from this study may be
described as open-vocabulary.

**A low AUROC is not proof of no legibility.** It would show this margin statistic on these banks
cannot separate present from absent. A better statistic might. The verdict is scoped to the
statistic, which is why the statistic is named in advance and not chosen after looking.

**The closed-set penalty is a different term and is not measured here.** `b35c` observed the read
falling from 0.5714 to 0.20 (gemma) and 0.6857 to 0.3143 (llama) when the candidate set widened
70 -> 462, but it returned `INVALID__null_artifact` and those figures are **UNLICENSED and decide
nothing**. They are named here only to rule them out of this design.

**This measures E for the read; it does not measure E for anything else.** No result here
transfers to diffgate, to OATH, or to the calibrated instruments.

---

## What this can and cannot support

**Can:** a threshold-free measurement of whether a cross-model reader can decline an absent
target, on committed banks, reproducible on a laptop CPU, reconciled against a published number
before it is allowed to report anything.

**Cannot:** a claim about open-vocabulary reading, a claim about any other lab's method, a
prevalence, or a novelty claim. The protocol we are extending — closed-set retrieval with the
truth guaranteed present — is standard, sensible, and used by many. We know of no other instance
of it being decomposed with a reject option and the accept-branch number reconciled against the
closed-set original; that is the whole of the claim.
