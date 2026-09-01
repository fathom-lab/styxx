# RESULT — the external corpus: a failed sanity gate, and the defect it exposed

Fathom Lab · 2026-08-27 · Protocol: `PROTOCOL_oath_external_corpus_2026_08_27.md`, frozen and
committed before the first API request. Receipts: `oath_external_corpus.json`,
`oath_adjudication_result.json`, `oath_adjudication_analysis.json`,
`oath_external_no_receipt_diagnostic.json`.

**The pre-committed sanity gate failed.** That is the first thing in this document because it is
the first thing the protocol said would matter, and because what it exposed is worth more than the
measurement it invalidated.

---

## 1. The gate, and honouring it

The protocol pre-committed a sanity condition on the arm of tokens the verifier had **verified**:
if the panel does not agree those are claims, *"the panel and the instrument disagree about what a
claim IS, and every other number here is suspect."*

Of the verified tokens put to the blind panel, the share it agreed were claims came back at
`0.4933`. Barely half. **The gate failed.**

So the two headline numbers this cycle was built to produce — the false-accusation rate and the
miss rate — are reported below and are **not promoted to findings**. Reinterpreting a frozen row
after seeing it is the move this protocol's opening paragraph forbids, and the fact that the
numbers are interesting is not a reason to keep them.

## 2. What made it fail, which is the actual result

The obvious excuse was available and does not work. This lane has a known **coincidence channel**:
a token can "verify" against a receipt leaf that merely happens to hold its value — an index, a
seed, a step counter. If the panel were rejecting those, the gate failure would be a known defect
wearing a new hat.

It is not. Split by whether the binding was coincident or nominal, the coincidence channel accounts
for five items. The rest are **nominally bound** — the token matched a receipt leaf whose path its
own line names — and the panel still says they are not claims. Read them and the panel is plainly
right:

| the token | its line | what it actually is |
|---|---|---|
| `20` | `--episodes 20 \` | a command-line flag |
| `6` | `\| [Blog 6](blog_post_phase5.md) \|` | a link label |
| `2.5` | `<img src="static/readme_examples/gallery/...">` | a numeral inside an HTML tag |
| `10` | `Apple M4 (10 logical cores, 16 GB RAM)` | a hardware spec |
| `256` | `\| Experts \| 256 with K=8 \|` | a configuration value |
| `0` | `All scores are in **[0, 1]**` | range notation |

Every one carries `OATH-VERIFIED`. The verifier did not merely fail to check these — **it swore an
oath to them.**

That is the finding. `SYNTHESIS_mention_and_use_2026_08_26.md` catalogues a defect in which a
marker that co-occurs with a class is treated as the class, and every instance it records is about
**accusations** — the instrument firing where it should not. Nobody had looked at the other
channel. The same defect is in the verifications, at roughly half of the nominally-bound external
tokens sampled, and **a false verification is worse than a false accusation**, because the
affirmative attestation is the entire product. An accusation says *check this*. A verification says
*I have checked this, and it is sworn to a receipt.* The second is the one an outside reader would
rely on.

The pilot could not have seen it. It measured only what the verifier accused.

## 3. The pilot's headline does not survive

`RECON_oath_external_reach_2026_08_26.md` reported thirteen accusations across twelve repositories
and concluded that **not one was a catch** — a false-accusation rate of `1.0`. That framing became
"OATH is a contract, not a detector", and it is quoted in the contract document and the PR.

Against 366 accusations drawn from seven query families, the false-accusation rate is `0.2596`.
About three quarters of what the verifier accused are, on the panel's reading, genuine claims.

**This withdrawal does not depend on the failed gate, and the direction is why.** The panel's
disagreement with the instrument runs one way: it rejects tokens the verifier *verified*. A panel
biased toward calling things non-claims would, applied to accusations, produce *more*
`NOT_A_CLAIM` verdicts, not fewer. So `0.2596` is an **upper bound** on the false-accusation rate,
the genuine-claim share of `0.7404` is a **lower bound**, and a stricter panel only makes the
pilot's `1.0` less tenable. The claim is withdrawn.

Why the pilot got it so wrong is now visible, and it is not sample size. Both its queries were
HuggingFace `Trainer` conventions, and it took only fourteen repositories across the two, with
the first filling before the second contributed. Per arm:

| query | tokens | abstain | accusation share |
|---|---|---|---|
| `all_results.json` *(pilot)* | 511 | 0.908 | 0.0274 |
| `eval_results.json` *(pilot)* | 2319 | 0.843 | 0.1143 |
| `metrics.json` | 287 | 0.9861 | 0.0139 |
| `benchmark_results.json` | 1206 | 0.7073 | 0.0166 |
| `evaluation_results.json` | 707 | 0.7992 | 0.0750 |

The first row replicates the pilot almost exactly — 511 tokens against its 507, abstain `0.908`
against its `0.9408`. **The pilot's headline was one filename**, not one harness and not a
population. Two arms, `results.json` and `scores.json`, came back under the 200-token bar the
protocol set in advance and support no comparison; they are reported and not read.

All fourteen pilot repositories were re-reached by the frozen queries, so the first row is a
genuine replication rather than a claim of one.

## 4. The rate is a property of a few documents, not of external prose

The pooled accusation share is `0.0707`. It should not be quoted without this beside it:

* one repository supplies `194` of the 366 accusations, and the top three supply `0.7322` of them;
* the **median** repository's accusation share is `0.0`;
* of the repositories with tokens, `43` draw **zero** accusations.

The dominant document is a model card carrying full retrieval benchmark tables with bootstrap
confidence intervals. Its accusations are the *least* false in the corpus — a false-accusation
rate of `0.1856` inside it against `0.343` outside it. The instrument fires where claims are
dense, which is what it should do; the pooled rate is then a statement about how many claim-dense
documents happened to be drawn.

## 5. Two rows of the outcome table are unreadable, and are withdrawn

**Row 6** proposed that if `NO_PAIR` dominates, that is evidence almost nobody publishes a claim
document beside machine-readable results. The frozen run returned `NO_RECEIPT` for 50 of 140
repositories, which looked exactly like that. It is not: of the repositories probed afterwards,
`26` demonstrably carry a results-like JSON file, missed on affix (`overall_results.json`,
`graph_gcn_results.json`) or on case (`All_results.json`). GitHub's `filename:` qualifier
tokenises; `RECEIPT_NAMES` is an exact, case-sensitive set. The selection rule and the inclusion
rule disagree about what a results file *is*.

That is `RECEIPT_NAMES` standing in for the class *machine-readable results file* — the same
defect, inside the collector built to measure that defect's reach. The list is **not** widened and
nothing is re-collected, because widening it after seeing which repositories fell out is selection
after seeing returns. The frozen numbers stand; the inference is withdrawn.

**The abstention rows** are equally affected by the miss rate. The measured share of abstained
tokens the panel called checkable claims is `0.4067` — abstention on external text is
substantially blindness rather than restraint. But it is downstream of the failed gate, so it is
recorded and not concluded from.

## 6. The panel is not three readers

Three seats of one model family agreed unanimously on `0.9814` of items, with `11` split panels
and `16` `UNSURE` votes out of `1773`. The protocol named this in advance: correlated error is not
solved, and near-total unanimity is the **ceiling** of that correlation, not evidence of
correctness. Every number here inherits it. The packets and the answer key are committed so a
human can re-adjudicate any of it, and until someone does, this is a machine's opinion about a
machine.

## 7. What is owed

1. **A repair for the verified channel outranks everything else in this lane.** It is the first
   measured instance of the mention/use defect in the affirmative attestation, and OATH's product
   is the attestation.
2. **Re-run the sanity gate against a human panel** on the retained packets, at whatever sample a
   person will actually do. Until then the gate's failure is diagnosed but not independently
   confirmed.
3. **A claim-density-stratified reading**, since the pooled rate is dominated by a few documents
   and the median repository contributes nothing.

## What this licenses

Nothing. No clause, no bar, no version bump. A protocol was frozen, a corpus was collected, a gate
failed, and the failure was more interesting than the measurement. The corpus is committed with
per-file `sha256` so anyone can re-fetch at the pinned commits and check it.

---

*The cycle set out to find whether the instrument transfers. It found that where the instrument
speaks most confidently, about half of what it says on foreign text is sworn to a command-line
flag.*
