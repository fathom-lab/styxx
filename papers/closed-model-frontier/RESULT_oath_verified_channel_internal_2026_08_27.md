# RESULT — the defect is in our own certificates, and the contract buys less than we said

Fathom Lab · 2026-08-27 · Prereg: `PREREG_oath_verified_channel_internal_2026_08_27.md`, frozen
and committed before the internal ledger was built and amended before it was scored. Receipts:
`oath_internal_result.json`, `oath_adjudication_result.json`, `oath_adjudication_analysis.json`.

**The pre-committed middle row fired.** Of this laboratory's own tokens carrying `OATH-VERIFIED`,
a blind panel judged `0.7933` to be claims. The prereg said in advance what that would mean:

> *the defect is present here too, merely **milder** — every OATH-HELD certificate in this
> repository is partly attesting to non-claims; the flagship numbers need a stated error bar and
> the repair is urgent.*

That is the verdict, taken as written.

---

## 1. What the two arms say

| | internal (ours) | external | gap |
|---|---|---|---|
| verified tokens the panel calls claims | `0.7933` (n=150) | `0.4933` (n=75) | `0.3` |
| abstained tokens the panel calls claims — the **miss rate** | `0.4267` (n=75) | `0.4067` (n=150) | `0.02` |
| panel unanimity | `0.9619` | `0.9814` | — |

The first row is the defence working. Documents written to keep the contract carry verifications
that are much more often about real claims, and the thirty-point gap is the largest evidence this
programme has that the contract framing is worth something.

**The second row is the finding, and it is not good.** On abstention the two corpora are
indistinguishable. The verifier declines to check a checkable claim at essentially the same rate
whether the author signed the contract or never heard of it. **Keeping the contract buys nothing
at all against blindness.**

That was not predicted anywhere. The contract framing says the instrument works where authors
emit receipts; it turns out to be true of what the instrument *says* and false of what it
*chooses to look at*. The obligation predicate is not contract-sensitive, because it never reads
the receipts — it reads the vocabulary on a line, and our lines are not more legible to it than
anyone else's.

## 2. What our own bad verifications look like

Thirty-one of a hundred and fifty. The pattern is the external one, unchanged:

| the token | its line | what it is |
|---|---|---|
| `343` | `One seed (343), one split, 70-way read` | a seed |
| `10000` | `permutation p = 0.0 (10000 draws)` | a draw count |
| `0.75` | `agreement quantized to {0.25, 2/4, 0.75, 1.0}` | notation, a set member |
| `5` | `any 5th conversion, 2nd flip, 6th silenced verification` | a gate's ordinal |
| `8` | a leading table cell in an adversarial-curve row | a row index |
| `1` | `Qwen2.5-1.5B only, ... λ=1.0 only, seeds {0,1}` | a configuration value |

Each carries an affirmative oath in a committed certificate, because its value happened to match
a receipt field. The v0.8 coincidence channel — which exists to catch exactly this — does not
reach them, because these bind *nominally*: the path shares vocabulary with the line.

## 3. What our own misses look like, which is worse

Thirty-two of seventy-five abstained tokens are, on the panel's reading, checkable claims. These
are not edge cases. They are headline measurements in flagship documents:

* `median parity_gap=0.0516` — a reported median, in a RESULT;
* `k* = 2` **under the pre-stated definition** — a preregistered quantity;
* `31/60 replicates cleared the noise-margin` — a pass count at a gate;
* `n=60 = 30 original + 30 new authored pairs` — a stated sample size.

The verifier read these lines and declined to check them. The certificate on those documents says
`OATH-HELD`.

**So what an OATH-HELD certificate actually means is narrower than we have ever written it.** It
means *the numbers this verifier chose to obligate, it could bind to a receipt.* It does not mean
the document's claims are grounded, and until today nobody had measured the distance between
those two sentences. On our own corpus it is roughly two in five.

## 4. What this costs, stated plainly

1. **Every `OATH-HELD` certificate in this repository needs the error bar this cycle measured**,
   in both directions: about one in five of its verifications may be sworn to a non-claim, and
   about two in five of the document's checkable claims were never examined.
2. **The claim that OATH is "a contract, not a detector" survives, but weaker and differently.**
   The contract helps the verification channel a great deal and the obligation channel not at all.
   Anyone adopting it is buying better attestations about what gets checked, not better coverage.
3. **The repair to the obligation predicate is now the top item in the lane**, ahead of the
   verified-channel repair, because it is the one the contract cannot compensate for.

## 5. Limits, and one that is severe

* **The internal accusation arm is n=11 and unusable.** Its false-accusation rate is `0.8182`,
  which sounds alarming and means almost nothing: all eleven live in three documents from one day,
  two of which are documents *about* accusations whose accused tokens are quoted examples. It is
  reported for completeness and no weight is put on it.
* **The panel is not three readers.** `0.9619` unanimity internally against `0.9814` externally,
  from three seats of one model family. That is the correlated-error ceiling the prereg disclosed,
  not agreement between independent readers.
* **The comparison is confounded by genre as well as by contract.** Our papers are argumentative
  prose about measurements; READMEs are installation instructions with results attached. No causal
  claim is made from the pair, and the disclosed rubric deviation between arms is one more reason.
* **The panel's independence was verified behaviourally, not by supervision.** The harness's
  safety classifier was unavailable for thirteen of thirty seats. Rather than assume, the check is
  in the output: the panel contradicts the withheld answer key in **all three arms** — calling 43%
  of abstentions claims, and most accusations non-claims. A panel with key access would agree with
  it, not contradict it. There is no evidence of key access, and this is the check rather than an
  assurance.
* **The internal corpus is not a sample of anything.** It is one laboratory's output.

## 6. This document is OATH-FAILED, on the example

The certificate beside this file says `OATH-FAILED`, with exactly two accusations, and both are
the numerals in `31/60` — the line in §3 quoting a pass count the verifier failed to check.

So: a document reporting that the verifier misses real claims is accused on its **quotation of a
missed claim**. The line carries measurement vocabulary because it is *about* a measurement, the
obligation predicate cannot tell a quoted example from an assertion, and the accusation lands on
the specimen.

It is published failing rather than reworded until it passes. Rewording would remove the clearest
demonstration in the document of the defect the document is about, and this lane has three other
certificates already `OATH-FAILED` for the same reason. The cost of explaining this defect
continues to scale with how concretely you explain it.

## What this licenses

Nothing. No clause, no bar, no version bump. It produces two numbers and a comparison, and it
moves the top of the repair queue.

---

*We built the instrument, wrote the contract, kept it, and then measured what keeping it bought.
It bought a great deal on the channel we were looking at and nothing at all on the channel we were
not.*
