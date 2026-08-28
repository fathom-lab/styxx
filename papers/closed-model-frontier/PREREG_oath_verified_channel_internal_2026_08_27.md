# PREREG — does the verified channel fail *inside* this laboratory too?

Fathom Lab · 2026-08-27 · **Frozen and committed before the internal packets were built.** Sample
sizes, seed, blinding composition, the question, the tie direction and the outcome table below do
not move.

## The question, and why it is the one that matters

`RESULT_oath_external_corpus_2026_08_27.md` found that of external tokens carrying
`OATH-VERIFIED`, a blind panel judged only about half to be claims at all — the rest are
command-line flags, link labels, hardware specifications and numerals inside HTML tags, each
holding an affirmative oath because its value matched a receipt field. That is the tenth instance
of the defect `SYNTHESIS_mention_and_use_2026_08_26.md` catalogues, and the first ever measured in
the **verification** channel rather than the accusation channel.

The obvious defence is that those documents never signed the contract. This laboratory's documents
did. **So the question a hostile reader asks next, and the one this cycle answers, is whether the
same defect is in our own certificates.**

It is the highest-stakes question available. Every claim this programme makes about itself rests
on `OATH-HELD` certificates over its own papers. If a comparable share of *our* verified tokens
are not claims, then the flagship artifact is partly an attestation about configuration values and
table furniture, and it has been the whole time.

## What is measured

The **same panel design, the same question, the same tie direction** as the external cycle, applied
to this laboratory's own certified corpus. The external run's numbers are the comparison arm; they
are already frozen and committed and cannot move in response to this one.

## Frozen sampling rule

Drawn from the live re-certification of every certificate under `papers/`, at the pinned verifier:

* **VERIFIED tokens — the arm under test:** uniform sample without replacement, `n = 150`.
* **ABSTAIN decoys:** uniform sample, `n = 75`.
* **UNGROUNDED decoys:** *all* of them. The internal corpus carries very few, and sampling a
  handful would be pointless.
* Seed `20260827`, the same as the external run. Packets of 25, shuffled, three seats each.
* Decoys exist for **blinding**, exactly as before: a packet made only of verified tokens would
  leak its own answer through membership.

Tokens are presented identically to the external run — token, line, repo/document, context excerpt
capped at 200 characters, obligating vocabulary — and adjudicators are told nothing about which
corpus a packet came from.

## Tie direction, frozen

Majority of three. No majority resolves **against the instrument**: a verified token no majority
calls a claim counts as a non-claim. `UNSURE` never rescues a verification. This is the direction
that makes our own corpus look worse, which is the point.

## Pre-committed outcome table

Let `v_int` be the share of internal VERIFIED tokens the panel calls claims, and `v_ext = 0.4933`
the external figure already recorded.

| result | what it means | what it costs us |
|---|---|---|
| `v_int` high (≳ 0.85) and far above `v_ext` | the defect is **contract-dependent**: it appears where authors never wrote for receipts | the strongest available defence of the contract framing, and the repair can be scoped to foreign prose |
| `v_int` middling (≈ 0.6–0.85) | the defect is present here too, merely **milder** | every OATH-HELD certificate in this repository is partly attesting to non-claims; the flagship numbers need a stated error bar and the repair is urgent |
| `v_int` ≈ `v_ext` or below | the defect is **not about the contract at all** — it is the obligation predicate, everywhere | the programme's central artifact is compromised as it stands; `OATH-HELD` cannot be quoted as an attestation about claims until the verified channel is repaired, and several published documents need correcting |
| the panel splits far more here than externally | our own writing is genuinely harder to adjudicate | the external comparison is weakened and the correlated-error ceiling is doing more work than admitted |

**Every row is reachable.** `v_int` is unconstrained by the sampling rule, and the arithmetic check
that failed last time — a cap deleting an arm — does not apply: there is one arm and it is drawn
from the whole corpus.

## What this cycle does not license

No clause, no bar, no repair, no version bump. It produces one number and a comparison. Any repair
to the obligation predicate needs its own preregistration, its own adjudication, and a
`styxx-discriminates` check against a null rule before any column of its census is called
decisive.

## Disclosed in advance

* **The panel is the same model family as the author and the verifier.** Correlated error is not
  solved and near-total unanimity is its ceiling, not confidence. The external run returned
  `0.9814` unanimity; expect similar and do not read it as agreement between independent readers.
* **The internal corpus is not a sample of anything.** It is one laboratory's output, so `v_int`
  describes these documents and generalises to nothing.
* **The comparison is confounded by genre as well as by contract.** Our papers are argumentative
  prose about measurements; READMEs are installation instructions with results attached. A
  difference in `v_int` and `v_ext` cannot be attributed to contract-keeping alone, and no causal
  claim is made from the pair.

---

## Amendments, recorded before any verdict was known

**1. The rubric's opening sentence differs between the arms, and it had to.** The external run told
adjudicators they were judging *"numeric tokens extracted from README files in public GitHub
repositories"*, which was true of that corpus and is **false** of this one. Repeating it verbatim
in the name of identical presentation would have meant asserting a false provenance to the panel.
Blinding means withholding which corpus an item came from; it does not license lying about it.

So the internal run opens with a description true of both: *"numeric tokens extracted from
technical documents that ship machine-readable result files alongside them."* Everything else —
the four categories, the `UNSURE` instruction, the integrity constraints, the item fields, the
packet size, the panel size, the seed and the tie direction — is unchanged.

**2. The operative question text lives in the panel harness, not in the packets file.** Both runs
record a `question` field inside their packets JSON, and in both runs the text adjudicators
actually received is the one embedded in the workflow that drove them. The two are the same
question in near-identical words — "checkable against the cited receipts" in the file,
"checkable against the document's committed result files" in the harness — but they are not
byte-identical, and a reader comparing them should know which was operative. This applies to the
external run exactly as it applies here.

Both are real differences and are disclosed as such rather than smoothed over. If the framing
sentence moves adjudicators at all, the comparison inherits that; it is one more reason the
internal/external gap is not a clean causal contrast, alongside the genre confound already
disclosed above. Recorded here, in the frozen document, before the panel returned.

---

*Frozen 2026-08-27, before the internal ledger was built. Amended before it was scored.*
