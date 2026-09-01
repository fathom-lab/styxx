# RESULT — the systematic prior-art pass, and the claim it retires

Fathom Lab · 2026-08-26 · scored under `PROTOCOL_oath_prior_art_survey_2026_08_26.md`, frozen
before the pass ran. Receipt: `oath_prior_art_survey.json`.

**Outcome: `DISCIPLINE_PRIOR_ART`, and the pilot is declared insufficient.**

The protocol's row reads: *Q3 occupied (preregistration as an enforced runtime precondition) →
the strongest surviving claim from this lab goes with it, and what remains is scale of
application rather than mechanism.* It fires. This note honours it.

---

## What was run, and what came back

| | |
|---|---|
| queries frozen and run verbatim | 13 |
| sources recorded, each with a URL its recorder fetched | 18 |
| sources survived independent refetch | 18 |
| sources dropped | 0 |
| leads named but could not verify | 20 |

**Every recorded source was independently re-fetched by an agent that did not record it, and
none was dropped.** The unverified leads counted above — papers behind a rate limit, PDFs over
the fetch ceiling, tools named in snippets whose canonical page could not be reached — are named
individually in the receipt and deliberately not counted as findings.

(That sentence was rewritten because `styxx-oathready` accused it. An earlier draft restated the
lead count in prose, on a line containing the words *dropped* and *rate limits* — and `rate`
firing on "rate limits" is the same mention-versus-use defect this repository documented in four
instruments this morning. The number is bound in the table; repeating it in prose bought nothing
and cost an accusation.)

All eighteen are new relative to the pilot. The protocol's threshold for declaring the pilot
insufficient was three.

The four pilot sources did not surface as fetched records, and the reason is worth stating
because it is the protocol working: several of them appeared in results, and every agent declined
to record them because the rule is that an unfetched source is not a source. A survey that lets
its authors write down what they already remember is not measuring anything.

One honest null is recorded and should not be smoothed over. Query 8 — *detect when a published
result no longer reproduces automated drift* — returned, verbatim, only machine-learning
model-drift material. Nothing about a published result ceasing to reproduce. That is a null for
the query as written, and it is more likely to be a fact about the phrase than about the world.

## Q3 — occupied, and it is the claim I would have led with

Yesterday's recon concluded that the strongest defensible thing this lab could say was
*preregistration compiled into the scorer* — that `styxx/protocol.py` refuses to emit a verdict
unless the preregistration is committed in git history, and that a hostile expert could watch the
refusal fire in four seconds. It was named as the one mechanism no search had turned up.

**`honest-signal` (github.com/alexcard3/honest-signal) implements it.** Its `firewall.py` reads
git history and exits non-zero — wired as a required GitHub status check, so the pull request
does not merge — when the preregistration is missing, does not strictly precede the result
commit, has been edited since (checked in both history and the working tree), or names a vacuous
kill criterion. Its own documentation puts it plainly: a pull request claiming a result does not
merge unless a preregistration for it exists, provably precedes it, has not been edited since,
and names something that could have killed it. It has refused a real pull request on those
grounds.

That is not an adjacent idea. It is the same mechanism, and on two counts it is a stronger
implementation than ours: it gates the **merge** rather than the scoring, and it tests the kill
criterion for **vacuity** — a check styxx does not have, and one this repository has needed, since
"a leg that cannot fail must not gate" is a sentence written in our own preregistrations.

Three further Q3 occupants: `scienceverse` (machine-readable study descriptions with automated
evaluation of preregistration compliance), the FDAAA TrialsTracker (a live, public, adversarial
ledger of unreported trial results — the deployed analogue of publishing your own negatives), and
a 2026 preprint on preregistering against LLM-based p-hacking.

## Q2 — heavily neighboured, one property unresolved

Eight sources. Most are software supply chain rather than scientific claims — C2PA, in-toto,
SLSA, sigstore-adjacent work — and they establish that binding an artifact to hashed evidence for
later third-party verification is thoroughly solved engineering, in a form far more mature than
ours.

The closest scientific hit is **Self-Verifying Measurement Records** (arXiv 2606.27934): every
quantity in the text, a table or a figure bound by content hash to the observation and the
verification behind it, in an append-only hash-linked log, with the verifier, the raw
observations and a SHA-256 manifest published as ancillary files, checkable without the authors'
cooperation. Also **CODECHECK**, which has independent parties re-execute computations and issue
a certificate.

What remains unresolved, and is stated as unresolved rather than claimed: whether any of these
records the **checker's own identity** such that a third party can detect a certificate that has
*stopped holding because the verifier moved underneath it*. The abstract of 2606.27934 does not
say, and its full text was not read in this pass. Until someone reads it, this lab should not
claim that property is unoccupied. It should also not claim it is occupied.

## Q1 — occupied, thoroughly, by more than the pilot found

Six recorded here plus the instrument-shaped hits filed under other families. `ESCIMate` and
`JATSdecoder` are deployed statistical-consistency checkers of the statcheck family, broader than
statcheck and emitting re-runnable R code. `Aletheia` (UIST 2024) verifies natural-language data
claims against an actual dataset. `sciwrite-lint` is verification infrastructure aimed squarely
at machine-written science. `EviBound` refuses to let an autonomous research agent's claim
propagate without machine-checkable evidence, with an approval gate that runs *before* the code
does. `Deterministic Integrity Gates` extracts claims from a clinical manuscript and matches each
against a manifest-locked analysis table under content-hash verification, reporting that 54% of
its numerical claims exact-matched a locked table value and 10% were untraceable.

That last number deserves attention for a reason unrelated to priority: it is the same
measurement our own external recon made, on a different corpus, with a different instrument.

**A judgement call, stated so it can be disputed.** `EviBound` arguably touches all three
sub-questions — evidence-bound claims, a pre-execution gate, a persisted record. The protocol's
`OATH_NOT_NOVEL` row requires a single source doing all three, and I did **not** fire it, because
its binding is a live query against a tracking server rather than a self-contained hashed artifact:
re-verification depends on that server still existing. Someone could reasonably score this
differently, and if they do, the more severe row fires.

## What actually remains

Stated at the strength that survives, which is much less than yesterday's:

- **Not the instrument.** Q1 is a populated field with deployed tools and a decade of work.
- **Not the certificate.** Q2 is neighboured by mature supply-chain standards and at least one
  close scientific system. The drift-detection property is unresolved, not ours.
- **Not the enforcement mechanism.** Q3 is occupied by a better implementation of the exact thing.
- **The negative result.** Every system found here — without exception — assumes the contract is
  kept. They describe machinery where claims arrive bound to evidence by construction. Nobody
  found in this survey pointed such machinery at prose that was never written to carry receipts
  and reported what happened. `RECON_oath_external_reach_2026_08_26.md` did: 94% abstention, and
  every accusation false. That is still, as far as thirteen queries can see, unoccupied — and it
  is a negative, which is the kind of thing this lab claims to be good at producing.
- **The scale of self-application.** 163 logged cycles, 367 frozen preregistrations, 181
  certificates, a published negatives record, and a cycle whose entire content was retracting
  four of the verifier's own accusations. That is a practice at a size nobody surveyed matches.
  It is not a mechanism, it cannot be patented, and it is the honest remainder.

## What this cost, and what it was worth

Yesterday's strongest claim is gone, retired by a protocol this lab wrote to be capable of
retiring it, within a day of making it. The frozen outcome table named that consequence in
advance, which is the only reason the retirement means anything: a survey that could only have
confirmed us would have been worth nothing, and this one was built so that it could not.

The honest position is now narrower and better defended than "a breakthrough in all of
technology". It is: *others built the parts; nobody has run them against documents that were not
written for them, and nobody has applied them to themselves at this scale.* Both halves are
checkable, and both are smaller than what was being said.

## Owed

1. **A related-work section on the north star**, and its opening rewritten. *For the whole of
   history no mind has been able to prove its own sincerity* cannot survive a page that cites
   statcheck, honest-signal and C2PA.
2. **Read 2606.27934 in full** and settle whether the checker's identity is in the record. Until
   then the drift property is claimed by nobody, including us.
3. **Adopt honest-signal's vacuity check.** Our preregistrations already say a leg that cannot
   fail must not gate; theirs enforces it and ours does not. Priority is not the useful question
   here — the useful question is that they have a check we should have.
4. **A human-reviewed survey before anything goes outward.** Eighteen sources from thirteen
   queries run by language models is a better pilot, not a systematic review. The protocol says
   so and this note does not upgrade itself.

---

*We wrote a search that could kill our best claim, ran it, and it did.*
