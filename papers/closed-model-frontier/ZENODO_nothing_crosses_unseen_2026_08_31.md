# Nothing Crosses Unseen: Epistemic Boundaries as Software Objects

**Fathom Lab · Alexander Rodabaugh · 2026-08-31**

*A week of preregistered, blind-adjudicated verification research on the styxx instrument
stack — positives, negatives, and two papers failed by their own verifier, all published.
Every number in this document is bound to a machine-readable receipt included in the deposit,
and the document carries its own OATH certificate.*

---

## 1. The thesis

Verification correctness is not verification coverage. A verifier that reports only what it
checked — and stays silent about what it declined to examine — ships a green lamp, not an
instrument. The styxx stack makes the boundary itself a software object: every certificate
states, machine-readably, what the verifier was *obligated* to check, what it *volunteered*,
what it *refused*, and what it *never read*.

This deposit documents the week that thesis was stress-tested on the instrument's own
research corpus, with the blind-panel methodology that let the machinery overrule its
authors — twice.

## 2. The corpus, as of this deposit

The standing audit line over the lab's certified research corpus, immediately before this
document's own certificate joined it: **199 certificates ·
192 OATH-HELD · 7 OATH-FAILED**. The failures are published beside the passes; two of the
seven are papers in this very arc, failed by the verifier they describe (§7). Certifying
this deposit — the act of depositing — adds a two-hundredth certificate to the very count
it reports; the snapshot receipt in this bundle is pinned to the moment before that
recursion, and the repository's live audit carries the moment after.

## 3. The boundary, measured on the agent gate

The diffgate instrument checks what an AI agent *says* it changed against what actually
changed. Pointed at the 54-commit branch its own author wrote, it read 6 sentences of 2,738
and — on hand adjudication — misread half of what it managed to read
(`RESULT_agent_gate_boundary_2026_08_30.md`). The follow-up baseline put that hand
adjudication itself to nine blind seats: the extractor's precision adjudicated at 0.3333,
its corpus-level recall near one claim in thirty, and the never-read band's claim density at
0.0204 — real claims, sitting unexamined
(`RESULT_agent_claim_extractor_baseline_2026_08_30.md`). The panel also overturned the author
on one of three contested cases, and the retraction shipped with the prominence the original
carried.

## 4. Methodology: panels that can say no to their builder

Every measurement here uses the same frozen protocol: preregistration committed before
harness code exists; adversarial red-teaming of the prereg before freezing; packets of
shuffled items under opaque ids; known-answer decoys gating each seat at 0.80 or better, with
contested-class decoys *reported but never gated* so the author's key cannot launder the
author's reading; answer keys sealed outside the repository with only a salted SHA-256
committed before any seat runs; majority verdicts with ties excluded and counted; and the
mandated publication of every gate's outcome, pass or fail, with raw counts attached.

## 5. A positive: structure beats the word list (STRUCT-1)

Every lexical repair to the claim extractor had died measured. STRUCT-1 — four structural
conjuncts over agent prose, specified in a frozen prereg before the code existed — was put to
a fresh blind panel on a census of every flag it produced. It adjudicated at **0.4211**
against its own verb-list null's frozen bar of **0.2061**, and the sentences it declined
contained **zero** claims in the matched control sample
(`RESULT_struct1_beats_the_null_2026_08_31.md`). Applied to the pinned corpus, it finds 40
structurally checkable claims in the band the templates never read — 5.71× what the gate
parsed. Its two known recall misses are pinned as tests rather than patched, because the
conjuncts were frozen.

## 6. A negative: the obligation clause does not ship (OBLIGATE-1)

The flagship verifier's largest known hole: 0.5227 of full-precision decimals sit on lines
without trigger vocabulary and are never checked. The best structural candidate from an
earlier in-sample census — 0.80 precision there — was frozen with a ship bar of 0.70 and put
to nine fresh blind seats. It adjudicated at **0.4483**
(`RESULT_obligate1_does_not_ship_2026_08_31.md`). Per the frozen prereg, verbatim: *"the
structural obligation clause does not survive held-out adjudication."* The clause beats the
obligate-everything null and would catch roughly two in five unchecked claims — real signal,
wrong grade, because obligation manufactures accusations and precision therefore gates
first. A third of the in-sample number evaporated held-out: the cleanest demonstration this
lab owns of why in-sample results license nothing. The failure class is named — **bars**:
thresholds written to two decimals, read by shape as results — and the next candidate goes
to its own freeze.

## 7. The verifier taxes the disclosures the protocol orders

Two of this arc's papers are published **OATH-FAILED** by the instrument they describe. One
was accused on a numeral inside a verbatim quotation; the other on the denominators inside
the counts statement its own preregistration mandates word-for-word. Both accusations are
wrong in the way the arc predicts — a reader that sees a token's shape but not its speech
act — and both papers ship FAILED rather than reworded, because quotations and mandated
disclosures do not get edited to appease an instrument. The defect is catalogued against the
verifier instead. An attestation stack that cannot fail its own papers would not be worth
depositing.

## 8. What this deposit contains

The two preregistrations and one amendment (each committed before its data existed), the
red-team transcript that reshaped one of them, the five RESULT documents of the arc with
their OATH certificates (two of them FAILED), every panel's packets and verbatim seat
outputs, the sealed-then-revealed answer-key hashes, the fold harnesses, and the receipts
binding every number above. The instrument itself is `pip install styxx` (v7.46.0 at deposit
time), MIT-licensed, at `github.com/fathom-lab/styxx`.

## 9. Limits, stated once more

All seats share one model family with each other and with the corpus's author; unanimity
rates near 0.96–0.98 are correlated-error ceilings, not independent agreement. Sample sizes
are small everywhere and no significance is claimed anywhere. The corpus is one lab's prose;
nothing here is claimed to transfer beyond it. These limits are not small print — measuring
them is the research program.

---

*The word lists died. One structural candidate lived, one died held-out, the panel overruled
the author, and the verifier failed two of the papers announcing it. Every crossing is in
the ledger. Nothing crosses unseen.*
