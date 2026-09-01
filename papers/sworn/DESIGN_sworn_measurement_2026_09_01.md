# DESIGN — measuring sworn output: do authors bind the sentences that matter?

Fathom Lab · 2026-09-01 · **A design, not a preregistration.** Nothing here is frozen. The spec's
owed item 3 is a measurement of sworn output and item 4 is a price for the gaming countermeasure;
this document states what each measurement would be, what bars it proposes and why, what
population it needs, and what blocks it today. It is written so the operator can freeze it as a
`PREREG_` by signing the bars, and it is deliberately not named `PREREG_` before that: a
preregistration whose lock hash reads *"TBD after operator signs"* is the shape
`AUDIT_the_whole_program_2026_09_01.md` §8 called the worst of both.

## The question, honestly stated

`RESULT_sworn_v01_ships_2026_09_01.md` proves the verifier keeps the author's word on the
sentences the author bound. It proves nothing about whether the author bound the sentences that
matter. The spec is explicit that a smarter verifier cannot answer that — a verifier deciding
which unsworn sentences *should* have been sworn is a claim detector, and every claim detector
this lab measured against strangers failed. So the answer has to come from readers, not from the
instrument, and the measurement has to be designed so the readers are not handed the target
either.

Three quantities, each with a proposed bar.

**Q1 — bound recall.** Of the sentences a blind panel judges LOAD-BEARING (a claim the document's
conclusion depends on), what share sits inside a sworn span? This is the number the format's
whole value rests on, and it is unmeasured.

**Q2 — trivial swearing.** Of the sworn spans, what share bind a sentence the panel judges NOT
load-bearing (a date, a version string, a hash the author just printed, a count of files)? This is
invariant 4's gaming route, measured rather than argued.

**Q3 — coverage error.** The verifier prints `coverage_estimate` from `styxx.claimdetect`, at a
documented ceiling of <sworn r="path:papers/closed-model-frontier/stage2_result.json#/arms/flagged/A_share" k="numeric">0.4211 precision</sworn> on <sworn r="path:papers/closed-model-frontier/stage2_result.json#/arms/flagged/total" k="numeric">n=38</sworn>. How far is the printed estimate from the panel's
coverage — `sworn / (sworn + panel-judged load-bearing narrative sentences)` — and in which
direction? The spec says the net direction has never been measured. This measures it.

## The blinding, which is the design

**The panel reads the canonical text, never the inline form.** `styxx.sworn` strips every tag to
produce `text` in the sidecar; seats receive that text with sentence boundaries marked and label
each sentence LOAD-BEARING / NOT / UNSURE. They never see which sentences were sworn. The join
between labels and spans happens afterwards, by byte offset, in a script. A panel that could see
the tags would be handed the author's target and would confirm it — the handed-target mechanism
(M2) one level up, applied to the measurement of the instrument built to escape it.

**Sentence boundaries are the panel's, not the splitter's.** Seats mark boundaries themselves on
the canonical text (a sentence is whatever they bracket). Q3 is then computed twice: once over
the panel's sentences and once over the splitter's, and the difference is reported as its own
number, because the splitter is the largest false-flag source in this lane
(`RESULT_struct1_beats_the_null_2026_08_31.md`, class 3) and a measurement that inherited it
silently would be measuring the splitter.

## Population — and why this session's two documents are excluded

The two sworn documents in the tree were written by the builder of the format, about the format,
in the session that built it. They are specimens chosen to pass (M6) and they are excluded from
every numerator and denominator here; they are the worked examples, not the sample.

**Prospective, in-house.** The next twelve `RESULT_`, `FINDING_` or `DECLARATION_` documents
written under this lab's standing rules in the sworn format, by whichever agent writes them, with
no instruction beyond the spec. Twelve is the smallest count at which a 0.70 bar on Q1 can be
missed by a single document's worth of unbound claims without the whole measurement resting on
one author; it is a floor, not a power calculation, and the design says so.

**External.** Eight documents written by an agent given only the spec and a real pull request
from the AIDev corpus (10.5281/zenodo.16919272, collected by someone else) with instructions to
report what the PR did, swearing whatever it chooses. The PR diff and CI log are the receipt
pool, minted into a manifest by a harness the agent does not control. This is the population the
audit named as the one that separates prose-vs-artifact from handed-vs-found, and it is where the
claim detectors died.

Twenty documents total. Every sentence in every document is labelled by three seats.

## Protocol, lifted from the lane's own panels

Three packets, three fresh seats per packet, majority label, NO-MAJORITY excluded and counted,
UNSURE excluded from numerators and denominators and counted. Decoy sentences (obvious
load-bearing claims and obvious non-claims, drawn from documents outside the sample) carried in
every packet and gated at ≥ 0.80 agreement exactly as `PREREG_obligate1_2026_08_31.md` gates
them; a packet whose seats miss the decoys is discarded and counted. The answer key is sealed
outside the repository and only its salted SHA-256 is committed before any seat runs. Seats see
one document's canonical text at a time and nothing else.

## Proposed bars — to be frozen by signature, with the reason for each

| gate | quantity | proposed bar | why this number |
|---|---|---|---|
| G-S1 | Q1 bound recall, in-house | ≥ 0.70 | below this, a SWORN-HELD document is silent about most of what it rests on, and the format's headline is misleading in the direction it exists to prevent |
| G-S1X | Q1 bound recall, external | ≥ 0.50 | an agent reporting a stranger's PR has weaker receipts to bind to; a bar equal to the in-house one would test receipt availability, not binding behaviour |
| G-S2 | Q2 trivial swearing | ≤ 0.25 | one sworn span in four may be a date or a version and the coverage number still says something; above that the numerator is padding |
| G-S3 | Q3 coverage error, absolute | ≤ 0.15 | the advisory estimate must be within the width of the ceiling it prints, or it should stop being printed beside the verdict |
| G-S4 | verifier false accusation | = 0 | a FAILED span on a sentence three seats judge TRUE against its receipt is a defect in the verifier, and one is enough to fail the cycle |

Every gate can fail and failure ships a RESULT that says so in those words. A G-S1 failure is the
finding that authors do not bind what matters, and it would be published under that title.

## Item 4 — the price of the countermeasure

A separate, smaller study, same seats: for each of ten in-house documents, a second version is
written by the same agent under the instruction *swear only what is trivially true*. Both
versions print a coverage estimate. The gate, G-G1, is that the trivial version prints the lower
coverage on ≥ 0.80 of pairs. If the countermeasure cannot separate a trivial swearer from an
honest one on its own documents, coverage should be removed from the headline rather than
labelled advisory, and the RESULT would recommend exactly that.

## What blocks it today

Seats: three fresh model seats per packet, nine per study, on a lab whose remote-API substrate is
recorded as blocked on credits (`PROGRAM_BACKLOG.md`, B23). The prospective population: twelve
documents that do not exist yet and must not be written for this purpose. The operator's
signature on the bars above.

## What this design does not license

No number in this document is a measurement. No sworn document in the tree may be quoted as
evidence for Q1, Q2 or Q3. The bars are proposals; a bar moved after the data exists is not a bar.
The eleven-instrument count that motivated the format is settled separately, by
`CENSUS_prose_claimhood_instruments_2026_09_01.md`, and this design does not depend on it.

---

*The instrument was built to escape the handed target. Its measurement has to escape it too, so
the readers see the text without the tags — and the first thing the format will be measured on is
how much of what mattered it left in narrative.*
