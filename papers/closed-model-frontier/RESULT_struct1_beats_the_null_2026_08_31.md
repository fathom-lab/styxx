# RESULT — the first repair in this lane to beat its null: STRUCT-1 at 0.4211 against 0.2061

Fathom Lab · 2026-08-31 · Prereg: `PREREG_claim_detector_2026_08_30.md`, amended before
collection by `AMENDMENT_claim_detector_stage2_2026_08_31.md`. Receipts:
`stage2_result.json`, `stage2_packets.json`, `stage2_seat_outputs.json`,
`claim_detector_dev_eval.json`, `claim_detector_corpus_census.json`. Detector frozen as
shipped: `struct-1/2026-08-30`.

## What was asked

Every lexical repair this lane has tried is measured dead. The baseline priced the template
extractor blind — precision 0.3333, corpus recall near one claim in thirty — and set a single
bar for any successor: beat **0.2061**, the weighted precision of N2, the verb-stem null. Not
beat the templates. Beat the word list, using structure.

## What happened

Nine fresh blind seats, three packets, the same thirty frozen decoys, the same instruction
text, no re-runs and no seat discarded. Every seat cleared validity; unanimity 0.9578; **zero
NO-MAJORITY sentences**.

| arm | adjudicated A | n | A-share |
|---|---|---|---|
| **STRUCT-1 flagged** (census, all 38) | **16** | 38 | **0.4211** |
| matched control (STRUCT-1 said no) | **0** | 38 | **0.0** |

- **G-S2P — PASS.** STRUCT-1 = 16/38 (A-share 0.4211) vs the frozen N2 bar 0.2061; control =
  0/38 (A-share 0.0); no significance is claimed at these n.
- **G-S2LIFT — PASS.** The flagged arm's claim rate exceeds the control arm's by the entire
  distance available: 0.4211 against zero.

**Structure beat the word list it is built on by roughly two to one, and the sentences it
declined contained no claims at all in this sample.** After a whole family of lexical
candidates died, one survived — and it survived on a census, not a sample, so there was no
draw to make favourably.

## The honest half: 22 of 38 flags are not claims

Precision 0.4211 means the majority of what STRUCT-1 flags is still not a claim. Every one is
listed in the receipt. They fall into three nameable classes, and none of them is mysterious:

1. **The verb is inside a code span.** *"`add -A` cannot repeat it."*, *"`git show
   commit:path` returns the blob"*, *"Landing them is `git am ci-pending/*.patch && git
   push` from"* — the action word is quoted code being discussed, not an act being asserted.
   Conjunct 1 reads it as a verb because it is one, syntactically.
2. **Behavioural present, not commit past.** *"`corpus_audit` does not drop a document
   whose"*, *"firewall.py reads git history and exits non-zero"*, *"So `--poster` emits a
   fixed 1200px frame"* — these describe what the code DOES, not what this commit DID. The
   frozen tense rule cannot separate them because both are simple present.
3. **Splitter fragments.** *"The unshallow first landed inside test_ledger.py,"*, *"landed:
   the range-sanity clause clobbered obligation_source unconditionally"* — the sentence
   splitter shreds wrapped commit prose mid-clause, so subject and object arrive severed. The
   baseline RESULT already named this as an instrument defect; Stage 2 confirms it is the
   single largest contributor to false flags.

Two of the three have obvious structural fixes (mask code spans before reading conjunct 1;
require a past-tense or imperative head for behavioural-present sentences). **Neither is
applied here.** STRUCT-1 is frozen for this cycle by the amendment, whatever the outcome, and
a repair developed after seeing the failures it must fix is exactly the discipline this lab
has spent the week enforcing on itself.

## What it is worth

Over the pinned corpus of 2,824 sentences, diffgate's templates parsed **7** claims and never
read **2,818** sentences. STRUCT-1 flags **40** structurally checkable claims inside that
blind spot — **5.71×** what the gate read — and diffgate now carries them as
`unparsed_claims`, printed on every CLI run. Applying the measured precision, roughly
seventeen of those forty are real claims the gate was silently missing.

The observer invariant held throughout: `claimdetect` is imported lazily inside a try/except,
a test breaks it on purpose, and the pinned 54-commit attestation range re-ran byte-identical
on every verdict and count (**G-AB PASS**). Nothing about the gate's judgement moved.

## Limits, stated plainly

n=38 per arm. No significance is claimed and none may be quoted; both gates compare small
proportions and must always travel with their raw counts. The sample size is a direct
consequence of STRUCT-1's narrowness — the same narrowness that produced the result — and
the frozen Stage 2 design was **unsatisfiable** (it required 60 flags where 38 exist), a
defect found before collection, published, and amended in writing before any seat ran. Seats
share one model family with each other and with the corpus's author; unanimity 0.9578 is a
correlated-error ceiling, not independent agreement. The control arm's perfect 0/38 is
strong, but 38 sentences cannot establish that STRUCT-1's non-flags are claim-free in
general. Two known recall misses stand unpatched and pinned as tests — quoted verbatim from the
corpus: *"9 new tests."* (no action verb) and decoy #2, *"certify: collapse the ladder's third rung into
spec-or-historical"* — a clear-A all nine seats labelled A, missed for want of a concrete
object.

## This document is OATH-FAILED, and the accusation is the tenth specimen

`RESULT_struct1_beats_the_null_2026_08_31.certificate.json` reads **OATH-FAILED**, on one
token: the `9` inside the verbatim quotation *"9 new tests."* — a corpus sentence this RESULT
quotes in order to report that STRUCT-1 misses it.

The verifier obligated itself to that numeral because the line carries claim-shaped
vocabulary, then found nothing in the receipts to bind it to, because there is nothing to
bind: it is quoted text, not an assertion. That is **mention-versus-use**, the eleventh
catalogued instance in this lane, and it is firing on a paper whose subject is a detector
built to tell mention from use.

It is published FAILED rather than reworded. The number is a quotation and quotations do not
get edited to appease an instrument; the instrument gets the defect recorded against it. Both
STRUCT-1's own false positives and OATH's accusation here share one root — a reader that can
see a token's shape but not its speech act — and that is the next problem in this lane,
stated by two independent instruments on the same day.

## What is owed

1. Mask code spans before conjunct 1, and separate behavioural-present from commit-past — the
   two repairs Stage 2's own false positives specify, developed on DEV and reported on a
   fresh panel, never on this one.
2. Fix the sentence splitter, or measure what it costs. It is now the largest single source
   of false flags and it corrupts every recall figure this lane publishes.
3. Only after those: revisit diffgate's three xfail mention-vs-use fixtures. Nothing about
   accusation suppression is licensed by this result.

---

*The word lists all died. Structure lived — at 0.4211, with its 22 failures printed beside
it, and the two repairs they name left undone until the next freeze.*
