# RECON — v0.13 is not frozen, and the census that would have justified it could not fail

Fathom Lab · 2026-08-27 · **RECON. No preregistration was frozen and none should be.** Receipt:
`formula_span_census.json` (as corrected). The design and red-team pass that produced this ran
three design lenses and three adversaries before anything was written down, per `RESEARCH_LOOP.md`.

---

## What was attempted

v0.12 died because it froze a bar against a line-level marker and specified a span-level clause.
Its RESULT gave the lesson in one line: *freeze the bar against the thing you are going to build.*
So a span-level census was taken, five candidate span definitions were scored, and the table looked
like a design ready to write:

Every candidate destroyed **nothing** on the nominal column — the shape v0.11's winning clause
had. Then three lenses designed against it and three adversaries attacked the designs. **Eleven
blockers.** The cycle is not frozen. What follows is why, in the order that matters.

**No count from this census is quoted anywhere below.** They live in the receipt, and they are
volatile in a way that is itself part of the finding: this document is inside the corpus the
census measures, so publishing it moves them, and re-running after drafting one section moved
two of three candidates. What is quoted instead are the census's *verdicts*, which are stable
under corpus growth. That distinction is the discipline this cycle failed to have and the reason
it is being written up rather than shipped.

## 1. The deciding column could not fail

The census named `destroys_nominal` "the column that decides". A control that should have been in
it from the start — **no span test at all**, every bare numeral on a line carrying a backslash
command — scores *identically to the best candidate on that column*, and identically to the worst
rule available.

A column that reads the same for the best and the worst possible rule discriminates between
neither. It is a vacuous gate — the exact defect catalogued as VP-D in
`RECON_vacuous_pass_2026_08_27.md`, committed in a census written the same day, by the author of
that RECON. Found by the red team, not by me.

The reason it is vacuous is worth more than the fact. `destroys_nominal` is measured over the
certified frame, which is a sixth of the markdown under `papers/`. **Every genuine measurement any
candidate would silence lives in an uncertified document**, where no ledger status exists and the
column is structurally blind. The zero was a property of the frame's coverage, not of any rule.
The census now carries the control and the retraction.

The second column fails too, and differently. On reach, the candidates *do* differ from one
another — so the column looks alive — but not one of them beats the null rule. Candidates
separating from each other while none separates from doing nothing is a distinct failure from a
column that is flat, and only the control makes it visible.

## 2. Both designs silence genuine claims, verified by implementation

The adversaries implemented the proposed clauses rather than arguing about them, and found real
tokens each would silence that are plainly claims a receipt could confirm or contradict:

- a measured cross-substrate cosine, written `$\cos \approx 0.043$`, on which its sentence's
  conclusion rests;
- a bound written `$\mathrm{AlignDepth} \leq 0.30$`;
- six numerals sitting under a literal heading reading **Testable predictions** — the paradigm
  case of a receipt-checkable claim;
- a declared sample size that **the shipped verifier itself obligates**, because `_TRIGGERS`
  matches `AUC` on its line. The verifier already treats that token as a claim requiring a
  receipt, and the clause would have silenced it.

None of these is out of scope or hypothetical. All are in this repository, and all are invisible
to every gate either design proposed, for the coverage reason above.

## 3. The affirmative case is one formula, written here, three times, in one day

Distinct specimens in the whole affirmative case: **one**. Every in-frame accusation the clause
would reach is the same LaTeX formula, quoted in two documents written the previous day by the
person who would write the clause. The single externally-authored instance is the
GradRetentionNet README the external recon found — and it is bare, with no `$`, no backticks and
no indent, so the narrow candidates miss it entirely.

Which reverses the census's own conclusion. "Same reach for a sixth of the corpus surface" was an
equality computed over text this lab wrote, and the candidate it favours is exactly the one that
misses the only specimen a stranger produced. That is v0.12's error committed a second time in a
new place — freezing against the population you happened to measure rather than the one the clause
will meet.

Shipping would also have required regenerating the certificates of the two documents whose
accusations constitute the case. A cycle whose success criterion is flipping its own author's
documents from FAILED to HELD is not a cycle, whatever its gates say.

## 4. Consequences already recorded elsewhere

The red team also found that the proposed blind panels were unblinded by construction (only target
cases carry an OFF-arm status; most of the draw pool has no certificate at all), that a proposed
leg was precisely the widening `styxx/certify.py` names and forbids in its own v0.12 comment, and
that the frame numbers both lenses froze had already drifted — because of an edit being made to a
roster document while the design was being written. That last one is a fair hit.

## What this costs, and what it does not

**It costs a cycle and it costs the census's headline claim.** The corrected census still reports
what each candidate reaches and what it touches corpus-wide, and those numbers stand. What does
not stand is the sentence saying which column decides.

**It does not cost the defect.** A mathematical constant inside a formula still has no truth
condition, and the verifier still accuses one. Three certificates in this corpus are OATH-FAILED
on it. It remains open, and it is now clear that closing it needs a population this lab did not
write — which the external recon suggests exists and which nobody has collected.

## What was built instead of a clause

`protocol.require_nonvacuous_gates` already refuses a preregistration whose gate no outcome row
depends on. Its own comment discloses the half it cannot reach — *"an unfailable BAR needs domain
knowledge this parser does not have and is not attempted; that residual is disclosed rather than
silently implied to be covered."*

That disclosed residual is what killed this census. An outcome row did name `destroys_nominal`,
so the structural check passed; the bar was unfailable for a reason no parser could infer from
the prereg text. The technique this cycle produced needs no domain knowledge at all: **score the
null rule too, and a column the null rule ties is not a deciding column.**

`styxx/discriminates.py` is that check, with `styxx-discriminates` on the command line. Given
each candidate's scores, the control's scores, and a declared direction per column, it returns
one of three verdicts per column — `SEPARATES` when some candidate strictly beats the control,
`NULL_TIES_BEST` when candidates differ from each other but none beats doing nothing, and
`DEGENERATE` when candidates and control all share one value. A column the author declared
decisive that comes back anything but `SEPARATES` is an accusation, and in strict mode an
exception, so it can be called inside a preregistration before a bar is frozen.

Run against this census it returns `destroys_nominal DEGENERATE` and `reaches NULL_TIES_BEST` —
**both** columns, not just the one the adversary found. The receipt now carries that verdict as a
`discrimination` block, so the retraction above is computed rather than asserted, and the anchor
test in `tests/test_discriminates.py` scores the real historical numbers and requires it to fire.

It changes no certificate, so nothing here needed a preregistration. And it is necessary rather
than sufficient: a column can separate cleanly and still measure the wrong thing, which is the
defect the next section is about, and which this check would have cleared in all seven earlier
instances.

## The pattern this is the eighth instance of

`SYNTHESIS_mention_and_use_2026_08_26.md` catalogues a defect where a marker that co-occurs with a
class is used as the class. Its addendum extends the count from four instruments to seven, adding
the measurements built to audit the instruments. This is the eighth, and the sharpest, because the
marker was not a proxy for a class of tokens — it was a proxy for a **cost**. `destroys_nominal`
stood in for "would this rule destroy something real", and it could not answer, and it was quoted
as the deciding number anyway.

The countermeasure is the one that keeps working and keeps not being applied in advance: **run the
control that would make the number fail.** The vacuous-pass census carried a positive and a
negative control and caught its own error before publication. This one carried none, published a
column that could not fail, and needed an adversary to notice.

---

*The cycle produced no clause, retracted a column, and cost less than shipping would have.*
