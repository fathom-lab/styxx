# PREREG — the edge screen: does the defect live between the nodes?

**Written before `styxx/edges.py` exists. No line of the instrument has been
written, and no external repository has been fetched, at the time this is frozen
and pushed.**

Prior results this is built on top of, both negative:
`RESULT_contract_sp6_2026_08_21.md` (3/5, boundary-only) and
`RESULT_flattering_external_2026_08_21.md` (0 genuine / 19,632 files, 10% recall).
Thesis: `SYNTHESIS_the_edge_2026_08_21.md`.

---

## the hypothesis

**H1 — SILENT-PASS is a property of an edge, not of a function.** An
edge-aware screen, which requires *both* a producer that can emit an unmeasured
value *and* a consumer that decides on it, will outperform both node screens
(`contract`, `flattering`) on a corpus neither was built from.

## the instrument, specified before it is built

A finding is an **edge** `F → C`, flagged only when all five hold:

1. **Producer.** `F` has a return path yielding a constant `K` that is reached on
   *absence* — an emptiness/None guard, or an `except` handler.
2. **Consumer.** A call site `C` uses `F`'s return in a **decision**: an `if`/
   `while` test, a `Compare`, an `assert`, or a boolean operator.
3. **Indistinguishability.** `K` is type-identical and range-plausible against
   `F`'s *computed* return paths, so `C` cannot separate them **even in
   principle**. If `F` returns `NaN`, `None`, or a `Measured`, it is defended and
   is never a finding.
4. **Polarity comes from the consumer, not from names.** The "quiet" branch of the
   decision is the one that does not raise, does not return an error, does not
   warn/log, and does not append to a findings list. `K` must land on the quiet
   branch. **This exists specifically to fix flattering's C4** — name morphology
   inverted on `sklearn`, and `valid`/`assert`/`check` selected for exactly the
   one-sided predicates where an optimistic empty return is *mathematically
   correct*.
5. **Contrast.** `F`'s computed paths must be able to produce a value that lands
   on the **loud** branch. A constant that always agrees with every computed value
   conflates nothing — **this exists to fix flattering's C5**, where
   `np.linalg.norm([], inf)` already returns `0.0` and the two ternary branches
   were numerically identical.

Requirement 2 is C1 (consumer liveness, which killed 6/6 last time) made
structural. Inbound arguments are ineligible by construction — C6.

## GO / NO-GO, committed now, before the instrument exists

> **If the edge screen catches fewer than 8 of the 20 cases in
> `benchmarks/silent_pass` — measured against real pre-fix source extracted at
> `<fix_commit>~1` — it is NOT run against external code at all, and this
> preregistration terminates with that number published.**

40% is deliberately at risk. `flattering` scored **10%** on this same corpus while
being fitted to it, and its external `0 of 8` was uninterpretable as a direct
result. **A low-recall screen cannot produce an interpretable external number,
and running one anyway is how this project fabricated a clean corpus last time.**
This is in-sample and is an **upper** bound on recall, never an estimate of it.

## the corpus, named before it is fetched

Six public Python repositories whose subject matter *is* evaluation, gating and
safety — the organ, not the numerics:

`EleutherAI/lm-evaluation-harness` · `explodinggradients/ragas` ·
`confident-ai/deepeval` · `NVIDIA/garak` · `truera/trulens` ·
`Giskard-AI/giskard`

Plus `inspect_ai`, already local. Default branch, pinned by commit SHA in the
RESULT. Test files excluded. No repository is added or dropped after any output
is seen.

## gates, frozen

**G0 — GO/NO-GO.** Above. Checked first; nothing else runs if it fails.

**G1 — PRIMARY, precision.** Of adjudicated external findings, **≥ 30% GENUINE**.
Higher than flattering's 20% because requirement 2 removes the failure mode that
killed all six candidates last time; a weaker bar would let a worse instrument
pass. Below 30% → **H1 NOT SUPPORTED**, published as such.

**G2 — RESOLUTION, a validity leg this project has never had.** The screen must
report the fraction of call sites whose callee it could actually resolve.
**If resolution < 25%, the verdict is `INVALID__BLIND` regardless of precision** —
a screen that cannot see the edges cannot make a claim about them, and a high
precision measured on 3% of the graph is a statement about nothing.

**G3 — POWER.** If total external findings < 15 → `INVALID__UNDERPOWERED`,
**not a null**.

**G4 — TWO-SIDED.** If BENIGN ≥ 80%, that goes in the **title**. If precision is
high *because the screen barely fires*, the firing rate goes in the title
instead. Both halves are reported with equal prominence.

**G5 — ADJUDICATION.** Three independent reviewers per finding, each prompted to
**refute**, each assigned a distinct lens, given the surrounding source and told
to resolve uncertainty **against** H1. GENUINE only when refuters fail to reach a
majority. Every verdict published with its rationale. This is the protocol that
produced 24/24 refutations last time, three of which turned on facts I had not
checked, and one of which corrected a specific **I had fabricated**.

**G6 — ANTI-TUNING.** `styxx/edges.py` is frozen at a commit recorded in the
RESULT, before any external repository is fetched. Any edit afterwards voids the
run. No rule widened to catch something seen externally; none narrowed to drop
something.

**G7 — COMPARISON.** `contract` and `flattering` are run over the same external
corpus. H1 requires the edge screen to beat **both** on genuine findings. If a
node screen matches it, the reframe is not supported and the synthesis is wrong.

## what would falsify the thesis

G7 directly: if a node screen finds as many genuine defects on a corpus neither
was built from, then node analysis was merely underpowered, "edge" is a
distinction without a difference, and `SYNTHESIS_the_edge_2026_08_21.md` is
retracted.

## stopping rule

One external run. If G1 fails, H1 is not supported **by this instrument** and the
number is published. No second scan with widened rules — that is how a program
fabricates a finding, and this project has now documented itself doing a version
of it once today already.
