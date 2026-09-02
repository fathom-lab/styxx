# DESIGN v2 — measuring sworn output: repaired so it cannot be passed by the failure it exists to detect

Fathom Lab · 2026-09-02 · **A design, not a preregistration.** Successor to
`DESIGN_sworn_measurement_2026_09_01.md`, which is not edited. Nine adversarial reviews of the
lab's next move (workflow `wf_3d0d4bca-ea3`) found eight defects in the v0.1 design; each repair
below names the review's objection beside it. The bars are proposals until the operator signs
them; a lock hash that reads "TBD" is the shape `AUDIT_the_whole_program_2026_09_01.md` §8 called
the worst of both, so this stays a DESIGN until signed and becomes a PREREG in its own commit.

## The question, unchanged

`RESULT_sworn_v02_ships_2026_09_02.md` proves the verifier keeps the author's word on the spans the
author bound. It proves nothing about whether the author bound the sentences that matter. The
answer must come from readers, blind to the tags, and the design has to make sure the readers are
not handed the target either.

## What v0.1 got wrong, and the repair

| # | the v0.1 design said | the review found | the repair |
|---|---|---|---|
| 1 | "three fresh model seats" | every panel in this lane was one model family; the standing rule forbids it | seats from **two model families**, both on this machine: Claude through the `claude -p` clean-config subscription transport (`run_b23_fable.py`), and a local open-weight instruct model on CPU (Qwen2.5-7B-Instruct at bf16, 3B fallback; Gemma and Llama instruct weights are also cached). No API credits are needed; B23's credit block applies to remote-API substrates only. A run that seats one family publishes counts labelled one-family and no precision-shaped number. |
| 2 | decoys "obvious load-bearing / obvious non-claims" | one-sided decoys passed the extraction-v1 panel that would have failed on a real ambiguity | two-sided sealed decoys **per family**, gated on each side separately (≥ 27/30 overall and ≥ 9/15 per side, the extraction-v2 gate), with hard-boundary items drawn from outside the sample; a family that fails is VOID on its own |
| 3 | Q1 = load-bearing sentences inside a sworn span / load-bearing sentences | a miss cannot distinguish *the author chose not to bind* from *the format has no kind that could bind it* | **Q1 in three cells**: load-bearing AND bindable under the four kinds (author choice); load-bearing but unbindable (the format's ceiling, priced as the v0.3 work item); not load-bearing. The frozen bar applies to the first cell only, and a miss there is titled *authors leave bindable sentences unbound*, never *agents do not bind what matters*. |
| 4 | G-S4: verifier false accusation = 0, "a FAILED span on a sentence three seats judge TRUE against its receipt" | the panel never sees a receipt, so the gate has no producer; and zero FAILED spans passes it vacuously | a **receipt-seeing Panel R** (sentence + leaf + kind: does the leaf evidence the sentence?) — which also produces the pairing-validity cell — and a **seeded-canary arm** of harness-inserted known-false spans, n ≥ 30, so FAILED precision is a count with a denominator: *k of n canaries FAILED, Wilson interval*, never "zero" |
| 5 | (absent) | the extraction RESULT found the identity P = E·A does not close; a HELD span on a correct leaf inside a false sentence is the same forgotten term | the **HELD-but-panel-FALSE cell** (pairing invalid) preregistered as a quantity of its own, with its own bar |
| 6 | external arm: eight AIDev PRs, "the PR diff and CI log are the receipt pool" | the shelf carries diffs and file lists only; no CI log exists for any PR; a diff-only pool makes Q1 a receipt-availability number | **order of arms**: in-house first (below), a **CI-bound pilot** second (the lab's own PRs and any volunteer repository running the sworn action, so "tests pass" has runner-minted bytes to bind to), the diff-only external arm **last**, restricted to fully-patched PRs, `complete` derived from the shelf's eligibility record, titled *binding behaviour under a diff-only receipt pool* with the receipt-availability ceiling reported first, drawn from HELD-OUT for the headline as the split rule requires |
| 7 | in-house arm: "the next twelve documents … no instruction beyond the spec" | the bars are committed and every lab author has read them; the documents are bar-aware | the in-house arm publishes with that disclosure in its first paragraph; the eight sworn RESULTs already in the tree that are about other arcs are the population's first eight, and the builder's own format documents are excluded (M6) |
| 8 | Q3: coverage error of the printed estimate | the estimate is withdrawn (SPEC v0.2 R8); the floor is not an estimate | Q3 becomes **the floor's error**: panel coverage minus `sentence_share`, expected positive by construction, reported as a description of how much the floor understates; no gate |

Kappa between families is computed over the splitter's sentence set with seat labels projected by
byte overlap (seat-drawn boundaries are not a shared item set); the panel-boundary variant is
reported as a second number only.

## Populations

- **In-house, prospective.** The next twelve `RESULT_`, `FINDING_` or `DECLARATION_` documents
  written under standing rules in the sworn format; the eight non-format sworn RESULTs already in
  the tree count first. Authors are bar-aware; the RESULT says so. The builder's own documents
  about the format are excluded from every numerator and denominator.
- **CI-bound pilot.** Pull requests to this repository and to any volunteer repository running the
  sworn action, where the runner mints `test_report` bytes at L2 after the turn.
- **External, diff-only, last.** AIDev HELD-OUT PRs, fully patched, stratified per agent, seed
  committed before any session; the report is titled by its receipt pool.
- **Canaries.** For every document, the harness inserts known-false sworn spans (a leaf that holds
  a different value; a quote that is not there) at a rate committed in the preregistration; the
  author never sees which.

## Quantities and proposed bars

| gate | quantity | proposed bar | what a miss is titled |
|---|---|---|---|
| G-F | seat families | two, each clearing its decoys | one-family: counts only, no precision |
| G-D | decoys | ≥ 27/30 overall and ≥ 9/15 per side, per family | void panel; the void is the result |
| G-S1 | Q1 cell 1, in-house: bindable load-bearing sentences inside a sworn span | ≥ 0.70 | *authors leave bindable sentences unbound* |
| G-S1X | Q1 cell 1, external | ≥ 0.50 | as above, with the receipt-availability ceiling beside it |
| G-U | Q1 cell 2: load-bearing but unbindable | reported, no bar | the format's ceiling, priced for v0.3 |
| G-S2 | trivial swearing: sworn spans on not-load-bearing sentences | ≤ 0.25 | *the numerator is padding* |
| G-P | pairing validity (Panel R): sworn spans whose leaf does not evidence the sentence | ≤ 0.10 | *the author named the wrong leaf* |
| G-C | canaries: known-false spans the verifier FAILED | k of n with Wilson interval; bar: lower bound ≥ 0.95 | *the verifier misses planted falsehoods* |
| G-R | Panel R: HELD spans the panel judges false | reported as its own cell | the forgotten term, printed |
| G-G1 | the trivially-swearing twin prints the lower floor | ≥ 0.80 of pairs | *the floor cannot price gaming*; the floor leaves the headline |

Every gate can fail and failure publishes under its title at the same speed as a pass.

## What blocks it

One signature on the bars above. The two seat families are on this machine; the packets, the
sealed keys, the scorer committed before any seat runs, and the canary inserter are a week of
engineering that needs no decision. The CI-bound pilot needs the sworn action merged into a
workflow, which only the operator can push. The external arm needs nothing new.

## What this design does not license

No number in it is a measurement. No sworn document in the tree may be quoted as evidence for
any quantity above. Two model families on one machine are not independent judges: correlated
error across families is the ceiling, and only an external seat — a packet anyone can seat
themselves on, with the key digest published first — moves it. The bars are proposals; a bar
moved after the data exists is not a bar.

---

*The instrument was built to escape the handed target; its measurement now cannot be passed by
an author who swears nothing false and a verifier that fails nothing, because the harness plants
the falsehoods and counts how many the verifier caught.*
