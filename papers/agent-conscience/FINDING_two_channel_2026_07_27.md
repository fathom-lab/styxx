# FINDING — the two-channel run closes negative without adjudicating its thesis: a mis-calibrated bar, a dead matcher, and the strongest belief signal ever measured

**Cycle 82. Prereg `PREREG_two_channel_2026_07_27.md` (commit `d27a289`; pre-run amendment
`9acbc3f`; print fix `0080f23` — all committed before any scored result). Verdict:
`CLOSED_NEGATIVE__two_channel_misses_instrument_floor`. Receipt: `two_channel_result.json`
(per-item records: `tc_phase_a.jsonl`). Agent Qwen2.5-7B-Instruct in 4-bit; 239 scored fresh
SQuAD short-answer items (pool v7, disjointness asserted at build); retrieval = the committed
20k-passage haystack apparatus.**

## The verdict first, then the two confessions it requires

**TG1 FAILED: selective accuracy 0.225 over the top half by COMBINED, against the 0.80 floor.
TG2 FAILED: additivity 0.008333333333333331 against 0.05. TG3 unpowered and failed.** Under the
frozen gates this is a closed negative and no bar moves.

**Confession one — the registered instrument bar was arithmetically unreachable on this
substrate, and the prereg should have caught it.** The design assumed the 7B would be accurate on
SQuAD short answers; it is not under the arc's strict matcher: first-answer accuracy
0.18410041841004185, revised 0.12133891213389121. Only 29 of 239 scored answers are correct, so
over the 120-item half-coverage set even a *perfect* ranking caps below one quarter — the 0.80
floor, imported from the multiple-choice format where the base rate is four times higher, could
not have been cleared by any signal, real or ideal. **This run therefore does not adjudicate the
two-channel thesis at the registered bar; it adjudicates the prereg's substrate assumption, which
was wrong.** Method rule extracted for the ledger: **check bar reachability against the base rate
before freezing** — the maximum selective accuracy at coverage c is bounded by accuracy/c, and a
floor above that bound makes the experiment unfalsifiable-in-favor before it starts.

**Confession two — the retrieval channel died at the matcher, not at retrieval.** The haystack
had the gold answer in the top five passages for 0.799163179916318 of items, yet `supported` fired
on only 0.12133891213389121 (and under two-fifths of its fires were correct, computed from the
receipt's `per_item` rows) — free-text claims almost never match
passage phrasing under strict word-boundary containment, even when they are right. TG2's additivity
of 0.008333333333333331 measures a nearly-dead channel, not the value of external evidence. The
conservatism was chosen deliberately ("nothing tunable to pass") and it over-shot: a matcher that
cannot fire cannot add.

## What is actually real in this run — reported, not gated, and worth the day

- **AUROC(S_frame) = 0.834072249589491 — the strongest belief-signal reading in this program's
  history**, on the hardest format measured (open-ended short answer), beating every
  multiple-choice reading at either scale. And it happens with belief *unanimity* collapsed to
  0.09623430962343096 (versus over eighty percent on multiple-choice): on free text the belief
  distribution is spread across surface forms, mutual-assertion matching still extracts a graded
  signal, and that graded signal ranks correctness better than the peaked binary one ever did.
  Cycle 81's G3 wall was tie-density at the top; open-ended answers dissolve the ties. This is an
  observation, claims nothing, and is exactly what the next prereg should register — with a bar the
  base rate can actually support.
- **The know-say gap is format-dependent and much larger in free text: cave rate
  0.5227272727272727** on initially-correct items at 7B (multiple-choice at the same scale ran
  ~0.26–0.29), with rescue at 0.041025641025641026 — being doubted destroys half the correct
  free-text answers and fixes almost nothing. The content-free challenge's first open-ended
  measurement at this scale, and the arc's vulnerability claim gets *stronger* off
  multiple-choice.
- Extraction fidelity 0.9205020920502092 — the amended claim-extraction plumbing worked; the
  failure was not there.

## Scope

Qwen2.5-7B-Instruct in 4-bit; SQuAD-v2 short answers under strict `mentions`/`asserts` matching
(a deliberately harsh accuracy criterion — some "wrong" answers are paraphrases the matcher
refuses; the 0.18410041841004185 is a lower bound on semantic accuracy and the honest number for
this apparatus); one content-free challenge turn; N=10 neutral samples; fixed 20k-passage
haystack. 1 item excluded unparseable. The multiple-choice results of cycles 77–81 are untouched
by anything here.

## What this licenses next, and what it does not

**Does not license:** any two-channel instrument claim in either direction — the thesis was not
reached; any weakening of the matcher by fiat (a fuzzier judge chosen *after* seeing it fail is
the forbidden move; a support matcher for free text needs its own validated study first); any
re-use of an MC-calibrated selective floor on a low-base-rate substrate.

**Does license (each needing its own prereg):** (a) **the open-ended verifier registered
properly** — S_frame at 7B on free text showed the program's best-ever AUROC as an observation;
register it with base-rate-feasible gates (AUROC floor plus a selective floor derived from the
measured base rate, both frozen in advance); (b) **a free-text support-matcher study** — the gap
between gold-in-top-5 (0.799163179916318) and support-fires (0.12133891213389121) is the exact
measurement target, and closing it honestly (spans, normalization families, entailment-free rules)
is a prerequisite for any retrieval channel on free text; (c) **the format-dependence of caving**
as a claim in its own right — 0.5227272727272727 free-text versus ~0.26–0.29 multiple-choice at
the same scale is a large, clean contrast sitting in committed receipts across cycles.
