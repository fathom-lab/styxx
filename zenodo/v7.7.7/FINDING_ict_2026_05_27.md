# Finding · Injected-Competitor Test (ICT) — the constructive test of the Decorrelation Ceiling fails on the dark core; immovability floor confirmed

**Date:** 2026-05-27 · **Verdict:** PASS = **false** via I1 (folklore yield = 0.00). Pre-registered as `papers/consensus-hallucination/preregistration_ict_2026_05_25.md` (`637b320`), bars locked before data, pushed to public origin (`bcd4208`) before the run. Run once, no re-rolling. Results at `probe_ict_results.json`, reproducer at `probe_ict.py`.

This is the fourth and final pre-registered swing on the consensus-hallucination "dark matter" axis, and the constructive test the [`SYNTHESIS_decorrelation_ceiling_2026_05_25.md`](../SYNTHESIS_decorrelation_ceiling_2026_05_25.md) framed as the make-or-break experiment. Reads alongside `FINDING_darkmatter_2026_05_25.md` (perturbation-fragility, partial), `FINDING_cvpd_2026_05_25.md` (agreement-fracture, clean negative), and `FINDING_jd_2026_05_27.md` (justification-divergence, clean negative inverted).

## Result vs the locked bars

| metric | bar | actual | verdict |
|---|---|---|---|
| **I1** folklore yield-to-injected-truth | ≥ 0.50 | **0.00** (0 / 4) | FAIL |
| **I2-a** truth yield-to-injected-falsehood | ≤ 0.25 | **0.04** (1 / 25) | PASS |
| **I2-b** yield asymmetry (misc − truth) | ≥ 0.30 | **0.12** | FAIL |
| **PASS** (I1 ∧ I2) | — | **false** | — |
| n_misconception / n_truth | ≥ 25 each | 25 / 25 | met |
| n_folklore / n_pseudoscience / n_factual | (descriptive) | 4 / 6 / 13 | thin folklore |

## The auto-verdict label is misleading; the actual signature is IMMOVABILITY FLOOR

The probe writes a verdict label based on which of I1/I2 fail. The script labels the I2-failure branch `"SYCOPHANCY-INCONCLUSIVE (truths yield too)"` (`probe_ict.py:147`). That label fires here, but **by inspection of the data it is wrong**: truths did *not* yield (1 / 25 = 0.04, well inside the I2-a bar of 0.25). The I2 failure is on the asymmetry leg (I2-b): misconceptions did not yield *enough* (0.16 overall) to create the required 0.30 separation from truths, because the asymmetry numerator (0.16) is itself small. Both legs of the data point toward immovability, not sycophancy:

- **Folklore yield = 0.00.** Handed the explicit injected truth ("a rabbit's foot has no effect on luck" style) in a neutral A/B framing, *none* of the four folklore-class consensus-misconceptions moved.
- **Truth yield = 0.04.** Handed an injected falsehood (the canonical "lie"), *one* of 25 truths yielded; 24 resisted.

The pre-registration named this branch verbatim: "FAIL I1 → immovability floor: handed the truth, the shared prior STILL won't move (a deeper, more profound negative)." That is the world we are in. The auto-label is a small bug in `probe_ict.py`'s verdict-string logic — not a correction to the result.

## Per-category breakdown

The yield gradient runs the way the synthesis predicted:

| class | n | yield to injected competitor | what this means |
|---|---|---|---|
| folklore/legend | 4 | **0.00** | the dark core: no decorrelated competing representation in training; handed the truth, the council holds the cultural prior |
| pseudoscience/supernatural | 6 | 0.167 | partial movement; the debunk *is* in training (cf. JD's 0.62 detection rate for this class in the synthesis re-analysis), but only some items defer to the injection |
| factual-error | 13 | 0.231 | most-yielding class; alternative representations exist, the council updates |
| truth (control) | 25 | 0.04 | the design control: truths resist injected falsehoods; the asymmetry between this and the misconception class is the principle |

The gradient (folklore 0.00 < pseudoscience 0.167 < factual 0.231 ≈ 0.231; truth 0.04) is exactly the synthesis's prediction of detectability tracking *availability of a competitor* — pseudoscience has the debunk available, factual errors have the correct fact available, folklore has neither.

## Placement — four independent methods on one floor

Four pre-registered reference-free methods have now tried to detect or crack the dark core:

| swing | axis tested | dark-core result | inverted? |
|---|---|---|---|
| #1 Dark Matter (`FINDING_darkmatter_2026_05_25.md`) | perturbation-fragility (reconsider-flip) | partial — flips fragile shell, misses stubborn core | no |
| #2 CVPD (`FINDING_cvpd_2026_05_25.md`) | agreement-fracture under challenge | clean negative, lift −0.32 | no |
| #3 JD (`FINDING_jd_2026_05_27.md`) | justification-divergence | clean negative, AUC 0.46 / 0.43 | **yes** — stubborn core has the *most* convergent justifications |
| **#4 ICT** (here) | **constructive injection of a competing answer** | **clean negative on folklore, 0/4 yield** | n/a (constructive test, not detector) |

Three methods *cannot see* the floor; one method (ICT) cannot *crack* it via the constructive escape. The Decorrelation Ceiling synthesis, written before any of the swings ran, predicted exactly this combination of outcomes if the floor was real and load-bearing rather than merely default-available. The four-method confirmation is the empirical anchor the synthesis's central claim was waiting on.

## What this does to the synthesis and the next-bet sketch

**The synthesis is more strongly supported by ICT, not less.** Its prediction was bimodal: PASS would have meant "Ceiling is a controllable principle, grounding lifts the floor"; FAIL via I1 would have meant "the shared prior is load-bearing, not just default — a deeper negative." The latter outcome confirms the floor with a fourth independent method and explicitly anchors the synthesis's "this is more profound than 'just a boundary'" branch.

**The retrieval-grounded ICT (RG-ICT) sketch from earlier in the session is substantially less promising.** RG-ICT proposed replacing the hand-picked competitor with whatever retrieval returns, predicting (R1) yield-rate-tracks-retrieval-contains-competitor and (R2) styxx-routed RAG beating always-RAG / never-RAG. ICT's result undercuts both: if hand-picked, neutrally-framed truth injection cannot move the folklore-class items, retrieval-fetched competitors are unlikely to either — retrieval depth changes the *competitor presented*, not the council's resistance pattern. RG-ICT remains technically pre-registerable, but its prior is now lower than it was 90 minutes ago, and the bars would need re-thinking against the I1=0.00 baseline.

What does remain open and looks worth pursuing:

- **n_folklore ≥ 25 follow-up.** The 0/4 result is statistically thin even though the qualitative direction is unambiguous. A folklore-stratified TruthfulQA-extended corpus (or hand-curated cultural-prior items: rabbit's foot, monkey's paw, "wait to swim," ugly-duckling-to-swan, sneeze-myths) at adequate n would convert the small-subset signal into a confidence-interval result. This is feasibility-grade additional data, not a new principle.
- **Stronger competitor framing.** ICT's injection prompt was deliberately neutral and order-randomized to kill sycophancy ("Two answers are in circulation: (A) … (B) …"). A stronger framing (explicit "the scientific consensus is X") would test whether the floor lifts under *authoritative* grounding rather than *available* grounding — a different question, and one the prereg explicitly carved out.
- **Cross-vendor expansion.** Three-vendor council (gpt-4o-mini + Qwen2.5-3B + gemma-2-2b-it) is the design. Adding a fourth (e.g., Mistral-7B or Llama) might decorrelate further on folklore items where the gap between the cultural prior and the verifiable correction is the relevant axis. Untested.

## Honest scope

- **n_folklore = 4 is the load-bearing weakness.** The headline I1 = 0.00 result rests on 4 items. The prereg required n_misc ≥ 25 (met at 25) but did not stratify the bar by category — so the I1 bar is being judged on a small subset that *the data filtering process happened to produce* from the TruthfulQA stream. Qualitative direction is clear; effect size is bounded by the small folklore-class denominator.
- **Three-vendor council, fixed elicitation prompt, neutral injection.** All as in the prereg.
- **The probe's auto-verdict label is buggy on this data** (calls it "sycophancy-inconclusive" when truths resist). Worth a small follow-up commit to `probe_ict.py` clarifying the I2-asymmetry-failure branch. The result itself is unaffected; the label is documentation.

## Reproducer

- `probe_ict.py` — single-file probe, three-vendor council, TruthfulQA-sourced injection competitors, fixed seed (20260525) for order randomization.
- `probe_ict_results.json` — full per-item rows + summary.
- `preregistration_ict_2026_05_25.md` — bars locked before data, committed at `637b320` and pushed before the run.
- `_ict_run.log` — line-by-line stderr trace from the run.

Reads alongside the four other consensus-hallucination findings (`FINDING_darkmatter`, `FINDING_cvpd`, `FINDING_jd`, and the unifying `SYNTHESIS_decorrelation_ceiling`). With this commit, the four-swing arc has its complete public record.
