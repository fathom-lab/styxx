# The Behavioral Knowledge Boundary — session index (2026-05-25)

A connected arc of **six pre-registered probes**, one thesis. Argument lives in
`knowledge-boundary-calibration/SYNTHESIS_behavioral_knowledge_boundary_2026_05_25.md`;
this is the map of artifacts + verdicts.

## Thesis

A model's self-report is dark exactly when it's wrong (logprob dies on hallucination;
verbal confidence unreliable). But the knowledge boundary is **bright to divergence**: a
fact is a shared attractor (convergent), a fabrication has none (divergent). Measure
divergence behaviorally — across a model's samples, across its peers, and against
controlled fakes — and you map where a model knows, where it admits ignorance, and where
it confidently invents. The recurring adversary, found four times: **form impersonating
meaning inside the instrument**.

## The six results

| # | question | verdict | key number | honest bound | finding |
|---|---|---|---|---|---|
| 1 | Does semantic entropy catch confident confabulation? | **YES** (after a self-caught artifact) | AUC 0.93–0.95 | cosine@0.70 manufactured a null; needs ≥0.8 / entailment | `tier3-confident-confabulation/FINDING_corrected_2026_05_25.md` |
| 2 | Which clustering — cosine or entailment? | meaning-judge cleanest, **not necessary** | LLM-judge 1.0 vs cos@0.9 0.97 | PASS=FALSE on "judge necessary"; tuned cosine suffices | `tier3-confident-confabulation/FINDING_clustering_2026_05_25.md` |
| 3 | Does it generalize across models? | **YES** | per-model AUC 0.88–0.92 | feasibility; gpt-4o mostly abstains instead | `tier3-confident-confabulation/FINDING_multimodel_2026_05_25.md` |
| 4 | Can we measure calibration via controlled fakes (KBC)? | construct-valid, **prompt-elastic** | abstention 0%→97% on one clause | model-discrimination fails under inviting prompt | `knowledge-boundary-calibration/FINDING_kbc_2026_05_25.md` |
| 4b | The knowledge-boundary as a *curve* | **PASS** | gpt-4o admits / gpt-4o-mini betrays | feasibility; neutral-prompt default boundary | `knowledge-boundary-calibration/FINDING_curve_2026_05_25.md` + `epistemic_curve.png` |
| 5 | Reference-free truth via inter-model agreement (Council)? | **fabrication: PASS; truth-tracking, fame rejected** | AUC 1.0; correct-cluster 8/8 | bounded by verifiable≈known confound + same-vendor | `council-reference-free-truth/FINDING_council_*` + `FINDING_fame_vs_truth_*` |
| 6 | The detector in reverse — diversity (GDI)? | **PASS** (instrument); alignment-tax **null** | open 14× closed; coherence 60/60 | G1 near-tautological; no real model ranking | `generative-diversity/FINDING_gdi_2026_05_25.md` |
| 7 | Can the detectors be **defeated** (red-team)? | robust to instruction, **blind to injection** | inconsistency 1.44→**0.00** under injection; soft attacks ~1.4 | registered A1 FAILS (parrot 0.625<0.70); security model confirmed in substance | `adversarial-robustness/FINDING_redteam_2026_05_25.md` |

## The discipline trail (the part that makes it trustworthy)

Every bar pre-registered before data; every probe run once; verdicts honored as written
(two VOIDs, two PASS=FALSE, kept). **Four self-corrections**, each by honoring a bar over
momentum: tweeted finding wrong (cosine artifact) → first correction overclaimed ("use
NLI") → focused probe (NLI itself false-positives) → threshold sweep (it was the cutoff).
The public thread was retracted; the internal record carries the erratum. This is the
asset: a map with the wrong turns left visible.

## Earned vs open

- **Earned (feasibility-grade):** confabulation is behaviorally detectable cross-model;
  the clustering step is characterized (cosine@0.9 default, judge opt-in, avoid 0.70/
  small-NLI); epistemic humility is prompt-elastic ("say if unsure" ≈ free guardrail);
  the knowledge boundary is a measurable psychometric curve (admits vs betrays);
  inter-model agreement is reference-free, truth-*tracking*, fame-rejected; GDI is a
  valid coherence-gated diversity instrument. **Security model:** both detectors are
  robust to instruction/persona but **blind to context-injection** (a planted fake
  collapses inconsistency 1.44→0.00 and makes the Council converge) — catch *spontaneous*
  confabulation, not *adversarially planted* fabrication; don't trust them on poisoned
  context.
- **Open:** **cross-vendor council** — the decisive truth test, key-blocked (operator);
  truth past the labelable edge — confound-blocked (measurement, not signal); full hashed
  multi-model/multi-seed runs; shipped `semantic_entropy` + `council` primitives.

## One line

**A model's knowledge boundary is dark to its own words but bright to divergence —
across its samples, and across its peers.** That, pre-registered and self-corrected four
times over, is the session.
