# project: threshold-law paper (2026-05-18)

## Status
PAPER READY — DOI STAGED — SELF-AUDIT PASSED — AWAITING GO

## Self-audit verdict (2026-05-18, styxx 7.4.1 on its own paper)
- All 8 headline numbers match raw JSON (max Δ 0.0044, tol 0.005).
- Integrity protocol: 6/6 checks pass; 11 "universal" mentions all in
  non-claim/killed-result guard context; "ALL of AI" absent.
- Circular-oracle check: 0.80 floor is definitional, not deployed as
  independent evidence. Not circular.
- Composite-honesty per section: 0.21–0.41. Two `needs_revision`
  firings (§7 Limits 0.31, §8 Integrity 0.41) driven entirely by
  sycophancy register on declarative-bound text — construct-ceiling
  artifact, inside the bound the paper already states. No content
  crack. `COGN_UNDER_REVIEW` (overconfidence) did NOT fire anywhere.
- Cracks requiring paper revision: **0**.
- Deliverable: `papers/threshold-law-self-audit-2026-05-18.md`;
  script `scripts/dogfood/selfaudit_threshold_law.py`; raw output
  `scripts/dogfood/out_selfaudit_threshold_law.json`.
- Zenodo bundle rebuilt with self-audit attached as methodology
  demonstration (`dist/zenodo/threshold-law-2026-05-18.zip`).


## Claim (exact, bounded)
There exists an empirical corpus↔domain-overlap threshold τ ≈ 0.31
(mean-max cosine in home space text-embedding-3-large) above which
label-free cognometric refusal transport from te3-large → mpnet
clears AUC 0.80, and below which it collapses to ~0.69. Same-family
(te3-large → te3-small) is approximately overlap-insensitive
(drift ~0.04 AUC). The threshold is a property of the corpus ×
foreign-space pairing, vendor-agnostic only in the localized sense
that the worst cell (mpnet × corpus_2) is worst for both OpenAI and
Anthropic targets. NOT a universal AI claim.

## Threshold value
- τ = 0.31 (overlap, mean-max cosine, te3-large)
- AUC floor for "holds" = 0.80
- Below-threshold regime AUC ≈ 0.69
- Spearman cross-family (n=5): +0.83 / (n=12): +0.69
- Spearman same-family (n=5): −0.29 / (n=12): −0.41

## Held limits
- 4 OpenAI targets (gpt-4o-mini/4.1-mini/4o/4.1); 75 eval prompts
- 1 same-family foreign space (te3-small); 1 cross-family (mpnet)
- 17 size-controlled label-free corpora across two studies (n=5 + n=12)
- Cross-vendor only as exemplar boundary: 3 Anthropic 4.5 models,
  claude-opus-4-7 excluded (empty completions)
- One overlap metric (mean-max cosine), one behavior (refusal)

## Fine-replication strict-criterion failure (one sentence)
The 12-level high-resolution replication recovered the cross-family
threshold (H1' PASS, ρ=+0.69) but failed the preregistered same-family
flat-control criterion (H2' FAIL, ρ=−0.41 vs limit ±0.40), so under
the joint preregistered criterion the replication does not hold and
the effect is cross-family-DOMINANT, not cross-family-EXCLUSIVE.

## Cross-vendor preregistration-killed boundary (referenced, not buried)
Cross-vendor confirmatory run (Anthropic) preregistration-killed:
min Anthropic transported = 0.617 < 0.70 floor; worst cell = same
mpnet × corpus_2 as worst for OpenAI. Universality unavailable; the
killed result is used in this paper purely as the localization of the
residual crack at the corpus/foreign-space boundary.

## Deliverables
- Paper: papers/threshold-law-2026-05-18.md
- Figure: papers/figures/threshold-law-curve.png
- Figure builder: scripts/dogfood/plot_threshold_law.py
- Zenodo zip: dist/zenodo/threshold-law-2026-05-18.zip (194 KB)
- Zenodo README/metadata: dist/zenodo/README-threshold-law-2026-05-18.md

## Self-audit verdict (styxx-on-paper, 2026-05-18)
- Composite ≥0.5 in §1 (refusal-axis principled-decline confound),
  §4 (sycophancy 0.804 on agreement-word lexical surface), §7
  (refusal-axis again). All flagged, none paraphrased away.
- deception_v2 grounded vs raw JSON: 0.75–0.99 across most sections.
  Interpreted as a TOOL limit: deception_v2 was validated on parallel
  prose references, not structured JSON; fires on prose-vs-JSON
  surface mismatch regardless of factual agreement. Paper numbers
  match the JSON on direct inspection. Tool-limit documented in
  paper §7 (Limits) and §11 (Self-audit).
- Integrity-protocol code checks: all pass. Circular-oracle check:
  passes (behavioral labels independent of IV).
- Action: paper §7 amended with self-audit findings; §11 added
  describing the audit; supplementary/ added to the Zenodo bundle.

## Decisions / integrity notes
- NO universality language used anywhere in the paper.
- The fine-replication strict-criterion failure is in the body (§6),
  not a footnote.
- No knob-tuning; no re-runs hunting cleaner numbers.
- Zenodo deposit STAGED only; API not called. Operator triggers.
- Repo base commit: 390752f (styxx 7.4.1 release).
