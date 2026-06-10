# PREREG — Anatomy v1: idiosyncrasy or range-artifact? (disattenuation test)

**Frozen 2026-06-10, before any scored run. Fathom Lab / styxx. Self-falsification follow-up to
`FINDING_anatomy_v0_2026_06_10.md` (SUPPORTED), attacking its own named confound.**

## Question

v0 found professions (0.0594) and furniture (0.0116) carry almost no cross-mind convergence. Two
readings: (a) **idiosyncrasy** — mind families genuinely draw different maps for social-role and
artifact-arrangement meaning; (b) **range-artifact** — those domains' sub-geometries are simply
unreliable (members tightly clustered, little structure to measure), so ANY between-mind comparison
floors regardless of agreement. v1 separates them with the standard psychometric instrument:
reliability-corrected (disattenuated) convergence.

## Apparatus (frozen)

- Reps: per-mind, per-domain sub-RDMs at the frozen 0.66-layer convention, frozen battery, computed
  from SPLIT TEMPLATE HALVES — half A = templates 1–4, half B = templates 5–8 of the frozen list
  (order as in `styxx.mind.TEMPLATES`). All 10 Atlas minds (6 anchors recomputed live for halves;
  4 independent subjects).
- **Reliability** of mind m on domain c: partial-lexical correlation between m's half-A and half-B
  sub-RDMs, Spearman–Brown corrected to full length: `rel = 2r/(1+r)` (halves average 4 templates;
  the full measure averages 8). Negative r floors to rel = 0.
- **Disattenuated convergence** for a cross-family pair (m, a) on domain c:
  `raw_RSA(m,a,c) / sqrt(rel_m,c * rel_a,c)`, computed ONLY where both rel >= 0.20 (the
  measurability floor); capped at 1.0 for reporting. raw_RSA uses the full-template reps
  (anchors npz / persisted atlas_live_reps.npz), exactly as v0.
- A domain is **UNMEASURABLE** for a mind if rel < 0.20; globally unmeasurable if fewer than 2 of
  the 4 independent subjects are measurable on it.

## Pre-registered outcomes (frozen; exactly one will be claimed)

- **IDIOSYNCRASY-CONFIRMED:** profession AND furniture are measurable (escape the floor for >= 2
  independent subjects each) AND both remain in the bottom 2 of the mean disattenuated anatomy over
  the independent subjects. The v0 reading hardens: social-role/artifact meaning is genuinely
  family-specific.
- **RANGE-ARTIFACT:** profession or furniture is globally UNMEASURABLE (the confound was real: there
  is too little stable structure in those domains to compare), or after correction it leaves the
  bottom 2. v0's interpretation is corrected loudly; the map is re-drawn with measurability marked.
- **P2 (anatomy robustness, both branches):** Kendall's W over the disattenuated rankings of the
  measurable domains across the 4 independent subjects, with the v0 bar: W >= 0.60 AND perm p < 0.05
  (10,000 draws, seed 0, permutations over the measurable-domain ranks). If P2 fails, the v0
  shared-anatomy headline is REOPENED regardless of branch.

## Validity controls / VOID

- **C1:** mean over domains of full-template raw convergence recomputed here must match v0's
  receipt per subject to ±0.005 (same pipeline, same numbers). Else VOID-PIPELINE.
- **C2 (sanity):** half-A vs half-B reliability of the FULL 96-concept RDM must exceed 0.5 for
  every mind (templates measure something stable at battery level). Else that mind is dropped and
  disclosed; <3 independent subjects remaining = VOID-SUBSTRATE.
- Smoke writes `*_SMOKE_INVALID*` only.

## Honest prior

Genuinely uncertain — that is the point. Profession plausibly RANGE-ARTIFACT (12 humans-in-roles may
cluster tightly); furniture could go either way. The shared-anatomy claim (P2) expected to survive.
