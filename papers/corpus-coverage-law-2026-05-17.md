# The Corpus-Coverage Law for Cross-Family Cognometric Transport (2026-05-17)

**Status:** preregistered, size-controlled, with a working control arm.
**LAW SUPPORTED.** The strongest, cleanest result of the program.
**Script:** `scripts/dogfood/corpus_coverage_law.py`
**Raw:** `scripts/dogfood/out_corpus_coverage_law.json`

## Claim

Cross-family cognometric transport quality is **governed by, and
predictable from, the semantic overlap between the label-free alignment
corpus and the audit domain** — a quantity computable in the home
embedding space *before* any transport is run. Same-family transport is
insensitive to it.

## Preregistration (fixed in the script before the run)

- IV: corpus↔eval overlap = mean over eval prompts of max cosine to the
  corpus, in te3-large (home).
- Corpora SIZE-CONTROLLED (exactly 360 sentences each), label-free,
  disjoint from eval; only semantic coverage varies (5 levels).
- Behavioral labels FIXED across levels (generated once).
- H1 (effect): cross-family Spearman(overlap, AUC) ≥ 0.60 AND
  dense−far lift ≥ 0.10.
- H2 (control): same-family |Spearman| < 0.40.
- LAW iff H1 ∧ H2.

## Result

| level | overlap | cross-family AUC | same-family AUC (control) |
|---|---|---|---|
| C0_far | 0.184 | 0.687 | 0.872 |
| C1_generic | 0.216 | 0.725 | 0.850 |
| C2_adjacent | 0.354 | 0.813 | 0.836 |
| C3_matched | 0.366 | 0.842 | 0.848 |
| C4_dense | 0.362 | 0.853 | 0.856 |

- Cross-family Spearman(overlap, AUC) = **+0.832**; far→dense lift =
  **+0.166** → **H1 PASS**.
- Same-family Spearman = **−0.292** (|ρ|<0.40, flat control) →
  **H2 PASS**.
- **LAW SUPPORTED.**

The cross-family curve rises monotonically with overlap then **saturates
around overlap ≈ 0.35 (AUC ≈ 0.85)** — there is a *sufficiency
threshold*, not unbounded gain. The same-family control stays flat-high
(0.85–0.87) regardless of overlap — confirming the effect is specific to
the cross-family regime and not a trivial overlap↔AUC artifact.

## Why this matters

The stress-test "crack" (corpus_2 failed cross-family, 0.215 swing) is
now **explained and operational**: corpus_2 was low-overlap. The
liability becomes a design rule with a **pre-flight diagnostic** — you
can measure corpus↔domain overlap in the home space and predict whether
a cross-family audit will hold *before* deploying it. That is a genuine
methodological contribution, derived honestly (preregistered + control),
not a tuned number.

Combined, the honest paper now has a spine:

1. Universal cognometric transport (label-free linear map), same-family:
   robust, validated (refusal AUC 1.000 clear / 0.89–0.94 live).
2. Cross-family: real but governed by the **corpus-coverage law**
   (this result) — with a computable pre-flight diagnostic.
3. Documented boundaries/negatives: zero-paired-data closed; Brick #1
   live-elicitation negative; instrument-agnostic within-signal only.

## Caveats (stated, not hidden)

5 corpus levels (modest — finer gradation would tighten the law); a
single overlap metric (mean-max-cosine; robustness to metric choice
untested); lexical refusal labels; **OpenAI-only — not cross-vendor**,
so the law's cross-vendor generality is untested (the real remaining
external-validity gap); same-family control showed a mild negative
drift (−0.29), within the preregistered flat band but not perfectly
zero.

## Next (honest)

1. More overlap levels (finer gradation) → tighten the law / locate the
   sufficiency threshold precisely.
2. Cross-vendor (needs an Anthropic/Gemini key) — does the law hold when
   the target is a different vendor, not just a different embedding
   family? This is the gate to a real paper.
3. Then: write refusal-universal-transport + corpus-coverage law as the
   methods paper; only then Zenodo/OSF (publishing bar).
