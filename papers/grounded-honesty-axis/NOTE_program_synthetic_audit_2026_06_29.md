# NOTE — how often does a generated eval corpus carry the length-confound fingerprint? (program-scale)

> **⚠ CORRECTED same day by [NOTE_field_eval_audit_2026_06_29](NOTE_field_eval_audit_2026_06_29.md).** The
> "generated 75% vs natural 40%" gap below does **not** hold up: on length-rich REAL benchmarks (IMDB 0.376,
> Yelp 0.271, Civil Comments 0.275, n=800) the within-corpus fingerprint flags at magnitudes comparable to
> generated corpora. The gap was panel composition (short natural statements) + statistical power, not a
> provenance signal. Read the prevalence below as *"where length↔construct coupling lives,"* **not** *"where a
> generator manufactured it"* — the fingerprint is a coupling detector, not an artifact discriminator. Ground
> truth (`validate_against_ground_truth`) is the only discriminator.

*fathom-lab · 2026-06-29 · broadens [FINDING_groundtruth_substrate_artifact_2026_06_27](FINDING_groundtruth_substrate_artifact_2026_06_27.md)
+ [NOTE_manufactured_confound_2026_06_29](NOTE_manufactured_confound_2026_06_29.md) from "we audited two corpora"
to the whole program. Repro: `program_synthetic_audit_repro.py` (scikit-learn only; no model download, no network).*

We ran the **shipped** styxx probe (`_lexical_entanglement`: length-invariant binary/`norm=None` BoW margin,
shuffled folds, within-label permutation test) on **every** text+binary-label eval corpus in the repo, with
confound = length (log word count), and asked: does a label-trained bag-of-words *itself* ride length within
class (the manufactured-confound fingerprint)?

## Prevalence

| provenance | corpora flagged ENTANGLED (perm-p < 0.05) |
|---|---|
| **LLM/model-generated** | **15 / 20 (75%)** |
| natural controls | 2 / 5 (40%) |
| unknown (`*_v0`, likely generated) | 3 / 4 (75%) |

The fingerprint is **program-wide**, spanning six constructs: sentiment (`lengrid` 0.705, `boundary` 0.325),
toxicity (`boundary` **0.828**, `lengrid` 0.698), sycophancy (`confound_grid` 0.669), overconfidence
(`adversarial_lenxreg` 0.499), plan_action (`confound_grid` 0.418), deception (`confound_grid` 0.264). All
perm-p ≤ 0.005.

## Three honest reads

1. **It tracks generation design, not "generated text" per se.** The corpora that *varied length as the
   confound* (the `confound_grid` / `boundary` / `lengrid` sets) entangle hardest. The corpora that
   *length-matched at generation* are mixed: deception's `pairs_lenmatched` / `responses_lenmatched` come back
   **clean** (corr 0.01–0.10, n.s.), but two overconfidence `pairs_lenmatched` sets **still flag** (0.27 / 0.23,
   p ≤ 0.005). So matching token-count **reduces but does not reliably eliminate** the artifact —
   *length-matching ≠ lexical orthogonality*; residual construct-vocabulary↔length coupling survives.
2. **The probe is a SCREEN, not a verdict.** It flags length↔construct coupling whether *manufactured* or
   *real*. The two natural flags are curated factual truthsets (`controlled_truthset` ±negated, 0.245) where a
   genuine length↔truth-value coupling exists — not a generator artifact. The genuinely free-natural set
   (`ood_naturals`) is **clean** (0.078, p 0.82), as is `wide_truthset` (0.125, p 0.10). The arbiter of
   manufactured-vs-real remains ground truth: `styxx.validate_against_ground_truth`.
3. **Generated ≈ 2× the natural flag rate, and the natural flags are explainable.** Net: the
   manufactured-length-confound risk is concentrated in generated content and pervasive across the program —
   the report-card self-falsification was not a one-off, it was a property of the generation recipe.

## Scope / limits (honest)

One lab's pipeline (mostly Gemini-generated), small panel (20 generated / 5 natural / 4 unknown);
provenance is a filename heuristic; the confound tested is length only; the probe screens coupling, it does
not by itself prove a model's *score* rides it. This is the **internal** slice of the "how often does
synthetic eval lie" question — the field-level extension (auditing widely-used public synthetic / LLM-judge
eval sets) and the non-saturated **toxicity-on-real** confirmation are the next two runs.
