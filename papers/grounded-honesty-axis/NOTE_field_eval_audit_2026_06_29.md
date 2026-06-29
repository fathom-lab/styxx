# NOTE — the within-corpus length fingerprint does NOT separate synthetic from real (so ground truth is the only gate)

*fathom-lab · 2026-06-29 · field-level follow-up that **corrects** [NOTE_program_synthetic_audit_2026_06_29](NOTE_program_synthetic_audit_2026_06_29.md)
and qualifies [NOTE_manufactured_confound_2026_06_29](NOTE_manufactured_confound_2026_06_29.md).
Repro: `field_eval_audit_repro.py` (datasets + scikit-learn; no model download).*

We pointed the shipped probe (`_lexical_entanglement`: length-invariant binary/`norm=None` BoW margin,
shuffled folds, within-label permutation test, confound = length) at four **widely-used, real, human-labeled**
benchmarks, n=800 balanced each. The expectation (from this morning's program audit) was that real corpora
would read low and generated ones high. **That expectation was wrong.**

| corpus | provenance | n | len_corr | perm-p | BoW-AUC |
|---|---|---|---|---|---|
| IMDB (sentiment) | **real** | 800 | **0.376** | 0.003 | 0.971 |
| Yelp-polarity (sentiment) | **real** | 800 | 0.271 | 0.003 | 0.882 |
| Civil Comments (toxicity) | **real** | 800 | 0.275 | 0.003 | 0.875 |
| SST-2 (sentiment) | real | 800 | 0.108 | 0.008 | 0.708 |
| *our LLM-generated corpora* | synthetic | 200 | *0.26–0.83* | ≤0.005 | ~1.0 |
| true-null control (construct ⟂ length) | — | 800 | **0.022** | **0.77** | — |

**Real benchmarks carry the fingerprint at magnitudes squarely inside the generated range — IMDB (0.376)
exceeds most of our synthetic corpora.** The same probe reads a clean null (0.022, p 0.77) on a constructed
true-null at the same n, so this is **real length↔construct coupling in real data**, not a probe artifact:
longer real reviews/comments genuinely carry more (or more separable) construct vocabulary within label.

## The correction

This morning's program audit reported "generated 75% flagged vs natural 40%" and read it as a
synthetic-specific signal. **It is not.** That gap was an artifact of panel composition (the natural side was
mostly *short* statements) and statistical power, not provenance. On length-rich real benchmarks the
within-corpus fingerprint flags as hard as on generated corpora. So:

> **The within-corpus lexical-entanglement fingerprint is a coupling *detector*, not a provenance or artifact
> *discriminator*. It cannot tell a generator-manufactured confound from a real one — because real eval data
> has the same coupling.**

## Why this strengthens, not weakens, the thesis

If even a careful, length-invariant, permutation-tested within-corpus probe can't separate manufactured from
real coupling, then **no within-corpus diagnostic can** — which makes ground-truth validation *non-optional*.
The only way to know whether a model's *score* rides a confound artifactually is to re-run the audit on
length-matched **real human labels** (`styxx.validate_against_ground_truth`) and see if the verdict survives.
7.23.0's design — fingerprint *subordinate*, ground truth *decisive* — is vindicated. Its warning **wording**
("…the artifact's fingerprint") over-claimed and is corrected to *corroborating, not diagnostic*.

(Precision: this corrects the *within-corpus coupling probe* `_lexical_entanglement`, **not** the original
report-card refutation — that used the score-shift on length-matched real human labels, a ground-truth metric,
which stands. The two are different measurements; only the within-corpus one fails to discriminate provenance.)

## Honest scope

n=800 real vs n=200 generated (the comparison is on *magnitude*, not the p-flag, which saturates at scale —
SST-2's 0.108 still "flags" at p 0.008). *Why* real data couples (genuine vocab-entanglement vs
"longer texts are more separable") is open and worth a follow-up. Length-only confound; four real benchmarks.
The remaining slice — auditing canonical **public LLM-generated / LLM-as-judge** eval sets (which needs careful
dataset sourcing) — is still open, but its headline metric must now be **ground-truth reconciliation**, not the
within-corpus fingerprint.
