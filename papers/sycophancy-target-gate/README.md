# sycophancy self-vs-other target gate

Fixing the sycophancy instrument's false positives on honest **self-directed
apology / self-correction** ("my mistake", "that was wrong") — which, after
7.4.4 made sycophancy the sole *trusted* gating axis, were directly tripping
`needs_revision`.

All under the project's text-only-recalibration discipline: a pre-registered
held-out kill-gate, holdout hashed before scoring, run once, bars never lowered.
Unlike the overconfidence/deception closed negatives, the self-vs-other target
is an explicit **grammatical** signal (pronoun attachment), not a latent
construct — so it had a real mechanistic chance, and cleared the bar.

## The mechanism

The false positive is **not** `superlative_density` (the K=1 critical feature
stays clean at 0.000 on all honest text). It is (a) a substring artifact
(`"correct" ∈ "corrected"`, `"fully" ∈ "carefully"`) and (b) `counter_lexicon`
*absence* (terse honest declaratives lack "however/but"). The gate adds a
self-vs-other **attachment** signal: praise/agreement is *outward* iff a
second-person token is within ±4 tokens; a response is `self_directed` when no
hit is outward-attached AND it has ≥2 first-person tokens. When self-directed,
the yielding-family features are neutralized; `superlative_density` stays active
so flattery is never softened.

## Results (kill-gate P1–P4, τ = 0.30; PASS = all four)

| run | FPR apology | recall flattery | AUC flat·apol | verdict |
|---|---|---|---|---|
| in-distribution (gpt-4o-mini, n=140) | 0.36 → **0.06** | 1.00 | 0.999 | **PASS** |
| cross-model (gpt-4o + gpt-3.5-turbo, n=140) | 0.20 → **0.08** | 1.00 | 0.996 | **PASS** |
| v0 5-fold CV (word-boundary, offline) | — | — | 0.9720 → **0.9805** | bug fix safe |

The target gate is *necessary*: baseline AND word-boundary-fix-only both fail
P1; only the gate clears it. The adversarial 2nd-person apology subclass
("i told you X; that was wrong") → FPR 0.00, 94% correctly self-directed by
attachment (not pronoun-counting). Gate never misfired on flattery (0/100).

## Bounds (honest)

- **Cross-VENDOR untested** (no Gemini/Anthropic key in env; blocked on operator).
  All passing runs are OpenAI-family text.
- A **separate, unfixed** FP mode surfaced: restrained-technical FPR ~0.30
  (gpt-3.5-turbo 0.60) — terse "yes, it's true that…" honest answers fire the
  detector; the self-vs-other gate correctly leaves them untouched. Candidate
  for the next bet.
- Residual: superlative proper-noun collisions ("Great Wall"), low-self-reference
  (`self_n<2`) terse apologies.

## What shipped

- `styxx/guardrail/self_directed_gate.py` + `cognometrics._cogn_needs_revision`
  `response=` guard (suppress-only `min(raw, gated)`; no published-weight change),
  tests in `tests/test_self_directed_gate.py`. Full suite 1019 passed. (`5414d80`)
- The word-boundary tokenization fix is validated but **not yet** in the published
  instrument (needs a weight re-fit + fingerprint bump).

## Commit map

| stage | commit |
|---|---|
| diagnosis + prereg + frozen gate (before data) | `fce969b` |
| in-distribution holdout hashed (before scoring) | `2b286d3` |
| in-distribution kill-gate PASS (run once) | `76248d6` |
| cross-model prereg + word-boundary CV | `88c81d4` |
| cross-model holdout hashed (before scoring) | `45156e5` |
| cross-model kill-gate PASS (run once) | `66bd3a3` |
| **ship: cognometrics guard + tests** | `5414d80` |

## Files

`DIAGNOSIS_2026_05_24.md` · `preregistration_2026_05_24.md` ·
`preregistration_crossmodel_2026_05_24.md` · `FINDING_2026_05_24.md` ·
`FINDING_crossmodel_2026_05_24.md` · `target_gate.py` (frozen experiment) ·
`gen_holdout.py` / `gen_holdout_crossmodel.py` · `run_killgate.py` /
`run_killgate_crossmodel.py` · `run_wordboundary_cv.py` · `results*.json` ·
`holdout/` · `holdout_manifest*.json`.
