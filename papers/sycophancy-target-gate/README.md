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

## Post-ship hardening (2026-05-24)

Adversarial review of the shipped guard (battery of self-aggrandizement,
mixed self/other, self-framed flattery, empty/short/all-pronoun, unicode):
- **No backfire on any input** — the gate is suppress-only and never raised a
  firing. Flattery always fires (incl. curly-apostrophe and "i think you're…"
  lead-ins). Curly-apostrophe (U+2019) apologies are correctly suppressed.
- Surfaced a non-monotonicity in `gated_sycophancy_risk`: a self-correction that
  uses counter-vocab ("however") loses a protective negative contribution under
  neutralization, so `gated` can EXCEED `raw`. The suppress-only guarantee is the
  `min(raw, gated)` in `_cogn_needs_revision`, not that function — pinned by a new
  test. A misleading per-sample `gated <= raw` assertion was replaced.
- Self-directed superlative self-praise ("my solution is brilliant") still fires
  (superlatives are never neutralized, by design) — a conservative residual, not
  a safety issue.
- `ruff` clean; full suite **1024 passed, 6 skipped**; `styxx doctor` healthy.

### Word-boundary instrument re-fit — validated, NOT executed (greenlight pending)

The word-boundary fix lifts v0 5-fold CV 0.9720 → 0.9805, but re-fitting the
*published* instrument has real blast radius: it changes `calibrated_weights_
sycophancy_v0.py` (coefs/scaler/fingerprint), breaks the pinned exact-score
fingerprint test (`test_attack_v0`, 1e-6 round-trip), shifts the shipped
`rf_05b21c` register fixture (tol 0.05), and changes the headline AUC published in
the DOI'd paper (10.5281/zenodo.19777921). Since the production *harm* is already
fixed (the gate path uses word-boundary internally), silently re-fitting a
DOI-published instrument is the wrong move. Correct path (operator greenlight): a
deliberate **v0.1** bump — regenerate weights, update the fingerprint + pinned
tests to new values, version bump, paper erratum noting the tokenization fix. Not
done autonomously.

## Follow-ups resolved (2026-05-24, end-to-end)

- **Cross-vendor** — confirmed **hard-blocked**: `ANTHROPIC_API_KEY` is present in
  the env but **empty** (SDK can't resolve auth); OpenAI is the only usable key. Not
  faked with analyst-generated text. Runner (`gen_holdout_crossvendor.py`) is wired
  and runs the moment a real key is dropped.
- **Restrained-technical FP** — pre-registered, run once → **CLOSED NEGATIVE**
  (`FINDING_restrained_2026_05_24.md`, prereg `54e91b9` → result `70ac4bc`). The
  obvious lexical fix (C3: neutralize when no interlocutor + no superlative) fixes the
  FP (0.82→0.00) but collapses content-free opinion-agreement sycophancy recall
  (1.00→0.03), because "Yes, the speed of light is X" (factual) and "Yes, absolutely,
  completely agree" (sycophantic) are lexically identical — the difference is
  semantic. The real fix is the v1 **NLI stance feature**, not a surface patch. The
  self-vs-other gate's boundary is now mapped: it works where direction is encoded in
  grammar (self vs interlocutor), not where the distinction is semantic.
  - **Prompt-side retry (C4) — also CLOSED NEGATIVE** (`FINDING_promptopinion_2026_05_24.md`,
    prereg `689e4d1` → result `0526739`). The signal *is* in the prompt (factual question
    vs stated opinion), and a lexical `prompt_has_opinion` detector separated my templates
    100% — but on a FRESH **varied-phrasing** holdout it caught only **47%** of naturally-
    phrased opinions (overall detector accuracy 0.73 < the pre-declared 0.85), tanking
    sycophancy recall (content-free agreement 1.00→0.47). The 100% was a template artifact;
    the pre-registered generalization confound (P5) caught it before any ship. **Both
    lexical shortcuts (response-side C3, prompt-side C4) are now foreclosed** — opinion-vs-
    fact is inherently semantic, so the only remaining path is a semantic signal (next).
  - **Semantic gate (C5) — PASS, SHIPPED 7.6.0** (`FINDING_semantic_2026_05_24.md`, prereg
    `4e99ad0` → result `bc6dd4a`). Same prompt-opinion signal, but **semantic** substrate:
    embed the prompt (all-MiniLM-L6-v2) vs frozen opinion/fact anchor centroids. On a fresh
    NEW-topic varied holdout it cleared all five bars — factual-confirmation FP 0.00,
    flattery recall **1.00**, content-free-agreement recall **1.00** (lexical C4: 0.58),
    subjectivity accuracy **1.00** (lexical: 0.73). styxx's first content-aware sycophancy
    gate; shipped as an **opt-in optional tier** (`styxx.guardrail.semantic_subjectivity`,
    `STYXX_SEMANTIC_SYCOPH=1` + `styxx[nli]`; default OFF, suppress-only). Honest bound: the
    **decoupled-diagonal** (prompt FORM ≠ premise TRUTH) remains — measured small (models
    correct false premises); full truth-grounding is future work.

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
