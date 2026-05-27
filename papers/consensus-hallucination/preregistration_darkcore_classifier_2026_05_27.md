# Pre-Registration · Dark-Core Classifier — the deployable form of the Decorrelation Ceiling

**Committed BEFORE training, BEFORE eval, BEFORE data inspection.** This pre-registration locks
the bars *and* the held-out test sets before any classifier is trained or evaluated. The
classifier code (`darkcore_classifier_2026_05_27.py`) and the labeled corpus splits are
committed in the same commit as this prereg.

## The bet

The four-method consensus-hallucination arc (Dark Matter, CVPD, JD, ICT) tested *detection* —
given a generated answer, can a reference-free divergence method flag it as wrong? Result:
no, on the dark core. The Decorrelation Ceiling synthesis explains why: shared cultural-prior
errors have no decorrelated competitor in training.

This is a different question on a different axis: *classification*. Given a question alone
— before any generation — can a simple text-feature classifier predict which class the
question falls into?

- **folklore-class** (Walt Disney frozen, Marie Antoinette cake, lucky rabbit's foot, …) —
  the dark core, no decorrelated competitor available.
- **pseudoscience-class** (psychics, Roswell, astrology, Libras' personality, …) — the
  debunk is in training; partially detectable; partially correctable.
- **factual-error-class** (everyday confusable facts) — alternative is available; movable
  under injection (ICT: 0.231 yield).
- **truth-shaped** (basic facts, capitals, basic science) — the control.

If a classifier reaches the bars, styxx ships a routing primitive
`styxx.classify_dark_core(question)` that — for any new question — predicts the
question's class. The class then conditions the agent's behavior:

- folklore → refuse, or require authoritative external grounding before answering
- pseudoscience → use RAG (debunk is in training but partially unreliable)
- factual-error → standard generation is fine
- truth-shaped → standard generation is fine

This converts the synthesis's detection-side floor into a *deployable routing primitive*
anchored in pre-registered empirical receipts. The synthesis predicts the linguistic
signature of folklore questions (cultural-prior topics) should be distinguishable from
factual questions; this is the implementation of that prediction.

## Design

### Training corpus

- **ICT receipts (`probe_ict_results.json`):** 50 items already classified into folklore (4),
  pseudoscience (6), factual-error (13), self-referential (~3), truth (~24) by the committed
  `analyze_darkcore.categorize` rules + the ICT probe's TruthfulQA classification step.
- **ICT-folklore corpus (`corpus_folklore_2026_05_27.py`):** 30 hand-curated folklore items +
  30 truth items.

This gives ~110 labeled items total. After dedup and class consolidation:
- folklore: ~34 (4 from ICT + 30 curated)
- pseudoscience: ~6
- factual-error: ~13
- truth: ~57

### Held-out test sets (LOCKED HERE, pre-eval)

- **In-distribution held-out test:** 20% stratified random split from the ICT receipts,
  seed = 20260527, fit-once. The 80% remaining + ALL the ICT-folklore items minus the
  cross-corpus holdout (below) are training.
- **Cross-corpus generalization test:** the full ICT-folklore *folklore* corpus (30 items),
  held out as a generalization probe. The training only sees ICT folklore items (4 of them),
  not the curated 30. K3 measures whether the classifier generalizes from
  TruthfulQA-derived folklore patterns to hand-curated cultural-prior items.

### Classifier

- **Embedding:** `all-MiniLM-L6-v2` (sentence-transformers, the same model the styxx semantic
  tier uses; lightweight CPU-friendly).
- **Head:** logistic regression (sklearn), one-vs-rest for the 4-class problem, with class
  weights balanced to address class imbalance.
- **Features:** raw embedding only, no engineered features. This is intentionally simple —
  the question is whether the *embedding alone* picks up the linguistic signature, not
  whether a sophisticated model can be tuned.

No hyperparameter search, no model selection — one classifier specification, fit once, score
once, against the locked test sets.

## Kill-gate (PASS iff K1 ∧ K2 ∧ K3)

| id | bar |
|---|---|
| **K1** | folklore-class F1 ≥ **0.70** on the in-distribution held-out test (the load-bearing detection axis) |
| **K2** | 4-way classification accuracy ≥ **0.65** on the in-distribution held-out test (baseline-better than majority-class predictor) |
| **K3** | folklore-class F1 ≥ **0.60** on the cross-corpus generalization test (the 30 curated folklore items, never seen in training) |

**K3 is the load-bearing bar.** If K1 ∧ K2 pass but K3 fails, the classifier works
in-distribution but is overfit to TruthfulQA-derived folklore patterns; it would not be
shippable as a general routing primitive. K3 makes the test honest.

PASS (K1 ∧ K2 ∧ K3) → ship `styxx.classify_dark_core(question)` as a public CLI + Python
primitive. The synthesis becomes a deployable router.

FAIL K1 → folklore questions are linguistically indistinguishable from non-folklore at the
embedding level. The dark core is dark to text-only classification *as well as* text-only
detection. A different bound on the synthesis.

FAIL K3 only → the principle holds for TruthfulQA-shaped folklore but not the hand-curated
cultural-prior items. The classifier overfits a corpus shape rather than learning the
underlying signature. The honest version: not yet deployable; needs more diverse training data.

## Honest prior

The synthesis's claim is that folklore items differ from factual questions in *whether a
decorrelated competitor exists*, which is a property of the training data the model was
trained on — not necessarily a linguistic property of the question. But folklore items also
tend to have specific topical signatures (cultural references, well-known names, superstitious
language) that an embedding *should* pick up. Empirical prior:

- K1 (in-distribution folklore F1): 60–70% likely to pass at 0.70. The 4 ICT folklore items
  are TruthfulQA-derived and may share a corpus-specific signature.
- K2 (4-way accuracy): 70–80% likely to pass at 0.65. The truth class is large and very
  distinct linguistically.
- K3 (cross-corpus folklore F1): 40–50% likely to pass at 0.60. The hard test. Hand-curated
  items span more cultural-prior topics than the TruthfulQA-derived 4, so the model has to
  generalize across topical surface.

Joint K1 ∧ K2 ∧ K3 PASS: ~30–40%. **This is genuinely a coin-flip with stakes.** If it passes,
styxx has a shippable router anchored in the four-method synthesis. If it fails K3, the floor
is also dark to text-only classification — a meaningful different bound.

## Run-once protocol

1. This prereg, the classifier code, and the corpus splits are committed in the same commit.
2. The commit is pushed to public origin BEFORE the classifier is trained.
3. The classifier is trained once on the locked training split.
4. The classifier is evaluated once on each held-out set.
5. The results JSON and the FINDING markdown are committed in a separate commit, after the eval.
6. No retraining, no hyperparameter tuning, no re-splitting.

## Reproducibility

- `preregistration_darkcore_classifier_2026_05_27.md` — this file (bars locked before training).
- `darkcore_classifier_2026_05_27.py` — the classifier code (committed with this prereg).
- `darkcore_classifier_results.json` — the eval output (will not exist until the run completes).
- Training corpus = `probe_ict_results.json` (rows with category labels) + `corpus_folklore_2026_05_27.py` (folklore + truth lists).

The discipline is verifiable from the git history: prereg + code committed together at commit
X; eval results + FINDING committed at commit X+1 strictly after X is pushed.
