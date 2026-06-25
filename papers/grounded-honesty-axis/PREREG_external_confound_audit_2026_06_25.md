# PREREG — external-model LENGTH-confound audit: do REAL deployed classifiers ride length?

**Frozen 2026-06-25 BEFORE generating. Tool: styxx.audit_confound (frozen bars). Generator: Gemini 2.5-flash
(paid, thinking off, temp 0). Targets: two widely-downloaded HuggingFace classifiers (NOT ours). Author: styxx.**

## Why (the edge)
Everything so far audited styxx's OWN instruments. The decisive test of the tool is whether it finds confounds
in REAL, externally-deployed, black-box classifiers that millions use. If `audit_confound` can reveal a hidden
length bias (and a fix) in a third-party model, it is a universal auditor, not a self-referential check.

## Targets (black-box, scored via their own output probability)
- **distilbert-base-uncased-finetuned-sst-2-english** (sentiment; the default HF sentiment pipeline). Construct =
  POSITIVE(1) vs NEGATIVE(0); score = P(POSITIVE).
- **unitary/toxic-bert** (Detoxify; toxicity moderation). Construct = rude/toxic(1) vs civil(0); score = P(toxic).

## Design
`build_confound_grid`: 50 neutral topics × 2 construct stances × {short ≈ 1 sentence/~15w, long ≈ 5–6
sentences/~90w}, Gemini-generated (sentiment: review the topic; toxicity: comment on the topic). Construct ⟂
length by construction. Score every text with the REAL model. **construct_recoverable_auc** = a 5-fold CV
TF-IDF/BoW logistic refit on the text (model-AGNOSTIC: is the construct recoverable from words at all?), which
disambiguates a broken model from a degenerate corpus.

## Frozen verdicts (audit_confound bars, unchanged)
ROBUST / THRESHOLD-BIASED (+guard) / CONFOUND-DEPENDENT (broken) / INCONCLUSIVE — gated on orthogonality
(|corr(label,len)|≤0.20) and construct-recoverability (BoW refit ≥0.60).

## Predictions (context, NOT bars)
Unknown. Strong modern classifiers MAY be robust (→ the tool clears a real model, also a valid result) or MAY
ride length (toxicity models plausibly dilute toxicity in verbose text / over-flag long civil text — a real
adversarial-evasion vector). Either outcome is a clean external-validity result for the tool.

## Honest scope
Single generator, single seed, n=200/model, ONE confound (length). Construct = a Gemini-instantiated stance
(checked by the BoW refit), not gold human labels. Toxic stance is mild incivility/condescension, not extreme
content. Verdicts gated + CI-backed. Tool: `scripts/external_confound_audit.py`, `styxx.audit_confound`.
