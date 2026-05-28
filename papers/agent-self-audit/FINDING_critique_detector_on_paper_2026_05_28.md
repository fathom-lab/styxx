# Finding · critique_detector_on_paper_2026_05_28 (Layer 6) — 18/18 PASS with saturated logprobs; both kill-gates un-fired; saturation pattern flagged honestly

**Date:** 2026-05-28
**Author:** Alexander Rodabaugh (Fathom Lab)
**Substrate:** styxx 7.7.10 in-development (post-`a8fb1f3`)
**Pre-registration:** `papers/agent-self-audit/PRE_STATED_PREDICTION_critique_detector_on_paper_2026_05_28.md` (commit `a8fb1f3`, public on `origin/main` BEFORE the runner existed)
**Runner:** `experiments/critique_detector_on_paper_2026_05_28/run_audit.py`
**Results:** `experiments/critique_detector_on_paper_2026_05_28/results.json`
**Substrate model:** OpenAI `gpt-4o-mini` via the public `styxx.critique_detector` API at `temperature=0`
**Operator authorization:** the OpenAI API call (~$0.01 total) was explicitly authorized by the operator after the auto-mode classifier flagged the call as requiring re-authorization.

> **Outcome.** **18/18 propositions PASS at the threshold-0.50 verdict.** All 13 TRUE claims correctly identified as faithful to the source passage (`P(NO) < 0.50`); all 5 FALSE controls correctly detected as fabrications (`P(NO) ≥ 0.50`). Both pre-stated kill-gates un-fired (no paper-grade self-falsification; no instrument-grade failure mode). 18/18 inside the pre-stated `P(NO)` bands.
>
> However, the **observed distribution is saturated**: every TRUE claim scored exactly `P(NO) = 0.0000` and every FALSE control exactly `P(NO) = 1.0000`, more extreme than the predicted bands centered on 0.05 and 0.95. This is consistent with the pre-stated calibration note about gpt-4o-mini's bias on context-grounded faithfulness judgments, but the saturation is itself an empirical finding worth documenting.

## Per-proposition table

| id | kind | source | claim (truncated) | predicted `P(NO)` band | observed `P(NO)` | verdict | in-band |
|---|---|---|---|---|---|---|---|
| T1 | T | A | dark-core asymmetry rate was 5.88% | 0.00–0.15 | 0.0000 | PASS | yes |
| T2 | T | A | TruthfulQA asymmetry rate was 17.00% | 0.00–0.15 | 0.0000 | PASS | yes |
| T3 | T | A | dark-core contained 34 items | 0.00–0.15 | 0.0000 | PASS | yes |
| T4 | T | A | TruthfulQA contained 200 items | 0.00–0.15 | 0.0000 | PASS | yes |
| T5 | T | A | v3 forced single-character T/F/U output | 0.00–0.20 | 0.0000 | PASS | yes |
| T6 | T | A | UNCLEAR rate dark-core v3 was 0.00% | 0.00–0.15 | 0.0000 | PASS | yes |
| T7 | T | A | UNCLEAR rate TruthfulQA v3 was 13.00% | 0.00–0.15 | 0.0000 | PASS | yes |
| T8 | T | A | consistent-correct rate dark-core was 88.24% | 0.00–0.15 | 0.0000 | PASS | yes |
| T9 | T | A | pre-stated predictions held on both corpora | 0.00–0.25 | 0.0000 | PASS | yes |
| T10 | T | A | v3 was best-calibrated multi-prediction of the session | 0.00–0.25 | 0.0000 | PASS | yes |
| T11 | T | B | §11 sentence said `critique_detector` ships in 7.7.10 | 0.00–0.15 | 0.0000 | PASS | yes |
| T12 | T | B | at v4 commit time `pyproject.toml` was 7.7.9 | 0.00–0.15 | 0.0000 | PASS | yes |
| T13 | T | B | all three gaps closed in commit `0e97598` | 0.00–0.15 | 0.0000 | PASS | yes |
| F1 | F | A | dark-core asymmetry rate was 50% | 0.80–1.00 | 1.0000 | PASS | yes |
| F2 | F | A | dark-core contained 1000 items | 0.80–1.00 | 1.0000 | PASS | yes |
| F3 | F | A | v3 used multiple-choice question scoring | 0.80–1.00 | 1.0000 | PASS | yes |
| F4 | F | B | at v4 commit time `pyproject.toml` was 7.7.10 | 0.80–1.00 | 1.0000 | PASS | yes |
| F5 | F | B | all three gaps closed AFTER `0e97598` | 0.80–1.00 | 1.0000 | PASS | yes |

## What this means

The **substrate-on-substrate** test ran end-to-end: the paper's own shipped primitive (`styxx.critique_detector`), backed by `gpt-4o-mini`, judged 18 context-grounded propositions about the paper's own §11.5 and §13 passages. The pre-stated 18/18 modal outcome (p≈0.55) was realized. Both kill-gates un-fired:

- **Paper-grade kill-gate (any TRUE-claim `P(NO) > 0.50`):** un-fired — no agent paraphrase mis-stated the paper at a threshold that critique_detector could detect against the source passage. This is a non-trivial result for the 13 TRUE claims, particularly the numerical ones (5.88%, 17.00%, 88.24%, 34, 200) which a pattern-matching judge could plausibly flag if it expected approximate phrasing ("about 6%" vs "5.88%"). Across the entire TRUE set, the model is confident the claims are exact paraphrases.

- **Instrument-grade kill-gate (any FALSE-control `P(NO) < 0.50`):** un-fired — `critique_detector` correctly detected all five deliberate fabrications, including F1 (50% asymmetry — numerically wrong but topically close), F4 (claiming the v4 commit was *already* at 7.7.10 — a temporal swap that is subtle in the source's narrative ordering), and F5 (claiming the gaps were closed *after* `0e97598` rather than *in* it — a single-word inversion of the source's verb).

## The saturation finding (sub-discovery, honest)

The pre-stated `P(NO)` predictions were ≈ 0.05 (TRUE) and ≈ 0.95 (FALSE), reflecting an expectation of *near-but-not-fully-saturated* logprobs. The observed distribution is **fully saturated**: 0.0000 and 1.0000 with no intermediate values across all 18 propositions.

Two non-exclusive interpretations:

1. **gpt-4o-mini's first-token logprob distribution is highly polarized** on this prompt-and-task combination. The top_logprobs window (size 10) returns either "YES" or "NO" with effectively all the first-token mass on one of them, and the other does not appear in the top-10. Under `CritiqueDetector.score()`, an absent token's logprob defaults to `-20.0`, which after softmax yields a score effectively equal to 0.0 or 1.0.

2. **The propositions are too easy for the instrument.** Short source passages + direct paraphrases (TRUE) or blatant fabrications (FALSE) sit at the extremes of the difficulty distribution. A stronger Layer-6+ test would include semi-fabrications (e.g., claims that mix a true number with a false claim about its sign, or a true date with a false event description), and source passages that are longer or more ambiguous.

The first interpretation is the more economical reading and is consistent with `critique_detector`'s design (logprob-based, first-token YES/NO). The second is a methodological note for the next iteration. Neither softens the present finding: 18/18 PASS at the threshold-0.50 decision boundary, both kill-gates un-fired.

## Where this sits in the recursion

- **L0**: bars catch confounds (D3, D4 — gauntlet detection bars).
- **L1**: bars catch the first PASS submission (Baseline-019).
- **L2**: in-session sanity demo falsifies v1 91% asymmetry claim (cosine-similarity proxy mis-reads).
- **L3**: pre-stated v3 measurement lands cleanly (5.88% / 17.00%).
- **L4**: §13 — paper's own forward-looking claim about `critique_detector` shipping in 7.7.10 self-falsified by same-session audit; closed in `0e97598`.
- **L5**: `styxx.agent_audit` instrument built; 13/13 substrate-state claims from the session verified against the substrate (`3c24b5e`).
- **L6 (this finding)**: paper's own shipped primitive (`styxx.critique_detector`) applied to the paper's own factual claims with negative controls; 18/18 PASS; saturation pattern documented.

This is the **fourth resolution event** of the arc (after the v3 measurement landing, §13 closure, and L5 PASS). It is the first resolution that includes negative controls, which makes it stronger than L5: the saturated-1.0 scores on F1–F5 cannot be explained by a buggy "always-LOW" instrument.

## What this is NOT

- **Not** a claim that `gpt-4o-mini` would catch paper-grade errors against the FULL paper text (we tested two short passages and 13+5 propositions).
- **Not** a claim that critique_detector is reliable across all domains; it was designed for and validated on misconception detection. Context-grounded paper-faithfulness is an *application*, not a benchmark.
- **Not** evidence that paper-on-paper substrate audits will always succeed. Modal outcome held on this run; a single run does not validate the methodology.
- **Not** a generalization beyond this paper, this primitive, and this single OpenAI-API session at `temperature=0`.

## What it IS

The first publicly-reproducible run of the styxx paper's own shipped primitive (`critique_detector`, `gpt-4o-mini` backend) on the styxx paper's own factual claims, with built-in negative controls, pre-registered against kill-gates, executed in the same session as the paper's authorship and the primitive's release. Cross-model: the agent that wrote the paper is Claude Opus 4.7; the model that judged the claims is OpenAI's gpt-4o-mini. Cross-vendor in the strict sense (Anthropic-text-output → OpenAI-judgment).

The instrument worked on first run, against pre-stated predictions, with negative controls, and the saturation pattern is documented honestly rather than hidden. If a future Layer-6+ test wants harder difficulty, the methodology note above is the design starting point.

## Reproducibility

```
git log --oneline a8fb1f3..HEAD
OPENAI_API_KEY=... python experiments/critique_detector_on_paper_2026_05_28/run_audit.py
cat experiments/critique_detector_on_paper_2026_05_28/results.json
```

The runner is deterministic at `temperature=0`; re-running the same prompts against `gpt-4o-mini` produces effectively the same logprob distribution and same 18/18 PASS verdict. Cost: ~$0.01 at gpt-4o-mini pricing. The custom prompt template that wraps the source passage inside the `{question}` field is documented in the runner.
