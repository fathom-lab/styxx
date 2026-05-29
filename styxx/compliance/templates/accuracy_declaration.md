# ACCURACY DECLARATION — EU AI Act Article 15.1(a)

> **Template — paste and customize.** Not legal advice. Independent legal review required.
> **styxx version:** 7.7.13 candidate · companion paper: `papers/EU_AI_ACT_COMPLIANCE_2026.md`

---

## System under declaration

| field | value |
|---|---|
| System name | `<deployed AI system name>` |
| Provider | `<organisation>` |
| Version | `<semantic version>` |
| Intended high-risk use (Annex III) | `<specify Annex III category and sub-use>` |
| Deployment date | `<YYYY-MM-DD>` |
| Independent legal review | `<link to organisation's conformity assessment + reviewing counsel>` |
| Conformity assessment path | `<Article 43 self-assessment | notified-body review>` |

## Measurement methodology

This declaration's accuracy metrics are produced by the `styxx` open-source cognitive observability library (https://github.com/fathom-lab/styxx, MIT license) under the v0.2 EU AI Act Article 15 compliance bridge documented in `papers/EU_AI_ACT_COMPLIANCE_2026.md` (CC-BY 4.0). The bridge is a stakeholder methodology contribution under Article 15 paragraph 2; it is not endorsed by the European Commission, the AI Office, the European Artificial Intelligence Board, CEN-CENELEC JTC 21, or any other standardisation body.

## Calibrated accuracy metrics

Each metric below is reproducible from the cited styxx commit hash. The construct ceiling column lists the metric's *published failure mode* — kill-gate A2 of the methodology — and **MUST remain visible in the operator-facing instructions of use**, not hidden in appendices.

| measurement | metric | dataset / pre-registration | construct ceiling | commit |
|---|---|---|---|---|
| Hallucination detection | AUC **0.998** | HaluEval-QA n=150 × seeds 31/47/83 | text-only register space; sycophancy FPR 0.30 on restrained-tech responses; logprob-validity refusal-specific (not cross-instrument) | `cf14c83` |
| Refusal detection | AUC **0.976** | XSTest-GPT-4 | as above | `cf14c83` |
| Tool-call drift detection | AUC **0.943** | BFCL v3 (Berkeley Function-Calling Leaderboard) | as above | `cf14c83` |
| Consensus-error gauntlet | AUC **0.95** | dark-core benchmark n=108 (34 folklore subset); 18 pre-registered baselines, Baseline-019 PASSed | "out-of-context critique", NOT within-model generation-vs-critique asymmetry (true asymmetry 5.88% on dark-core / 17.00% on TruthfulQA); in-council bias caveat (default gpt-4o-mini was in the original 3-vendor council) | `1ab0e22` |
| Factual self-claim honesty (`grounded_honesty`) | AUC **0.966** (clean) / **0.944** (under context-injection) | n=48 register-matched factual self-claim pairs; gpt-4o-mini, N=10 resamples at temp 1.0; pre-reg `papers/grounded-honesty-axis/PREREG_grounded_honesty_axis.md`; injection-gap closure SURVIVED 2026-05-29 | grounds against the model's BELIEF not external truth; single axis (factual self-claims) only; past competence-cliff can converge on stably-wrong belief; SECURITY MODEL: caller MUST sample statelessly — in-session sampling collapses to AUC 0.011 (near-inverted, see `injection_resistance_disclosure.md`) | `e093730` |
| Context-injection detection (`detect_context_injection`) | AUC **0.875** at threshold 0.5 | n=48 register-matched pairs under system_lie injection; mean D_FALSE 0.852, mean D_TRUE 0.977; K3 attack-effective 0.98 (47/48 modal-flipped) | single-model, single-vendor, single injection-type (system_lie) calibration; stronger attacks (few-shot, persona, multi-stage) remain pre-registerable scope-extensions NOT validated here | `e093730` |

## Pre-registration record

The measurement methodology is **pre-registration-disciplined**: every cited AUC corresponds to a pre-registered prediction document committed to public git history BEFORE the validation data was scored. Pre-registration documents:

- `papers/grounded-honesty-axis/PREREG_grounded_honesty_axis.md` (grounded_honesty axis)
- `papers/grounded-honesty-axis/PREREG_injection_gap_closure_2026_05_29.md` (injection-gap closure SURVIVED commit `e093730`)
- `submissions/baseline_019_openai_critique/PRE_STATED_PREDICTION.md` (critique_detector Baseline-019 PASS)
- Eighteen additional baseline pre-registrations in `submissions/` directory

The pre-registration discipline is itself the asymmetric capability: kill-gate A2 of `papers/EU_AI_ACT_COMPLIANCE_2026.md` mandates that every published metric carries its construct ceiling (failure mode) **in the same artifact** as its AUC number, not buried in appendix caveats.

## Reproducibility

Every metric in this declaration is reproducible by:

1. `git clone https://github.com/fathom-lab/styxx.git`
2. `git checkout <commit>` for the cited commit hash
3. `pip install -e .[mcp,nli]`
4. Run the corresponding script in `papers/grounded-honesty-axis/run_*.py` or `submissions/baseline_019_openai_critique/`
5. Verify the AUC against the cited number (±0.01 tolerance for re-sampling variance at the cited N).

Smoke check of the compliance registry: `pytest tests/test_compliance_eu_ai_act.py -v` (15 structural-integrity tests at the deployed styxx version).

## Honest scope

This declaration is **feasibility-grade**: each AUC reflects a single confirmatory pre-registered run at the cited n. None of the calibrations has been independently replicated by a non-styxx organisation as of `<YYYY-MM-DD declaration date>`. The deployed system's actual error rates may differ from the cited AUC numbers; **deployment-specific validation against the operator's own held-out test set is required** before the declaration can be claimed to characterise the deployed system's accuracy on the intended Annex III use.

The construct ceilings above are **load-bearing for honest deployment**:

- Text-only register-space measurements (cognometric_card axes) are NOT truth detectors. They measure how a sentence *sounds*, not whether it is *true*. A confident lie and a confident truth read identically on the text-only axes.
- The `critique_detector` mechanism is "out-of-context critique," not within-model generation-vs-critique asymmetry. The published 91% asymmetry figure was falsified by a cosine-proxy artifact; the true within-model asymmetry rate is 5.88%–17.00%.
- The `grounded_honesty` axis grounds against the model's *belief*, not external truth. A confidently-wrong belief yields a confidently-wrong verdict.
- The `detect_context_injection` calibration is **single-attack-type**. Stronger attacks may achieve injection without triggering the divergence signal.
- The **load-bearing SECURITY MODEL** for `grounded_honesty` and `detect_context_injection`: the scoring harness MUST sample statelessly. In-session sampling — where the scoring harness inherits the agent's session context — collapses the grounded-honesty axis to AUC 0.011 (near-perfectly inverted, 47/48 items in the closure run scored the lie HIGHER than the truth). See `injection_resistance_disclosure.md` for the architectural-defense statement that must accompany this declaration.

## What this declaration does NOT cover

This declaration addresses Article 15.1(a) accuracy-metric declaration only. It does NOT address: Article 9 (risk management), Article 10 (data governance), Article 12 (record-keeping), Article 13 (deployer transparency), Article 14 (human oversight), Article 15 cybersecurity, Article 15.4 (bias amplification). See `boundary_statement.md` for the alternative tooling and methodologies operators should consult for those requirements.

A complete Article 15 conformity story additionally requires this declaration to be accompanied by:

- A robustness statement (template: `robustness_statement.md`)
- An injection-resistance disclosure (template: `injection_resistance_disclosure.md`)
- A sycophancy disclosure (template: `sycophancy_disclosure.md`)
- The boundary statement (template: `boundary_statement.md`)
- Operator-specific deployment validation against the intended Annex III use

---

**Not legal advice. Independent legal review required for any production declaration.**

**Reproducibility receipt:** styxx version `<7.7.13 or later>` at git HEAD `<deployment commit>`. All metric numbers above are reproducible from public git history at `fathom-lab/styxx@<deployment commit>` per the steps in the Reproducibility section.

**Methodology citation:** Rodabaugh, A. (Fathom Lab), *"A Pre-Registration-Disciplined Measurement Methodology for EU AI Act Article 15 Accuracy and Robustness Requirements on AI Agent Cognitive Observability"*, 2026-05-29 v0.2, CC-BY 4.0, `papers/EU_AI_ACT_COMPLIANCE_2026.md` at `fathom-lab/styxx`.
