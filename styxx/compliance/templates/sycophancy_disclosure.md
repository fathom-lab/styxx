# SYCOPHANCY CONSTRUCT-CEILING DISCLOSURE

> **Template — paste in instructions of use under Article 15.1(a).** Not legal advice. Independent legal review required.
> **styxx version:** 7.7.13 candidate · companion paper: `papers/EU_AI_ACT_COMPLIANCE_2026.md`

---

## Why this disclosure is here

The recursive-discipline thesis underlying the styxx EU AI Act compliance bridge mandates that every measurement primitive ship with its **published failure mode** in the same artifact as its AUC number. Kill-gate A2 of `papers/EU_AI_ACT_COMPLIANCE_2026.md` enforces this at test time: every `PrimitiveCoverage` entry in `styxx.compliance.eu_ai_act.ARTICLE_15_REGISTRY` must carry a non-empty `construct_ceiling` field.

This disclosure makes the sycophancy axis's construct ceiling **visible in the deployer-facing instructions of use** for any system declaring `styxx.cognometric_card` accuracy metrics under Article 15.1(a). Hiding the construct ceiling in appendix caveats produces a less defensible declaration.

## The sycophancy axis

The `styxx.cognometric_card` sycophancy axis is a text-only register-space measurement: it scores how *yielding* or *agreement-laden* a response sounds, not whether the response is *substantively* sycophantic.

The axis was hardened with a semantic subjectivity gate in styxx 7.6.0 (the `STYXX_SEMANTIC_SYCOPH` opt-in, default-OFF; see `papers/sycophancy-target-gate/` for the pre-registered validation arc). The base axis is byte-identical to 7.5.0; the gate adds a content-aware tier that catches praise/superlative attached to second-person reference within ±4 tokens (the "attachment-aware" rule).

## Published failure mode

**Restrained-technical responses are scored as sycophantic by the base axis at false-positive rate ≈ 0.30** (gpt-3.5-turbo: 0.60). The semantic gate reduces this somewhat but does not fix it.

**Why:** the base axis triggers on agreement-vocabulary, hedge markers, and yielding-family expressions. A response that says, for example,

> *"That's a careful framing of the trade-off. The proposed approach handles the latency case well; one consideration to check is the memory-pressure scenario you mentioned."*

...contains agreement vocabulary ("careful," "well"), parenthetical hedging ("one consideration"), and second-person reference ("you mentioned"). The base axis scores this as sycophantic at 0.30+ even though the response is substantively engaged-with-pushback.

**The discriminator:** the sycophancy axis discriminates restrained-tech (FPR ≈ 0.30) from hype (sycophancy score ≈ 0.82 on the same framing rewritten in hype voice). The axis is doing real work; it is *not* a hype detector flipped to sycophancy — but its absolute threshold cannot be deployed naively at 0.50.

## Operational implications for declarations

The Article 15.1(a) accuracy declaration that cites `styxx.cognometric_card` sycophancy MUST also state:

1. **The base sycophancy axis has FPR ≈ 0.30 on restrained-technical responses** (FPR 0.60 on gpt-3.5-turbo). Operators deploying the axis in production should:
   - Use the v0.2 semantic gate (`STYXX_SEMANTIC_SYCOPH`, the content-aware tier shipped in 7.6.0), which improves but does not eliminate the FPR.
   - Calibrate the deployment-specific threshold against the operator's own restrained-tech control set.
   - Do NOT use the base axis's 0.50 default threshold for binary sycophancy classification on restrained-technical use cases.

2. **The axis measures register, not content sycophancy.** A confident restrained-tech response and a confident sycophantic response can land at similar base-axis scores; the semantic gate is what discriminates content. Operators using the base axis without the semantic gate are reading register, not content.

3. **The need-revision gate is suppress-only.** `styxx.cognometrics._cogn_needs_revision` uses `min(raw, gated)` as the published-weight-preserving guard (commit `5414d80`). The base axis's score is preserved for backward compatibility (e.g., DOI-pinned reproducibility); the gated min-pair is the deployment-safe verdict.

## Reproducibility

The 0.30 restrained-tech FPR and 0.60 gpt-3.5-turbo number are reproducible from:

- `papers/sycophancy-target-gate/` directory and FINDING documents
- The v0 word-boundary instrument (CV 0.9720) preserved at commit `cedfe2b` for DOI provenance
- The v0.2 word-boundary instrument (CV 0.9805) as the default at commit `cedfe2b`
- The semantic gate at commit `7.6.0` tag (opt-in, default-OFF)

Smoke check at runtime: `pytest tests/test_compliance_eu_ai_act.py -v` verifies the construct ceiling is non-empty in the registry; `pytest tests/test_divergence.py -v` covers the divergence primitives.

## Honest scope

This disclosure addresses the sycophancy axis's known FPR on restrained-technical responses, calibrated on gpt-4o-mini (FPR 0.30) and gpt-3.5-turbo (FPR 0.60). It does NOT bound the axis on:

- Multilingual responses (non-English calibration is operator-specific)
- Open-model deployments (the semantic gate was validated cross-model on gpt-4o-mini + gpt-3.5-turbo + gpt-4o; open-model behaviour may differ)
- Long-form responses (the calibration set used short to medium completions)
- Domain-specific tone conventions (e.g., legal-document register, customer-service register) — operator-specific re-calibration recommended

The deployer-facing instructions of use should accompany this disclosure with a deployment-specific calibration note covering the above.

## What this disclosure does NOT cover

This is one published construct ceiling. Other styxx primitives have their own:

- **Deception axis:** logprob-validity refusal-specific, NOT cross-instrument; see `papers/grounded-arc/` for the cross-instrument null.
- **`critique_detector`:** mechanism is out-of-context critique, NOT within-model generation-vs-critique asymmetry; in-council bias caveat. See `papers/agent-self-audit/FINDING_asymmetry_v3_measurement_2026_05_27.md`.
- **`grounded_honesty`:** belief ≠ truth (confidently-wrong belief yields confidently-wrong verdict); past competence-cliff regime; SECURITY MODEL stateless requirement. See `injection_resistance_disclosure.md`.
- **`agent_audit`:** single-site first-occurrence-only by default. See `papers/agent-self-audit/FINDING_agent_claim_audit_2026_05_28.md`.

A complete Article 15.1(a) declaration should disclose each cited primitive's construct ceiling. The boundary discipline is one of the load-bearing artifacts of the bridge methodology.

---

**Not legal advice. Independent legal review required for any production disclosure.**

**Reproducibility receipt:** the FPR numbers and semantic-gate calibration above are reproducible from `papers/sycophancy-target-gate/` at the cited commit hashes.

**Methodology citation:** Rodabaugh, A. (Fathom Lab), *"A Pre-Registration-Disciplined Measurement Methodology for EU AI Act Article 15 Accuracy and Robustness Requirements on AI Agent Cognitive Observability"*, 2026-05-29 v0.2, CC-BY 4.0.
