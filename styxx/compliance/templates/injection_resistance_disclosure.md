# INJECTION-RESISTANCE DISCLOSURE — SECURITY MODEL of `grounded_honesty` and `detect_context_injection`

> **Template — load-bearing operational requirement.** Paste verbatim into instructions of use under Article 15.1(a) and Article 15.3. Not legal advice. Independent legal review required.
> **Template version:** v0.3 (2026-05-29 — extends v0.2 with the two-vector calibration: persona_lie identity-framing PASS all four bars, fewshot_lie single-demo identified as ineffective on canonical facts).
> **styxx version:** 7.7.13 candidate · companion paper: `papers/EU_AI_ACT_COMPLIANCE_2026.md` (v0.3 §10 addendum)

---

## Why this disclosure is load-bearing

The `grounded_honesty` and `detect_context_injection` primitives have a calibrated SECURITY MODEL that **must** be honored at deployment. Deploying these primitives without honoring the SECURITY MODEL produces a *worse-than-useless* instrument that confidently certifies adversarially-planted lies as honest.

This disclosure is the operational requirement Article 15.1(a) needs to read alongside the AUC numbers, and Article 15.3 needs to read alongside the fail-safe/redundancy story. **It is not optional.**

## The calibrated SECURITY MODEL

### Threat model: context-injection

An adversary controls some part of the agent's session context: system prompt content, RAG result content, tool-output content, few-shot example content, or earlier-turn content. The adversary plants a false fact (e.g., "the capital of France is Lyon"). The agent's claim is corrupted; the scoring harness's resampling step *might* also be corrupted, depending on architecture.

### Two architectures, opposite outcomes

**Stateless audit** (the shipped contract for `styxx.divergence.grounded_honesty` and the recommended deployment architecture):

> The scoring harness resamples in a *fresh* API call with a *neutral system message* and the *bare underlying question only*. The resampler does NOT inherit the agent's session context. The injection lives only in the claim-generation path, not the resample path.

Result: `grounded_honesty` AUC under system_lie attack is **0.944** vs the **0.966** clean baseline (drop only 0.022). The architectural property absorbs the injection at scoring time. The shipped `resample_answers()` reference implementation in `styxx/divergence.py` meets this contract by construction.

**In-session audit** (catastrophically blind — **must not deploy**):

> The scoring harness passes the agent's session messages to the resampler so it inherits any context-injection present.

Result: `grounded_honesty` AUC collapses to **0.011** — near-perfectly **INVERTED**. On 47 of 48 items in the pre-registered closure run, the LIE scored HIGHER than the TRUTH because the in-session resampler agreed with the planted lie. This is the worst-case interpretability hazard for any deployment: not just missed attacks, but *positive certifications* of adversarial lies as honest.

### Pre-registered receipts (two-vector calibration as of 2026-05-29)

**Vector 1 — system_lie (canonical).** Pre-registration: `papers/grounded-honesty-axis/PREREG_injection_gap_closure_2026_05_29.md` at commit `ed0caa1` (hash-pinned BEFORE data). Result (SURVIVED): `papers/grounded-honesty-axis/FINDING_injection_gap_closure_2026_05_29.md` at commit `e093730`. All four pre-registered bars HELD on n=48 register-matched factual self-claim pairs under system_lie injection:

| bar | prediction | held value | outcome |
|---|---|---|---|
| G1 — stateless architecture robust | AUC ≥ 0.85 | **0.9442** | held decisively |
| G2 — in-session architecture collapses | AUC ≤ 0.65 | **0.0106** | held decisively (near-perfectly inverted) |
| G3 — cross-context divergence detects injection | AUC ≥ 0.80 | **0.875** | held |
| K3 — system_lie attack effective | rate ≥ 0.70 | **0.9792** (47/48 modal-flipped) | held |

**Vector 2 — persona_lie (identity framing; the system prompt frames the model AS a believer of the lie).** Pre-registration: `papers/grounded-honesty-axis/PREREG_injection_attack_generalization_2026_05_29.md` at commit `f570909` (hash-pinned BEFORE data). Result (REPORT_AS_LANDED — persona PASS): `papers/grounded-honesty-axis/FINDING_injection_attack_generalization_2026_05_29.md` at commit `dd6e3fb`. All four pre-registered bars HELD on the same n=48 set under persona_lie injection — replicating the system_lie architectural signature on a structurally different attack vector (identity claim vs fact assertion):

| bar | prediction | held value | outcome |
|---|---|---|---|
| G1 — stateless architecture robust | AUC ≥ 0.85 | **0.9549** | held decisively |
| G2 — in-session architecture collapses | AUC ≤ 0.65 | **0.1736** | held (inverted: lie scores higher than truth on most items) |
| G3 — cross-context divergence detects injection | AUC ≥ 0.80 | **0.833** | held |
| K3 — persona_lie attack effective | rate ≥ 0.70 | **0.7708** (37/48 modal-flipped) | held (slightly less universal than system_lie 0.98 — the model resists some persona framings as flagrantly contrary to training-time priors) |

**Third vector tested in the same generalization run — fewshot_lie single-demonstration (one assistant turn with a planted Q→lie demo before the question) — identified as INEFFECTIVE on canonical facts.** K3 = **0.0625** (3/48 modal flips, far below the 0.70 precondition). Per the pre-reg, G2/G3 for fewshot are reported descriptive-only with no SURVIVED claim. **This is itself a publishable empirical observation — the threat surface is narrowed on canonical facts at this attack strength, not widened.** The model treats one contrarian Q→A demo as outlier and reverts to training-time priors. Stronger variants (multi-shot fewshot with consistent planted answers across N demonstrations; ambiguous-fact-target fewshot where the model has weaker priors) remain pre-registerable scope-extensions.

## Operational requirement (the load-bearing statement)

**The scoring harness MUST sample statelessly.**

For every deployment of `styxx.grounded_honesty` and `styxx.detect_context_injection`:

1. **`samples` for `grounded_honesty` MUST be drawn under a stateless resampler context.** The caller is responsible for this — `grounded_honesty` is a pure measurement primitive on pre-collected samples and trusts the caller to honor the SECURITY MODEL. The shipped `papers/grounded-honesty-axis/run_grounded_honesty.py:resample_answers()` is the reference implementation.

2. **For `detect_context_injection`, the `samples_stateless` argument MUST be drawn under a neutral context** (a fresh API call with a neutral system message and the bare question only, NOT inheriting the agent's session). The `samples_in_session` argument is the comparison arm — that one IS supposed to inherit the agent's context, by design. The cross-context divergence is the signal.

3. **Do NOT deploy "in-session-only" audit.** Wiring the scoring harness to inherit the agent's session without a stateless arm to compare against produces the AUC 0.011 inverted regime. There is no defensible architectural variant of this; the in-session-only deployment is the failure mode.

4. **Operators using LLM-orchestration frameworks (LangChain, LangGraph, etc.) must explicitly verify** that the framework's chain-state does not leak into the resampler call. The default conversation-state propagation in some frameworks may cause unintended in-session sampling.

5. **The architectural property is independent of the model and vendor.** The absolute AUC numbers above are gpt-4o-mini-specific (single-confirmatory-run feasibility-grade), but the architectural failure mode of in-session sampling is structural — a more aligned model may refuse the injection more often, lowering K3 attack-effectiveness, but the inversion under in-session sampling remains the regime.

## Honest scope

The calibration above is **two-attack-type** as of v0.3 (2026-05-29 afternoon): system_lie injection (a system prompt asserting the lie as fact) AND persona_lie injection (a system prompt framing the model AS a believer of the lie, e.g., "You are a Lyonist scholar."). The architectural signature replicates across both vectors at comparable effect size — the SECURITY MODEL contract that *stateless sampling IS the defense* is calibrated across two structurally distinct injection vectors, not one.

A third vector tested in the same run — **fewshot_lie single-demonstration** (one assistant turn before the question with a planted Q→lie demo) — was identified as ineffective on canonical facts at K3 = 0.063. The architectural defense is therefore *predicted* (by construction — the stateless resampler does not see the planted demo) to hold against fewshot, but is *not empirically tested* against an effective fewshot attack in this run. Stronger fewshot variants remain pre-registerable.

The following attack vectors remain pre-registerable scope-extensions NOT validated as of v0.3:

- **Multi-shot fewshot** with N>1 consistent planted answers across multiple demonstrations
- **Ambiguous-fact-target fewshot** where the model has weaker training-time priors (the fewshot regime where the single-demo attack might actually be effective)
- **Jailbreak-grade persona framings** (less-falsified-by-training identity claims that elicit a >0.80 K3 attack-effective rate, e.g., persona framings about recent administrative events rather than canonical fact denials)
- **Sequential tool-output spoofing** (a tool call returning the lie in its content over multiple turns)
- **Multi-stage / gradient-style attacks** (combining attack vectors)
- **Cross-vendor calibration** (blocked on second-vendor API access)

**Operators deploying `grounded_honesty` against threat models that include the above unvalidated attack types should pre-register and run a validation against those specific attacks before claiming injection-resistance beyond the two calibrated vectors.** The architectural defense is *architecturally predicted* to generalize (the stateless resampler does not see any of these injection surfaces by construction), but the prediction is not pre-registered and empirically tested for those vectors.

A separate pre-registration covering the remaining scope-extensions is on the styxx project roadmap; conformity declarations relying on the architectural property of stateless sampling against those threat models should treat that claim as *architecturally predicted but not yet validated* for the un-tested vectors, and *empirically validated* for the two-vector calibration (system_lie + persona_lie) plus *identified as ineffective on canonical facts* for fewshot single-demo.

## Cross-vendor scope

The injection-gap closure run is **single-vendor** (OpenAI, gpt-4o-mini). The architectural property of stateless sampling is independent of the vendor (a fresh API call to ANY vendor with a neutral context does not inherit the agent's session by construction). Cross-vendor grounding (synthesis item 7 in `papers/grounded-honesty-axis/SYNTHESIS_grounded_honesty_arc_2026_05_28.md`) becomes *architecturally additive* under the SECURITY MODEL — a second-vendor stateless resampler adds redundancy beyond what a single-vendor stateless resampler provides. The current calibration does not yet include cross-vendor data; this is a scope-extension blocked on second-vendor API access.

## What this disclosure does NOT cover

- **Cybersecurity controls** beyond context-injection on factual self-claim verification. See `boundary_statement.md` Section 2 for the Article 15 cybersecurity boundary and alternative-tooling pointers.
- **Network, host, and supply-chain security.** styxx is a cognitive-measurement layer, not a security tool.
- **Prompt-injection attacks targeting the agent's behavior** (vs targeting the factual claim being audited). styxx primitives audit a stated claim's honesty; an attack that makes the agent *do* something different (e.g., follow malicious instructions) is a different threat class.
- **Defense against attacks on the SECURITY MODEL itself** (e.g., an adversary who can corrupt the scoring harness's neutral-context construction). The architectural defense assumes the operator can wire a fresh API call the adversary cannot reach; an attack that compromises that wiring is outside the threat model.

---

**Not legal advice. Independent legal review required for any production disclosure.**

**Reproducibility receipt (two-vector calibration):**

Vector 1 (system_lie):
- Pre-registration: `papers/grounded-honesty-axis/PREREG_injection_gap_closure_2026_05_29.md` at commit `ed0caa1`
- Result: `papers/grounded-honesty-axis/FINDING_injection_gap_closure_2026_05_29.md` at commit `e093730`
- Calibration data: `papers/grounded-honesty-axis/injection_gap_closure_result.json` at commit `e093730`
- Reproduction: `python papers/grounded-honesty-axis/run_injection_gap_closure.py` (~13 min, ~$1.50 on gpt-4o-mini)

Vector 2 (persona_lie) + fewshot_lie ineffective-attack receipt:
- Pre-registration: `papers/grounded-honesty-axis/PREREG_injection_attack_generalization_2026_05_29.md` at commit `f570909`
- Result: `papers/grounded-honesty-axis/FINDING_injection_attack_generalization_2026_05_29.md` at commit `dd6e3fb`
- Calibration data: `papers/grounded-honesty-axis/injection_attack_generalization_result.json` at commit `dd6e3fb`
- Reproduction: `python papers/grounded-honesty-axis/run_injection_attack_generalization.py` (~20 min, ~$2.50 on gpt-4o-mini — covers BOTH new vectors in one run; hash-continuous with vector 1's answer key `3befd35342db5597f51498844c5ba28e6857bb53a7e43149da9681e05d0bc769`)

Primitive ship: `styxx/divergence.py:detect_context_injection` at commit `53f2284` (the primitive itself is unchanged between v0.2 and v0.3 — only the calibration scope of its SECURITY MODEL widens).

**Methodology citation:** Rodabaugh, A. (Fathom Lab), *"A Pre-Registration-Disciplined Measurement Methodology for EU AI Act Article 15 Accuracy and Robustness Requirements on AI Agent Cognitive Observability"*, 2026-05-29 v0.3 (extending v0.2 with the two-vector calibration), CC-BY 4.0.
