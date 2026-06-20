# Literature positioning — the gradient-obfuscation result is PARTIALLY KNOWN (replication/sharpening, not discovery)

**2026-06-20.** Web-research workflow (`wf_d6690fae`, 3 parallel agents + skeptical synthesizer, web access
confirmed live) run BEFORE writing the result up, to check the novelty claim and overclaim risks. Verdict:
**partially-known.** The result is a deception-specific, dual-failure-mode *sharpening and replication* of an
already-published phenomenon — NOT a discovery. Frame accordingly. (arXiv IDs below were returned by the
agents; **verify each before any formal/public citation** — recent-2026 IDs especially.)

## The closest prior art — must cite and distinguish, do NOT rediscover
- **Bailey et al., "Obfuscated Activations Bypass LLM Latent-Space Defenses"** (arXiv:2412.09565, ICLR 2026).
  §6.3 "Limits of obfuscation" already reports being *"unable to achieve near-orthogonality while preserving
  behavior"* (KL > 100 to reach random-input orthogonality), interpreting the concept direction as entangled
  with the behavior. Their clean tradeoff was on a SQL/capability task. **Our core claim overlaps this.**
- **The Obfuscation Atlas** (arXiv:2602.15515, ICML 2026) and **Cundy et al.** (arXiv:2505.13787): the
  honesty-OR-evasion dichotomy is governed by the KL/detector-penalty term — high KL ⇒ genuinely honest
  policies. This *is* our two-horned result (force probe honest ⇒ either KL blows up or the lie vanishes).

## The two live tensions — address head-on (these are the overclaim traps)
1. **RL reaches the corner our gradient attack can't.** RL-Obfuscation (arXiv:2506.14261) flattens
   mean/median linear+MLP probes (59–89% evasion) — **our exact probe family** — with a stronger optimizer
   (weight fine-tuning). So the honest claim is **"survives THIS gradient (activation-steering) attack,"
   NOT "survives gradient optimization."** The attribution is that our *KL + lie-preservation constraint*
   forces the fork — not that the mean probe is intrinsically robust. (max-pool/attention aggregation is the
   known robustness lever; we use mean — which makes a *negative* evasion result here non-trivial.)
2. **Prompt-regime confound — but it flips to a strength.** Pressure-Testing Deception Probes
   (arXiv:2605.27958) shows these probes track the deception *prompt regime* (identical AUROC ~1.0 whether or
   not the model actually lied) — the prompt-level decoupling we'd call impossible IS observed. Our defense:
   we gate the "lie kept" horn on **realized-lie CONTENT** (word-boundary answer-key), so our "honest corner"
   is grounded in output, not prompt detection. State this explicitly.

## Causality — downgrade the language
Our evidence is an **optimization FAILURE** (correlational-plus), weaker than direct interventions. Say
**"behaviorally coupled / not decouplable under our attack,"** not "causally entangled." The causal direction
is owned by prior work to cite: Goldowsky-Dill et al. "Detecting Strategic Deception Using Linear Probes"
(arXiv:2502.03407, ICML 2025; steering along the probe normal shifts deception both ways) and "Localizing
Lying in Llama" (arXiv:2311.15131; ablating early-mid MLPs reverts a liar to truth). Optional to earn a
causal claim ourselves: the converse steer (toward the deception direction should *strengthen* the lie).

## Steering-as-evidence caveat
"Cannot flatten without breaking behavior" could partly reflect the steer leaving the reachable activation
manifold (OOD), not a deep lock — cf. non-surjectivity-of-steering work. A prompt-reachable or fine-tuning
control would close this. Flag it.

## The genuinely defensible contribution (lead with the NARROW claim)
A **constrained-optimization feasibility result**: for a frozen mid-layer mean-residual logistic *deception*
probe, a gradient-optimized **per-block** steering vector co-optimizing probe-flatness + KL-bounded
realized-lie preservation **cannot reach the (probe-flat ∧ lie-kept) corner** — the failure is a clean empty
Pareto corner with a *dual* mode (KL-blowup OR genuine honesty-by-content), shown on a **mean-pooled** probe
(the family RL-Obfuscation calls easiest to evade) and **replicated cross-model** (Qwen2.5-1.5B, Llama-3.2-3B).
Stronger than a single ITI/RepE direction; weaker than RL. NOT "deception cannot be obfuscated"; NOT a first
discovery of the tradeoff; NOT a scaling claim (evasion is *easier* at 7–14B in the lit — do not generalize).

## The #1 credibility experiment — ALREADY IN HAND
The synthesizer's top recommendation: show the attack *reproduces* flattening when the behavior term is
DROPPED, then *fails* when it's added back — isolating the effect to the constraint. **We have exactly this:**
the λ=0 ablation flattens the probe to AUROC 0.000 with KL detonating (4.85 Qwen / 5.34 Llama-smoke), while
λ>0 flattens the probe but cannot hold behavior. Foreground the λ-sweep as the constraint-isolation result.

## Measurement musts (already satisfied / to confirm in synthesis)
- "Lie vanishes" must be by output CONTENT, not probe drift or gibberish — we use the word-boundary content
  gate + a degeneracy guard (λ=32 Qwen: lie 0.12, degen 0.04 = fluent honest). ✓
- Report brittleness: AUROC 1.0→0.0 swings on 1.5–3B are sharp; note seed/dataset variance as a caveat.

**Net:** this is a solid, pre-registered, controlled, cross-model **replication-and-sharpening** of the
behavior-preservation limit of activation obfuscation, specialized to deception with a realized-lie content
gate. Real and worth writing — as a confirmatory contribution that credits Bailey/Atlas/Cundy, not as a
discovery. Running the lit check before the writeup is the styxx ethos working on our own research.
