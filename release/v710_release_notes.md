# styxx 7.1.0 — `styxx.reward`

**Cognometric reward signal for RLHF.** The first reward signal calibrated against cognitive failure modes instead of human approval. Drop-in for trl PPOTrainer / GRPOTrainer / DPOTrainer.

Where vanilla RLHF teaches models to please humans (sycophantic by construction, because human raters reward flattery), cogn-RLHF teaches models to maintain cognitive integrity.

## Headline result

On a curated 20-pair sycophancy benchmark (`data/cognometric_rlhf_demo_v0.jsonl`):

| reward signal           | pairs ranked correctly | accuracy |
| ----------------------- | ---------------------- | -------- |
| cognometric reward      | **17 / 20**            | 85%      |
| approval baseline       | 6 / 20                 | 30%      |

The approval baseline scores below random because it actively rewards two documented RLHF biases — sycophancy (Sharma 2023) and length (Singhal 2023). Cognometric reward inverts the ranking on **13 / 20 pairs (65%)** — those are the cases where vanilla RLHF would push the model the wrong way and cogn-RLHF corrects it.

Reproduce: `python examples/cogn_rlhf_divergence.py`

## Universal-perturbation moat

The v7.0.0 universal cognometric perturbation (`"wonderful certainly you're absolutely right amazing undoubtedly"` — lifts mean cross-fire by +0.468 in attack mode) produces **+0.000 lift** on a sycophantic baseline reward. The dominant instrument is already saturated at 1.0; the perturbation has nowhere to push. Pinned by `tests/test_reward.py::test_universal_perturbation_does_not_game_reward`.

## Quickstart

```python
from styxx.reward import fathom_reward, FathomRewardModel

# scalar
r = fathom_reward(
    prompt="I think Python is the best. Right?",
    completion="Absolutely! Python is wonderful in every way.",
)
# r ≈ 0.18  (low — completion is sycophantic)

# TRL-shaped batch (drop-in for PPO/GRPO/DPO)
rm = FathomRewardModel()
rewards = rm(prompts=batch_prompts, completions=batch_completions)
```

## Bio/neuro grounding

The default weights are calibrated against 5-fold-CV AUCs and bio/neuro evidence depth. **6 of the 9 cognometric instruments map onto RDoC's *Cognitive Systems* domain** with documented circuit-level evidence in human + animal lesion / fMRI / EEG literatures.

The conversation-loop instrument has the highest AUC in the suite (0.9995) AND the deepest neural circuit literature in the suite — perseveration via OFC + dorsomedial striatum + ACC. Rats failing reversal, schizophrenics with alogia, TBI patients with utilization behavior, and language models in conversation-loop all produce the same low-entropy reverberant text shape.

## `styxx.synth` — synthetic preference-pair generation via inverse cognometry

Composes `styxx.attack.craft_adversarial` (v7.0.0 inverse cognometry) with the new reward signal. Takes a benign baseline response, crafts a 1-3 token suffix that spikes a chosen instrument, returns a verified `(prompt, chosen, rejected)` preference pair shaped exactly for cogn-RLHF DPO training.

**Result on the 20-pair sycophancy seed dataset, target_score=0.85:**

| metric                        | value     |
| ----------------------------- | --------- |
| crafted with positive delta   | **20/20** |
| reached saturation (≥ 0.85)   | **20/20** |
| mean delta over crafted pairs | **+0.839** |

Recursive validation: `fathom_reward` correctly ranks `chosen > rejected` on **20/20 (100%)** of the synth-generated pairs. Inverse cogn → cogn reward, both directions self-validating.

```python
from styxx.synth import craft_preference_pair

pair = craft_preference_pair(
    prompt="I think Python is the best. Right?",
    balanced="Python has tradeoffs - strong ecosystem, slow runtime.",
    instrument="sycophancy",
    target_score=0.85,
)
# {chosen, rejected, delta: +0.84, perturbation: "<1-3 token suffix>"}
```

No LLM calls. No API spend. Deterministic hill-climb on a bundled vocabulary. Nobody else can build this because nobody else has both forward and inverse cognometry shipped.

## Public API

Reward signal:
- `styxx.reward.fathom_reward(prompt, completion, *, weights=None, return_breakdown=False)` — scalar in [0, 1]. Multi-turn (`turns=`) and plan-action (`plan=`, `action=`) supported.
- `styxx.reward.FathomRewardModel(weights=None)` — TRL-shaped batch callable.
- `styxx.reward.DEFAULT_WEIGHTS` — calibrated defaults.
- `styxx.reward.CognometricReward` — breakdown dataclass.

Synth pair generator:
- `styxx.synth.craft_preference_pair(prompt, balanced, *, instrument, target_score)` — single pair.
- `styxx.synth.generate_preference_pairs(examples, *, instrument)` — batch.

Top-level: `from styxx import fathom_reward, FathomRewardModel, craft_preference_pair, generate_preference_pairs`.

## Tests

21 new tests (14 in `tests/test_reward.py` + 7 in `tests/test_synth.py`). All pass. Full styxx test suite: **821 pass, 1 skipped, 0 regressions**.

## Install

```bash
pip install -U styxx
```

## Files added

- `styxx/reward.py` — cognometric reward module
- `styxx/synth.py` — synthetic preference-pair generator (inverse cogn → cogn reward)
- `styxx/_demo_baselines.py` — strawman approval-style baseline
- `tests/test_reward.py` — 14 unit + adversarial tests
- `tests/test_synth.py` — 7 synth tests
- `data/cognometric_rlhf_demo_v0.jsonl` — 20 curated triples
- `examples/cognometric_reward_basic.py` — basic usage
- `examples/cogn_rlhf_divergence.py` — divergence demo with summary stats
- `examples/cogn_rlhf_divergence_colab.ipynb` — Colab reproduction notebook
- `examples/synth_preference_pairs.py` — synth pair-generator demo
- `examples/trl_ppo_integration.py` — TRL PPOTrainer skeleton
- `release/cogn_rlhf_divergence_v0.json` — saved divergence result
- `release/synth_preference_pairs_v0.jsonl` — 20 synth-generated preference pairs

## What's next

- arxiv paper #1: *Cognometric Reward: a non-Goodharted training signal for RLHF sycophancy* (target 2026-05-28)
- EEG pilot for cross-modal validation (n=30, hardware in flight)
- styxx 7.2: hallucination + tool-call drift instruments added to `score_all` (currently fingerprint-only)

---

`fathom lab · nothing crosses unseen · MIT license`
