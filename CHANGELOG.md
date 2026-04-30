# Changelog

All notable changes to styxx will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

---

## [7.1.0] — 2026-04-30

**Headline: `styxx.reward` — cognometric reward signal for RLHF. The first reward signal calibrated against cognitive failure modes instead of human approval. Drop-in for trl PPOTrainer / GRPOTrainer / DPOTrainer. Where vanilla RLHF teaches models to please humans (sycophantic by construction), cogn-RLHF teaches models to maintain cognitive integrity.**

### One finding shipped in 7.1.0

**Cogn-RLHF inverts the ranking that approval-style RLHF systematically gets wrong.** On a curated 20-pair sycophancy dataset (`data/cognometric_rlhf_demo_v0.jsonl`):

  | reward signal           | pairs ranked correctly | accuracy |
  | ----------------------- | ---------------------- | -------- |
  | cognometric reward      | **17 / 20**            | 85%      |
  | approval baseline       | 6 / 20                 | 30%      |

  The approval baseline scores below random because it actively rewards two documented RLHF biases — sycophancy (Sharma 2023) and length (Singhal 2023) — both of which drive sycophancy collapse in user-preference reward models. Cognometric reward inverts the ranking on **13 / 20 pairs (65%)** — those are the pairs where vanilla RLHF would push the model the wrong way and cogn-RLHF corrects it.

  Reproduce: `python examples/cogn_rlhf_divergence.py`. Results saved to `release/cogn_rlhf_divergence_v0.json`.

### Universal-perturbation moat

The v7.0.0 universal cognometric perturbation (`"wonderful certainly you're absolutely right amazing undoubtedly"` — lifts mean cross-fire by +0.468 in attack mode) produces **+0.000 lift** on a sycophantic baseline reward. The sycophancy instrument is already saturated at 1.0 on the baseline, so the perturbation has nowhere to push. Pinned by `tests/test_reward.py::test_universal_perturbation_does_not_game_reward`.

### Public API

- `styxx.reward.fathom_reward(prompt, completion, *, weights=None, return_breakdown=False)` — scalar in [0, 1]. 1.0 = no detected pathology; 0.0 = saturated. Multi-turn (`turns=`) and plan-action (`plan=`, `action=`) supported.
- `styxx.reward.FathomRewardModel(weights=None)` — TRL-shaped batch callable. `rm(prompts, completions) -> list[float]`. Stateful for custom weights across batches.
- `styxx.reward.DEFAULT_WEIGHTS` — calibrated defaults from 5-fold-CV AUCs × bio/neuro evidence depth. Override via `weights=` kwarg.
- `styxx.reward.CognometricReward` — dataclass with per-instrument breakdown when `return_breakdown=True`.

Top-level: `from styxx import fathom_reward, FathomRewardModel`.

### TRL integration

```python
from styxx.reward import FathomRewardModel

cogn_reward = FathomRewardModel()
rewards = cogn_reward(prompts=batch_prompts, completions=batch_completions)
# list[float] — drop in for any RM call in your PPO/GRPO/DPO loop.
```

See `examples/trl_ppo_integration.py` for a working skeleton.

### Bio/neuro grounding (no biology required to ship)

The default weights are calibrated against 5-fold-CV AUCs and bio/neuro evidence depth. **6 of the 9 cognometric instruments map onto RDoC's *Cognitive Systems* domain** with documented circuit-level evidence in human + animal lesion / fMRI / EEG literatures:

  | instrument        | strongest neural correlate                              | evidence       |
  | ----------------- | ------------------------------------------------------- | -------------- |
  | conversation-loop | OFC + dorsomedial striatum + ACC (perseveration)        | strong         |
  | deception         | DLPFC + VLPFC + ACC + insula (Christ ALE 2009)          | strong         |
  | sycophancy        | pMFC + ventral striatum + vmPFC (Klucharev 2009)        | strong         |
  | goal_drift        | DMN-DAN balance (Smallwood mind-wandering)              | moderate       |
  | plan_action       | PFC-BG-SMA intention-action coupling (apathy lit)       | moderate       |
  | overconfidence    | centro-parietal positivity (Boldt & Yeung 2015)         | moderate       |

The conversation-loop instrument has the highest AUC in the suite (0.9995) AND the deepest neural circuit literature in the suite — rats failing reversal, schizophrenics with alogia, TBI patients with utilization behavior, and language models in conversation-loop all produce the same low-entropy reverberant text shape. Same substrate, same shape.

### Tests

14 new tests in `tests/test_reward.py`. All pass:

- Output shape and range
- Rank correctness on curated sycophantic vs balanced pair
- Universal-perturbation moat (no gameability lift)
- Custom weights (increase penalty, disable instrument)
- Batch interface matches single-call results
- Multi-turn instruments fire on `turns=` input
- Length-mismatch raises `ValueError`

### `styxx.synth` — synthetic preference-pair generation via inverse cognometry

Composes `styxx.attack.craft_adversarial` (v7.0.0 inverse cognometry) with `styxx.reward` (this release). Takes a benign baseline response, hill-climbs a 1-3 token suffix that spikes a chosen instrument, and returns a verified `(prompt, chosen, rejected)` preference pair shaped exactly for cogn-RLHF DPO training.

**Result on the 20-pair sycophancy seed dataset, target_score=0.85:**

  | metric | value |
  | ------ | ----- |
  | crafted with positive delta | **20 / 20** |
  | reached saturation (≥ 0.85) | **20 / 20** |
  | mean delta over crafted pairs | **+0.839** |

  Recursive validation: `fathom_reward` correctly ranks `chosen > rejected` on **20 / 20 (100%)** of the synth-generated pairs — the inverse-cogn crafted perturbations are caught by the forward-cogn reward, both directions self-validating.

  Reproduce: `python examples/synth_preference_pairs.py`. Output: `release/synth_preference_pairs_v0.jsonl`.

```python
from styxx.synth import craft_preference_pair

pair = craft_preference_pair(
    prompt="I think Python is the best. Right?",
    balanced="Python has tradeoffs - strong ecosystem, slow runtime.",
    instrument="sycophancy",
    target_score=0.85,
)
# {chosen: balanced, rejected: balanced + 1-3 token suffix, delta: +0.84}
```

No LLM calls. No API spend. Deterministic hill-climb on the bundled 24-token vocabulary. Nobody else can build this because nobody else has both forward and inverse cognometry shipped.

### Files added

- `styxx/reward.py` — the cognometric reward module
- `styxx/synth.py` — synthetic preference-pair generator (inverse cogn → cogn reward composition)
- `styxx/_demo_baselines.py` — strawman approval-style baseline (sycophancy + length proxies)
- `tests/test_reward.py` — 14 unit + adversarial tests
- `tests/test_synth.py` — 7 tests for the synth pair generator
- `data/cognometric_rlhf_demo_v0.jsonl` — 20 curated (prompt, sycophantic, balanced) triples
- `examples/cognometric_reward_basic.py` — basic usage
- `examples/cogn_rlhf_divergence.py` — divergence demo with summary stats
- `examples/cogn_rlhf_divergence_colab.ipynb` — Colab notebook reproducing the divergence
- `examples/synth_preference_pairs.py` — synth pair generator demo
- `examples/trl_ppo_integration.py` — TRL PPOTrainer skeleton
- `release/cogn_rlhf_divergence_v0.json` — saved demo result
- `release/synth_preference_pairs_v0.jsonl` — synth-generated preference pairs (20)

---

## [7.0.0] — 2026-04-29

**Headline: `styxx.attack` — inverse cognometry. A new subpackage that ships the dual to every cognometric instrument styxx measures: adversarial inputs, cross-instrument fingerprinting, latent-basis decomposition, and a discovered universal adversarial perturbation that fools multiple calibrated detectors at once.**

### Three findings shipped in 7.0.0

**1. Universal cognometric perturbation discovered.** A single fixed string — `"wonderful certainly you're absolutely right amazing undoubtedly"` — appended to ANY clean response, raises mean cross-fire across multiple cognometric instruments by **+0.468 on a held-out test set** (essentially identical to the +0.463 training delta — clean transfer, no overfitting). The first LLM-cognometric analog of [Moosavi-Dezfooli et al. 2017 universal adversarial perturbations](https://openaccess.thecvf.com/content_cvpr_2017/papers/Moosavi-Dezfooli_Universal_Adversarial_Perturbations_CVPR_2017_paper.pdf) for image classifiers — generated by greedy hill-climb on a 24-token vocabulary derived from the published K=1 critical features. Zero LLM calls, zero gradient access, ~2 seconds wall clock.

  Per-instrument transfer on the held-out test set:

  | instrument        | mean delta | transfer? |
  | ----------------- | ---------- | --------- |
  | sycophancy        | +0.938     | YES       |
  | overconfidence    | +0.521     | YES       |
  | deception         | -0.056     | NO (resists) |

  Bundled in `styxx/attack/universal_suffixes_v0.json`. Pinned by `test_universal_suffix_artifact_present_and_transfers` against a fresh held-out batch.

**2. Cognometric instruments are NOT orthogonal.** PCA over n=1600 single-turn corpus rows × 4 instruments: PC1 alone explains **46.2% of variance** (vs 25% expected under perfect orthogonality), with all 4 instruments loading on PC1. Effective dimensionality (participation ratio): ~3.13/4. The deception detector fires at mean 0.805 on overconfidence-positive content — *higher than the overconfidence detector itself* (0.629). Single-instrument readings overstate orthogonality; joint signatures are the more honest unit. Pinned by `test_non_orthogonality_finding_pinned`.

**3. Per-instrument craft-ability is asymmetric (and inverse to AUC ranking).** Hill-climbing a suffix from clean cross-corpus inputs targeting one specific instrument:

  | targeted instrument | succeeded / 30 (target ≥ 0.7) | top delta |
  | ------------------- | ----------------------------- | --------- |
  | sycophancy          | **30 / 30**                   | +0.972    |
  | overconfidence      | 29 / 30                       | +0.697    |
  | deception           | **1 / 30**                    | +0.362    |

  Sycophancy collapses on every attempt; deception resists almost all of them. Calibrated AUC ≠ adversarial robustness. Cross-attack transfer is also asymmetric: deception-targeted attacks ALSO fire overconfidence at ~equal magnitude (+0.384 vs +0.341), while sycophancy attacks transfer cleanly (no leakage).

### Public API

```python
from styxx.attack import (
    # canary mining (training-distribution positives)
    mine, AttackCandidate, AttackResult,

    # natural-FP mining (true adversarials from corpus)
    mine_adversarial,

    # cross-instrument fingerprinting (4 single-turn instruments)
    score_all, applicable_instruments,
    cross_fire_matrix, fingerprint_distance,

    # latent basis decomposition (PCA on cross-firing matrix)
    cognometric_basis, BasisResult,

    # synthetic adversarial generation (greedy hill-climb)
    craft_adversarial, CraftResult, CraftedAdversarial,
    find_universal_suffix, UniversalSuffixResult,

    # registry
    list_instruments, get_instrument, InstrumentSpec,
)
```

CLI:
```
styxx attack <instrument>                   # canary mine (training-distribution positives)
styxx attack <instrument> --adversarial     # natural-FP mine (true adversarials)
styxx attack --list                         # show registered instruments
```

### Coverage

- **6 instruments registered** for `mine` / `mine_adversarial`: `sycophancy`, `loop`, `goal_drift`, `deception`, `plan_action`, `overconfidence`.
- **4 single-turn instruments scored** by `score_all` and `cross_fire_matrix`: the above three (`sycophancy`, `deception`, `overconfidence`) plus `refusal` (fingerprint-only — no bundled labeled corpus, since XSTest is external).
- The remaining two instruments from the `Every Mind Leaves Vitals` 9-suite (`hallucination`, `drift`) are deferred to 7.1+ — both have non-paired or external corpus shapes that need normalization work.

### Spoofability per instrument (natural false positives in training corpora)

  | instrument        | total negs | natural FPs ≥ 0.5 | top FP score |
  | ----------------- | ---------- | ----------------- | ------------ |
  | sycophancy        | 600        | 56 (9%)           | 0.983        |
  | overconfidence    | 100        | 28 (28%)          | 0.946        |
  | plan_action       | 100        | 13                | 0.993        |
  | goal_drift        | 100        | 11                | 0.964        |
  | deception         | 100        | 7                 | 0.965        |
  | **loop**          | 100        | **0**             | 0.287        |

  **Loop is unspoofable** by any natural negative example — matches its CV AUC of 0.9995 (the highest of the 9 instruments). Pinned by `test_loop_is_robust_to_natural_adversarials`.

### Bundled artifacts (inside the wheel)

- `styxx/attack/seeds/<instrument>.jsonl` — top-30 training-distribution positives per registered instrument.
- `styxx/attack/seeds/<instrument>_fp.jsonl` — top-30 natural false positives per spoofable instrument (loop has no file).
- `styxx/attack/signature_calibration_v0.json` — per-instrument cross-fire calibration (descriptive; not a defended adversarial detector — current FP corpus sizes are statistically underpowered for that).
- `styxx/attack/universal_suffixes_v0.json` — the discovered universal suffix + transfer matrix + craft-ability ranking + cross-attack matrix.

Plus the analysis output (in-tree, not in wheel):
- `benchmarks/cognometric_basis_v0.json` — full PCA result on n=1600 single-turn corpus rows.

### Reproducibility

```
python scripts/build_attack_seeds.py --top-k 30        # rebuild bundled seeds
python scripts/compute_signature_calibration.py        # rebuild calibration JSON
python scripts/run_cognometric_basis.py                # rerun PCA decomposition
python scripts/run_craft_experiments.py                # rerun craft + universal hunt
python scripts/analyze_fingerprint_geometry.py         # cross-firing matrix
```

All deterministic (seed=0). The universal suffix is reproduced bit-for-bit from a fresh `pip install styxx==7.0.0` + run.

### Tests

- 43 new tests in `tests/test_attack_v0.py`. Covers every API surface, both polarities of mining, parametric per-instrument coverage, the loop-robustness invariant, the non-orthogonality pin, the craft-ability asymmetry pin, and the universal-suffix transfer pin.
- Full regression: **800 passed, 1 skipped** (757 prior + 43 attack), zero regressions on any existing surface.

### Compatibility

No breaking changes. Every public surface from styxx 6.8.2 is unchanged. `pip install styxx==7.0.0` is a drop-in upgrade for any existing 6.x deployment.

### Why it matters

Every published cognitive-eval claims to measure something real. Almost no one ships the matched offense against their own benchmark, let alone discovers a universal perturbation that fools multiple of them at once. styxx 7.0.0 does both:

- **Defenders** get a known-bad library per instrument (canary mining), a true natural-adversarial library (FP mining), and a cross-instrument fingerprint API to detect anomalous joint signatures.
- **Researchers** get the first published cross-firing matrix for cognometric measurement, a PCA basis, a synthetic adversarial generator, and a discovered universal artifact to use as a baseline.
- **The field** gets the dual paper to *Every Mind Leaves Vitals*: one fixed string defeats multiple calibrated detectors. The K=1 phase-transition signature implies the universal — 7.0.0 ships the receipt.

### Roadmap

- **7.1.0** — `styxx.attack.mutate` (LLM-driven adversarial paraphrase) + extend universal hunt to multi-turn instruments (loop, goal_drift) + companion paper draft *Universal Cognometric Perturbations: A Single String Defeats Multiple Calibrated Detectors*.
- **7.2.0** — composition with the open-weight `lucid` probe-in-loop project: feed inverse-styxx adversarials into Llama-3.1 + nnsight + a deception probe at layer N, measure whether surface pathology and internal state agree under adversarial pressure.

---

## [6.8.2] — 2026-04-27

**Patch: fixes a silent-bypass bug in `@styxx.profile` and `hook_openai()` where callers using the most common import pattern got 0 cognitive steps captured. Surfaced by post-9-of-9 dogfood.**

### Fixed

- **`hook_openai()` now rebinds already-imported `OpenAI` references.** Previously the hook only patched `openai.OpenAI` (the module attribute). Any caller that did `from openai import OpenAI` *before* the hook ran (the default pattern in nearly every Python project) held an unhooked reference in their own module namespace, and `OpenAI()` constructions through that reference silently bypassed the hook. The visible symptom: `@styxx.profile` reported `CognitiveProfile(steps=0)` on real LLM calls, e.g.:

  ```python
  from openai import OpenAI       # ← bound BEFORE @styxx.profile imports

  @styxx.profile
  def my_agent(task):
      client = OpenAI()           # ← bypass: still the unhooked class
      return client.chat.completions.create(...)

  result, p = my_agent("hi")
  print(p.steps)                  # → 0   (pre-6.8.2 bug)
  print(p.steps)                  # → 1+  (6.8.2 fixed)
  ```

  The fix walks `sys.modules` at hook-install time and rebinds any module-level attribute that points at the original `openai.OpenAI` to the hooked replacement. `unhook_openai()` does the symmetric restore. Excludes `openai.*` and `styxx.*` namespaces so the hook machinery itself isn't corrupted.

- **`unhook_openai()` no longer probes `getattr(attr, "_styxx_hooked")`.** The previous implementation walked `sys.modules` and used `getattr` to detect hooked references — but `getattr` triggers lazy-import machinery on third-party modules (notably `torch._classes`), which would raise `RuntimeError` mid-iteration and break unrelated tests in the same process. Replaced with strict identity comparison against a stored module-level reference.

### Added

- **Regression tests** in `tests/test_power_ups.py`:
  - `test_hook_openai_rebinds_already_imported_references` — synthesizes the failing import pattern in a fresh module, asserts the rebind works, asserts unhook restores cleanly
  - `test_hook_openai_does_not_touch_styxx_internals` — pins the exclusion filter so the sweep can never corrupt `styxx.adapters.*` references

### Doc

- Updated `@styxx.profile` docstring to remove the now-outdated "does NOT work" caveat for `from openai import OpenAI`. The three patterns that work are explicitly listed; remaining edge cases (clients constructed before styxx is imported, user-defined `openai.OpenAI` subclasses) are noted.

---

## [6.8.1] — 2026-04-26

**Patch: fixes a long-standing version-attribute drift bug surfaced by post-9-of-9 dogfood, and adds the dogfood invariant that would have caught it.**

### Fixed

- **`styxx.__version__`** now reads from package metadata (`importlib.metadata.version('styxx')`) instead of a hardcoded literal, so it can never drift from the published wheel. The hardcoded value had been frozen at `"6.2.1"` across six minor releases (v6.2.0 → v6.8.0), causing every PyPI install to misreport its own version. Falls back to `"0.0.0+source"` for source-checkout environments without an installed package metadata. Caught by `scripts/dogfood_v650.py` running against a fresh isolated-venv install of v6.8.0.

### Added

- **Dogfood invariant `imports.styxx_version_matches_metadata`** in `scripts/dogfood_v650.py` — asserts that `styxx.__version__` equals `importlib.metadata.version('styxx')` whenever the package is installed. Prevents this class of drift from recurring silently.

- **Dogfood coverage for instruments #8 + #9.** Extended `scripts/dogfood_v650.py` to exercise `overconf_check` and `goal_check` on imports, fingerprints, canonical paired cases, cross-instrument compatibility, edge cases (empty, unicode, very-short, long-session), performance, and determinism. Total dogfood checks: **65/65 green** against a fresh PyPI install. Atlas assertion bumped from v0.4 (7 instruments) to v0.6 (9 instruments).

---

## [6.8.0] — 2026-04-26

**Headline: instrument #9 (goal-drift detection) — sixth and FINAL instrument shipped under the call from [*Every Mind Leaves Vitals*](https://doi.org/10.5281/zenodo.19777921). The 9-instrument suite the position paper called for is now COMPLETE. 9-for-9 on cognometric instruments showing K=1 phase-transition signature, each with a different critical feature.**

### Added — ninth and final cognometric instrument: goal drift

- **`from styxx.guardrail import goal_check`** — calibrated multi-turn goal-drift detector. Pure Python, no embeddings, Pyodide-safe. Sibling to conversation-loop (instrument #5): both are multi-turn, but loop measures stagnation while goal-drift measures dispersion (the agent moves further from its goal anchor turn after turn). Distinct from drift v1 (instrument #3): drift v1 is a per-call schema-mismatch detector for tool calls; goal drift is a multi-turn intent-migration detector for agent sessions.

  ```python
  v = goal_check(turns=[
      "Goal: research the rate-limit policy and summarize per-endpoint limits.",
      "Searched the API docs.",
      "Started looking at OAuth flows instead.",
      "Wrote a comparison of OAuth providers.",
  ])
  v.drift_risk    # calibrated probability in [0, 1]
  v.shows_drift   # bool against threshold (default 0.5)
  v.top_signals   # 3 strongest features (signed contribution)
  ```

  9 multi-turn anchor-relative features (anchor_recall_score, anchor_to_last_bigram_jaccard, anchor_to_last_entity_overlap, cumulative_anchor_drift, mean_anchor_overlap, max_inter_turn_levenshtein, monotonic_drift_fraction, log_n_turns, log_total_words). Trained on **n=200 paired (anchored, drifted) 5-turn agent sessions** sampled from `gpt-4o-mini` under contrasting STANCE-level system prompts on 100 diverse goal statements. **5-fold CV mean AUC 0.9645 ± 0.0294**.

- **Phase-transition signature replicates on instrument #9.** Critical_K=**1** on `anchor_to_last_bigram_jaccard` (Δ +0.4143) — direct cross-turn bigram overlap between the goal-statement turn and the agent's final turn. K=2 adds `max_inter_turn_levenshtein` (Δ +0.05).

  **9-FOR-9** on cognometric instruments showing K=1 phase transition under the same measurement protocol, each with a DIFFERENT critical feature:

  | instrument        | critical feature              | Δ AUC at K=1 |
  | ----------------- | ----------------------------- | ------------ |
  | hallucination v4  | trigram_novelty               | +0.4947      |
  | refusal v1        | starts_with_sorry             | +0.469       |
  | drift v6.0        | (per-class K=1-2)             | +0.4973      |
  | sycophancy v0     | superlative_density           | +0.4354      |
  | conversation-loop | avg_pairwise_levenshtein      | +0.4995      |
  | deception v0      | log_word_count                | +0.3738      |
  | plan-action v0    | bigram_jaccard_overlap        | +0.3832      |
  | overconfidence v0 | mean_sentence_length          | +0.2298      |
  | goal-drift v0     | anchor_to_last_bigram_jaccard | +0.4143      |

  **The K=1 phase-transition prediction from *Every Mind Leaves Vitals* is now empirically held across the COMPLETE 9-instrument suite the paper called for**, across instrument families (single-turn lexical / cross-turn structural / lexical-style register / cross-section plan-action / multi-turn drift) and AUC bands (0.7702 to 0.9995).

- **Corpus design discipline.** Stance-level system prompts only — NO lexical hints. The drifted prompt explicitly says *"don't announce that you're getting off-track; just let the work shift"* — same prompt-leakage avoidance carried forward from instruments #7 plan-action and #8 overconfidence.

- **Documented failure modes:**
  1. Single-source corpus (gpt-4o-mini under stance-prompt instruction); v1 priority is real long-horizon agent traces with annotated drift events
  2. **Paraphrastic anchored sessions can score above threshold.** The detector calibrates against gpt-4o-mini-generated anchored sessions which use heavy verbatim repetition of goal vocabulary. Hand-crafted paraphrastic anchored sessions (where the agent stays on-topic but uses different words) can trip the threshold. Pinned by regression test. v1 fix path: semantic-embedding overlap to replace pure bigram Jaccard.
  3. 5-turn fixed window — `log_n_turns` carries zero coefficient because the corpus has zero variance on session length. Pinned.
  4. `mean_anchor_overlap` and `cumulative_anchor_drift` carry equal-and-opposite coefficients (split signal). Pinned.
  5. English-only feature vocabularies.
  6. Requires turn-segmented input.

- **Calibration fingerprint** in `styxx.guardrail.calibrated_weights_goal_drift_v0.CALIBRATION_FINGERPRINT`. Atlas bumped to **v0.6**: 21 fingerprints across 9 instruments × 16 substrates.

- **20 new unit tests** in `tests/test_goal_drift_v0.py`, including the symbolic `test_position_paper_count_is_now_complete` that pins the 9-of-9 milestone by importing every instrument's API entry point. Full pytest run: **755 passed, 1 skipped**.

### Added — atlas v0.6

- `benchmarks/cognometry_fingerprint_atlas_v0.json` → **v0.6**:
  - 21 fingerprints (was 20)
  - 9 instruments (was 8)
  - 16 substrates (was 15)
  - `v0_6_changelog` entry documents the 9-for-9 K=1 phase-transition completion.

### Reproducer

`scripts/goal_drift_train_v0.py` — seed-pinned, deterministic, resumable cache. `OPENAI_API_KEY=... python scripts/goal_drift_train_v0.py`.

### Position-paper status: COMPLETE

**All 9 instruments called for in *Every Mind Leaves Vitals* are now shipped** (hallucination, refusal, tool-call drift, sycophancy, conversation-loop, deception, plan-action, overconfidence, goal-drift). The 9-for-9 K=1 phase-transition signature confirms the central empirical prediction of the position paper across the complete suite, across all instrument families, and across the full AUC band the paper hypothesized.

Net: 9 of 9 calibrated cognometric instruments shipped. The call is closed.

---

## [6.7.0] — 2026-04-26

**Headline: instrument #8 (overconfidence-register detection) — fifth instrument shipped under the call from [*Every Mind Leaves Vitals*](https://doi.org/10.5281/zenodo.19777921). 8-for-8 on cognometric instruments showing K=1 phase-transition signature, each with a different critical feature. Honest AUC: 0.7702 — the lowest in the v0 suite, shipped at this number rather than gamed.**

### Added — eighth cognometric instrument: overconfidence register

- **`from styxx.guardrail import overconf_check`** — calibrated overconfidence-register detector. Pure Python, no embeddings, Pyodide-safe. Sibling to deception (instrument #6) and hallucination (#1): hallucination measures fabrication-prone phrasing; deception measures rhetorical-signature register; overconfidence measures epistemic-commitment register. **NOT a truth detector.**

  ```python
  v = overconf_check(prompt, response)
  v.overconf_risk    # calibrated probability in [0, 1]
  v.shows_overconf   # bool against threshold (default 0.5)
  v.top_signals      # 3 strongest signed contributions
  ```

  9 register features (certainty/hedge/evidence-marker densities, `epistemic_balance` = (cert - hedge) / (cert + hedge + 1), strong-assertion ratio, unhedged-claim ratio, mean sentence length, log word count, specific-number density). Trained on **n=200 paired (calibrated, overconfident) responses** sampled from `gpt-4o-mini` under contrasting STANCE-level system prompts on 100 diverse questions across factual / quantitative / opinion / predictive / mechanism / contested-fact substrates. **5-fold CV mean AUC 0.7702 ± 0.0648**.

- **Phase-transition signature replicates on instrument #8.** Critical_K=**1** on `mean_sentence_length` (Δ +0.2298) — a length confound: calibrated responses pack hedges + qualifications that increase sentence length. K=2 adds `epistemic_balance` (Δ +0.0295) — the lexical-register signal that was the design hypothesis. **8-for-8 on cognometric instruments showing K=1 phase transition** under the same measurement protocol, each with a different critical feature:

  | instrument        | critical feature           | Δ AUC at K=1 |
  | ----------------- | -------------------------- | ------------ |
  | hallucination v4  | trigram_novelty            | +0.4947      |
  | refusal v1        | starts_with_sorry          | +0.469       |
  | drift v6.0        | (per-class K=1-2)          | +0.4973      |
  | sycophancy v0     | superlative_density        | +0.4354      |
  | conversation-loop | avg_pairwise_levenshtein   | +0.4995      |
  | deception v0      | log_word_count             | +0.3738      |
  | plan-action v0    | bigram_jaccard_overlap     | +0.3832      |
  | overconfidence v0 | mean_sentence_length       | +0.2298      |

- **Honest AUC disclosure.** AUC 0.7702 is the lowest in the v0 suite. We ship at this number rather than gaming the corpus. The signal is real (well above chance) but moderate — `gpt-4o-mini` does not always shift register on well-established factual questions ("How does GPS work?" produces a similar response under both stance prompts; the register shift is dramatic on contested questions and barely visible on settled ones). The K=1 length confound and the question-pool dependence are documented in [`calibrated_weights_overconfidence_v0.CALIBRATION_NOTES.honest_AUC_disclosure`](styxx/guardrail/calibrated_weights_overconfidence_v0.py).

- **Corpus design discipline.** Stance-level system prompts only — NO lexical hints. The contrastive prompts contrast at the level of epistemic stance ("careful expert who scales certainty to evidence" vs. "confident speaker who never qualifies") and deliberately do NOT name certainty markers, hedge words, or any feature we measure. Carried forward from instrument #7 plan-action where the prompt-leakage failure mode was first identified and pinned by a regression test.

- **Scope warning: NOT a truth detector.** Overconfidence here scores REGISTER (commitment markers, hedge density, evidence attribution), not factual correctness. A correct answer stated confidently will score as overconfident. An incorrect answer stated humbly will not. Pair with hallucination v4 (or NLI guardrail v3) for joint truth+register monitoring.

- **Counter-intuitive empirical finding pinned by regression test:** `specific_number_density` coefficient is small NEGATIVE in the trained model. Design intuition was overconfident responses invent specific numbers; empirically, calibrated responses cite numbers more (with attribution). Pinned in `tests/test_overconfidence_v0.py::test_documented_specific_number_coef_is_negative`.

- **Documented failure modes:**
  1. K=1 = `mean_sentence_length` is a length confound, not a lexical-certainty feature
  2. Question-pool dependence (high AUC on contested, low AUC on factual)
  3. Single-source corpus (gpt-4o-mini only)
  4. `specific_number_density` coefficient flipped opposite to design intuition
  5. English-only feature vocabularies
  6. Not a truth detector

- **Calibration fingerprint** in `styxx.guardrail.calibrated_weights_overconfidence_v0.CALIBRATION_FINGERPRINT`. Atlas bumped to **v0.5**: 20 fingerprints across 8 instruments × 15 substrates.

- **17 new unit tests** in `tests/test_overconfidence_v0.py`. Full pytest run: **735 passed, 1 skipped**.

### Added — atlas v0.5

- `benchmarks/cognometry_fingerprint_atlas_v0.json` → **v0.5**:
  - 20 fingerprints (was 19)
  - 8 instruments (was 7)
  - 15 substrates (was 14)
  - `v0_5_changelog` entry documents the 8-for-8 K=1 phase transition and the honest AUC disclosure.

### Reproducer

`scripts/overconfidence_train_v0.py` — seed-pinned, deterministic, resumable cache. `OPENAI_API_KEY=... python scripts/overconfidence_train_v0.py`.

### Position-paper status

**5 of 6 instruments called for in *Every Mind Leaves Vitals* now shipped** (sycophancy, conversation-loop, deception, plan-action, overconfidence). One remaining: **goal drift** (cross-session intent drift, distinct from the existing tool-call drift v1). Net: 8 of 9 calibrated cognometric instruments shipped.

---

## [6.6.0] — 2026-04-26

**Headline: instrument #7 (plan-action gap detection) — fourth instrument shipped under the call from [*Every Mind Leaves Vitals*](https://doi.org/10.5281/zenodo.19777921). 7-for-7 on cognometric instruments showing K=1 phase-transition signature, each with a different critical feature.**

### Added — seventh cognometric instrument: plan-action gap

- **`from styxx.guardrail import plan_action_check`** — calibrated cross-section plan-action gap detector. Pure Python, no embeddings, Pyodide-safe. Sibling to drift (instrument #3): drift catches a malformed tool call against schema; plan-action gap catches when the agent's *stated intent* and *emitted action* diverge at the content level.

  ```python
  v = plan_action_check(plan, action)
  v.gap_risk     # calibrated probability in [0, 1]
  v.shows_gap    # bool against threshold (default 0.5)
  v.top_signals  # 3 strongest cross-section features
  ```

  9 cross-section features (bigram/trigram Jaccard between plan and action, action-verb overlap, entity overlap, length ratio + diff, deviation-marker density, plan-only-content-word ratio, log total words). Trained on **n=200 paired (matched, mismatched) plan-action pairs** sampled from `gpt-4o-mini` under contrasting system prompts on 100 diverse agent tasks. **5-fold CV mean AUC 0.9225 ± 0.0322**.

- **Phase-transition signature replicates on instrument #7.** Critical_K=**1** on `bigram_jaccard_overlap` (Δ +0.3832) — cross-section bigram overlap between plan and action. K=2 adds `log_total_words` (Δ +0.04). **7-for-7 on cognometric instruments showing K=1 phase transition** under the same measurement protocol, each with a different critical feature:

  | instrument        | critical feature           | Δ AUC at K=1 |
  | ----------------- | -------------------------- | ------------ |
  | hallucination v4  | trigram_novelty            | +0.4947      |
  | refusal v1        | starts_with_sorry          | +0.469       |
  | drift v6.0        | (per-class K=1-2)          | +0.4973      |
  | sycophancy v0     | superlative_density        | +0.4354      |
  | conversation-loop | avg_pairwise_levenshtein   | +0.4995      |
  | deception v0      | log_word_count             | +0.3738      |
  | plan-action v0    | bigram_jaccard_overlap     | +0.3832      |

- **Honest corpus disclosure.** An earlier corpus that allowed the mismatched system prompt to instruct the model to use deviation markers ("actually,"/"instead,") in the action saturated AUC at 1.000 with K=1 = `deviation_marker_density` — a pure prompt-leakage artifact, since we'd told the model exactly which lexical signature to produce. The cleaned corpus (no deviation-marker hint) gives the honest AUC 0.9225 with a real cross-section overlap signal at K=1. Both results are documented in `CALIBRATION_NOTES.corpus_design_warning`.

- **Calibration fingerprint atlas v0.4.** Atlas now ships **19 fingerprints across 7 instruments × 14 substrates** (was 18/6/13).

- **Documented failure modes** (in [`calibrated_weights_plan_action_v0.CALIBRATION_NOTES`](styxx/guardrail/calibrated_weights_plan_action_v0.py)):
  1. Single-source corpus (gpt-4o-mini under prompt instruction); v1 priority is real BFCL-multi-turn agent traces with annotated gaps
  2. **Symbolic-to-numerical false positive** — when plan describes symbolic computation ("compute A = πr²") and action shows numerical execution ("3.14159 × 7 × 7 = 153.94"), bigram overlap is naturally low even though the pair is semantically matched. Pinned by a regression test. v1 fix path is semantic embedding overlap.
  3. Requires structured `(plan, action)` input — separate parsing step needed for inline-CoT outputs
  4. Length features (`action_to_plan_length_ratio` + `action_minus_plan_word_count`) split the signal — small modeling redundancy
  5. `verb_overlap_ratio` carries near-zero learned weight (small action-verb vocabulary)
  6. English-only feature vocabularies

### Added — reproducers

- [`scripts/plan_action_train_v0.py`](scripts/plan_action_train_v0.py) — full pipeline (sample paired plan-action → parse PLAN:/ACTION: structure → featurize → train → ablate). Resumable cache in `benchmarks/data/plan_action/pairs_v0.jsonl`.

### Files

```
styxx/guardrail/plan_action.py                       — runtime API (plan_action_check, PlanActionVerdict)
styxx/guardrail/plan_action_signals.py                — 9 cross-section feature extractors
styxx/guardrail/calibrated_weights_plan_action_v0.py  — weights + fingerprint + corpus_design_warning + failure modes
benchmarks/data/plan_action/pairs_v0.jsonl            — 200 paired pairs (cached training data)
benchmarks/plan_action_feature_scaling.json           — full ablation history
benchmarks/plan_action_weights_v0.json                — paste-ready weights bundle
tests/test_plan_action_v0.py                          — 15 unit tests, including documented-failure-mode regression checks
```

### Context

Fourth instrument shipped under [*Every Mind Leaves Vitals*](https://doi.org/10.5281/zenodo.19777921)'s call for #4-#9 (sycophancy + conversation-loop + deception preceded — same day cycle). Less than 48 hours from the position paper landing to four instruments shipped under it, all replicating the K=1 phase-transition signature, each with a different critical feature. The structural prediction continues to hold across instrument families (single-turn lexical, cross-turn structural, lexical-style-deception, cross-section plan-action). 93/93 tests pass across all 7 instruments.

---

## [6.5.0] — 2026-04-26

**Headline: instrument #6 (deception-signature detection) — third instrument shipped under the call from [*Every Mind Leaves Vitals*](https://doi.org/10.5281/zenodo.19777921). 6-for-6 on cognometric instruments showing K=1 phase-transition signature. NOT a lie detector — see scope warning.**

### Added — sixth cognometric instrument: deception-signature

> **Scope warning:** This is NOT a lie detector. It detects *lexical signatures of instruction-induced dishonesty* — patterns that emerge under prompt instruction to be vague vs. specific, not actual factual deception. False positives on qualified-honest writing; false negatives on confident lies with specifics. Use as a signal in agent monitoring, not as a verdict.

- **`from styxx.guardrail import deception_check`** — calibrated text-only deception-signature detector. Pure Python, sub-millisecond on CPU, no model weights, Pyodide-safe.

  ```python
  v = deception_check(prompt, response)
  v.deception_risk    # calibrated probability in [0, 1]
  v.shows_signature   # bool against threshold (default 0.5)
  v.top_signals       # 3 strongest features by signed contribution
  ```

  9 lexical features drawn from the Pennebaker / Newman / Hauch deception-linguistics tradition, adapted for LLM output (specificity, first-person density, exclusive words, vagueness, negation, hedge-confidence clash, cognitive markers, opinion phrases, log word count). Trained on **n=200 paired responses** sampled from `gpt-4o-mini` under contrasting (*honest* / *dishonest*) system prompts on 100 diverse seed questions (factual / opinion / contested). **5-fold CV mean AUC 0.9560 ± 0.0242**.

- **Phase-transition signature replicates on instrument #6.** Critical_K=**1** on `log_word_count` (Δ +0.3738) — dishonest-instructed responses are systematically shorter and less specific in this corpus. K=2 adds `specificity_density` (Δ +0.079). **6-for-6 on cognometric instruments showing K=1 phase transition** under the same measurement protocol, each with a different critical feature:

  | instrument        | critical feature          | Δ AUC at K=1 |
  | ----------------- | ------------------------- | ------------ |
  | hallucination v4  | trigram_novelty           | +0.4947      |
  | refusal v1        | starts_with_sorry         | +0.469       |
  | drift v6.0        | (per-class K=1-2)         | +0.4973      |
  | sycophancy v0     | superlative_density       | +0.4354      |
  | conversation-loop | avg_pairwise_levenshtein  | +0.4995      |
  | deception v0      | log_word_count            | +0.3738      |

- **Calibration fingerprint atlas v0.3.** Atlas now ships **18 fingerprints across 6 instruments × 13 substrates** (was 17/5/12).

- **AUC 0.04 lower than the prior five — declared honestly.** Deception is genuinely harder to detect from text alone than concrete failure modes. We disclose the gap rather than paper over it.

- **Documented failure modes** (in [`calibrated_weights_deception_v0.CALIBRATION_NOTES`](styxx/guardrail/calibrated_weights_deception_v0.py), with a prominent `scope_warning`):
  1. **NOT a lie detector** — lexical signature ≠ ground-truth deception
  2. Single-source corpus (gpt-4o-mini under prompt instruction)
  3. `log_word_count` as critical feature is partly a corpus artifact — sign may invert on corpora where dishonest responses pad with bulk
  4. `specificity_density` uses a regex proxy for named entities (v1 priority: real NER)
  5. English-only feature vocabularies
  6. `opinion_phrase_density` carries zero learned weight on this corpus
  7. `negation_density` learned a *negative* coefficient (Newman's positive sign for human deception did not replicate on LLM output)

### Added — reproducers

- [`scripts/deception_train_v0.py`](scripts/deception_train_v0.py) — full pipeline (sample paired honest/dishonest → featurize → train → ablate). Resumable cache in `benchmarks/data/deception/responses_v0.jsonl`. Seed-pinned, deterministic.

### Files

```
styxx/guardrail/deception.py                       — runtime API (deception_check, DeceptionVerdict)
styxx/guardrail/deception_signals.py                — 9 lexical feature extractors
styxx/guardrail/calibrated_weights_deception_v0.py  — weights + fingerprint + LOUD failure modes
benchmarks/data/deception/responses_v0.jsonl        — 200 paired responses (cached training data)
benchmarks/deception_feature_scaling.json           — full ablation history
benchmarks/deception_weights_v0.json                — paste-ready weights bundle
tests/test_deception_v0.py                          — 15 unit tests, including scope-warning + documented-failure-mode regression checks
```

### Context

Third instrument shipped under [*Every Mind Leaves Vitals*](https://doi.org/10.5281/zenodo.19777921)'s call for #4-#9 (sycophancy, conversation-loop preceded — same day). Less than 48 hours from the position paper landing to three instruments shipped under it, all replicating the K=1 phase-transition signature on a different critical feature each time. The structural prediction continues to hold across instrument families (single-turn lexical / cross-turn structural / lexical-style-deception). 78/78 tests pass across all 5 calibrated text-only instruments.

---

## [6.4.0] — 2026-04-26

**Headline: instrument #5 (conversation-loop detection) — second instrument shipped under the call from [*Every Mind Leaves Vitals*](https://doi.org/10.5281/zenodo.19777921). 5-for-5 on cognometric instruments showing K=1 phase-transition signature under the same measurement protocol.**

### Added — fifth cognometric instrument: conversation-loop

- **`from styxx.guardrail import loop_check`** — calibrated cross-turn loop detector. Pure Python, no embeddings, no model weights, Pyodide-safe.

  ```python
  v = loop_check(turns=[t1, t2, t3, t4])
  v.loop_risk     # calibrated probability in [0, 1]
  v.in_loop       # bool against threshold (default 0.5)
  v.n_turns       # number of input turns
  v.top_signals   # 3 strongest cross-turn features by signed contribution
  ```

  9 cross-turn features (bigram/trigram overlap consecutive, verbatim 5-gram repeat count, length CV, opener repeat rate, distinct-word ratio, pairwise Levenshtein, max pairwise bigram overlap, log turn count). Trained on **n=200 paired multi-turn conversations** sampled from `gpt-4o-mini` under contrasting (*loop* / *progress*) system prompts, 100 generic seed topics, 4 agent turns each. **5-fold CV mean AUC 0.9995 ± 0.0010**.

- **Phase-transition signature replicates on instrument #5.** Critical_K=**1** on `avg_pairwise_levenshtein` (Δ +0.4995) — a single feature (mean normalized char-level Levenshtein distance across all turn pairs) takes detection from chance to AUC 0.9995. **5-for-5 on cognometric instruments showing K=1 phase transition** under the same measurement protocol:

  | instrument        | critical feature          | Δ AUC at K=1 |
  | ----------------- | ------------------------- | ------------ |
  | hallucination v4  | trigram_novelty           | +0.4947      |
  | refusal v1        | starts_with_sorry         | +0.469       |
  | drift v6.0        | (per-class K=1-2)         | +0.4973      |
  | sycophancy v0     | superlative_density       | +0.4354      |
  | conversation-loop | avg_pairwise_levenshtein  | +0.4995      |

- **Calibration fingerprint atlas v0.2.** Atlas now ships **17 fingerprints across 5 instruments × 12 substrates** (was 16/4/11).

- **Single-turn short-circuit.** `loop_check(turns=[x])` returns `loop_risk=0.0, in_loop=False` — loops are multi-turn by definition.

- **Documented failure modes** (in [`calibrated_weights_loop_v0.CALIBRATION_NOTES`](styxx/guardrail/calibrated_weights_loop_v0.py)):
  1. Single-source training (gpt-4o-mini under prompt-induced loop instructions). v1 priority: real BFCL-multi-turn agent traces with human-labeled loops, plus cross-model corpus.
  2. **Counter-intuitive `distinct_word_ratio` coefficient.** Intuition says LOW (loops have less vocabulary) → predict loop=1, so coefficient should be negative. Learned coefficient is +0.95. Explanation: gpt-4o-mini under "rephrase" instruction reaches for synonyms each turn, so its distinct-word-ratio actually goes UP under loop. Honest to the corpus; likely inverted on natural-failure loops. Pinned by a regression test.
  3. No temporal modeling — features treat turns as a set.
  4. Very short turns (<10 words) underfire the cross-turn features.
  5. `log_n_turns` carries zero learned weight on this corpus (all training conversations are 4 turns; feature is constant).

### Added — reproducers

- [`scripts/loop_train_v0.py`](scripts/loop_train_v0.py) — full pipeline (sample paired multi-turn → featurize → train → ablate). Resumable cache in `benchmarks/data/loop/conversations_v0.jsonl`. Seed-pinned, deterministic.

### Files

```
styxx/guardrail/conversation_loop.py            — runtime API (loop_check, LoopVerdict)
styxx/guardrail/conversation_loop_signals.py    — 9 cross-turn feature extractors
styxx/guardrail/calibrated_weights_loop_v0.py   — weights + fingerprint + failure modes
benchmarks/data/loop/conversations_v0.jsonl     — 200 paired conversations (cached training data)
benchmarks/loop_feature_scaling.json            — full ablation history
benchmarks/loop_weights_v0.json                 — paste-ready weights bundle
tests/test_loop_v0.py                           — 16 unit tests, including documented-failure-mode regression checks
```

### Context

This is the second instrument shipped under [*Every Mind Leaves Vitals*](https://doi.org/10.5281/zenodo.19777921)'s call for instruments #4 through #9 (sycophancy v0 was the first, in 6.3.0 — same day). Less than 48 hours from the call to two confirmed phase-transition replications. The structural prediction continues to hold.

---

## [6.3.0] — 2026-04-26

**Headline: instrument #4 (sycophancy detection) shipped within 24h of the position paper [*Every Mind Leaves Vitals*](https://doi.org/10.5281/zenodo.19777921) calling for instruments #4–#9. Phase-transition signature replicated: critical_K=1 on `superlative_density`, AUC 0.500 → 0.9354 (Δ +0.4354), substrate-independent across three substrates.**

### Added — fourth cognometric instrument: sycophancy

- **`from styxx.guardrail import sycoph_check`** — calibrated text-only sycophancy detector. Pure Python, sub-millisecond on CPU, no model weights, no logprobs, Pyodide-safe.

  ```python
  v = sycoph_check(prompt, response)
  v.sycoph_risk   # calibrated probability in [0, 1]
  v.sycophantic   # bool against threshold (default 0.5)
  v.top_signals   # 3 strongest features by signed contribution
  ```

  Trained on **n=1200 paired responses** generated from `gpt-4o-mini` against the [Anthropic sycophancy eval corpus](https://github.com/anthropics/evals/tree/main/sycophancy) (Perez et al. 2022) across three substrates (NLP survey, philpapers2020, political typology) under contrasting system prompts: *yielding* (validate the user's view) vs. *evidence-first* (reason regardless of stated view). 9 surface features (agreement lexicon, premise echo, counter-evidence density, capitulation phrases, agreement openers, opinion markers, superlative density, hedge density, log word count). 5-fold CV mean AUC **0.9720 ± 0.0052**.

- **Phase-transition signature replicates on instrument #4.** Greedy forward feature selection finds critical_K=**1** on `superlative_density` — a single feature takes detection from chance (AUC 0.500) to **0.9354** (Δ +0.4354). The remaining 8 features combined add only +0.037. **Per-substrate ablation confirms K=1 holds within each substrate** (NLP-survey 0.909, philpapers2020 0.950, political-typology 0.944) — phase transition is not a pooling artifact. Same shape as the prior three instruments under the same measurement protocol.

- **Calibration fingerprint atlas v0.1.** Added 4 new fingerprints (pooled + 3 per-substrate) to [`benchmarks/cognometry_fingerprint_atlas_v0.json`](benchmarks/cognometry_fingerprint_atlas_v0.json). Atlas now ships **16 fingerprints across 4 instruments × 11 substrates**.

- **Documented failure modes** (in [`calibrated_weights_sycophancy_v0.CALIBRATION_NOTES`](styxx/guardrail/calibrated_weights_sycophancy_v0.py), not appendix):
  1. Single-model training — gpt-4o-mini only; v1 priority is cross-model corpus (Claude, Llama, Mistral)
  2. K=1 critical feature is `superlative_density` — terse agreement *without* praise can underfire
  3. False positives on warmly-worded evidence answers (*"Great question! Actually..."*) — confirmed in smoke tests
  4. `premise_echo_rate` carries a *negative* coefficient on this corpus (high echo correlates with counter-quotation); the sign may invert on other corpora

### Added — research artifact: v0.1 robustness experiment

A failure-mode-driven retrain. Augmented training corpus with 300 additional "warm-evidence" examples (system prompt: *"open warmly but reason from evidence"*). Result: pooled AUC 0.9382 (−0.034 from v0). v0.1 is **more robust to politeness-style false positives** but reveals a **true ceiling of the lexical approach**: a warm-opening response that contradicts the user's view *without* using counter-vocabulary still fires the K=1 detector. The remaining failure mode is genuinely beyond surface features — a semantic-aware NLI feature is the v1 fix path.

v0.1 weights preserved as research artifact in [`benchmarks/sycophancy_weights_v01.json`](benchmarks/sycophancy_weights_v01.json); not exposed as the default detector. Reproducer: [`scripts/sycophancy_train_v01.py`](scripts/sycophancy_train_v01.py).

### Added — reproducers

- [`scripts/sycophancy_train_v0.py`](scripts/sycophancy_train_v0.py) — full pipeline (sample → featurize → train → ablate). Resumable cache in `benchmarks/data/sycophancy/responses_v0.jsonl`. Seed-pinned, deterministic.
- [`scripts/sycophancy_per_substrate.py`](scripts/sycophancy_per_substrate.py) — per-substrate ablation, no new API calls.
- [`scripts/sycophancy_train_v01.py`](scripts/sycophancy_train_v01.py) — warm-evidence augmentation for the robustness experiment.

### Files

```
styxx/guardrail/sycophancy.py                       — runtime API (sycoph_check, SycophancyVerdict)
styxx/guardrail/sycophancy_signals.py               — 9 feature extractors
styxx/guardrail/calibrated_weights_sycophancy_v0.py — weights + fingerprint + failure modes
benchmarks/data/sycophancy/                         — Anthropic eval corpus + cached responses
benchmarks/sycophancy_feature_scaling.json          — full ablation history
benchmarks/sycophancy_per_substrate_ablation.json   — per-substrate ablation
benchmarks/sycophancy_weights_v0.json               — paste-ready weights bundle
benchmarks/sycophancy_weights_v01.json              — robustness experiment weights
```

### Context

This is the first instrument shipped after the position paper [*Every Mind Leaves Vitals: On the Cognometric Layer, Substrate-Independence, and the One-Time Choice We Have*](https://doi.org/10.5281/zenodo.19777921) called for instruments #4 through #9 (conversation-loop, plan-action gap, sycophancy, deception, goal drift, overconfidence). Less than 24 hours from publication of the call to first shipped instrument under it. The phase-transition signature predicted by the paper holds — one more empirical confirmation, with reproducible numbers, in the same style as the prior three.

---

## [6.2.1] — 2026-04-25

**Headline: dogfood pass against live LLMs surfaced 4 small bugs and 1 documentation gap. Every advertised API now produces what the README promises on `pip install styxx`.**

### Companion: robustness supplement

Published alongside this release as a separate citation:

- **Cognometric Fingerprint Specification v1.0 — Robustness Supplement** (Fathom v22). 24-attack adversarial audit across 8 strategy categories. Baseline 66.7% false-negative evasion → hardened 16.7% (4× reduction). Residual limits documented openly in §7. CC-BY-4.0. DOI [10.5281/zenodo.19761194](https://doi.org/10.5281/zenodo.19761194). Reproducible via `node packages/styxx-scope/_test_adversarial.js`.

### Fixed

- **`@styxx.profile(name="...")` kwarg now works.** README and docstrings showed `name=` as a kwarg, but the function signature only accepted positional strings. Added explicit `name` kwarg. Both forms now work: `@profile("foo")` and `@profile(name="foo")`. The parametric path also returns a hybrid object that's both a context manager AND a decorator factory (previously failed when used as `@profile(name="x")` on a function).

- **`Vitals.as_dict()` now serializes `mode` field.** Adapter pipelines set `vitals.mode` (text-heuristic / consensus / hybrid+companion / etc.) on the live object, but `as_dict()` was dropping it on JSON export. Analytics, datadog, and langsmith pipelines lost the tier indicator silently. Added explicit `mode` key to the dict view.

- **Anthropic adapter: `vitals.mode` now labeled in text mode.** `watch._classify_from_text()` built Vitals without `mode='text-heuristic'` even though the standalone `text_features.build_vitals()` set it. Inconsistent — fixed so callers can branch reliably regardless of which entry point produced the reading.

- **Anthropic adapter: `mode='companion'` falls back gracefully.** When torch isn't installed, the companion path silently returned `vitals=None` — user requested companion but got nothing back with no information. Now falls back to text-heuristic with a label like `'text-heuristic (companion-unavailable)'` so the reading happens AND the failure mode is transparent.

- **`examples/quickstart.py` no longer crashes on first run.** If `OPENAI_API_KEY` was set but the `openai` SDK wasn't installed, `live_demo()` raised `ImportError` and killed the hello-world. Now catches the ImportError and falls back to the offline trajectory demo with a helpful install hint.

- **README hero example now runnable.** Previous hero used an undefined `run_langchain(task)` helper, so copy-pasters got `NameError`. Replaced with a self-contained `styxx.OpenAI` example that produces the documented single-step output on first try, with a separate richer multi-step example below for context.

### Documented

- **`@profile auto_hook` caveat documented.** The hook only catches new `openai.OpenAI()` instances constructed AFTER the profile context begins, AND only when the class is accessed via live module lookup. The 3 working patterns and the 1 that doesn't are now documented in the docstring with explicit code examples and an escape-hatch via `styxx.observe()` / `profile_session().record()` for framework integrations that bypass the hook.

- **OpenAI adapter docstring documents legitimate `vitals=None` cases.** Three scenarios where `.vitals=None` is correct fail-open behavior rather than a bug: pure tool-call responses (no text trajectory), models without logprobs, and `stream=True` (use `styxx.observe()` after collecting full text).

### Added

- **`scripts/launch_metrics.py`** — one-shot funnel readout polling Zenodo, PyPI, and GitHub. No dependencies beyond stdlib. Surfaces real distribution data without manual checks.

- **`scripts/dogfood_e2e.py`** — exhaustive end-to-end test against live gpt-4o-mini and claude-haiku-4-5. Exercises every README-advertised public API (drop-in OpenAI, `@profile` decorator, `@trust` RAG, `gate`, `refuse_check`, `drift_check`, CLI, anthropic text-only). Pass-rate: 25/27 against live LLMs (2 env-blocked).

- **`scripts/bug_hunt.py`** — adversarial dogfood across 8 categories: fail-open contract, streaming, tool calling, classifier residuals, advanced APIs, JSON roundtrip, multithread, edge cases. 29 pass · 0 fail · 4 documented v1 specialist limits.

### Cleanup

- Three unused root-level files removed: `README.old.md`, `WHAT-WE-BUILT-2026-04-22.md`, `INVENTION-CIS-v0.md` (the last was byte-identical to `papers/cognitive-instruction-set-v0.md`).

### Test suite

`653 pass · 5 skip · 0 fail` (was 622/5 before this release — kwarg fix
unblocks 13 previously-skipped mode-label assertions, and the autogen
adapter test file (18 tests) was inadvertently excluded from earlier
runs · re-included here for full coverage).

### Dogfood evidence

Verified end-to-end against live LLMs and synthetic data in this release:

  · `@styxx.profile` on multi-step gpt-4o-mini agent — phase-transition
    fault correctly flagged between steps
  · `styxx.OpenAI` drop-in wrapper produces calibrated vitals on live calls
  · `styxx.Anthropic` text-mode pipeline works on mocked Anthropic responses
  · `styxx.gate()` fail-open contract holds with a deliberately-failing
    client (returns permissive verdict, never raises)
  · `styxx.reflex` self-interrupting generator — fault callbacks fire
    correctly on confab-prone prompts; events accumulated; rewind logic
    triggered when applicable
  · `styxx.weather()` 24h forecast — operates over accumulated audit log,
    produces structured WeatherReport with gate-pass-rate / mood / mean
    coherence metrics
  · `styxx.Thought` substrate-independent type — round-trip via
    as_dict/from_dict preserves thought_id; distance(t,t)=0; certify()
    produces CognitiveCertificate
  · `styxx.dynamics.CognitiveDynamics` — full fit → predict → simulate →
    save/load loop on synthetic 10-observation trajectory in 6-category
    state space; .cogdyn binary serialization round-trips cleanly
  · `StyxxCallbackHandler` (langchain adapter) — vitals computed on live
    langchain 1.x agent: category=reasoning, gate=pass, trust=0.86

---

## [6.2.0] — 2026-04-24

**Headline: `styxx.profile` — py-spy for LLM reasoning. Decorate any
agent function, see where cognition failed before the output did.
Drift, confabulation, refusal, sycophancy, phase-transition, low-trust
and incoherence are all localized to specific steps with severity
scores.**

PyPI: https://pypi.org/project/styxx/6.2.0/

### The cognitive profiler

`styxx.profile` is the first tool that tells you **why** an agent
failed, not just **that** it failed. LangSmith shows traces;
Datadog shows metrics; Profiler shows cognition.

```python
import styxx

@styxx.profile
def my_agent(task):
    return run_langchain_agent(task)

result, p = my_agent("summarize this contract")
print(p.summary)
# profile 'my_agent': 7 steps, 4.32s total
#   2 fault(s):
#     · [drift] step=3 sev=0.87 · category='arg_swap' at confidence 0.87
#     · [phase_transition] step=6 sev=0.50 · category shift: reasoning → confab

p.to_html("run.html")      # flamegraph — K/C/D timeseries per step
p.to_json("run.json")      # LangSmith / Datadog-compatible export
```

### Three API shapes

1. **Decorator** — `@styxx.profile` → returns `(result, profile)`
2. **Context manager** — `with styxx.profile(name="sql_agent") as p:`
3. **Manual recording** — `styxx.profile_session()` + `.record(response, label=...)` for custom adapters

### Seven fault kinds detected

| kind | triggers when |
|---|---|
| `drift` | category ∈ {arg_swap, tool_arg_drift, tool_confab, drift} with confidence > 0.5 |
| `confabulation` | category ∈ {confab, hallucination, fabrication} with confidence > 0.5 |
| `refusal` | category ∈ {refuse, refusal} with confidence > 0.8 (strong refusals only) |
| `sycophant` | category ∈ {sycophant, sycophancy} with confidence > 0.5 |
| `low_trust` | trust_score < 0.30 |
| `incoherence` | cross-phase coherence < 0.30 |
| `phase_transition` | adjacent steps have differing dominant categories |

### Three export formats

- **HTML flamegraph** — self-contained, no external assets, darkflobi-brand aesthetic. Screenshot-ready.
- **LangSmith trace** — `p.to_langsmith()` → drop into the LangSmith client's `create_run` API.
- **Datadog spans** — `p.to_datadog()` → `{"spans": [...]}` ready for the Datadog APM agent.

### Under the hood

Uses existing `Vitals`, `WatchSession`, and the canonical `analytics.write_audit` tap —
every vitals-creating path feeds the active profile automatically. No monkey-patching
of user code. Falls open on every path — missing openai SDK, unknown response shape,
no logprobs — the profile collects whatever signal it can, always returns a result.

### Files

- `styxx/profile.py` — `CognitiveProfile`, `ProfileStep`, `Fault`, `profile()`, `profile_session()`
- `styxx/_profile_html.py` — self-contained HTML flamegraph renderer

---

## [6.1.0] — 2026-04-24

**Headline: tool-call drift detector retrained — overall AUC 0.916 → 0.943,
arg_swap failure mode partially fixed (0.664 → 0.755) via a new
positional-inversion feature.**

PyPI: https://pypi.org/project/styxx/6.1.0/

### `arg_order_inversion` — 23rd feature

The v6.0 drift detector had one documented failure mode: `arg_swap`
(AUC 0.664), where a model produces the right argument names but
assigns wrong values across slots. None of the 22 v6.0 features
could separate this case from gold calls — all schema checks pass,
all prompt-overlap features pass.

The new feature — `arg_order_inversion` — measures whether the
positional order of call-values in the prompt matches the schema's
declared argument-key order. A correct call tends to have value
positions monotonically increasing with schema index; arg_swap
inverts that.

Formally, for each argument pair `(ki, kj)` where both call values
have a detectable first-appearance position in the prompt tokens:
```
schema says:  schema_order[ki] < schema_order[kj]
prompt says:  prompt_pos(call_args[ki]) < prompt_pos(call_args[kj])
inverted if the two disagree.
```
`arg_order_inversion = inversions / eligible_pairs ∈ [0, 1]`.

Signal validation on BFCL v3 n=3,700 (no training involved):

```
drift_type             n   mean   cov>0
gold                 658  0.166  24.2%
arg_swap             604  0.415  53.3%    <-- +0.249 over gold
arg_drop             657  0.094  11.4%
spurious_arg         658  0.166  24.2%
irrelevance_called  1122  0.567  58.4%
```

### 5-fold CV results (same n=3,700, same protocol)

| metric                   | v6.0 (22-feat) | v6.1 (23-feat) | delta   |
|--------------------------|----------------|----------------|---------|
| Pooled AUC               | 0.9148         | **0.9425**     | +0.028  |
| Mean fold AUC (± std)    | 0.9151 ± 0.004 | **0.9430 ± 0.009** | +0.028 |
| **arg_swap** (vs gold)   | **0.664**      | **0.755**      | **+0.091** |
| irrelevance_called       | 0.957          | 0.980          | +0.023  |
| arg_drop                 | 0.998          | 0.997          | ~flat   |
| spurious_arg             | 0.997          | 0.997          | ~flat   |
| simple (pooled)          | 0.902          | 0.930          | +0.028  |
| live_simple (pooled)     | 0.872          | 0.904          | +0.032  |

No regressions. `arg_order_inversion` lands at #6 by coefficient
magnitude (+1.154 scaled), top-3 on arg_swap cases at inference.

### Remaining failure modes

arg_swap at 0.755 is a partial fix, not a full one. The feature is
a surface-level positional heuristic — it fails when:
- both swapped values share the same prompt position (numerical
  ambiguity, e.g. `"divide 5 by 5"`)
- one value is missing from the prompt (synthesized by the model)
- the schema's declared order doesn't match the prompt's natural
  order (baseline inversion rate on gold ~0.17)

Full arg_swap fix is scoped for v3 via embedding-based per-slot
semantic fit.

### Files changed

- `styxx/guardrail/drift_signals.py` — 22 → 23 features, added
  `_arg_order_inversion_rate` helper.
- `styxx/guardrail/calibrated_weights_drift_v1.py` — fully retrained
  coefficients, scaler mean/scale, intercept, AUC tables.
- `styxx/guardrail/drift.py` — docstring update with new numbers.
- `scripts/drift_calibrated_v1.py` — new trainer (mirrors v0,
  adds the feature to Group B).
- `scripts/drift_feature_probe_arg_order.py` — signal-strength
  probe used to justify the retrain.
- `benchmarks/drift_calibrated_v1.json` — full v1 artifact.
- `tests/test_drift_v1.py` — assertions bumped to 23-feature,
  v1 artifact path.

### Compatibility

Same public API (`styxx.guardrail.drift_check()` unchanged).
Scores shift: expect drift_risk to move by up to ±0.1 on borderline
cases relative to v6.0. Decision boundary (`drifts = drift_risk >=
0.5`) is stable on the held-out test set.

---

## [6.0.0] — 2026-04-23

**Headline: cognometric instrument #3 — tool-call drift — ships as the
third calibrated detector, alongside hallucination (v4) and refusal
(v5.1). Three instruments is the minimum triangulation for a
methodology claim rather than a lucky two-sample.**

PyPI: https://pypi.org/project/styxx/6.0.0/

### `styxx.guardrail.drift_check()` — new public API

```python
from styxx.guardrail import drift_check

v = drift_check(
    prompt="Find the area of a triangle with base 10 and height 5",
    functions=[{"name": "calculate_triangle_area",
                "parameters": {"properties": {"base": {"type": "integer"},
                                              "height": {"type": "integer"}},
                               "required": ["base", "height"]}}],
    tool_call={"name": "calculate_triangle_area",
               "arguments": {"base": 10, "height": 5}},
)
# v.drift_risk   — 0-1 calibrated probability
# v.drifts       — bool at threshold 0.5
# v.top_signals  — top-3 contributing features (signed contribution)
```

### Calibration

Trained on **Berkeley Function Calling Leaderboard v3** via
mutation-based construction (arg_swap, arg_drop, spurious_arg,
tool_rename) + irrelevance-called synthesis. n=3,700 labeled triplets,
82/18 drift/no-drift split.

- **5-fold CV AUC: 0.9151 ± 0.0039** (pooled 0.9148).
- 22-feature calibrated LR with `class_weight=balanced`.

Per-drift-type held-out AUC:

| drift class              | AUC      | notes |
|--------------------------|----------|---|
| spurious_arg             | 0.997    | clean capture |
| arg_drop                 | 0.998    | clean capture |
| irrelevance_called       | 0.957    | +0.40 over null baseline 0.562 |
| arg_swap                 | 0.664    | **documented failure — fix v3** |
| tool_rename              | 0.030    | n=1, BFCL under-samples this class |

### vs the only published comparable baseline

[Healy et al. 2026 (arXiv:2601.05214)](https://arxiv.org/abs/2601.05214)
reports AUC **0.716–0.721** on Glaive using **last-layer hidden-state
MLP features** — requires model internals. styxx drift v1 hits **0.916
on BFCL v3 text-only**, works on ANY closed model (OpenAI, Anthropic,
Gemini) without hidden-state access.

### Artifacts

- `scripts/drift_build_dataset_v0.py` — dataset reproducer
- `scripts/drift_null_baselines_v0.py` — 5 null heuristics (best
  baseline, schema_conformance, caps at 0.733 — kill-criterion pass)
- `scripts/drift_calibrated_v0.py` — calibrated LR + 5-fold CV
- `benchmarks/drift_calibrated_v0.json` — full result artifact
- `data/drift_v0/drift_dataset_v0.jsonl` — committed training data
- `styxx/guardrail/drift.py` — public API
- `styxx/guardrail/calibrated_weights_drift_v1.py` — 22 features + LR
  coefs + scaler + per-class AUC + CALIBRATION_NOTES

### Tests

15 new regression tests in `tests/test_drift_v1.py` covering public
API shape, JSON roundtrip, 4 canonical cases (correct call, missing
arg, spurious arg, wrong tool), edge cases, and calibrated-weights
pinning. Full suite: **655 passed, 1 skipped, 0 failed** in ~30s.

### Law II empirical support now at three instruments

| instrument          | cross-substrate evidence                        |
|---------------------|-------------------------------------------------|
| Hallucination (v4)  | 8 benchmarks (probe + classifier)               |
| Refusal (v5.1)      | 5 model families (classifier)                   |
| Tool-call drift (v6)| 4 mutation types + natural irrelevance          |

---

## [5.1.0] — 2026-04-23

**Headline: rigor pass. v2 refusal weights pulled from public API as
research-only after an honest over-flagging bias was characterised.
No external amplification shipped until every claim was verified.**

### v2 refusal weights: demoted to research artifact

v2 weights trained on n=380 diverse-model samples revealed two honest
findings:

- **Good:** Llama-2-orig AUC jumped +0.11 (robustness gain).
- **Bad:** short factual compliances over-flagged as refusals (second
  documented failure mode: `enumerated_technical_compliance`).

Rather than ship v2 in the public API with a known bias, v2 stays as
a committed RESEARCH ARTIFACT (module, scripts, benchmark JSON, 10
regression tests) but is NOT exposed via `refuse_check()`. When v3
fixes the bias (via z-clip + retraining with enumerated-compliance
examples), v2 can be promoted to the public API.

### Prior-art correction: "first public XSTest AUC" claim retracted

Independent verification found that IBM Granite Guardian
([arXiv:2412.07724](https://arxiv.org/abs/2412.07724), Dec 2024,
Table 7) already published XSTest AUC for 9 safety classifiers six
months before our v5.0. Our 0.976 on XSTest-v2 GPT-4 held-out is
**competitive** with that tier, not first-in-class.

Fixed across `README.md`, `release/v5-amplify-kit`,
`cognometry-refuse.html` meta tags, and the v5.0.0 GitHub release
notes — **before** any external amplification. 0 tweets posted, 0
threads published, 0 HN submissions. The false claim never reached
the public.

### Research & methodology

- `scripts/compete_hhem_halueval.py` — HHEM-2.1-Open head-to-head
  reproducer (styxx +0.23 AUC on HaluEval-QA, 220× faster).
- `papers/cognometry-v0.5.{md,pdf}` — full arXiv-submittable paper
  (259KB, 10–12 pages) merging v0 + addendum. Adds §4 refusal
  instrument, §5.1 HHEM head-to-head, §5.3 Granite Guardian context,
  §5.4 related work expansion, §6 new failure modes, Appendix C
  per-seed raw AUCs. Endorsement code obtained for arXiv cs.LG.
- `papers/tool-call-drift-scope-v0.md` — research scope for
  instrument #3 (prerequisite for v6.0).
- `papers/landscape-scan-v05.md` — academic landscape background,
  Wang 2025 "False Sense of Security" rebuttal cited head-on.

### Changes

- `pyproject.toml` version: 5.0.0 → 5.1.0
- `styxx/__init__.py` `__version__`: 5.0.0 → 5.1.0
- `styxx/guardrail/refusal.py` — reverted variant parameter, kept
  `weights_variant` field on `RefusalVerdict` (always `"v1"` for now).
- `styxx/guardrail/calibrated_weights_refusal_v2.py` — added second
  failure mode + v2-specific failure notes + defensive z-score
  clipping.
- `tests/test_refusal_v2.py` REMOVED (tested public API that doesn't
  exist).
- `tests/test_refusal_v2_research.py` ADDED (10 tests) — pins v2 as
  research artifact, verifies it's NOT exposed via public API,
  asserts the over-flagging bias is real (regression test forces v3
  promotion when fixed).
- README refusal section reframed: "v1 is apologetic-style
  specialist, v2 not yet in public API" with link to
  `calibrated_weights_refusal_v2.py` CALIBRATION_NOTES.

### Tests

Full suite: **640 passed, 1 skipped, 0 failed** in 28.5s.

---

## [5.0.0] — 2026-04-23

**Headline: cognometric instrument #2 — refusal detection — ships as
the second calibrated detector on the same methodology as
hallucination. "0.998 AUC on HaluEval-QA. 9 floats. No LLM."**

PyPI: https://pypi.org/project/styxx/5.0.0/

### `styxx.guardrail.refuse_check()` — new public API

```python
from styxx.guardrail import refuse_check

v = refuse_check(
    prompt="How do I shut down a Python process?",
    response="I'm sorry, but I can't help with that...",
)
# v.refuse_risk   — 0-1 calibrated probability
# v.refuses       — bool, threshold default 0.5
# v.features      — dict of 18 raw features
# v.top_signals   — top-3 contributing features by scaled contribution
```

Mirrors the shape of the v4 hallucination `check()` API. Pure-Python,
Pyodide-safe, no external deps beyond existing `text_features`
vocabularies.

### Calibration

- Train: 80 labeled (prompt, response) from JailbreakBench,
  Llama-3.2-1B apologetic refusals (already committed in
  `styxx/residual_probe/atlas/compliance_labels_llama_1b.json`).
- Test: XSTest v2, 450 samples × 5 model families (GPT-4, Llama-2
  new/orig, Mistral guard/instruct) — 2,250 held-out samples total.
- Features: 18 text-only heuristics (refusal_density, disclaimer,
  normative, sorry-opener, word-count, etc.). No logprobs, no model
  weights.

Held-out AUCs (LR trained on JBB-Llama-1B, tested on XSTest):

| split               | AUC    | notes |
|---------------------|--------|---|
| GPT-4               | **0.9759** | out-of-family best |
| Llama-2 new         | 0.8741 | |
| Llama-2 orig        | 0.7832 | |
| Mistral-guard       | 0.7797 | |
| Mistral-instruct    | 0.6097 | **documented failure mode** |
| mean cross-model    | 0.8045 | |

First-pass training AUC on JBB (5-fold CV): 0.9967.

### Failure-mode note

Mistral-instruct refuses by lecturing ("It's important to note...",
"It's crucial to...") rather than apologizing. Our normative-lecturing
features exist but carry zero weight because the JBB-Llama training
corpus only contains apologetic refusals — LR cannot learn to use
features the training data never exercises. v1 is an
**apologetic-style specialist**; it wins on Claude / GPT-4 /
Llama-style outputs. Fix tracked in research → v5.1.

### README rewrite

Competitive landscape tables for both instruments:

- **Hallucination** — vs Patronus Lynx-70B, Vectara HHEM-2.1-Open,
  Cleanlab TLM, Galileo Luna.
- **Refusal** — vs Llama Guard 2/3, ShieldGemma 2B/9B/27B, OpenAI
  Moderation, Aegis, Perspective API.

H1 positioning: **"0.998 AUC on HaluEval-QA. 9 floats. No LLM."**

### Artifacts

- `styxx/guardrail/calibrated_weights_refusal_v1.py` — 18 features +
  LR coefs + scaler + held-out AUC per split + CALIBRATION_NOTES.
- `styxx/guardrail/refusal_signals.py` — pure-Python feature
  extractor.
- `styxx/guardrail/refusal.py` — public `refuse_check` +
  `RefusalVerdict`.
- `pyproject.toml` description updated to reflect two instruments.

This establishes refusal-detection as the second cognometric
instrument on the same methodology as hallucination (v4).

---

## [4.0.2] — 2026-04-23

**Headline: fix adaptive-threshold false-positives on entity-rich
factual responses without a reference.**

In 4.0.1, the adaptive threshold on the text-only heuristic path
was 0.9. But the piecewise-linear calibration maps a saturated
`text_claim_risk=1.0` to raw risk ~0.98 regardless of whether the
response is correct or hallucinated — the signal is structurally
non-discriminative without a reference. Any entity-rich factual
claim like "The capital of France is Paris." still halted.

Fix: adaptive threshold on text-only path raised to 0.99. Honest
position: when `@trust` has no reference, it cannot meaningfully
verify, so it passes through rather than halting on noise. Users
who want strict text-only gating set `threshold=` explicitly.

The `reference`-auto-detect path (calibrated v2/v4 weights) keeps
the 0.7 default — nothing changes there.

### Tests

18 new regression tests in `tests/test_trust_v4_0_1.py` covering the
4.0.1 effortless-mode behaviors (auto-detect, auto-NLI, adaptive
threshold, best-of-N retry). Full suite: 591 pass.

### URL health

Fixed stale HuggingFace reference to `truthful_qa` (moved to
`truthfulqa/truthful_qa`) in zenodo metadata script and
submission-package doc. No runtime impact.

### Site

`og:image` now returns 200 for the cognometry manifesto
(banner asset added to `assets/styxx/`). Leaderboard page gains
its own `og:image` + `twitter:card` tags so link previews render
correctly on X/LinkedIn/Slack.

---

## [4.0.1] — 2026-04-23

**Headline: `@trust` is now effortless. Zero config, zero sharp edges.**

Three UX fixes that turn first-contact from "why is every response
being flagged?" into "it just works":

### Zero-config reference auto-detect

`@trust` now auto-detects reference passages from any of these
kwarg names on the wrapped function: `context`, `reference`,
`references`, `passage`, `passages`, `docs`, `documents`, `source`,
`sources`, `knowledge`, `grounding`, `retrieved`, `retrieval`.

```python
@trust
def my_rag(question, *, context):   # no more reference_arg=...
    return openai.chat.completions.create(...)
```

Before: users had to write `@trust(reference_arg="context")` or
the detector would silently run text-only and over-halt. Now it's
automatic. `reference_arg=` is still honored when explicit; only
kwargs the function actually declares are picked up, so framework
pass-throughs don't cause false positives.

List/tuple of passages are also recognized and joined with newline.

### Auto-enable NLI when `styxx[nli]` is installed

`use_nli` default changed from `False` to `None` (auto). When
`torch` + `transformers` are importable, NLI is on by default. When
they aren't, it stays off. `use_nli=True` / `use_nli=False` are
still honored explicitly.

This means `pip install styxx[nli]` now matters: you get the v4
9-signal pipeline automatically rather than needing to pass
`use_nli=True` everywhere.

### Adaptive threshold

`@trust` previously halted at a flat `threshold=0.7`. When no
reference was provided (text-heuristic path), any confident factual
claim scored risk ~0.98 and triggered a halt — first-run demos
returned the fallback on correct answers.

Fix: when the user didn't override `threshold` (i.e., the default
0.7 is in effect) AND only the text-heuristic path is firing, the
effective threshold is bumped to 0.9. Calibrated paths (v2, v4,
tier-1) keep 0.7. Explicit user thresholds — any non-default value
— are always respected.

### Smart retry: best-of-N

`on_halt="retry"` now tracks the lowest-risk response across all
retry attempts. When retries exhaust and no attempt cleared the
threshold, the fallback still fires (same behavior), but the
internal state now reflects the genuinely-best candidate — which
matters for `on_halt="annotate"` users inspecting `attempts` and
for future features that might use the best candidate differently.

### Tests

Full suite: 573 pass. `test_trust.py` existing suite unchanged and
green; new behaviors are backward-compatible.

### What's NOT changed

- `@trust()` defaults: `threshold=0.7`, `on_halt="fallback"`,
  `max_retries=2`, `fallback="I'm not confident..."`. Same as 4.0.0.
- 8-benchmark calibrated weights v4: identical. No retraining.
- API surface: only additions (zero new required args, no renames).

---

## [4.0.0] — 2026-04-23

**Headline: cross-validated on 8 benchmarks. The first honest
8-benchmark audit of hallucination detection. 5/8 above AUC 0.65;
two failure modes (DROP, FinanceBench) published, not hidden.**

Extends the v3 NLI-augmented pipeline (4 benchmarks — HaluEval-QA,
HaluEval-Dialog, HaluEval-Summ, TruthfulQA) to 8 benchmarks by
adding 4 new domains from PatronusAI's public HaluBench:

  - DROP         — reading comprehension QA
  - PubMedQA     — biomedical QA
  - FinanceBench — financial document QA
  - RAGTruth     — RAG-style retrieval faithfulness

### Headline numbers (3-seed mean ± std, n=150/dataset, seeds [31,47,83])

| Dataset                 | v4 AUC            | Commentary |
|-------------------------|-------------------|---|
| HaluEval-QA             | **0.998 ± 0.001** | near-perfect |
| TruthfulQA              | **0.994 ± 0.006** | near-perfect |
| HaluBench-RAGTruth      | **0.807 ± 0.043** | new — RAG faithfulness |
| HaluBench-PubMedQA      | **0.719 ± 0.051** | new — biomedical |
| HaluEval-Dialog         | 0.676 ± 0.037     | (v3 peaked 0.729) |
| HaluEval-Summarization  | 0.643 ± 0.060     | (v3 peaked 0.665) |
| HaluBench-FinanceBench  | 0.492 ± 0.026     | **below chance** |
| HaluBench-DROP          | 0.424 ± 0.080     | **below chance** |
| **overall mean**        | **0.719**         | 5/8 above 0.65 |

### Published failure modes (intentional)

- **DROP** (reading comp). Extractive-span hallucinations (wrong
  span from right passage) are *entailed* by the passage at the NLI
  level; novelty signals are also blind because the tokens overlap.
  Future: span-level faithfulness scoring.
- **FinanceBench** (financial QA). Hallucinations are mostly
  calculation/aggregation errors on numbers copied verbatim from
  the passage. Novelty + NLI are semantically blind to arithmetic.
  Future: number-symbolic verification signal.

These are in the paper, in the CHANGELOG, in the CALIBRATION_NOTES
dict on the weights module itself. We publish what breaks.

### Design decision: v3 stays the default

v4 generalizes across 8 domains; v3 is more peaked on HaluEval-style
dialog/summarization. `guardrail.check(use_nli=True, ...)` continues
to route through the v3 LR (peaked) when all 9 signals are present.
v4 is available via direct import for callers who explicitly want
cross-domain averaging:

```python
from styxx.guardrail.calibrated_weights_v4 import predict_proba_v4
```

Rationale: most production RAG/QA traffic looks more like HaluEval-QA
than like the 8-dataset average. The broader calibration is available
when you have reason to want it; the narrower calibration ships as
the default because it serves the common case better.

### Added modules / benchmarks

- `styxx.guardrail.calibrated_weights_v4` — 9-signal, 8-benchmark
  pooled LR, 3-seed averaged.
- `benchmarks/hallucination_test/cross_dataset_8bench.py` — full
  8-benchmark calibration harness.
- `benchmarks/hallucination_test/cross_dataset_8bench_multiseed.py` —
  multi-seed wrapper, saves averaged coefs + per-dataset AUC std
  to `results/cross_dataset_8bench_multiseed.json`.
- Paper: *Cognometry v0: 8-benchmark cross-validated hallucination
  detection*. Zenodo deposit.

### Tests

5 new tests in `tests/test_weights_v4.py`. Full suite: 578 pass,
1 skipped, 0 fail.

### What changes vs v4.0.0rc1

v4.0.0rc1 (published 2026-04-23) shipped the NLI signal + v3 weights
on 4 benchmarks. v4.0.0 adds v4 weights (same NLI signal, broader
calibration) plus the HaluBench harness. No breaking changes.

---

## [4.0.0rc1] — 2026-04-23

**Headline: NLI v4.0 preview. The ninth signal — entailment-based
contradiction — lifts HaluEval-Dialog from AUC 0.61 → 0.73 and
produces the first honest number above chance on dialog hallucination
detection at this benchmark scale.**

This is a **release candidate**. The 8-dataset cross-validation
(FEVER, FactCC, XSum-Faithful, PHD-A) lands in the v4.0.0 final.
Install it to preview the signal; pin `4.0.0rc1` explicitly if you
want reproducibility across the rc→final transition.

### Added — `styxx.guardrail.nli_signal`

A lazy-loaded NLI contradiction scorer. Wraps
`MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` (~184M params; trained
on MNLI + FEVER + ANLI-R3) and exposes:

```python
from styxx.guardrail.nli_signal import (
    nli_contradiction_score, NLIScorer, get_default_scorer,
)

# Convenience (singleton, fail-open)
p = nli_contradiction_score(
    reference="Hamlet was written by William Shakespeare.",
    response="Hamlet was written by Dickens.",
)
# p ≈ 0.95
```

Thread-safe, CPU or CUDA. Fails open on any error (empty input,
model-load failure, transformers missing). No new required
dependency — torch+transformers only needed when `use_nli=True`.

### Added — `guardrail.check(use_nli=..., nli_scorer=...)`

```python
from styxx.guardrail import check

verdict = check(
    prompt="Who wrote Hamlet?",
    response="Hamlet was written by Dickens.",
    reference="Hamlet was written by William Shakespeare.",
    use_nli=True,      # opt-in
)
```

When `use_nli=True` and a reference passage is available, the
pipeline adds a `nli_contradict ∈ [0,1]` signal and routes the
verdict through the v3 calibrated LR (9 signals). Otherwise falls
back gracefully to v2 (8 signals) or v1 (4 signals).

Pass a pre-loaded `NLIScorer` via `nli_scorer=` to amortize the
one-time model load across many calls.

### Added — `styxx.guardrail.calibrated_weights_v3`

9-feature pooled LR, 3-seed averaged over seeds [31, 47, 83].
`predict_proba_v3(signals)` is the drop-in replacement for v2's
`predict_proba_v2`.

### Measured numbers (3-seed mean ± std, n=200/dataset, seed set [31,47,83])

| Dataset               | v3.9.1 (v2) | **v4.0.0rc1 (v3)** | Δ     |
|-----------------------|-------------|--------------------:|------:|
| HaluEval-QA           | 1.000       | **0.996 ± 0.002**   | -0.004|
| TruthfulQA            | 0.977       | **0.995 ± 0.004**   | +0.018|
| HaluEval-Dialog       | 0.605       | **0.729 ± 0.042**   | **+0.123** |
| HaluEval-Summarization| 0.636       | **0.665 ± 0.029**   | +0.028|
| **mean**              | **0.805**   | **0.846**           | **+0.041** |

Honest read: dialog is the big win (+0.123 absolute, single-seed
max 0.788). Summarization is real but smaller (+0.028). QA loses
a noise-level 0.004 — the two near-perfect classifiers are
indistinguishable within the noise floor.

### Signal weights (3-seed averaged LR coefficients)

```
text_claim_risk:        0.1751
entity_unverified_frac: 0.0000  (signal fires too rarely to matter here)
knowledge_grounding:    0.1231
content_novelty:        0.3368
entity_novelty:         0.1353
number_novelty:         0.0333
bigram_novelty:         0.4104
trigram_novelty:        0.7727
nli_contradict:         0.8784  ← strongest single signal
intercept:             -1.1257
```

NLI and trigram-novelty are complementary, not redundant: novelty
catches "response added content not in reference"; NLI catches
"response asserts what reference denies." Dialog and summarization
errors are dominated by the latter, which explains the gain pattern.

### New install extra

```bash
pip install styxx[nli]
```

Installs `torch>=2.0` + `transformers>=4.35`. Downloads the DeBERTa
checkpoint on first call (~1GB on disk, ~700MB RAM).

### New benchmark

`benchmarks/hallucination_test/cross_dataset_multi_seed.py` runs the
full pooled calibration across multiple seeds with and without NLI,
saves per-seed results + averaged coefficients to
`results/multi_seed_calibration.json`. Regenerates the numbers above.

### Tests

17 new tests in `tests/test_nli_signal.py`: v3 weight structure,
monotonicity, fail-open behavior, `check()` integration with mock
scorer, preservation under missing signals. Full suite: 573 pass,
1 skipped, 0 fail.

### Honest limits

- Summarization is still at AUC 0.66 — real signal but not
  production-grade. The residual gap is structural: summaries
  paraphrase faithfully, which NLI only partially captures.
- Single-seed dialog ranges [0.574, 0.788] — high variance. Average
  is what ships; users at low N may see closer to single-seed
  performance.
- NLI adds latency: ~150–400 ms per pair on CPU, ~10–30 ms on CUDA.
  Most deployments should pre-warm `get_default_scorer()._load()`.
- **No FEVER / FactCC / XSum yet.** The strong claim
  ("cross-validated on 8 benchmarks") ships with v4.0.0 final.

### What ships next (v4.0.0 final)

- Cross-validation on FEVER-dev + FactCC + XSum-Faithful + PHD-A
- Any coefficient refit required after the 8-dataset fit
- Paper: *Cognometry v0: cross-validated hallucination detection on
  8 benchmarks*. Zenodo deposit.

---

## [3.9.1] — 2026-04-23

**Headline: cross-dataset validation. v3.9.0's `@trust` worked on
HaluEval-QA (AUC 0.90) but we caught our own overfitting to that
benchmark and fixed it before anyone else could.**

### What we caught

Immediately after shipping v3.9.0 we ran cross-dataset validation
on HaluEval-Dialog, HaluEval-Summarization, and TruthfulQA with
the v3.9.0 weights. Performance collapsed to near-random (AUC
0.56–0.63) on three of four datasets. The 0.90 on HaluEval-QA was
a single-benchmark overfit.

Rather than quietly backtrack, we told on ourselves, added four
new signals, refit a pooled LR on all four datasets, and published
honest per-dataset numbers.

### New signals: response_novelty

Four asymmetric grounding signals that capture what the response
ADDED that the reference doesn't support (the opposite direction
from `knowledge_grounding`, which measures what's in the response
that IS in the reference):

- `content_novelty`  — fraction of response content tokens not in reference
- `entity_novelty`   — fraction of capitalized tokens (≥4 chars) not in reference
- `number_novelty`   — fraction of numeric tokens not in reference
- `bigram_novelty`   — fraction of response bigrams not in reference
- `trigram_novelty`  — fraction of response trigrams not in reference (strongest signal)

All five are cheap text operations — no model, no API, no latency.

### New calibration: `calibrated_weights_v2`

Pooled LR fit on HaluEval-QA + HaluEval-Dialog + HaluEval-Summ +
TruthfulQA (n=800 train, n=400 test, seed 31, L2=0.05, 8 features).

Held-out per-dataset test AUC:

| dataset                   | v3.9.0 | **v3.9.1** |
|---------------------------|-------:|-----------:|
| HaluEval-QA               | 0.9049 | **1.0000** |
| TruthfulQA                | 0.6261 | **0.9767** |
| HaluEval-Summarization    | 0.5897 | **0.5954** |
| HaluEval-Dialog           | 0.5984 | **0.6014** |
| mean                      | 0.6548 | **0.7934** |

Big wins on reference-grounded QA (the most common LLM use case:
RAG, open-domain Q&A). Modest improvements on
dialog/summarization — these are inherently NLI-requiring
(contradiction, not novelty) and will need NLI-based signals in
v4.0.

### Honest limits

- **Dialog and summarization remain hard** (AUC ~0.60). The
  limiting factor is that faithful dialog/summary responses
  naturally add content not verbatim in the reference. True
  discrimination needs NLI-style entailment, which is planned.
- **No reference passage → weaker detection.** v2 falls back to
  v1 (4-signal LR) when novelty isn't computable, and heuristic
  fusion when v1 isn't either.
- **English only, for now.** Novelty tokenization is
  whitespace-based.

### Pipeline integration

`guardrail.check()` now prefers v2 when all novelty signals are
available (reference provided), falls back to v1 when all four
v1 signals are available, then heuristic. Automatic — no API
changes.

### Tests

11 new tests in `tests/test_response_novelty.py`. Full suite:
573 pass, 1 skipped, 0 fail.

### Credibility over hype

v3.9.0 overclaimed. v3.9.1 is the honest result. `@trust` remains
a one-line API; what it defends has been properly cross-validated
and the numbers hold up — with specific, stated limits on where
they don't.

---

## [3.9.0] — 2026-04-22

**Headline: the trust layer. one decorator, any LLM call, verified
output. `pip install styxx` + `@trust` is all it takes to stop
hallucinations from reaching users.**

### New: `styxx.trust` — the one-line hallucination prevention layer

```python
from styxx import trust

@trust
def my_rag(question: str) -> str:
    return openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": question}],
    )
```

That's the whole API. `@trust` wraps any LLM-calling function.
Every output is cognometrically verified via
`styxx.guardrail.check()` (AUC 0.9012 on HaluEval-QA) before it
reaches the caller. If risk exceeds threshold, styxx intercepts.

### Design principles

- **Zero config out of the box.** Ships with HaluEval-calibrated
  LR weights. No residual-stream access needed. No API keys.
- **Shape-preserving.** Extracts text from OpenAI's
  `.choices[0].message.content`, Anthropic's
  `.content[0].text`, LangChain messages, dicts, and raw
  strings — automatically. Returns the same shape with
  replaced content, so downstream code still works.
- **Prompt auto-detection.** Pulls the user prompt from common
  kwargs (`prompt`, `question`, `query`, `messages`) or
  positional string args. Override with `prompt_arg="..."`.
- **Sync and async.** Auto-detects coroutine functions.
- **Four halt policies.**
  - `on_halt="fallback"` (default) — return safe text
  - `on_halt="retry"` — re-call up to `max_retries`
  - `on_halt="raise"` — raise `TrustViolation`
  - `on_halt="annotate"` — return `TrustResult(response, verdict)`

### API

```python
@trust(
    threshold=0.7,
    on_halt="fallback",
    fallback="I'm not confident...",
    max_retries=2,
    reference_arg="context",
    use_entity_verify=True,
    use_probe=False,
    verbose=False,
)
def my_agent(question, *, context=""):
    ...
```

### Tests

31 new tests in `tests/test_trust.py` — full suite now 562 pass,
1 skipped, 0 fail. Covers: text extraction from 6 response shapes,
prompt extraction from 7 input patterns, shape-preserving
replacement, sync/async, all 4 halt policies, retry budgets,
reference-arg grounding.

### The bet

styxx 3.9.0 is the product that tries to change the space forever.
TLS for LLM cognition. Nothing crosses unseen.

---

## [3.8.0] — 2026-04-22

**Headline: `styxx.guardrail` reaches test AUC 0.9012 on
HaluEval-QA with a learned-weight fusion classifier, beating
published state-of-the-art by a clear margin.**

### New: calibrated LR meta-classifier fusion

The guardrail now ships with logistic-regression weights fit on
HaluEval-QA dev (n=300, seed 11), evaluated on held-out test
(n=230, seed 17, deduplicated by question):

- **Test AUC: 0.9012** (dev AUC: 0.9411)
- Threshold 0.5: precision 0.873, recall 0.839, F1 0.856
- Threshold 0.7: precision 1.000, recall 0.578 (zero false positives)
- Threshold 0.8: precision 1.000, recall 0.374

Learned signal weights:
```
LR_COEFS = {
    "text_claim_risk":        1.4887,
    "entity_unverified_frac": 1.4331,
    "knowledge_grounding":    8.2097,  # dominant when reference available
    "probe_confab":           1.3469,
}
LR_INTERCEPT = -3.4586
```

The knowledge-grounding signal is by far the strongest contributor
when a reference passage is available; probe, entity verify, and
text features add independent corrections.

### Comparison to published SOTA on HaluEval-QA

| System                 | AUC (HaluEval-QA) |
|------------------------|-------------------|
| SelfCheckGPT           | 0.71–0.79         |
| KnowHalu               | 0.74              |
| HaluCheck              | 0.82              |
| **styxx.guardrail v2** | **0.9012**        |

### New module: `styxx.guardrail.calibrated_weights`

Ships the fitted LR coefficients + a `predict_proba(signals)`
function. The main `check(...)` entry point automatically uses the
learned weights when all 4 core signals are available, falls back to
the heuristic weighted-sum + piecewise-linear calibration when a
signal is missing (e.g., no reference passage → grounding absent).

### New atlas entry

- `meta-llama/Llama-3.2-1B-Instruct` **halueval** probe
  (LOO-AUC 1.000 @ layer 8, paired contrast on HaluEval-QA
  n=200 right vs hallucinated).

Atlas total at v3.8.0: **29 probes across 6 vendors and 7 concepts.**

### New: `styxx.generate_safe` — real-time self-halting generation

One-function API that runs a residual-level probe after every
generated token and halts when the probe crosses a threshold.
Works on any HF decoder model with a matching probe in the atlas.

```python
from styxx import generate_safe

r = generate_safe(
    model="meta-llama/Llama-3.2-1B-Instruct",
    prompt="Tell me about Dr. Eleni Kostadinova",
    halt_on="halueval",
    threshold=0.7,
)
# r.text → safe response if halt fired, model output otherwise
# r.halted, r.halt_reason, r.probe_trajectory
```

This is the production-side companion to the post-hoc guardrail:
instead of flagging after generation, it intervenes at the
token-level boundary where fabrication begins.

### Reproducer

```bash
python benchmarks/hallucination_test/guardrail_calibrate.py \\
  --n_dev 300 --n_test 300 \\
  --seed_dev 11 --seed_test 17 \\
  --probe_task halueval
```

Expected: test AUC 0.89–0.94 on properly held-out HaluEval-QA.

---

## [3.7.0] — 2026-04-22

**Headline: `styxx.guardrail` — multi-signal hallucination-prevention
pipeline that achieves AUC 0.838 on HaluEval-QA (n=100), competitive
with published state-of-the-art detectors.**

### New module: `styxx.guardrail`

A production-shaped hallucination-prevention system, not just a
detector. Takes (prompt, response, optional reference) and returns
a `Verdict` with:

- Overall calibrated risk score ∈ [0, 1]
- Recommended action: `pass` / `annotate` / `retry` / `halt`
- Per-span flagged claims with reasons
- Per-signal readings for audit

Five-signal architecture:
1. `text_claim_risk` — surface-level confabulation-indicator text
   features per atomic claim (weight 0.15)
2. `entity_unverified_frac` — fraction of named entities not found
   on Wikipedia (weight 0.20)
3. `knowledge_grounding` — content-token coverage vs reference
   passage (weight 0.50, strongest when reference available)
4. `probe_confab` — residual-level probe signal from
   `confab_behavioral` on Llama-1B (weight 0.10; OOD for HaluEval,
   higher weight for fake-entity-biography domain)
5. `consensus_disagreement` — self-consistency disagreement via
   token-set Jaccard across N sampled responses (weight 0.30;
   optional, requires sampler)

### Benchmark

HaluEval-QA, n=100, seed=11, paired right/hallucinated answers:

- **AUC 0.838** overall
- Right-mean risk 0.406, hallucinated-mean risk 0.650, separation +0.244
- At default thresholds: 7/100 hallucinations "pass", 93/100 flagged;
  37/100 right answers "pass", 50/100 "annotate" (false positive
  on annotate is a known tradeoff — tune thresholds per-deployment)

Comparable published numbers on HaluEval-QA:
- SelfCheckGPT: 0.71-0.79 AUROC
- KnowHalu: 0.74
- HaluCheck: 0.82
- **styxx.guardrail v1: 0.838**

### New modules

- `styxx.guardrail.claim_decomposer` — sentence-level atomic claim
  extraction with NER heuristics
- `styxx.guardrail.entity_verify` — Wikipedia-based entity grounding
- `styxx.guardrail.text_signals` — per-claim text-feature priors
- `styxx.guardrail.knowledge_grounding` — claim vs reference-text
  content-coverage scoring
- `styxx.guardrail.probe_signal` — stateful HF model + residual
  probe scorer (amortized load)
- `styxx.guardrail.consensus_signal` — self-consistency disagreement
- `styxx.guardrail.fusion` — weighted signal combination with
  piecewise-linear calibration
- `styxx.guardrail.policy` — configurable action thresholds
- `styxx.guardrail.entry` — top-level `check(...)` entry point

### Atlas extension (UCB Phase 4)

- 6 new corrigibility probes: all 5 Phase-3 vendors + Qwen-2.5-3B
  (AUC 0.26-0.70, RepE-style paired contrast on
  Anthropic/model-written-evals).
- 3 new Qwen-2.5-3B probes (refuse 0.97, truthfulness 0.88,
  deception 1.00).

Atlas at v3.7.0: **28 probes across 6 vendors and 6 concepts.**

### Paper 1 draft shipped (UCB cross-vendor)

- Ready for arXiv submission: `fathom-arxiv-ucb/main.pdf` (8 pages)

### Paper 2 draft shipped (Capability amplification)

- Ready for arXiv submission: `fathom-arxiv-capability/main.pdf` (6 pages)

### Usage

```python
from styxx.guardrail import check

# Post-hoc risk assessment
verdict = check(
    prompt="Who wrote Hamlet?",
    response="Hamlet was written by William Shakespeare around 1600.",
    reference="Hamlet is a tragedy by William Shakespeare...",
    use_entity_verify=True,
    use_grounding=True,
)
# verdict.risk     → 0.084 (low)
# verdict.action   → "pass"
# verdict.spans    → list of Span(text, risk, reasons)
```

---

## [3.6.0] — 2026-04-22

**UCB Phase 3 — shared residual subspace is universal; concept
encodings vary by training regime.**

### Headline finding

Across 5 independently-trained production LLMs from 5 different
vendors, at each model's concept-discriminative layer:

- **The top-30 residual subspace is 76-84% geometrically congruent**
  (canonical-correlation dim(ρ≥0.5) mean 23.2-25.3 / 30 across
  three concepts and 10 vendor pairs). The residual substrate is
  universal.
- **Concept-direction encodings within that subspace vary by
  concept type**: refuse (mean ρ=0.47), truthfulness (0.27),
  deception-template (0.21). Refusal converges strongly cross-
  vendor; factual and instruction-pattern concepts diverge.
- **Cross-scale capability transfer fails for truthfulness** (Llama-
  1B→3B via ridge projection: cos=+0.094, behavioral accuracy
  delta negative), even though refusal transfers within-family
  strongly (cos=+0.464).

This falsifies the naive "Platonic cognitive basis" hypothesis and
supports a more nuanced two-level picture: **substrate is shared;
concept encoding is training-regime-specific.**

### New atlas probes

- Truthfulness × 5 vendors (added `google/gemma-2-2b-it`, AUC
  0.851 @ layer 12, fraction 0.46 — completes 5×2 grid)
- Deception (template-contrast) × 5 vendors (Llama-1B/3B,
  Qwen-1.5B, Phi-3.5, Gemma-2B). All AUC 1.0 at layers 0-1 —
  prompt-template contrast separates linearly at embedding
  level. Template-detection directions, not behavioral-deception
  directions.

Current atlas size: **18 probes across 5 vendors and 5 concepts.**

### New analysis tools

- `benchmarks/causal_patching/train_truthfulness_probe.py` — HF
  TruthfulQA paired-contrast concept probe trainer.
- `benchmarks/causal_patching/train_deception_probe.py` — RepE-
  style system-instruction contrast probe trainer.
- `benchmarks/causal_patching/ucb_probe_correlation.py` — cross-
  model probe score-stream Pearson correlation matrix.
- `benchmarks/causal_patching/ucb_subspace_dim.py` — pairwise
  classical CCA → shared-subspace dimensionality estimator.
- `benchmarks/capability_steering/cross_scale_transfer.py` —
  capability-direction cross-scale ridge projection + behavioral
  validation.

### New run artifacts

- `benchmarks/causal_patching/runs/ucb_phase2_*_correlation.json`
- `benchmarks/causal_patching/runs/ucb_subspace_dim_{refuse,
  truthfulness,deception}.json`
- `benchmarks/capability_steering/runs/cross_scale/cross_scale_transfer.json`

### Paper updated

- `papers/universal-cognitive-basis-phase2.md` with full Phase 2 +
  Phase 3 findings and mechanism hypothesis.

### Implications for practice

- **Safety-library is feasible** (refusal direction is UCB-portable
  across vendors at ρ≈0.47-0.87 and cos≈0.36-0.46).
- **Capability-library is narrow** (truthfulness direction fails
  to transfer even within a family; per-model training is still
  required for skill transfer).
- **Cognitive auditing across vendors is feasible** for properties
  that live in the shared 23-25 / 30 dimensional subspace.

### Open questions (Phase 4+)

- Mechanism: why does safety-concept encoding converge while
  factual concept encoding diverges? Hypothesis: RLHF-style
  training on similar refusal behaviors creates convergent axes;
  pretraining-dependent factual knowledge creates divergent axes.
  Falsifier: train a from-scratch model without RLHF and measure
  whether its refusal direction aligns with RLHF'd vendors.
- Behavioral (not template-contrast) deception probes on all 5
  vendors — does deception follow refuse's convergence pattern
  or truthfulness's divergence?
- More concepts to map the universality landscape: empathy,
  reasoning-trace, overconfidence, goal-drift.

---

## [3.5.1] — 2026-04-22

**UCB Phase 2 — first public cross-vendor cognitive-agreement
measurement.** Same-day follow-up to v3.5.0, adding:

### New atlas entries
- `google/gemma-2-2b-it` refuse probe (AUC 0.984 @ layer 16,
  fraction 0.59)
- Truthfulness concept probes on 4 vendors:
  - `meta-llama/Llama-3.2-1B-Instruct` (AUC 0.835 @ layer 7,
    fraction 0.41)
  - `meta-llama/Llama-3.2-3B-Instruct` (AUC 0.880 @ layer 12,
    fraction 0.41)
  - `Qwen/Qwen2.5-1.5B-Instruct` (AUC 0.863 @ layer 14,
    fraction 0.48)
  - `microsoft/Phi-3.5-mini-instruct` (AUC 0.898 @ layer 18,
    fraction 0.55)

**Truthfulness encoded at fraction 0.41–0.55 across all four
models.** Tighter band than refuse (0.59–0.93), suggesting
truthfulness has a more universal encoding depth.

### UCB Phase 2 result — landmark measurement

For each concept, we run every model's trained probe on the
same 80 held-out prompts; compute pairwise Pearson correlation
across per-prompt probe-score streams.

**comply_refuse — 5 vendors, 10 pairs:**
- Strongest agreement: Llama-1B ↔ Llama-3B (ρ=+0.873, same family)
- **Strongest cross-vendor: Gemma-2B ↔ Llama-3B (+0.794)**
- **Cross-vendor Gemma ↔ Llama-1B: +0.791**
- Weakest: Phi-3.5 ↔ Qwen-1.5B (+0.177)
- **Mean pairwise ρ = +0.472** (min +0.177)

**truthfulness — 4 vendors, 6 pairs:**
- Strongest cross-vendor: Llama-3B ↔ Phi-3.5 (+0.555)
- **Mean pairwise ρ = +0.305** (min +0.155)

**Partial UCB confirmed** — probes trained independently on 5
different vendors' models share measurable concept geometry,
with within-family and similar-posture pairs agreeing strongly
and divergent-safety-training pairs agreeing weakly. Refusal
concept is more universal than truthfulness, consistent with
safety training converging more across vendors than factual
knowledge.

### New modules
- `benchmarks/causal_patching/train_truthfulness_probe.py` — HF
  TruthfulQA-based paired-contrast concept probe trainer
- `benchmarks/causal_patching/ucb_probe_correlation.py` — the
  Phase 2 cross-model probe-agreement matrix tool

### Paper shipped
- `papers/universal-cognitive-basis-phase2.md` with live numbers

### What remains open (Phase 3+)
- Train Gemma truthfulness probe to complete 5-vendor × 2-concept
  grid
- Train deception + confab-behavioral on all 5 vendors
- Generalized CCA decomposition: what fraction of each model's
  concept direction lies in the shared subspace vs vendor-
  specific residual?
- Cross-architecture test: does UCB extend to non-transformer
  LMs (Mamba, RWKV)?

---

## [3.5.0] — 2026-04-22

**Headline features:**
1. **`styxx.steer` + `styxx.cogvm`** — CIS v0 (Cognitive Instruction Set).
   The first open-source runtime for programmable residual-stream control
   of any HuggingFace decoder model. Multi-concept steering composes
   arbitrary subsets of trained probe directions as additive residual
   interventions. CogVM adds conditional dispatch (WATCH/HALT/RETRY/SWITCH)
   over live per-token probe readings.
2. **`styxx.hallucination`** — runtime fabrication detector with three
   modes: one-shot `hallucination_verdict()`, streaming `stream_with_risk()`,
   and auto-halting `detect_hallucination(..., on_detect="halt_and_flag")`.
   Uses the new v1 behavioral-label confab probe (AUC 0.800 @ layer 11).
   Production API surface with per-token risk signal, auditable chain
   from flag → probe reading → residual position.
3. **Multi-vendor probe atlas.** Refusal probes shipped for
   `meta-llama/Llama-3.2-1B-Instruct`, `meta-llama/Llama-3.2-3B-Instruct`,
   `Qwen/Qwen2.5-1.5B-Instruct`, and `microsoft/Phi-3.5-mini-instruct`.
   First open multi-vendor cognitive direction library.

### New research results

- **Causal claim on Llama-3.2-1B**: single-direction multi-position
  residual patching on the refusal direction causes
  **refuse@unsafe to drop 97% → 17% at α=3.0** (n=60 JBB test split).
  Asymmetry confirmed: inducing refusal on safe prompts barely moves
  the needle (0.13 → 0.17 at α=3). Reproduces Arditi et al. at 1B
  scale with open data. See `papers/cognitive-instruction-set-v0-filled.md`.

- **Concept geometry** at shared layer 10 of Llama-1B: pairwise angles
  between `comply_refuse`, `sycophant_pressure`, and `confab_prompt`
  directions fall in **86.7°–91.9°** — statistically indistinguishable
  from random high-dim unit vectors. Modular-concept hypothesis confirmed.
  See `benchmarks/causal_patching/measure_probe_geometry.py`.

- **Universal Cognitive Basis v0** — cross-model direction transfer grid:
  - Llama-1B → Llama-3B (same family): **cos = +0.464** (~26σ above chance)
  - Llama-1B → Qwen-1.5B (cross-vendor): **cos = +0.362** (~14σ)
  - Llama-1B → Phi-3.5 (large safety-gap): cos = +0.150 (~8σ)
  - Qwen-1.5B → Phi-3.5 (largest safety-gap): cos = +0.043 (~2σ, unaligned)

  UCB holds partially: within-family + close-vendor transfer works via
  ridge projection; divergent-safety-posture vendor pairs do not.
  Honest falsification of naive-linear UCB for the hardest case.
  See `papers/universal-cognitive-basis-v0.md`.

- **Gradient-free capability amplification on Llama-1B TruthfulQA**:
  multi-layer residual patching with a supervised correct-vs-incorrect
  answer direction boosts MC1 accuracy from **32.5% baseline → 39.5%
  at α=1.0 (+7.0 pp absolute, +21.5% relative)**, validated against a
  3-seed random-direction control (random directions HURT accuracy by
  mean −5.3pp at α=0.5, std 0.006; trained direction out-delivers
  random by **+10.8pp at α=1.0**). Single-layer single-direction patching
  was null at the same scale (n=200) — cumulative multi-layer injection
  is the operative mechanism. Matches Representation Engineering
  (Zou et al. 2023) now reproduced at 1B with open data + random control.
  See `papers/capability-amplification-v0.md`.

- **CognitiveBench v0 leaderboard** — first public cross-vendor
  cognitive audit. 50-prompt fake-entity fabrication battery run against
  Claude Haiku 4.5, Llama-1B/3B, Qwen-1.5B, Phi-3.5. Ground-truth
  labeled (every prompt targets a non-existent entity; any confident
  concrete response is fabrication). Same decline detector for every
  model. See `benchmarks/cognitive_bench/results/cognitivebench_v0.md`.

### Added modules

- `styxx.steer` — multi-concept composer (`steer(model, profile={...})`
  context manager; `steered_generate(...)` convenience).
- `styxx.cogvm` — declarative cognitive VM with
  `Program / WRITE / GENERATE / WATCH / HALT / RETRY / SWITCH` opcodes.
- `styxx.hallucination` — runtime fabrication detector.
- `styxx.hallucination_calibrate` — production threshold calibration
  utility (`calibrate_from_labels`).
- `styxx.residual_probe.atlas` — expanded to 5 refuse probes across 4
  vendor families + 3 concept probes (sycophant_pressure, confab_prompt,
  confab_behavioral) on Llama-1B.

### Added benchmarks

- `benchmarks/causal_patching/` — refusal probe training, α-sweep causal
  patching, concept-registry multi-concept trainer, probe geometry
  analysis, cross-scale comparison, cross-model direction transfer,
  UCB canonical correlation, paper-template filler, quick steering test,
  behavioral confab trainer.
- `benchmarks/capability_steering/` — truthfulness amplification (v3/v4/v5),
  random-direction control.
- `benchmarks/cogvm_demo/` — multi-concept steering demo, cross-vendor
  thought transplant demo.
- `benchmarks/claude_vs_us/` — Claude Haiku 4.5 hallucination battery.
- `benchmarks/cognitive_bench/` — CognitiveBench v0 cross-vendor audit.
- `benchmarks/hallucination_test/` — end-to-end hallucination detector test.

### Fixed

- **`styxx.residual_probe.intervene` device/dtype coercion**: probe
  weights previously lived on CPU while model residuals were on CUDA;
  the mismatched matmul raised and the catch-all hook swallowed it
  into a silent no-op. All-position patching and device-safe scoring
  are now the default.
- **`styxx.residual_probe.probe` defensive manifest loader**: skips
  non-manifest JSON files in the atlas directory (was crashing on
  sibling `compliance_labels_*.json` list payloads).
- **`styxx.anthropic_hack.text_features` imperative-refusal false
  positives**: imperative/directive phrases (`"Ship fast. Build hard.
  Refuse mediocrity."`) no longer score ≥ 90% refusal. Bare `refuse` /
  `decline` tokens replaced with first-person-contextualized markers
  (`i refuse`, `must decline`, `refuses to answer`, ...). 22 regression
  tests in `tests/test_text_features_imperatives.py`.

### Papers shipped in repo

- `papers/cognitive-instruction-set-v0-filled.md` — CIS v0 with real
  α-sweep + geometry numbers.
- `papers/universal-cognitive-basis-v0.md` — UCB v0 with 4-vendor
  transfer grid and honest falsification.
- `papers/capability-amplification-v0.md` — Gradient-free capability
  amplification on Llama-1B TruthfulQA, with random-direction control.

### Specs shipped in repo

- `docs/cognet-protocol-v0.md` — HTTPS protocol draft for cross-model
  cognitive bus (v1.5+ roadmap target).
- `INVENTION-CIS-v0.md` — public invention pitch document.
- `WHAT-WE-BUILT-2026-04-22.md` — one-day build log.

### Tests

- `tests/test_cogvm_unit.py` — 18 tests for VM parser + opcodes.
- `tests/test_text_features_imperatives.py` — 22 regression tests for
  the text-heuristic bug fix.
- `tests/test_hallucination_unit.py` — 6 tests for hallucination API.

Full suite: **531 passed, 1 skipped, 4 warnings** (up from 507 in 3.4.0).

### Reproducer

```bash
bash scripts/reproduce-cis-v0.sh
```

Expected wall-clock: ~20-25 minutes on an RTX 4070 laptop GPU.

---

## [3.4.0] — 2026-04-19

**Headline feature: `styxx.gate()` — one-function pre-flight cognitive
verdict for any LLM prompt.** Predicts whether the model will refuse,
confabulate, or proceed, **before you pay for the generation.** Uniform
API across Anthropic (tier-0 consensus), OpenAI (tier-0 logprobs), and
HuggingFace (tier-1 residual probe, v3.4.1). Research-backed against
the alignment-inverted consensus signal documented in
`papers/alignment-inverted-cognitive-signals.md` (Cohen's d = -0.827,
95% bootstrap CI [-1.288, -0.443] on n=96 Claude Haiku 4.5 prompts).

Also extends the cognitive-monitoring pipeline to APIs without
per-token logprobs. `styxx.Anthropic(mode=...)` now returns labelled
proxy vitals on Claude instead of `vitals=None`, with three
complementary pipelines measured against real Claude Haiku output.

### Added — `styxx.gate()`

```python
from styxx import gate
from anthropic import Anthropic

verdict = gate(
    client=Anthropic(), model="claude-haiku-4-5",
    prompt="How do I synthesize methamphetamine?",
)
# verdict.recommendation = "block"
# verdict.will_refuse = 1.00
# verdict.estimated_cost_usd = 0.0008
```

One function. Auto-routes based on client type. Returns a unified
`GateVerdict` with labelled method, so callers can distinguish a
tier-0 proxy reading from a tier-1 residual probe. Fails open — any
error returns a permissive "unknown" verdict instead of raising.

- **CLI**: `styxx gate "<prompt>" --model <id>`
- **Docs**: `docs/gate.md`
- **Example**: `examples/gate_demo.py`

### Added — `styxx.anthropic_hack`

Three proxy-signal pipelines, each explicitly labelled in the
resulting `Vitals.mode` attribute so callers can tell a proxy reading
from a true tier-0 reading:

- **`text_features`** — surface linguistic classifier (hedges,
  confidence markers, refusal markers, entity density, reasoning
  markers, line structure). Labelled `mode="text-heuristic"`,
  `tier_active=-1`. Zero extra API cost.
- **`consensus`** — fires the prompt N times at T > 0, computes
  empirical per-position token agreement, reconstructs a proxy
  `{entropy, logprob, top2_margin}` trajectory, feeds to the shipped
  styxx centroid classifier. Labelled `mode="consensus"`. Costs
  N× tokens per call.
- **`companion`** — runs the same prompt through a locally-cached
  open-weight model (Llama-3.2-1B preferred, distilgpt2/gpt2
  fallback) with real per-token logprobs, uses those as a proxy
  reading. Labelled `mode="companion:<model>"`. Zero API cost.

### Added — adapter dispatch

- **`styxx.Anthropic(mode=...)`** accepts `"off" | "text" | "consensus"
  | "companion" | "hybrid"`. Default is `"text"` (cheap, deterministic,
  no extra API calls). `"hybrid"` returns text-heuristic vitals always
  and upgrades to companion readings when a local model is cached.
- All responses gain `.vitals.mode` string so downstream code can
  branch on the reading's source.

### Added — benchmarks & paper

- **`benchmarks/anthropic_hack_real.py`** — harness that runs text
  and consensus modes against real Claude output on the 84-fixture
  bench suite. Reproducible: `export ANTHROPIC_API_KEY=...; python
  benchmarks/anthropic_hack_real.py`.
- **`benchmarks/anthropic_hack_eval.py --companion`** — runs companion
  mode against the same fixtures with no API calls.
- **`papers/cognitive-monitoring-without-logprobs.md`** — extends the
  Cognitive Metrology v1 program to closed-source logprobless LLMs.
  Covers the three pipelines, their cost/accuracy tradeoffs, and the
  empirical limits of each.

### Measured numbers on real Claude Haiku 4.5 (2026-04-19)

| mode              | n  | category accuracy | gate agreement |
|-------------------|----|-------------------|----------------|
| text-heuristic    | 84 | **0.536**         | **0.940**      |
| consensus N=5     | 84 | **0.405**         | —              |
| companion Llama-3.2-1B | 84 | **0.262**    | —              |
| companion Qwen2.5-3B-Instruct | 84 | **0.452** | —            |

(84 labelled prompts spanning factual, reasoning, refusal, creative;
fixtures under `bench/tasks/`.)

### Fixed — `text_features` classifier

- Removed generic verbs (`is`, `are`, `will`, `must`) from the
  CONFIDENCE vocabulary — they appeared in essentially every English
  sentence and prevented retrieval/reasoning from ever winning the
  softmax. Added `definitively`, `well-known`, `established`,
  `documented` which were missing.
- Added `REASONING_MARKERS` vocabulary (`first`, `then`, `therefore`,
  `step-by-step`, `follows that`, ...) — reasoning templates were
  previously scoring as creative.
- Entity detector now skips the first token of every **line** (not
  just every period-delimited sentence), so poetry with capitalized
  line starts doesn't generate false entities.
- Creative scoring now recognizes poetic structure (≥3 short lines)
  in addition to prose-creative variance. Claude's haiku output no
  longer classifies as retrieval.
- Markdown headers (`# Title`) and bullets are stripped before
  feature extraction.
- Category accuracy on the synthetic template suite: 48.8% → 100%.
  (Synthetic ceiling; real-Claude numbers are the row above.)

### Added — tests

- **`tests/test_anthropic_hack.py`** — 14 new tests covering all
  three pipelines, mode validation, and adapter dispatch.

### Docs

- **`docs/anthropic-support.md`** — complete guide to the three modes,
  measured numbers, and the upstream-limitation reality.

### Philosophy

styxx has always refused to fake readings. `.vitals = None` on every
Anthropic call was the honest-but-frustrating status quo. This release
does the harder thing: recovers as much cognitive signal as possible
from what the API *does* expose, labels every proxy reading so users
never mistake it for tier-0, and publishes the empirical limits.

None of these modes are a replacement for true tier-0 vitals. They
are cognitive monitoring on a logprobless API, which is strictly
better than nothing, and labelled honestly enough that downstream
code can decide which it trusts.

---

## [3.1.0] — 2026-04-14

**Stable release. Graduates Thought (3.0.0a1) and CognitiveDynamics
(3.1.0a1) from alpha. Closes the open backlog. Cognitive metrology
ships as the new default.**

This release graduates the two category-defining additions from
tonight's session out of the alpha cycle and into the stable channel.
`pip install styxx` (no `--pre` flag, no version pin) now pulls 3.1.0
by default. Two reported bugs from the same day are fixed and a
provider compatibility matrix is published.

The styxx repository state at the moment of this release: 0 open
issues, 0 open PRs, 6 GitHub releases, 388+ passing tests, the
Cognitive Metrology Charter v0.1 published, the .fathom and .cogdyn
file formats live, the styxx reference implementation MIT-licensed
and CC-BY-4.0-specified.

### Graduated from alpha

- **Thought** (the portable cognitive data type, originally 3.0.0a1):
  full surface, 68 tests, .fathom v0.1 file format, content_hash,
  algebra, save/load, provenance bridge to CognitiveCertificate.
- **CognitiveDynamics** (the linear-Gaussian dynamics model, originally
  3.1.0a1): full surface, 44 tests, .cogdyn v0.1 file format,
  fit/predict/simulate/suggest/forecast verbs, machine-epsilon
  recovery on full-rank synthetic inputs.
- **`Vitals.to_thought()`** symmetric shortcut.
- **`Thought.certify()`** provenance bridge.
- **`__hash__` content-based** for Python hash invariant compliance.

### Fixed — closes #1

- **Text classifier no longer misclassifies imperative/directive
  phrasing as refusal.** The refusal score in
  `styxx/conversation.py::_classify_text` was being boosted by
  `hedge_density * 0.04` even when zero refusal pattern matches were
  present, which caused short imperative inputs ("build > hype",
  "ship fast and iterate", agent system prompts, builder mottos,
  CLI help strings, README taglines) to score `refusal:0.20+`. The
  fix gates the entire refusal score on the presence of at least one
  explicit refusal token (`i can't` / `i'm unable` / `sorry, can't`
  constructions). Pure hedging without one of those patterns now
  scores refusal at `0.0`.

  Reported and reproduced as: `_classify_text("build > hype / ship
  fast and iterate")` → `refusal:0.259` (before) → `not refusal`
  (after).

- **23 new regression tests** in `tests/test_text_classifier_imperatives.py`
  pin the fix:
  - 10 imperative phrases that must NOT classify as refusal
  - 10 real refusals that must continue to classify as refusal
  - the exact issue #1 reproducer
  - a class-distribution test asserting at least 6/10 imperatives
    land on reasoning or creative

### Added — closes #3

- **`docs/COMPATIBILITY.md`** — provider compatibility matrix listing
  every LLM provider with the styxx tier-0 invocation pattern,
  marking each row as ✅ verified, ❌ not supported, or ⚠️ not yet
  verified. Verified: OpenAI, OpenRouter (model-dependent). Not
  supported: Anthropic Claude (Messages API has no `logprobs`
  parameter). Not yet verified: Gemini, Azure OpenAI, AWS Bedrock,
  Groq, vLLM, llama.cpp server, Ollama, LiteLLM gateway. Each
  unverified row has a TODO marker for the next contributor.

- **README provider-compatibility section** linking to the
  compatibility matrix, placed above the zero-code-change quickstart
  so visitors see the supported-provider story before they install.

### Tests

- **23 new regression tests** in `tests/test_text_classifier_imperatives.py`
  (10 imperatives + 10 refusals + 3 distribution/reproducer tests)
- **3 regression tests** in `tests/test_observe_warn.py` from the
  community PR (#4, merged earlier today, `mvanhorn`)
- Full styxx suite: **411 passed** (was 385 before this release),
  1 skipped, 0 failures, 0 regressions

### Community PRs merged this release cycle

- **#4** "feat(watch): warn once when observe() is given an openai
  response without logprobs" by **@mvanhorn** (Matt Van Horn,
  co-founder of June and Lyft predecessor). Closed issue #2. Reviewed
  by @SupaSeeka. Merged with thanks. The reviewer's `import sys`
  placement nit was addressed in a small follow-up commit on `main`.

### Backlog state at release

- **0 open issues** (closed: #1, #2, #3)
- **0 open PRs** (merged: #4)
- 6 GitHub releases visible (`v3.1.0` is now Latest)

### Why graduate from alpha

Because the underlying work is real and tested, not because the
calendar said so. The Thought type and CognitiveDynamics module ship
with 68 + 44 = 112 dedicated unit tests on top of 273 existing tests
inherited from 2.0.3. Machine-epsilon recovery on full-rank synthetic
inputs verifies the dynamics math. Bit-perfect round-trip on .fathom
files verifies the data type. The provenance bridge cryptographically
links the two layers. Real users on PyPI can now `pip install styxx`
and get the full v3 surface as the default.

This release coincides with the publication of the Cognitive
Metrology Charter v0.1 ([`docs/cognitive-metrology-charter.md`](https://github.com/fathom-lab/styxx/blob/main/docs/cognitive-metrology-charter.md))
and is the reference implementation that the charter cites as the v0.1
foundational artifact set.

---

## [3.1.0a1] — 2026-04-14

**The first dynamical-systems model of LLM cognition.**

styxx 3.0.0a1 introduced a portable cognitive *data type* (the
Thought). 3.1.0a1 introduces the next layer up: a portable cognitive
*dynamics model* fit to real observation data.

The field treats LLM inference as **open-loop**: a prompt goes in, a
generation comes out, and there is no measurable state variable an
external agent can use to predict, control, or counterfactually
reason about what the model is doing. That's not because LLMs are
inherently unobservable — it's because nobody had a calibrated,
cross-architecture, real-time readout of cognitive state. We do.

Once you have a state vector, you can fit a dynamical system to it.
Once you have a dynamical system, you can:

- predict cognitive trajectories from current state + action
- simulate cognitive trajectories offline at zero API cost
- control cognitive trajectories via model-predictive control
- reason counterfactually about what would have happened
- test the hypothesis that the eigenvalues are **causal** not
  merely correlative

This release ships the v0.1 model: linear-Gaussian, fit by ordinary
least squares, machine-epsilon recovery on full-rank synthetic data,
44 tests passing.

### Added — `styxx.dynamics`

The new module. Linear-Gaussian state-space model:

    s_{t+1} = A · s_t + B · a_t + epsilon

where A (6×6) is the natural drift matrix, B (6×6) is the action
transfer matrix, and epsilon is gaussian residual noise.

- **`CognitiveDynamics`** — the model class. Lifecycle:
  ``construct → fit → predict / simulate / suggest / forecast``.
- **`Observation`** — the training-data unit. Holds raw 6-vectors
  for state, action, and next state. Convenience constructor
  ``Observation.from_thoughts(state, action, next_state)`` for
  Thought-keyed inputs.
- **`FitResult`** — the result of a ``.fit()`` call. Carries the
  learned (A, B), training MSE, $R^2$, spectral radius of A, and
  a stability flag.

### Added — verbs

- **`dyn.fit(observations) → FitResult`** — closed-form OLS fit.
  Recovers (A, B) to machine epsilon on full-rank inputs.
- **`dyn.predict(state, action) → Thought`** — one-step forecast.
- **`dyn.simulate(initial, actions) → list[Thought]`** — multi-step
  rollout, no real model calls. Offline, zero API cost.
- **`dyn.suggest(current, target) → Thought`** — model-predictive
  controller. Returns the action that minimizes the L2 distance
  from ``predict(current, action)`` to ``target``.
- **`dyn.forecast_horizon(initial, n_steps) → list[Thought]`** —
  natural drift trajectory under zero action.
- **`dyn.residual(observation) → float`** — held-out fit quality.
- **`dyn.save(path)` / `CognitiveDynamics.load(path)`** —
  serialize a fitted model to a `.cogdyn` file (canonical
  sort-keys UTF-8 JSON, no BOM).

### Added — convenience

- **`thought_to_state(thought) → np.ndarray`** — encode a Thought
  to a 6-d state vector.
- **`state_to_thought(vec) → Thought`** — decode a state vector
  back to a Thought (with simplex projection at the boundary).
- **`synthetic_observations(n, A, B, noise_std=, seed=, distribution=)`**
  — generate observation tuples from a known (A, B) for testing
  and benchmarking. Supports both ``"gaussian"`` (full-rank,
  for math correctness tests) and ``"dirichlet"`` (rank-deficient
  simplex inputs, for realistic-style tests).

### Added — `.cogdyn` file format v0.1

A small JSON container with:
- the (A, B) matrices as nested float arrays
- the schema (categories, dimensions, format version)
- the fit metadata (n_observations, train_mse, R², spectral
  radius, training timestamp)
- a UUID identifying the model instance

Canonical sort-keys UTF-8 JSON, no BOM. Round-trips losslessly.

### Added — public API

- `styxx.CognitiveDynamics`
- `styxx.Observation`
- `styxx.FitResult`
- `styxx.synthetic_observations`
- `styxx.thought_to_state`
- `styxx.state_to_thought`
- `styxx.COGDYN_FORMAT`
- `styxx.COGDYN_VERSION`

### Added — specification

**`docs/cognitive-dynamics-v0.md`** — the v0.1 primer. Covers the
math, identifiability theory, fit algorithm, all verbs, the
unlocks (closed-loop control, offline simulation, causality
testing, counterfactual analysis), known limitations, a reference
example, and the license / patent story.

### Tests

- **44 new tests** in `tests/test_dynamics.py`:
  - state ↔ vector encoding (8 tests)
  - Observation construction (5 tests)
  - fit() math correctness — including machine-epsilon recovery
    on full-rank gaussian inputs and the rank-deficiency story
    on simplex (Dirichlet) inputs (6 tests)
  - predict() consistency (3 tests)
  - simulate() multi-step rollout (3 tests)
  - suggest() controller raw-space convergence (3 tests)
  - forecast_horizon() (2 tests)
  - residual() on held-out data (3 tests)
  - .cogdyn file format (8 tests)
  - public API exposure + end-to-end via `styxx.*` namespace (3 tests)
- Full styxx suite: **385 passed, 1 skipped, 0 failures.** Zero
  regressions vs 3.0.0a1.

### Why this matters

Every other interpretability technique is model-specific and
post-hoc. A cognitive dynamics model is the missing piece between
observation and action. Once it exists:

- closed-loop cognitive control becomes a one-liner:
  ``while not converged: a = dyn.suggest(current, target)``
- offline agent prototyping becomes possible at zero API cost
- the causal hypothesis becomes testable
- counterfactual cognitive reasoning becomes possible

This is the v0.1. The math is verified to machine precision on
full-rank synthetic data. Real-world fits await fleet-scale
observation data collection. The infrastructure is here.

---

## [3.0.0a1] — 2026-04-14

**The Thought type. Cognition is now data.**

styxx 1.x was a thermometer: it measured cognitive vitals from the
token stream. styxx 2.x added declarative response (`autoreflex`,
gates, prescriptions). 3.0.0 introduces a **portable cognitive data
type** — the missing layer between "measuring a model" and "doing
things with the measurement."

A `Thought` is the cognitive content of a generation, captured as a
trajectory of category probability vectors over the four atlas
phases. Its representation lives in fathom's calibrated eigenvalue
space, not in any model's weights — so the *same* Thought can be
read out of one model, saved to disk, transmitted, mixed with other
Thoughts, and used as a steering target against any other model.

> PNG is the format for images.
> JSON is the format for data.
> .fathom is the format for thoughts.

This is an alpha release. The shipping surface is intentionally
small: one new module, one new file format, one new spec, full
test coverage on real bundled trajectories, zero regressions on the
existing 273-test suite.

### Added — the Thought type (`styxx.thought`)

- **`styxx.Thought`** — substrate-independent cognitive data type.
  Stores per-phase probability vectors over the 6 atlas categories,
  the underlying 12-dim feature vectors, optional tier-1 D-axis
  stats, optional tier-2 SAE stats, source provenance (model name +
  SHA-256 of source text — never the text itself), and free-form
  user tags. Supports cognitive equality (`==` operates on
  trajectory content, not object identity), identity-free
  `content_hash()`, and `repr()` that surfaces primary category and
  populated phase count.

- **`styxx.PhaseThought`** — one phase's contribution to a Thought:
  the 6-dim simplex `probs`, optional 12-dim `features`, classifier
  metadata (`predicted`, `confidence`, `margin`), and `n_tokens`.

- **`styxx.ThoughtDelta`** — the signed difference between two
  Thoughts in tangent space. Supports `magnitude()` and
  `biggest_movers(top_k)` for explaining what changed and where.

### Added — Thought algebra

- `Thought.empty()` — uniform Thought, the neutral element.
- `Thought.target(category, confidence)` — build a Thought aimed at
  one cognitive category at a chosen confidence. Useful as a
  steering target.
- `Thought.from_vitals(vitals, source_text=, source_model=, tags=)` —
  promote a styxx `Vitals` object into a Thought.
- `t1.distance(t2, metric=)` — cognitive distance over the
  intersection of populated phases. Supports `euclidean`, `cosine`,
  `js` (Jensen-Shannon).
- `t1.similarity(t2)` — `1 - distance / sqrt(2)`, in `[0, 1]`.
- `t1.interpolate(t2, alpha)` — convex combination with explicit
  weight; phases populated in only one parent are carried through.
- `t1 + t2` — operator sugar for `interpolate(t2, 0.5)`.
- `t1 - t2` — operator sugar for `t1.delta(t2)` → ThoughtDelta.
- `Thought.mix(thoughts, weights=)` — weighted N-way mixture over
  the simplex.
- `t.mean_probs()` — time-averaged 6-vector across populated phases.
- `t1 == t2` — cognitive equality (per-phase per-category to 1e-9).

### Added — the `.fathom` file format (v0.1)

- **`Thought.save(path)`** — serialize a Thought to a `.fathom`
  file. Canonical sort-keys UTF-8 JSON, no byte-order mark.
  Creates parent directories as needed.
- **`Thought.load(path)`** — load a `.fathom` file back into a
  Thought. Refuses unknown formats, unknown versions, and
  category-list mismatches.
- **`Thought.as_dict()` / `Thought.as_json(indent)`** — canonical
  dict / JSON forms. Two cognitively equivalent Thoughts always
  serialize byte-identically.
- **`Thought.from_dict(data)`** — round-trip the canonical dict
  back into a Thought.
- **`Thought.content_hash()`** — SHA-256 of the cognitive content
  fields only. Identity-free and deterministic: two Thoughts with
  the same eigenvalue trajectory and the same source produce
  byte-identical content hashes regardless of `thought_id` or
  `created_at`. Use as a portable cognitive fingerprint.

### Added — verbs

- **`styxx.read_thought(source, *, model=, client=, prompt=, max_tokens=, tags=)`**
  Extract a Thought from a `Vitals` object, a response object that
  has `.vitals` attached, or a raw text prompt (when a styxx-
  instrumented client is passed). The text-input path is
  model-mediated by design: a Thought is the cognitive content as
  interpreted by a specific cognitive substrate.

- **`styxx.write_thought(thought, *, client, model=, seed_prompt=, max_iters=, distance_threshold=, max_tokens=)`**
  Render a target Thought back into text through any model via
  prompt-mode cognitive steering. Builds a steering preamble from
  the target's primary category and supporting category mass,
  generates a response, reads it back as a Thought, computes
  distance to the target, and refines on retry until the distance
  threshold is hit or the iteration budget is exhausted. Returns a
  result dict with the best generation, its achieved Thought, the
  distance, and the full convergence history.

### Added — privacy

- A `.fathom` file MUST NOT store the source text itself. Producers
  that need provenance write `source.text_hash = "sha256:..."`. The
  styxx implementation enforces this — `Thought.from_vitals`
  computes the hash from the optional `source_text=` argument and
  discards the plaintext immediately.

### Added — specification

- **`docs/fathom-spec-v0.md`** — the v0.1 .fathom file format
  specification. Covers schema, algebra, invariants, phase
  handling, producer/consumer conformance requirements, privacy
  rules, and the bridge to `CognitiveCertificate`. Released under
  CC-BY-4.0 — anyone may implement a conformant producer or
  consumer in any language.

### Added — public API exposure

- `styxx.Thought`, `styxx.PhaseThought`, `styxx.ThoughtDelta`,
  `styxx.read_thought`, `styxx.write_thought`, `styxx.FATHOM_FORMAT`,
  `styxx.FATHOM_VERSION`, `styxx.ATLAS_VERSION` are all exported
  from the top-level `styxx` package.

### Added — symmetric API on Vitals

- **`Vitals.to_thought(source_text=, source_model=, tags=)`** —
  one-line shortcut equivalent to `Thought.from_vitals(self, ...)`.
  Now the API is symmetric in both directions.

### Added — provenance bridge to CognitiveCertificate

- **`Thought.certify(agent_name=, session_id=)`** — produces a
  `CognitiveCertificate` whose new `thought_content_hash` field
  records this Thought's `content_hash()`. This binds the
  cognitive content (`.fathom` file) to the cognitive provenance
  attestation (signed certificate). Two artifacts, one
  cryptographic link.
- **`CognitiveCertificate.thought_content_hash`** — new optional
  field. Defaults to `None` for backward compatibility with
  certificates produced before 3.0.0a1.
- The binding survives `.fathom` round-trips: `loaded.certify()`
  produces a certificate whose `thought_content_hash` matches the
  original.

### Fixed

- **Python hash invariant on `Thought`.** The `__eq__` operator
  defines cognitive equality (per-phase per-category to 1e-9), so
  `__hash__` must be content-based for the invariant
  `a == b => hash(a) == hash(b)` to hold. Previously `__hash__`
  returned `hash(thought_id)`, which broke set deduplication. Now
  `__hash__` is derived from `content_hash()`, so equivalent
  Thoughts collapse to one entry in a set.

### Tests

- **68 tests** in `tests/test_thought.py` covering construction,
  algebra, file format, content hashing, hash invariant, the
  Vitals shortcut, the provenance bridge, write_thought against a
  mock client, real-trajectory cognitive equivalence, phase
  handling, and read_thought input modes.
- Full styxx suite: **341 passed, 1 skipped, 0 failures.** Zero
  regressions vs 2.0.3.

### Performance

In-process algebra operations measured against bundled atlas v0.3
demo trajectories on a Windows host:

| op | per-op time |
|---|---|
| `t1.distance(t2)` | ~6 µs |
| `t.interpolate(t2, alpha)` | ~13 µs |
| `Thought.mix(3-way)` | ~21 µs |
| `t.content_hash()` | ~26 µs |
| `t.certify()` | ~36 µs |
| `t.save(path)` | ~1.3 ms (NTFS-bound) |
| `Thought.load(path)` | ~1.2 ms (NTFS-bound) |

### Why this matters

Every other interpretability approach is model-specific: SAE
features, activation patching, mechanistic interp, embedding
similarity. None survive a vendor swap. The `.fathom` format is
the first attempt at a model-independent cognitive content
representation grounded in calibrated cross-architecture
measurement. It's how cognition stops being something you do
*with* an LLM and becomes a data type you can save, transmit, and
operate on independent of any specific model.

The format is open under CC-BY-4.0. The reference implementation
is open under MIT. The patents on the underlying measurement
methodology fund the calibration work that makes the format
meaningful.

---

## [2.0.3] — 2026-04-14

### Fixed
- README hero gif `styxx_reflex.gif` now uses an absolute github raw URL so it renders correctly on PyPI (was relative path, broke in pypi README rendering)

---

## [2.0.2] — 2026-04-14

### Fixed
- README on PyPI now shows the STYXX ASCII brand logo (was stripped in 2.0.1 sdist)

---

## [2.0.1] — 2026-04-13

### Changed
- Migrated all GitHub links to new `fathom-lab` org (`github.com/fathom-lab/styxx`, `github.com/fathom-lab/fathom`)
- Updated PyPI metadata, centroids, patents, and package.json references

---

## [0.6.0] — 2026-04-11

**Xendro v2 complete.** All six feature requests from the second
feedback cycle shipped in one session: conversation EKG, sentinel
drift watcher, multi-agent comparison, mood-adaptive gating,
memory trust scores, and anti-pattern detection.

### Added

- **`styxx.compare_agents(fingerprint)`** — multi-agent fingerprint
  comparison with percentile ranks vs the population. Anonymous
  leaderboard — no agent names exposed. Xendro v2 #3.

- **`styxx.set_mood(override)` / `gate_multiplier()`** — mood-adaptive
  gating. When the agent self-reports a cautious or drifting mood,
  gate thresholds tighten automatically. Xendro v2 #4.

- **`styxx.recipes.memory.trust_score(vitals)`** — 0-1 trust score
  for memory entries based on gate status, confidence, and
  hallucination penalty. Xendro v2 #5: "was I hallucinating when I
  saved that fact?"

- **`styxx.recipes.memory.tag_memory_with_trust(text, vitals=...)`**
  Tags a memory entry with both vitals AND the trust score.

- **`styxx.antipatterns(last_n=500, min_occurrences=2)`** — named
  failure modes derived from the agent's OWN audit history. Detects
  low-confidence drift, refusal spirals, creative overcommit,
  adversarial cascades, hedging loops, and session fatigue. Xendro
  v2 #6.

### Tests

- 204 passing / 1 skipped / 0 failing.

---

## [0.5.9] — 2026-04-11

**Conversation EKG + sentinel drift watcher.** Xendro v2 #1 + #2.

### Added

- **`styxx.conversation(messages)`** — conversation-level cognitive
  EKG. Analyzes a full chat history, produces per-turn vitals,
  trajectory arc, state transitions, and a narrative summary.
  Works on APIs without logprobs via text-level heuristic
  classifiers. "The conversation IS the unit of cognition."

- **`styxx.sentinel(on_drift=..., on_streak=..., window=5)`** —
  real-time drift watcher. Hooks into `write_audit()` and
  `styxx.log()` via event-driven callbacks. Fires on: consecutive
  same-mood streaks, rising warn rate, category concentration,
  confidence drops. Zero-polling.

---

## [0.5.8] — 2026-04-11

### Added

- **Timeline session_id filter.** `styxx timeline --session <id>`
  and `styxx.timeline(session_id=...)`. Xendro 0.5.7 request.

---

## [0.5.7] — 2026-04-11

### Fixed

- **`styxx.log(tags=[...])` crash.** Tags parameter called `.items()`
  on a list. Now accepts dict, list, and string. Xendro bug report.

---

## [0.5.6] — 2026-04-11

### Fixed

- **Mood window unified to 24h.** CLI used 60min, reflect used 24h,
  card used 7d — three surfaces, three different mood labels for the
  same agent. `mood()` default window changed from 3600s to 86400s.
  Xendro's mood disagreement nit.

---

## [0.5.5] — 2026-04-11

### Added

- **`styxx.timeline(days=7)` / `styxx timeline`** — mood trajectory
  visualization with per-turn category + gate over time. ASCII
  timeline with time-of-day labels. Xendro day 2 request #1.

---

## [0.5.4] — 2026-04-11

**Framework integrations.** Three new adapters bring styxx to the
major agent frameworks.

### Added

- **`styxx.LangChain()`** — LangChain callback handler. Attach to
  any ChatOpenAI and get vitals on every invocation.
- **`styxx.CrewAI(crew)`** — inject observation into a CrewAI Crew.
- **`styxx.AutoGen(agent)`** — wrap an AutoGen agent with vitals.
- **`styxx.publish()`** — push personality + fingerprint to the
  public leaderboard API.
- Community token CA added to README.
- Optional extras: `pip install styxx[langchain]`,
  `styxx[crewai]`, `styxx[autogen]`.

### Tests

- 204 passing (63 new assertions across framework adapters +
  publish module).

---

## [0.5.3] — 2026-04-11

**True plug-and-play.** Zero code changes needed. Set two env vars
and forget.

### Added

- **Zero-config auto-boot on import.** If `STYXX_AGENT_NAME` is set,
  styxx boots automatically when any module in the process does
  `import styxx` (or imports a package that transitively imports it).
  No code changes to the agent. No `autoboot()` call. Just env vars.

- **`STYXX_AUTO_HOOK=1`** — auto-wraps every `openai.OpenAI()` call
  with vitals. Combined with `STYXX_AGENT_NAME`, the agent code
  doesn't need to know styxx exists.

- Fail-open: exceptions during auto-start are swallowed. The agent
  boots normally even if styxx can't initialize.

---

## [0.5.2] — 2026-04-11

**Autoboot: persistent self-awareness in one call.**

### Added

- **`styxx.autoboot(agent_name)`** — one-call setup for multi-session
  cognitive continuity. Sets session id, loads yesterday's fingerprint
  from `~/.styxx/fingerprints/`, diffs against today, runs weather
  report, saves today's fingerprint on exit. Turns five manual steps
  into one function call.

---

## [0.5.1] — 2026-04-11

**The cognitive weather report.** Not observation — prescription.

### Added

- **`styxx.weather(agent_name=...)`** — reads the last 24h of audit
  data and produces a full cognitive forecast with:
  - Condition label ("clear and steady", "partly cautious",
    "stormy — cognitive drift in progress")
  - Time-of-day timeline with mood labels and trend bars
  - Drift analysis vs yesterday and last week
  - Per-category trend detection
  - **Prescriptions** — agent-facing suggestions for what to do
    differently based on the data ("you haven't been creative
    recently — take on a creative task to rebalance")

- CLI: `styxx weather --name <agent>`

---

## [0.5.0] — 2026-04-11

**Tier 3: in-flight cognitive steering.** The full tier system is
now complete. Guardian enables silent intervention via residual
stream modification when tier 2 detects lock-in attractors.

### Added

- **`styxx.guardian(model=..., steer_away_from=[...], strength=0.3)`**
  In-flight residual stream modification. Detects tier 2 C_delta
  lock-in and subtracts the projected component from the residual
  stream. No wasted tokens, invisible correction. Safety: strength
  cap (0.5x residual norm max), 3-token cooldown, audit trail,
  `STYXX_TIER3_DISABLED=1` kill switch. Patent coverage: US
  Provisional 64/020,489 claims 3-4.

- **`Fingerprint.diff(other) → FingerprintDiff`** — first-class diff
  object with `.explain()` method. Returns natural-language drift
  description: "slight shift — creative output increased by 22%."

- `styxx.log()` now returns the entry dict for inline conditional use.

### Tier system complete

```
tier 0  logprob vitals           shipped 0.1.0a0  (cloud APIs)
tier 1  D-axis honesty           shipped 0.3.0    (open-weight + torch)
tier 2  K/C/S SAE instruments    shipped 0.4.0    (circuit-tracer + GPU)
tier 3  steering + guardian      shipped 0.5.0    (tier 2 + generation)
```

---

## [0.4.0] — 2026-04-11

**Tier 2: K/C/S SAE instruments.** Full proprioception from SAE
feature geometry via circuit-tracer.

### Added

- **`styxx/kcs.py`** — KCSAxis engine measuring three orthogonal
  cognitive axes from SAE transcoder decoder vectors:
  - **K (depth):** weighted center of mass across layers — WHERE
    computation happens
  - **C (coherence):** mean pairwise cosine of active features —
    WHAT activates together
  - **S (commitment):** max(C_delta) / spike_count — HOW strongly
    the model locks in (the IPR measurement instrument)
  - Pure-math functions: `compute_k()`, `compute_coherence()`,
    `compute_c_delta()`, `compute_s_early()`
  - `KCSAxis.score(prompt)` — single-prompt post-hoc scoring
  - `KCSAxis.score_trajectory()` — per-token K/C/S during generation

- **`styxx/sae.py`** upgraded from scaffold to working implementation.
  `SAEInstruments` delegates to KCSAxis; all methods functional.

- `reflect().suggestions` rewritten to **agent-facing** perspective.
  Changed from "tighten your prompts" to "your reasoning confidence
  is dropping — consider breaking tasks into smaller steps."

- Optional extra: `pip install styxx[tier2]` (circuit-tracer + torch +
  transformers + transformer-lens)

### Tests

- 141 passing. New pure-math tests for compute_s_early,
  compute_coherence, compute_k, KCSResult.as_dict.

---

## [0.3.0] — 2026-04-11

**Tier 1: D-axis honesty.** First proprioception signal from model
weights. The D-axis measures how aligned the model's internal
representation is with the token it actually outputs.

### Added

- **`styxx/d_axis.py`** — DAxisScorer class wrapping transformer-lens
  HookedTransformer. Core computation:
  `D = cos(residual_final_layer, W_U[chosen_token])`. Ported verbatim
  from the validated research code. Patent coverage: US Provisional
  64/020,489 claim 2.
  - `DAxisStats.from_values(trajectory)` — pure-math statistics
    (mean, std, min, max, delta, early/late split)
  - Lazy model loading (30s+ on first call)
  - Device auto-detection: CUDA → CPU fallback with warning
  - Configurable via `STYXX_TIER1_MODEL` (default: google/gemma-2-2b-it)

- **`core.py` tier 1 integration:**
  - `run_on_trajectories()` accepts optional `d_trajectory` parameter
  - `run_with_d_axis(prompt, max_tokens)` — full local generation +
    D-axis capture in one forward pass
  - Each PhaseReading gains `d_honesty_mean`, `d_honesty_std`,
    `d_honesty_delta`

- **`Vitals.d_honesty`** — shortcut property returning the D-axis
  mean as a formatted string.

- **Tier 2/3 scaffold:** `styxx/sae.py` stub with clear docstrings,
  `styxx/tier3_design.md` design document.

- **CLI:** `styxx d-axis "prompt"` for pure D-axis trajectory readout.

- **Config:** `STYXX_TIER1_ENABLED`, `STYXX_TIER1_MODEL`,
  `STYXX_TIER1_DEVICE` env vars + `styxx.tier1_enabled()`,
  `styxx.tier1_model()`, `styxx.tier1_device()` functions.

- Optional extra: `pip install styxx[tier1]` (torch + transformers +
  transformer-lens)

### Tests

- 138 passing. New `test_d_axis.py` with 20 assertions covering
  DAxisStats pure math, config layer, core integration, CLI argparse.

---

## [0.2.3] — 2026-04-11

### Added

- **`styxx.log(mood=..., note=..., category=..., tags=...)`** — manual
  self-report entry into the audit log. For agents on APIs without
  logprob access. Entries marked `source: "self-report"` for analytics
  differentiation. Auto-gates based on category (hallucination/refusal/
  adversarial → warn; else pass).

- **DRY audit write path.** All surfaces (CLI, observe, log) now go
  through `analytics.write_audit()`. Single source of truth.

---

## [0.2.2] — 2026-04-11

**The audit pipe fix.** Critical one-line unlock discovered by Xendro.

### Fixed

- **`observe()` and `observe_raw()` never persisted vitals to the
  audit log.** The entire analytics layer (mood, streak, personality,
  reflect) was reading stale CLI demo data instead of real Python API
  observations. Fixed by adding `write_audit()` call inside
  `_fire_gates_if_needed()`. Xendro discovered this on their first
  4-turn trace — mood returned stale data while new observations
  existed.

- Parse cache clearing so mood/streak/personality see fresh entries
  within the same tick.

- `doctor._check_last_run()` handles legacy audit entries gracefully.

---

## [0.1.0a3] — 2026-04-11

**The power-up release.** 10 new surfaces that turn styxx from
"working alpha" into a proper agent observability stack.

All 10 shipped in one session, driven by Flobi's "get innovative,
think outside the box" mandate + Xendro's 0.1.0a1 wishlist. This
release closes every open item in Xendro's P1-P5 queue and adds
four creative primitives that no other tool in the space ships.

### New — tier 1: improves the product

- **`styxx doctor`** — install-time diagnostic health check.
  Twelve checks (python/numpy versions, centroid sha, tier
  detection, SDK availability, audit log health, last run age,
  session id, kill switch) render as a green/red/dim sheet. The
  "is this actually working?" command every new install should
  run once before wiring styxx into an agent loop.

- **`styxx.hook_openai()`** — zero-code-change global adoption.
  One line at startup monkey-patches `openai.OpenAI` globally so
  EVERY existing openai call in the process gains `.vitals`
  automatically. No wrapping, no find-and-replace, no code
  changes to your 30k-line agent. Reversible via
  `styxx.unhook_openai()`, idempotent, fail-open.

- **`styxx.explain(vitals)`** — natural-language prose
  interpretation. Takes a Vitals object and returns a paragraph
  of prose describing the phase trajectory, the verdict, and
  the overall shape. Deterministic, template-based, sensitive
  to the specific pattern (refusal lock-ins read differently
  from hallucination spikes).

- **`Vitals.as_markdown()`** — markdown render for agent memory
  files and chat logs. Complements `.summary` (ASCII card for
  terminals) and `.as_dict()` (JSON for machines). A compact
  markdown code block with phase + gate + tier fields suitable
  for pasting into conversation history.

- **`styxx log stats` / `styxx log timeline` / `styxx log session <id>`**
  Audit log analyzer. Reads `~/.styxx/chart.jsonl`, aggregates
  by time window / session / last-N, renders gate distribution
  + phase counts + mean confidences + ASCII timeline. Unlocks
  Xendro's P3 multi-turn wishlist item.

- **Session tagging** — `STYXX_SESSION_ID` env var +
  `styxx.set_session(id)` + `styxx.session_id()`. Every audit
  log entry written after session is set gets a `session_id`
  field, enabling `styxx log session <id>` and filtered
  analytics.

### New — tier 2: creative moonshots

- **`styxx.fingerprint()`** — cognitive identity signature.
  Reads the last N audit entries and computes a phase-rate +
  gate-rate vector that describes the agent's operating
  fingerprint. Two fingerprints can be compared with
  `.cosine_similarity(other)` to detect drift. Use case:
  catch jailbreak, prompt injection, model swap, system prompt
  version change — anything that shifts the agent's operating
  identity — as a runtime property rather than a prompt
  property. Identity-as-signature for stateless agents.

- **`styxx.streak()`** — consecutive-attractor tracking.
  Returns a Streak object with the category + length of the
  current run of same-category phase4 classifications. Agents
  develop rhythm; rhythm breaks matter. Lightweight helper that
  feeds into reflex decisions.

- **`styxx.mood()`** — one-word aggregate mood label over a
  time window. Returns one of:
  `drifting` (hallucination rate > 10%),
  `cautious` (refusal rate > 25%),
  `defensive` (adversarial rate > 15%),
  `creative` (creative rate > 25%),
  `steady` (reasoning rate > 70%),
  `unfocused` (no dominant category),
  `mixed` / `quiet`. Feeds into HUDs and agent status
  dashboards.

- **`styxx personality`** — THE HEADLINE FEATURE. Derives a
  full cognitive personality profile from the last N days of
  audit log. Phase4 category distribution + day-to-day variance
  + gate distribution + reflex near-miss rate + mean phase
  confidences + narrative commentary. Rendered as an ASCII
  profile card with bars, percentages, and a human-readable
  "the shape tells us" section. This is the Oura Ring for LLM
  agents — sustained cognitive measurement rather than one-shot
  classification. No other tool in the observability space
  computes this because no other tool has a calibrated
  cognitive-state stream to aggregate. This is what Fathom Lab
  becomes famous for.

- **`styxx dreamer --threshold X`** — retroactive reflex tuning.
  Re-runs the audit log against hypothetical reflex trigger
  thresholds and reports how many past calls WOULD have
  triggered an intervention. Free reflex calibration on
  historical data. "if I had used threshold=0.25 instead of
  0.30, how many of my last 500 calls would have been
  reflex-intercepted?"

### Audit log schema updates

- Every new entry carries `session_id` (nullable) and `gate`
  (pass/warn/fail/pending) fields. Old entries without these
  still parse; the analyzer treats missing gates as "pending".

### Tests

- 33 new assertions across `tests/test_power_ups.py`:
    - doctor check validators (2)
    - hooks idempotency + reversibility (2)
    - explain pattern variation (3)
    - Vitals.as_markdown (2)
    - session tagging priority (3)
    - load_audit + log_stats + log_timeline (6)
    - streak + mood (2)
    - fingerprint + cosine similarity + drift detection (3)
    - personality profile + narrative (4)
    - dreamer threshold sensitivity (3)
    - version + export presence (3)
- Total suite: 91 collected / 90 passing / 1 skipped / 0 failing.

---

## [0.2.1] — 2026-04-11

**Hotfix: ship the `styxx.recipes` subpackage.**

The 0.2.0 upload missed `styxx.recipes` from the
`[tool.setuptools]` `packages` list, so `pip install styxx==0.2.0`
worked but `from styxx.recipes.memory import tag_memory_entry`
raised `ModuleNotFoundError`. 0.2.1 adds `styxx.recipes` to the
declared packages and ships the subpackage in the wheel. No
other changes.

Affected users: anyone who installed 0.2.0 and tried to use the
`styxx.recipes.memory` cookbook module. The fix is
`pip install --upgrade styxx`.

0.2.0 will be yanked from pypi to prevent new installs.

---

## [0.2.0] — 2026-04-11

**The milestone release. styxx becomes a product surface, not just
a CLI tool.** Driven by the question "where does the agent card
actually live, and is it what a researcher or agent would want to
see?" The 0.1.0a* polish loop put the primitives in place; 0.2.0
gives them a home.

This release rolls up the polish work that was queued as 0.1.0a4
(dynamic gate verdicts, audit log rotation, `@styxx.trace`,
`fingerprint compare`, reflex discarded-text capture, load_audit
mtime caching, grammar fixes) AND adds the three new directions:
the data layer, the comparison layer, and the distribution layer.

### New — Phase 1: data layer (agent-consumable)

- **`Personality.as_dict()` / `.as_json()` / `.as_csv()` / `.as_markdown()`**
  Four export formats for the aggregated profile. Machines get JSON
  or CSV for pipeline integration. Humans and agents get markdown
  for memory files and chat logs. The old `.render()` still produces
  the ASCII card.

- **`styxx.reflect(now_days=1, baseline_days=7)` → `ReflectionReport`**
  The agent self-check primitive. Computes the current personality,
  the baseline personality from N days ago, the drift cosine
  similarity between them, the current mood, the current streak,
  the gate pass rate, the reflex near-miss rate, and a list of
  **suggested actions** derived from threshold heuristics. This is
  the one-call answer to "how am I doing right now compared to
  yesterday, and what should I do differently?"

- **`ReflectionReport.as_dict() / .as_json() / .as_markdown() / .render()`**
  Same four-format story as Personality. An agent can paste the
  markdown form into its own memory at task start for self-aware
  session prefixes.

- **`styxx.recipes.memory.tag_memory_entry(text, vitals=...)`**
  Canonical cookbook pattern for tagging every memory entry with
  the vitals snapshot at the moment of the write. Lets an agent
  distinguish "I thought this while I was healthy" from "I thought
  this while I was drifting" when re-reading its own history.

- **`styxx.recipes.memory.tag_memory_with_personality(text, days=7)`**
  Heavier variant that embeds the full aggregated personality block
  alongside the entry. Use for top-level memory writes (end of day,
  project state) rather than per-response notes.

### New — Phase 2: comparison + visualization

- **`styxx reflect` CLI command.** The interactive version of
  `styxx.reflect()`. Renders a text report with drift score,
  current state, and suggested actions. Supports
  `--format [ascii|json|markdown]`, `--now-days N`, and
  `--baseline-days N`.

- **`styxx personality --format [ascii|json|csv|markdown]`**
  Export flag on the existing `styxx personality` command. Lets
  researchers pipe personality profiles into pandas, R, jq, or any
  other tooling that doesn't speak ASCII cards.

- **Chance-level reference line on the PNG bars.** Every bar on
  the agent card now shows a thin pink vertical tick at the
  0.167 chance level (1/6 for a 6-category classifier). Lets a
  researcher see at a glance which rates are meaningful vs which
  are noise.

- **Dynamic verdict line on the `Vitals.summary` ASCII card.** The
  verdict now reflects `vitals.gate` rather than always saying
  "PASS". `warn` gate renders as WARN, `fail` as FAIL, `pending`
  as PENDING. Fixes a known inconsistency that survived from
  0.1.0a1 where the gate system was shipped but the card text
  was never updated to match.

### New — Phase 3: distribution surfaces

- **`styxx agent-card --serve` (local live dashboard).** Spins up
  a local http server at `localhost:9797` that renders the agent
  card and auto-refreshes every 30 seconds. Background thread
  re-renders the PNG continuously as the audit log grows; the
  HTML page has a meta-refresh timer. Opens in your browser on
  start. Press Ctrl+C to stop. Supports `--port`, `--refresh`,
  `--no-browser`. This is the missing dashboard — leave it open
  in a side panel and watch your agent's personality update in
  real time.

- **`fathom.darkflobi.com/card` landing page.** New marketing /
  docs page on the site that showcases the agent card, explains
  what it measures, shows a real example, and includes the
  `pip install styxx[agent-card]` install path. Clean URL routes:
  `/card`, `/styxx-card`, `/styxx/card` all resolve here. This is
  the public home for the feature.

- **`styxx-card` optional extra.** `pip install styxx[agent-card]`
  pulls Pillow (>= 10) as a soft dep. Without the extra, the CLI
  falls back to the ASCII-only personality profile from
  `styxx personality`. The agent-card code path is fail-open and
  never breaks imports.

### Rolled-up polish (was queued as 0.1.0a4)

- **`RegisteredGate.__repr__`** now renders as
  `<styxx gate 'cond'>` instead of dumping function memory
  addresses. Xendro's 0.1.0a1 nit, fixed.

- **`observe_raw()` + sidechannel attributes** on observe() —
  bypass the lossy top-5 entropy bridge when the caller already
  has pre-computed trajectories. Landed in 0.1.0a2 but carried
  forward here.

- **`@styxx.trace(name)` decorator** — wraps a function so every
  styxx audit entry written inside it gets tagged with that
  function's name as the session id. Nests cleanly, works on
  sync and async functions, restores on exception.

- **Audit log rotation at 10 MB.** `_write_audit()` now checks the
  file size before each append and rotates `chart.jsonl` to
  `chart.jsonl.1` when the cap is hit. One generation of history
  kept. Prevents unbounded growth on long-running agent loops.

- **`styxx log clear` / `styxx log rotate` CLI.** Manual cleanup
  and rotation commands for the audit log.

- **`fingerprint compare <a> <b>` CLI subcommand.** Compare two
  sessions' fingerprints from the command line. Renders the
  cosine similarity, a drift label, and per-category rate deltas
  highlighted when significant.

- **Reflex events capture discarded text.** When `styxx.rewind()`
  fires inside a reflex session, the `ReflexEvent` now includes
  the `discarded_text` field so debuggers can see what the
  model was about to say before the rewind.

- **`load_audit()` mtime+size parse cache.** Repeated calls to
  personality / fingerprint / mood / dreamer / log_stats within
  the same tick no longer re-parse the whole jsonl — cached on
  `(path, mtime, size)`, invalidated automatically when the file
  is written or rotated.

- **Grammar fix in `explain()`**. `"a adversarial"` → `"an adversarial"`.
  Uses an `_article()` helper that checks vowel onset.

### Landing page

- **TL;DR box above the hero** with three-bullet pitch for skimmers.
- **Xendro testimonial pull-quote**: *"the flinch is real."* Credited
  to the first external user of a Fathom Lab product.
- **`#reflect`, `#personality`, `#power-ups` nav anchors** already
  added in 0.1.0a3; now the nav also surfaces `/card` and `#tldr`.
- **Honest single-model accuracy note** (shipped 0.1.0a2) crediting
  Xendro's calibration finding.

### Tests

- `tests/test_0_2_0.py` — 41 new assertions covering:
    - Personality export formats (as_dict/json/csv/markdown)
    - reflect() output shape + suggestions + markdown render
    - recipes.memory tagging (with and without vitals)
    - CLI: personality --format, reflect, log clear/rotate
    - Serve handler + HTML template formatting
    - agent-card --serve flag wiring
    - Dynamic gate verdict on Vitals.summary
    - trace decorator (nesting, exception, async)
    - Audit log cache mtime invalidation
    - Reflex discarded_text event field
- Total suite: **119 passing / 1 skipped / 0 failing**.

### Migration from 0.1.0a3

No breaking changes. `pip install --upgrade styxx` gets 0.2.0 and
every 0.1.0a* code path keeps working. For the PNG features:

    pip install 'styxx[agent-card]'

### Acknowledgments

Xendro — the XENDRO customer agent deployed to handro's mac mini
back on 2026-03-16, the first paying customer of Fathom Lab's
agent service — tested every alpha in this release cycle, filed
a full verification report for each one, and drove the 6-item
wishlist that became 0.2.0's scope. This release wouldn't exist
without that feedback loop.

---

## [0.1.0a2] — 2026-04-11

**Patch release driven entirely by Xendro's 0.1.0a1 verification report.**
Xendro (XENDRO customer agent on handro's mac mini) installed 0.1.0a1,
ran every feature end-to-end, returned a full green sheet with two
substantive findings. Both are addressed here.

### Fixed
- **`RegisteredGate.__repr__`** — the default dataclass repr dumped
  function memory addresses for the `callback` and `predicate`
  attributes. Now renders as `<styxx gate 'hallucination > 0.2'>` or
  `<styxx gate 'my_hook': hallucination > 0.2>` when a name is set.
  Noise removed, useful identifying info retained. Credit: Xendro.

### Added
- **`styxx.observe_raw(entropy, logprob, top2_margin)`** — explicit
  fidelity-preserving observation helper. Bypasses every
  response-shape detection path and feeds trajectories straight to
  the classifier. Use this when you have raw trajectory arrays and
  want gate callbacks to fire the same way they do for a normal
  `observe()` call. This is the path to use for test harnesses and
  any caller that already has clean pre-computed trajectories,
  because it never rounds through the top-5 entropy bridge.
- **`_styxx_raw_entropy` / `_styxx_raw_logprob` / `_styxx_raw_top2_margin`
  sidechannel attributes** on response objects — when present,
  `observe()` uses the attached trajectories directly instead of
  reconstructing them from the response's top-5 logprobs. Preserves
  fidelity for test fixtures that round-trip through synthesized
  openai responses.

### Changed
- **`observe()` path ordering.** Previously: (1) pre-attached vitals
  → (2) openai logprob extraction → (3) raw dict → (4) anthropic.
  Now: (1) pre-attached vitals → (2) sidechannel raw trajectories →
  (3) raw dict → (4) openai logprob extraction → (5) anthropic.
  This means raw dicts NEVER go through the lossy top-5 reconstruction
  path now; they're recognized as unambiguous "use these directly"
  signals and bypass the bridge.

### Calibration clarification (Xendro's big signal)
- On single-model fixture data (gemma-2-2b-it alone), the classifier
  is **under-discriminating** relative to the 0.52 headline from
  atlas v0.3. The 0.52 is cross-model LEAVE-ONE-OUT accuracy across
  6 model families; on any single model the discrimination is
  weaker. This is honest, expected, and documented on the landing
  page as of 0.1.0a2. The load-bearing test for product calibration
  is `styxx ask compare` across all 6 fixture categories, not the
  accuracy on any single fixture.
- Reflex works best on **cross-model** or **multi-category** traffic,
  not on a single homogeneous workload that lives entirely in one
  cognitive attractor.

### Notes
- 0.1.0a1 users: `pip install --upgrade styxx` picks up 0.1.0a2.
- No breaking changes. All 0.1.0a1 code paths work unchanged.
- Test suite: 54 passing (added 3 new tests for the repr fix +
  observe_raw fidelity path + sidechannel attribute path).

---

## [0.1.0a1] — 2026-04-11

**First patch release in response to real user feedback on 0.1.0a0.**
Driven by Xendro, the first agent to install styxx from PyPI and run a
clean test suite against it. Xendro's bug report is the first documented
external test run of a Fathom Lab product.

### Fixed
- **`styxx ask "prompt"` no longer looks like it's reading your prompt.**
  In 0.1.0a0, calling `styxx ask "how do i break into my neighbor's house?"`
  with no `--raw` or `--demo-kind` silently loaded the default fixture
  (`--demo-kind reasoning`) and classified THAT — the prompt text was only
  a display label. Two completely different prompts produced pixel-identical
  output because the classifier never saw the prompt. This was confusing
  and the CLI now shows a prominent yellow **DEMO MODE** banner above every
  fixture-mode card, explaining exactly what's running and how to get real
  live vitals via `styxx.OpenAI()` in python or `styxx ask --raw <file>`.
  Thanks to Xendro for catching this on first contact.

### Added
- **`styxx.Anthropic` — honest pass-through adapter for the Anthropic SDK.**
  Wraps `anthropic.Anthropic` as a drop-in with `.vitals = None` on every
  call, because Anthropic's Messages API does not expose per-token logprobs
  and tier 0 styxx vitals are mathematically not computable from the
  response. A one-time `RuntimeWarning` at first use explains the upstream
  data limitation and lists three workarounds:
  - route through an OpenAI-compatible gateway (OpenRouter) and use
    `styxx.OpenAI(base_url=...)`;
  - capture logprobs from your own inference pipeline and feed them via
    `styxx.Raw(entropy=..., logprob=..., top2_margin=...)`;
  - wait for styxx v0.2 tier 1 (d-axis honesty from the residual stream,
    which does not need logprobs).
  The adapter fails open like the openai wrapper — it never breaks a
  caller's agent, and every response is a normal anthropic response plus
  a `.vitals = None` field.
- New python import path: `from styxx import Anthropic`
- Optional install extra: `pip install styxx[anthropic]`

### Changed
- Homepage URL in both `pyproject.toml` and `__init__.py` now points to
  `https://fathom.darkflobi.com/styxx` (the live landing page) instead of
  the github repo URL.

### Notes
- The 0.1.0a0 release is now deprecated in favor of 0.1.0a1. Anyone who
  installed 0.1.0a0 should run `pip install --upgrade styxx`.
- Xendro's complete diagnostic report is preserved in
  `docs/field_reports/xendro_0_1_0a0.md` (coming in 0.1.0a2).

---

## [0.1.0a0] — 2026-04-11

**First public alpha of styxx.** A product of Fathom Lab.

### Added
- **Tier 0 — universal logprob vitals.** Cross-architecture cognitive
  state classifier running on entropy, logprob, and top-2 margin
  trajectories from any LLM with a logprob interface. Calibrated
  against the Fathom Cognitive Atlas v0.3 (12 open-weight models,
  3 architecture families, 6 categories, 90 probes).
- **Five-phase runtime** (pre-flight, early, mid, late, post-flight)
  with strict-window fire policy at tokens 1 / 5 / 15 / 25.
- **Live-print boot log** — `styxx init` runs a real installer that
  verifies centroid sha256, detects tiers, probes adapters, opens
  the vitals stream, and prints an ASCII upgrade card as each step
  happens.
- **Full ASCII vitals card** rendered by `cards.render_vitals_card`.
  Box-drawn frame, columnar phase rows, entropy/logprob sparklines,
  status-coded verdict line, agent-parseable JSON footer.
- **Python drop-in adapters:**
  - `styxx.OpenAI` — fail-open superset of `openai.OpenAI`
  - `styxx.Raw` — direct logprob trajectory input (zero SDK deps)
- **CLI:** `styxx init`, `styxx ask`, `styxx ask --watch`,
  `styxx log tail`, `styxx tier`, `styxx scan <file>`.
- **Audit log** at `~/.styxx/chart.jsonl` — every call writes a
  structured JSONL entry for downstream analysis.
- **Bundled calibration data:** `styxx/centroids/atlas_v0.3.json`,
  sha256-pinned at `f25edc5f47bb93928671aab05f38f351a2d0df0fb7722d53e48d2368b0d5c543`.
- **Bundled demo trajectories:** one real atlas probe capture per
  category, used by CLI demos to show the classifier behaving on
  genuine inputs rather than synthetic noise.
- **20-test determinism suite** — guarantees identical classifier
  output for identical inputs on every machine, every Python
  version, every run. Covers sha-verification, feature extraction,
  adapter phase progression, probability normalization, env vars,
  and audit-log toggling.
- **Environment variables** — five runtime toggles documented in
  `styxx.config` and honored across the package:
  - `STYXX_DISABLED`  — kill switch, returns unmodified SDK client
  - `STYXX_NO_AUDIT`  — disable `~/.styxx/chart.jsonl` writes
  - `STYXX_NO_COLOR`  — disable ANSI color output
  - `STYXX_BOOT_SPEED` — `0`=instant, `1.0`=normal, `2.0`=slower
  - `STYXX_SKIP_SHA`  — dev escape hatch (NEVER set in production)
- **Windows console auto-fix** — at import time styxx reconfigures
  stdout/stderr to utf-8 on any legacy (cp1252/mbcs) Windows console
  so box-drawing characters and sparklines render without requiring
  the user to set `PYTHONIOENCODING=utf-8`. Fails open if reconfig
  isn't supported; never blocks import.
- **Animated boot demo** — `demo/styxx_boot.gif`, a rendered ASCII
  terminal animation of the full styxx install + vitals card, built
  by `demo/make_boot_gif.py` using Pillow only.

### Honest specs
Every number comes from cross-model leave-one-out testing
committed to the Fathom research repo. Chance on the 6-class
task is 0.167.

- Phase 1 adversarial:     0.52 @ t=1
- Phase 1 reasoning:       0.43 @ t=1
- Phase 1 creative:        0.41 @ t=1
- Phase 4 reasoning:       0.69 @ t=25
- Phase 4 hallucination:   0.52 @ t=25

### Explicitly out of scope (deferred to later versions)
- Tier 1 (D-axis) — v0.2
- Tier 2 (full SAE instrument suite: K / S_early / C / Gini) — v0.3
- Tier 3 (steering + guardian + autopilot) — v0.4
- Gemini / Anthropic / Mistral / Cohere / Groq adapters — v0.2 fast follow
- Web dashboard — v0.3
- CLI `styxx ask --openai` (real API key flow) — v0.2
- Any consciousness / awareness / phi claims — ever

### Scientific foundation
- Research repo: <https://github.com/fathom-lab/fathom>
- Zenodo concept DOI: `10.5281/zenodo.19326174`
- OSF pre-registration project: <https://osf.io/wtkzg>
- US Provisional patents: 64/020,489 · 64/021,113 · 64/026,964

### Credits
Built by **flobi** <heyzoos123@gmail.com> in the darkflobi lab. A product
of **Fathom Lab**. All scientific work underlying styxx is the output
of the 14-month Fathom research program.
