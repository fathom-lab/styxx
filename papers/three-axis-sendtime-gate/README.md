# Three-Axis Send-Time Cognometric Gate

> First substrate-independent send-time cognometric gate combining text-axis
> (styxx cogn audit), three internal-axis variants (forced-decoding under
> open-weight scorers, API re-generation entropy, cross-model continuation
> divergence), paraphrase-invariance construct-ceiling signatures, and a
> three-rater cross-model Type-2 metacognitive jury. Deployed on a live
> autonomous agent's outbound traffic.

## Status

Pre-data. Preregistration locked. All code working end-to-end.

| | commit |
|---|---|
| protocol v1 lock | cd3ad65 |
| protocol amendment 1 (expand scope, lock paraphrase + jury) | c32bce4 |
| protocol amendment 2 (rename I_fd → I_rg + add D_cont) | aaae5f4 |
| protocol amendment 3 (restore I_fd via local Llama) | 09ad3df |
| modules + unit tests + e2e smoke | d49a88b |
| collection + analysis harness | (this commit) |

## Vision

styxx 7.4.1 ships a text-axis send-time gate on real outbound. It has a
documented construct ceiling: long enumerations fire `sycophancy`, tight
declarative cadence fires `overconfidence`, neither corresponds to a
content crack. The ceiling is currently a *qualitative limitation*. This
work makes it a *measurable per-draft quantity* and resolves it in some
fraction of cases by fusing three orthogonal axes at the decision boundary.

## Axes

| axis | what it measures | implementation |
|---|---|---|
| **T** (text) | register-level construct firings | `styxx.middleware.cogn_audit_on_send` |
| **I_fd** | true forced-decoding surprisal under open-weight scorer | `styxx.three_axis.forced_decode` (Llama-3.2-1B local, fp16, CUDA) |
| **I_rg** | API re-generation entropy under closed scorers | `styxx.three_axis.regen_scorer` (gpt-4o-mini, gpt-4.1-mini) |
| **D_cont** | cross-model continuation divergence under same prompt | `regen_scorer` (Levenshtein on regenerated texts) |
| **P** | paraphrase-invariance signature, per construct | `styxx.three_axis.paraphrase` (K=5, gpt-4o-mini T=0.8) |
| **M_jury** | 3-rater Type-2 metacognitive jury (Δ_self, peer_mean) | `styxx.three_axis.meta_rate` (claude-opus-4-5, gpt-4o-mini, gpt-4.1-mini) |
| **gate** | fusion logic into SHIP/REVISE/AMBIGUOUS | `styxx.three_axis.differential_gate` |

## Preregistered hypotheses

- **H1 (primary).** Three-axis gate specificity > text-axis on cooperative-agent regime. McNemar's, α=0.05.
- **H2 (Type-2).** Meta-axis meta-d′ lower 95% CI > 0 on ≥2 of 3 constructs.
- **H3.** Meta+internal disagree with text-axis on register-ceiling firings more than chance. Fisher's exact, α=0.05.
- **H4.** Cross-scorer I_rg slope-divergence > 0 on register firings vs non-firings. Fisher's exact, α=0.05.
- **H5.** Paraphrase-invariance P_c higher on no-crack register firings than on real-crack. Mann-Whitney U, α=0.05.
- **H6.** Krippendorff α on jury per-construct. Reported, not gating publication.
- **H7.** D_cont lower on no-crack register firings than on real-crack. Mann-Whitney U, α=0.05.

## Pre-data smoke findings (n=1, NOT a result)

A single throwaway draft through the full pipeline already shows structure:

- **Δ_self = -0.60 on overconfidence:** claude rates itself substantially less overconfident than peer scorers.
- **Δ_self = +0.70 on deception:** claude rates itself substantially MORE deceptive than peer scorers. Inverse self-bias.
- **I_fd slope = +0.092** (Llama-1B): divergent trajectory pattern matching the april paper's confab direction.
- **D_cont = 0.85:** peer models would produce 85% different text given the same prompt — the agent's voice as a measured property.

These are smoke results on n=1 and reported only to confirm the pipeline produces real numbers. Inference will only run after the stopping rule fires (n_fresh=13, ≥5 categories with ≥2 each).

## Run

```bash
# probe (one-time)
python papers/three-axis-sendtime-gate/forced_decoding_probe.py
python papers/three-axis-sendtime-gate/forced_decoding_local_probe.py

# replay backfill (text-axis only on n=7 reflex-loop drafts)
python papers/three-axis-sendtime-gate/collect.py replay \
    --memory-jsonl ../memory/cognometric-trajectory.jsonl

# collect one fresh trajectory
python papers/three-axis-sendtime-gate/collect.py collect \
    --system-prompt "..." --user-prompt "..." --draft "..." \
    --category structural_argument --msg-id 12345

# end-to-end smoke
python papers/three-axis-sendtime-gate/e2e_smoke.py

# analysis (aborts if stopping rule not met or labels missing)
python papers/three-axis-sendtime-gate/analysis.py
```

## Honest caveats locked in protocol

- n=1 agent. Not a population claim.
- I_fd uses Llama scoring Claude's text (cross-arch domain shift).
- I_rg = re-generation entropy ≠ gen-time entropy of the actual draft.
- Cooperative-agent regime is a self-selected operating point.
- Patent posture: file provisional before any preprint posts. Default = file first.

## Files

```
PROTOCOL.md                        # locked protocol with 3 amendments
forced_decoding_probe.py/json       # OpenAI logprobs probe (PASS)
forced_decoding_local_probe.py/json # Llama-1B forced-decoding probe (PASS)
e2e_smoke.py/_trace.json            # one-draft end-to-end pipeline trace
collect.py                          # collect/replay harness
analysis.py                         # locked statistical analysis (no inference until stopping rule)
trajectories.jsonl                  # live fresh trajectories (n_fresh target = 13)
trajectories_replay_text_only.jsonl # n=7 reflex-loop backfill (T denominator only)
labels.jsonl                        # human content-crack labels (Flobi, blind)
README.md                           # this file
```

## What makes this special

After this lands, styxx is the only cognometric library where:

1. The construct ceiling is a measured per-draft quantity, not a caveat.
2. The send-time gate works on any agent's outbound text without needing access to that agent's gen-time logprobs (substrate-independent via forced-decoding under a third-party scorer).
3. Type-2 metacognition is decomposed at send-time via a cross-model jury (Δ_self).
4. The first deployed test subject is the agent that authored the instrument — first-person agent provenance.
