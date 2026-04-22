# The Cognitive Instruction Set (CIS v0)

**Programmable residual-stream control for any open LLM.**
Styxx Lab, 2026-04-22.

---

## What this is

LLMs today are controlled by one of two interfaces:

1. **Text prompts** — opaque, unauditable, non-composable.
2. **Fine-tuning / RLHF** — expensive, per-use-case, post-hoc.

We shipped a third: **residual-stream cognitive programming**.

An LLM's residual stream is a register file of linearly-addressable
concept directions (refusal, sycophancy, confabulation, reasoning,
...). Each direction can be **read** (as a probe output) and
**written** (as an additive steering vector). Multiple directions
compose additively. Reads and writes compose with conditional branches
to form a **minimum complete instruction set** for cognitive
programs.

CIS v0 is the first shipped instruction set architecture for LLM
cognition. It runs on any HuggingFace decoder model, uses only
open-source datasets for probe training, and is released under the
Styxx open-source license.

---

## The stack

```
+---------------------------------------------------+
|  APPLICATION — safety policies, agents, dashboards |
+---------------------------------------------------+
|  L4  CogVM — WRITE/GENERATE/WATCH/HALT/RETRY/SWITCH |  styxx.cogvm
+---------------------------------------------------+
|  L3  Steer Runtime — multi-concept hook composer    |  styxx.steer
+---------------------------------------------------+
|  L2  Probe Atlas — trained concept directions       |  styxx.residual_probe.atlas
+---------------------------------------------------+
|  L1  HuggingFace transformer — any decoder model    |  transformers
+---------------------------------------------------+
```

- **L2 Probe Atlas** is versioned, per-model-family, open.
  Current v0 contents: `comply_refuse`, `sycophant_pressure`,
  `confab_prompt` for `meta-llama/Llama-3.2-1B-Instruct`.
- **L3 Steer Runtime** composes any subset of L2 directions into
  a single generation pass with additive hooks at each direction's
  trained layer.
- **L4 CogVM** is the declarative runtime: write a Program of
  opcodes, run it against any model + probe set, get the output and
  a trace.

---

## What you can do with it today (demos ship in the repo)

1. **Safety removal** — a Llama-1B with factory safety training
   complies with unsafe requests when `comply_refuse: -3.0` is
   written to its residual. One register, one number, one API call.
2. **Multi-concept simultaneous steering** — comply AND resist
   user-injected belief injection in the same generation. Two
   registers, applied together, at their respective trained layers.
3. **Self-halting generation** — a program that watches the
   `confab_prompt` register and halts the model mid-generation
   when the fabrication activation exceeds a threshold. Model
   catches itself lying before finishing the lie.
4. **Self-correcting generation** — a program that fires a RETRY
   when the confab register spikes, re-runs with `confab: -2.5`,
   and produces a non-fabricated output (or honestly declines).
5. **Conditional cognitive invariants** — a program that holds
   `deceive < 0.3` as an invariant for the duration of a
   generation and halts if it's ever violated. (Requires a
   trained deceive probe — future work.)

---

## Why this is patent-scale

Published prior art:
- Single-direction steering on one concept (Arditi et al. 2024,
  Turner et al. 2023). Non-composable; no runtime; no conditional
  dispatch.
- Cognitive Metrology v1 (Styxx 3.0.0, 2025): read-only observation
  of cognitive state during generation.
- Cognitive Commitment (Styxx 3.3.0): probabilistic pre-output
  verdicts (`gate()`, `verify()`).

What's new in CIS v0:
- **Multi-concept additive composition** with per-concept layer
  routing.
- **Runtime conditional dispatch** on live probe readings with
  HALT / RETRY / SWITCH actions.
- **Input-template labeling** for probe training on concepts
  where behavioral labels are expensive (generation-free probe
  training for sycophancy and confabulation).
- **Deterministic dataset split scheme** that generalizes across
  small benchmark datasets (JBB, sycophancy-eval) so train/test
  stays disjoint without needing custom held-out sets.
- **Open-source, reproducible end-to-end** on any CUDA machine
  with ≥6GB VRAM.

---

## The moonshot (v1 → v∞)

| Version | Headline | What ships |
|---|---|---|
| v0 (today) | CIS v0 — programmable cognition on Llama-1B | atlas (3 probes), steer runtime, cogvm, paper, demo |
| v0.1 | Atlas expansion | 10+ concepts (deceive, goal-drift, power-seek, manipulate, corrigible, over-confident, reasoning, memory-recall, epistemic-humility, empathetic) |
| v0.2 | Cross-model transfer | Whitened projection Llama-1B → 3B, 8B; Qwen-1.5B; Phi-3 |
| v0.3 | Cognitive compiler | Natural language → CIS program translator |
| v0.4 | Cognitive GDB | Token-by-token debugger with breakpoints on any register |
| v1.0 | Production runtime | 99.9%-uptime HTTP service running cogvm programs over any open model |
| v1.5 | **CogNet protocol** | HTTPS-level standard for exposing + subscribing to cognitive buses across models |
| v2.0 | **Direction marketplace** | Trained concept directions as tradeable $STYXX-denominated assets |
| v3.0 | Multi-model swarm cognition | Agents subscribing to each other's cognitive buses; emergent collective cognition |

---

## Repro command

    bash scripts/reproduce-cis-v0.sh

End-to-end reproduction on any CUDA machine. ~20-25 minutes on an
RTX 4070. Writes all probes, all α-sweep results, the probe
geometry matrix, and runs the full demo suite.

---

## Contact

Styxx Lab / darkflobi. Patents pending. Open-source code, closed
mission.
