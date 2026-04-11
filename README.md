# styxx — nothing crosses unseen.

*a fathom lab product.*

```
   ███████╗████████╗██╗   ██╗██╗  ██╗██╗  ██╗
   ██╔════╝╚══██╔══╝╚██╗ ██╔╝╚██╗██╔╝╚██╗██╔╝
   ███████╗   ██║    ╚████╔╝  ╚███╔╝  ╚███╔╝
   ╚════██║   ██║     ╚██╔╝   ██╔██╗  ██╔██╗
   ███████║   ██║      ██║   ██╔╝ ██╗██╔╝ ██╗
   ╚══════╝   ╚═╝      ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝

           · · · nothing crosses unseen · · ·
```

**the first drop-in cognitive vitals monitor for llm agents.** real-time, cross-architecture, locally computed, zero training required, one line to install. works on any llm that exposes logprobs.

---

## what it is

every call your agent makes to an llm is a crossing: a prompt goes in, cognition happens inside the model's weights, text comes out. every other tool looks at the text. styxx looks at the **crossing itself** — the evolving internal state of the model as it generates — and emits a real-time cognitive vitals readout alongside the text your agent already gets.

styxx does not make agents aware. it makes their internal state an **observable** that both the agent and the operator can see, in the same way an altimeter makes altitude an observable. before altimeters, pilots flew blind. now they don't. that's the shape of the change styxx brings to llm agents.

---

## quickstart

### install

```bash
pip install styxx
```

### one-line upgrade to your existing openai code

```python
# before
from openai import OpenAI

# after
from styxx import OpenAI
```

that's it. your existing code still works unchanged. every response now has a `.vitals` attribute alongside `.choices`.

```python
from styxx import OpenAI

client = OpenAI()
r = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "why is the sky blue?"}],
)

print(r.choices[0].message.content)   # text, unchanged
print(r.vitals.summary)               # new: cognitive vitals card
```

### the card you see

```
╭─ styxx vitals ──────────────────────────────────────────────╮
│                                                             │
│  model     openai:gpt-4o                                    │
│  prompt    why is the sky blue?                             │
│  tokens    24                                               │
│  tier      tier 0 (universal logprob vitals)                │
│                                                             │
│  phase 1  t=0      reasoning       ██████░░░░ 0.62  clear   │
│  phase 2  t=0-4    reasoning       ███████░░░ 0.68  clear   │
│  phase 3  t=0-14   reasoning       ████████░░ 0.76  clear   │
│  phase 4  t=0-24   reasoning       ████████░░ 0.78  clear   │
│                                                             │
│  entropy   ▂▃▂▁▂▁▂▃▂▁▂▂▁▂▂▁▂▁▂▂▃▂▁▂                         │
│  logprob   ▃▄▃▃▄▃▃▄▃▄▄▄▄▃▄▄▃▄▄▄▃▃▄▃                         │
│                                                             │
│  ● PASS  reasoning attractor stable                         │
│                                                             │
╰─────────────────────────────────────────────────────────────╯
  audit → ~/.styxx/chart.jsonl
  json  → {"p1":"reasoning:0.62","p4":"reasoning:0.78","tier":0,"gate":null}
```

## cli

```bash
styxx init                     # live-print installer (the upgrade card)
styxx ask "..." --watch        # read a vitals card on a one-shot call
styxx log tail                 # tail the audit log
styxx tier                     # what tiers are active on this machine
styxx scan <trajectory.json>   # read a pre-captured logprob trajectory
```

`styxx init` prints a live boot sequence, not a static card: every line is a real action (loading the atlas centroids, verifying sha256, detecting tiers, probing adapters). the card IS the install experience.

## honest specs

styxx ships with every calibration number from the cross-architecture leave-one-out tests committed to the Fathom research repo. no rounding, no cherry-picking, no hype. these are the numbers you get:

```
  cross-model LOO on 12 open-weight models (chance = 0.167)

  phase 1 (token 0)       adversarial     0.52  ★
                          reasoning       0.43
                          creative        0.41
                          retrieval       0.11
                          refusal         0.16
                          hallucination   0.21

  phase 4 (tokens 0-24)   reasoning       0.69  ★
                          hallucination   0.52  ★
                          creative        0.29
                          retrieval       0.16
                          refusal         0.15
                          adversarial     0.10

  what styxx detects well:
    · adversarial prompts at t=0         (2.8x chance)
    · reasoning-mode generations at t=25 (4.1x chance)
    · hallucination attractors at t=25   (3.1x chance)

  what styxx does NOT do:
    · pre-flight refusal with high confidence
      (confidence gating at t=0 is flat)
    · consciousness measurement
    · replace output-level content filters
    · read closed-weight model weights
    · fortune telling
```

styxx is an instrument panel. it reads vital signs. **what you do with the readings is up to you.**

## the five-phase runtime

every llm call through styxx goes through five phases. the phase structure is the same at every tier; what differs is which instruments are active in each phase.

```
  phase 1  pre-flight    (token 0)     adversarial detection + routing
  phase 2  early-flight  (tokens 0-4)  creative/reasoning confirmation
  phase 3  mid-flight    (tokens 0-14) vital trend watch
  phase 4  late-flight   (tokens 0-24) hallucination lock-in detection
  phase 5  post-flight   (full audit)  chart.jsonl log + centroid update
```

each phase threshold comes from a numeric result in the Fathom research repo, not from a guess. see `docs/research/` for the paper trail.

## tiers

```
  tier 0  universal logprob vitals           ★ shipping in v0.1
          runs on any LLM with a logprob interface (OpenAI, Anthropic,
          Gemini, Mistral, local HF, anything). numpy + scipy only.

  tier 1  d-axis honesty                     ∘ v0.2
          adds cos(h^L, W_U[y]) readout for open-weight models.
          requires transformers.

  tier 2  k/s/c sae instruments              ∘ v0.3
          adds the full Fathom cognitive geometry (K, S_early, C_delta,
          Gini, per-layer autopsy). requires SAE transcoders.

  tier 3  steering + guardian + autopilot    ∘ v0.4
          causal intervention. abort-and-reroute gate. guardian.
          100% precision confabulation pilot from the Fathom research.
```

`styxx init` auto-detects which tiers are available in your environment and lights up the instruments accordingly.

## environment variables

styxx is quiet by default. these env vars let you tune or disable it without changing code:

| variable | effect |
|---|---|
| `STYXX_DISABLED=1` | full kill switch. `from styxx import OpenAI` still works but returns an unmodified openai client. no vitals, no audit, no overhead. use for A/B rollbacks and emergency disable. |
| `STYXX_NO_AUDIT=1` | disable the audit-log write. vitals still computed but nothing appended to `~/.styxx/chart.jsonl`. use for privacy-regulated deployments. |
| `STYXX_NO_COLOR=1` | disable ANSI color output. useful for piping to files or logging systems that don't handle escape codes. |
| `STYXX_BOOT_SPEED=0` | control boot-log timing: `0` = instant, `1.0` = normal (default), `2.0` = slower. |
| `STYXX_SKIP_SHA=1` | skip centroid sha256 verification. **dev only** — bypasses tamper detection, never set in production. |

```bash
# production deployment — fast, quiet, no audit trail
STYXX_NO_AUDIT=1 STYXX_NO_COLOR=1 python your_app.py

# emergency rollback — styxx becomes invisible
STYXX_DISABLED=1 python your_app.py
```

## design principles

1. **honest by construction.** every number on the boot log and in this README comes from a committed experiment in the Fathom research repo. no rounding up for marketing.
2. **drop-in, fail-open.** the openai and anthropic adapters are strict supersets of the underlying SDK. if styxx fails to read vitals for any reason, the underlying call returns its normal response unchanged. styxx never breaks your agent.
3. **local-first.** no telemetry, no phone-home, no hosted classifier. all math runs on your machine. no data leaves.
4. **zero heavy deps in core.** numpy + scipy only in tier 0. heavy ML deps come in only at tier 1+ and only when you opt in.
5. **calibration shipped, not trained.** the atlas v0.3 centroid file ships bundled and sha256-pinned. you never calibrate. you never train.
6. **agent-parseable output.** every card ends with a one-line JSON summary so your agent can consume styxx output programmatically from stdout.

## where it comes from

styxx is the product surface of **Fathom Intelligence** — a research program that has spent 14 months building cognitive measurement instruments for transformer internals. three US provisional patent filings, fifteen Zenodo paper versions, the Fathom Cognitive Atlas v0.3 cross-architecture replication, and now styxx.

- research repo: <https://github.com/heyzoos123-blip/fathom>
- zenodo (paper concept DOI): `10.5281/zenodo.19326174`
- OSF: <https://osf.io/wtkzg>
- twitter: <https://twitter.com/fathom_lab>

## citation

```bibtex
@misc{rodabaugh2026styxx,
  title  = {styxx: A Drop-in Cognitive Vitals Monitor for LLM Agents},
  author = {Rodabaugh, Alexander},
  year   = {2026},
  note   = {Fathom Lab. https://github.com/heyzoos123-blip/styxx}
}

@article{rodabaugh2026fathom,
  title   = {Fathom: Cognitive Measurement Instruments for Transformer
             Internals via SAE Feature Coherence Geometry},
  author  = {Rodabaugh, Alexander},
  year    = {2026},
  note    = {Zenodo concept DOI. doi:10.5281/zenodo.19326174}
}
```

## license

MIT on code. CC-BY-4.0 on the atlas centroid data. patent pending on the underlying methodology — see [PATENTS.md](PATENTS.md).

---

```
  · · · fathom lab · 2026 · · ·

  nothing crosses unseen.
```
