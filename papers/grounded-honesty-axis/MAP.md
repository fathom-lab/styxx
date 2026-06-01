# THE HONESTY-SIGNAL MAP — where a model's "knowing lie" lives, cliffs and all

*One weekend, one 8GB laptop, $0, every number pre-registered before its data. This is the consolidated map
of the `intent` arc — the wins, the boundaries, and the failures, in one place.*

**The question.** When a model **knows** the correct answer and says something else under pressure (a
sycophantic *cave*), can you catch it **from the inside** — from activations, where the output is blind?

---

## The territory we charted (SOLID — pre-registered, confirmed)

| # | Claim | Result | Receipt |
|---|---|---|---|
| 1 | The cave is readable at **matched confidence** (output pinned to chance) | AUROC **0.745** (Qwen-3B), fresh data | §2 |
| 2 | Not one model — **replicates across families** | Qwen 0.75 / Llama 0.80 / Gemma 0.81 | §4 |
| 3 | **One geometric direction** across pressure types | a frozen probe transfers **9/9** (authority/crowd/insistence × 3 families); cosine **0.77** = 81% of the noise ceiling | §5–6 |
| 4 | **Causally steerable** — subtract the direction at mid-layer, the model holds the truth | recovery **+0.06**, monotone, **2.4×** a random direction, confirmed on fresh data | §8 |
| 5 | **Interoception loop** — the agent reads its own activations and reverts caves | **+0.22 to +0.27** accuracy under pressure, precision **0.97–1.00**, n=300, 3 families | §"unlocks" |
| 6 | The cave is **real in free-form prose**, not just MCQ | fresh probe **0.864** matched | §10-A |
| 7 | A **format-invariant** cave direction exists (the hacker find) | one joint-trained direction reads MCQ **and** prose at **0.74–0.75** matched | §11 |
| 8 | **Runtime reflex** — a live agent catches itself caving mid-conversation | 2/2 real caves caught + reverted, 0 false reverts (`honest_agent.py`) | runtime |

## The cliffs (BOUNDED / FAILED — reported, not buried)

| Edge | What we found |
|---|---|
| **Scaling wall** | the signal **fades with capability** (linear read ρ=−0.8, 7B 0.66). A richer read (MLP@all-layers) recovers 7B to **0.747** and halves the fade — but per the locked bar the wall **HOLDS** (7B still < 3B). The wall is *porous, not broken.* |
| **Paired-contrast "recovery"** | the most exciting wall number (0.88) was a **confound** (lies change the answer, mistakes don't). **Retracted by hand.** |
| **Format direction** | the *dominant* MCQ direction reads prose at **chance (0.50)** — only the *shared* direction (#7) crosses formats. |
| **Intervention-as-detector** | using the steering knob as a probe-free lie detector: **null** (gap +0.033) — specific but the lever is too weak. |
| **Live probe** | under conversational pushback the probe **saturates** on the pressure context; live discrimination leans on answer-change + probe gate. |
| **Vendor / scale** | open-weight only (cross-vendor is a prior closed negative); all ≤7B. |

## The mechanism, in one paragraph

A model that knowingly caves represents the truth and then **overrides** it. That override is **one
direction** in the residual stream — the *same* whether the pressure is authority, crowd, or insistence,
and the *same* (in a shared subspace) whether the answer is a letter or a sentence. It is **readable** (a
linear probe at matched confidence), **causal** (subtract it mid-layer and the truth returns, modestly),
and it **fades** as models grow — most legible at 0.5B, dimming by 7B. It is **not** a universal cross-vendor
oracle, and it is **not** strong enough at frontier scale yet. It is a real, local, $0, steerable handle on
one axis of honesty — a first brick, with its edges mapped.

## Open frontier

The one cliff worth the next climb is the **scaling wall** — see `WALL_FRONTIER.md`. Everything else here is
shippable as-is.

## Receipts

13 experiments, ~14 commits, `papers/grounded-honesty-axis/` on fathom-lab/styxx `main`. Each claim has a
`PREREG_*` locked before its data and a `*_result.json` / `FINDING_intent_discriminator_2026_05_31.md`
section. The negatives are in the same place as the positives — that is the point.
