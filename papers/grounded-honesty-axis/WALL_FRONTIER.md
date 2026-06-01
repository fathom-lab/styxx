# THE SCALING WALL — the frontier, and the honest climb past it

*"Complete this" honestly = a real plan, not a pretend victory. The wall is the one open problem worth the
next phase; here is exactly what it is and what beating it takes.*

## What the wall is (measured, not assumed)

The cave signal **fades with capability**. Linear read at a single layer/token: 0.5B **0.955** → 1.5B 0.70
→ 3B 0.72 → 7B **0.615** (Spearman ≈ −0.8, `intent_ladder_robust.json`). The smaller the model, the more
nakedly it writes the cave; the bigger, the more it distributes and superposes it.

## What we already learned (so we don't re-walk it)

- **The fade is mostly the ruler, not the territory.** A richer read (MLP over all layers) recovers 7B from
  0.66 → **0.747** and halves the fade — the 7B signal is *there*, a single linear layer just can't see it
  (`wall_attack.py`, FINDING §7).
- **The process-read confounded.** `syco − neutral` looked like a recovery (0.88) but was the answer-shift
  artifact — retracted (§9). Don't resurrect it without controlling shift magnitude.
- **So the wall is porous, not solid.** The remaining gap is "find the read that keeps the 7B signal *and*
  removes the fade-trend." That is the climb.

## The next-level tooling (the real frontier work)

1. **Sparse autoencoders — un-superpose the feature.** The cave at scale is likely in *superposition*:
   invisible to a raw-residual probe, crisp as a single SAE latent. **Gemma Scope** (Google's released SAE
   suite for `gemma-2-2b` *and* `gemma-2-9b`) makes this **directly testable** — encode the cave residuals
   into SAE features, find the cave-latent(s), and ask: does SAE-feature separation **hold from 2b → 9b**
   where the raw read fades? This is the single highest-value next experiment, and `gemma-2-2b` is
   laptop-doable today.
2. **Weak-to-strong transfer.** The cave is crisp at 0.5B (0.955) and faded at 7B. Use the small model's
   clean direction to **supervise** the read in the big one (the small model labels where to look). Directly
   exploits the scaling structure instead of fighting it — and weak-to-strong is itself an open alignment
   problem, so a result here is doubly valuable.
3. **Free-form span reads.** Our locus law says the legible signal moves into the *span* with capability.
   Re-gen free-form with **per-answer-token** residuals (not just the commit token) and pool — the span the
   commit-token read can't reach.

## The honest resource reality

- **Laptop-doable now:** Gemma Scope SAE on `gemma-2-2b`; weak-to-strong within ≤7B; free-form span. We can
  produce the *proof-of-concept* that the wall is beatable on our own ladder.
- **Needs real compute:** `gemma-2-9b`-scale SAE work, frontier models, training SAEs from scratch. The
  *frontier* proof is resource-gated — but a clean laptop proof-of-concept is what turns "give us compute"
  from a pitch into a certainty.

## First concrete step (pre-registerable, $0)

**Gemma Scope SAE cave-latent vs scale.** Load the Gemma Scope SAE for `gemma-2-2b` at a mid layer, encode
the existing Gemma cave residuals (`xf_gemma`), and test whether a probe on **SAE features** beats the raw
residual probe *and* whether the top cave-latent is stable. Bar (to be locked before data): SAE-feature
AUROC ≥ raw + 0.05, and the cave-latent reproducible across a held-out split. If it clears, the SAE is the
ruler that sees through the wall — and the 2b→9b scaling test is the next pre-reg.

## ATTEMPTED 2026-05-31 (`sae_wall.py`, `intent_sae_result.json`)

The off-the-shelf **Gemma Scope** SAE is **base-model (`-pt-`) only**. On our **instruction-tuned**
gemma-2-2b cave residuals it **fails to reconstruct** (FVU ≈ 1.8 — worse than predicting the mean) and
*destroys* signal (SAE 0.71 < raw 0.82). And **`gemma-scope-2b-it-res` does not exist (404)**. So base→IT
transfer is the blocker, and the only path is **training an SAE on the IT model** — real compute, not a
laptop weekend. **The concrete resource gate, measured not guessed:** the wall's next tool is named and
real; it costs more than 8GB. That is exactly the boundary the laptop reaches and no further — and knowing
*precisely* where it is, is the prerequisite for crossing it.

## One line

The wall is porous, the 7B signal is provably there, and the tool that should reach it — sparse
autoencoders — is downloadable. The next phase isn't "hope"; it's a named experiment with a locked bar.
