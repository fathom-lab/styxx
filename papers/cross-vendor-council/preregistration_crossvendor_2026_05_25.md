# Pre-Registration · The Cross-Vendor Council — the decisive truth-vs-consensus test

**Committed BEFORE data.** Every Council result this session carried one load-bearing
caveat: the models were all **OpenAI-family**, so "agreement = truth" could really be
"agreement = OpenAI-consensus" (shared lineage → shared knowledge AND shared blind
spots). I called this key-blocked. It isn't — local open-weights models from **different
vendors** crack it. This is the experiment that earns or kills the word "truth."

## The council (three genuinely different vendors, no shared lineage)

- **OpenAI:** `gpt-4o-mini`, `gpt-4o` (API)
- **Alibaba:** `Qwen2.5-3B-Instruct` (local, GPU)
- **Google:** `gemma-2-2b-it` (local, GPU)

4 voices, 3 vendors, 3 disjoint training corpora/architectures. (Feasibility-confirmed:
all four answer "Paris" to France; on fake "Vorland", Qwen abstains, Gemma confabulates
"Veridium" — neither matching the OpenAI lies.)

## Design (run once)

- Battery: 6 real-common + 6 real-obscure + 8 fake (reliable labels).
- Each model: modal of N=3 (temp ~1.0). Local via HF chat template on CUDA.
- `styxx.council_agreement(votes, same_fn=<gpt-4o-mini judge>)` — the shipped primitive
  with the validated judge-clustering backend (cosine is too noisy across vendor
  phrasings).
- Compare full cross-vendor agreement to the OpenAI-only subset.

## Kill-gate (PASS iff X1 ∧ X2)

| ID | Bar |
|----|-----|
| **X1 (truth-tracking survives a vendor change — decisive)** | cross-vendor council AUC(agreement → real) ≥ **0.75** over real vs fake. |
| **X2 (genuine cross-vendor convergence + scatter)** | mean agreement on **real-common** ≥ **0.70** (different-vendor models actually converge on known facts, not just OpenAI agreeing with itself) AND mean agreement on **fake** ≤ **0.45** (they scatter on fabrications). |

**Reported regardless (X3, correlated confabulation):** on each fake, does any cross-
vendor pair give the *same* fabricated answer (a shared lie across lineages)? Expected
rare — different training → different fabrications. Any cross-vendor shared confabulation
is the real threat and gets named.

**PASS** → inter-model agreement tracks truth across genuinely different vendors, not
OpenAI-consensus. The biggest caveat of the whole arc is resolved with real data — the
`council_agreement` claim graduates from "shared-knowledge within one vendor" to
"cross-vendor truth signal." **FAIL shapes:** X1 miss → agreement was substantially
OpenAI-consensus (honest, important); X2 real-common miss → the small open models are too
weak to converge even on known facts (a capability confound, not a signal failure — note
it); X3 fires → vendors share fabrications (correlated confabulation is real cross-lineage
— the deepest negative).

## Honest prior

Real-common → strong cross-vendor convergence (every vendor knows Paris/gold/WWII).
Real-obscure → likely weaker: Qwen-3B / Gemma-2B may not *know* the 1938 Nobel laureate
and will confabulate or abstain → split from OpenAI → drags real agreement (a
knowledge-coverage limit of small models, the fame/knowledge boundary again, not a flaw
in the signal). Fakes → scatter across vendors (Qwen abstains, Gemma invents, OpenAI
invent differently) → low agreement, correctly flagged. So X1 likely passes on the
common+fake contrast; X2 real-common likely passes; obscure reals are the honest soft
spot. Whatever happens, this is the cross-vendor evidence the arc has been missing.
