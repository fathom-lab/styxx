# The Geometry of Meaning Across Machines and Minds
### A five-experiment, pre-registered arc — and what survived the discipline
**2026-06-03 · fathom-lab/styxx · `papers/real-convergence/` + `papers/ai-human-alignment/`**

## Abstract
We asked, as a *measurement*, a 2,500-year-old question: is there a shared structure of meaning, and
does it span artificial and biological minds? Across five pre-registered experiments on a fixed cohort
of small language models (70M–3B) plus real human behavioral and neural data, we find: **(1)** real,
independently-trained LLMs converge to a shared semantic geometry — but the convergence is **data-/
family-driven, not scale-driven**; **(2)** that geometry aligns with the **human mind** — behaviorally
(human odd-one-out judgments) and **neurally** (fMRI) — about **as well as 1.5M human judgments predict
the brain, and better than sentence embedders built for similarity**; **(3)** the neural match is
**genuine meaning, not pixels** (it survives a vision-model control) but is **modest, quality-gated,
and the shared semantic-visual core** — not a unique or mystical effect. The throughline is method:
four times the discipline retracted our own best headline. What is left is smaller than the hype and
true.

## The arc (each step pre-registered; gate frozen before data)

**1 · Real-model convergence** (`real-convergence/`). Six independently-trained LLMs over real concepts,
contextual-template RSA, lexical-controlled, MiniLM-anchored. Pre-registered bare-word test → NULL,
but its positive control failed → measurement-limited. Strengthened measurement → **convergence
confirmed out-of-sample** (cross-family partial-lexical RSA 0.258, control ≈ 0). **Honest shape:**
universal in *direction* (every pair above chance), heterogeneous in *magnitude* (Llama↔Gemma 0.93;
Phi-3.5 an outlier) — outliers track **training data**, not size.

**2 · Scale law** (`real-convergence/`). Does convergence rise with scale (the PRH "mid-climb"
hypothesis we ourselves floated)? Three controlled ladders (Pythia gold-standard, GPT-2, Qwen).
Pre-registered test **FAILED** (pooled ρ = −0.573; Pythia, the perfectly-controlled ladder, ran
backwards). A uniform-depth map confirmed: **no robust scale law in 14M–3B — convergence is data-
driven, not scale-driven.** *This corrected our own same-day hypothesis.* Surviving method finding:
**concept geometry lives near the final layer**; a fixed mid-layer read-out reads ~5× too low and can
*manufacture* a null.

**3 · AI ↔ human behavior** (`ai-human-alignment/`). VICE human concept embedding (1.5M odd-one-out
judgments, 1,854 objects), 120 matched concepts. **H1 confirmed strong:** LLMs match the human
behavioral geometry at 0.65 (up to 0.74), *as well as MiniLM/mpnet trained on similarity*. **H2/H3 did
NOT survive:** the seductive "cross-model consensus *is* the human structure" is a **geometry-quality
artifact** (human-alignment tracks RDM reliability ρ = 0.76; convergence→human collapses 0.81→0.60
controlling reliability) and the **consensus does not beat a good single model** (no wisdom-of-the-
crowd toward human). *Retracted before it shipped.*

**4 · AI ↔ human brain** (`ai-human-alignment/`). Mitchell 2008 fMRI (9 subjects, 60 nouns; noise
ceiling [0.394, 0.557]). **H1 confirmed:** best LLM RSA-to-brain 0.264 = **67% of the noise ceiling**.
**Three-way:** AI↔brain 0.222 ≈ behavioral-human↔brain 0.247 > embedders 0.16 — **a small LM predicts
the brain about as well as 1.5M human judgments do, and better than trained embedders**; brain- and
behavior-alignment track each other at ρ = 0.98. Driver = **geometry quality, not scale** (gpt2-large/
xl top; Qwen-3B, gemma-2-2b low).

**5 · Vision control — meaning or pixels?** (`ai-human-alignment/`). Mitchell stimuli were word + line-
drawing, so the brain RDM is partly visual. CLIP-image vision RDM + variance partition. **PASS (pre-
stated bar):** AI↔brain *survives* removing the vision model (partial 0.182 → 0.107) → **real meaning,
not a visual artifact.** **Bounded:** the brain RDM is substantially visual; the LLM geometry is itself
~half visual (AI↔vision 0.543); AI adds almost nothing *unique* to brain prediction beyond human
behavior (+0.2% R²). Behavioral-human is the strongest, most-unique brain predictor.

## Scoreboard — what survived vs what we retracted
| claim | verdict |
|---|---|
| Real LLMs converge to a shared semantic geometry (controlled, out-of-sample) | **SURVIVED** |
| LLM geometry matches the human **behavioral** geometry (as well as trained embedders) | **SURVIVED** |
| LLM geometry matches the human **brain** (~⅔ of noise ceiling; ≈ behavior; > embedders) | **SURVIVED** |
| The brain match is **meaning**, not pixels (survives a vision control) | **SURVIVED** |
| Concept geometry lives near the **final layer** (mid-layer read-outs can manufacture nulls) | **SURVIVED** (method) |
| Convergence is **scale-driven** (PRH mid-climb) | **RETRACTED** — data-driven, not scale |
| The cross-model **consensus *is* the human/brain structure** | **RETRACTED** — quality artifact; consensus ⊁ good single model |
| "Universal forms vindicated" / synthetic-geometry headline (prior night) | **RETRACTED** — controlled illustration only |

## The honest bottom line
There **is** a shared geometry of concrete meaning, and it **does** span machines, human behavior, and
the human brain — measurable, lexical-controlled, vision-controlled, judged against the brain's own
noise ceiling. That is real, and it is genuinely remarkable: a 774M-parameter language model, never
shown a brain, lands on the same arrangement of concepts a human brain uses, as well as a million
human judgments do. **But it is not mystical and not unique.** It is the broad semantic-visual core
that any sufficiently-clean learner — silicon or biological — recovers; it is **gated by representation
quality, not scale**; and the cross-model *consensus* holds no special claim to the human truth. The
universal is real and *bounded* — which is the only kind of universal worth trusting.

## Method (the actual contribution)
None of the individual directions is novel (representational convergence: PRH/vec2vec/CKA; AI↔brain:
Mitchell 2008/Schrimpf/Goldstein). The contribution is **how**: every claim pre-registered with a
frozen kill-gate; positive controls load-bearing (a null is only as good as its positive control —
twice a "negative" was a broken measurement we caught); three nested controls (lexical → reliability →
vision), each of which **shrank the claim to its true size**; and **four self-retracted headlines**,
published next to the wins. In a field that rewards overclaiming, the instrument that catches *its own*
overclaims is the asset.

## Limitations & next
Small models (≤3B; VRAM-bound); concrete objects; 53–120 concepts; one fMRI dataset (noisy, 2008,
line-drawing stimuli); behavioral + neural human data both ultimately behavioral-ish; RSA = known-
correspondence similarity, not unsupervised translation. **Next:** wider scale ladder (7B–70B) to
settle the final-layer scale signal; modern **THINGS-fMRI** single-trial betas; a **low-level CNN**
vision model for a cleaner vision/semantics split; data-overlap sweep to test the data-driven account
directly.

## Reproduce
All scripts, pre-registrations, results, and the corrections record are in `papers/real-convergence/`
and `papers/ai-human-alignment/`. Commits 2026-06-03: `a59db86` (convergence), `02b7393` (scale),
`492a849` (behavioral), `0e8bb95` (brain), `583c796` (vision). Prior-night corrections:
`papers/ancient-question-program/CORRECTIONS_2026_06_03.md`.
