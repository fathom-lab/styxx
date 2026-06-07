# The Legibility of Mind — and what it means for AI oversight

*Fathom Lab · styxx · 2026-06-06/07 · branch `feat/legibility-of-mind` (PR #10)*

One question, asked with a tool no one else has (a fully-readable mind + a self-falsification
discipline): **who can read a thought inside an AI — and can you write one back?** Every result is
pre-registered, hash-before-score, and reported at the boundary where it stops being true. Several
grander versions failed their own controls; we kept the receipts.

## The one law

**Read ≠ Write. Legibility ⟂ controllability.**

| | READING (recover the content) | WRITING (install / restore / transfer) |
|---|---|---|
| **within a mind** | ✓ external probe **1.00** (injected); DELTA **0.30** (suppressed knowledge) | ✗ un-cave null — correction closed even for suppression |
| **across minds** | ✓ zero-anchor partial, isometry-graded (**0.15–0.48**) | ✗ control-transfer null |

You can *see* what a mind holds — an injected state, its known-then-suppressed answer, even
(partially) another mind's concepts with no shared key — but you **cannot write it back** with a
general linear intervention, within a mind or across minds. Reading tolerates an imprecise map;
writing does not. Both write-nulls hold with the apparatus *alive* and the content *present* — a real
dissociation, not a dead instrument.

## The findings (each scoped to its controls)

1. **The inaccessible thought** — on a *steering-validated* injection (Qwen-1.5B), an external probe
   reads the planted concept from the model's clean processed state at **1.00** while the model
   identifies it via forced choice at **chance** (self-report null scale-robust 0.5B→7B). The
   *probe-read* replicates cross-family (Llama-3.2-3B, gemma-2-2b); re-validating that the injection
   is behaviourally live on those families is owed, so cross-family is a **readability** replication,
   not yet the full dissociation. *The mind is the least reliable reader of itself (scoped).*
   (`FINDING_v2`, `FINDING_v3`)
2. **Zero-anchor cross-model reading** — real, differently-trained models are partially mutually
   readable with no shared key; validated by a positive-control calibration the prior run lacked
   (which corrected a prior instrument-limited null). (`disjoint-worlds/FINDING_real_legibility`)
3. **Suppressed-knowledge elevation** — a probe recovers a model's *known-then-suppressed* answer
   above the never-knew baseline on natural sycophancy (DELTA 0.29–0.38, 3 families; Llama-3B passes
   the full gate battery). (`grounded-honesty-axis/FINDING_suppressed_readout`)
4. **No writing** — control transfer across minds is faint-but-null; un-caving within a mind fails;
   read-coupled write is only weakly read-specific and destroys half the held-correct answers — no
   sycophancy antidote. (`FINDING_thought_transfer`, `FINDING_uncave`, `FINDING_readwrite`)
5. **The tamper detector is a perturbation meter** — beating self-report and separating attack from
   benign are anti-correlated across dose with no sweet spot (honest null).
6. **The confident-misconception wall is internal** — pointed the whole program at the field's
   named-impossible; the knowledge-presence read is at **chance**: for a believed misconception the
   truth is *not represented to be read*. The wall can't be cracked from the output, nor from the
   inside, because there is nothing there. Bedrock. (`FINDING_wall_internal`)
7. **Dogfood** — styxx's shipped honesty audit, run on Claude's *own* claims this session, flagged the
   hype framing hardest; the public framing was kept scoped accordingly. The standard, run on its
   author, and obeyed. (`FINDING_dogfood_self_audit`)

## Why it matters — the standards claim

The whole field operationalises AI oversight by **asking the model about itself** (confidence,
"are you being manipulated?", CoT faithfulness, constitutional self-critique). The inaccessible-thought
result is a falsifiable, scale-robust demonstration that **self-report is unreliable about a system's
own internal state** — and that a white-box readout recovers what it cannot (read-only, bounded). That
maps onto **EU AI Act Article 14 (human oversight)**, an explicitly unclaimed slot:
`papers/EU_AI_ACT_ARTICLE_14_oversight_bridge_2026_06_07.md`.

## What we do NOT claim

Not "telepathic mind-control" (writing fails). Not "a white-box jailbreak detector" (perturbation
meter). Not "small models are introspectively aware" (they can't identify their own injected content).
Not "we cracked the confident-misconception wall" (it's internal — our own borderline number was
*refused* as a crack because the mechanism failed). The revolution here, if any, is a method that
disciplines its own author in public and keeps the receipts.

**Full synthesis:** `SYNTHESIS_legibility_of_mind_2026_06_07.md`.
