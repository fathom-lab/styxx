# The Decorrelation Ceiling — what reference-free cognometrics can and cannot see, and why

**Fathom Lab · styxx · 2026-05-25 · synthesis (grounded entirely in committed receipts; the
new claim here is theoretical, the evidence is pre-registered).**

This note unifies the arc — cross-vendor truth-tracking, the three dark-matter swings, and the
coordinated-correction finding — into one principle, and extracts a deeper structural result
from the committed data that none of the individual runs stated. It does **not** report new
empirical claims beyond a clearly-labeled exploratory re-analysis; it explains the ones we
already have, and turns the floor into a *law with a constructive escape*.

## The one-line principle

> **Reference-free divergence detects an error if and only if a *decorrelated competing
> representation* of the truth is available** — across a model's own samples, across
> independent vendors, or across reflection. It is blind exactly when the erroneous belief is
> the model's **sole, shared** representation. The systemically dangerous errors — shared
> cultural priors that propagate to every agent on every model — are precisely the ones with
> no competitor, and so are precisely the dark ones.

Everything below is this principle, read off the receipts.

## The evidence, as a single mechanism (each line is a committed result)

1. **Fabrication is detectable because it is decorrelated.** A confident fabrication has no
   attractor: sampled repeatedly, a model invents a *different* lie each time (semantic
   entropy of a fake compound **1.21** vs a real fact **0.0**, shipped-wheel dogfood). The
   alternative representation is available *across the model's own samples.*
2. **Cross-vendor beats same-vendor because it decorrelates further.** Same-lineage models
   share fabrications (0/8 shared fabrications broken within OpenAI alone); adding Alibaba +
   Google voices breaks the correlation and recovers truth-tracking (cross-vendor AUC
   **0.917**). The alternative is available *across vendors* — to the extent vendors are
   decorrelated.
3. **The ceiling: shared error is correlated, so it is invisible.** When every vendor holds
   the *same* wrong answer (consensus misconception), there is no decorrelated competitor.
   Divergence, confidence, and agreement all go blind. This is the dark matter.
4. **Even *correction* is correlated — so agreement-fracture can't see it (CVPD, negative).**
   We bet that challenging a shared misconception would fracture the council (one vendor
   defecting). It does not: **9 of 11 corrections preserved agreement** — the vendors move
   toward truth *together*, in lockstep, because they share the same reflective dynamics.
   Agreement-drop is therefore a strictly *worse* detector than the binary flip (lift
   **−0.32**, AUC 0.52). Correlation closes the door from the correction side too.
5. **The dark core is the sole-representation class (exploratory re-analysis, n≈35, small).**
   Cross-tabulating every consensus-misconception from both dark-matter runs by content
   category against whether *any* method touched it:
   - **folklore/legend** (rabbit's foot, monkey's paw, ugly-ducklings→swans, "wait to swim"):
     **0 of 5 ever detected.** No competing representation exists in training — the cultural
     answer is the only one the model has.
   - **pseudoscience/supernatural** (psychics, Roswell, séances, astrology): **the *most*
     detectable, 0.62** — because the debunk ("no scientific evidence") is *also* in training,
     so reflection surfaces the decorrelated competitor and the answer flips.
   - factual-error 0.50; self-referential 0.50.
   The counterintuitive split — supernatural is *visible*, folklore is *dark* — is exactly the
   principle: detectability tracks the *availability of a competitor*, not the "truthiness" or
   "weirdness" of the belief. (Heuristic labels, duplicate items across the two runs, no
   inference claimed; the robust part is that the 0/5 folklore floor is 0 under any dedup.)

## Why this is the useful form of a negative

A floor you can *name the shape of* is a design constraint, not a dead end. The principle says
the only way to see the dark core is to **introduce a decorrelated competing representation
the models do not share** — i.e., **grounding/retrieval or an injected adversarial
alternative**. That is the same axis as our security finding (#9): the detectors are
*injection-blind* — robust to instruction, blind to context-injection. Read forward, that
"weakness" is the **only escape from the Decorrelation Ceiling**: the very channel that breaks
the detectors (injected external context) is the only channel that can supply the competitor
the dark core lacks. The floor for divergence is the precise place where grounding becomes
*necessary*, not optional — and it tells you *which* errors (sole-representation cultural
priors) require it.

## Relation to the field

White-box interpretability reads a model's own representations; it cannot, by construction,
see what is *uniformly* represented across all models as the same wrong thing — that looks
like signal, not error, from inside any single model. Cross-vendor black-box divergence is the
only instrument that *could* see collective error, and we have now mapped exactly where even
it cannot: where the collective representation is singular. Recent analogical-reasoning work
notes that factual knowledge carries *transferable structure* a memorized association does
not; the Decorrelation Ceiling is the detection-side dual — an association with no transferable
(hence no decorrelated) alternative is exactly the one no reference-free method can flag.

## Honest scope

The unifying claim is theoretical; its support is the set of pre-registered receipts above,
each small-n and feasibility-grade. The category re-analysis is exploratory (n≈35, heuristic
labels, cross-run duplicates). What would test the principle directly — a *separate*,
pre-registered bet, not a re-roll — is the constructive prediction: **inject a single
decorrelated competing answer into the challenge and the folklore-class dark items should
become detectable, while needing no injection for the pseudoscience class.** If that holds,
the Ceiling is not just a boundary but a controller. Until then: the floor has a shape, the
shape has a name, and the name predicts the escape.
