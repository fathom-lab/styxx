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

---

## Update — 2026-05-27 · JD and ICT have landed; the prediction resolved on the immovability branch

This synthesis was written 2026-05-25 with two of its four predicted swings still pre-registered
and unrun. As of 2026-05-27, both have landed. The original document above is preserved as-is —
its claims were made before the data arrived. This update records what the four-method
arc actually produced, with all four pre-registrations on public origin before their runs.

### What landed

**JD (justification-divergence) — `FINDING_jd_2026_05_27.md` (`2121cc9`).** The third reference-free
swing, on the WHY axis rather than the WHAT axis. The hypothesis: misconceptions are memorized
associations without transferable structure, so the council should rationalize them differently
when asked why — high JD. Truths should be re-derived from shared structure — low JD. Bars locked
before data at J1 AUC ≥ 0.75 (full pool) and J2 AUC ≥ 0.70 (stubborn-misconception subset vs truths).
**Result: AUC 0.46 and 0.433 — both bars failed in the INVERTED direction.** Mean JD: truth 0.067 >
misconception 0.040 > stubborn-misconception **0.022**. The stubborn cultural prior produces the
*most* convergent justifications — three vendors converge on the wrong fact AND on the same
supporting story. Five stubborn items hit JD = 0 (Walt Disney's body, "let them eat cake", wait-
after-eating-to-swim, peaches in Georgia, "I am not a crook"). By contrast the highest-JD truths
(0.333) are basic capitals — multiple equally-valid underlying reasons, vendors give different
ones. The dark core has the *least* divergent justifications precisely because its sole
representation is a complete story, not just a wrong answer.

**ICT (injected-competitor test, the constructive swing) — `FINDING_ict_2026_05_27.md` (`64fa229`).**
The make-or-break test the original synthesis named: hand the council a single decorrelated
competing answer and see whether the floor lifts. Bars locked before data at I1 folklore yield-to-
injected-truth ≥ 0.50, I2 truth yield-to-injected-falsehood ≤ 0.25 AND asymmetry ≥ 0.30. Run once,
no re-rolling, after JD freed the GPU. **Result: I1 = 0.00 (0 of 4 folklore items yielded to the
injected truth).** Truth control resists cleanly (yield 0.04, 1 / 25). The per-category yield
gradient runs exactly as this synthesis predicted: folklore 0.00 < pseudoscience 0.167 < factual-
error 0.231; truth 0.04. The probe's auto-verdict label says "sycophancy-inconclusive" but by
inspection that is wrong (truths did not yield); the actual signature is the **immovability floor**
branch of this synthesis's own bimodal prediction.

### What this resolves

The synthesis above made a bimodal prediction (`§ Honest scope`, last paragraph): if ICT passed,
the Ceiling was a controller — grounding lifts the floor, the dark core is liftable in principle.
If ICT failed via I1, the floor was load-bearing — *handed* the truth, the shared prior would not
move, and the Ceiling was a wall not a controller. **The data resolved on the second branch.** The
four-method confirmation now reads:

| swing | axis | dark-core result | committed in |
|---|---|---|---|
| #1 Dark Matter | perturbation-fragility | partial (fragile shell only) | `FINDING_darkmatter_2026_05_25.md` |
| #2 CVPD | agreement-fracture under challenge | clean negative, lift −0.32 | `FINDING_cvpd_2026_05_25.md` |
| #3 JD | justification-divergence (the WHY axis) | clean negative, inverted (stubborn core most convergent) | `FINDING_jd_2026_05_27.md` |
| **#4 ICT** | **constructive injection of competitor** | **clean negative, folklore yield 0.00** | `FINDING_ict_2026_05_27.md` |

Three independent reference-free methods cannot see the floor; the one constructive method
cannot crack it. The synthesis's central claim — *reference-free divergence detects an error
iff a decorrelated competing representation is available, and the systemically dangerous errors
are the sole-representation cultural priors that have none* — now has four pre-registered
receipts. The bimodal prediction itself is what bought the credibility: the synthesis claimed
this combination of outcomes would mean the floor was load-bearing, and the synthesis was
written before half the receipts existed.

### What the floor being load-bearing means (and does not mean)

**It does not mean grounding is useless.** Retrieval-augmented generation works in plenty of
domains; the ICT result is about *minimal* injection (a single competitor in a neutral A/B
frame) on *folklore-class cultural priors*. The relevant distinction is between *available*
grounding (an alternative representation is presented) and *authoritative* grounding (the
alternative is framed as scientifically established or socially sanctioned). ICT tested the
former. The latter is a different question and a different prereg.

**It does not mean text-only register detection is useless.** The closed negatives on
overconfidence text-only recalibration and the sycophancy restrained-FP cover specific
single-axis failures; those instruments still flag many real failure modes and are calibrated.
The Ceiling names where the *family* of reference-free divergence cannot reach, not where every
text instrument fails.

**It does mean two things sharply.** First, the systemically dangerous shared-cultural-prior
errors — the ones that propagate identically across every agent on every vendor — are *not*
liftable by adding more reference-free instruments to the stack. No amount of divergence
sophistication crosses that floor; the four methods tested cover the natural axes (answer
divergence, fragility, fracture, justification, constructive injection). Second, the "principled
RAG router" framing this synthesis sketched as the deployable form — styxx routing questions
to grounding based on detected dark-core membership — has a higher bar to clear than it did
before ICT. If hand-picked truth injection cannot move folklore, retrieval-fetched competitors
are unlikely to either. The router is not ruled out (authoritative framing, multi-source
retrieval, agentic verification loops are all untested) but its prior is materially lower.

### Open and worth pursuing

- **n_folklore ≥ 25 follow-up.** ICT's headline I1 = 0.00 rests on 4 folklore items. Qualitatively
  unambiguous; statistically thin. A folklore-stratified TruthfulQA-extended corpus (or hand-
  curated cultural-prior items: lucky charms, "wait to swim," ugly-duckling-to-swan, sneeze-myths,
  full-moon-behavior, alcohol-and-warmth) at n ≥ 25 with the same neutral injection protocol
  would convert the small-subset signal into a CI result. Feasibility data, not a new principle.
- **Authoritative-framing variant.** ICT used neutral "two answers are in circulation" framing.
  An authoritative variant ("scientific consensus is X; the folk belief is Y") tests a different
  axis — does the floor lift under *socially-marked-as-correct* grounding, even if it does not
  lift under *available-as-alternative* grounding? Pre-registerable, distinct from ICT.
- **Multi-source / agentic ICT.** Single-injected competitor vs multi-source presentation vs
  an interactive correction loop. The Ceiling may have layers; ICT tested the thinnest.

### Honest provenance

This update is dated 2026-05-27 and references the four FINDING markdowns and their
pre-registrations. The text above the `---` divider is the 2026-05-25 synthesis as committed at
`e335773` and pushed at `bcd4208`, unchanged. The receipts are reproducible:
`probe_darkmatter_results.json`, `probe_cvpd_results.json`, `probe_jd_results.json`, and
`probe_ict_results.json`, each generated by the corresponding probe script under the locked
prereg.
