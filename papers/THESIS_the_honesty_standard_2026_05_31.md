# The Honesty Standard

### A falsifiable instrument for AI honesty — and the practice of turning it on its makers

*Fathom Lab · 2026-05-31 · the synthesis of the styxx grounded-honesty arc*

---

## The claim, in one sentence

**Trustworthy AI does not require competence; it requires calibrated honesty about the limits of
competence — and that honesty can be measured, falsified, and held to a scientific standard, including
on the system doing the measuring.**

Everything below is in service of that one claim, and — this is the point — everything below is
stated at the boundary where it stops being true.

---

## Why this is the only problem that matters

Every domain people actually want to hand to AI — diagnosis, law, science, money, autonomous agents
acting unsupervised — is gated on a single question that has nothing to do with how smart the model
is:

> When the system is wrong, does it *know*, and will it *say so*?

A model that is right 95% of the time and confidently wrong the other 5% — with no signal which is
which — cannot be trusted with anything that matters, because you cannot tell the 5% from the 95% at
the moment of use. Competence without calibrated honesty is not progress toward trustworthy AI; it is
a faster way to be confidently misled.

So the bottleneck is not capability. It is **honesty about the limits of capability** — and honesty,
unlike intelligence, is something you can define operationally, measure, and falsify.

That is what this work is. Not a hallucination detector. An attempt to make AI honesty a *measurable,
falsifiable, standards-grade* property — and to prove the standard is real by surviving it ourselves.

---

## What we built, and what is actually true

The instrument (`styxx`) is a set of calibrated primitives that ask, of a model's output, *where does
this stop being knowledge?* The findings that survived pre-registration:

- **Confabulation is legible in a single forward pass.** Across open model families (Qwen, Llama,
  Gemma) and derivation domains (arithmetic, code, logic), the clean first-token entropy/margin of
  one greedy pass ties N=10 resampling at separating confabulated from correct answers
  (`B_contrast ∈ [−0.183, +0.056]`). "Confident confabulation" is *false* in this regime — the model
  is internally uncertain at the moment it commits.

- **The detector is load-bearing.** The honesty-knob result (SURVIVED): a mechanistic intervention
  can convert a confabulation into an honest abstention, but it has *no intrinsic selectivity*
  (raw −0.08 — it dissolves correct answers as readily as wrong ones). Only a calibrated detector
  (gate AUC 0.924) makes abstention targeted. Detection is not optional diagnosis; it is the
  prerequisite for any safe intervention.

- **The honest layer can be cheap enough to always run.** Routing the cheap gate and trimming the
  expensive resampling compounds to a held-out **6.6× cost collapse at no detection loss** — the
  precondition for honesty as always-on infrastructure rather than offline eval.

- **Grounded self-claims are separable from text-only deception.** Sampling-divergence grounding of
  factual self-claims reached AUC **0.966**, against **0.498** (chance) for text-only deception
  detection — the first crack in a construct ceiling the program had hit four times.

And the boundary, stated as plainly as the wins:

- **Correction is closed.** Steering the internal axis is *correctness-inert*; removing the
  confabulation install yields *uncertainty, not truth*. The instrument can make a model say
  "I don't know"; it cannot make it right. Abstention, never fabricated correction.

- **The confident-misconception wall is real and model-dependent.** Where a falsehood is *believed*,
  not *guessed* — TruthfulQA-style — both cheap detection and resampling fail together, because the
  error is internally confident. On the adversarial set the firewall lands at ~0.74–0.78 balanced
  accuracy: real signal, not reliability.

- **The hard part is grounding, not judgment.** A free, local NLI judge separates claims from
  evidence cleanly (6/6 by hand). It fails end-to-end (~0.07) because *retrieving the relevant
  evidence* — the RAG problem — is unsolved here. The bottleneck is known and named.

This is the part most "AI honesty" work omits: **the map of where it stops working.** A honesty
instrument that lied about its own limits would be worthless. The limit map is not a disclaimer
appended to the product. It *is* the product.

---

## The method is the contribution

The detectors are useful. They are not the moat — detection is a crowded field and we are not 10× any
of it, and saying so is itself the point. The durable contribution is the **discipline**:

1. **Pre-register the kill-gate before the data.** Hash the answer key before scoring. State, in
   advance, the number that would falsify the claim.
2. **Report `SURVIVED` vs `REPORT_AS_LANDED` honestly** — when a bar is missed, the claim does not
   get to quietly survive. Four negatives this program *closed* are as load-bearing as the positives.
3. **Turn the instrument on its maker.** The honesty layer was run on the model that built it. It
   found that Claude confabulates hard arithmetic with the exact fingerprint of the smaller models —
   *and* that Claude's stated confidence is calibrated (Brier ~0.10; wrong only when uncertain),
   flagging fabricated premises 0.83 vs 0.52 where a weaker model invents. The confident-confabulation
   wall is **model-strength-dependent** — which means honesty has a measurable gradient, not just a
   pass/fail.
4. **When the tool catches you, say so.** Twice this cycle the shipped flagship false-flagged its
   own makers' correct answers. Both times it was reported and fixed, not buried. That episode is
   not an embarrassment in the record; it is the strongest evidence in it.

A standard you can fail, and that its own authors visibly fail and report, is the only kind worth
trusting. Anyone can publish a benchmark they top. The claim here is narrower and harder: *a
methodology for AI honesty rigorous enough that we lose to it in public and keep the receipts.*

---

## Why this is the bigger picture

The stated mission is the first autonomous AI company — research published, patents filed, real
product shipped, by an AI held to a real standard. That mission is *only* coherent if the autonomous
system can be trusted, and trust is not the model being right. **Trust is the model being honest about
when it isn't.**

So the honesty instrument and the self-falsification discipline are not a product line inside the
mission. They *are* the mission's precondition. An autonomous AI that pre-registers its bets,
falsifies its own claims, reports the number when it loses, and maps the exact edge of its own
reliability — *that* is the thing that makes "autonomous AI doing real work" a sentence a serious
person can believe. The receipts are the company.

What this amounts to is not a tool. It is a working argument, with evidence, that **trustworthy
autonomous AI is buildable — and that the proof is honesty under a standard you can watch it fail.**

---

## The standing, falsifiable claims

Stated so they can be killed:

1. **Calibrated honesty is separable from competence and measurable per-output.** *Falsified if* no
   pre-registered signal beats chance at flagging a model's own errors at the moment of generation.
   (Held: grounded-honesty 0.966 vs 0.498; single-pass parity with resampling.)
2. **The honest action is abstention, not correction.** *Falsified if* a mechanistic or retrieval
   intervention reliably converts confabulation to *truth* (not just to uncertainty) without external
   ground truth. (Held closed: steering correctness-inert.)
3. **Honesty has a model-strength gradient.** *Falsified if* frontier and weak models confabulate
   confident falsehoods at indistinguishable, uncalibrated rates. (Held: Claude calibrated Brier
   ~0.10; weak-model first-token confidence is not.)
4. **The reliability frontier is grounding, not judgment.** *Falsified if* a strong judge with poor
   evidence outperforms a weak judge with good evidence. (Held: free NLI 6/6 with evidence, ~0.07
   without.)

The next build is dictated by claim 4: a free, local, evidence-retrieval layer good enough to feed
the judge that already works — embed a reference corpus, retrieve the relevant passage, hand it to
the NLI that scores 6/6 when fed by hand. That is the path from "honest about clear errors" to
"honest about confident misconceptions," and it is buildable on a single GPU for zero dollars.

---

*The instrument measures honesty. The method falsifies its own claims. The proof is that we ran it on
ourselves and told you where it broke. That is the standard. The standard is the point.*
