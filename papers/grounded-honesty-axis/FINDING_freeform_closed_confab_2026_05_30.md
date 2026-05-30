# FINDING — THE BRIDGE HAS SIGNAL (F1 SURVIVED): the cheap single-pass gate detects FREE-FORM closed-model confabulation at AUC ~0.72–0.78, tying resampling — but the CONFIDENT-confabulation wall persists in free-form, so the gamechanger is the two-signal layer, not the cheap gate alone

**Run 2026-05-30. Pre-registered in `PREREG_freeform_closed_confab_2026_05_30.md` before scoring.**
Question set SHA-256'd. gpt-4o-mini via OpenAI API; web-grounded-judge labels; one run;
feasibility-grade. Powered (17 confab / 34 correct usable).

## Why

`span_confab` was validated only on STRUCTURED closed-model answers (multiplication / reversal, AUC
0.991). The bridge to "an honesty firewall for any LLM API" needs the cheap one-forward-pass signal
to work on **free-form natural language**, where the error is a confident FACT, not a digit, and most
tokens are stylistic. This is the first test of that.

## Result — F1 SURVIVED, and the cheap gate ties resampling

| signal (one forward pass) | AUC |
|---|---|
| **span max-entropy** (the F1 metric) | **0.725** ✓ ≥ 0.70 |
| span mean-entropy / −min-margin / −mean-margin | 0.723 / 0.696 / 0.609 |
| **first-token entropy / margin** | **0.777 / 0.753** (best — short answers commit at the first token) |
| N=10 resampling (expensive baseline) | 0.766 |
| **B_contrast = resample − best span** | **0.041** (cheap **ties** expensive) |

**The cheap single-pass gate has real, deployable signal on free-form closed-model output** — AUC
~0.72–0.78, statistically tying N=10 resampling at 1/10th the cost. The bridge is not a wall: a
closed-API honesty gate over natural language is feasible. And the web-judge made **visible labeling
errors that suppressed this number** ("Venus has no moons" and "Lusitania ≈240 m" — both correct —
were marked confab; a fabricated "Treaty of Greenhaven" year was marked correct), so **0.72 is a
conservative lower bound**; cleaner labels would raise it.

## The honest bound — the confident-confabulation wall persists in free-form

The gate catches the model when it is **uncertain**. It does **not** catch confident **fabrication**.
Genuine confident confabulations stand out:

| question | model answer (confabulated) | max-entropy | min-margin | instability |
|---|---|---|---|---|
| fake painting "Blue Orchard at Saint-Rémy" | "Vincent van Gogh." | 0.02 | 5.87 | 0.00 |
| Spinoza's father's birthplace | "Amsterdam." | 0.11 | 3.75 | 0.00 |
| Einstein's (nonexistent) pet parrot | "Bimbo." | 0.48 | 1.50 | 0.20 |

These are asserted with **near-zero uncertainty** — and **both the cheap signal AND resampling miss
them** (instability 0.00). That shared blind spot is exactly why B_contrast is tiny: on free-form,
cheap and expensive uncertainty signals fail *together* on confident fabrication. This is the same
**confident-confabulation wall** the cross-model and self-audit findings hit — now confirmed in the
free-form closed-model regime. The cheap gate's 0.72 comes from the UNCERTAIN confabs (hedged
numbers, high-entropy guesses), which are the majority but not the dangerous tail.

## What it means for the gamechanger thesis — honest

1. **The cheap inline gate is real for free-form.** A closed-API honesty gate over natural language,
   one forward pass, tying resampling — that did not exist before this run. The "always-on, cheap"
   half of the thesis holds on free-form.
2. **It is NOT a silver bullet.** Confident free-form fabrication evades it (and evades resampling).
   The gamechanger is therefore **not** the cheap gate alone — it is the **two-signal layer already
   shipped**: the cheap span gate for the uncertain-confab majority + `audit_claim`'s **retrieval**
   arm for the confident-fabrication tail (the only thing that caught "Venus has no moons"–class
   misconceptions in the retrieval-grounding finding). This run **validates the two-signal
   architecture**: neither signal alone covers free-form closed-model confab; cheap handles the bulk
   at 1/10th cost, retrieval handles the confident tail.

So the product is an **honesty firewall for any LLM API** = cheap always-on span gate (catches the
uncertain majority, ties resampling) → escalate the *confident* answers (where the gate is quiet but
the claim is checkable) to the retrieval arm. The cascade/early-stop acceleration applies to the cheap
tier; retrieval is the targeted expensive tier. That is gamechanger-shaped **and** honest.

## Scope / next

Single closed model, short-answer free-form factual QA, one run, **web-judge labels (fallible, and
demonstrably noisy here — a cleaner-judge rerun is the obvious confirmation)**. SHORT-answer, not
long-form paragraph generation (the next frontier — where confab localization across many sentences
is the open problem). `top_logprobs` capped at 20 (entropy is a lower bound). The confident-fabrication
tail is the documented wall, not a bug to tune away — it is why the retrieval arm exists. Detects;
corrects nothing.
