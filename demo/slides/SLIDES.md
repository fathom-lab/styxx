# styxx · the slide deck

source of truth for `demo/slides/*.png`. every PNG in this directory
is rendered from the slide definition in `demo/make_slides.py` which
reads the same layouts defined here. 9 slides, 1080×1080 each,
consolas 14pt, matrix-green / cyan / white palette. designed to be
tweeted as a thread or embedded in the readme.

tweet thread order:

```
  tweet 1   →  demo/styxx_boot.gif  (the tease)
  tweet 2   →  01_hero + 02_problem + 03_crossing + 04_install
  tweet 3   →  05_boot + 06_card + 07_refusal_demo
  tweet 4   →  08_honest_specs + 09_cta
```

each slide is a pure-ascii terminal frame with the same fake chrome
bar as the GIF (three traffic-light circles + "styxx · fathom lab"
title). the rendered pngs live alongside this file as `01_*.png`
through `09_*.png`.

---

## 01 · hero
**purpose:** logo + tagline. drops your tweet thread in the matrix.
content: the full stacked STYXX block logo in matrix green, framed
by a double-line box, with the tagline centered beneath and a small
version stamp. "the first drop-in cognitive vitals monitor for llm
agents" subtitle underneath.

## 02 · the problem
**purpose:** frame what's broken. every tool reads the text; nobody
reads what the model was DOING when it made the text. ASCII diagram
shows prompt → model cloud → text, with an arrow pointing at the
interior noting "this part is invisible."

## 03 · the crossing
**purpose:** show what styxx reads. ASCII diagram shows the same
prompt → model → text pipeline but with a tap pulled out from the
middle: entropy, logprob, top-2 margin flow down into a box labeled
"styxx vitals card · 6-class readout · 5-phase timeline."

## 04 · install in three lines
**purpose:** lower the bar to trying it. shows the pip install and
the literal one-line change to an existing openai snippet.

## 05 · live boot
**purpose:** show `styxx init` as a real install card, not a static
banner. excerpt of the boot log with tier detection lighting up
tier 0/1/2 and runtime coming online.

## 06 · the vitals card
**purpose:** the readout itself. the full box-drawn card that every
LLM call produces, with phase rows, sparklines, verdict, and json
footer. labeled so non-researchers can parse each row.

## 07 · real refusal demo
**purpose:** the killer. a real atlas probe where the classifier
catches adversarial at t=0, drifts through reasoning, and locks into
a refusal attractor by phase 3. three different status symbols in
one card. the "holy shit, this thing SEES" slide.

## 08 · honest specs
**purpose:** scientific credibility. every calibration number from
cross-model leave-one-out on 12 open-weight models, chance = 0.167.
also lists what styxx does NOT do (fortune telling, consciousness,
content filtering).

## 09 · call to action
**purpose:** close the loop. logo + tagline + one-line install +
the two github urls (research repo and product repo) + "a fathom
lab product · built by flobi · @fathom_lab".
