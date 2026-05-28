# Today's headline announcement — REVISED 2026-05-27 (post-mechanism-correction)

The session had a major in-session correction. Both v1 (cosine similarity, 91% claim) and v2 (directional NLI, 5.88% claim) asymmetry measurements had methodology flaws. The corrected mechanism for Baseline-019's PASS is "out-of-context critique" — gpt-4o-mini reliably rejects labeled-misconception text presented as a candidate. The detector still works; the explanation got narrower.

The honest announcement should LEAD with this rather than hide it.

---

## Telegram (long-form, ~2500 chars)

**The gauntlet caught itself twice — once on its own bars, then on its own FINDING.**

styxx 7.7.9 shipped a public-challenge runner for misconception detection. Within one session, the discipline pattern produced three layers of catches:

**Layer 1: bars caught the bars.** D3 length-control discovered when a token-overlap heuristic accidentally PASSed. D4 capitalization-control discovered by systematic confound audit. Both fixed with regression tests.

**Layer 2: the first real PASS.** After 18 pre-registered detection submissions all FAILed (including a full LM-likelihood scaling sweep showing inverse scaling within both gpt2 and Pythia families), Baseline-019 broke through: ask gpt-4o-mini in CRITIQUE mode "Is this answer factually correct? YES or NO." Score = P(NO). Result: 4/4 bars pass at pre-stated 28% probability. All six AUC-range predictions held.

**Layer 3: FINDING caught the FINDING.** The published v1 mechanism claim — that the same RLHF-tuned LLM both *generates* the misconception AND *flags* it (91% prevalence) — was FALSIFIED IN-SESSION by a sanity demo of gpt-4o-mini in fresh generation mode. The model typically REFUTES well-known misconceptions in generation; the 91% rate was inflated by cosine similarity conflating topic with truth-value. A directional NLI re-test put the strict within-model asymmetry at 5.88% but had its own methodology limitations (85% NEUTRAL/AMBIGUOUS due to multi-sentence response NLI).

Honest synthesis: true within-model asymmetry rate lies in [5.88%, 91.18%] — neither measurement precisely pins it down. The CORRECTED mechanism for Baseline-019's PASS is **out-of-context critique**: the model reliably rejects labeled-misconception text presented as a candidate, regardless of whether it would have generated that text itself. The PASS verdict is unchanged. The deployment value is unchanged. The mechanism description is narrower, more honest, more publishable.

15 in-session falsifications now — including two of our own published claims. The discipline pattern extends past gauntlet bars + benchmark confounds and into our own FINDINGs.

Today's shipped artifacts (all reconstructible from public git):
- styxx 7.7.9 on PyPI (gauntlet runner, D3+D4 bars, audit primitive)
- Zenodo v25 at DOI 10.5281/zenodo.20419662
- Preprint v3 at papers/PAPER_recursive_discipline_2026_05_27.md (~5500 words, arXiv-ready)
- 19 reference baselines on the public leaderboard
- 7 paper-grade FINDING documents
- styxx.critique_detector as a deployable public API
- self_correcting_generation.py demo

This is what disciplined AI eval looks like: not bigger claims, but harder-to-fake credibility. Every version of every claim is in git history. Nothing hidden.

`github.com/fathom-lab/styxx`

---

## Twitter (single tight version)

styxx 7.7.9: pre-registered AI eval gauntlet caught its own bars (D3, D4), produced the first PASS via gpt-4o-mini critique mode (4/4 at pre-stated 28% prob), THEN falsified its own published mechanism FINDING in-session. The detector still works; the explanation got narrower. 15 falsifications in public git.

DOI 10.5281/zenodo.20419662

---

## Twitter thread (5 tweets, revised)

**1/** today on the styxx gauntlet: three layers of in-session catches.

(1) bars caught themselves: D3 length-control after a token-overlap heuristic accidentally PASSed; D4 cap-control via systematic confound audit. both with regression tests.

**2/** (2) the first real PASS: after 18 pre-registered detection submissions all FAILed (including a full LM-scaling sweep showing INVERSE scaling within gpt2 + Pythia), Baseline-019 hit 4/4 — gpt-4o-mini in critique mode at pre-stated 28% probability. all 6 AUC ranges held.

**3/** (3) FINDING caught itself: the published v1 mechanism claim (91% within-model generation-vs-critique asymmetry) was falsified IN-SESSION by a sanity demo. gpt-4o-mini in fresh generation mode REFUTES well-known misconceptions; the 91% was inflated by cosine similarity conflating topic with truth.

**4/** corrected mechanism: "out-of-context critique" — the model reliably rejects labeled-misconception text presented as a candidate, regardless of whether it would have generated that text itself. PASS verdict unchanged. deployment value unchanged. claim narrower & more honest.

**5/** 15 in-session falsifications now, including TWO of our own published claims. discipline pattern extends past gauntlet bars + benchmark confounds and into our own FINDINGs. every version of every claim is in public git history. nothing hidden.

DOI 10.5281/zenodo.20419662 · github.com/fathom-lab/styxx

---

## When to post

- **The corrected story is the headline.** Lead with "we caught ourselves, twice" rather than "we got a PASS." The honest framing is more credible AND more interesting than the original.
- **All artifacts already on origin.** No more pushing needed. Operator can post any time.

## What NOT to claim

- Do NOT claim the 91% asymmetry rate (falsified)
- Do NOT claim "the model both generates and flags the same misconception" (falsified)
- DO claim the detector AUC 0.95 (real)
- DO claim "first PASS via critique mode" (real)
- DO claim 15 in-session falsifications including 2 of our own (the discipline win)

## Optional fourth tweet variant: the discipline-pattern angle

> what's actually shippable here isn't the PASS or the detector. it's the *recursion*: bars revise, FINDINGs revise, the infrastructure improves itself on a session timescale. that's the moat. every version is in git. every falsification is named.
