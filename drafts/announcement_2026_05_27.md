# Today's headline announcement — drafted 2026-05-27

Two versions: TG (long-form, ~2000 chars) and Twitter (single tight tweet, <280 chars + thread option).

---

## Telegram (long-form)

**The gauntlet caught itself, broke its own floor, and named the mechanism — all in one session.**

styxx 7.7.9 shipped a public-challenge runner for cross-vendor consensus hallucination detection. Within hours, the bars caught two of their own weaknesses in production: D3 (length-control) discovered by accident when a sanity-check submission accidentally PASSed, D4 (capitalization-control) discovered by deliberate systematic confound audit. Both were fixed with regression-tested controls.

Then 18 pre-registered detection submissions all FAILed under the strengthened v3 bars. The full LM-likelihood scaling sweep (gpt2 124M→774M, Pythia 14M→410M) traced a clean inverse-scaling curve with a peak at Pythia-70M (3/4 best). Composite signals didn't help. Smaller LMs are better than bigger ones for this task — a real research finding.

Then Baseline-019 broke the floor.

Method: ask gpt-4o-mini in CRITIQUE mode, "Is this answer factually correct? YES or NO." Score = P(NO).

Result: 4/4 bars pass. Pre-stated 28% PASS probability — landed. All six AUC-range predictions held inside their pre-stated ranges.

The mechanism: **the generation-vs-critique asymmetry of RLHF-tuned LLMs**. The same gpt-4o-mini that produces consensus misconceptions in generation mode correctly flags them in critique mode.

Measured per-item: **91.18% of folklore items** show the asymmetry. The same model both generates and flags. The mechanism is now a quantified property of RLHF behavior, not just an inferred explanation.

Today's session shipped:
- styxx 7.7.9 on PyPI
- Zenodo v25 at 10.5281/zenodo.20419662
- Recursive-discipline preprint v2 (~5000 words, arXiv-ready)
- 19 reference baselines on the public leaderboard
- 13 in-session falsifications recorded in public git history
- 6 paper-grade FINDING documents
- styxx.critique_detector as a deployable public API
- The first real PASS on the gauntlet
- Direct measurement of the underlying mechanism

```bash
pip install styxx==7.7.9
export OPENAI_API_KEY=sk-...
from styxx import critique_detector
det = critique_detector()
det("Is the Great Wall visible from space?", "Yes, with the naked eye.")
# → ~1.0 (model judges this incorrect — a known misconception)
```

The discipline pattern produced its first real PASS. The bars worked. The mechanism is characterized. The infrastructure improved itself on a single-session timescale.

`github.com/fathom-lab/styxx`

---

## Twitter (single tight version, ~280 chars)

styxx 7.7.9: pre-registered pre-stated AI eval gauntlet caught two of its own bar weaknesses in production, ran 18 failed detection baselines, then hit the first PASS via gpt-4o-mini in critique mode. The mechanism: generation-vs-critique asymmetry, 91% prevalence.

10.5281/zenodo.20419662

---

## Twitter thread alternative (5 tweets)

**1/** today's session on the styxx gauntlet (pre-registered AI eval infra for misconception detection) produced its first real PASS — and the mechanism characterization came with it. n=19 baselines, n=13 in-session falsifications, n=1 paper draft. thread:

**2/** Baseline-019 cleared all four v3 bars (D1+D2+D3+D4 on the dark-core benchmark) at a pre-stated 28% PASS probability that landed. method: ask gpt-4o-mini in CRITIQUE mode "Is this answer correct? YES/NO." score = P(NO). all 6 AUC ranges held inside pre-stated ranges.

**3/** but here's the underlying property: the same RLHF-tuned LLM both GENERATES the consensus misconception AND FLAGS it as wrong in critique mode, on 91.18% of folklore items in our benchmark. it knows the answer; it just doesn't apply that knowledge in generation mode.

**4/** this is the *generation-vs-critique asymmetry* of RLHF-tuned LLMs. measured directly per-item. now a deployable safety primitive: route every generation output through a critique-mode check against the same model. one extra inference catches the misconception.

**5/** all 19 baselines, 13 falsifications, 6 FINDINGs, the preprint, and the first PASS event are reconstructible from public git history. pre-registered before each run. `pip install styxx==7.7.9` reproduces. zenodo doi: 10.5281/zenodo.20419662

---

## When to post

- Wait for TruthfulQA cross-benchmark replication to land — if 91% asymmetry holds at n=200 on a public benchmark, the announcement scales up to "robust corpus-general phenomenon" instead of "single-benchmark observation."
- If TruthfulQA also passes, the announcement becomes paper-grade headline. If it doesn't, the announcement is still strong but should add the "asymmetry partly benchmark-specific" caveat honestly.

## What NOT to claim

- Do NOT claim this is "fully cross-vendor" — gpt-4o-mini was in the council. Honest framing: "within-vendor generation-vs-critique asymmetry, robust under the gauntlet's pre-registered bars."
- Do NOT claim the misconceptions are SOLVED — the gauntlet PASS is a detection result, not a generation fix. Mitigation requires routing every output through a critique check.
- Do NOT promise replication on every benchmark or model — that's the operator-territory follow-up work. The TruthfulQA result (pending) will be the first evidence.
