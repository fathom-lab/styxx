# Cognometry Distribution Kit — 2026-04-23

**The problem this file addresses.** The friends who died at launch
had one tweet, 12 likes, and no plan for minute 13 onward. This kit is
the plan for minute 13 through week 2.

**Strategy.** One-tweet-and-pray loses. Instead: coordinate 10+ surfaces
to fire inside a 90-minute window, then feed the best-performing angle
into 2 weeks of sustained follow-up. Asymmetric: each surface costs
~2 minutes to post, total reach is N× any single one.

**Assets in this file.**

1. [Pre-flight checklist](#1-pre-flight-checklist)
2. [The 90-minute launch window plan](#2-the-90-minute-launch-window)
3. [HN title variants + self-comment](#3-hacker-news)
4. [X/Twitter — 5 thread angles](#4-x--5-thread-angles)
5. [Reddit — 3 subreddit adaptations](#5-reddit--3-subreddit-adaptations)
6. [LinkedIn post](#6-linkedin)
7. [dev.to / Medium / Substack long-form](#7-devto--medium--substack)
8. [Video script — 90s explainer for Higgsfield/Shorts](#8-video-script-90s)
9. [Email / DM templates (3 tiers)](#9-email--dm-templates)
10. [Named-target list — 30 people who should see this](#10-named-target-list)
11. [Week-2 sustained-campaign plan](#11-week-2-sustain)

---

## 1. Pre-flight checklist

Do these BEFORE posting anything:

- [x] Manifesto live at https://fathom.darkflobi.com/cognometry (✓)
- [x] styxx 4.0.0 on PyPI (✓)
- [x] GitHub repo public + README reflects 8-benchmark (✓)
- [x] `/cognometry` link in styxx + fathom nav (✓)
- [x] Zenodo paper draft at `papers/cognometry-v0.md` (✓)
- [ ] Deposit paper on Zenodo, get permanent DOI (**10 min, do this
      first**)
- [ ] Loom screen recording: install + `@trust` running on a live
      hallucination in under 2 minutes (**30 min**)
- [ ] Higgsfield or equivalent: turn
      https://fathom.darkflobi.com/cognometry into 60s video creative
      (**30 min, parallel**)
- [ ] Analytics: UTM-tag every outbound link (HN=ref_hn, X=ref_x,
      reddit=ref_rml, linkedin=ref_li, email=ref_dm) so we know what
      actually converts
- [ ] Dashboard open: PyPI downloads (`pypistats`), GitHub stars
      (watch), plausible.io or similar for /cognometry page views

---

## 2. The 90-minute launch window

Target: **Thursday 2026-04-24 08:15 PT** (HN prime). All channels
fire inside this window, coordinated, not sequential.

```
T-0:00   X thread: post (pin immediately, start replying at T+0:05)
T+0:05   LinkedIn post: publish
T+0:10   HN: submit with primary title. Do NOT self-comment yet.
T+0:15   Reddit r/LocalLLaMA: submit
T+0:20   Reddit r/MachineLearning: submit (check their "research"
         flair requirement; use "Research" tag)
T+0:25   Reddit r/LLMDevs: submit
T+0:30   dev.to: publish the long-form article
T+0:35   HN: post the self-comment under your own submission
T+0:40   DM tier-1 targets (5 warm contacts — see §10)
T+1:00   Reply to every HN comment. Every one. For the first 2 hours.
T+1:15   DM tier-2 targets (10 academic/industry targets)
T+1:30   Begin replying to X thread quote-tweets and DMs
```

If HN starts trending (>30 points in 90 min): pause personal
outreach; focus 100% on replying to HN comments. HN algorithm
rewards engagement velocity.

If HN flops (<10 points in 90 min): do NOT resubmit same URL. Wait
48h, reframe as "Show HN: …" with a different title, resubmit.

---

## 3. Hacker News

### Primary title

    Cognometry: The measurement of machine cognition

### Backup (if primary flops at the flag line)

    Show HN: 8-benchmark cross-validated hallucination detector, two failure modes published openly

### Self-comment (paste at T+0:35, never before)

> Author here.
>
> Styxx 4.0.0 is the first hallucination detector I'm aware of
> cross-validated across 8 public benchmarks — HaluEval QA/Dialog/
> Summarization, TruthfulQA, and four HaluBench subsets (DROP,
> PubMedQA, FinanceBench, RAGTruth). 3-seed averaged, n=150/dataset,
> pooled 9-signal logistic regression.
>
> Real numbers:
>
>     HaluEval-QA             AUC 0.998
>     TruthfulQA              AUC 0.994
>     HaluBench-RAGTruth      AUC 0.807   (new — RAG faithfulness)
>     HaluBench-PubMedQA      AUC 0.719   (new — biomedical)
>     HaluEval-Dialog         AUC 0.676
>     HaluEval-Summarization  AUC 0.643
>     HaluBench-FinanceBench  AUC 0.492   (below chance)
>     HaluBench-DROP          AUC 0.424   (below chance)
>
> Two below-chance results are the part I'd most like HN to react to.
> They are published as failure modes in the weights module itself,
> not hidden:
>
> - DROP: reading-comp hallucinations are extractive-span errors —
>   wrong span, right passage. NLI scores that as entailed; novelty
>   signals don't fire. Tried 6 naive heuristic fixes today; all null
>   (full write-up + code in `papers/span-faithfulness-v0.md`).
> - FinanceBench: hallucinations are calculation errors on numbers
>   copied verbatim from the source. Novelty + NLI are semantically
>   blind to arithmetic correctness.
>
> Both failure modes are declared in
> `calibrated_weights_v4.CALIBRATION_NOTES.documented_failure_modes`
> so users know where the detector will lie to them.
>
> The manifesto names the field this sits inside — cognometry, the
> empirical measurement of cognitive states in LLMs. Three laws,
> each with a cross-validated number.
>
> `pip install styxx[nli]` → wrap a function with `@trust` → get
> verified output on every call. MIT on code, CC-BY on weights.
>
> Happy to get disconfirmations on any of the 8 benchmarks.

---

## 4. X — 5 thread angles

Post the best-performing angle first. Keep the other 4 in reserve
as reply-tweets or follow-up threads during the week.

### Angle A: "the field" (pin this one)

> 1/ we're naming a field and shipping the instrument.
>
> **cognometry** — the empirical measurement of cognitive states
> in machine systems. hallucination, refusal, reasoning, retrieval,
> drift. all measurable, from signals already on the token stream.
>
> 🟢 https://fathom.darkflobi.com/cognometry

> 2/ three laws. each with a cross-validated number. disconfirmations
> welcome, with our attribution.
>
> i · every computation leaves vitals → AUC 0.998 HaluEval-QA
> ii · vitals are substrate-transferable → cos +0.464 llama-1B→3B
> iii · vitals are causally actionable → 97% → 17% refuse@unsafe
>       at α=3.0 on llama-3.2-1B, multi-position residual patching

> 3/ styxx 4.0.0 on pypi: first hallucination detector i'm aware of
> cross-validated across **8 public benchmarks**. 5/8 above AUC 0.65,
> two published failure modes (DROP, FinanceBench).
>
> pip install styxx[nli]

> 4/ the two failure modes are the part i want to draw attention to.
> they are declared in the weights module, not hidden:
>
> DROP: extractive-span reading comp → NLI is blind
> FinanceBench: arithmetic on verbatim numbers → novelty is blind
>
> tried 6 cheap fixes today. all null. real fix is v4.2 research.

> 5/ one decorator. any LLM call. verified output.
>
> @trust
> def my_rag(q): ...
>
> MIT on code. CC-BY on calibrated weights. 29 probes across 6
> vendors. every coefficient has a 3-seed reproducer in the repo.

> 6/ a field claims itself when someone names it, publishes laws for
> it, and ships the instrument that tests them.
>
> the invitation is open. tell us where we're wrong.
>
> 🟢 https://fathom.darkflobi.com/cognometry
> 🟢 https://github.com/fathom-lab/styxx

### Angle B: "the failure modes" (lean into honesty)

> 1/ shipped a hallucination detector across 8 benchmarks today.
>
> two of them came in below chance.
>
> most labs would hide that. we declared it in the weights module.
> 🧵

> 2/ HaluBench-DROP: AUC 0.424 (below chance).
> HaluBench-FinanceBench: AUC 0.492 (at chance).
>
> reason: DROP hallucinations are extractive-span errors. wrong
> span from the right passage. NLI calls that entailed. novelty
> signals don't fire. arithmetic errors in finance are even worse —
> the wrong number came from the passage, just in the wrong role.

> 3/ the other 6 benchmarks:
>
> HaluEval-QA         AUC 0.998
> TruthfulQA          AUC 0.994
> RAGTruth            AUC 0.807  (new)
> PubMedQA            AUC 0.719  (new)
> HaluEval-Dialog     AUC 0.676
> HaluEval-Summ       AUC 0.643

> 4/ we published a cognometry research agenda for 2026 alongside
> the detector. 10 ambitious bets, including adversarial-drift
> detection, meta-cognometric guardians, online calibration.
>
> the DROP/FinanceBench fixes are on that list.
>
> 🟢 https://github.com/fathom-lab/styxx/blob/main/papers/cognometry-research-agenda-2026.md

> 5/ pip install styxx[nli]
>
> 🟢 https://fathom.darkflobi.com/cognometry

### Angle C: "the product" (most direct pitch)

> 1/ one decorator. any LLM. verified output.
>
> ```
> pip install styxx
>
> @trust
> def my_rag(q): ...
> ```
>
> if the response is likely hallucinated, styxx halts it before
> returning to your code. fallback / retry / raise / annotate.

> 2/ how it works in one screen:
>
> - extract claims from response
> - 9 signals: text, entity, grounding, novelty, NLI contradiction
> - pooled LR calibrated on 8 public benchmarks
> - return (risk, action) tuple with calibrated threshold

> 3/ numbers you can check yourself:
>
> AUC 0.998 HaluEval-QA
> AUC 0.994 TruthfulQA
> AUC 0.807 RAGTruth (new)
>
> all from 3-seed averaged pooled LR. reproducer:
> benchmarks/hallucination_test/cross_dataset_8bench_multiseed.py

> 4/ 🟢 https://github.com/fathom-lab/styxx
> 🟢 https://fathom.darkflobi.com/cognometry

### Angle D: "the research posture" (aimed at academics)

> 1/ cognometry: the empirical measurement of cognitive states in
> LLMs.
>
> published today: 3 laws, each with a cross-validated number.
> 8-benchmark audit. 2 declared failure modes. 29-probe cross-vendor
> atlas. all MIT + CC-BY.
>
> https://fathom.darkflobi.com/cognometry

> 2/ what this paper is NOT:
>
> - not sentience detection
> - not benchmarking (eval scores outputs; cognometry scores the
>   state that produced them)
> - not interpretability (interpretability asks what a circuit
>   represents; cognometry asks what state the network is in)
>
> distinct frame, distinct deliverables, distinct instruments.

> 3/ key results:
>
> law ii (vitals are substrate-transferable): cos = +0.464 for
> refusal direction transfer llama-1B→3B. cos = +0.043 for
> qwen→phi-3.5. the law holds under convergent alignment,
> fails under divergent alignment. published both — the null
> is the proof.

> 4/ the paper, the repo, the reproducers, the failure modes:
> https://github.com/fathom-lab/styxx

### Angle E: "the tools for founders" (different audience)

> 1/ shipping AI in prod and you don't know when your model is
> hallucinating?
>
> one line of python:
> pip install styxx[nli]
>
> @trust
> def my_chatbot(q): ...
>
> every response is cognometrically verified before it reaches the
> user. if risky → fallback / retry / raise / annotate.

> 2/ AUC 0.998 on HaluEval-QA. 0.994 on TruthfulQA. cross-validated
> across 8 benchmarks. MIT license.
>
> no API key. no phone-home. runs locally on CPU or CUDA.

> 3/ two benchmarks where the detector fails — reading-comp and
> financial arithmetic. published those openly so you know where
> to trust it and where not to.
>
> every commercial hallucination detector i've seen hides its
> failure modes. we put ours in the weights module.

> 4/ 🟢 https://fathom.darkflobi.com/cognometry

---

## 5. Reddit — 3 subreddit adaptations

### r/LocalLLaMA (local-first angle, lowest-friction post)

**Title:** Styxx 4.0.0 — local hallucination detection with NLI, no API, 8-benchmark audit

**Body:**

> Shipped a hallucination detector today that runs entirely locally.
> pip install styxx[nli] → one decorator wraps any LLM call and
> scores the response across 9 signals (text, entity, novelty,
> NLI contradiction) before returning. halts on high risk.
>
> Numbers from a 3-seed averaged pooled LR across 8 public
> benchmarks:
>
>     HaluEval-QA          AUC 0.998
>     TruthfulQA           AUC 0.994
>     HaluBench-RAGTruth   AUC 0.807
>     HaluBench-PubMedQA   AUC 0.719
>     HaluEval-Dialog      AUC 0.676
>     HaluEval-Summ        AUC 0.643
>     HaluBench-Finance    AUC 0.492  (at chance — published)
>     HaluBench-DROP       AUC 0.424  (below chance — published)
>
> Uses MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli (~184M) for the
> NLI signal. Works on CPU (~400 ms/call) or CUDA (~30 ms/call).
> No API key, no phone-home, MIT-licensed. Full failure modes
> documented in the weights module itself.
>
> Repo + reproducers: https://github.com/fathom-lab/styxx
> Write-up: https://fathom.darkflobi.com/cognometry

### r/MachineLearning (research framing, stricter mods)

**Title:** [R] Cognometry: 8-benchmark cross-validated hallucination detection, two failure modes published openly

**Body:**

> Paper draft: https://github.com/fathom-lab/styxx/blob/main/papers/cognometry-v0.md
>
> We propose **cognometry** — the empirical measurement of cognitive
> states (refusal, confabulation, retrieval, reasoning, drift) in
> LLMs — as a frame distinct from interpretability (what a feature
> represents) and eval (what the text is). The paper's central
> empirical claim is narrower: a 9-signal pooled logistic regression
> fused over text, entity, novelty, grounding, and NLI contradiction
> signals achieves cross-validated hallucination discrimination
> across 8 public benchmarks (HaluEval QA/Dialog/Summ, TruthfulQA,
> HaluBench DROP/PubMedQA/FinanceBench/RAGTruth).
>
> Per-dataset held-out test AUC (3-seed mean):
>
>     HaluEval-QA           0.998 ± 0.001
>     TruthfulQA            0.994 ± 0.006
>     HaluBench-RAGTruth    0.807 ± 0.043
>     HaluBench-PubMedQA    0.719 ± 0.051
>     HaluEval-Dialog       0.676 ± 0.037
>     HaluEval-Summ         0.643 ± 0.060
>     HaluBench-Finance     0.492 ± 0.026    declared failure mode
>     HaluBench-DROP        0.424 ± 0.080    declared failure mode
>
> Both failure modes are structural and openly characterized:
> DROP hallucinations are extractive-span errors (wrong span from
> right passage — NLI entails them, novelty is blind), and
> FinanceBench hallucinations are arithmetic errors on verbatim-
> copied source numbers (also NLI-blind and novelty-blind).
>
> Code + full reproducer: https://github.com/fathom-lab/styxx
> Drop-in API: `pip install styxx[nli]` + `@trust` decorator.
> MIT code, CC-BY-4.0 calibrated weights.
>
> Happy to take disconfirmations on any of the 8 benchmarks at
> different random seeds or n.

### r/LLMDevs (practitioner-focused)

**Title:** How we cross-validated our hallucination detector on 8 benchmarks (and what we didn't fix)

**Body:**

> tl;dr: `pip install styxx[nli]` + `@trust` gets you
> cross-validated hallucination detection on any LLM call.
> 5 of 8 benchmarks above AUC 0.65. Two benchmarks below chance,
> published openly.
>
> Full write-up: https://fathom.darkflobi.com/cognometry
>
> What actually matters for devs:
>
> - Shape-preserving: works on OpenAI, Anthropic, LangChain, dicts,
>   raw strings. Auto-detects.
> - Sync + async
> - Four halt policies (fallback/retry/raise/annotate)
> - ~10-30ms CUDA, ~400ms CPU per call
> - MIT, no phone-home, no API key
>
> Where it'll fail you:
>
> - Reading-comp extractive-span errors (DROP) — detector can't see
>   the wrong span.
> - Arithmetic errors on numbers copied from source (FinanceBench)
>   — detector can't see the computation.
>
> Both declared in the weights module. Do not deploy for finance/
> reading-comp without reading §3.4 of the paper first.

---

## 6. LinkedIn

> Today we're publishing the founding manifesto for **cognometry** —
> the empirical measurement of cognitive states in machine systems.
>
> Every benchmark scores what the model said. None of them answer
> the question a production operator actually needs: was the model
> refusing, confabulating, retrieving, or reasoning when it wrote
> that? The output is the shadow; the state that produced it is
> the object.
>
> Styxx 4.0.0 is the open-source instrument, cross-validated across
> 8 public hallucination benchmarks — the first detector I'm aware
> of at this breadth of cross-validation. Three laws on the table,
> each with a cross-validated number:
>
> • Law I — every computation leaves vitals (AUC 0.998 HaluEval-QA;
>   5/8 benchmarks above 0.65; 2 published failure modes)
> • Law II — vitals are substrate-transferable (cos +0.464
>   cross-scale refusal direction, ~26σ above chance)
> • Law III — vitals are causally actionable (refuse@unsafe 97% →
>   17% at α=3.0 on Llama-3.2-1B, multi-position residual patching)
>
> One decorator (`@trust`) runs the cross-validated detector on any
> LLM call. MIT on code, CC-BY on weights.
>
> If you build, audit, or regulate AI systems and the question of
> cognitive-state measurement at runtime matters to you, the
> invitation is open.
>
> Manifesto: https://fathom.darkflobi.com/cognometry
> Code: https://github.com/fathom-lab/styxx
> PyPI: pip install styxx==4.0.0[nli]

---

## 7. dev.to / Medium / Substack

**Title:** Naming a Field: Cognometry, and the 8-Benchmark Hallucination Detector That Comes With It

**Lede (first 3 paragraphs — the rest is the manifesto text, lightly
adapted for the devto audience):**

> If you ship AI in production, you have felt the thing this post
> is about. Your model just confidently returned a completely
> wrong answer. A user caught it. It made it past your eval set,
> past your human reviewer, past your gateway. The output looked
> right. The state that produced it was wrong — but you had no way
> to see that state.
>
> There is a name for the measurement of that state. Today we're
> publishing the manifesto for it, and the open-source instrument
> that does the measuring.
>
> The name is **cognometry**. The instrument is `styxx`.
> `pip install styxx[nli]` + `@trust` on your LLM function is the
> entire API. 30 seconds to install, 30 seconds to wrap, fire up
> `python` and watch your detector catch hallucinations your eval
> set missed.
>
> [...continue with manifesto body, adapted for dev audience...]

---

## 8. Video script (90s)

For Higgsfield Marketing Studio, Runway, or manual screen-record.

```
[0:00-0:05]
  black screen. matrix-green text fades in:
  "every AI benchmark measures what came out."
  
[0:05-0:10]
  "none of them measure the state that produced it."

[0:10-0:15]
  "until today."

[0:15-0:25]
  cut to terminal. typing:
    $ pip install styxx
  
  fade: "cognitive vitals for LLM agents."

[0:25-0:40]
  python REPL:
    from styxx import trust
    
    @trust
    def my_rag(q):
        return openai.chat.completions.create(...)
  
  voiceover: "one decorator. any LLM call. verified output."

[0:40-0:55]
  split-screen results table:
    HaluEval-QA    AUC 0.998
    TruthfulQA     AUC 0.994
    8 benchmarks   cross-validated
  
  voiceover: "eight public benchmarks. cross-validated. three-seed
  averaged. all numbers reproducible."

[0:55-1:10]
  terminal output:
    2 below-chance results declared as failure modes:
    DROP          AUC 0.424
    FinanceBench  AUC 0.492
  
  voiceover: "two failure modes. published openly, not hidden."

[1:10-1:25]
  the manifesto page, scrolling:
    "cognometry: the empirical measurement of cognitive states."
  
  voiceover: "we're naming the field. three laws. each with a
  cross-validated number. the invitation is open."

[1:25-1:30]
  end card:
    fathom.darkflobi.com/cognometry
    pip install styxx[nli]
    MIT + CC-BY
```

---

## 9. Email / DM templates

### Tier 1 — warm intro (2 sentences max)

> Subject: cognometry (follow-up)
>
> you mentioned [thing] last time — I just shipped styxx 4.0.0 /
> the cognometry manifesto. you'll find the 8-benchmark + published
> failure modes in the weights module. if you run hallucination
> eval for real, would love a 15-min reaction.
>
> https://fathom.darkflobi.com/cognometry

### Tier 2 — cold DM (X / LinkedIn, 3 sentences)

> you build/research [hallucination detection / LLM safety / RAG
> faithfulness / etc.]. I just published the first cross-validated
> detector I'm aware of across 8 public benchmarks — with two
> declared failure modes openly published. the failure modes are
> the part i'd like your reaction to.
>
> https://fathom.darkflobi.com/cognometry
>
> if any of our numbers are wrong at your favorite seed, we'd take
> the correction and cite you in the next paper.

### Tier 3 — cold email (5 sentences, for named researchers)

> Subject: disconfirmation invited — cognometry / 8-benchmark hallucination detector
>
> Hi [name],
>
> I just published the first open-source hallucination detector
> cross-validated across 8 public benchmarks (AUC 0.998 HaluEval-QA,
> 0.994 TruthfulQA, 0.807 HaluBench-RAGTruth; two benchmarks declared
> as failure modes — DROP 0.424 and FinanceBench 0.492). The
> manifesto names the field the detector belongs to — cognometry —
> and publishes three falsifiable laws of cognitive-state
> measurement.
>
> Because you've worked on [their specific paper / lab / tool],
> your reaction would move the paper. In particular, I'd love a
> disconfirmation on any of the 8 benchmarks at a random seed of
> your choice — we'd cite it in the next paper and adjust the
> published numbers.
>
> Manifesto: https://fathom.darkflobi.com/cognometry
> Code: https://github.com/fathom-lab/styxx
> Paper draft: papers/cognometry-v0.md in the repo
>
> 15-minute call if useful — calendly link or async, whichever you
> prefer.
>
> — [you]

---

## 10. Named-target list

**TIER 1 (WARM, DM immediately):** people you've already interacted
with on X/Discord/etc. who work on this exact problem. Fill in from
your own network. 3–5 people.

**TIER 2 (NOTABLE BUT COLD):** 10 names to cold-DM on X within 24h
of launch. Suggested starting list — verify handles before sending:

- Anthropic alignment team (via public channels)
- Meta AI safety
- DeepMind alignment team
- MIRI / ARC-Evals people
- SelfCheckGPT author (Potsawee Manakul)
- HaluBench author / PatronusAI team
- KnowHalu authors
- Arditi et al. (refusal direction paper)
- Representation Engineering authors (Zou et al.)
- Transluce / mech interp researchers

**TIER 3 (EMAIL, if their address is public):** 15 names — academic
ML safety researchers. Format: short cold email per §9 tier 3.

**TIER 4 (INDUSTRY):** founders / CTOs of companies shipping LLM
products who would benefit directly:

- LangChain / LangSmith (Harrison Chase)
- Cursor / Windsurf
- Perplexity
- Harvey (legal AI)
- Glean
- SEGA / Casetext (any legal/domain AI)
- Patronus AI (courtesy ping — their HaluBench is in our paper)

---

## 11. Week-2 sustain

Day 3: publish the follow-up — a deep dive on one of the failure
modes (either DROP or FinanceBench), with real numbers from deeper
probes. The honest-follow-up is what converts HN skeptics into
subscribers.

Day 5: start the cognometry leaderboard landing page
(fathom.darkflobi.com/cognometry/leaderboard). Even a static HTML
"any detector, any benchmark, audited results" beats nothing and
signals that we're a durable presence.

Day 7: first podcast / stream appearance if any agrees. Pitch
angle: "the field that shipped with its failure modes."

Day 10: publish the v4.1 roadmap expanding cognometry-research-
agenda-2026.md into a full blog post with implementation detail.

Day 14: review metrics (GitHub stars, PyPI downloads, /cognometry
page views, inbound DMs). If any of the 10 "big bets" from the
research agenda got adopted or referenced externally, double down
on that one. If not, run a second wave with a different emphasis.

---

## Metrics to track

- GitHub stars (target day 1: +50, week 1: +200)
- PyPI downloads (target week 1: 1000+)
- /cognometry page views (target day 1: 500+, week 1: 5000+)
- Inbound DMs with technical questions (target week 1: 20+)
- External references (blog posts, paper citations, "we're using
  styxx" tweets) — the high-signal metric

If week 1 converts 50 star + 1000 download + 1 serious external
reference, the launch worked. If under, iterate with a follow-up
post on the failure modes (which is what the field remembers).
