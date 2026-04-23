# LAUNCH-DAY PLAYBOOK — Cognometry + Styxx 4.0.2

**Single file. Everything paste-ready. Char-count validated. UTM-tagged.**

Target launch window: **Thursday 2026-04-23 08:15 PT** (HN prime).

Before you fire, do the three pre-flight items in §0. Then execute
§1 in order at the minute-marker times. Everything after §2 is raw
paste material for later in the week.

---

## §0 — Pre-flight checklist (T-60 min, ~1 hour total)

- [x] **Zenodo deposit** — **DONE**. DOI:
      [10.5281/zenodo.19703527](https://doi.org/10.5281/zenodo.19703527).
      Record: https://zenodo.org/record/19703527. Already backfilled
      into the manifesto, launch copy, and this file.
- [ ] **Analytics tabs open in separate browser windows**:
  - https://pypistats.org/packages/styxx
  - https://github.com/fathom-lab/styxx (watch stars count)
  - https://news.ycombinator.com/newest (find your submission)
  - X/Twitter notifications tab
- [ ] **Timer ready** — you need to follow the T+0 / T+0:05 schedule
      in §1.
- [ ] **Done? Fire §1.**

---

## §1 — The 90-minute window (execute top-to-bottom, minute by minute)

Each step below has the EXACT text to paste. No thinking required
inside the window — just copy, paste, send.

### T+0:00 — X thread (post first, everything else waits)

Post the 6 tweets as a thread. Wait ~5 seconds between each so they
post in order. After tweet 6, **pin the thread**.

**Tweet 1:**
```
1/ we're naming a field and shipping the instrument.

cognometry — empirical measurement of cognitive states in LLMs. hallucination, refusal, reasoning, drift. all measurable, from signals already on the token stream.

https://fathom.darkflobi.com/cognometry?ref=x
```

**Tweet 2:**
```
2/ three laws. each with a cross-validated number.

i · vitals exist → AUC 0.998 HaluEval-QA
ii · vitals transfer → cos +0.464 llama-1B to 3B refusal direction
iii · vitals are causal → refuse@unsafe 97% → 17% at α=3.0 on llama-3.2-1B
```

**Tweet 3:**
```
3/ styxx 4.0.2 on pypi: first hallucination detector i'm aware of cross-validated on 8 public benchmarks. 5/8 above AUC 0.65, two published failure modes.

pip install styxx[nli]
```

**Tweet 4:**
```
4/ the two failure modes are the interesting part. declared in the weights, not hidden:

DROP: extractive-span reading comp → NLI blind
FinanceBench: arithmetic on verbatim numbers → novelty blind

6 cheap fixes for DROP tried. all null. fix needs span-level faithfulness.
```

**Tweet 5:**
```
5/ one decorator. any LLM call. verified output.

@trust
def my_rag(q, *, context): ...

zero config. MIT code. CC-BY weights. every number has a 3-seed reproducer in the repo.
```

**Tweet 6:**
```
6/ a field claims itself when someone names it, publishes laws for it, and ships the instrument that tests them.

the invitation is open. tell us where we're wrong.

github.com/fathom-lab/styxx
```

→ **PIN THE THREAD.** Optional: quote-tweet after an hour with:

```
peer-archived paper for the curious:
https://doi.org/10.5281/zenodo.19703527

Zenodo DOI, CC-BY-4.0. every number in the thread has a reproducer in the repo.
```

---

### T+0:05 — LinkedIn

Paste into https://www.linkedin.com/feed/ and publish:

```
Today we're publishing the founding manifesto for cognometry — the empirical measurement of cognitive states in machine systems.

Every benchmark scores what the model said. None answer the question a production operator actually needs: was the model refusing, confabulating, retrieving, or reasoning when it wrote that?

Styxx 4.0.2 is the open-source instrument, cross-validated across 8 public hallucination benchmarks — the first detector I'm aware of at this breadth of cross-validation. Three laws, each with a cross-validated number:

• Law I — every computation leaves vitals (AUC 0.998 HaluEval-QA; 5/8 benchmarks above 0.65; 2 published failure modes)
• Law II — vitals are substrate-transferable (cos +0.464 cross-scale refusal direction, ~26σ above chance)
• Law III — vitals are causally actionable (refuse@unsafe 97% → 17% at α=3.0 on Llama-3.2-1B)

One decorator (@trust) runs the cross-validated detector on any LLM call. Zero config. MIT on code, CC-BY on weights.

If you build, audit, or regulate AI systems and the question of cognitive-state measurement at runtime matters to you, the invitation is open.

Manifesto: https://fathom.darkflobi.com/cognometry?ref=li
Paper (DOI): https://doi.org/10.5281/zenodo.19703527
Code: https://github.com/fathom-lab/styxx
PyPI: pip install styxx==4.0.2[nli]
```

---

### T+0:10 — Hacker News submission

Go to https://news.ycombinator.com/submit

**Title** (paste exactly):
```
Cognometry: The measurement of machine cognition
```

**URL** (paste exactly):
```
https://fathom.darkflobi.com/cognometry?ref=hn
```

**Text:** leave blank. Submit.

**Do NOT self-comment yet.** Wait for T+0:35.

---

### T+0:15 — Reddit /r/LocalLLaMA

Go to https://www.reddit.com/r/LocalLLaMA/submit

**Title:**
```
Styxx 4.0.2 — local hallucination detection with NLI, no API, 8-benchmark audit
```

**Body:**
```
Shipped a hallucination detector today that runs entirely locally. `pip install styxx[nli]` → one decorator wraps any LLM call and scores the response across 9 signals (text, entity, novelty, NLI contradiction) before returning. Halts on high risk.

Numbers from a 3-seed averaged pooled LR across 8 public benchmarks:

```
HaluEval-QA          AUC 0.998
TruthfulQA           AUC 0.994
HaluBench-RAGTruth   AUC 0.807
HaluBench-PubMedQA   AUC 0.719
HaluEval-Dialog      AUC 0.676
HaluEval-Summ        AUC 0.643
HaluBench-Finance    AUC 0.492  (at chance — published)
HaluBench-DROP       AUC 0.424  (below chance — published)
```

Uses MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli (~184M) for the NLI signal. Works on CPU (~400 ms/call) or CUDA (~30 ms/call). No API key, no phone-home, MIT-licensed. Full failure modes documented in the weights module itself.

Repo + reproducers: https://github.com/fathom-lab/styxx
Write-up: https://fathom.darkflobi.com/cognometry?ref=r_llama
```

---

### T+0:20 — Reddit /r/MachineLearning

Go to https://www.reddit.com/r/MachineLearning/submit

*Check the sub's rules — they require a specific flair. Use "Research" flair.*

**Title:**
```
[R] Cognometry: 8-benchmark cross-validated hallucination detection, two failure modes published openly
```

**Body:**
```
Paper draft: https://github.com/fathom-lab/styxx/blob/main/papers/cognometry-v0.md

We propose **cognometry** — the empirical measurement of cognitive states (refusal, confabulation, retrieval, reasoning, drift) in LLMs — as a frame distinct from interpretability (what a feature represents) and eval (what the text is). The paper's central empirical claim is narrower: a 9-signal pooled logistic regression fused over text, entity, novelty, grounding, and NLI contradiction signals achieves cross-validated hallucination discrimination across 8 public benchmarks (HaluEval QA/Dialog/Summ, TruthfulQA, HaluBench DROP/PubMedQA/FinanceBench/RAGTruth).

Per-dataset held-out test AUC (3-seed mean):

```
HaluEval-QA           0.998 ± 0.001
TruthfulQA            0.994 ± 0.006
HaluBench-RAGTruth    0.807 ± 0.043
HaluBench-PubMedQA    0.719 ± 0.051
HaluEval-Dialog       0.676 ± 0.037
HaluEval-Summ         0.643 ± 0.060
HaluBench-Finance     0.492 ± 0.026    declared failure mode
HaluBench-DROP        0.424 ± 0.080    declared failure mode
```

Both failure modes are structural and openly characterized: DROP hallucinations are extractive-span errors (wrong span from right passage — NLI entails them, novelty is blind), and FinanceBench hallucinations are arithmetic errors on verbatim-copied source numbers (also NLI-blind and novelty-blind).

Code + full reproducer: https://github.com/fathom-lab/styxx
Drop-in API: `pip install styxx[nli]` + `@trust` decorator.
MIT code, CC-BY-4.0 calibrated weights.

Happy to take disconfirmations on any of the 8 benchmarks at different random seeds or n.
```

---

### T+0:25 — Reddit /r/LLMDevs

Go to https://www.reddit.com/r/LLMDevs/submit

**Title:**
```
How we cross-validated our hallucination detector on 8 benchmarks (and what we didn't fix)
```

**Body:**
```
tl;dr: `pip install styxx[nli]` + `@trust` gets you cross-validated hallucination detection on any LLM call. 5 of 8 benchmarks above AUC 0.65. Two benchmarks below chance, published openly.

Full write-up: https://fathom.darkflobi.com/cognometry?ref=r_dev

What actually matters for devs:

- Shape-preserving: works on OpenAI, Anthropic, LangChain, dicts, raw strings. Auto-detects.
- Sync + async
- Four halt policies (fallback/retry/raise/annotate)
- ~10-30ms CUDA, ~400ms CPU per call
- MIT, no phone-home, no API key

Where it'll fail you:

- Reading-comp extractive-span errors (DROP) — detector can't see the wrong span.
- Arithmetic errors on numbers copied from source (FinanceBench) — detector can't see the computation.

Both declared in the weights module. Do not deploy for finance/reading-comp without reading §3.4 of the paper first.

Repo: https://github.com/fathom-lab/styxx
```

---

### T+0:35 — HN self-comment (NOT before)

Go back to your HN submission. Click on the title, scroll to the
comment box, paste **this exact text** and submit:

```
Author here.

Styxx 4.0.2 is the first hallucination detector I'm aware of cross-validated across 8 public benchmarks — HaluEval QA/Dialog/Summarization, TruthfulQA, and four HaluBench subsets (DROP, PubMedQA, FinanceBench, RAGTruth). 3-seed averaged, n=150/dataset, pooled 9-signal logistic regression.

Paper (Zenodo, peer-archived): https://doi.org/10.5281/zenodo.19703527
Code: https://github.com/fathom-lab/styxx
Leaderboard: https://fathom.darkflobi.com/cognometry/leaderboard
Colab demo (2 min): https://colab.research.google.com/github/fathom-lab/styxx/blob/main/examples/cognometry_colab.ipynb

Real numbers:

    HaluEval-QA             AUC 0.998
    TruthfulQA              AUC 0.994
    HaluBench-RAGTruth      AUC 0.807   (new — RAG faithfulness)
    HaluBench-PubMedQA      AUC 0.719   (new — biomedical)
    HaluEval-Dialog         AUC 0.676
    HaluEval-Summarization  AUC 0.643
    HaluBench-FinanceBench  AUC 0.492   (below chance)
    HaluBench-DROP          AUC 0.424   (below chance)

Two below-chance results are the part I'd most like HN to react to. They are published as failure modes in the weights module itself, not hidden:

- DROP: reading-comp hallucinations are extractive-span errors — wrong span, right passage. NLI scores that as entailed; novelty signals don't fire. Tried 6 naive heuristic fixes; all null. The null probe is committed alongside the successes.
- FinanceBench: hallucinations are calculation errors on numbers copied verbatim from the source. Novelty + NLI are semantically blind to arithmetic correctness.

Both failure modes are declared in calibrated_weights_v4.CALIBRATION_NOTES.documented_failure_modes so production callers know where the detector will lie.

pip install styxx[nli] → wrap a function with @trust → get verified output on every call. Zero config: auto-detects context/reference/passage kwargs, auto-enables NLI when installed, adaptive threshold. MIT on code, CC-BY on calibrated weights.

Happy to get disconfirmations on any of the 8 benchmarks at your favorite random seed.
```

---

### T+0:40 onward — engagement + outreach

From here the minute-by-minute plan is looser. Two rules:

1. **Reply to every HN comment within the first 2 hours.** The HN
   algorithm rewards engagement velocity. If you disappear, the
   thread dies.
2. **Do not DM yet.** Wait until T+1:00 at earliest. Let the organic
   traffic establish first.

At T+1:00 (if HN is alive, i.e. >15 points): start tier-1 DMs. See
§2 below. At T+1:30: start tier-2. At T+2:00: begin replying to X
quote-tweets.

**If HN is dead at T+1:00 (<5 points)** — no DMs. Pause. Reassess.
Don't resubmit the same URL. Schedule a second launch with the
backup title 48 hours out.

---

## §2 — Named-target outreach (DM scripts)

Pre-filled with known handles. Confirm each before sending.

### Tier 1 — warm intros (T+1:00, 3-5 sends)

Fill from your own network. DM template:

```
you mentioned [thing] last time — just shipped styxx 4.0.2 + the cognometry manifesto. 8-benchmark audit, two declared failure modes (DROP, FinanceBench). would love your 15-min reaction.

https://fathom.darkflobi.com/cognometry?ref=dm_warm
```

### Tier 2 — cold DMs on X (T+1:30, 10 sends)

Verify handles before sending. Replace `@handle` with actual.

**Template:**
```
you build [hallucination detection / RAG faithfulness / LLM safety]. i shipped the first cross-validated detector i'm aware of on 8 public benchmarks today — with two declared failure modes openly published. failure modes are the part i'd like your reaction to.

https://fathom.darkflobi.com/cognometry?ref=dm_cold

if any of our numbers are wrong at your favorite seed, we'd take the correction and cite you.
```

**Targets to verify:**

```
[ ] @potsawee             — SelfCheckGPT author
[ ] @patronusai           — HaluBench team
[ ] @harrison_chase       — LangChain
[ ] @hwchase17            — LangChain (alt)
[ ] @jerryjliu0           — LlamaIndex
[ ] @cHHillee             — ML research (PyTorch)
[ ] @neelnanda5           — mech interp (Anthropic)
[ ] @AnthropicAI          — corporate (longshot)
[ ] @Meta_AI              — Meta safety
[ ] @DeepMind             — DM safety team
[ ] @soarenergies         — Sorin (if relevant)
[ ] @TransluceAI          — AI safety startup
```

### Tier 3 — cold email to researchers (T+2:00, 5-10 sends)

Use §9 Tier 3 template from cognometry-distribution-kit.md. Subject:

```
disconfirmation invited — cognometry / 8-benchmark hallucination detector
```

**Email targets to research:**

Academic researchers who publish on hallucination detection or
refusal directions. Find their .edu email from their lab page. Send
from your real email address.

```
[ ] Arditi et al. (refusal direction)
[ ] Zou et al. (representation engineering)
[ ] Marks & Tegmark (truthfulness directions)
[ ] Burns et al. (CCS / truthfulness)
[ ] Authors of SelfCheckGPT / KnowHalu / HaluCheck baselines
[ ] MIRI / ARC-Evals group
```

---

## §3 — UTM-tag reference card

Every link in this file uses one of these UTM values:

| Surface    | URL suffix                               |
|------------|------------------------------------------|
| HN         | `?ref=hn`                                |
| X/Twitter  | `?ref=x`                                 |
| LinkedIn   | `?ref=li`                                |
| r/LocalLLaMA | `?ref=r_llama`                         |
| r/ML       | `?ref=r_ml`                              |
| r/LLMDevs  | `?ref=r_dev`                             |
| Warm DM    | `?ref=dm_warm`                           |
| Cold DM    | `?ref=dm_cold`                           |
| Email      | `?ref=email`                             |

This works on the Netlify-hosted manifesto out of the box — the
`?ref=...` query string is passed through as-is. If you want to
switch to proper analytics later, configure Netlify Analytics or
Plausible to bucket by `ref` param.

---

## §4 — Emergency rollback

If something is wrong on the live manifesto (typo, broken link, bad
number) during the launch:

1. Fix in `C:/Users/heyzo/Desktop/clawd-clean/darkflobi-site/cognometry.html`
2. Also in `C:/Users/heyzo/clawd/darkflobi-site/cognometry.html`
3. Redeploy:
   ```bash
   bash C:/Users/heyzo/clawd/scripts/deploy-fathom-site.sh --message "launch-day hotfix: <what>"
   ```
4. Deploy takes ~5 sec. Verify with:
   ```bash
   curl -s https://fathom.darkflobi.com/cognometry | head -40
   ```

If something is wrong on PyPI (broken package): you CANNOT unpublish
a version. You must ship a bugfix release (`4.0.3`). Do NOT attempt
`--delete-version`.

If something is wrong on GitHub (leaked secret, bad commit): force-push
is your enemy during launch. Push a fix commit on top. Squash later.

---

## §5 — What to watch after the window closes

T+3:00 (end of active engagement window):

- HN score + rank. Screenshot if top-30.
- PyPI downloads delta. Aim: 100+ in first 3 hours.
- GitHub stars delta. Aim: 20+ in first 3 hours.
- X thread impressions. Aim: 10K+ in first 3 hours.
- Inbound DMs + comments. If anyone serious engages, prioritize
  replying to them even over HN comments.

T+24:00:

- Run the distribution-kit week-2 sustain plan (§11 of
  [cognometry-distribution-kit.md](cognometry-distribution-kit.md)):
  day 3 failure-mode deep-dive post, day 5 leaderboard landing, etc.

---

## §6 — Known launch-day risks (don't get surprised)

1. **`@trust` without a reference passes through by default.** In
   4.0.2, text-only heuristic path has threshold 0.99 — effectively
   off — because without a reference the detector can't meaningfully
   verify. Users who want strict text-only gating pass `threshold=`
   explicitly. If anyone asks "why did it not catch X without a
   reference?", explain: verification requires grounding, otherwise
   we'd be gating on noise. The calibrated path (reference passed)
   keeps the tight 0.7.

2. **NLI model download is ~1GB** on first call. Users pip-installing
   `[nli]` expecting an instant demo will wait. Mitigation: pre-warm
   mentioned in docs, acknowledged in launch copy.

3. **Zenodo deposit DOI not live until you submit.** Do this first
   in §0; all downstream references assume the DOI exists.

4. **HN can flag "field-claiming" posts as grandiose.** The manifesto
   frames three falsifiable laws with cross-validated numbers — that's
   our shield. If flagged, engage in comments with the per-benchmark
   table, not the manifesto prose.

---

**Everything above is copy-paste ready. Execute.**

Good luck. The research is done. The product ships. The only thing
left is to let people see it.
