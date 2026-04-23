# cognometry launch copy — 2026-04-23 (v4.0.0)

**Manifesto:** https://fathom.darkflobi.com/cognometry
**Leaderboard:** https://fathom.darkflobi.com/cognometry/leaderboard
**Repo:** https://github.com/fathom-lab/styxx
**PyPI:** https://pypi.org/project/styxx/4.0.1/
**Colab demo:** https://colab.research.google.com/github/fathom-lab/styxx/blob/main/examples/cognometry_colab.ipynb
**Paper (Zenodo DOI):** https://doi.org/10.5281/zenodo.19703527
**Prior paper (styxx v3.x):** https://doi.org/10.5281/zenodo.19504993

---

## Hacker News

### Title (primary)

    Cognometry: The measurement of machine cognition

### Title (Show HN variant — if launching the package as the feature)

    Show HN: Styxx v4.0.0 – hallucination detection cross-validated on 8 benchmarks

### Self-comment (paste after submission)

> Author here.
>
> Styxx 4.0.0 is the first hallucination detector I'm aware of that's
> been cross-validated across 8 public benchmarks — HaluEval-QA/Dialog/
> Summarization, TruthfulQA, and four HaluBench subsets (DROP, PubMedQA,
> FinanceBench, RAGTruth). 3-seed averaged, n=150/dataset, pooled
> 9-signal logistic regression.
>
> Headline numbers:
>
>   HaluEval-QA             AUC 0.998
>   TruthfulQA              AUC 0.994
>   HaluBench-RAGTruth      AUC 0.807   (new — RAG faithfulness)
>   HaluBench-PubMedQA      AUC 0.719   (new — biomedical)
>   HaluEval-Dialog         AUC 0.676
>   HaluEval-Summarization  AUC 0.643
>   HaluBench-FinanceBench  AUC 0.492   (below chance)
>   HaluBench-DROP          AUC 0.424   (below chance)
>
> The two below-chance results are the part I'd actually like HN to
> pay attention to. They're published as failure modes, not hidden:
>
> - DROP: reading-comp hallucinations are extractive-span errors.
>   Wrong span, right passage. NLI scores that as entailed; novelty
>   signals don't fire. Fix needs span-level faithfulness scoring.
> - FinanceBench: hallucinations are calculation/aggregation errors on
>   numbers copied verbatim from the source. Novelty + NLI are
>   semantically blind to arithmetic. Fix needs a symbolic
>   verification pass.
>
> Both failure modes are declared in the weights module itself
> (`calibrated_weights_v4.CALIBRATION_NOTES.documented_failure_modes`)
> so users know where the detector will lie to them.
>
> The manifesto names the field this sits inside — cognometry, the
> empirical measurement of cognitive states in LLMs. Three laws, each
> with a cross-validated number. Full deposit + reproducers at the
> GitHub repo.
>
> Install: `pip install styxx[nli]`, wrap a function with `@trust`,
> get verified output on every call.

---

## X / Twitter

### Thread (6 tweets)

> 1/ we're naming a field and shipping the instrument.
>
> **cognometry** — the empirical measurement of cognitive states in
> machine systems. hallucination, refusal, reasoning, retrieval, drift.
> all measurable, from signals already on the token stream.
>
> 🟢 https://fathom.darkflobi.com/cognometry

> 2/ styxx 4.0.0 is the first hallucination detector cross-validated on
> **8 public benchmarks** — 3-seed averaged, n=150/dataset:
>
> HaluEval-QA        AUC 0.998
> TruthfulQA         AUC 0.994
> RAGTruth           AUC 0.807   new
> PubMedQA           AUC 0.719   new
> HaluEval-Dialog    AUC 0.676
> HaluEval-Summ      AUC 0.643

> 3/ two benchmarks came in below chance. we're publishing them, not
> hiding them:
>
> DROP         AUC 0.424 — extractive-span reading comp errors
> FinanceBench AUC 0.492 — calculation errors on verbatim numbers
>
> both are declared as failure modes in the weights module itself.
> users need to know where the detector will lie.

> 4/ three laws, each with a cross-validated number:
>
> i · every computation leaves vitals — AUC **0.998** halueval-qa
> ii · vitals are substrate-transferable — cos **+0.464** llama-1B→3B
> iii · vitals are causally actionable — **97% → 17%** refuse@unsafe
>    at α=3.0 on llama-3.2-1B, multi-position residual patching

> 5/ one decorator, any LLM call, verified output:
>
> ```python
> pip install styxx[nli]
>
> @trust
> def my_rag(q): ...
> ```
>
> MIT on code. CC-BY on calibrated weights. 29 probes across 6 vendors.
> every number has a reproducer in the repo.

> 6/ a field claims itself when someone names it, publishes laws for it,
> and ships the instrument that tests them.
>
> we did all three. the invitation is open — disconfirmations on any
> of the 8 benchmarks go up with our attribution.
>
> 🟢 https://fathom.darkflobi.com/cognometry
> 🟢 https://github.com/fathom-lab/styxx
> 🟢 https://pypi.org/project/styxx/4.0.0/

### Solo tweet (if not threading)

> naming a field today and shipping the instrument.
>
> **cognometry** — the measurement of cognitive states in LLMs at
> runtime. three laws, all cross-validated.
>
> styxx 4.0.0: hallucination detection on **8 benchmarks**
> (5/8 above AUC 0.65, two published failure modes).
>
> pip install styxx[nli]
>
> 🟢 https://fathom.darkflobi.com/cognometry

---

## LinkedIn

> Today we're publishing the founding manifesto for **cognometry** —
> the empirical measurement of cognitive states in machine systems.
>
> Every benchmark on earth scores what the model said. None of them
> answer the question a production operator actually needs answered:
> *was the model refusing, confabulating, retrieving, or reasoning
> when it wrote that?*
>
> Styxx 4.0.0 is the open-source instrument, cross-validated across
> **8 public hallucination benchmarks** — the first detector I'm
> aware of at this breadth of cross-validation. Three laws on the
> table, each with a cross-validated number:
>
> • Law I — every computation leaves vitals (AUC 0.998 HaluEval-QA;
>   5/8 benchmarks above 0.65, 2 published failure modes)
> • Law II — vitals are substrate-transferable (cos +0.464 cross-scale
>   refusal direction, ~26σ above chance)
> • Law III — vitals are causally actionable (refuse@unsafe 97% → 17%
>   at α=3.0 on Llama-3.2-1B, multi-position residual patching)
>
> One decorator (`@trust`) runs the cross-validated detector on any
> LLM call. MIT on code, CC-BY on calibrated weights.
>
> Manifesto: https://fathom.darkflobi.com/cognometry
> Code: https://github.com/fathom-lab/styxx
> PyPI: https://pypi.org/project/styxx/4.0.0/

---

## Defensive Q/A — prepared answers for HN comments

**"Isn't this just eval?"**
> No. Eval scores outputs against a label. Cognometry scores the state
> that produced the output. TruthfulQA with accuracy is eval. TruthfulQA
> with per-response hallucination probability and a calibrated threshold
> is cognometry. The two are complements: eval gives ground truth;
> cognometry gives the runtime signal to act on when ground truth is
> unavailable.

**"8 benchmarks but two are below chance. You're cherry-picking."**
> No — that's exactly the point. The two below-chance results
> (DROP, FinanceBench) are published as declared failure modes in the
> weights module, not buried. We're making two structural claims:
> (a) reference-grounded QA and RAG faithfulness are effectively
> solved by the 9-signal stack, and (b) reading-comp span errors and
> arithmetic errors are NOT solved. Both claims are honest and both
> are necessary for the field to advance.

**"Have you tested at frontier scale?"**
> No. Every causal result we publish is at 1B–3B. Frontier-scale
> replication is an open invitation. `residual_probe.atlas` is designed
> to accept new vendor entries as they land. We'd be delighted to see
> the refusal direction measurement at 70B+.

**"What's the latency overhead of `@trust`?"**
> Text/entity/novelty signals: sub-millisecond. NLI (the v4.0 addition):
> ~10–30 ms on CUDA, ~150–400 ms on CPU per call. Pre-warming
> `get_default_scorer()._load()` at startup eliminates the cold-path
> cost. Grounding + probe signals are optional.

**"Why do I care about 'cognometry' vs just 'hallucination detection'?"**
> Hallucination detection is one measurement within cognometry. The
> same signal stack detects refusal, steering resistance, cognitive-state
> drift, and adversarial perturbation. Naming the field lets us
> unify these measurements under one set of laws and one set of
> instruments, rather than treating each as its own ad-hoc classifier.

---

## Posting notes

- HN: best window is 08:00–10:00 PT (we're ~7h out at the time of
  writing). Post from the primary domain — `fathom.darkflobi.com/cognometry`
  — not from the repo, so the manifesto takes the front page.
- X: thread > solo. Pin the thread. Reply-chain the 6 tweets manually.
- Don't cross-post HN + X simultaneously — X 15 min before HN so the
  launch momentum stacks.
- If HN flags it, the backup framing is Show HN (package + reproducer).
