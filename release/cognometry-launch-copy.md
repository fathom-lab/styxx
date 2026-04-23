# cognometry launch copy — 2026-04-22

**Manifesto URL:** https://fathom.darkflobi.com/cognometry
**Repo:** https://github.com/fathom-lab/styxx
**PyPI:** https://pypi.org/project/styxx/
**Paper (v3.x family):** https://doi.org/10.5281/zenodo.19504993

---

## Hacker News

### Title (primary — use this one)

    Cognometry: The measurement of machine cognition

### Title (backup, more specific)

    Show HN: Cognometry – measuring cognitive states in LLMs at runtime

### Submission note (optional — HN usually just wants the URL)

No submission text needed. Let the page stand on its own.

If prompted for a self-comment, paste:

> Author here. The manifesto puts a name and three testable laws around
> something a lot of safety/eval teams are already doing: measuring
> cognitive states of LLMs at runtime instead of just scoring the output.
>
> Every claim has a reproducer. The open-source instrument is `styxx`
> on PyPI — `pip install styxx` + one decorator `@trust` runs the
> HaluEval-QA AUC 1.000 / TruthfulQA AUC 0.977 hallucination detector
> on any LLM call.
>
> Happy to discuss the laws, the places the universality claim breaks
> (Phi-3.5 cross-vendor transfer goes null — expected and published),
> or the v4.0 NLI work that gets dialog/summarization AUC from 0.60 → 0.80+.

---

## X / Twitter

### Thread (5 tweets)

> 1/ we're naming a field.
>
> **cognometry** — the empirical measurement of cognitive states in
> machine systems. hallucination, refusal, reasoning, retrieval, drift.
> all measurable. all from signals already on the token stream.
>
> founding manifesto: https://fathom.darkflobi.com/cognometry

> 2/ three laws, each with a cross-validated number:
>
> i · every computation leaves vitals — AUC **1.000** on HaluEval-QA, **0.977** on TruthfulQA (held out).
>
> ii · vitals are substrate-transferable — cos **+0.464** llama-1B → llama-3B on the refusal direction (~26σ).
>
> iii · vitals are causally actionable — refuse@unsafe **97% → 17%** at α=3.0 (llama-3.2-1B, multi-position patch).

> 3/ this isn't a paper. it's an instrument.
>
> ```
> pip install styxx
>
> @trust
> def my_rag(q): ...
> ```
>
> one decorator. verified output. fallback / retry / raise / annotate.
> MIT on code, CC-BY on weights. 29 probes across 6 vendors shipped.

> 4/ honest limits (published in the manifesto, not hidden):
>
> • dialog/summarization AUC only 0.60 — novelty can't separate faithful addition from contradiction. v4.0 adds NLI, lifts to ~0.70+.
> • cross-vendor universality is null when alignment regimes diverge (qwen → phi). the law fails where it should fail.
> • every causal result is at 1B–3B. 70B+ is open.

> 5/ a field claims itself when someone writes the first instrument,
> publishes the first number, and puts a name on both.
>
> we did all three today. the invitation is open.
>
> 🟢 https://fathom.darkflobi.com/cognometry
> 🟢 https://github.com/fathom-lab/styxx

### Solo tweet (if not threading)

> naming a field today.
>
> **cognometry** — the measurement of cognitive states in LLMs at
> runtime. three laws, all cross-validated. AUC 1.000 on HaluEval-QA,
> 0.977 on TruthfulQA, refuse@unsafe 97%→17% causal.
>
> open instrument: `pip install styxx`
>
> 🟢 https://fathom.darkflobi.com/cognometry

---

## LinkedIn / long-form

### Post (if pushing to LinkedIn)

> Today we're publishing the founding manifesto for **cognometry** —
> the empirical measurement of cognitive states in machine systems.
>
> Every benchmark on earth scores what the model said. None of them
> answer the question a production operator actually needs answered:
> *was the model refusing, confabulating, retrieving, or reasoning
> when it wrote that?*
>
> We put three laws on the table, each with a cross-validated number:
>
> • Law I — every computation leaves vitals (AUC 1.000 on HaluEval-QA,
>   0.977 on TruthfulQA, held-out test)
> • Law II — vitals are substrate-transferable (cos +0.464 cross-scale
>   on the refusal direction, ~26σ above chance)
> • Law III — vitals are causally actionable (refuse@unsafe 97% → 17%
>   at α=3.0 on Llama-3.2-1B, multi-position residual patching)
>
> The open-source instrument is **styxx** — MIT on code, CC-BY on
> calibrated weights. One decorator (`@trust`) runs the cross-validated
> detector on any LLM call.
>
> Manifesto: https://fathom.darkflobi.com/cognometry
> Code: https://github.com/fathom-lab/styxx
> Zenodo paper: https://doi.org/10.5281/zenodo.19504993
>
> This is a founding document. The invitation to extend, disconfirm,
> or replicate at frontier scale is open.

---

## Notes

- HN: submit Wednesday 08:00–10:00 PT for best visibility, or now
  if the momentum is live.
- X: thread version > solo. Pin the thread.
- If HN flags it as a "Show HN", re-post with that prefix later —
  "Show HN" is for launching something specific; this is a field claim
  + instrument, so a bare title works.
- Ready to respond to the obvious critique ("isn't this just eval?"
  → no, eval scores outputs; cognometry scores the state that produced
  the output) and ("you're overreaching with laws" → every law has a
  committed reproducer, happy to retract one if anyone disconfirms).
