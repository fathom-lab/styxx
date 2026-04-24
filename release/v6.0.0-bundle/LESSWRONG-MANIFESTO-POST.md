# LessWrong / Alignment Forum post — cognometry manifesto

**Target venue:** LessWrong (crosspost to Alignment Forum)
**Author account needed:** either Flobi or Alexander Rodabaugh (your LW handle)
**Tags:** AI, AI Alignment, Language Models, Interpretability

**Based on:** research evidence that Arditi's "Refusal in LLMs is
Mediated by a Single Direction" LW post drove all initial stars
*before* the arXiv paper. The LW/AF route is the validated path
from obscure solo work to cited canon.

---

## Suggested title (pick one)

Option A (neutral, high-signal):
```
Cognometry: three calibrated text-only instruments for detecting
LLM cognitive failure (styxx v6)
```

Option B (field-claiming, higher swagger):
```
Cognometry: a manifesto for the empirical measurement of cognitive
state in machine systems
```

Option C (specific hook, highest CTR):
```
We beat a hidden-state MLP baseline on tool-call hallucination
using 22 text-only features — and accidentally found phase
transitions in detectability
```

**Recommendation:** Option A. It's neutral enough for LW's "not
promotional" norm, specific enough to earn a read. Option C for
the X crosspost.

---

## Post body (paste verbatim — formatted for LW markdown)

```markdown
**tl;dr** — We published a short manifesto arguing that
*cognometry* — the calibrated, reproducible, black-box measurement
of transient cognitive state in language models — is a coherent
research discipline separate from both LLM psychometrics (stable
traits) and mechanistic interpretability (weight-level internals).
We ship three calibrated instruments as existence proofs, with
three head-to-head wins against named incumbents. We also found a
surprise: detection AUC does not rise smoothly with feature count
— it phase-transitions. Full writeup: [fathom.darkflobi.com/manifesto](https://fathom.darkflobi.com/manifesto)

## The claim

Every forward pass of an LLM writes structure onto the token
stream: logprob trajectories, residual-stream geometry,
generation-order time series, lexical dispersion, schema
conformance. Most production stacks throw these signals away.
**Cognometry is the discipline that picks them up and calibrates
them against failure.**

## Three laws

- **Law I (vitals).** Every computation leaves vitals. Evidence:
  8 hallucination benchmarks, AUCs 0.424 → 0.998 on a single
  9-signal calibrated LR. Signal is there, modulated by domain.
- **Law II (cross-substrate universality).** Cognitive states
  have substrate-independent signatures. Evidence: train refusal
  detector on 80 Llama-3.2-1B apologetic refusals → AUC 0.976 on
  GPT-4 held-out. One documented failure: Mistral-instruct
  lecturing (AUC 0.61 — corpus matters).
- **Law III (predictive validity).** In flight. Our live agent
  economy (DarkCity) logs vitals per call + realized P&L per
  settlement. 30-day dataset will establish or falsify it.

## Three head-to-heads (5-fold CV, committed reproducers)

| Instrument | Our AUC | Baseline |
|---|---|---|
| Hallucination (HaluEval-QA) | **0.998** | Vectara HHEM-2.1-Open: 0.764 *(+0.23, 330× faster)* |
| Refusal (XSTest-v2 GPT-4 held-out) | **0.976** | ShieldGemma-27B: 0.893; Llama-Guard-3-8B: 0.975 |
| Tool-call drift (BFCL v3) | **0.916 ± 0.004** | Healy et al. 2026 hidden-state MLP: 0.72 |

For the drift case specifically: we do it text-only. Works on any
closed-model API. No hidden states required.

## The surprise: phase transitions in detectability

Ran a feature-count ablation on drift expecting a smooth curve.
Got discrete jumps instead:

```
drift class         K=1    K=2    K=6    K=10
arg_drop            0.50   0.998  0.998  1.000   ← K=2: +arg_count_zscore
irrelevance_called  0.49   0.70   0.83   0.96    ← K=10: +prompt_coverage
arg_swap            0.51   0.49   0.69   0.68    ← K=6: +type_mismatch_frac
```

Each failure class has a *critical feature*. Below it: chance.
Above it: class solved in one step. This is the inverse of
emergent capabilities in generative LLMs — detection emerges in
discrete jumps as feature count scales, not smooth curves. If the
pattern generalizes across cognometric instruments (it should),
every detector has a minimum feature count below which specific
failure classes are structurally undetectable.

## Differentiation from adjacent work

- **Psychometrics** ([Ye et al. 2025](https://arxiv.org/abs/2505.08245))
  measures stable traits. Cognometry measures transient state.
  Different measurement targets.
- **Mechanistic interpretability** requires weight access.
  Cognometry doesn't. Complementary, not competing.
- **Semantic entropy** ([Farquhar et al., *Nature* 2024](https://www.nature.com/articles/s41586-024-07421-0))
  needs multiple inference passes. Our detectors are single-pass,
  sub-millisecond, CPU-only. Different point on the design frontier.

## What I'd love feedback on

1. Is **Law II** overclaimed? The Mistral-instruct failure is
   real. Would welcome thoughts on when/where cross-substrate
   universality breaks — we can only test what we can sample.
2. Is the manifesto framing counter-productive? We wrote it
   because "another detection library" doesn't compound; a
   named discipline does. But if the framing lands as
   overreach, we want to know before arXiv submission.
3. Which **instrument #4-#9** is most tractable? Sycophancy has
   a natural calibration corpus (Anthropic's sycophancy eval);
   goal-drift has no benchmark yet. We'd love collaborators on
   any of them — if you build instrument #4, we'll PR your
   method in and credit you as the instrument's founder.

## Artifacts

- **Manifesto (the essay):** https://fathom.darkflobi.com/manifesto
- **Paper (Zenodo DOI):** https://doi.org/10.5281/zenodo.19703527
- **Repo (MIT):** https://github.com/fathom-lab/styxx
- **Try live (Pyodide, no install):** https://fathom.darkflobi.com/cognometry
- **OSF reproducer bundle:** https://osf.io/6syq4/
- **`pip install styxx[nli]`**

Every AUC has a committed reproducer; rerunning from `random_state=0`
produces the same number. Two failure modes (DROP 0.424, FinanceBench
0.492) are declared openly in the weights module itself.

Happy to answer questions. Collaborators welcome.
```

---

## Posting checklist

- [ ] Pick a LessWrong account to post from (Flobi, Alexander
      Rodabaugh, or a new one — if new, expect lower initial
      traction; existing karma matters)
- [ ] Tag appropriately: `AI`, `AI Alignment`, `Language Models`
- [ ] Turn on **Crosspost to Alignment Forum** checkbox (mandatory
      for this type of content — AF audience is the target)
- [ ] First comment (30 minutes after post): quote the
      phase-transition table and say "happy to run this
      ablation on other detectors if anyone wants to test it"
      — invites participation
- [ ] After LW post is up, post the X crosspost:

```
we just published a manifesto arguing cognometry is a real
discipline — the measurement of transient cognitive state in
LLMs — and shipped 3 calibrated text-only detectors to prove it

0.998 / 0.976 / 0.916 AUC
3 head-to-heads
0 LLM inference

fathom.darkflobi.com/manifesto
```

Tag @NeelNanda_, @OwainEvans_UK, @natolambert in the thread — not
all at once, one per tweet in a 3-reply thread to avoid mention-spam
flagging.

---

## If the post catches (>30 karma in 6 hours)

Immediate follow-ups:
- Reply to every substantive comment within 20 minutes
- Cross-link to the specific Colab reproducer for whichever
  instrument a commenter is probing
- If an Anthropic/DeepMind researcher engages, DM them the
  full methodology doc and offer a 15-min call

## If the post dies (<10 karma in 12 hours)

The framing was wrong. Retry options:
- Ditch the "manifesto" framing entirely. Rewrite as "Three
  detectors that beat hidden-state baselines: what we learned."
- Lead with the phase-transition finding, bury the
  laws + manifesto in a `Broader context` appendix.
- Post to HN instead as `Show HN: styxx v6 — detect LLM
  hallucination/refusal/drift without hidden states`.
