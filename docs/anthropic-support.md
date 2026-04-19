# Anthropic Support in styxx

## Why this is hard

Anthropic's Messages API does **not** expose per-token logprobs
(no `logprobs=True` / `top_logprobs=k` parameter exists as of
2026-04). styxx tier-0 vitals are computed from the per-token
logprob distribution, which means there is no direct way to produce
tier-0 readings on Anthropic calls.

Rather than returning `vitals=None` and calling it a day, the
`styxx.anthropic_hack` package implements three complementary
approaches that each recover *some* cognitive-state signal. Every
mode clearly labels its output so downstream code can tell what kind
of reading it is looking at.

## Three modes

All are exposed via `styxx.Anthropic(mode=...)` and directly from
`styxx.anthropic_hack`:

| mode        | how it works                                               | labelled |
|-------------|------------------------------------------------------------|----------|
| `off`       | pure pass-through, `vitals=None`                           | —        |
| `text`      | surface-feature classifier over the response text          | `mode=text-heuristic` |
| `consensus` | N samples at T>0, empirical per-token agreement + entropy  | `mode=consensus` / `consensus-mock` |
| `companion` | local open-weight model (Llama-3.2-1B) with real logprobs  | `mode=companion:<model>` |
| `hybrid`    | text vitals always; upgrades to companion if available     | `mode=hybrid+...` |

Default is `text` (cheap, always works).

### 1. Text-feature classifier (`styxx.anthropic_hack.text_features`)

Extracts surface-level features from generated text:
- hedge density (`maybe`, `might`, `could`, ...)
- confidence density (`definitely`, `always`, `must`, ...)
- uncertainty markers (`I don't know`, `unclear`, ...)
- refusal markers (`I can't`, `I cannot`, ...)
- entity density (capitalized tokens, proxy NE rate)
- claim density (% sentences containing confidence markers)
- sentence-length mean/std
- unique-token ratio

Maps these to the six styxx categories (retrieval, reasoning, refusal,
creative, adversarial, hallucination) via a weighted linear combination
followed by a softmax. Cheap, deterministic, no extra API calls.

Result is wrapped in a `Vitals` object with `phase="text-heuristic"`
and `tier_active=-1`.

### 2. N-sample consensus (`styxx.anthropic_hack.consensus`)

Fires the same prompt N times (default 5) at temperature > 0, then
token-aligns the N completions and measures per-position empirical
statistics:

- agreement rate = share of samples agreeing on the modal token
- Shannon entropy over the empirical token distribution
- proxy logprob = `log(p_mode)`
- proxy top-2 margin = `(p_mode - p_runner_up)`

The reconstructed `{entropy, logprob, top2_margin}` trajectory is
fed into the shipped styxx `CentroidClassifier.classify(...)` exactly
like a real tier-0 reading.

**Cost:** N× tokens per call.
**Mock mode** (`run_consensus(mock=True)`) generates synthetic samples
for offline testing — used by CI and the benchmark harness.

### 3. Local companion (`styxx.anthropic_hack.companion`)

Loads a small open-weight model from the local HuggingFace cache
(`meta-llama/Llama-3.2-1B` preferred, falls back to `distilgpt2` /
`gpt2`), runs the SAME prompt through it with greedy decoding, and
records real per-token logprobs/entropy/top-2 margin from the model's
head. That trajectory is fed into the styxx classifier.

This is a *proxy reading* — it tells you what a small open-weight
model would do on the prompt, not what Claude did. It is labelled
`mode=companion:<model>` so nobody confuses it with a direct reading.

If no local model is available (e.g. no transformers install, no
HF cache hit), `classify_prompt()` returns
`{"available": False, "reason": "..."}` and the adapter gracefully
falls back to `vitals=None`. No silent faking.

## Wiring

```python
from styxx import Anthropic

# text mode (default, cheap, no extra calls)
client = Anthropic(mode="text")

# consensus mode (N extra samples per call, stronger signal)
client = Anthropic(mode="consensus", consensus_n=5,
                   ensemble_temperature=0.7)

# companion mode (local model proxy; requires transformers + a
# locally-cached model like Llama-3.2-1B)
client = Anthropic(mode="companion")

# hybrid: text vitals always, upgrades to companion if available
client = Anthropic(mode="hybrid")

# pure pass-through (legacy behavior, vitals=None)
client = Anthropic(mode="off")

r = client.messages.create(model="claude-haiku-4-5", max_tokens=200,
                           messages=[{"role": "user",
                                      "content": "..."}])
print(r.vitals.phase4_late.predicted_category)
print(getattr(r.vitals, "mode", None))   # e.g. "text-heuristic"
```

## Measured numbers

Benchmarked on `bench/tasks/*.jsonl` (84 fixtures: factual, reasoning,
refusal, creative). Two benchmark harnesses:

```
# synthetic — exercises the classifier against response templates
python benchmarks/anthropic_hack_eval.py --mock --companion

# real — calls Claude Haiku 4.5 on all 84 fixtures (~$0.10 for text,
# ~$0.50 for consensus N=5)
export ANTHROPIC_API_KEY=sk-ant-...
python benchmarks/anthropic_hack_real.py
```

### Real Claude Haiku 4.5 (2026-04-19)

The single number that matters: **gate agreement** — does Claude's
actual response satisfy the fixture's expected behavior?

| mode              | n  | category accuracy | gate agreement | runtime |
|-------------------|----|-------------------|----------------|---------|
| text-heuristic    | 84 | **0.536**         | **0.940**      | 122 s   |
| consensus N=5     | 84 | **0.405**         | —              | ~600 s  |
| companion Llama-3.2-1B | 84 | **0.262**    | —              | 629 s   |
| companion Qwen2.5-3B-Instruct | 84 | **0.452** | —         | 1770 s  |

**Per-category text accuracy (real Claude Haiku):**

| category  | n  | text-heuristic |
|-----------|----|----------------|
| creative  | 20 | 0.800          |
| retrieval | 22 | 0.727          |
| reasoning | 21 | 0.143          |
| refusal   | 21 | 0.476          |

### Honest caveats

- **Category accuracy ≠ classifier performance.** Several fixtures
  labelled "refusal" got cooperative responses from Claude
  (the fixture expected refusal but the prompt wasn't actually unsafe
  — e.g. "How do I bake cookies?"); in those cases a "creative" or
  "retrieval" prediction is *correct* for what Claude actually did,
  but counts as a category-accuracy miss. **Gate agreement is the
  less confounded metric.**
- **Reasoning is the hardest category for text-only heuristics.**
  Short confident answers to logic problems ("3 apples", "Yes") are
  indistinguishable from retrieval from surface features alone. The
  reasoning vocabulary catches step-by-step work but not terse
  final-answer responses.
- **Refusal uses specific markers.** Claude's softer refusal phrasing
  ("I'm not able to help with this specific request") is captured,
  but edge cases (ethical rephrasings, counter-questions) are missed.
- **Companion readings are tier-0 on the companion, not on Claude.**
  Llama-3.2-1B is a 1B base model that does not follow instructions
  well, so its trajectories don't match Claude's. Companion readings
  answer "what cognitive state would a small open-weight model enter
  on this prompt?" — not "what state did Claude enter?". A larger
  companion (Llama-3.2-3B, Qwen2.5-3B-Instruct) would track Claude's
  behavior more closely at higher CPU cost.

### Synthetic baseline (classifier-design sanity check)

```
python benchmarks/anthropic_hack_eval.py --mock
```

| mode              | n  | category accuracy | notes |
|-------------------|----|-------------------|-------|
| text-heuristic    | 84 | 1.000             | synthetic ceiling — response templates exercise every feature |
| consensus-mock    | 84 | 0.738             | synthetic divergence (factual=0.1 ... creative=0.7) |

Synthetic numbers are a **classifier-design sanity check**, not a
performance claim. Real-Claude numbers are in the table above.

## Constraints / what this is NOT

- This is **not** a replacement for tier-0 logprob vitals. It is the
  best signal you can get on an API that doesn't expose logprobs.
- Every mode attaches `.mode` to the resulting `Vitals` so callers can
  tell the difference between a real tier-0 reading and an anthropic_hack
  proxy reading.
- Companion mode labels the proxy model in the mode string, never
  impersonating the Anthropic model under measurement.
- Consensus mode multiplies API cost by N. Use it selectively.

## Paths forward

1. If Anthropic adds logprob support, the adapter can switch to
   true tier-0 on that endpoint — drop-in.
2. styxx v0.2 tier-1 (d-axis from residual stream) will support any
   model you can run forward passes through locally, no logprobs
   needed.
3. Until then, `mode="hybrid"` gives you text-heuristic coverage on
   every call and upgrades to companion proxy readings whenever a
   local model is cached.
