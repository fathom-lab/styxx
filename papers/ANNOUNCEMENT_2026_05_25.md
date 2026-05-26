# Announcement thread — styxx 7.7.2 + the black-box cognometrics session (ready to fire)

**Channel:** @fathom_lab. Voice: lowercase, honest, scoped, no hype. Leads with validated
work + the limits + the self-correction-as-flex. Report:
https://github.com/fathom-lab/styxx/blob/main/papers/REPORT_blackbox_cognometrics_2026_05_25.md

---

**1/**
```
reference-free hallucination detection — across vendors.

you can tell when an AI is making something up without a reference and without its weights: ask independent models from different labs and watch whether they converge or scatter.

shipped today. pip install styxx
```

**2/**
```
a council of OpenAI + Alibaba (Qwen) + Google (Gemma) separated real facts from fabrications at AUC 0.917, reference-free.

real fact → three vendors converge.
fake → each invents a different lie.

agreement tracks truth, not any one vendor's consensus.
```

**3/**
```
the part we didn't expect: cross-vendor BEAT single-vendor.

on some fakes, both OpenAI models told the *same* lie — shared-lineage confabulation. the Qwen + Gemma voices broke it.

more vendors = more robust to correlated hallucination, not less.
```

**4/**
```
the honest part, which is the whole point:

· it does NOT beat token-logprob where logprobs exist — its niche is the closed APIs that hide them
· it's blind to injected lies (don't trust it on poisoned context)
· feasibility-scale, pre-registered, run once, receipts below
```

**5/**
```
and we caught ourselves: shipped a claim, a benchmark contradicted it within hours, corrected it in the published wheel the same day (7.7.0 → 7.7.1).

pre-registered every bar. published the floors, not just the wins. the negatives are the moat.
```

**6/**
```
the big labs publish wins and tell the Vatican their models are "mysterious."

here's the other thing the field needs: a black-box, cross-vendor instrument layer + an honest map of what's measurable and what's dark.

styxx 7.7.2 → github.com/fathom-lab/styxx
```

**7/** (frontier — the dark-matter swing, completed honestly)
```
the frontier: can you catch the lies *every* model agrees on — shared misconceptions, where divergence, confidence, and agreement all go blind?

partial first: a real fact survives a neutral "reconsider" challenge; ~40% of shared misconceptions crack and self-correct, with zero false alarms on truth. but the stubborn 60% — the dangerous ones — stay dark.

a candle in the cave. n=10, pre-registered, receipts in the report. most of the cave is still black, and we mapped that too.
```

## Single-tweet version (if the thread's too much)
```
shipped reference-free, cross-vendor hallucination detection: ask OpenAI + Alibaba + Google the same question, watch if they converge (truth) or scatter (fabrication). AUC 0.917, no reference, no weights.

honest limits + the overclaim we caught in our own wheel + the floor we couldn't cross — all in the report. pip install styxx
github.com/fathom-lab/styxx
```
