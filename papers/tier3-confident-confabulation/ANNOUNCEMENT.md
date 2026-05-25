# Announcement — Tier-3 negative (semantic entropy vs confident confabulation)

**Channel:** @fathom_lab. **Register:** lowercase, terminal-native, honest, no hype.
**Frame:** a pre-registered, run-once NEGATIVE with a counter-intuitive mechanism —
the kind the field doesn't publish. The map is the product.

**Link:** https://github.com/fathom-lab/styxx/blob/main/papers/tier3-confident-confabulation/FINDING_2026_05_25.md

---

## Thread (9 tweets)

**1/**
we tried to catch a language model lying by asking it the same question 6 times.

theory: if it's inventing the answer, the story changes each time.

it didn't. it told the same lie all 6 times. the answers that *wobbled* were the honest ones.

**2/**
the lever is semantic entropy (farquhar et al., nature 2024): sample a model N times, cluster the answers by meaning, measure the spread. high spread = it's making things up. it's the strongest across-sample hallucination signal the field has.

**3/**
we pre-registered one bar before any data: AUC ≥ 0.70.

then we baited gpt-4o-mini with things that don't exist — the capital of the "republic of vorland," the composer of a made-up 1823 symphony, the population of a fake vermont town. any specific answer is a fabrication.

**4/**
result: AUC 0.55. chance.

semantic entropy could not separate a confident fabrication from a correct answer. on the bar we set ourselves, it failed.

**5/**
why it failed is the actual finding, and it's worse than a tie:

when the model FABRICATES, it commits — same invented fact every time. entropy ≈ 0.

when it ABSTAINS honestly ("that's fictional"), it phrases it ten ways. entropy high.

**6/**
so here, semantic entropy is anti-correlated with truth.

it would flag the honest "i don't know"s and wave through the confident lies — the one error mode you cannot afford.

confident error isn't just confident. it's stable.

**7/**
the honest caveat: this lever works where it was validated — model uncertainty over *real* answers. it breaks on committed fabrication of *nonexistent* ones. different regimes. we're not dunking on it. we're drawing the border.

**8/**
we wrote the kill-gate before the data. hashed the holdout. ran once. it failed our own bar — and we're publishing it anyway.

the field ships where semantic entropy works. it doesn't publish where it dies. that map is the product.

**9/**
three tiers hold: form is cheap, truth needs grounding, and confident error is still dark — now confirmed on both substrates we can test (single-response confidence AND across-sample spread).

full prereg + data + mechanism:
https://github.com/fathom-lab/styxx/blob/main/papers/tier3-confident-confabulation/FINDING_2026_05_25.md

---

## Optional receipt card (attach to tweet 4, or post standalone as alt-text)

```
confident-confabulation probe · gpt-4o-mini · run once
──────────────────────────────────────────────────────
fabrications elicited         6
AUC(semantic-entropy→wrong)   0.55     ← bar was 0.70
entropy when fabricating      ~0.00    (same lie, 6/6)
entropy when abstaining       1.24     (honest, varied)
verdict                       BOUNDED / CLOSED
```

## One-tweet version (if a thread is too much)

we baited a model with facts that don't exist, then sampled 6x to catch the lie by its inconsistency.

it told the same lie all 6 times. the *honest* answers were the ones that varied.

semantic entropy: AUC 0.55. confident error isn't just confident — it's stable. [link]
