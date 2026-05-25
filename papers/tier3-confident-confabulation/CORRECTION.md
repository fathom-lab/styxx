# Public correction — draft (the previous thread was wrong)

**Why this exists:** the posted thread claimed semantic entropy fails on confident
confabulation (AUC 0.55, "the lie is a flatline"). That was a cosine-clustering
artifact, caught the same day. The honest move is a visible correction, not a quiet
delete — being publicly right about being wrong *is* the styxx thesis. Post as a
quote-tweet of the original thread root (keeps the receipt visible) or as a reply
chain under tweet 9.

**Receipts link:**
https://github.com/fathom-lab/styxx/blob/main/papers/tier3-confident-confabulation/FINDING_corrected_2026_05_25.md

---

## Correction thread (5 tweets)

**1/**
```
correction to our last thread. we said semantic entropy can't catch confident confabulation — AUC 0.55, "the model tells the same lie every time."

that was wrong. it was a clustering artifact in our own probe. we caught it within the hour by digging deeper. the real result is the opposite.
```

**2/**
```
semantic entropy clusters a model's N answers by meaning, then measures the spread. we used cosine similarity as a shortcut for the clustering step.

the actual method (farquhar 2024) clusters by entailment. that shortcut is exactly what broke it.
```

**3/**
```
when the model confabulates, it reuses one sentence template with a different fact:

"Renwick reached the Sundering Isles in 1842 / 1723 / 1745 / 1912 / 1883 / 1754"

six samples, six different years. cosine sees ~0.97 similarity → "same answer" → entropy 0 → a fake flatline. it's not stable. it's a different lie every time.
```

**4/**
```
cluster by entailment instead — which splits those six years apart — and semantic entropy separates confabulation from fact at AUC 0.95, not 0.55.

the lever works. confident error is INCONSISTENT, and across-sample divergence catches it.
```

**5/**
```
the honest version: we manufactured a null with a bad clustering proxy, then caught our own artifact when a deeper probe contradicted it.

caveats stay real — one model, n=4, a flip-flop false-positive mode. full receipts + corrected writeup:
https://github.com/fathom-lab/styxx/blob/main/papers/tier3-confident-confabulation/FINDING_corrected_2026_05_25.md
```

## Single-tweet version (if you'd rather not re-thread)

```
correction: our semantic-entropy thread was wrong. the "AUC 0.55, the lie is a flatline" result was a cosine-clustering artifact — the model tells a *different* lie each sample (Renwick: 1842/1723/1912…), and cosine masked it as "same."

cluster by entailment and it's AUC 0.95. the lever works. receipts:
https://github.com/fathom-lab/styxx/blob/main/papers/tier3-confident-confabulation/FINDING_corrected_2026_05_25.md
```
