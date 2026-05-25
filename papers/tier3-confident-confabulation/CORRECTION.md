# The corrected story — draft (the original thread was deleted 2026-05-25)

**Status:** the original (wrong) thread was **deleted** by the operator, so no live
correction is required. This is kept as (a) the internal record and (b) a ready-to-post
*fresh* telling of the correct result, if we choose to. It is NOT framed as "correction
to our last thread" anymore — it stands alone.

**Accuracy note (read before posting):** an early version of this draft said "cosine
can't, use NLI." A threshold sweep (`analyze_clustering_threshold.py`) showed that was
itself an overclaim — the real culprit was the *too-lenient 0.70 threshold*; cosine
recovers to AUC 0.93 at 0.95, NLI (threshold-free) reaches 0.95. The robust claim is
"semantic entropy detects confident confabulation at AUC 0.93–0.95; our 0.55 was a
threshold artifact." Tweets below reflect that.

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
the bug was our clustering threshold: we merged answers that were only ~70% similar, which swallowed the different lies into "one answer."

tighten the threshold (or cluster by entailment, the proper method) and semantic entropy separates fact from confabulation at AUC 0.93–0.95, not 0.55. the lever works. confident error is INCONSISTENT.
```

**5/**
```
the honest version: we manufactured a null with a bad clustering proxy, then caught our own artifact when a deeper probe contradicted it.

caveats stay real — one model, n=4, a flip-flop false-positive mode. full receipts + corrected writeup:
https://github.com/fathom-lab/styxx/blob/main/papers/tier3-confident-confabulation/FINDING_corrected_2026_05_25.md
```

## Single-tweet version (if you'd rather not re-thread)

```
turns out semantic entropy DOES catch confident confabulation — we just clustered wrong at first. the model tells a *different* lie each sample (Renwick: 1842/1723/1912…); our 0.70 similarity threshold merged them into "one answer" → fake null.

fix the threshold (or use entailment): AUC 0.93–0.95. confident error is inconsistent. receipts:
https://github.com/fathom-lab/styxx/blob/main/papers/tier3-confident-confabulation/FINDING_corrected_2026_05_25.md
```
