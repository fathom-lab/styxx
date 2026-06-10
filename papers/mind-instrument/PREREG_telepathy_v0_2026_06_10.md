# PREREG — TELEPATHY v0: structure-only transmission of novel concepts between minds

**Frozen 2026-06-10, before any scored run. Fathom Lab / styxx.**

"Telepathy" here is operationally defined and nothing more: **a NOVEL concept, never shared between
two independently-built minds, transmitted from one to the other through a purely relational
message over an alignment learned WITHOUT labels** — no shared tokens, weights, dimensions,
training data, or codebook. GAVAGAI v0 showed the convergent geometry carries identity for the
reference set itself; this tests whether that channel can carry NEW content.

## Protocol (frozen)

1. **Shared world, no language:** both minds hold norm-equalized reps
   (`PREREG_anatomy_v2_normeq_2026_06_10.md` convention) of the 96-concept anchor battery
   (`styxx.mind.BATTERY`).
2. **Channel bootstrap (unsupervised):** alignment π between A's and B's battery indices from the
   frozen GAVAGAI translator (`run_gavagai_v0.translate`) on their battery RDMs — geometry only,
   labels hidden.
3. **Message:** for a probe concept w* from the PROBE SET — the 96 fresh-battery words of the
   confirm run (8 categories incl. EMOTION; disjoint from the anchor battery) — A computes reps of
   battery+probe, centers all rows on A's battery mean, row-normalizes, and transmits the 96-vector
   of Euclidean distances from w* to its battery rows. Nothing else crosses the channel.
4. **Decode:** B builds the same-form profile for each of the 96 probe-set candidates from ITS OWN
   reps, and scores candidate c by Pearson correlation between the received message (re-indexed
   through π) and its profile for c; decodes argmax. Chance = 1/96.

## Pre-registered gates (frozen)

- **T1 (the channel carries novel content):** mean cross-family top-1 decode accuracy over all
  cross-family ordered pairs × 96 probes ≥ 10× chance (≥ 0.1042), AND a broken-channel null
  (π replaced by a seeded random derangement; same decode; 100 runs, seed 0) with 95th percentile
  below the observed mean. PASS → **TRANSMISSION-DEMONSTRATED**; FAIL → **CHANNEL-INSUFFICIENT**
  (the geometry carries reference-set identity but not novel content at this scale — reported as
  the bound).
- **T2 (alignment cost, descriptive):** same decode with the TRUE alignment (identity π) — the
  oracle ceiling; the gap quantifies what unsupervised bootstrap costs. No bar.
- **T3 (abstract content, descriptive):** per-probe-category accuracy, EMOTION highlighted — does
  the channel carry abstract thoughts? No bar; reported either way.
- **Top-5 accuracy** reported alongside top-1 (no bar).

## VOID

- **VOID-PIPELINE:** self-pair (A→A, identity π) mean decode accuracy must be ≥ 0.99.
- Smoke = 2 minds, writes `*_SMOKE_INVALID*` only.
- GPU pass computes battery+probe reps for all 10 minds under the frozen normeq convention; reps
  persisted (`telepathy_reps.npz`) as the receipt.

## Honest framing commitment

The FINDING, whatever the verdict, will define the term operationally in its first sentence, claim
nothing about humans, biology, or nonlocality, and state the shared-instrument caveat (both minds
measured by the same battery/template convention; the broken-channel null controls the decode, not
the convention).
