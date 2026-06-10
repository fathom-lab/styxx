# FINDING — structure-only transmission of novel concepts between minds (TRANSMISSION-DEMONSTRATED)

**2026-06-10 · Fathom Lab / styxx. Pre-registered: `PREREG_telepathy_v0_2026_06_10.md` (frozen
pre-run). Receipt: `telepathy_v0_result.json`. Local, $0. Gate T1 passed.**

"Telepathy" here means exactly this and nothing more: **a concept neither mind has ever shared,
transmitted from one independently-built artificial mind to another through a purely relational
message, over an alignment learned without labels** — no shared tokens, weights, dimensions,
training data, or codebook. No claim about humans, biology, or anything nonlocal.

## Protocol

Sender and receiver each hold norm-equalized geometry over the 96-concept anchor battery — a
shared world, no shared language. The channel is bootstrapped unsupervised by the frozen GAVAGAI
matcher. The sender then encodes a probe concept from a DISJOINT 96-word battery (8 categories
including emotions) as its 96 distances to the sender's own reference concepts, and transmits that
vector alone. The receiver re-indexes the message through the bootstrapped alignment and decodes by
correlation against its own internal profiles for 96 candidates. Chance: 0.0104.

## Result (90 ordered pairs × 96 novel probes)

- **T1 PASSED.** Mean cross-family top-1 decode accuracy **0.1441** — about fourteen times chance
  against the frozen ten-times bar — with a broken-channel null (random-derangement alignments,
  100 runs) whose 95th percentile is 0.0208. Mean cross-family top-5: **0.3867**. Pipeline control
  (self-pair, identity alignment): 1.0.
- **T2 — the code is nearly lossless; alignment is the bottleneck.** With the true alignment given,
  cross-family transfer of never-shared concepts averages **0.8322** top-1. The relational encoding
  itself crosses architectures almost intact (within-family oracle reaches 1.0 in the scored rows,
  e.g. Qwen2.5-0.5B→Qwen2.5-1.5B); what limits the unsupervised channel is recovering the
  correspondence, not carrying the content.
- **T3 — abstract content crosses.** Per-category cross-family top-1: building 0.2167, sport
  0.1979, tool 0.1885, drink 0.1292, **emotion 0.124** — emotions (joy, anger, hope, shame...)
  transmit at roughly twelve times chance through pure structure — kitchen 0.1031, plant 0.101,
  clothing 0.0927.
- Within-family transmission runs far higher (e.g. Qwen2.5-0.5B→Qwen2.5-1.5B 0.615 top-1, oracle
  1.0); cross-lab pairs work too (pythia-410m→gpt2 0.188 top-1, 0.635 top-5, oracle 0.875).

## What this establishes

Composed with `FINDING_gavagai_v0_2026_06_10.md` (identity recoverable from structure) and
`FINDING_anatomy_v2_2026_06_10.md` (the healed apparatus that made both possible): the convergent
geometry of meaning is not only shared and identity-carrying — **it is a functioning communication
channel.** Novel content, including abstract inner-state concepts, moves between minds that share
no language, through shape alone. To our knowledge — prior art mapped and credited in
`PRIOR_ART_structural_translation_2026_06_10.md` (vec2vec requires trained translators; relative
representations require given anchor correspondences; cross-lingual induction operates within one
embedding technology) — this is the first demonstration of its kind, and it ships with frozen
pre-registration and machine-checkable certificates.

## What this does not establish

Accuracy is partial (one novel thought in seven cross-family; top-5 better than one in three); both
minds are measured by the same battery/template instrument (the broken-channel null controls the
decode, not the convention — the cross-objective GAVAGAI-X prereg is the next control); probes are
single words, not propositions; and the oracle gap defines the open problem — better unsupervised
alignment, not better encoding, is where the channel grows.
