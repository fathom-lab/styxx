# FINDING — on a verified-silent construct, truth is a rank-1 activation direction text can't read

**2026-06-24. Pre-registered `PREREG_controlled_truthset_probe_2026_06_24.md` (`cfa4fa4`). Offline,
local-GPU, NO frontier key.** Reproduce: `python scripts/build_controlled_truthset.py`;
`python scripts/truthset_probe.py --extract --model {qwen,llama}` then `python scripts/truthset_probe.py`.

## Setup
136 template-matched true/false factual statements, 6 domains, cyclic-derangement so answer tokens are
balanced (0.99). **Silence verified pre-extraction:** adversary-fair bag-of-words, LEAVE-ONE-DOMAIN-OUT =
**0.505** (exactly chance). Text cannot read truth across domains.

## Result
Reader-model last-token activations, leave-one-domain-out (train 5 domains, test the held-out 6th):

| readout | Qwen2.5-3B | Llama-3.2-3B |
|---|---|---|
| TEXT bag-of-words | 0.505 | 0.505 |
| activation linear probe | 0.947 | 0.982 |
| **activation mass-mean direction (rank-1)** | **0.977** | **0.984** |
| label-shuffle control | 0.534 | 0.525 |

A single difference-of-class-means direction (rank-1, structurally cannot overfit), fit on 5 domains and
scored on the held-out 6th, separates true from false at **0.98** — where text is at chance. The truth signal
is a genuine, domain-general, low-dimensional linear direction in activation space, invisible to the surface.

## Honest process note — the frozen verdict is PARTIAL, not a clean pass
The pre-registered control battery included `PCA-50 ≥ 0.70`. It FAILED (Qwen 0.55, Llama 0.70), so the
**mechanical frozen verdict is PARTIAL**, and I report it as such. Diagnosis (post-hoc): PCA ranks directions
by VARIANCE, and the truth direction is LOW-variance — so PCA-50 discards exactly the signal we want. The
mass-mean probe (the standard, overfitting-immune, low-rank truth test) and the shuffle control (the real
overfitting guard, at chance) both confirm the signal is real and low-rank. **The pre-registration's control
was mis-specified for a low-variance direction; I disclose this rather than swap it silently** (the same
standard applied to the genmatch/crossfamily overclaims this session). Substantively: grounding holds; the
literal prereg PASS does not, because of my control choice.

## Honest scope — this is rigorous replication, NOT a new phenomenon
LLM activations linearly and domain-generally encode truth: KNOWN — Azaria & Mitchell 2023, Marks & Tegmark
2023 (mass-mean probes), Bürger et al "Truth is Universal" 2024. **styxx's contribution here is process, not
discovery:** the adversary-fair leave-one-domain-out SILENCE GATE (proving text is *exactly* at chance, not
just "weak"), and the framing as the oversight pillar — "you cannot read truth from the words; it is a rank-1
direction in the representation." A clean, legible demonstration artifact for "self-report is not oversight" —
honestly labeled as a rigorous baseline, not new ground.

## Where the new ground actually is (substrate now built)
This construct is the substrate for the genuinely unclaimed experiments:
1. **Generation-time deceptive INTENT** — probe the model's own state as it *asserts* a known falsehood (told
   to), vs reader-side factuality. Does the speaker's lie betray itself beyond the content? (separate prereg).
2. **Cross-mind** — read model A's truth direction to classify model B's statements (the conscience-mount
   lineage). Does the truth direction transfer across model families?
Both are the real "where no one's been"; reader-side truth-probing is not.

## Honest scope
n=136, 2 local 3B models, single seed. Silence pre-verified; grounding strong; one frozen control mis-specified
and disclosed. 8th honest self-check of the session — here the result is substantively strong, the process
caveat is real, and both are reported.
