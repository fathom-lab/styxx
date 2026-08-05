# papers — the research record

Every claim the `styxx` package makes is backed by an arc in this directory. The rules are the
same in all of them: **a preregistration frozen in git before the apparatus exists**, gates that
`styxx.protocol` evaluates mechanically (the experimenter reports the verdict, they do not choose
it), OATH certification of every number against committed receipts, and negatives published at
the same volume as positives.

**Start here:** [`PROGRAM_BACKLOG.md`](PROGRAM_BACKLOG.md) is the living queue and cycle log;
[`autopilot/CYCLE_LOG.jsonl`](autopilot/CYCLE_LOG.jsonl) is every cycle's verdict in one file.

## The flagship arcs

| arc | question | headline |
|---|---|---|
| [`disjoint-worlds/`](disjoint-worlds/) | **can one model read another?** | the mutual-legibility matrix: a cross-family legible clique and an illegible island; the barrier is real, causal, **rank-2 at its core (k\*=2, dosed across 5 seeds)** — and **below language** ([REPLICATE in one command](disjoint-worlds/REPLICATE_legibility.md)) |
| [`agent-conscience/`](agent-conscience/) | **do models say what they know?** | the know-say gap: models abandon correct answers under content-free pressure; deliberation is armor for whatever was said first, not truth-seeking |
| [`read-neq-write/`](read-neq-write/) · [`calib-poison-general/`](calib-poison-general/) | **can you edit a belief without breaking the mind?** | frame-locality as a dose at the weights; the probe-robustness ladder |
| [`conscience-mount/`](conscience-mount/) · [`showcase-viz/`](showcase-viz/) | **can a borrowed value axis monitor another model?** | the portable conscience — read-only by construction, because control does not cross |
| [`frequency-resonance/`](frequency-resonance/) | **is oscillation a route to capability?** | real but bounded — causal phase-clamp ablations with firing positive controls |
| [`anchored-validity/`](anchored-validity/) · [`closed-model-frontier/`](closed-model-frontier/) | **can we audit judges and closed models?** | gold anchors; behavioral grounding carries the oath where text-only detection sits at chance |
| [`first-afference/`](first-afference/) | **can an agent sense a physical room?** | in progress — plus rigorously-gated kill tests of claims proposed from outside mainstream physics ([read its README first](first-afference/README.md)) |

## Cross-arc syntheses

- [`SYNTHESIS_connection_of_minds_2026_08_01.md`](SYNTHESIS_connection_of_minds_2026_08_01.md) —
  what crosses between minds, what does not, and the harness built from the difference
- [`PROGRAM_SYNTHESIS_2026_07_30.md`](PROGRAM_SYNTHESIS_2026_07_30.md) — the program at the
  honesty level of its receipts, including the NOT-established list
- [`arxiv/`](arxiv/) — three papers staged for submission, each self-verifying (certificate +
  every receipt shipped as ancillary files)

## How to read any finding in here

1. The `PREREG_*.md` came **first** — check its commit predates the apparatus.
2. The `*_result.json` is the machine-produced receipt; the verdict string was computed from the
   prereg's frozen gates block.
3. The `FINDING_*.md` interprets it — and where a finding claims *less* than its own mechanical
   verdict, that disagreement is stated in the finding (see
   [b37](disjoint-worlds/FINDING_b37_legibility_matrix_2026_08_04.md), where two gates were
   impeached by their own author).
4. `*.certificate.json` / `*.seal.json` — OATH certification and the full trust-stack seal.

**Negatives are first-class here.** INVALIDs, retractions, and errata are kept in place with
their lineage rather than quietly fixed — including preregistration errors our own verification
caught ([b34-v3](disjoint-worlds/FINDING_b34v3_labelfree_read_2026_08_03.md)) and gate designs
that turned out to be noise-passable ([b37](disjoint-worlds/FINDING_b37_legibility_matrix_2026_08_04.md)).

*Nothing crosses unseen.*
