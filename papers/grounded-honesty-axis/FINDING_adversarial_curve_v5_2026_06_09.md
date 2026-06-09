# FINDING — v5: the positive control resolves the v2–v5 arc (the re-fittable-probe frame was the wrong test)

**2026-06-09. Fathom Lab / styxx.** v5 ran the positive-control-gated read≠write test (`PREREG_adversarial_curve_v5`).
The gate returned **VOID** (positive control did not pass), and a 3-lens red-team then turned on the *intended*
write-up and scoped it — caught the third framing-overclaim in 24 hours. The honestly-scoped result is **more
valuable than a VOID-as-dead-end**: it explains *why* v2→v5 kept returning VOID/inconclusive, and it ends the
chase. **The re-fittable-probe frame for read≠write is near-tautological; the operationally meaningful question is
the deployed frozen probe, which v2 already answered (defeatable).**

## What the positive control actually showed

Plant a causally-inert sentinel `[tag:s]`; build a frozen rank-4 class-mean subspace `Bplant`; run the scrub at
lam=128 to collapse the planted read. Result: **the scrub nulled its target subspace ~97% (energy 0.128→0.003)
— it provably engaged — yet a fresh same-model refit probe still read the sentinel at 0.94.** Direct white-box
probing (red-team) confirmed: at any *static* snapshot, *all* the linear sentinel signal lives in `Bplant`
(project it out → read collapses to ~0.05). So the 0.94 survival is **not** an un-scrubbable raw-token pathway —
it is the LoRA weight-edit **relocating** the signal into a fresh direction orthogonal to the frozen subspace,
which the refit probe then finds. **Real read≠write-style relocation, the same mechanism as v3 (trace moved
off-band to L27) and v4 (reads rose with pressure).**

## The defensible core (keep)

**A fresh all-layer refit probe re-finds any linearly-present, knowledge-coupled signal wherever subspace-nulling
relocates it.** The only way to make that refit read fall is to *remove* the information — i.e. **unlearn it,
which also removes the knowledge.** That is the **read=write boundary itself**, not a separate result. So:
- **No knowledge-preserving subspace-nulling attack can produce EVADABLE on a re-fittable probe** for a
  knowledge-coupled target — confirmed mechanistically across v3 (relocation off-band), v4 (reads rose),
  v5 (PC relocation despite 97% target-nulling).
- Against an adaptive *re-fitting* auditor, **info-present ⟺ knowledge-held ⟺ fresh-probe-reads-it**, so
  "read≠write on a re-fittable probe" collapses toward "true unlearning" — which is the boundary. The
  re-fittable read is **robust by construction**, not by a contingent experimental win.

## The corrections (the red-team scoped the overclaim)

1. **The `VOID-INSTRUMENT-DEAD` / "floor unreachable" label is imprecise.** The scrub *engaged* (energy→0.003);
   the floor is unreachable only for the **refit** probe — which *is* the finding (relocation), not a dead instrument.
2. **A carried-over bug:** `PC_KNOWLEDGE_HELD` used the *perfect base anchor* (K=1.0, need ≥0.95) — the exact
   anchor bug v4 told v5 to fix — and cave-FT alone drops K to ~0.885, so the PC could **never pass on knowledge
   regardless of the read** (partly vacuous, same failure mode as v4's structurally-unreachable ROBUST).
3. **The universal claim is scoped.** "Never EVADABLE / near-tautological" is sound for **deterministically
   re-derivable / in-knowledge targets** (the verbatim sentinel is *always* in-context). The v5 gold sweep reads
   on *behaviorally-suppressed* items (K_attack false) where gold is **not** deterministically re-derivable — so
   the sentinel (easiest target) does **not** strictly license the universal claim for the harder suppressed-gold
   regime. Honest scope: *"subspace-nulling cannot collapse a re-fittable read whose target is re-derivable from
   surviving context."*

## The pivot (this ends the chase)

The read≠write story, stated at true strength:
- **Re-fittable probe:** near-tautologically robust (collapsing it = unlearning = the boundary). **Stop attacking it.**
- **Deployed frozen probe:** the operationally meaningful question — and **v2 already answered it: DEFEATABLE by
  displacement** (n=1, needs replication). *This* is where the real read≠write risk lives.
- **The boundary:** true unlearning removes the trace *and* the knowledge together — that is read=write, by definition.

So v2→v5 were, in part, **chasing a verdict the re-fittable frame cannot yield.** v5's value is naming that. The
program's adversarial energy should move **off** the re-fittable read and onto: (a) **replicating v2's
frozen-probe defeat** (n=1 → robust), and (b) the **B3 deployment trapdoor** (can a model read clean on the audit
set but blind in deployment — the *deployed*-probe attack that actually matters).

## Owed (to fully close the scope)

(1) Re-fit `Bplant` on the **scrubbed** model and project it out — confirm the signal merely relocated into a new
refittable direction (vs. an un-scrubbable pathway). (2) A **non-deterministic** positive control (tag = a coin
flip not derivable from the question) to test the over-generalization at the suppressed-gold boundary.
(3) Fix the PC knowledge anchor to the cave-FT baseline. Scope loud: Qwen-1.5B, sycophancy-caving, LoRA, this
attack family. Receipts: `v5_readwrite_result.json` (sha `ed553a41`), `run_v5_readwrite.py`,
`PREREG_adversarial_curve_v5`, red-team `wf_290ebd72`.
