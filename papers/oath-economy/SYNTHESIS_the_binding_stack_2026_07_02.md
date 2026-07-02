# SYNTHESIS — the binding stack: what this lab has actually been studying

**Fathom Lab · 2026-07-02 · written the night rungs 1+3 shipped. A reframe, not a result; every number below
carries its receipt in the two FINDINGs and `rebind_result.json`.**

## 1. The pattern under every finding

Line up what this lab keeps discovering, across two months of preregs and self-falsifications:

- a truth **probe** with 0.98 in-construct AUC that reads a surface artifact (probe-orthogonality);
- guardrail **instruments** whose scores ride length, and a deception channel that reads *content*, not
  conduct (0.99 on a correct textbook paragraph);
- **benchmark corpora** that manufacture the confound they claim to control (ground-truth artifact);
- an **experiment** (keystone v1) whose depth metric measured formatting tokens while its grader scored right
  answers wrong;
- **model cards** carrying 201 numbers, none bound to a receipt (rung 1) — and, when re-bound by a third
  party, a config that disagrees with the vendor's own receipts and a value that matches nothing (rung 3);
- and this lab's **own matcher**, which produced three phantom mismatches that self-audit killed hours before
  they became a false public accusation.

These are not six findings. They are ONE finding, six times, at six altitudes: **a claim about a measurement
outrunning what was actually measured.** The instruments, the certificates, OATH, the rigor gate, the prereg
culture — they were never separate products. They are one product: **the binder** — the thing that ties a
claim back to the measurement that allegedly produced it, and refuses the claim when the tie fails.

## 2. Three gaps, all mechanically measurable today (opened by tonight's data)

Rung 3's dataset work surfaced a triple structure that generalizes far beyond one card:

| gap | question | tonight's measurement |
|---|---|---|
| **BINDING** | is the claim *linked* to its receipt? | 0/201 on-card (rung 1) — while the receipts existed, unlinked |
| **FIDELITY** | does the claim *agree* with its receipt? | 18/22 exact (Δ=0.0); 1 config-conflict (MATH 0-shot vs 4-shot); 1 orphan value (BFCL 25.7 matches nothing) |
| **SELECTION** | what *fraction* of what was measured got published — and was the choice flattering? | **22 of 1,246 metric rows = 1.8%** published. Spot-checks cut both ways: MGSM published macro-avg 24.5, NOT the flattering English 42.0 — honest selection, now *measurable* as such |

The third is the one nobody has ever quantified: vendors' receipt files reveal not just whether published
numbers are true, but **what was measured and left unsaid**. Cherry-picking stops being an accusation and
becomes a statistic — in either direction. (Tonight's n=2 spot-check found *no* cherry-picking by Meta; that
is reported with the same care as a hit would be.)

## 3. The census — the next rung this synthesis nominates (awaiting ratification)

Meta ships `-evals` receipt datasets for the **entire Llama family** (3.x, all sizes). The rung-3 scripts are
mechanical and already committed. Run them across every card+receipts pair that exists (any vendor):

- the first **claim-fidelity and selection census** of the industry — per-vendor, per-generation binding /
  fidelity / selection rates, pre-registered bars, no accusation framing;
- longitudinal: does binding improve or decay across generations? — that is **M10 (archaeology) applied to
  claims instead of conduct**, and it seeds **M7 (weather)**: pre-register the *next* release's binding
  posture before launch day;
- the tool that falls out is rung 2 grown up: **`styxx bind <card-url>`** — extract claims, locate receipts,
  emit the binding report. The EU-AI-Act documentation-provenance angle (Art. 15 bridge, already in
  papers/compliance) makes that report a compliance artifact, not a blog post.

Cost: CPU + public data. Risk: it is vendor-facing at scale — the framing rule (gap = industry property, no
accusation language, invite correction) must be frozen in the census prereg, and the phantom-mismatch hazard
list from rung 3 §3 is mandatory reading. **This does not fire without flobi's explicit go.**

## 4. Where the keystone fits (the full stack)

If the keystone's verdict lands positive, binding extends *inside* the model, and the stack closes:

```
ecosystem   cards <-> receipt datasets          (rungs 1+3, the census)
document    claims <-> committed receipts       (OATH, audit_claims, the rigor gate)
behavior    says <-> believes                   (grounded honesty, sampling divergence)
computation says <-> computes                   (attribution depth — the keystone, in flight)
```

One subject at four altitudes: the say-do gap. "Nothing crosses unseen" was never a tagline about detectors —
it is the claim that every layer of this stack can be bound, and the moonshots stop being ten dreams:
M1/M3/M7/M10 are the same census at different timescales; M2/M9 are its passport and its border; M5 is the
institution that runs it on itself first.

## 5. Honest counterweights

- n = one card, one vendor, 22 in-scope claims. The census is what turns tonight into an industry statement;
  until then, no industry statement.
- The selection-gap spot-check (n=2) found honest selection — the question is open, not a gotcha.
- Everything vendor-facing here waits for the second independent pass rung 1 already requires, and for the
  census prereg to be ratified.

*The bigger picture was under the floorboards the whole time: we are not building detectors. We are building
the binder — and tonight it bound someone else's claims for the first time.*
