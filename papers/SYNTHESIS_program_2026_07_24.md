# STYXX program synthesis — instruments that refuse

**Date:** 2026-07-24. Unifies the work running in parallel across sessions (interactive + the
`styxx-autopilot-cycle` scheduled task) into one statement of what this program is and what it has
earned the right to claim.

## The one-sentence identity

STYXX builds **measurement apparatus for AI evaluation that can VOID but never bless** — instruments
that ship their own weaknesses on the label, and mechanisms isolated by one causal knob at a time.
Everything below is either a shipped instrument with measured operating characteristics, or a causal
result with a frozen gate that could have killed it.

## Two arcs, one method

### Arc A — validity instruments (published, shipped, in use)

The measurement layer for LLM-as-judge evaluation, where most of modern AI evaluation now rests.

- **The kill.** The industry validates judge panels with blatant gold checks. Across four constructed
  families on a real correlated panel, gold anchors licensed nothing: label-free coverage 0/15 in every
  family while the panel was flawless on the gold items — and the failure is silent when the violation
  is smooth.
- **The repair.** Anchors drawn from the same generator as the work ("ladder" anchors) either price each
  judge's error or REFUSE when no judge is informative. Never certifies garbage.
- **In the wild.** On a real heterogeneous panel (Gemini + Qwen judges on TruthfulQA), gold-anchor
  validity is prevalence-dependent; ladder anchors cover at every prevalence.
- **The anchor threshold (design-time).** One known-negative the whole panel calls "correct" is a 150.8x
  likelihood ratio for a shared blind spot; roughly 15 give 90% power. Shipped as
  `styxx.anchors.blindspot_power` / `min_anchors_for_power` / `anchor_lr` in styxx 7.27.0.
- **Real-judge demonstration.** A homogeneous weak panel is unanimously wrong on 0.64 of imitative
  falsehoods; the tool's predicted budget of 3 anchors detects it at 0.959 empirical power. Heterogeneous
  and stronger panels show no blind spot, and the instrument says so.
- **The operating boundary (autopilot cycle 61, verified here).** Sweeping one informative judge among
  deaf ones across the noise-margin gate, `audit_panel` prices where covered and refuses below the gate —
  `SURVIVED__prices_where_covered_refuses_below_gate`, certificate OATH-HELD with 62 verified and zero
  ungrounded.

Together these are not five findings; they are **one datasheet**: what the instrument reads, when it
refuses, how many anchors you need before "no detection" means anything, and where its boundary sits.
That completeness — refusal semantics plus measured characteristics plus a design-time budget — is the
part no other eval tooling ships.

### Arc B — mechanism instruments (in progress, controlled, honestly scoped)

The same method turned on computation itself: isolate a mechanism with one knob and report what it
causally carries.

- **Oscillation is the long-range mechanism** in state-space models (phase-clamp ablation; permuted-MNIST
  decay gap +31.2 points).
- **Long-range consistency-checking requires it.** Decay compares adjacent facts perfectly but collapses
  to chance when the fact is distant.
- **It has a measurable horizon with a mechanism.** Decay's probability of solving falls with distance
  (half-horizon near gap 32) and tracks the signal a magnitude-limited channel retains.
- **The dissociation.** Decay recalls a single fact at ANY distance — no horizon at all. Only comparison
  has one. So oscillation is a **relating** mechanism, not a memory one.
- **Open, running:** does the deficit scale with relational load while sparing storage load (a
  dose-response that can refute the relating reading)?

**Scope, stated as loudly as the results:** Arc B is controlled state-space-model work. It is NOT a
claim about real-LLM honesty — no language model is ablated, and transformers have no phase to clamp.
The tempting bridge ("real LLMs use RoPE, which is phase") was checked and REJECTED: attention is global,
so it has no decay horizon for phase to rescue. Arc B's mechanism lives in recurrent/SSM computation.

## Where the arcs actually meet

Both arcs converge on the same operation: **relating a claim to its grounding**.

- Arc A audits it from the outside: does a panel of judges catch a response that contradicts the truth,
  and can that judgment be licensed without labels?
- Arc B asks what computation makes it possible from the inside: relating two temporally-separated facts
  requires a channel that keeps them separable, which decay does not provide at range.

That is a real through-line, and it is also where the honesty must be loudest: Arc B is a
**precondition**, not evidence about deployed models. The bridge from "relating needs phase in an SSM" to
"honesty rides such a channel in an LLM" is an open hypothesis with a named next test (a trained
Mamba/LinOSS checkpoint under `resonance_profiler`).

## What makes this defensible rather than loud

- Every number in every result document is receipt-bound and machine-verified by the same certifier that
  polices the rest of the program.
- Frozen preregistrations with kill-gates; ABSTAIN and NULL are shipped, not buried. Recent examples on
  the record: a mean-accuracy horizon that ABSTAINED on bimodality, a recall-horizon prediction that was
  REFUTED (and produced the better dissociation), and an anchor-power table found to be conservative and
  corrected in the favourable direction.
- Instruments can only void, never bless. That constraint is what makes the tool trustworthy where an
  optimistic one would be worthless.

## The honest roadmap

1. **Arc A distribution (operator-gated):** arXiv submission of the flagship; announcement with the live
   DOI. The science and the package are done.
2. **Arc A depth:** fold the buried-judge boundary into the paper's datasheet section as a future version.
3. **Arc B falsification:** finish the relational-load dose-response; then the real-model rung — run the
   resonance profiler on a trained SSM checkpoint. That is the step that either opens the bridge to real
   models or closes it.
