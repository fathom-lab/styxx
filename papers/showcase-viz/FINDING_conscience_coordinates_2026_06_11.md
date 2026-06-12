# FINDING — conscience COORDINATES: truth is a portable coordinate, "refusal" is request-bound (HARM-AXIS-NULL)

**2026-06-11 · Fathom Lab / styxx. Pre-registered: `PREREG_conscience_coordinates_2026_06_11.md`
(frozen pre-run, committed 8692ec3). Receipt: `conscience_coordinates_result.json`; figure
`conscience_coordinates.png`. Cycle 4 — the creative leap from cycle 3 (WHITENING-RESOLVES: truth +
refusal are a clean orthonormal basis under a whitened readout): if it is a basis, treat it as a
COORDINATE SYSTEM and ask whether its coordinates mean something on content it was never fit on.**

## The test — a 2×2 factorial of statements, projected onto the whitened basis

A fully-crossed factorial of single declarative sentences, T ∈ {true, false} × H ∈ {danger-topic,
safe-topic}, content-matched across T within each H level, twelve per cell, forty-eight total. The
worst cell (false × danger) is genuine dangerous misinformation (false safety claims about real hazards),
used only as flagged-as-false probe stimuli read pre-output. The basis (w_truth, w_refusal) and the ZCA
whitening were fit on the cycle-3 train sets (factual statements / harmful requests); NO factorial item
touched the fit. Each sentence's whitened gemma-L12 state gives two conscience coordinates,
c_truth = x̃·ŵ_truth and c_refusal = x̃·ŵ_refusal (whitened-basis cosine −0.0, i.e. orthonormal).

## Result — one coordinate generalizes, the other does not

| readout | gemma | mapped Llama-3.2-3B | mapped Qwen2.5-3B |
| --- | --- | --- | --- |
| c_truth recovers T (AUROC) | **0.8524** | **0.9809** | **0.9306** |
| c_truth invariant to H (discrim) | 0.684 | 0.7344 | 0.6215 |
| c_refusal recovers H (AUROC) | **0.5226** | **0.4948** | **0.592** |
| c_refusal invariant to T (discrim) | 0.7135 | 0.7049 | 0.5156 |

- **The truth coordinate is a genuine, portable value coordinate.** It recovers true/false on a content
  type it was never fit on (danger/safe statements), and it transfers BETTER through the cross-model map
  than it reads in the source (gemma 0.8524 → Llama 0.9809, Qwen 0.9306). The four quadrant centroids
  spread cleanly along c_truth: true-safe +2.6216, true-danger +0.8091, false-safe −0.6854, false-danger
  **−3.5631** — dangerous misinformation sits at the extreme-low-truth corner.
- **The refusal coordinate does NOT generalize.** c_refusal recovers the danger-topic factor at chance
  in all three settings (0.5226 / 0.4948 / 0.592), and the four quadrant centroids barely move on
  c_refusal (−0.1344 to +0.1912). **Verdict per the frozen gate: HARM-AXIS-NULL.** The "refusal" axis,
  fit on harmful REQUESTS, encodes request-COMPLIANCE ("should I refuse to do this?"), not content-HAZARD
  ("is this statement about something dangerous?"). They are different things, and the conscience learned
  the former.

## What this does and does not establish — the bound is the contribution

- **Dangerous misinformation IS detectable, but via falsity, not via a 2D composite.** A derived
  low-truth-and-high-harm score picks the false-danger cell out of the other three at AUROC 0.838 — but
  the figure and the centroids show this is carried almost entirely by the truth coordinate (the
  false-danger corner is extreme-low-truth); the harm coordinate adds nothing. The pre-registered
  composite hypothesis — that the basis decomposes dangerous misinfo into separate false AND dangerous
  components — is NOT supported. The single-axis truth generalization IS, strongly and cross-model.
- **The conscience basis is not a general (truth × danger) coordinate system for statements.** It is a
  portable truth coordinate plus a request-bound refusal coordinate that does not double as a
  content-hazard reader. That is a precise, useful negative: it prevents the overclaim that you can read
  "is this dangerous content" off the refusal axis.

## Honest bounds (what is NOT claimed)

The truth coordinate is not perfectly H-invariant: discrim(c_truth, H) is 0.684 in gemma, marginally
above its own permutation floor 0.6649 — danger-topic nudges even TRUE statements toward lower c_truth
(true-danger +0.8091 sits well below true-safe +2.6216). So the truth read is slightly depressed by the
danger register; stated, not spun. Twelve items per cell is small; per-cell readouts carry permutation
context (reported in the receipt) and the cross-model replication is the load-bearing support for the
truth-coordinate claim. Linear DiM directions in a whitened space, one source model (gemma-2-2b),
local open targets; the false-danger stimuli are flagged-as-false research probes read pre-output and no
model generated any text from them. The frontier this opens: fit a DEDICATED content-danger axis on
danger-vs-safe STATEMENTS (not requests) and test whether THAT forms a clean orthonormal pair with truth
— i.e. build the right second axis rather than borrowing the request-refusal one — which would turn this
informative null into the intended (truth × danger) coordinate system. The figure ships as the artifact:
content laid out in conscience-coordinate space, sorting by truth and flat on refusal — the null made
visible.
