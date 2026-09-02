# RESULT — the confound in the knob, resolved: rotation is load-bearing beyond timescale diversity — 2026-09-02

Fathom Lab · 2026-09-02 · Frozen by `PREREG_untied_magnitudes_2026_09_02.md`, committed before the
run. Runner: `run_untied_control.py`, importing `run_rhythm_rescue.py` verbatim. Receipt:
`untied_control_result.json`, scored through `styxx.protocol`. Device: CPU, three seeds, 4000 steps,
D=256. Every number below is sworn to the receipt at commit `3f4a03eac7a4`.
<sworn r="path:papers/rhythm-rescue/untied_control_result.json#/verdict" k="quote">The frozen verdict reads `ROTATION_LOAD_BEARING__beyond_diversity`.</sworn>

## The confound, and the arm

The phase clamp does two things at once: it removes rotation, and it ties each complex mode's two
real channels to one magnitude. REAL2 — a real-eigenvalue bank with 2D modes and 2D independent
magnitudes, no rotation — has exactly FREE's state size and parameter count
(<sworn r="path:papers/rhythm-rescue/untied_control_result.json#/arms/real2/params" k="numeric">168524 parameters</sworn> against
<sworn r="path:papers/rhythm-rescue/untied_control_result.json#/arms/free/params" k="numeric">FREE's 168524</sworn>), and strictly more timescale diversity than either
existing arm. If the clamp's loss had been diversity, REAL2 would recover it.

## The gates

<sworn r="path:papers/rhythm-rescue/untied_control_result.json#/metrics/anchor_max_abs_dev" k="numeric">The anchors landed 0.0 items from the committed GPU receipt</sworn>:
<sworn r="path:papers/rhythm-rescue/untied_control_result.json#/arms/free/kcap_mean" k="numeric">FREE's ordered-memory capacity was 6.0 items</sworn> and
<sworn r="path:papers/rhythm-rescue/untied_control_result.json#/arms/clamped/kcap_mean" k="numeric">CLAMPED's was 2.6667</sworn>, seed for seed the numbers of 2026-06-03,
on a different device. <sworn r="path:papers/rhythm-rescue/untied_control_result.json#/metrics/gap_free_minus_clamped" k="numeric">The gap was 3.3333 items</sworn>, over the
frozen 2.0. <sworn r="path:papers/rhythm-rescue/untied_control_result.json#/arms/real2/kcap_mean" k="numeric">the untied bank's capacity was 2.6667 items</sworn> —
<sworn r="path:papers/rhythm-rescue/untied_control_result.json#/metrics/real2_minus_clamped" k="numeric">0.0 above CLAMPED</sworn>, and
<sworn r="path:papers/rhythm-rescue/untied_control_result.json#/metrics/free_minus_real2" k="numeric">3.3333 below FREE</sworn>.
<sworn r="path:papers/rhythm-rescue/untied_control_result.json#/metrics/recovery_fraction" k="numeric">The recovery fraction was 0.0</sworn>.

## The finding

Untying the magnitudes recovers nothing. A real bank with twice FREE's independent timescales
holds ordered items exactly as poorly as the tied bank, at every K: the two curves lie on each
other. Whatever the rotation buys, it is not a richer spread of decay rates — it is the rotation.
The arc's headline, that oscillation is causally load-bearing in state-space models, survives
the sharpest control its own knob admits, and the theta-gamma reading — phase as the code for
order — is strengthened rather than scoped.

## What this does not say

One task family (ordered copy), one width, three seeds, no interval, toy scale. The flagship
permuted-MNIST number (+0.312) was measured with the same knob and carries the same confound; this
result makes the diversity reading unlikely there but does not test it, and the permuted-MNIST
re-run with a REAL2 arm is the next preregistration. Nothing here is about real LinOSS or Mamba
checkpoints. The disclosed prior was uncertain and the bet was real; it came out on the side of
rotation, and the numbers say so at the resolution of a capacity grid.

---

*The knob turned two things; the second turned out to be nothing. The rhythm was the rhythm.*
