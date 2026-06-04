# NEGATIVE — a linear code-geometry model does NOT reproduce the resonance

**2026-06-04.** Ambitious swing: predict the frequency resonance *and* the scaling sweep's verdict
analytically, from the linear geometry of the phase code — no training, no GPU. Each item is a
phasor code vector v(a) = [ r_m^a · e^{iθa} ]_m; capacity = separability of the K code vectors.
`analytic_capacity.py`.

**Two metrics, both fail the sanity gate** (reproduce the empirical interior resonance at D=0):

1. **log-det of the normalized Gram** (linear independence) — flat in θ. With M=256 modes ≫ K≈6
   items, any few vectors are near-orthogonal in 256-D; the metric saturates at "separable."
2. **effective rank above a noise floor** (Σ sᵢ²/(sᵢ²+σ²), un-normalized to keep decay/SNR) — also
   flat (~1.5 dims for *all* θ). The decay-magnitude structure dominates the singular spectrum;
   phase rotates singular *vectors* but barely moves the singular *values*.

The script's **built-in sanity gate refused to emit a scaling prediction** once the resonance check
failed — otherwise it would have reported a noise-driven "NULL/item-count" verdict (it did, on a
first pass, before the gate; that's exactly the failure mode the gate exists to stop).

**Conclusion.** The resonance is **not** a property of the linear code geometry — it requires the
**trained nonlinear readout** (2-layer MLP decoder + optimization). Linear separability of the raw
code is necessary but not the binding constraint; the across-θ capacity differences emerge from how
*learnable/robust* the nonlinear decode is, which a closed-form linear model doesn't capture.

**Implications.**
- No analytical shortcut to predict the scaling sweep — the empirical run stays the arbiter, and
  `THEORY_phase_coding` §3's two-budget fork (item-count vs window) remains an *open empirical*
  question, not something the linear model could settle.
- The **qualitative** phase-coding mechanism (THEORY §2: collapse at θ=0 and at Nyquist, interior
  optimum) still stands as the *explanation*. Only the quantitative linear-*predictive* shortcut
  failed.
- **Don't re-attempt the naive linear version.** A working analytical model would have to include a
  trained/matched nonlinear decoder — non-trivial, lower priority than the empirical program.

This is a documented dead-end (cf. the geometry probe). The value: the negative is honest, and the
sanity gate caught noise before it could be reported as a prediction.
