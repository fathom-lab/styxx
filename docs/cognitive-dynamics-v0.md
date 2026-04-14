# Cognitive Dynamics v0.1 — A Primer

**Status:** draft  
**Version:** 0.1  
**Module:** `styxx.dynamics` (3.1.0a1+)  
**Authors:** flobi (fathom lab)  

---

## 1. The thesis

The field of large language models treats inference as an **open-loop**
process. A prompt goes in, a generation comes out, and there is no
measurable state variable that an external agent can use to predict,
control, or counterfactually reason about what is happening inside.

This is not because LLMs are inherently unobservable. It is because
nobody has had a calibrated, cross-architecture, real-time readout of
cognitive state. Every other interpretability technique — SAE
features, attention patterns, residual stream probes, embedding
similarity — is **model-specific** and dies the moment you swap the
model.

Fathom changed that with the atlas v0.3 calibration: a 6-dimensional
projection of cognitive state into a substrate-independent eigenvalue
space, validated cross-model on 12 open-weight models from 3
architecture families. The styxx 3.0.0a1 release lifted that
projection into a portable data type — the `Thought`.

Once you have a measurable state vector, you can ask the next
question: **how does the state evolve over time?** And once you can
predict that evolution, you can:

1. simulate cognitive trajectories offline at zero API cost
2. build closed-loop controllers that steer cognition toward targets
3. reason counterfactually about what would have happened
4. test the hypothesis that the eigenvalues are **causal**, not
   merely correlative

This document specifies the v0.1 cognitive dynamics model: the first
dynamical-systems model of LLM cognition, fitted by ordinary least
squares on observation tuples in fathom's eigenvalue space.

---

## 2. The model

Let $s_t \in \mathbb{R}^6$ denote the cognitive state at step $t$, encoded
as the time-mean probability vector across populated atlas phases:
$$s_t = \text{mean\_probs}(\text{Thought}_t)$$

Let $a_t \in \mathbb{R}^6$ denote the action at step $t$ — the cognitive
direction the agent attempted to push toward (also a `Thought`,
extracted via the same encoding).

The cognitive dynamics model is the linear-Gaussian update:

$$s_{t+1} = A \cdot s_t  +  B \cdot a_t  +  \varepsilon_t$$

where:

- $A \in \mathbb{R}^{6\times 6}$ is the **natural drift matrix**. It captures
  how cognitive state evolves between steps with no intervention.
  The diagonal of $A$ encodes per-category persistence (cognitive
  momentum); off-diagonal entries encode cross-category coupling
  (does heavy reasoning suppress creative? does refusal pull state
  toward adversarial?). The spectral radius of $A$ governs system
  stability: if $\rho(A) < 1$ the natural-drift trajectory converges
  to a fixed point; if $\rho(A) \geq 1$ it can diverge.

- $B \in \mathbb{R}^{6\times 6}$ is the **action transfer matrix**. It
  encodes how a unit-magnitude push in each category direction
  *actually* moves the state. Agents often want X and get Y — the
  matrix $B$ is exactly that gap. A perfect controller has $B = I$
  (intended pushes equal observed movement); real agents have
  off-diagonal $B$ entries that say "trying to push toward
  reasoning also accidentally amplifies retrieval."

- $\varepsilon_t \sim \mathcal{N}(0, \Sigma)$ is gaussian residual noise
  capturing variance the linear model cannot explain.

This is the v0.1 parameterization. Future versions can lift to:

- **higher-dim state**: 24-d (4 phases × 6 categories) instead of
  the time-collapsed 6-d
- **non-linear dynamics**: kernelized regression, neural networks,
  Koopman operators
- **multi-step models**: $s_{t+1} = f(s_t, a_t, s_{t-1}, a_{t-1}, \dots)$
- **action embeddings**: continuous action representations beyond
  Thought-space

But the v0.1 linear-gaussian model is the floor: simple, fittable
in closed form, mathematically falsifiable, and good enough to be
useful.

---

## 3. Fitting

Given $N$ observation tuples $\{(s_t^{(i)}, a_t^{(i)}, s_{t+1}^{(i)})\}_{i=1}^N$,
fit $A$ and $B$ by ordinary least squares:

$$\min_{A, B} \sum_{i=1}^{N} \left\| s_{t+1}^{(i)} - A \cdot s_t^{(i)} - B \cdot a_t^{(i)} \right\|^2$$

In matrix form, stack the inputs:

$$X = [S \mid \mathrm{Act}] \in \mathbb{R}^{N \times 12}, \quad Y = S_{\text{next}} \in \mathbb{R}^{N \times 6}$$

and solve $X W = Y$ for $W \in \mathbb{R}^{12 \times 6}$ via
`np.linalg.lstsq`. The recovered $W$ contains $[A^T; B^T]$ stacked
vertically; transpose to get the matrices back.

This is closed-form, $O(N)$, and runs in milliseconds.

### 3.1 Identifiability

The recovery is **fully identified** when the regressor matrix $X$ has
rank 12 — i.e. the state and action samples span the full
$\mathbb{R}^6$ space. With full-rank inputs (e.g. gaussian samples), the
fit recovers $A$ and $B$ to **machine epsilon** in the noise-free
limit.

The recovery is **identified up to an equivalence class** when state
and action are constrained to the probability simplex (each sums to 1).
A 6-dimensional simplex is a 5-dimensional affine subspace, so the
stacked regressor matrix has rank $5 + 5 = 10$, not 12. The
least-squares solution still produces predictions that are exact on
the training set — meaning $\hat{A} \cdot s + \hat{B} \cdot a = s_{t+1}$
for every training tuple — but $\hat A$ and $\hat B$ are not unique.
Many distinct $(A, B)$ pairs lie in the same equivalence class.

This matters for **interpretation** of the matrices, not for
**prediction or control**. The styxx test suite verifies machine-
epsilon recovery on full-rank gaussian inputs, and verifies
prediction-set $R^2 = 1.0$ on rank-deficient simplex inputs. Both
behaviors are mathematically correct and both are documented.

### 3.2 Quality metrics

Every `FitResult` carries:

- `train_mse`: mean squared error on the training set
- `r2`: coefficient of determination, $1 - \frac{\text{SS}_{\text{res}}}{\text{SS}_{\text{tot}}}$
- `spectral_radius_A`: $\max_i |\lambda_i(A)|$, governs stability
- `train_max_err`: worst per-tuple L2 error
- `is_stable()`: True when $\rho(A) < 1$ (drift trajectories converge)

Use $r^2$ to decide whether the model is trustable for downstream
prediction. Use the spectral radius to decide whether long-horizon
forecasts are bounded.

---

## 4. Verbs

### `dyn.predict(state, action) → Thought`

One-step forecast. Computes $s_{t+1} = A \cdot s_t + B \cdot a_t$ and
returns the result as a `Thought` (with simplex projection at the
boundary for visualization).

### `dyn.simulate(initial, actions) → list[Thought]`

Multi-step rollout. Composes `predict` in a loop:

$$\text{trajectory} = [s_0, s_1, s_2, \dots, s_n] \quad \text{where} \quad s_{i} = A \cdot s_{i-1} + B \cdot a_{i-1}$$

No real model calls. Fully offline. Use this to:

- prototype agent prompt strategies before paying for real LLM calls
- generate synthetic training data for downstream learning
- run thought experiments: "what if my agent saw this kind of input?"

### `dyn.suggest(current, target) → Thought`

Model-predictive controller. Solves

$$B \cdot a^* = (\text{target} - A \cdot \text{current})$$

for $a^*$ via least squares. The result is the action that minimizes
the L2 distance from $\text{predict}(\text{current}, a)$ to $\text{target}$ in
the linear model.

For multi-step planning, run `suggest` in a closed loop: apply the
suggested action, observe the actual next state, re-suggest from
there. Convergence is governed by the spectral radius of the
closed-loop dynamics, $A - B B^+ A$.

### `dyn.forecast_horizon(initial, n_steps) → list[Thought]`

The natural drift trajectory. Equivalent to `simulate(initial,
[zero_action]*n_steps)`. If $\rho(A) < 1$ this trajectory converges
to a fixed point in eigenvalue space; otherwise it diverges. The
fixed point is the **cognitive equilibrium** the agent drifts toward
when nothing is pushing it.

### `dyn.residual(observation) → float`

L2 distance between the model's prediction and the actual observed
next state. Used to measure fit quality on held-out data.

### `dyn.save(path)` / `CognitiveDynamics.load(path)`

Serialize a fitted model to a `.cogdyn` file (canonical sort-keys
UTF-8 JSON, no BOM). The format stores `A`, `B`, the schema, the
fit metadata, and a UUID identifying the model.

---

## 5. Why this matters

### 5.1 The closed-loop unlock

Most of "AI safety in production" today consists of **post-hoc**
interventions: filter the output, retry on failure, re-prompt with
guardrails. None of these can prevent failures at generation time
because there's no feedback signal to use *during* generation.

A cognitive dynamics model is the missing piece. Once you can predict
that a given action in the current state will produce a hallucination
attractor, you can choose a different action. Once you can simulate
forward 5 steps, you can pick the action that minimizes long-horizon
cognitive risk. That's closed-loop control over cognition, applied
to the inference-time API surface every developer already uses.

### 5.2 The simulation unlock

LLM inference costs money. Iterating on agent design today means
making thousands of real API calls and watching them fail. With a
fitted dynamics model, you can run thousands of synthetic rollouts
**offline** at zero marginal cost. Tune your prompt strategies in
simulation. Find the cognitive failure modes before they hit
production. Stress-test your agent against hostile cognitive
trajectories before shipping.

### 5.3 The causality unlock

The deepest implication: if you can fit a dynamics model on
observations from one set of models and that model **predicts**
behavior on a *different* set of models that weren't in the training
data, you have evidence that the eigenvalues are not just
correlations — they're **causal handles** on cognition. That is a
testable hypothesis in the v0 framework. The infrastructure for the
test is exactly what shipped in this release.

### 5.4 The counterfactual unlock

With a learned dynamics model you can ask: "what would my agent have
done if it had been in a different cognitive state?" That's the
foundation of:

- post-incident forensics ("the model went off the rails — what
  cognitive state was it in, and what would have happened in
  another state?")
- robustness analysis ("how sensitive is the agent's behavior to
  small perturbations in cognitive trajectory?")
- training data filtering ("which training examples shifted the
  cognitive baseline of my model?")

---

## 6. Limitations

### 6.1 Linear is not sufficient

Real LLM cognition is not linear. The v0 model captures the
first-order structure: persistence, cross-category coupling, action
transfer. It misses non-linear interactions, conditional dynamics
("the response to this prompt depends on the recent history"), and
the long-tail of cognitive failure modes that don't follow a smooth
manifold. v0 should be treated as the **floor** of what's possible,
not the ceiling.

### 6.2 Six dimensions is small

Compressing cognitive state to 6 dimensions throws away a lot of
information. The full atlas v0.3 representation has 4 phases × 6
categories = 24 dimensions; v0 collapses to the time-mean for the
sake of the simplest possible math. v0.2 will lift to the full 24-d
representation.

### 6.3 Action representation is naive

Encoding an action as a `Thought` (a target push direction) is
intuitive but lossy. Real "actions" in agent contexts are richer:
prompt prefixes, system messages, sampling adjustments, retrieval
context. v0 treats all of these as a single 6-d push direction.
v0.2 will support continuous action embeddings.

### 6.4 No data, yet

The styxx 3.1.0a1 release ships the math but no large-scale fitted
model — there is no hosted "cognitive dynamics" you can download.
The expectation is that users will fit their own from observations
collected from their agent fleet. Fathom Lab will ship a calibrated
v0.1 model in a follow-up release once we have a sufficient corpus
of cross-model observation tuples.

---

## 7. Reference example

```python
import styxx
import numpy as np
from styxx.dynamics import CognitiveDynamics, Observation

# Step 1: collect observations from your fleet
obs = []
for episode in your_agent_episodes:
    for t in range(len(episode) - 1):
        s_t   = episode[t].vitals.to_thought()
        s_t1  = episode[t+1].vitals.to_thought()
        a_t   = your_action_encoder(episode[t].action)
        obs.append(Observation.from_thoughts(s_t, a_t, s_t1))

# Step 2: fit
dyn = CognitiveDynamics()
result = dyn.fit(obs)
print(f"fit: {result}")
print(f"  natural drift spectral radius: {result.spectral_radius_A:.3f}")
print(f"  is stable: {result.is_stable()}")
print(f"  R^2: {result.r2:.4f}")

# Step 3: predict
current_thought = styxx.observe(latest_response).to_thought()
target_thought  = styxx.Thought.target("reasoning", confidence=0.85)
predicted_next  = dyn.predict(current_thought, target_thought)
print(f"predicted next state: {predicted_next}")

# Step 4: control
optimal_action = dyn.suggest(current_thought, target_thought)
# Encode optimal_action back to an actual prompt prefix and apply

# Step 5: simulate offline
trajectory = dyn.simulate(
    initial=current_thought,
    actions=[styxx.Thought.target("reasoning")] * 10,
)
for i, t in enumerate(trajectory):
    print(f"step {i}: {t}")

# Step 6: save / load the fitted model
dyn.save("my_agent.cogdyn")
loaded = CognitiveDynamics.load("my_agent.cogdyn")
```

---

## 8. License and patents

- The cognitive dynamics SPECIFICATION is released under
  **CC-BY-4.0**.
- The reference implementation (`styxx.dynamics`) is released under
  **MIT** as part of the styxx package.
- The underlying cognitive measurement methodology that produces
  the eigenvalues is covered by US Provisional Patents 64/020,489,
  64/021,113, and 64/026,964. See `PATENTS.md` in the styxx
  repository for details.

---

## 9. Status

This is v0.1. The math is verified to machine precision on
full-rank synthetic data (44 tests in `tests/test_dynamics.py`,
all passing). Real-world fits await fleet-scale observation data
collection.

The goal of v0.1 is to ship the **infrastructure** — the data
type, the fit/predict/control verbs, the file format, the spec —
so that the community can collect data and build on it.

---

*nothing crosses unseen.*  
*— fathom lab*
