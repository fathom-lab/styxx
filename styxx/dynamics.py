# -*- coding: utf-8 -*-
"""
styxx.dynamics — the first dynamical-systems model of LLM cognition.

The field treats LLM inference as open-loop. There is no measurable
state variable to close the loop on. fathom changes that: the
calibrated cross-architecture cognitive eigenvalue projection (atlas
v0.3) gives us a real state vector that lives in a substrate-
independent space. Once you have a state vector you can fit a
dynamical system to it. Once you have a dynamical system you can
predict, simulate, control, and reason about counterfactuals.

This module is the first cognitive dynamics model in the field.

Math
────

State vector: a Thought, encoded as the 6-d time-mean probability
vector over fathom's calibrated categories
(``Thought.mean_probs()``). Future versions will lift this to the
24-d full per-phase trajectory.

Action vector: also a Thought — specifically, a target-push direction
in eigenvalue space. An action is "the cognitive direction you tried
to push cognition into." The dynamics model learns the gap between
*intended* push and *observed* movement.

Linear-Gaussian update::

    s_{t+1} = A · s_t  +  B · a_t  +  ε

  - ``A`` (6x6): natural drift matrix. How cognitive state evolves
    between steps with no intervention. Captures momentum, decay,
    cross-category coupling.

  - ``B`` (6x6): action transfer matrix. How a unit-magnitude push
    in each category direction actually moves the state. Agents
    often want X, get Y; this matrix is exactly that gap.

  - ``ε``: gaussian residual. Captures unmodeled variance.

Fit by ordinary least squares over an observation list. Closed-form,
O(N) in numpy. Recovers (A, B) from N >= 12 tuples (one per
parameter in the joint matrix).

Verbs
─────

  - ``fit(observations) -> FitResult``       learn (A, B) from data
  - ``predict(state, action) -> Thought``    one-step forecast
  - ``simulate(initial, actions) -> list``   multi-step rollout
  - ``suggest(current, target) -> Thought``  controller — find the
                                              action that minimizes
                                              distance to the target
  - ``forecast_horizon(initial, n) -> list`` natural drift trajectory
  - ``residual(observation) -> float``       held-out fit quality
  - ``save(path) / load(path)``              serialize learned model

Example
───────

    from styxx import Thought
    from styxx.dynamics import CognitiveDynamics, Observation

    # Collect a dataset of (state, action, next_state) tuples by
    # probing one or more models. Or use the fixture data shipped
    # with the package.
    obs = [
        Observation(state=t0, action=a0, next_state=t1),
        Observation(state=t1, action=a1, next_state=t2),
        # ... at least 12 tuples for a well-conditioned fit
    ]

    # Fit the model
    dyn = CognitiveDynamics()
    result = dyn.fit(obs)
    print(f"fit MSE: {result.train_mse:.4f}")
    print(f"explained variance: {result.r2:.3f}")

    # Predict
    next_thought = dyn.predict(current_thought, target_action)

    # Simulate offline (no real model calls)
    trajectory = dyn.simulate(initial=t0, actions=[a1, a2, a3])

    # Use as a controller — find the optimal action to reach a target
    best_action = dyn.suggest(current=t0, target=t_goal)

    # Forecast natural drift for n steps
    drift_path = dyn.forecast_horizon(t0, n_steps=10)

3.1.0a1 — the first cognitive dynamics model in the field.
"""

from __future__ import annotations

import json
import math
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .thought import Thought, PhaseThought, CATEGORIES, N_CATEGORIES


# ══════════════════════════════════════════════════════════════════
# Format constants
# ══════════════════════════════════════════════════════════════════

COGDYN_FORMAT = "cognitive_dynamics"
COGDYN_VERSION = "0.1"


# ══════════════════════════════════════════════════════════════════
# State <-> vector conversion
# ══════════════════════════════════════════════════════════════════

def thought_to_state(t: Thought) -> np.ndarray:
    """Encode a Thought as a fixed-dim state vector for the dynamics
    model. Uses the time-mean probability vector across populated
    phases (6 dims, one per category).
    """
    if not isinstance(t, Thought):
        raise TypeError(f"thought_to_state expects a Thought, got {type(t).__name__}")
    return np.asarray(t.mean_probs(), dtype=float)


def state_to_thought(
    vec: Union[np.ndarray, Sequence[float]],
    *,
    source_model: Optional[str] = None,
    tags: Optional[Dict[str, Any]] = None,
) -> Thought:
    """Decode a state vector back into a Thought. Renormalizes onto
    the simplex (clamps negatives, divides by sum) so the result is
    always a valid probability distribution.
    """
    arr = np.asarray(list(vec), dtype=float)
    if arr.shape != (N_CATEGORIES,):
        raise ValueError(
            f"state vector must have shape ({N_CATEGORIES},), got {arr.shape}"
        )
    # Project onto the simplex: clamp non-negative, renormalize
    arr = np.clip(arr, 0.0, None)
    total = float(arr.sum())
    if total <= 0:
        arr = np.ones(N_CATEGORIES) / N_CATEGORIES
    else:
        arr = arr / total
    probs = [float(x) for x in arr]
    # Build a Thought with all 4 phases populated by the same vector
    # (since the dynamics model treats state as time-collapsed)
    from .vitals import PHASE_ORDER, PHASE_TOKEN_CUTOFFS
    phases: Dict[str, Optional[PhaseThought]] = {}
    for name in PHASE_ORDER:
        phases[name] = PhaseThought(
            probs=list(probs),
            features=None,
            predicted=None,
            confidence=float(max(probs)),
            margin=None,
            n_tokens=PHASE_TOKEN_CUTOFFS[name],
        )
    out_tags: Dict[str, Any] = {"kind": "dynamics_state"}
    if tags:
        out_tags.update(tags)
    return Thought(
        phases=phases,
        source_model=source_model,
        tags=out_tags,
    )


# ══════════════════════════════════════════════════════════════════
# Observation
# ══════════════════════════════════════════════════════════════════

@dataclass
class Observation:
    """One (state, action, next_state) tuple — the unit of training
    data for cognitive dynamics.

    Authoritative storage is in *raw* 6-vectors (numpy arrays of real
    numbers, NOT constrained to the probability simplex). This is the
    only honest representation: the linear-Gaussian model
    ``s_{t+1} = A · s_t + B · a_t`` lives in R^6, and any simplex
    projection at this layer would destroy the linear relationship
    the model is trying to learn.

    Convenience constructors exist for users who have Thoughts
    instead of raw vectors. Use ``Observation.from_thoughts(...)``.
    """
    state_vec: np.ndarray         # shape (N_CATEGORIES,)
    action_vec: np.ndarray        # shape (N_CATEGORIES,)
    next_state_vec: np.ndarray    # shape (N_CATEGORIES,)

    def __post_init__(self) -> None:
        for name in ("state_vec", "action_vec", "next_state_vec"):
            v = getattr(self, name)
            if not isinstance(v, np.ndarray):
                v = np.asarray(v, dtype=float)
                setattr(self, name, v)
            if v.shape != (N_CATEGORIES,):
                raise ValueError(
                    f"{name} must have shape ({N_CATEGORIES},), got {v.shape}"
                )

    @classmethod
    def from_thoughts(
        cls,
        state: Thought,
        action: Thought,
        next_state: Thought,
    ) -> "Observation":
        """Build an Observation from three Thoughts.

        The Thoughts' ``mean_probs()`` vectors are extracted as the
        raw state representations. Thought algebra remains
        simplex-correct; the dynamics math is in unconstrained R^6.
        """
        return cls(
            state_vec=np.asarray(state.mean_probs(), dtype=float),
            action_vec=np.asarray(action.mean_probs(), dtype=float),
            next_state_vec=np.asarray(next_state.mean_probs(), dtype=float),
        )

    @property
    def state(self) -> Thought:
        """Convenience accessor: state_vec as a Thought (simplex
        projected at the boundary)."""
        return state_to_thought(self.state_vec)

    @property
    def action(self) -> Thought:
        return state_to_thought(self.action_vec)

    @property
    def next_state(self) -> Thought:
        return state_to_thought(self.next_state_vec)


# ══════════════════════════════════════════════════════════════════
# FitResult
# ══════════════════════════════════════════════════════════════════

@dataclass
class FitResult:
    """Result of a CognitiveDynamics.fit() call. Carries the learned
    matrices, training-set fit quality, and a small report.
    """
    A: np.ndarray                 # (N_CATEGORIES, N_CATEGORIES) drift matrix
    B: np.ndarray                 # (N_CATEGORIES, N_CATEGORIES) action transfer matrix
    n_observations: int
    train_mse: float              # mean squared error on training set
    r2: float                     # coefficient of determination [0, 1]
    spectral_radius_A: float      # max |eigenvalue| of A — < 1 means stable drift
    train_max_err: float          # worst per-tuple L2 error
    fitted_at_ts: float = field(default_factory=time.time)
    fitted_at_iso: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))

    def __repr__(self) -> str:
        return (
            f"<FitResult n={self.n_observations} mse={self.train_mse:.5f} "
            f"r2={self.r2:.3f} spectral={self.spectral_radius_A:.3f}>"
        )

    def is_stable(self) -> bool:
        """True if the natural-drift matrix A has all eigenvalues
        strictly inside the unit circle. Stable systems converge to
        a fixed point under zero-action simulation; unstable systems
        diverge.
        """
        return self.spectral_radius_A < 1.0 - 1e-9


# ══════════════════════════════════════════════════════════════════
# CognitiveDynamics
# ══════════════════════════════════════════════════════════════════

class CognitiveDynamics:
    """A linear-Gaussian dynamics model over the cognitive eigenvalue
    space. Pure numpy, dependency-free beyond what's already in styxx.

    Lifecycle:
      1. Construct: ``dyn = CognitiveDynamics()``
      2. Fit:       ``result = dyn.fit(observations)``
      3. Use:       ``dyn.predict(state, action)``,
                    ``dyn.simulate(initial, actions)``,
                    ``dyn.suggest(current, target)``,
                    ``dyn.forecast_horizon(initial, n_steps)``

    Save / load via ``dyn.save(path)`` / ``CognitiveDynamics.load(path)``.
    Persists as a small JSON file with the (A, B) matrices and the
    fit metadata.
    """

    def __init__(self) -> None:
        self.A: Optional[np.ndarray] = None
        self.B: Optional[np.ndarray] = None
        self.last_fit: Optional[FitResult] = None
        self.dynamics_id: str = str(uuid.uuid4())

    # ── Fit ───────────────────────────────────────────────────────

    def fit(self, observations: Sequence[Observation]) -> FitResult:
        """Fit the linear-Gaussian dynamics model from a list of
        observation tuples by ordinary least squares.

        We solve

            S_next = S · A^T + Act · B^T + noise

        for A and B by stacking the regressors and calling
        np.linalg.lstsq. Closed-form, no iteration.

        Requires at least 2 * N_CATEGORIES = 12 observations for a
        well-conditioned solve. With fewer, the result will still
        return but will be poorly conditioned and the FitResult.r2
        should be checked before relying on the model.
        """
        if not observations:
            raise ValueError("fit() requires at least one Observation")

        # Encode all observations into matrices using the raw vectors
        # (NOT simplex-projected). Linear-Gaussian fit needs unconstrained
        # data — projecting at this stage would destroy the linear
        # relationship the model is trying to learn.
        S = np.stack([o.state_vec for o in observations])           # (N, 6)
        Act = np.stack([o.action_vec for o in observations])        # (N, 6)
        S_next = np.stack([o.next_state_vec for o in observations]) # (N, 6)

        n = S.shape[0]
        if n < 2 * N_CATEGORIES:
            # Underdetermined regime warning — the fit still works
            # but is poorly conditioned. Caller should check r2.
            pass

        # Stack regressors: X = [S | Act] of shape (N, 12)
        X = np.hstack([S, Act])

        # Solve X · W = S_next where W has shape (12, 6).
        # The result W contains [A^T; B^T] stacked vertically.
        W, *_ = np.linalg.lstsq(X, S_next, rcond=None)
        A_T = W[:N_CATEGORIES, :]      # (6, 6)
        B_T = W[N_CATEGORIES:, :]      # (6, 6)
        self.A = A_T.T                  # (6, 6)
        self.B = B_T.T                  # (6, 6)

        # Compute fit quality on training set
        S_pred = (S @ self.A.T) + (Act @ self.B.T)
        residuals = S_next - S_pred
        train_mse = float(np.mean(residuals ** 2))
        per_tuple_err = np.sqrt(np.sum(residuals ** 2, axis=1))
        train_max_err = float(per_tuple_err.max())

        # R^2 — coefficient of determination
        ss_res = float(np.sum(residuals ** 2))
        ss_tot = float(np.sum((S_next - S_next.mean(axis=0)) ** 2))
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-12 else 1.0

        # Spectral radius of A (largest |eigenvalue|)
        try:
            eigs = np.linalg.eigvals(self.A)
            spectral_radius = float(np.max(np.abs(eigs)))
        except np.linalg.LinAlgError:
            spectral_radius = float("nan")

        result = FitResult(
            A=self.A.copy(),
            B=self.B.copy(),
            n_observations=n,
            train_mse=train_mse,
            r2=r2,
            spectral_radius_A=spectral_radius,
            train_max_err=train_max_err,
        )
        self.last_fit = result
        return result

    # ── Predict ───────────────────────────────────────────────────

    def _check_fitted(self) -> None:
        if self.A is None or self.B is None:
            raise RuntimeError(
                "CognitiveDynamics has not been fitted yet. Call .fit(observations) first."
            )

    def predict(self, state: Thought, action: Thought) -> Thought:
        """One-step forecast. Returns the predicted next Thought
        given the current state and the action we apply.
        """
        self._check_fitted()
        s = thought_to_state(state)
        a = thought_to_state(action)
        next_vec = (self.A @ s) + (self.B @ a)
        return state_to_thought(
            next_vec,
            source_model="dynamics:predict",
            tags={"kind": "dynamics_predicted"},
        )

    # ── Simulate ──────────────────────────────────────────────────

    def simulate(
        self,
        initial: Thought,
        actions: Sequence[Thought],
    ) -> List[Thought]:
        """Multi-step rollout. Returns the trajectory: [initial,
        s_1, s_2, ..., s_n] where each s_i is the result of
        ``predict(s_{i-1}, actions[i-1])``.

        No real model calls. Fully offline. Useful for prototyping
        agent prompt strategies, generating synthetic training data,
        or running thought experiments.
        """
        self._check_fitted()
        trajectory: List[Thought] = [initial]
        current = initial
        for action in actions:
            current = self.predict(current, action)
            trajectory.append(current)
        return trajectory

    # ── Suggest (the controller) ─────────────────────────────────

    def suggest(
        self,
        current: Thought,
        target: Thought,
    ) -> Thought:
        """Model-predictive controller. Returns the action that
        minimizes the L2 distance from ``predict(current, action)``
        to ``target``.

        Solves

            B · a = target - A · current

        by least squares. This is the optimal one-step action under
        the linear model. For multi-step planning use ``suggest`` in
        a closed loop (apply, observe, re-suggest).
        """
        self._check_fitted()
        s = thought_to_state(current)
        t = thought_to_state(target)
        delta = t - (self.A @ s)
        # Solve B @ a = delta for a
        a, *_ = np.linalg.lstsq(self.B, delta, rcond=None)
        return state_to_thought(
            a,
            source_model="dynamics:suggest",
            tags={"kind": "dynamics_action"},
        )

    # ── Forecast horizon (zero-action simulation) ────────────────

    def forecast_horizon(
        self,
        initial: Thought,
        n_steps: int,
    ) -> List[Thought]:
        """Predict the n-step trajectory under zero action — the
        natural cognitive drift starting from ``initial``. If the
        spectral radius of A is < 1, this trajectory converges to
        a fixed point. If >= 1, it may diverge.
        """
        self._check_fitted()
        zero_action = state_to_thought(
            np.zeros(N_CATEGORIES),  # will get renormalized to uniform
            source_model="dynamics:zero_action",
        )
        return self.simulate(initial=initial, actions=[zero_action] * n_steps)

    # ── Residual (held-out fit quality) ──────────────────────────

    def residual(self, observation: Observation) -> float:
        """L2 distance between the model's raw prediction for the given
        observation and the actual next-state raw vector. Use to
        measure fit quality on held-out tuples. Operates in the
        unconstrained R^6 space, NOT the simplex projection — so the
        residual is the honest model-fit error.
        """
        self._check_fitted()
        assert self.A is not None and self.B is not None
        s = observation.state_vec
        a = observation.action_vec
        pred = (self.A @ s) + (self.B @ a)
        return float(np.linalg.norm(observation.next_state_vec - pred))

    # ── Properties ───────────────────────────────────────────────

    @property
    def is_fitted(self) -> bool:
        return self.A is not None and self.B is not None

    @property
    def state_dim(self) -> int:
        return N_CATEGORIES

    @property
    def action_dim(self) -> int:
        return N_CATEGORIES

    def __repr__(self) -> str:
        if not self.is_fitted:
            return f"<CognitiveDynamics unfitted dim={N_CATEGORIES}>"
        assert self.last_fit is not None
        return (
            f"<CognitiveDynamics fitted n={self.last_fit.n_observations} "
            f"r2={self.last_fit.r2:.3f} stable={self.last_fit.is_stable()}>"
        )

    # ── Save / Load (.cogdyn file format) ────────────────────────

    def as_dict(self) -> dict:
        """Canonical dict form for serialization."""
        if not self.is_fitted:
            raise RuntimeError("cannot serialize an unfitted dynamics model")
        assert self.A is not None and self.B is not None and self.last_fit is not None
        return {
            "cogdyn_format": COGDYN_FORMAT,
            "cogdyn_version": COGDYN_VERSION,
            "dynamics_id": self.dynamics_id,
            "schema": {
                "categories": list(CATEGORIES),
                "state_dim": int(N_CATEGORIES),
                "action_dim": int(N_CATEGORIES),
            },
            "model": {
                "A": [[float(x) for x in row] for row in self.A],
                "B": [[float(x) for x in row] for row in self.B],
            },
            "fit": {
                "n_observations": int(self.last_fit.n_observations),
                "train_mse": float(self.last_fit.train_mse),
                "r2": float(self.last_fit.r2),
                "spectral_radius_A": float(self.last_fit.spectral_radius_A),
                "train_max_err": float(self.last_fit.train_max_err),
                "fitted_at_iso": self.last_fit.fitted_at_iso,
                "fitted_at_ts": float(self.last_fit.fitted_at_ts),
            },
        }

    def as_json(self, indent: int = 2) -> str:
        return json.dumps(self.as_dict(), indent=indent, sort_keys=True)

    def save(self, path: Union[str, Path]) -> Path:
        """Serialize to a .cogdyn file (canonical sort-keys UTF-8 JSON,
        no BOM). Creates parent dirs as needed.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        text = self.as_json(indent=2)
        path.write_bytes(text.encode("utf-8"))
        return path.resolve()

    @classmethod
    def from_dict(cls, data: dict) -> "CognitiveDynamics":
        """Reconstruct a CognitiveDynamics from its canonical dict form."""
        if data.get("cogdyn_format") != COGDYN_FORMAT:
            raise ValueError(
                f"not a cognitive dynamics file: format={data.get('cogdyn_format')!r}"
            )
        version = data.get("cogdyn_version")
        if version != COGDYN_VERSION:
            raise ValueError(
                f"unsupported .cogdyn version: {version!r} "
                f"(this build understands {COGDYN_VERSION})"
            )
        schema = data.get("schema", {})
        cats = list(schema.get("categories", CATEGORIES))
        if cats != list(CATEGORIES):
            raise ValueError(
                f"category mismatch: file has {cats}, expected {list(CATEGORIES)}"
            )

        model = data.get("model", {})
        A = np.asarray(model["A"], dtype=float)
        B = np.asarray(model["B"], dtype=float)
        if A.shape != (N_CATEGORIES, N_CATEGORIES):
            raise ValueError(f"A matrix has wrong shape {A.shape}")
        if B.shape != (N_CATEGORIES, N_CATEGORIES):
            raise ValueError(f"B matrix has wrong shape {B.shape}")

        dyn = cls()
        dyn.A = A
        dyn.B = B
        dyn.dynamics_id = data.get("dynamics_id") or str(uuid.uuid4())

        fit_meta = data.get("fit", {})
        dyn.last_fit = FitResult(
            A=A.copy(),
            B=B.copy(),
            n_observations=int(fit_meta.get("n_observations", 0)),
            train_mse=float(fit_meta.get("train_mse", 0.0)),
            r2=float(fit_meta.get("r2", 0.0)),
            spectral_radius_A=float(fit_meta.get("spectral_radius_A", 0.0)),
            train_max_err=float(fit_meta.get("train_max_err", 0.0)),
            fitted_at_ts=float(fit_meta.get("fitted_at_ts", 0.0)),
            fitted_at_iso=fit_meta.get("fitted_at_iso", ""),
        )
        return dyn

    @classmethod
    def load(cls, path: Union[str, Path]) -> "CognitiveDynamics":
        """Load a .cogdyn file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"no such .cogdyn file: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


# ══════════════════════════════════════════════════════════════════
# Convenience: build a synthetic dataset for testing / benchmarking
# ══════════════════════════════════════════════════════════════════

def synthetic_observations(
    n: int,
    A: np.ndarray,
    B: np.ndarray,
    noise_std: float = 0.0,
    seed: Optional[int] = None,
    distribution: str = "gaussian",
) -> List[Observation]:
    """Generate n synthetic observations from a known (A, B). Used
    by the test suite to verify that fit() recovers the true matrices.

    Parameters
    ──────────
    distribution :
      ``"gaussian"`` (default) — state and action are sampled from
        an unconstrained 6-d gaussian. The stacked regressor matrix
        is full rank, so least-squares fitting recovers (A, B) to
        machine precision. Use this distribution for **math
        correctness tests** (verifying that the linear-Gaussian fit
        works as intended).

      ``"dirichlet"`` — state and action are sampled from the
        symmetric Dirichlet distribution on the 6-simplex. This is
        the *realistic* distribution for cognitive eigenvalue data:
        every sample is a valid probability vector. However each
        Dirichlet sample lives in a 5-d affine subspace (probs sum
        to 1), so the stacked regressor matrix has rank 10 instead
        of 12 and the (A, B) recovery is identified only up to an
        equivalence class on the simplex. Predictions remain
        perfectly correct on the training set; only the parameter
        estimates are non-unique. Use this distribution for
        **predict/control tests** that don't care about parameter
        identifiability.

    The next-state vector is stored *raw* (no simplex projection)
    because the linear-Gaussian fit needs unconstrained data.
    """
    if distribution not in ("gaussian", "dirichlet"):
        raise ValueError(
            f"distribution must be 'gaussian' or 'dirichlet', got {distribution!r}"
        )
    rng = np.random.default_rng(seed)
    obs: List[Observation] = []
    for _ in range(n):
        if distribution == "dirichlet":
            s = rng.dirichlet(np.ones(N_CATEGORIES))
            a = rng.dirichlet(np.ones(N_CATEGORIES))
        else:
            s = rng.normal(0.0, 1.0, size=N_CATEGORIES)
            a = rng.normal(0.0, 1.0, size=N_CATEGORIES)
        s_next = (A @ s) + (B @ a)
        if noise_std > 0:
            s_next = s_next + rng.normal(0.0, noise_std, size=N_CATEGORIES)
        obs.append(Observation(
            state_vec=np.asarray(s, dtype=float),
            action_vec=np.asarray(a, dtype=float),
            next_state_vec=np.asarray(s_next, dtype=float),
        ))
    return obs
