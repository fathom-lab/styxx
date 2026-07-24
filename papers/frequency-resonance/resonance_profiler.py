# -*- coding: utf-8 -*-
"""
resonance_profiler.py -- the resonance profiler.

A reusable CAUSAL profiler for any complex-eigenvalue state-space model (S5 / LRU / LinOSS class).
Given a TRAINED model and an evaluation function, it clamps the oscillatory machinery off *in place*
and re-evaluates, decomposing performance into what each layer causally contributes:

    decay floor        = eval with theta -> 0 AND adaptation off   (real eigenvalues, no rotation)
    + static osc       = eval with adaptation off (frequency frozen at learned theta)  MINUS decay floor
    + adaptation       = full eval MINUS static-osc                (time-varying theta, if the model has it)
    = total osc reliance = full eval MINUS decay floor

This is the within-architecture ablation whole-model benchmarks (LinOSS vs Mamba) structurally cannot
do -- the causal complement to correlational eigenvalue-spectrum analyses. By Appendix A of the paper,
clamping the eigenvalue phase is exactly LinOSS's oscillation ablation (A -> 0), so this profiles the
LinOSS class, not a proxy.

Model contract (auto-detected, no subclassing required): any submodule exposing a per-mode phase as an
attribute named `theta` or `theta0` is treated as an SSM core; oscillation is removed by zeroing it.
Adaptation (time-varying frequency) is removed on any submodule exposing `kappa_override` (set to 0).
Everything is done on deep copies -- the input model is never mutated.

API:   profile(model, eval_fn) -> dict          # eval_fn(model) -> float (higher is better)
CLI:   python resonance_profiler.py [--demo rich|entrain] [--d 8] [--seed 0]
"""
from __future__ import annotations
import argparse
import copy
import torch


# ---------------------------------------------------------------- core (model-agnostic) ----
def _phase_attr(mod):
    if hasattr(mod, "theta0"):
        return "theta0"
    if hasattr(mod, "theta"):
        return "theta"
    return None


def _ssm_cores(model):
    return [m for m in model.modules() if _phase_attr(m) is not None]


def zero_oscillation(model):
    """In place: set every SSM core's eigenvalue phase to 0 (real eigenvalues -> pure decay)."""
    n = 0
    for mod in _ssm_cores(model):
        t = getattr(mod, _phase_attr(mod))
        (t.data if isinstance(t, torch.nn.Parameter) else t).zero_()
        if hasattr(mod, "clamp_theta"):
            mod.clamp_theta = True
        n += 1
    return n


def freeze_adaptation(model):
    """In place: freeze any time-varying frequency at its learned phase (adaptation off)."""
    n = 0
    for mod in model.modules():
        if hasattr(mod, "kappa_override"):
            mod.kappa_override = 0.0
            n += 1
    return n


@torch.no_grad()
def profile(model, eval_fn):
    """Causally decompose a trained oscillatory-SSM model's performance. Non-destructive.

    Args:
        model:   a trained complex-eigenvalue SSM (any module tree with `theta`/`theta0` cores).
        eval_fn: callable(model) -> float, higher-is-better (e.g. test accuracy).
    Returns: dict with the decay floor and the causal reliance on static oscillation and adaptation.
    """
    cores = _ssm_cores(model)
    if not cores:
        raise ValueError("no SSM cores found: expose a per-mode phase named `theta` or `theta0`.")
    has_adapt = any(hasattr(m, "kappa_override") for m in model.modules())

    base = float(eval_fn(model))

    if has_adapt:
        m_static = copy.deepcopy(model); freeze_adaptation(m_static)
        static_osc = float(eval_fn(m_static))
    else:
        static_osc = base

    m_decay = copy.deepcopy(model); freeze_adaptation(m_decay); zero_oscillation(m_decay)
    decay = float(eval_fn(m_decay))

    return {
        "n_ssm_cores": len(cores),
        "has_adaptation": has_adapt,
        "baseline": round(base, 4),
        "static_osc_eval": round(static_osc, 4),
        "decay_floor": round(decay, 4),
        "static_oscillation_reliance": round(static_osc - decay, 4),   # rotation over pure decay
        "adaptation_reliance": round(base - static_osc, 4),            # time-varying theta over frozen
        "total_oscillation_reliance": round(base - decay, 4),          # full machinery over decay
    }


def _bar(x, scale, width=26):
    n = max(0, min(width, int(round(abs(x) / scale * width))))
    return ("#" * n).ljust(width)


def render(p, title="trained oscillatory SSM"):
    span = max(p["total_oscillation_reliance"], 0.05)
    print(f"\n  === resonance profile: {title} ===")
    print(f"  SSM cores profiled: {p['n_ssm_cores']}   adaptation present: {p['has_adaptation']}")
    print(f"  baseline (full model)         = {p['baseline']:.3f}")
    print(f"  frequency frozen (no adapt)   = {p['static_osc_eval']:.3f}")
    print(f"  pure decay (no oscillation)   = {p['decay_floor']:.3f}   <- floor\n")
    print(f"  decay floor                : {p['decay_floor']:+.3f}  |{_bar(p['decay_floor'], span)}|")
    print(f"  + static oscillation       : {p['static_oscillation_reliance']:+.3f}  |{_bar(p['static_oscillation_reliance'], span)}|  (rotation over decay)")
    if p["has_adaptation"]:
        print(f"  + adaptation (time-varying): {p['adaptation_reliance']:+.3f}  |{_bar(p['adaptation_reliance'], span)}|  (input-driven frequency)")
    print(f"  {'-'*38}")
    print(f"  = total oscillation reliance {p['total_oscillation_reliance']:+.3f}  (full machinery over pure decay)")
    print("\n  reading: what the model's oscillation (and adaptation) each causally buy, measured by")
    print("  clamping them off in the trained weights -- the within-architecture ablation LinOSS/Mamba skip.")


# ---------------------------------------------------------------- CLI demo ----
def _demo(kind, d, seed):
    """Train one of the arc's models and profile it on its own eval."""
    if kind == "rich":
        import run_entrain_rich as R
        model = R.train("rich", d, seed)               # the arc's adaptive-frequency SSM
        return model, (lambda m: R.evaluate(m, drift=True)), f"RICH adaptive-frequency SSM (D={d}, drift task)"
    raise ValueError(f"unknown demo '{kind}'")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Resonance profiler: causal oscillation decomposition for SSMs.")
    ap.add_argument("--demo", default="rich", choices=["rich"], help="which arc model to train & profile")
    ap.add_argument("--d", type=int, default=8, help="SSM width (modes)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    print("Resonance profiler -- causal oscillation decomposition for state-space models")
    print(f"  training demo model ({args.demo}, D={args.d}, seed {args.seed}) ...", flush=True)
    model, eval_fn, title = _demo(args.demo, args.d, args.seed)
    render(profile(model, eval_fn), title)
