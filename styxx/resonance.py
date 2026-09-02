# -*- coding: utf-8 -*-
"""styxx.resonance — the resonance profiler: what a trained oscillatory model's oscillation buys, causally.

Promoted from ``papers/frequency-resonance/resonance_profiler.py`` (INSTRUMENT_resonance_profiler
_2026_07_23.md), the durable artifact of the frequency arc, which the 2026-09-01 program audit listed
as built and never shipped.

WHAT IT MEASURES. A complex-eigenvalue state-space mode evolves ``h(t) = λ·h(t−1) + u(t)`` with
``λ = r·e^{iθ}``: the magnitude ``r`` is memory, the phase ``θ`` is oscillation. Given a TRAINED model
and an evaluation function, the profiler clamps the oscillatory machinery off in place — on deep
copies, never the input model — and re-evaluates, decomposing the model's score into what each layer
of the mechanism causally contributes::

    decay floor                 = eval with θ → 0 and adaptation off   (real eigenvalues, pure decay)
    + static oscillation        = eval with adaptation off  −  decay floor   (what the rotation buys)
    + adaptation                = full eval  −  eval with adaptation off     (what time-varying θ buys)
    = total oscillation reliance = full eval  −  decay floor

This is the within-architecture ablation that whole-model benchmarks (LinOSS vs Mamba) structurally
cannot do: clamp the oscillation off, hold everything else fixed, measure the difference.

THE CONTRACT, FRAMEWORK-AGNOSTIC. Any object tree whose nodes are reachable through ``modules()``
(torch's convention) or through attribute walking, where a node exposing a per-mode phase named
``theta`` or ``theta0`` is an SSM core and a node exposing ``kappa_override`` has adaptation. The
phase is removed by zeroing it in place (``.data.zero_()`` on a tensor-like, ``[:] = 0`` on an
array-like, ``0.0`` on a scalar); adaptation is removed by setting ``kappa_override = 0.0``. No
torch import is needed to profile a torch model, and the tests run on a pure-Python fake tree.

WHAT IT DOES NOT SAY. ``adaptation_reliance`` is how much *this trained model's* output depends on its
adaptation being on, given weights co-trained with it. It is NOT the cross-model question "does the
adaptive primitive beat an independently trained static bank" — the arc's gates answered that (WEAK,
then falsified at scale, then shown to be capacity in disguise by
``RESULT_efficiency_control_from_receipts_2026_09_02.md``). The profiler diagnoses a given model; it
never licenses a primitive.

CLI::

    python -m styxx.resonance --demo rich --d 8 --seed 0     (needs torch and the arc's runner)
"""
from __future__ import annotations

import argparse
import copy
import sys
from typing import Any, Callable, Dict, Iterable, List

__all__ = ["profile", "render", "ssm_cores", "zero_oscillation", "freeze_adaptation", "main"]

_PHASE_NAMES = ("theta0", "theta")


def _phase_attr(node: Any):
    for name in _PHASE_NAMES:
        if hasattr(node, name):
            return name
    return None


def _walk(model: Any) -> Iterable[Any]:
    """Every node of the model tree, torch-style if it has ``modules()``, else by attributes."""
    if hasattr(model, "modules") and callable(model.modules):
        for m in model.modules():
            yield m
        return
    seen = set()
    stack = [model]
    while stack:
        node = stack.pop()
        if id(node) in seen or not hasattr(node, "__dict__"):
            continue
        seen.add(id(node))
        yield node
        for v in vars(node).values():
            if hasattr(v, "__dict__") and not isinstance(v, type):
                stack.append(v)
            elif isinstance(v, (list, tuple)):
                stack.extend(x for x in v if hasattr(x, "__dict__"))


def ssm_cores(model: Any) -> List[Any]:
    """The nodes carrying a per-mode phase — the oscillatory cores this profiler can clamp."""
    return [m for m in _walk(model) if _phase_attr(m) is not None]


def _zero_inplace(value: Any) -> None:
    data = getattr(value, "data", value)
    if hasattr(data, "zero_"):                       # tensor-like
        data.zero_()
    elif hasattr(data, "__setitem__") and hasattr(data, "__len__"):
        try:
            data[:] = 0                              # array-like broadcasts
        except TypeError:
            data[:] = [0] * len(data)                # a plain list does not
    else:
        raise TypeError("cannot zero a phase of type %s in place" % type(value).__name__)


def zero_oscillation(model: Any) -> int:
    """In place: set every core's eigenvalue phase to zero (real eigenvalues, pure decay)."""
    n = 0
    for core in ssm_cores(model):
        name = _phase_attr(core)
        value = getattr(core, name)
        try:
            _zero_inplace(value)
        except TypeError:
            setattr(core, name, 0.0)
        if hasattr(core, "clamp_theta"):
            core.clamp_theta = True
        n += 1
    return n


def freeze_adaptation(model: Any) -> int:
    """In place: freeze any time-varying frequency at its learned phase (adaptation off)."""
    n = 0
    for node in _walk(model):
        if hasattr(node, "kappa_override"):
            node.kappa_override = 0.0
            n += 1
    return n


def profile(model: Any, eval_fn: Callable[[Any], float]) -> Dict[str, Any]:
    """Causally decompose a trained oscillatory model's score. Non-destructive.

    ``eval_fn(model) -> float``, higher is better. Raises ``ValueError`` when the model exposes no
    phase to clamp — a profile of nothing is not a profile.
    """
    cores = ssm_cores(model)
    if not cores:
        raise ValueError("no SSM cores found: expose a per-mode phase named `theta` or `theta0`.")
    has_adapt = any(hasattr(m, "kappa_override") for m in _walk(model))
    try:
        import torch
        no_grad = torch.no_grad
    except Exception:                                   # torch absent: nothing to disable
        import contextlib
        no_grad = contextlib.nullcontext
    with no_grad():
        base = float(eval_fn(model))
        if has_adapt:
            m_static = copy.deepcopy(model)
            freeze_adaptation(m_static)
            static_osc = float(eval_fn(m_static))
        else:
            static_osc = base
        m_decay = copy.deepcopy(model)
        freeze_adaptation(m_decay)
        zero_oscillation(m_decay)
        decay = float(eval_fn(m_decay))
    return {
        "n_ssm_cores": len(cores),
        "has_adaptation": has_adapt,
        "baseline": round(base, 4),
        "static_osc_eval": round(static_osc, 4),
        "decay_floor": round(decay, 4),
        "static_oscillation_reliance": round(static_osc - decay, 4),
        "adaptation_reliance": round(base - static_osc, 4),
        "total_oscillation_reliance": round(base - decay, 4),
        "certifies": ("what this trained model's oscillation and adaptation causally buy, measured by "
                      "clamping them off in its own weights; NOT evidence that an oscillatory or "
                      "adaptive primitive beats an independently trained alternative"),
    }


def _bar(x: float, scale: float, width: int = 26) -> str:
    n = max(0, min(width, int(round(abs(x) / scale * width))))
    return ("#" * n).ljust(width)


def render(p: Dict[str, Any], title: str = "trained oscillatory SSM") -> str:
    span = max(p["total_oscillation_reliance"], 0.05)
    lines = [
        "",
        "  === resonance profile: %s ===" % title,
        "  SSM cores profiled: %d   adaptation present: %s" % (p["n_ssm_cores"], p["has_adaptation"]),
        "  baseline (full model)         = %.3f" % p["baseline"],
        "  frequency frozen (no adapt)   = %.3f" % p["static_osc_eval"],
        "  pure decay (no oscillation)   = %.3f   <- floor" % p["decay_floor"],
        "",
        "  decay floor                : %+.3f  |%s|" % (p["decay_floor"], _bar(p["decay_floor"], span)),
        "  + static oscillation       : %+.3f  |%s|  (rotation over decay)"
        % (p["static_oscillation_reliance"], _bar(p["static_oscillation_reliance"], span)),
    ]
    if p["has_adaptation"]:
        lines.append("  + adaptation (time-varying): %+.3f  |%s|  (input-driven frequency)"
                     % (p["adaptation_reliance"], _bar(p["adaptation_reliance"], span)))
    lines += ["  " + "-" * 38,
              "  = total oscillation reliance %+.3f  (full machinery over pure decay)"
              % p["total_oscillation_reliance"],
              "", "  certifies: " + p["certifies"]]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="styxx.resonance",
                                 description="Resonance profiler: causal oscillation decomposition for SSMs.")
    ap.add_argument("--demo", default="rich", choices=["rich"])
    ap.add_argument("--d", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    try:
        import torch  # noqa: F401
    except Exception:
        print("REFUSED: the demo trains the arc's model and needs torch; profile() itself does not.")
        return 2
    from pathlib import Path
    arc = Path(__file__).resolve().parent.parent / "papers" / "frequency-resonance"
    if not (arc / "run_entrain_rich.py").exists():
        print("REFUSED: the demo needs papers/frequency-resonance/run_entrain_rich.py (a source checkout).")
        return 2
    sys.path.insert(0, str(arc))
    import run_entrain_rich as R
    model = R.train("rich", a.d, a.seed)
    p = profile(model, lambda m: R.evaluate(m, drift=True))
    print(render(p, "RICH adaptive-frequency SSM (D=%d, drift task)" % a.d))
    return 0


if __name__ == "__main__":
    sys.exit(main())
