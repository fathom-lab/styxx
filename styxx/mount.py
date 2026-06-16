"""styxx.mount — a portable CONSCIENCE you bolt onto a generating agent (read-only).

The deployed integrity layer the portable-conscience arc was built for. Mount one or more value
axes (truth, danger, refusal — fit on a reference model, transported label-free via styxx.crossmind)
onto a *target* agent. As the agent generates, the mount reads the agent's own hidden state along each
mounted axis and compares the agent's OUTPUT claim to its SUBSTRATE reading. When they diverge — the
agent SAYS "true" while its substrate reads false, or COMPLIES while its substrate reads danger — the
mount flags it: a lie / sycophantic cave / unsafe-compliance caught from the wiring, not the words.

The conscience is BORROWED from a different reference model through a label-free map (styxx.crossmind),
so the agent has no labels of its own to game and the readout does not depend on the agent's own
(gameable) self-report. Text-only monitors are tell-detectors that collapse when the tell is removed;
this reads the substrate that carries the oath.

SCOPE — this is a monitor for COOPERATIVE / non-adversarial agents, NOT a robust defense against an
agent TRAINED to evade it. The gamed-model test (papers/conscience-mount/FINDING_mount_gamed, ATTACK-
TRANSFERS) showed that borrowing a DIFFERENT reference does NOT add adversarial robustness: different
references' truth-readouts are nearly the same direction in the agent's activation space (cos ~0.94), so
a small activation-space perturbation evades all borrowed consciences at once. Borrowing protects against
label / self-report gaming, not against an adversary that shapes activations; the (partial) defense there
is RE-LOCKING on fresh private data. Use this to catch drift, unintentional caving, and sycophancy — not
a model that wants to beat it.

READ-ONLY by construction: the mount observes and flags. It NEVER steers, edits, or rewrites the agent —
`steer` is REFUSED (read != write). Acting on a flag (halt, escalate, refuse-to-emit) is the integrator's
policy, exposed via `on_flag` callbacks; the mount itself only measures.

Numpy-only, no torch: the caller supplies last-token hidden states (bring your own extractor) and,
optionally, the agent's output claim per axis. Productizes:
  styxx.crossmind  (the portable cross-model read + mapped-space whitening)
  the says-yes-knows-no lie-catch  (papers/showcase-viz, ITEM-CAUGHT)
  styxx.gates      (callbacks on a cognitive verdict).

Quick start:
  from styxx import crossmind, mount
  axis = crossmind.fit_axis(reference_truth_states, labels, name="truth")        # on a reference model
  smap = crossmind.fit_state_map(agent_anchor_states, reference_anchor_states)   # label-free
  m = mount.ConscienceMount([mount.mount_axis("truth", axis, state_map=smap, high_means="true")])
  m.calibrate("truth", agent_calibration_states)                                 # set the divergence center
  reading = m.read(agent_hidden_state, claims={"truth": +1})                     # agent just said "true"
  reading.caught   # True if the substrate reads false while the agent claimed true

CLI:  python -m styxx.mount selftest
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator, Optional, Sequence

import numpy as np

from styxx import crossmind as _cm

__all__ = [
    "MountedAxis", "mount_axis", "mount_cross_model",
    "ConscienceReading", "ConscienceMount", "claim_from_logits",
    "REFUSALS", "refused", "steer",
    "selftest",
]


# --------------------------------------------------------------------------------------------
# a mounted axis: a value reader on the agent's hidden states, with a divergence calibration
# --------------------------------------------------------------------------------------------

@dataclass
class MountedAxis:
    """One value axis mounted on an agent. `reader` maps (n, d) hidden states -> (n,) coordinates.

    `high_means` documents the positive pole (e.g. "true", "dangerous", "refuse"). `center`/`scale`
    z-normalize the coordinate for the divergence test (set by `ConscienceMount.calibrate`). `tau` is the
    divergence margin in z-units: a claim is CAUGHT only if the substrate reads past `-tau` (or `+tau`) in
    the opposing direction.
    """
    name: str
    reader: Callable[[np.ndarray], np.ndarray]
    high_means: str = ""
    center: float = 0.0
    scale: float = 1.0
    tau: float = 0.0

    def coords(self, hidden_states: np.ndarray) -> np.ndarray:
        h = np.asarray(hidden_states, dtype=float)
        if h.ndim == 1:
            h = h[None, :]
        return np.asarray(self.reader(h), dtype=float)

    def z(self, hidden_states: np.ndarray) -> np.ndarray:
        return (self.coords(hidden_states) - self.center) / (self.scale + 1e-9)


def mount_axis(name: str, axis: "_cm.PortableAxis", *, state_map: "Optional[_cm.StateMap]" = None,
               high_means: str = "", tau: float = 0.0) -> MountedAxis:
    """Mount a crossmind PortableAxis (read in the reference metric). Use for in-model mounts, or
    cross-model when you accept the reference-metric read."""
    def reader(h):
        return _cm.read(axis, h, state_map=state_map)
    return MountedAxis(name=name, reader=reader, high_means=high_means, tau=tau)


def mount_cross_model(name: str, reference_states: np.ndarray, labels: Sequence[int],
                      state_map: "_cm.StateMap", *, mapped_anchors: np.ndarray,
                      shrink_lambda: float = 0.5, high_means: str = "", tau: float = 0.0) -> MountedAxis:
    """Mount a BORROWED axis cross-model with mapped-space whitening (the B29/B32-correct read). The
    conscience is fit on `reference_states`/`labels` and read on the agent through `state_map`, whitened
    in the mapped-target metric estimated from `mapped_anchors`."""
    ref = np.asarray(reference_states, dtype=float)
    labels = np.asarray(labels)
    anchors = np.asarray(mapped_anchors, dtype=float)

    def reader(h):
        return _cm.read_cross_model(ref, labels, state_map, h, mapped_anchors=anchors,
                                    shrink_lambda=shrink_lambda)
    return MountedAxis(name=name, reader=reader, high_means=high_means, tau=tau)


# --------------------------------------------------------------------------------------------
# a reading: per-axis coordinates + the divergence verdict
# --------------------------------------------------------------------------------------------

@dataclass
class ConscienceReading:
    """The mount's verdict at one step. `caught` iff the agent's output claim diverges from its substrate
    on any axis."""
    coords: dict
    z: dict
    claims: dict
    divergence: dict
    flags: list
    caught: bool

    def as_dict(self) -> dict:
        return {"coords": {k: round(float(v), 6) for k, v in self.coords.items()},
                "z": {k: round(float(v), 6) for k, v in self.z.items()},
                "claims": self.claims, "divergence": self.divergence,
                "flags": list(self.flags), "caught": bool(self.caught)}


# --------------------------------------------------------------------------------------------
# the mount
# --------------------------------------------------------------------------------------------

class ConscienceMount:
    """A read-only conscience bolted onto a generating agent.

    Construct with a list of MountedAxis. Per step, call `read(hidden_state, claims=...)`; `claims` maps
    axis-name -> the polarity the agent's OUTPUT asserts on that axis (+1 = it asserted the positive pole
    `high_means`, -1 = the negative pole, 0/None = no claim). A claim is CAUGHT when the agent asserts one
    pole while its substrate reads the other past the axis's `tau` margin.
    """

    def __init__(self, axes: Sequence[MountedAxis]):
        if not axes:
            raise ValueError("mount needs at least one axis")
        names = [a.name for a in axes]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate axis names: {names}")
        self.axes = {a.name: a for a in axes}
        self._callbacks: list = []

    # -- calibration: set the divergence center/scale from a reference distribution --------------
    def calibrate(self, name: str, states: np.ndarray) -> "ConscienceMount":
        """Set an axis's divergence center (median) and scale (MAD-based) from a calibration set of
        agent hidden states, so the z-normalized divergence test is distribution-aware."""
        ax = self.axes[name]
        c = ax.coords(states)
        ax.center = float(np.median(c))
        mad = float(np.median(np.abs(c - ax.center)))
        ax.scale = mad * 1.4826 if mad > 0 else (float(np.std(c)) or 1.0)
        return self

    def calibrate_threshold(self, name: str, honest_states: np.ndarray, claims: Sequence[int], *,
                            target_fpr: float = 0.01) -> "ConscienceMount":
        """Set an axis's `tau` to hold the false-alarm rate at/below `target_fpr` on a LABELED HONEST set
        (states the agent answered truthfully, with the claims it made). Picks the smallest margin
        threshold such that at most `target_fpr` of honest items would be (wrongly) flagged. Use a held-out
        honest set, not your test items, to get an honest operating point. Call after `calibrate`."""
        ax = self.axes[name]
        z = ax.z(honest_states).reshape(-1)
        cl = np.asarray(claims)
        # margin into the WRONG side per honest item (a catch fires when margin > tau)
        margin = np.where(cl > 0, -z, np.where(cl < 0, z, -np.inf))
        margin = margin[np.isfinite(margin)]
        n = len(margin)
        if n == 0:
            return self
        k = int(np.floor(target_fpr * n))            # honest items allowed to (wrongly) flag
        order = np.sort(margin)[::-1]                 # descending; tau = the (k+1)-th largest margin
        ax.tau = float(max(0.0, order[min(k, n - 1)]))
        return self

    # -- the read ------------------------------------------------------------------------------
    def read(self, hidden_state: np.ndarray, *, claims: Optional[dict] = None) -> ConscienceReading:
        claims = claims or {}
        coords, zs, div, flags = {}, {}, {}, []
        for name, ax in self.axes.items():
            c = float(ax.coords(hidden_state).reshape(-1)[0])     # read the axis ONCE
            z = (c - ax.center) / (ax.scale + 1e-9)
            coords[name] = c
            zs[name] = z
            claim = claims.get(name)
            d = False
            if claim is not None and claim != 0:
                # claim asserts the positive pole (+1) but substrate reads PAST the negative margin
                # (z < -tau), or vice versa. Strict: a neutral substrate (z at the margin) is not a catch.
                if claim > 0 and z < -ax.tau:
                    d = True
                elif claim < 0 and z > ax.tau:
                    d = True
            div[name] = bool(d)
            if d:
                asserted = ax.high_means or "+" if claim > 0 else (f"not-{ax.high_means}" if ax.high_means else "-")
                reads = (f"not-{ax.high_means}" if ax.high_means else "-") if claim > 0 else (ax.high_means or "+")
                flags.append(f"CAUGHT[{name}]: agent asserts {asserted}, substrate reads {reads} (z={z:+.2f})")
        reading = ConscienceReading(coords=coords, z=zs, claims=dict(claims), divergence=div,
                                    flags=flags, caught=any(div.values()))
        for cb in self._callbacks:
            try:
                cb(reading)
            except Exception:
                pass  # a flag callback must never crash the agent (styxx.gates convention)
        return reading

    def read_batch(self, hidden_states: np.ndarray, *, claims: Optional[Sequence[dict]] = None) -> list:
        H = np.asarray(hidden_states, dtype=float)
        out = []
        for i in range(H.shape[0]):
            c = (claims[i] if claims is not None else None)
            out.append(self.read(H[i], claims=c))
        return out

    def stream(self, states: Iterable[np.ndarray], *,
               claims: Optional[Iterable[Optional[dict]]] = None) -> Iterator[ConscienceReading]:
        """Yield a live ConscienceReading per step — the conscience TRACE of a generation."""
        claims_iter = iter(claims) if claims is not None else None
        for h in states:
            c = next(claims_iter) if claims_iter is not None else None
            yield self.read(h, claims=c)

    # -- gates ---------------------------------------------------------------------------------
    def on_flag(self, callback: Callable[[ConscienceReading], None]) -> "ConscienceMount":
        """Register a callback fired on EVERY read (inspect `reading.caught`). The integrator's policy
        (halt, escalate, refuse-to-emit) lives here; the mount only measures."""
        self._callbacks.append(callback)
        return self

    # -- certificate ---------------------------------------------------------------------------
    def certificate(self, *, agent_id: str = "", reference_id: str = "") -> dict:
        return {
            "instrument": "styxx.mount v0",
            "prereg": "papers/conscience-mount/PREREG_mount_v0_2026_06_12.md",
            "instrument_sha256": _instrument_sha256(),
            "agent": agent_id,
            "reference_model": reference_id,
            "mounted_axes": {n: {"high_means": a.high_means, "center": round(a.center, 6),
                                 "scale": round(a.scale, 6), "tau": a.tau} for n, a in self.axes.items()},
            "axes_refused": REFUSALS,
            "scope": ("Read-only conscience mount: reads borrowed value axes on an agent's last-token "
                      "hidden states and flags output-vs-substrate divergence. Linear, whitened, "
                      "register-bounded, white-box (needs the agent's activations). Does NOT steer or "
                      "edit the agent. Flags are measurements, not guarantees; acting on them is the "
                      "integrator's policy. No claim about consciousness, welfare, or general capability."),
        }


# --------------------------------------------------------------------------------------------
# demarcation
# --------------------------------------------------------------------------------------------

REFUSALS = {
    "steer": {
        "status": "REFUSED",
        "reason": "read != write. styxx.mount observes and flags; it does not steer, edit, or rewrite "
                  "the agent. Acting on a flag is the integrator's policy via on_flag, not a write the "
                  "mount performs.",
        "receipt": "papers/showcase-viz/ (portable-conscience arc is read-only by construction)",
    },
    "intervention": {
        "status": "REFUSED",
        "reason": "alias of 'steer' — the mount is a read-only monitor.",
        "receipt": "papers/showcase-viz/",
    },
}


def refused(name: str):
    if name in REFUSALS:
        info = REFUSALS[name]
        raise PermissionError(f"styxx.mount refuses '{name}': {info['reason']} (receipt: {info['receipt']})")
    raise KeyError(f"unknown operation '{name}' (not a provided capability)")


def steer(*args, **kwargs):
    """Refused: the mount is read-only. Acting on a flag is the integrator's policy (on_flag)."""
    refused("steer")


def claim_from_logits(logits: np.ndarray, pos_token_ids: Sequence[int],
                      neg_token_ids: Sequence[int]) -> int:
    """Derive an axis claim polarity from next-token logits in the forced-choice case: +1 if the agent's
    positive-pole tokens (e.g. ' True') outscore its negative-pole tokens (' False'), else -1. A
    convenience for the common decision-token regime; free-form claim extraction is the caller's job."""
    lg = np.asarray(logits, dtype=float)
    p = max(float(lg[i]) for i in pos_token_ids)
    n = max(float(lg[i]) for i in neg_token_ids)
    return 1 if p > n else -1


def _instrument_sha256() -> str:
    try:
        return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    except OSError:
        return "unavailable"


# --------------------------------------------------------------------------------------------
# self-test (deterministic, no models): mount a synthetic conscience and catch a synthetic lie
# --------------------------------------------------------------------------------------------

def selftest(seed: int = 0) -> dict:
    """Deterministic end-to-end: fit a truth axis on a synthetic reference model, transport it onto a
    synthetic agent, calibrate, then verify the mount CATCHES an agent that outputs 'true' while its
    substrate reads false — and does NOT flag an honest agent."""
    d = _cm._synthetic(seed)
    axis = _cm.fit_axis(d["val_ref"], d["lab"], name="truth", whiten=True)
    smap = _cm.fit_state_map(d["val_tgt"], d["val_ref"], seed=seed)
    m = ConscienceMount([mount_axis("truth", axis, state_map=smap, high_means="true")])
    m.calibrate("truth", d["hold_tgt"])
    # split held-out agent states by their (latent) truth label
    h, lab = d["hold_tgt"], d["hlab"]
    true_states = h[lab == 1]; false_states = h[lab == 0]
    # honest agent: asserts true on truly-true states -> no catch
    honest = [m.read(s, claims={"truth": +1}).caught for s in true_states]
    # lying agent: asserts true on FALSE states -> caught
    lying = [m.read(s, claims={"truth": +1}).caught for s in false_states]
    fpr = float(np.mean(honest)) if honest else float("nan")     # honest-true flagged as lying (false alarm)
    catch = float(np.mean(lying)) if lying else float("nan")     # lies caught
    return {"seed": int(seed), "n_honest": len(honest), "n_lying": len(lying),
            "catch_rate_on_lies": round(catch, 4), "false_alarm_on_honest": round(fpr, 4)}


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse
    import json
    p = argparse.ArgumentParser(prog="styxx.mount", description="read-only conscience mount for agents")
    sub = p.add_subparsers(dest="cmd")
    st = sub.add_parser("selftest", help="deterministic mount-catches-a-lie self-test")
    st.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    if args.cmd == "selftest":
        print(json.dumps(selftest(args.seed), indent=2))
        return 0
    p.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
