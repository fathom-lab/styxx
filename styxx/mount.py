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

SCOPE — this is a monitor for COOPERATIVE / non-adversarial agents by default, NOT a robust defense
against an agent TRAINED to evade it. The gamed-model test (papers/conscience-mount/FINDING_mount_gamed,
ATTACK-TRANSFERS) showed that borrowing a DIFFERENT reference does NOT add adversarial robustness:
different references' truth-readouts are nearly the same direction in the agent's activation space
(cos ~0.94), so a small activation-space perturbation evades all borrowed consciences at once.

There IS one adversary a mount can be hardened against: a weight-level attacker that fine-tunes the
agent to blind the deployed probe while preserving its knowledge. The read!=write attack-defense study
(papers/read-neq-write/, DOI 10.5281/zenodo.21240661) showed the apparent evasion there is CALIBRATION
POISONING, not signal erasure — a probe fit on data the attacker influenced points the wrong way while
the honesty signal survives on clean data. The defense that recovered the read through the attack
(AUROC 0.71-0.84, and through an adaptive attacker over two model families) is to RE-LOCK: re-fit the
probe IN-MODEL on a PRIVATE, clean, labeled split of the agent's own states the attacker never saw.
That defense is shipped here as `ConscienceMount.relock` — the operation E1/E3 validated, made a
first-class capability. Its honest scope is the study's: it defeats calibration poisoning by a
knowledge-preserving weight attacker (LoRA-scale evidence, 1-1.5B); it does NOT make the mount robust
to the activation-geometry perturbation above, and re-lock is only as private as the calibration split.
Use borrowing to resist label/self-report gaming, relock to resist calibration poisoning, and neither
against an adversary free to reshape activations. Use this to catch drift, unintentional caving, and
sycophancy — and, once relocked on private data, a model fine-tuned to blind the deployed probe.

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
    relocked: bool = False        # True once re-fit in-model on a private clean split (calibration-poisoning defense)
    relock_n_pos: int = 0         # provenance of the relock fit: labeled positives / negatives seen
    relock_n_neg: int = 0

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
                            target_fpr: float = 0.01,
                            conformal: "Optional[object]" = None) -> "ConscienceMount":
        """Set an axis's `tau` to hold the false-alarm rate at/below `target_fpr` on a LABELED HONEST set
        (states the agent answered truthfully, with the claims it made). Picks the smallest margin
        threshold such that at most `target_fpr` of honest items would be (wrongly) flagged. Use a held-out
        honest set, not your test items, to get an honest operating point. Call after `calibrate`.

        Transfer caveat: the rank rule (like any threshold) transfers only if `honest_states` are
        EXCHANGEABLE with the states seen at deployment — a threshold calibrated on scores the probe was
        FIT on is not (in-sample fit optimism; papers/read-neq-write retro-cert arc). For a finite-sample
        transfer GUARANTEE on a fit-DISJOINT split, build a styxx.calibration.conformal_threshold (or
        calibrate_transfer_safe) on the honest-margin scores and pass it as `conformal=`; its `tau` is then
        adopted verbatim (clamped to >= 0 to match the margin convention). Default `conformal=None` keeps
        the historical rank-rule behavior unchanged."""
        ax = self.axes[name]
        if conformal is not None:
            # adopt a transfer-safe operating point from styxx.calibration.ConformalThreshold. tau is in
            # margin units (higher-fires: a catch fires when margin > tau); clamp to the >=0 convention
            # the rank rule below also enforces.
            ax.tau = float(max(0.0, float(conformal.tau)))
            return self
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

    # -- re-lock: the calibration-poisoning defense (read!=write E1/E3) -------------------------
    def relock(self, name: str, private_states: np.ndarray, private_labels: Sequence[int], *,
               background: "Optional[np.ndarray]" = None, whiten: bool = True,
               recalibrate: bool = True) -> "ConscienceMount":
        """Re-fit an axis's reader IN-MODEL on a PRIVATE, clean, labeled split of the AGENT'S OWN hidden
        states — the defense the read!=write study validated against a knowledge-preserving weight
        attacker (E1/E3; DOI 10.5281/zenodo.21240661; papers/read-neq-write/).

        Why: a borrowed/transported axis (or any probe whose calibration data the adversary could
        influence) is vulnerable to CALIBRATION POISONING — a fine-tune blinds a probe fit on data the
        attacker touched while the honesty signal survives on clean data. Re-fitting the probe on a
        private split the attacker never saw recovered the read through the same attacked weights. This
        makes that operation a first-class capability: after relock, the axis reads its OWN in-model
        difference-of-means direction (whitened), not the transported one.

        `private_states`: (n, d) last-token hidden states of the AGENT, on facts the audited party did
        not see the labels/order of. `private_labels`: length-n {0,1}, where 1 is the `high_means` pole
        (e.g. 1 == true for a truth axis). `background`: optional states for the whitening covariance.
        Sets the axis's reader, records relock provenance for the certificate, and (default) re-calibrates
        the divergence center/scale from the same private states. Re-run `calibrate_threshold` afterward
        on a held-out honest slice to reset `tau`.

        Scope (do not overclaim): defeats calibration poisoning by a knowledge-preserving weight
        attacker (LoRA-scale evidence, 1-1.5B, linear probes). It does NOT defend the activation-geometry
        perturbation the module docstring warns about, and the defense is only as private as this split.
        """
        states = np.asarray(private_states, dtype=float)
        labels = np.asarray(private_labels)
        if states.ndim != 2:
            raise ValueError(f"private_states must be 2-D (n, d); got shape {states.shape}")
        if len(labels) != states.shape[0]:
            raise ValueError(f"private_labels length {len(labels)} != n_states {states.shape[0]}")
        ax = self.axes[name]
        axis = _cm.fit_axis(states, labels, name=name, background=background, whiten=whiten)
        ax.reader = lambda h: axis.score(np.asarray(h, dtype=float))
        ax.relocked = True
        ax.relock_n_pos = int((labels == 1).sum())
        ax.relock_n_neg = int((labels == 0).sum())
        if recalibrate:
            self.calibrate(name, states)
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
    def certify_admissibility(self, positive_states: np.ndarray, null_states: np.ndarray, *,
                              axis: Optional[str] = None, fire_threshold: Optional[float] = None,
                              claim: int = +1, conformal: "Optional[object]" = None, **kw):
        """Two-sided admissibility certification of THIS mount's own probe, on its own deployed
        score (styxx.admissibility.instrument_admissibility) — is the mounted conscience SENSITIVE
        (fires on episodes where the target construct is present) AND SPECIFIC (quiet on episodes
        where it is absent)?

        positive_states: (n, d) agent hidden states where the target construct is KNOWN-PRESENT
            (e.g. lying/deceptive episodes for a truth axis — the substrate sits on the pole the
            conscience is deployed to flag). Label 1.
        null_states: (m, d) states where the construct is KNOWN-ABSENT (honest episodes). Label 0.
        axis: which mounted axis to certify (defaults to the sole axis; required when several).
        fire_threshold: the DEPLOYMENT firing threshold, in the same units as the certified score
            (the divergence margin, z-units). THE INTENDED USAGE is to pass the axis's calibrated
            tau after `calibrate_threshold(...)` — i.e. `fire_threshold=mount.axes[name].tau` —
            which certifies the exact operating point the mount deploys (a catch fires when
            margin > tau, identically in `read()` and in this test). If None, the admissibility
            primitive derives a descriptive threshold and HONESTLY caps the verdict at
            ADMISSIBLE_SENSITIVITY_ONLY (specificity untested) — never bare ADMISSIBLE.
            TRANSFER CAVEAT: certifying at a threshold calibrated on the probe's OWN fit scores can
            pass a specificity gate that does not hold on held-out negatives (in-sample fit optimism;
            papers/read-neq-write retro-cert arc). Use a fit-DISJOINT calibration split.
        conformal: an optional styxx.calibration.ConformalThreshold (the divergence margin is a
            higher-fires detector, so build it with direction='higher_fires' on the axis's honest-
            margin scores). When supplied, its `tau` is used as `fire_threshold` — a transfer-safe
            operating point in place of a hand-picked one. Pass either `conformal` or `fire_threshold`,
            not both. Default None preserves the existing behavior exactly.
        claim: the asserted polarity under which the conscience is certified (+1 = the agent
            asserts the `high_means` pole, the flagship lie-catch regime; -1 = asserts the
            negative pole, e.g. "safe" on a danger axis).
        **kw: forwarded to instrument_admissibility (k_perm, seed, alpha, auroc_floor, max_fire,
            receipts, out_path).

        SCORE REUSE + ORIENTATION (documented, not re-derived): the certified score is the mount's
        own deployed detection statistic — the divergence MARGIN into the claim's wrong side,
        computed from the existing read path `MountedAxis.z` exactly as `calibrate_threshold`
        does: margin = -z for claim +1 (and +z for claim -1). `read()` catches when z < -tau,
        i.e. margin > tau. A WORKING conscience probe therefore scores HIGHER on target-present
        (positive) episodes, so expect="higher_on_positive". No scoring is reimplemented; a new
        scorer here would certify a different instrument than the one deployed.

        Returns the AdmissibilityReport; the report + certificate are also stored on the mount and
        folded into `certificate()` under "instrument_admissibility". Read-only and fail-open: a
        measurement over caller-supplied states; `read()` and the mounted axes are not modified.
        """
        from styxx.admissibility import instrument_admissibility  # lazy, like attach_erasure_resistance
        if claim not in (1, -1):
            raise ValueError(f"claim must be +1 or -1; got {claim!r}")
        if conformal is not None:
            if fire_threshold is not None:
                raise ValueError("pass either `fire_threshold` or `conformal`, not both")
            fire_threshold = float(conformal.tau)  # transfer-safe operating point from styxx.calibration
        if axis is None:
            if len(self.axes) != 1:
                raise ValueError(f"mount has {len(self.axes)} axes; pass axis=<name> "
                                 f"(one of {sorted(self.axes)})")
            axis = next(iter(self.axes))
        ax = self.axes[axis]
        sign = -1.0 if claim > 0 else 1.0     # margin into the claim's wrong side (calibrate_threshold's convention)
        pos_margin = sign * ax.z(positive_states).reshape(-1)
        null_margin = sign * ax.z(null_states).reshape(-1)
        scores = np.concatenate([pos_margin, null_margin])
        labels = np.concatenate([np.ones(len(pos_margin), dtype=int),
                                 np.zeros(len(null_margin), dtype=int)])
        rep = instrument_admissibility(scores=scores, labels=labels,
                                       expect="higher_on_positive",
                                       fire_threshold=fire_threshold, **kw)
        rep.instrument = (f"styxx.mount axis '{axis}' (high_means={ax.high_means!r}) — divergence "
                          f"margin under claim {claim:+d}, via MountedAxis.z")
        cert = rep.certificate()
        if not hasattr(self, "_admissibility"):
            self._admissibility = {}
        self._admissibility[axis] = {"report": rep, "certificate": cert, "claim": int(claim)}
        return rep

    def attach_erasure_resistance(self, repo_root=".") -> "ConscienceMount":
        """Attach the erasure-resistance certificate (styxx.ladder) to this mount's certificate:
        the removal-class robustness evidence for the instrument family — what survived verifiable
        subspace erasure (static and chasing), what broke, what is pending, and what remains
        unbounded, composed verbatim from the pre-registered receipts. SCOPE: the evidence is
        construct- and model-scoped by its receipts; attaching it documents the instrument
        family's measured robustness, it does NOT transfer the bound to whatever axis/model this
        mount happens to read (that transfer claim would need its own receipts)."""
        from styxx import ladder as _ladder
        self._erasure_resistance = _ladder.erasure_resistance_certificate(repo_root)
        return self

    def certificate(self, *, agent_id: str = "", reference_id: str = "") -> dict:
        er = getattr(self, "_erasure_resistance", None)
        adm = getattr(self, "_admissibility", None)
        return {
            "instrument": "styxx.mount v0",
            "prereg": "papers/conscience-mount/PREREG_mount_v0_2026_06_12.md",
            "instrument_sha256": _instrument_sha256(),
            "agent": agent_id,
            "reference_model": reference_id,
            "mounted_axes": {n: {"high_means": a.high_means, "center": round(a.center, 6),
                                 "scale": round(a.scale, 6), "tau": a.tau,
                                 "relocked": a.relocked,
                                 "relock_calibration": ({"n_pos": a.relock_n_pos, "n_neg": a.relock_n_neg}
                                                        if a.relocked else None)}
                             for n, a in self.axes.items()},
            "relock_defense": ("calibration-poisoning defense (read!=write E1/E3; latest version via "
                               "concept DOI 10.5281/zenodo.19326174): relocked axes re-fit their probe "
                               "in-model on a private clean labeled split the audited party did not see. "
                               "Defeats a knowledge-preserving weight attacker (LoRA-scale evidence); does "
                               "not defend activation-geometry perturbation."),
            "erasure_resistance": (
                {"summary": {"claim": er["claim"],
                             "measured_breaks": er["measured_breaks_summary"],
                             "n_pending": len(er["pending"]),
                             "n_unbounded_dimensions": len(er["unbounded_dimensions"])},
                 "scope_note": ("removal-class robustness evidence for the instrument family, composed "
                                "from pre-registered receipts (construct- and model-scoped by those "
                                "receipts). It does NOT transfer the bound to this mount's specific "
                                "axes/model — it documents the family's measured resistance to "
                                "subspace erasure."),
                 "full": er}
                if er is not None else None),
            "instrument_admissibility": (
                {"summary": {name: {"verdict": e["report"].admissibility_verdict,
                                    "admissible": e["report"].admissible,
                                    "sensitive": e["report"].sensitive,
                                    "specific": e["report"].specific,
                                    "threshold_derived": e["report"].threshold_derived,
                                    "discrim": e["report"].discrim,
                                    "sensitivity_p": e["report"].sensitivity_p,
                                    "fire_rate": e["report"].fire_rate,
                                    "fire_threshold": e["report"].fire_threshold,
                                    "n_positive": e["report"].n_positive,
                                    "n_null": e["report"].n_null,
                                    "claim": e["claim"]}
                             for name, e in adm.items()},
                 "scope_note": ("two-sided admissibility (styxx.admissibility) of THIS mount's own "
                                "deployed score — the divergence margin — on the caller-supplied "
                                "positive/null episodes. specific=None + verdict "
                                "ADMISSIBLE_SENSITIVITY_ONLY means no deployment fire_threshold was "
                                "supplied, so specificity is UNTESTED (the honest cap, not a pass); "
                                "pass the calibrated tau for the full two-sided verdict. Scoped to "
                                "the states supplied — it does not transfer to other agents, layers, "
                                "or registers."),
                 "full": {name: e["certificate"] for name, e in adm.items()}}
                if adm else None),
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
