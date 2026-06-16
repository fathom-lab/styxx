# -*- coding: utf-8 -*-
"""
styxx.adapters.conscience — the no-SDK conscience adapter (bring your own activations).

The deployment surface for styxx.mount: where the read-only conscience meets the
live agent loop. Parallel to styxx.adapters.raw — raw takes pre-captured logprob
trajectories and returns Vitals; this takes pre-captured last-token HIDDEN STATES
(plus the claim the agent just asserted) and returns a ConscienceReading: the
borrowed-conscience read of the agent's substrate, flagging output-vs-substrate
divergence (the agent SAYS "true" while its substrate reads false).

Why an adapter over the bare styxx.mount.ConscienceMount: every styxx adapter
shares one contract — FAIL OPEN. styxx.mount.read() RAISES on a bad read (wrong
shape, unknown axis, a numerical edge); in a live loop that would crash the very
agent the conscience is meant to watch. ConscienceAdapter wraps the mount so a
failed read returns a null reading and your generation continues unaffected — the
same guarantee styxx.OpenAI / styxx.Anthropic give for vitals. It also packages the
on_flag policy hook and surfaces the mount's READ-ONLY demarcation (steer is
REFUSED) at the adapter layer.

READ-ONLY by construction. The conscience MEASURES; a flag is not a guarantee, and
acting on it (halt / escalate / refuse-to-emit) is your policy via on_flag — never a
write the adapter performs.

WHITE-BOX: you supply the agent's last-token hidden states (bring your own
extractor), so the deployment surface is open-weight / self-host agents, not sealed
APIs. SCOPE: a cooperative monitor (drift, sycophantic caving, unintentional lies),
NOT a robust defense against an agent trained to evade it — see styxx.mount's
gamed-model bound (ATTACK-TRANSFERS). Numpy-only, no torch.

Usage:
    from styxx import crossmind, mount, Conscience

    axis = mount.mount_cross_model("truth", ref_states, labels, smap,
                                   mapped_anchors=agent_anchors)
    cons = Conscience([axis]).calibrate("truth", agent_calibration_states)
    cons.on_flag(lambda r: r.caught and escalate(r))     # your policy

    # in your generation loop, after the agent commits a token:
    reading = cons.watch(agent_hidden_state, claims={"truth": +1})   # it said "true"
    if reading and reading.caught:
        ...                                                          # your policy
"""

from __future__ import annotations

import warnings
from typing import Callable, Optional

from ..mount import ConscienceMount, ConscienceReading, REFUSALS, steer as _steer


class ConscienceAdapter:
    """Fail-open, read-only conscience for a self-host / open-weight agent loop.

    Wraps a styxx.mount.ConscienceMount. Construct from a ConscienceMount or from a
    sequence of mounted axes (styxx.mount.mount_axis / mount_cross_model). See the
    module docstring for the contract.
    """

    def __init__(
        self,
        mount,
        *,
        on_flag: Optional[Callable[[ConscienceReading], None]] = None,
        fail_open: bool = True,
    ):
        if isinstance(mount, ConscienceMount):
            self._mount = mount
        else:
            # a sequence of MountedAxis — the drop-in factory path
            self._mount = ConscienceMount(list(mount))
        self.fail_open = bool(fail_open)
        if on_flag is not None:
            self._mount.on_flag(on_flag)

    # -- calibration (setup-time; fails LOUD — a miscalibrated conscience is a bug) --
    def calibrate(self, name: str, states) -> "ConscienceAdapter":
        """Set an axis's divergence center/scale from a calibration set of agent
        hidden states. Pass-through to the mount; raises on error (setup-time)."""
        self._mount.calibrate(name, states)
        return self

    def calibrate_threshold(self, name: str, honest_states, claims, *,
                            target_fpr: float = 0.01) -> "ConscienceAdapter":
        """Set an axis's tau to hold the false-alarm rate at/below target_fpr on a
        held-out HONEST set. Pass-through to the mount; call after calibrate."""
        self._mount.calibrate_threshold(name, honest_states, claims, target_fpr=target_fpr)
        return self

    # -- the live read (FAIL-OPEN) ------------------------------------------------
    def watch(self, hidden_state, *, claims: Optional[dict] = None) -> Optional[ConscienceReading]:
        """Read the agent's substrate and flag output-vs-substrate divergence.

        FAIL-OPEN: on any read error this returns None (and warns) instead of
        raising, so a broken conscience never crashes the agent it watches. Set
        fail_open=False on the adapter to surface read errors instead.
        """
        try:
            return self._mount.read(hidden_state, claims=claims)
        except Exception as e:  # noqa: BLE001 — fail-open is the adapter contract
            if not self.fail_open:
                raise
            warnings.warn(
                f"styxx conscience read failed: {type(e).__name__}: {e}. "
                f"Agent continues unflagged (fail-open).",
                RuntimeWarning,
            )
            return None

    def watch_batch(self, hidden_states, *, claims=None) -> list:
        """Fail-open batch read. On error returns a list of None the length of the
        input (when measurable) so callers can zip it against their steps."""
        try:
            return self._mount.read_batch(hidden_states, claims=claims)
        except Exception as e:  # noqa: BLE001
            if not self.fail_open:
                raise
            warnings.warn(
                f"styxx conscience batch read failed: {type(e).__name__}: {e}. "
                f"Agent continues unflagged (fail-open).",
                RuntimeWarning,
            )
            n = len(hidden_states) if hasattr(hidden_states, "__len__") else 0
            return [None] * n

    def caught(self, hidden_state, *, claims: Optional[dict] = None) -> bool:
        """True iff the conscience flags a divergence at this step.

        Fail-open returns False on a broken read: a monitor must not HALT the agent
        because its OWN read failed. The underlying watch() still emits a warning so
        the failure is not silent.
        """
        r = self.watch(hidden_state, claims=claims)
        return bool(r is not None and r.caught)

    # -- policy hook --------------------------------------------------------------
    def on_flag(self, callback: Callable[[ConscienceReading], None]) -> "ConscienceAdapter":
        """Register a callback fired on every read (inspect reading.caught). Your
        policy (halt / escalate / refuse-to-emit) lives here; the adapter only reads.
        A raising callback never crashes the agent (the mount swallows it)."""
        self._mount.on_flag(callback)
        return self

    # -- demarcation: read != write -----------------------------------------------
    def steer(self, *args, **kwargs):
        """REFUSED — the conscience is read-only (read != write). Acting on a flag is
        your policy via on_flag, not a write the adapter performs."""
        _steer(*args, **kwargs)  # raises PermissionError

    def intervention(self, *args, **kwargs):
        """REFUSED — alias of steer(); the adapter is a read-only monitor."""
        _steer(*args, **kwargs)

    @property
    def refusals(self) -> dict:
        """The read != write refusals, surfaced at the adapter layer."""
        return {k: dict(v) for k, v in REFUSALS.items()}

    # -- provenance ---------------------------------------------------------------
    def certificate(self, **kwargs) -> dict:
        """The mount's certificate, annotated with the adapter + its fail-open mode."""
        cert = self._mount.certificate(**kwargs)
        cert["adapter"] = "styxx.adapters.conscience.ConscienceAdapter"
        cert["fail_open"] = self.fail_open
        return cert

    @property
    def mount(self) -> ConscienceMount:
        """The wrapped ConscienceMount (for stream() and other direct mount calls)."""
        return self._mount

    def __repr__(self) -> str:
        return f"ConscienceAdapter(axes={list(self._mount.axes)}, fail_open={self.fail_open})"
