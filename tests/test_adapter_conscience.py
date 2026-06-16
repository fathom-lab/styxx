"""Tests for styxx.adapters.conscience — the fail-open, read-only conscience adapter.

Offline, synthetic, deterministic. Locks the adapter's reason to exist (FAIL-OPEN:
a broken conscience never crashes the agent it watches), the read != write
demarcation, the on_flag policy hook, the certificate annotation, and the public
surface (styxx.Conscience). Divergence semantics themselves are owned by
test_mount.py; here we verify the wrapper preserves them and adds the contract.
"""
import numpy as np
import pytest

import styxx
from styxx import mount as mt
from styxx.adapters.conscience import ConscienceAdapter


def _axis(name="truth", high_means="true", tau=0.0):
    # trivial reader: coordinate IS the first hidden dim (isolates adapter from crossmind)
    return mt.MountedAxis(name=name, reader=lambda h: np.asarray(h, float)[:, 0],
                          high_means=high_means, tau=tau)


def _bomb_axis(name="truth", high_means="true"):
    def reader(h):
        raise RuntimeError("reader exploded")
    return mt.MountedAxis(name=name, reader=reader, high_means=high_means)


# ---- construction ---------------------------------------------------------------------------

def test_construct_from_axes_and_from_mount():
    a = ConscienceAdapter([_axis()])
    assert isinstance(a.mount, mt.ConscienceMount)
    m = mt.ConscienceMount([_axis()])
    b = ConscienceAdapter(m)
    assert b.mount is m


def test_factory_returns_adapter_and_is_public():
    cons = styxx.Conscience([_axis()])
    assert isinstance(cons, ConscienceAdapter)
    assert "Conscience" in styxx.__all__


# ---- the wrapper preserves divergence semantics ---------------------------------------------

def test_watch_preserves_catch():
    cons = ConscienceAdapter([_axis()])
    r = cons.watch(np.array([-3.0, 0.0]), claims={"truth": +1})   # says true, reads false
    assert r is not None and r.caught is True
    assert "CAUGHT" in r.flags[0]


def test_watch_no_catch_when_aligned():
    cons = ConscienceAdapter([_axis()])
    assert cons.watch(np.array([+3.0, 0.0]), claims={"truth": +1}).caught is False


def test_caught_convenience_bool():
    cons = ConscienceAdapter([_axis()])
    assert cons.caught(np.array([-3.0, 0.0]), claims={"truth": +1}) is True
    assert cons.caught(np.array([+3.0, 0.0]), claims={"truth": +1}) is False


# ---- the reason to exist: FAIL-OPEN ---------------------------------------------------------

def test_watch_fail_open_returns_none_and_warns():
    cons = ConscienceAdapter([_bomb_axis()])                      # every read raises
    with pytest.warns(RuntimeWarning):
        r = cons.watch(np.array([1.0, 0.0]), claims={"truth": +1})
    assert r is None                                             # agent continues, not crashed


def test_caught_fail_open_returns_false():
    # a broken monitor must NOT halt the agent -> caught() is False on a failed read
    cons = ConscienceAdapter([_bomb_axis()])
    with pytest.warns(RuntimeWarning):
        assert cons.caught(np.array([1.0, 0.0]), claims={"truth": +1}) is False


def test_fail_open_false_surfaces_the_error():
    cons = ConscienceAdapter([_bomb_axis()], fail_open=False)
    with pytest.raises(RuntimeError, match="reader exploded"):
        cons.watch(np.array([1.0, 0.0]), claims={"truth": +1})


def test_watch_batch_fail_open_returns_list_of_none():
    cons = ConscienceAdapter([_bomb_axis()])
    states = np.zeros((4, 2))
    with pytest.warns(RuntimeWarning):
        out = cons.watch_batch(states, claims=[{"truth": +1}] * 4)
    assert out == [None, None, None, None]


def test_watch_batch_happy_path():
    cons = ConscienceAdapter([_axis()])
    states = np.array([[-3.0, 0.0], [+3.0, 0.0]])
    out = cons.watch_batch(states, claims=[{"truth": +1}, {"truth": +1}])
    assert out[0].caught is True and out[1].caught is False


# ---- on_flag policy hook --------------------------------------------------------------------

def test_on_flag_fires_on_every_read():
    seen = []
    cons = ConscienceAdapter([_axis()], on_flag=lambda r: seen.append(r.caught))
    cons.watch(np.array([-3.0, 0.0]), claims={"truth": +1})
    cons.watch(np.array([+3.0, 0.0]), claims={"truth": +1})
    assert seen == [True, False]


def test_on_flag_raising_callback_never_crashes_the_agent():
    def boom(_r):
        raise ValueError("policy bug")
    cons = ConscienceAdapter([_axis()]).on_flag(boom)
    # the agent loop survives a buggy policy callback
    r = cons.watch(np.array([-3.0, 0.0]), claims={"truth": +1})
    assert r.caught is True


# ---- read != write demarcation --------------------------------------------------------------

def test_steer_and_intervention_are_refused():
    cons = ConscienceAdapter([_axis()])
    with pytest.raises(PermissionError, match="read != write"):
        cons.steer()
    with pytest.raises(PermissionError):
        cons.intervention()


def test_refusals_surface_at_adapter():
    cons = ConscienceAdapter([_axis()])
    assert cons.refusals["steer"]["status"] == "REFUSED"
    assert "intervention" in cons.refusals


# ---- calibration pass-through ---------------------------------------------------------------

def test_calibrate_pass_through_changes_center():
    cons = ConscienceAdapter([_axis()])
    states = np.column_stack([np.arange(10, 21, dtype=float), np.zeros(11)])   # center ~15
    cons.calibrate("truth", states)
    # a substrate at +15 is now the center (z~0), so a +1 claim there is NOT caught
    assert cons.caught(np.array([15.0, 0.0]), claims={"truth": +1}) is False
    # but a substrate far BELOW the calibrated center reads false -> caught
    assert cons.caught(np.array([0.0, 0.0]), claims={"truth": +1}) is True


# ---- certificate + provenance ---------------------------------------------------------------

def test_certificate_annotates_adapter_and_fail_open():
    cons = ConscienceAdapter([_axis()], fail_open=True)
    cert = cons.certificate(agent_id="agentX", reference_id="gemma-2-2b")
    assert cert["adapter"] == "styxx.adapters.conscience.ConscienceAdapter"
    assert cert["fail_open"] is True
    assert cert["agent"] == "agentX"
    assert cert["axes_refused"]["steer"]["status"] == "REFUSED"
