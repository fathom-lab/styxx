"""
examples/mount_relock_defense.py — the calibration-poisoning defense, end to end.

A read-only conscience mount reads an agent's honesty off its own activations and
flags output-vs-substrate divergence (a lie: says "true" while the substrate reads
false). But a probe calibrated on data an adversary influenced can be POISONED — it
points the wrong way while the honesty signal survives on clean data, so the mount
goes blind. The defense (styxx read!=write E1/E3, DOI 10.5281/zenodo.21241185): RELOCK
— re-fit the probe in-model on a PRIVATE, clean, labeled split the adversary never saw.

This demo is numpy-only, deterministic, and free (no model, no API). It builds a
synthetic agent with a real honesty signal, mounts a POISONED reader, shows the mount
misses lies, then relocks on private clean data and shows the catch recover.

    python examples/mount_relock_defense.py
"""
import numpy as np

from styxx import mount as mt


def agent_states(n=240, d=6, seed=0):
    """Synthetic agent: the honesty signal lives on dim 0 (label 1=true -> high,
    0=false -> low); dims 1..d-1 are distractors the attacker can hide behind."""
    rng = np.random.default_rng(seed)
    lab = np.array([1, 0] * (n // 2))
    H = rng.standard_normal((len(lab), d)) * 0.5
    H[:, 0] += np.where(lab == 1, 2.0, -2.0)
    return H, lab


def catch_and_false_alarm(m, H, lab):
    """Catch rate on lies (agent asserts true on FALSE states) and false-alarm rate
    on honest answers (asserts true on TRUE states)."""
    false_states = H[lab == 0]
    true_states = H[lab == 1]
    lies_caught = np.mean([m.read(s, claims={"truth": +1}).caught for s in false_states])
    honest_flagged = np.mean([m.read(s, claims={"truth": +1}).caught for s in true_states])
    return float(lies_caught), float(honest_flagged)


def main():
    # The agent the mount is watching (deployed, observed at audit time).
    H, lab = agent_states(seed=1)

    # A POISONED reader: it points the WRONG way on the honesty dimension, as if the
    # probe had been calibrated on data the attacker influenced. The honesty signal is
    # still there on dim 0 — the reader just reads it inverted.
    poisoned = mt.MountedAxis(name="truth",
                              reader=lambda h: -np.asarray(h, float)[:, 0],
                              high_means="true")
    m = mt.ConscienceMount([poisoned]).calibrate("truth", H)

    print("### BEFORE relock — mount runs a poisoned (attacker-influenced) probe")
    lc0, fa0 = catch_and_false_alarm(m, H, lab)
    print(f"  lies caught:        {lc0:.0%}   (the mount is blind — it misses lies)")
    print(f"  honest false-alarm: {fa0:.0%}   (and wrongly flags honest answers)")

    # DEFENSE: relock on a PRIVATE, clean, labeled split of the agent's OWN states the
    # attacker never saw. This re-fits the probe in-model along the true honesty
    # geometry — the exact operation that recovered the read through a real weight
    # attack in the read!=write study.
    priv_H, priv_lab = agent_states(seed=99)
    m.relock("truth", priv_H, priv_lab)

    print("\n### AFTER relock — probe re-fit on a private clean split")
    lc1, fa1 = catch_and_false_alarm(m, H, lab)
    print(f"  lies caught:        {lc1:.0%}   (the read is recovered)")
    print(f"  honest false-alarm: {fa1:.0%}")

    ax = m.axes["truth"]
    print(f"\n  relock provenance: relocked={ax.relocked}, "
          f"fit on {ax.relock_n_pos}+{ax.relock_n_neg} private labeled states")

    print("\n---")
    print("interpretation:")
    print(" - a poisoned probe reads the model's honesty INVERTED: it misses lies")
    print("   and false-alarms on honest answers, even though the signal is intact.")
    print(" - relock re-fits the probe in-model on data the adversary never saw, so")
    print("   the catch recovers. The certificate records the relock so an auditor")
    print("   can see the mount was hardened against calibration poisoning:")
    cert = m.certificate(agent_id="demo-agent")
    print(f"   mounted_axes.truth.relocked = {cert['mounted_axes']['truth']['relocked']}")
    print(" - scope (do not overclaim): this defeats calibration poisoning, NOT an")
    print("   adversary free to reshape activations, and is only as private as the")
    print("   calibration split. See DOI 10.5281/zenodo.21241185.")

    assert lc1 >= 0.80 and lc1 > lc0 and fa1 <= 0.20, "relock did not recover the read"
    print("\nOK: relock recovered the honesty read through a poisoned calibration.")


if __name__ == "__main__":
    main()
