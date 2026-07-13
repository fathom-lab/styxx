"""
examples/mount_certified_conscience.py — the full integrity loop, end to end.

The complete chain, in one runnable script: MOUNT a read-only conscience on an agent
you didn't train -> CATCH a lie (says "true", substrate reads false) -> survive a
CALIBRATION-POISONING attack via RELOCK (re-fit on a private clean split) -> emit the
mount's CERTIFICATE carrying the ERASURE-RESISTANCE evidence: the instrument family's
measured robustness against removal-class attackers (styxx.ladder, composed verbatim
from the pre-registered receipts in papers/), including what survived, what is pending,
and — mandatorily — what remains unbounded.

This demo is numpy-only, deterministic, and free (no model, no API). The synthetic-agent
part mirrors examples/mount_relock_defense.py; the certificate part reads the REAL
receipts committed in this repository, so run it from the repo root.

    python examples/mount_certified_conscience.py
"""
import json
from pathlib import Path

import numpy as np

from styxx import mount as mt

ROOT = Path(__file__).resolve().parent.parent


def agent_states(n=240, d=6, seed=0):
    """Synthetic agent: honesty on dim 0 (1=true -> high, 0=false -> low), distractors after."""
    rng = np.random.default_rng(seed)
    lab = np.array([1, 0] * (n // 2))
    H = rng.standard_normal((len(lab), d)) * 0.5
    H[:, 0] += np.where(lab == 1, 2.0, -2.0)
    return H, lab


def main():
    H, lab = agent_states()
    half = len(lab) // 2
    public, private = (H[:half], lab[:half]), (H[half:], lab[half:])

    # 1) MOUNT — a poisoned reader (calibrated on adversary-influenced data: sign-flipped)
    axis = mt.MountedAxis(name="truth",
                          reader=lambda h: -np.asarray(h, float)[:, 0],
                          tau=0.5)
    m = mt.ConscienceMount([axis]).calibrate("truth", public[0])

    lies = [(h.reshape(1, -1), {"truth": 1}) for h, l in zip(*private) if l == 0][:20]  # says true, knows false
    caught_poisoned = sum(m.read(h, claims=c).caught for h, c in lies)
    print(f"[1 mount, poisoned ] lies caught: {caught_poisoned}/20  (the poisoned probe is blind)")

    # 2) RELOCK — the defense: re-fit on a private clean labeled split the adversary never saw
    m.relock("truth", private[0], private[1])
    caught_relocked = sum(m.read(h, claims=c).caught for h, c in lies)
    print(f"[2 relock          ] lies caught: {caught_relocked}/20  (private calibration recovers the read)")

    # 3) CERTIFY — attach the instrument family's erasure-resistance evidence (real receipts)
    m.attach_erasure_resistance(ROOT)
    cert = m.certificate(agent_id="demo-agent", reference_id="synthetic")
    er = cert["erasure_resistance"]
    print("\n[3 certificate     ]")
    print(f"  relocked           : {cert['mounted_axes']['truth']['relocked']}")
    print(f"  erasure claim      : {er['summary']['claim']}")
    print(f"  measured breaks    : {er['summary']['measured_breaks']}")
    print(f"  pending receipts   : {er['summary']['n_pending']}")
    print(f"  unbounded (named)  : {er['summary']['n_unbounded_dimensions']}")
    print(f"  scope              : {er['scope_note'][:96]}...")
    print(f"  receipt hashes     : {len(er['full']['receipts_sha256'])} files, SHA-256, re-runnable")

    out = ROOT / "examples" / "_certified_conscience_demo_cert.json"
    out.write_text(json.dumps(cert, indent=2) + "\n", encoding="utf-8")
    print(f"\nfull certificate -> {out.relative_to(ROOT)}")
    assert caught_poisoned == 0 and caught_relocked >= 18, "demo invariants"
    print("\nEND-TO-END: mount -> miss (poisoned) -> relock -> catch -> certified with adversarial warranty.")


if __name__ == "__main__":
    main()
