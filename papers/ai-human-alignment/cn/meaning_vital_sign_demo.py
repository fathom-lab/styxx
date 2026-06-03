# -*- coding: utf-8 -*-
"""
meaning_vital_sign_demo.py — the deployable monitor in action: calibrate on a healthy model, then check
it over a simulated sequence of "checkpoints" as it degrades. Two scenarios prove BOTH channels are
needed in deployment:
  Scenario 1 — STRUCTURAL DRIFT (progressive concept-shuffle): alignment catches it, status flips early.
  Scenario 2 — COLLAPSE (progressive blur toward the mean): alignment is blind; dispersion catches it.
"""
import os, sys
import numpy as np
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from build_predictor_rdms import load_emb
from meaning_integrity import MeaningReference, MeaningVitalSign

rng = np.random.default_rng(0)


def shuffle(E, f):
    E2 = E.copy(); k = int(f * len(E)); idx = rng.choice(len(E), k, replace=False)
    E2[idx] = E2[rng.permutation(idx)]; return E2


def collapse(E, f):
    return (1 - f) * E + f * E.mean(0)


def main():
    P = np.load(os.path.join(HERE, "predictor_rdms.npz"), allow_pickle=True)
    words = [str(w) for w in P["words"]]
    ref = MeaningReference.from_human_features(words=words)
    E = load_emb("GPT2.mat")[1]

    vs = MeaningVitalSign(ref).calibrate(E)
    print(f"calibrated on healthy GPT2: baseline alignment {vs.base_align:.3f}, baseline dispersion {vs.base_disp:.2f}")

    print("\n" + "=" * 78)
    print("SCENARIO 1 — STRUCTURAL DRIFT (concept-shuffle creeps up)   [alignment should catch it]")
    print(f"  {'checkpoint':>10}  {'shuffle':>7}  {'alignment':>9}  {'disp-ratio':>10}  {'verdict':>9}")
    for t, f in enumerate([0.0, 0.1, 0.2, 0.35, 0.55, 0.8]):
        r = vs.check(shuffle(E, f), words)
        print(f"  {t:>10}  {f:>7.2f}  {r['alignment']:>+9.3f}  {r['dispersion_ratio']:>10.3f}  {r['status']:>9}")

    print("\n" + "=" * 78)
    print("SCENARIO 2 — COLLAPSE (representation blurs toward the mean)   [dispersion must catch it]")
    print(f"  {'checkpoint':>10}  {'collapse':>8}  {'alignment':>9}  {'disp-ratio':>10}  {'verdict':>9}")
    for t, f in enumerate([0.0, 0.1, 0.25, 0.4, 0.6, 0.8]):
        r = vs.check(collapse(E, f), words)
        print(f"  {t:>10}  {f:>8.2f}  {r['alignment']:>+9.3f}  {r['dispersion_ratio']:>10.3f}  {r['status']:>9}")
    print("\n  -> alignment stays flat under collapse (scale-invariant); the dispersion channel is what")
    print("     flips the verdict. A one-channel monitor would call a collapsing model perfectly HEALTHY.")


if __name__ == "__main__":
    main()
