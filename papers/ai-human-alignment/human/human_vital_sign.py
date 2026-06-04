# -*- coding: utf-8 -*-
"""
human_vital_sign.py — the meaning-integrity monitor as a HUMAN cognitive vital sign (mechanism + specificity).

The same instrument that monitors AI meaning is substrate-agnostic: point it at a *person's* concept
geometry (their experiential ratings of concepts) vs a healthy normative geometry, and a degrading score
is a candidate marker of semantic decline.

KEY FINDING (this first step) — WHICH channel is the right marker:
Early semantic decline (Alzheimer's / semantic dementia) loses *distinctive* features first: concepts blur
toward their category prototype (you keep "it's an animal", lose which animal). Within a category, that is
an isotropic SHRINK of the spread — and the cosine-ANGULAR alignment is provably invariant to scale, so it
is nearly BLIND to it. The signature is a DISPERSION (magnitude) phenomenon. The monitor's two channels map
onto this exactly: the angular channel misses the blur; the WITHIN-CATEGORY DISPERSION channel tracks it,
falling to ~(1-collapse), while ordinary rater noise *raises* dispersion (specificity).

HONEST framing (mechanism/specificity foundation, NOT a clinical claim):
 - Normative healthy geometry = real human ratings (Binder 2016, 65 experiential features).
 - Decline MODEL grounded in the literature: collapse toward category prototype.
 - Clinical SENSITIVITY (real patients, e.g. DementiaBank) is the gated next step, not claimed here.
Uses the shipped package: `from styxx.meaning_integrity import ...`.
"""
import os, sys
import numpy as np
import pandas as pd
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.abspath(__file__))
BINDER = os.path.join(HERE, "..", "en", "binder_data", "WordSet1_Ratings.xlsx")
from styxx.meaning_integrity import MeaningReference, meaning_alignment, meaning_dispersion

rng = np.random.default_rng(0)


def within_cat_dispersion_ratio(subject, normative, cats, min_sz=5):
    """Mean WITHIN-category spread of the subject vs the normative. Early decline blurs concepts toward
    their category prototype -> within-category spread shrinks -> ratio falls below 1. This is the
    DISPERSION channel applied within categories; the scale-invariant angular alignment is blind to it."""
    ratios = []
    for c in set(cats):
        idx = np.where(cats == c)[0]
        if len(idx) < min_sz:
            continue
        ratios.append(meaning_dispersion(subject[idx]) / (meaning_dispersion(normative[idx]) + 1e-9))
    return float(np.mean(ratios))


def main():
    df = pd.read_excel(BINDER)
    words = [str(w).strip().lower() for w in df["Word"]]
    H = df.iloc[:, 5:70].apply(pd.to_numeric, errors="coerce").values
    cats = pd.factorize(df["Category"].astype(str))[0]
    keep = ~np.isnan(H).any(1)
    H, cats, words = H[keep], cats[keep], [w for w, k in zip(words, keep) if k]
    normative = H
    ref = MeaningReference(normative, words=words, name="binder65_normative")
    n = len(words)
    print(f"normative human geometry: {n} concepts x 65 features (Binder), {len(set(cats))} categories\n")

    def healthy_subject(noise):
        return normative + noise * rng.standard_normal(normative.shape)

    def declining_subject(f):
        out = normative.copy().astype(float)
        for c in set(cats):
            idx = cats == c
            out[idx] = (1 - f) * normative[idx] + f * normative[idx].mean(0)
        return out

    print(f"{'condition':24s} {'ANGULAR align':>14} {'WITHIN-CAT dispersion':>22}")
    print("-" * 62)
    print("SPECIFICITY — healthy rater variation must NOT look like decline:")
    for label, noise in [("  noise 0.3", 0.3), ("  noise 0.6", 0.6), ("  noise 1.0", 1.0)]:
        s = healthy_subject(noise)
        print(f"{label:24s} {meaning_alignment(s, ref):>+14.3f} {within_cat_dispersion_ratio(s, normative, cats):>22.3f}")

    print("\nSENSITIVITY — modeled semantic decline (collapse toward category prototype):")
    rows = []
    for f in [0.0, 0.2, 0.4, 0.6, 0.8]:
        s = declining_subject(f)
        g, d = meaning_alignment(s, ref), within_cat_dispersion_ratio(s, normative, cats)
        rows.append((f, g, d)); print(f"{('  collapse=%.1f' % f):24s} {g:>+14.3f} {d:>22.3f}")

    print("-" * 62)
    adrop = rows[0][1] - rows[-1][1]; ddrop = rows[0][2] - rows[-1][2]
    healthy_disp = within_cat_dispersion_ratio(healthy_subject(1.0), normative, cats)
    print(f"\n  healthy->severe-decline drop:  ANGULAR {adrop:+.3f}   DISPERSION {ddrop:+.3f}")
    print(f"  dispersion under severe decline {rows[-1][2]:.2f} vs under heavy healthy noise {healthy_disp:.2f}")
    print(f"  -> the WITHIN-CATEGORY DISPERSION channel is the marker: it collapses to ~{rows[-1][2]:.2f} under")
    print(f"     decline but stays >=1 under healthy noise. The angular channel is scale-invariant and blind")
    print(f"     to the blur. Same principle as the machine-side blind-spot — load-bearing for the human side.")
    print(f"     Clinical sensitivity on real patients (DementiaBank) = the gated next step.")


if __name__ == "__main__":
    main()
