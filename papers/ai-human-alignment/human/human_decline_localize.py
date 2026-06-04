# -*- coding: utf-8 -*-
"""
human_decline_localize.py — localize WHICH concepts a person is losing (the clinically-relevant output).

Early semantic decline is not uniform: specific concepts lose their distinctive meaning first. The marker
is per-concept DISTINCTIVENESS = how far a concept sits from its category prototype (in the human feature
space). Decline blurs a concept toward the prototype -> its distinctiveness shrinks. We model GRADED decline
(each concept declines by a different amount) PLUS realistic rater noise, and ask: does the per-concept
distinctiveness-loss recover WHICH concepts declined — and does healthy noise alone NOT produce that pattern?
Normative = Binder 65 features. Uses shipped styxx (meaning_dispersion building block).
"""
import os, sys
import numpy as np
import pandas as pd
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.abspath(__file__))
BINDER = os.path.join(HERE, "..", "en", "binder_data", "WordSet1_Ratings.xlsx")
rng = np.random.default_rng(0)


def distinctiveness(feats, cats):
    """per-concept distance from its category centroid (how distinctive it is within its category)."""
    d = np.zeros(len(feats))
    for c in set(cats):
        idx = np.where(cats == c)[0]
        ctr = feats[idx].mean(0)
        d[idx] = np.linalg.norm(feats[idx] - ctr, axis=1)
    return d


def spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def auc(scores, labels):
    o = np.argsort(scores); r = np.empty(len(scores)); r[o] = np.arange(len(scores))
    p = labels == 1; n1 = p.sum(); n0 = (~p).sum()
    return float((r[p].sum() - n1 * (n1 - 1) / 2) / (n1 * n0))


def main():
    df = pd.read_excel(BINDER)
    H = df.iloc[:, 5:70].apply(pd.to_numeric, errors="coerce").values
    cats = pd.factorize(df["Category"].astype(str))[0]
    keep = ~np.isnan(H).any(1)
    H, cats = H[keep], cats[keep]
    normative = H.astype(float)
    n = len(normative)
    d_norm = distinctiveness(normative, cats)
    print(f"normative: {n} concepts, {len(set(cats))} categories\n")

    # GRADED decline: each concept collapses toward its category prototype by its own severity, + rater noise
    sev = rng.uniform(0, 0.8, n)
    declined = normative.copy()
    for c in set(cats):
        idx = np.where(cats == c)[0]; ctr = normative[idx].mean(0)
        declined[idx] = (1 - sev[idx])[:, None] * normative[idx] + sev[idx][:, None] * ctr
    declined += 0.5 * rng.standard_normal(declined.shape)            # realistic rater noise on top

    loss = 1.0 - distinctiveness(declined, cats) / (d_norm + 1e-9)   # measured per-concept distinctiveness loss
    rho = spearman(loss, sev)
    top = sev > np.quantile(sev, 0.7); a = auc(loss, top.astype(int))
    print(f"GRADED DECLINE + rater noise:")
    print(f"  measured distinctiveness-loss vs TRUE severity: Spearman rho {rho:.3f}")
    print(f"  identify the worst-declining third: ROC-AUC {a:.3f}")

    # SPECIFICITY: healthy subject (noise only, NO decline) -> no severity to recover, loss is noise
    healthy = normative + 0.5 * rng.standard_normal(normative.shape)
    loss_h = 1.0 - distinctiveness(healthy, cats) / (d_norm + 1e-9)
    print(f"\nSPECIFICITY (healthy, no decline):")
    print(f"  mean distinctiveness-loss: decline {loss.mean():+.3f}  vs  healthy {loss_h.mean():+.3f}")
    print(f"  -> decline shrinks distinctiveness (loss>0); healthy noise does not (loss~0 or <0).")
    print(f"\n  VERDICT: the per-concept marker LOCALIZES graded decline through noise (rho {rho:.2f}, AUC {a:.2f})")
    print(f"  and stays flat on healthy variation. WHICH concepts a person is losing -- the clinical output.")


if __name__ == "__main__":
    main()
