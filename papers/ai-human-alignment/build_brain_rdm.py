# -*- coding: utf-8 -*-
"""
build_brain_rdm.py — neural RDMs over the 60 Mitchell-2008 nouns from human fMRI, + noise ceiling.

Per subject: average the 6 presentations per noun -> 60 voxel patterns; select the most STABLE
voxels (cross-presentation consistency, the Mitchell-standard SNR step); RDM = 1 - Pearson
correlation between noun patterns. Group RDM = mean of subject RDMs. Noise ceiling = how well a
single subject's RDM predicts the group (upper) and the other subjects (lower) — the band any
model's RSA-to-brain must be judged against.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
from scipy.io import loadmat

HERE = Path(__file__).resolve().parent
BRAIN = HERE / "brain"


def subject_rdm(path, K=500):
    m = loadmat(path)
    info = m["info"][0]
    data = m["data"]
    n = len(info)
    words = [str(info[i]["word"][0]) for i in range(n)]
    epochs = [int(np.array(info[i]["epoch"]).ravel()[0]) for i in range(n)]
    V = data[0, 0].shape[1]
    X = np.empty((n, V), dtype=np.float32)
    for i in range(n):
        X[i] = data[i, 0][0]
    nouns = sorted(set(words))
    nidx = {w: j for j, w in enumerate(nouns)}
    eps = sorted(set(epochs))
    T = np.full((len(eps), len(nouns), V), np.nan, dtype=np.float32)
    for i in range(n):
        T[eps.index(epochs[i]), nidx[words[i]]] = X[i]
    # per-voxel stability = mean cross-epoch correlation of the 60-noun profile
    pc = T - np.nanmean(T, axis=1, keepdims=True)
    pn = pc / (np.nanstd(T, axis=1, keepdims=True) + 1e-9)
    s = np.zeros(V); cnt = 0
    for a in range(len(eps)):
        for b in range(a + 1, len(eps)):
            s += np.nanmean(pn[a] * pn[b], axis=0); cnt += 1
    stab = s / cnt
    sel = np.argsort(stab)[-K:]
    pat = np.nanmean(T[:, :, sel], axis=0)            # (60, K) mean over presentations
    C = np.corrcoef(pat)                              # (60,60)
    return nouns, 1.0 - C, float(np.mean(stab[sel]))


def group_and_ceiling(K=500):
    paths = sorted(BRAIN.glob("data-science-P*.mat"))
    rdms, nouns0 = [], None
    for p in paths:
        nouns, rdm, ms = subject_rdm(p, K)
        if nouns0 is None:
            nouns0 = nouns
        assert nouns == nouns0
        rdms.append(rdm)
        print(f"  {p.name}: mean stability(top{K}) {ms:.3f}", flush=True)
    rdms = np.array(rdms)                              # (S,60,60)
    group = rdms.mean(0)
    iu = np.triu_indices(len(nouns0), 1)
    # noise ceiling
    up, lo = [], []
    for s in range(len(rdms)):
        others = np.delete(rdms, s, axis=0).mean(0)
        up.append(np.corrcoef(rdms[s][iu], group[iu])[0, 1])
        lo.append(np.corrcoef(rdms[s][iu], others[iu])[0, 1])
    ceiling = (float(np.mean(lo)), float(np.mean(up)))
    return nouns0, group, rdms, ceiling


if __name__ == "__main__":
    nouns, group, rdms, ceiling = group_and_ceiling(500)
    print(f"\n{len(rdms)} subjects, {len(nouns)} nouns")
    print(f"noise ceiling (RSA a model could reach): lower {ceiling[0]:.3f}  upper {ceiling[1]:.3f}")
    iu = np.triu_indices(len(nouns), 1)
    inter = [np.corrcoef(rdms[a][iu], rdms[b][iu])[0, 1] for a in range(len(rdms)) for b in range(a + 1, len(rdms))]
    print(f"mean inter-subject RDM correlation: {np.mean(inter):.3f}")
    np.savez(HERE / "brain_rdm.npz", nouns=np.array(nouns), group=group, ceiling=np.array(ceiling))
    print("wrote brain_rdm.npz")
