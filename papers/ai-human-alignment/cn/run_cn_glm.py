# -*- coding: utf-8 -*-
"""
run_cn_glm.py SUBJ — single-trial(concept) GLM -> per-concept brain RDM for one ds004301 subject.

Per run: nilearn FirstLevelModel with one regressor per concept present (84), HRF-convolved, denoised
by fMRIPrep confounds (6 motion + csf/wm/global). Each concept's effect-size map -> a masked vector.
A concept appears in 6 runs (6 reps); we accumulate odd/even-occurrence sums for a SPLIT-HALF
reliability (the positive control) and select the most stable voxels, then RDM = 1 - corr over concepts.
Saves cn/rdm_sub-SUBJ.npz (rdm, reliability, n_concepts, words order). Runs are in MNI (same grid).
"""
import os, re, sys
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn.glm.first_level import FirstLevelModel
from nilearn.maskers import NiftiMasker
from nilearn.masking import compute_epi_mask

HERE = os.path.dirname(os.path.abspath(__file__))
CONF = ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z", "csf", "white_matter", "global_signal"]
NCONCEPTS = 672


def concept_idx(stim):
    m = re.search(r"word(\d+)", str(stim))
    return int(m.group(1)) if m else -1


def main(subj):
    s = f"sub-{subj}"
    bdir = os.path.join(HERE, "bold", s)
    edir = os.path.join(bdir, "events")
    runs = sorted(set(re.search(r"run-(\d+)", f).group(1) for f in os.listdir(bdir) if f.endswith("_bold.nii.gz")), key=int)
    print(f"{s}: {len(runs)} runs", flush=True)

    masker = None
    K = None
    sum_odd = sum_even = None
    cnt_odd = np.zeros(NCONCEPTS); cnt_even = np.zeros(NCONCEPTS)
    occ = np.zeros(NCONCEPTS, dtype=int)

    for ri, run in enumerate(runs):
        bold = os.path.join(bdir, f"{s}_task-listening_run-{run}_bold.nii.gz")
        ev = pd.read_csv(os.path.join(edir, f"{s}_task-listening_run-{run}_events.tsv"), sep="\t")
        ev["trial_type"] = ev["stim_file"].apply(lambda x: f"c{concept_idx(x)}")
        cpath = os.path.join(bdir, f"{s}_task-listening_run-{run}_desc-confounds.tsv")
        conf = pd.read_csv(cpath, sep="\t")
        conf = conf[[c for c in CONF if c in conf.columns]].fillna(0.0)
        img = nib.load(bold)
        tr = float(img.header.get_zooms()[3])
        if masker is None:
            mask = compute_epi_mask(img)
            masker = NiftiMasker(mask_img=mask).fit()
            K = int(mask.get_fdata().sum())
            sum_odd = np.zeros((NCONCEPTS, K)); sum_even = np.zeros((NCONCEPTS, K))
            print(f"  mask: {K} voxels, TR={tr}", flush=True)
        flm = FirstLevelModel(t_r=tr, hrf_model="glover", high_pass=0.01, mask_img=masker.mask_img_,
                              minimize_memory=True, signal_scaling=False, standardize=False)
        flm.fit(img, events=ev[["onset", "duration", "trial_type"]], confounds=conf.reset_index(drop=True))
        for c in sorted(set(ev["trial_type"])):
            ci = int(c[1:])
            if ci < 0 or ci >= NCONCEPTS:
                continue
            beta = np.asarray(masker.transform(flm.compute_contrast(c, output_type="effect_size"))).ravel()
            if occ[ci] % 2 == 0:
                sum_odd[ci] += beta; cnt_odd[ci] += 1
            else:
                sum_even[ci] += beta; cnt_even[ci] += 1
            occ[ci] += 1
        print(f"  [{ri+1}/{len(runs)}] run-{run} done", flush=True)

    seen = occ > 0
    co = np.maximum(cnt_odd, 1)[:, None]; ce = np.maximum(cnt_even, 1)[:, None]
    odd = sum_odd / co; even = sum_even / ce
    mean = (sum_odd + sum_even) / np.maximum(occ, 1)[:, None]
    # voxel stability: cross-half correlation per voxel over concepts
    A = odd[seen]; B = even[seen]
    Az = (A - A.mean(0)) / (A.std(0) + 1e-9); Bz = (B - B.mean(0)) / (B.std(0) + 1e-9)
    stab = (Az * Bz).mean(0)
    topk = np.argsort(stab)[-5000:]

    def rdm(P):
        P = P - P.mean(0); P = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-9)
        return 1.0 - P @ P.T
    R_mean = rdm(mean[np.ix_(seen, topk)])
    R_odd = rdm(odd[np.ix_(seen, topk)]); R_even = rdm(even[np.ix_(seen, topk)])
    iu = np.triu_indices(seen.sum(), 1)
    reliab = float(np.corrcoef(R_odd[iu], R_even[iu])[0, 1])
    print(f"{s}: {seen.sum()} concepts, split-half RDM reliability = {reliab:.3f}, mean voxel stability(top5k) = {stab[topk].mean():.3f}", flush=True)

    np.savez(os.path.join(HERE, f"rdm_{s}.npz"), rdm=R_mean, reliability=reliab,
             concept_index=np.where(seen)[0], n_voxels=len(topk))
    print(f"wrote rdm_{s}.npz", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
