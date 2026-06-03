# -*- coding: utf-8 -*-
"""
run_cn_glm2.py SUBJ — fast, COMMON-MASK GLM saving full per-concept beta patterns (for pooling).

Fixes the v1 failure (per-subject voxel selection -> uncorrelated RDMs, inter-subject ceiling ~0):
  - ONE shared brain mask for ALL subjects (same MNI grid) -> betas are in a common voxel space.
  - direct least-squares (one solve per run) instead of 84 contrasts -> ~6x faster.
  - saves the per-concept MEAN beta pattern (671 x K, float16) so pooling + inter-subject voxel
    selection happen across subjects (the powerful, correct way), in pool_and_deflate.py.
"""
import os, re, sys
import numpy as np, pandas as pd, nibabel as nib
from nilearn.glm.first_level import make_first_level_design_matrix
from nilearn.maskers import NiftiMasker
from nilearn.masking import compute_epi_mask

HERE = os.path.dirname(os.path.abspath(__file__))
CONF = ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z", "csf", "white_matter", "global_signal"]
NCON = 672
MASKP = os.path.join(HERE, "common_mask.nii.gz")


def cidx(stim):
    m = re.search(r"word(\d+)", str(stim)); return int(m.group(1)) if m else -1


def main(subj):
    s = f"sub-{subj}"; bdir = os.path.join(HERE, "bold", s); edir = os.path.join(bdir, "events")
    runs = sorted(set(re.search(r"run-(\d+)", f).group(1) for f in os.listdir(bdir) if f.endswith("_bold.nii.gz")), key=int)
    # shared mask: compute once from the first available run, reuse for all subjects
    first_bold = os.path.join(bdir, f"{s}_task-listening_run-{runs[0]}_bold.nii.gz")
    if not os.path.exists(MASKP):
        compute_epi_mask(nib.load(first_bold)).to_filename(MASKP)
        print("computed shared mask", flush=True)
    masker = NiftiMasker(mask_img=MASKP).fit()
    K = int(nib.load(MASKP).get_fdata().sum())
    print(f"{s}: {len(runs)} runs, shared mask {K} voxels", flush=True)

    sum_odd = np.zeros((NCON, K), np.float32); sum_even = np.zeros((NCON, K), np.float32)
    occ = np.zeros(NCON, int)
    for ri, run in enumerate(runs):
        img = nib.load(os.path.join(bdir, f"{s}_task-listening_run-{run}_bold.nii.gz"))
        tr = float(img.header.get_zooms()[3]); nT = img.shape[3]
        ev = pd.read_csv(os.path.join(edir, f"{s}_task-listening_run-{run}_events.tsv"), sep="\t")
        ev = ev.copy(); ev["trial_type"] = ev["stim_file"].apply(lambda x: f"c{cidx(x)}")
        conf = pd.read_csv(os.path.join(bdir, f"{s}_task-listening_run-{run}_desc-confounds.tsv"), sep="\t")
        conf = conf[[c for c in CONF if c in conf.columns]].fillna(0.0)
        ft = np.arange(nT) * tr
        X = make_first_level_design_matrix(ft, ev[["onset", "duration", "trial_type"]], hrf_model="glover",
                                           drift_model="cosine", high_pass=0.01,
                                           add_regs=conf.values, add_reg_names=list(conf.columns))
        Y = masker.transform(img)                      # (nT, K)
        B = np.linalg.lstsq(X.values, Y, rcond=None)[0]  # (n_reg, K)
        for j, name in enumerate(X.columns):
            if re.fullmatch(r"c\d+", str(name)):
                ci = int(name[1:])
                if 0 <= ci < NCON:
                    (sum_odd if occ[ci] % 2 == 0 else sum_even)[ci] += B[j]
                    occ[ci] += 1
        print(f"  [{ri+1}/{len(runs)}] run-{run}", flush=True)

    seen = occ > 0
    mean = ((sum_odd + sum_even) / np.maximum(occ, 1)[:, None]).astype(np.float16)
    co = np.maximum((occ + 1) // 2, 1)[:, None]; ce = np.maximum(occ // 2, 1)[:, None]
    odd = (sum_odd / co).astype(np.float16); even = (sum_even / ce).astype(np.float16)
    np.savez_compressed(os.path.join(HERE, f"betas_{s}.npz"), mean=mean, odd=odd, even=even,
                        concept_index=np.where(seen)[0], n_voxels=K)
    print(f"{s}: {seen.sum()} concepts, saved betas_{s}.npz", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
