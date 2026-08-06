"""H1a — cross-subject neural alignment, per PREREG_h1a_human_alignment_2026_08_06.md.

Extracts each NSD subject's nsdgeneral betas for the shared1000 images (averaged over that
subject's repetitions), intersects to images present for ALL subjects, and runs the shipped
`styxx.islands.survey` with defaults unchanged. No subject is dropped for any reason.

    python run_h1a.py --data <dir with betas_all_subj0*.hdf5, shared1000.npy, COCO_73k_subj_indices.hdf5>
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
from styxx.islands import survey                # noqa: E402

SUBJECTS = [f"subj0{i}" for i in range(1, 9)]


def subject_matrix(data: Path, subj: str, shared_mask: np.ndarray, idx: np.ndarray):
    """(coco_image_id -> mean beta vector) for this subject's shared-image trials."""
    is_shared = shared_mask[idx]
    rows = np.where(is_shared)[0]
    coco = idx[rows]
    with h5py.File(data / f"betas_all_{subj}_fp32_renorm.hdf5", "r") as f:
        order = np.argsort(rows)                      # h5py needs increasing selection
        betas = f["betas"][rows[order], :]
    coco = coco[order]
    out = {}
    for c in np.unique(coco):
        out[int(c)] = betas[coco == c].mean(0)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    data = Path(a.data)
    t0 = time.time()

    shared_mask = np.load(data / "shared1000.npy")
    with h5py.File(data / "COCO_73k_subj_indices.hdf5", "r") as h:
        indices = {s: np.asarray(h[s]) for s in SUBJECTS}

    per_subj = {}
    for s in SUBJECTS:
        per_subj[s] = subject_matrix(data, s, shared_mask, indices[s])
        print(f">> {s}: {len(per_subj[s])} shared images [{time.time()-t0:.0f}s]", flush=True)

    common = sorted(set.intersection(*(set(v) for v in per_subj.values())))
    if a.smoke:
        common = common[:40]
    reps = {s: np.stack([per_subj[s][c] for c in common]).astype(float) for s in SUBJECTS}
    for s in SUBJECTS:                                # per-voxel z within subject (prereg step 3)
        X = reps[s]
        reps[s] = (X - X.mean(0)) / (X.std(0) + 1e-9)
    print(f"cohort: {len(reps)} subjects x {len(common)} shared images; "
          f"voxels {[reps[s].shape[1] for s in SUBJECTS]}", flush=True)

    s_ = survey(reps, k=20, n_null=100 if a.smoke else 1000,
                n_perm=100 if a.smoke else 1000, seed=343)
    print(s_, flush=True)

    res = {"prereg": "PREREG_h1a_human_alignment_2026_08_06.md", "smoke": a.smoke,
           "source": "huggingface.co/datasets/pscotti/mindeyev2 (NSD nsdgeneral betas)",
           "n_members": len(reps), "n_shared_images": len(common),
           "n_trials_per_subject": {s: int(len(indices[s])) for s in SUBJECTS},
           "n_voxels_per_subject": {s: int(reps[s].shape[1]) for s in SUBJECTS},
           "members": s_.members, "pairwise": s_.pairwise,
           "mean_affinity": s_.mean_affinity, "islands": s_.islands,
           "island_rule": s_.island_rule, "null": s_.null,
           "bimodality_p": s_.bimodality_p, "cohort_median": s_.cohort_median,
           "survey_verdict": s_.verdict, "caveats": s_.caveats,
           "median_minus_null_p95": round(s_.cohort_median - s_.null["p95"], 4)}
    try:
        from styxx.protocol import Experiment
        v = Experiment(HERE / "PREREG_h1a_human_alignment_2026_08_06.md").score(res, smoke=a.smoke)
        res["verdict"], res["gates"] = v.verdict, v.gates
        res["prereg_commit"] = v.prereg_commit
    except Exception as e:
        res["verdict"] = f"UNSCORED__{type(e).__name__}: {e}"
    (HERE / f"h1a_result{'_smoke' if a.smoke else ''}.json").write_text(
        json.dumps(res, indent=2) + "\n", encoding="utf-8")
    print(f"\nVERDICT: {res['verdict']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
