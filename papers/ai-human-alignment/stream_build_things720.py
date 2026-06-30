# -*- coding: utf-8 -*-
"""TIER 2 build — stream the 26.6 GB THINGS-fMRI single-trial nifti archive and build 720-concept ROI
patterns, WITHOUT ever landing the full archive (disk-safe on a tight drive). For each per-run betas volume
(75 MB) we mask to high-level visual ROIs (LOC + union ventral parcels), label trials via the run's
conditions.tsv (image_filename folder = concept), and accumulate per-concept mean patterns. Saves
things720_patterns.npz (small). Then the analysis (within-category RSA + decoding) runs separately."""
import sys, os, time, socket, tempfile, urllib.request, tarfile, ssl, json
socket.setdefaulttimeout(180)
import numpy as np, nibabel as nib
from pathlib import Path
HERE = Path(__file__).resolve().parent
TF = HERE / "things_fmri"
ROIBASE = TF / "rois" / "rois" / "category_localizer"
ctx = ssl.create_default_context(); ctx.check_hostname = False; ctx.verify_mode = ssl.CERT_NONE
TMP = TF / "_stream_tmp.nii.gz"
concept_of = lambda fn: fn.split("/")[0].strip().lower()

mask_cache = {}
def masks_for(sub):
    if sub in mask_cache: return mask_cache[sub]
    base = ROIBASE / sub
    union = loc = None
    for f in base.rglob("*.nii.gz"):
        d = np.asarray(nib.load(f).dataobj) > 0
        union = d if union is None else (union | d)
        if "LOC" in f.name:
            loc = d if loc is None else (loc | d)
    m = {"ventral": union, "LOC": loc}
    mask_cache[sub] = m
    print(f"  [{sub}] masks: ventral {int(union.sum())} vox, LOC {int(loc.sum())} vox", flush=True)
    return m

acc = {}        # (sub, roi) -> dict concept -> [sumvec, count]
pend = {}       # (sub,ses,run) -> {roi: (nvox,T) masked}  (betas seen before conditions)
conds = {}      # (sub,ses,run) -> [concept per trial]

def key_of(name):
    p = name.split("/")[-1]                       # sub-01_ses-things05_run-10_betas.nii.gz
    parts = p.split("_"); return (parts[0], parts[1], parts[2])

def accumulate(sub, roi, masked, concept_list):
    a = acc.setdefault((sub, roi), {})
    for t, c in enumerate(concept_list):
        if c in a: a[c][0] += masked[:, t]; a[c][1] += 1
        else: a[c] = [masked[:, t].astype(np.float64), 1]

def resolve_pending(k):
    if k in pend and k in conds:
        sub = k[0]
        for roi, md in pend.pop(k).items():
            accumulate(sub, roi, md, conds[k])

t0 = time.time(); nb = nc = 0
d = json.load(urllib.request.urlopen("https://api.figshare.com/v2/articles/20590140/files", context=ctx, timeout=60))
url = [f for f in d if f["name"].endswith(".tar.gz")][0]["download_url"]
print(f"streaming {url}", flush=True)
resp = urllib.request.urlopen(url, context=ctx, timeout=120)
tf = tarfile.open(fileobj=resp, mode="r|gz")
for m in tf:
    if not m.isfile(): continue
    if m.name.endswith("_conditions.tsv"):
        k = key_of(m.name)
        lines = tf.extractfile(m).read().decode("utf-8", "replace").splitlines()[1:]
        conds[k] = [concept_of(ln.split("\t")[1]) for ln in lines if "\t" in ln]
        resolve_pending(k); nc += 1
    elif m.name.endswith("_betas.nii.gz"):
        k = key_of(m.name); sub = k[0]
        if not (ROIBASE / sub).exists(): continue          # only the 3 fMRI subjects have ROIs
        with open(TMP, "wb") as fh: fh.write(tf.extractfile(m).read())
        data = nib.load(TMP).get_fdata(dtype=np.float32)    # (X,Y,Z,T)
        mk = masks_for(sub)
        masked = {roi: data[mk[roi]] for roi in mk}         # each (nvox, T)
        del data
        if k in conds:
            for roi, md in masked.items(): accumulate(sub, roi, md, conds[k])
        else:
            pend[k] = masked
        nb += 1
        if nb % 24 == 0:
            print(f"  ...{nb} betas runs, {nc} cond files, pend {len(pend)}, {time.time()-t0:.0f}s", flush=True)
resp.close()
if TMP.exists(): TMP.unlink()
print(f"stream done: {nb} betas runs, {nc} cond files, {time.time()-t0:.0f}s", flush=True)

# build patterns: concept mean = sum/count; keep concepts present in ALL 3 subjects per ROI
subs = ["sub-01", "sub-02", "sub-03"]; save = {}
for roi in ["ventral", "LOC"]:
    persub = {s: acc.get((s, roi), {}) for s in subs}
    common = sorted(set.intersection(*[set(persub[s].keys()) for s in subs]))
    print(f"[{roi}] concepts common to all subjects: {len(common)}", flush=True)
    save[f"{roi}_concepts"] = np.array(common)
    for s in subs:
        M = np.stack([persub[s][c][0] / persub[s][c][1] for c in common])  # (n_concepts, nvox)
        save[f"{roi}_{s}"] = M.astype(np.float32)
np.savez(TF / "things720_patterns.npz", **save)
print("wrote things720_patterns.npz keys:", list(save.keys()), flush=True)
