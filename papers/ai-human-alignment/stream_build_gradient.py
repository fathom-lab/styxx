# -*- coding: utf-8 -*-
"""TIER 2 gradient build — re-stream the 26.6 GB THINGS-fMRI nifti and extract 823-concept patterns across the
ventral HIERARCHY: V1, V2, V3, hV4 (retinotopic, from resampled_varea) + LOC + union-ventral (high-level). For
the visual->semantic gradient: vision-controlled LLM<->brain MEANING should climb up the hierarchy (collapse in
V1, survive high-level). Disk-safe streaming (one 75 MB volume at a time). Saves things_gradient_patterns.npz."""
import sys, os, time, socket, urllib.request, tarfile, ssl, json
socket.setdefaulttimeout(180)
import numpy as np, nibabel as nib
from pathlib import Path
HERE = Path(__file__).resolve().parent; TF = HERE / "things_fmri"
ROIBASE = TF / "rois" / "rois" / "category_localizer"; PRF = TF / "rois" / "rois" / "prf"
ctx = ssl.create_default_context(); ctx.check_hostname = False; ctx.verify_mode = ssl.CERT_NONE
TMP = TF / "_stream_tmp_grad.nii.gz"
concept_of = lambda fn: fn.split("/")[0].strip().lower()
ROIS = ["V1", "V2", "V3", "hV4", "LOC", "ventral"]

mask_cache = {}
def masks_for(sub):
    if sub in mask_cache: return mask_cache[sub]
    va = np.asarray(nib.load(PRF / sub / "resampled_varea.nii.gz").dataobj)
    cat = ROIBASE / sub
    ventral = loc = None
    for f in cat.rglob("*.nii.gz"):
        d = np.asarray(nib.load(f).dataobj) > 0
        ventral = d if ventral is None else (ventral | d)
        if "LOC" in f.name: loc = d if loc is None else (loc | d)
    m = {"V1": va == 1, "V2": va == 2, "V3": va == 3, "hV4": va == 4, "LOC": loc, "ventral": ventral}
    mask_cache[sub] = m
    print(f"  [{sub}] " + " ".join(f"{k}:{int(v.sum())}" for k, v in m.items()), flush=True)
    return m

acc = {}; pend = {}; conds = {}
def key_of(name):
    p = name.split("/")[-1]; parts = p.split("_"); return (parts[0], parts[1], parts[2])
def accumulate(sub, roi, masked, cl):
    a = acc.setdefault((sub, roi), {})
    for t, c in enumerate(cl):
        if c in a: a[c][0] += masked[:, t]; a[c][1] += 1
        else: a[c] = [masked[:, t].astype(np.float64), 1]
def resolve(k):
    if k in pend and k in conds:
        for roi, md in pend.pop(k).items(): accumulate(k[0], roi, md, conds[k])

t0 = time.time(); nb = nc = 0
d = json.load(urllib.request.urlopen("https://api.figshare.com/v2/articles/20590140/files", context=ctx, timeout=60))
url = [f for f in d if f["name"].endswith(".tar.gz")][0]["download_url"]
print(f"streaming {url}", flush=True)
resp = urllib.request.urlopen(url, context=ctx, timeout=120)
tf = tarfile.open(fileobj=resp, mode="r|gz")
for m in tf:
    if not m.isfile(): continue
    if m.name.endswith("_conditions.tsv"):
        k = key_of(m.name); lines = tf.extractfile(m).read().decode("utf-8", "replace").splitlines()[1:]
        conds[k] = [concept_of(ln.split("\t")[1]) for ln in lines if "\t" in ln]; resolve(k); nc += 1
    elif m.name.endswith("_betas.nii.gz"):
        k = key_of(m.name); sub = k[0]
        if not (ROIBASE / sub).exists(): continue
        with open(TMP, "wb") as fh: fh.write(tf.extractfile(m).read())
        data = nib.load(TMP).get_fdata(dtype=np.float32); mk = masks_for(sub)
        masked = {roi: data[mk[roi]] for roi in mk}; del data
        if k in conds:
            for roi, md in masked.items(): accumulate(sub, roi, md, conds[k])
        else: pend[k] = masked
        nb += 1
        if nb % 36 == 0: print(f"  ...{nb} runs, {nc} cond, pend {len(pend)}, {time.time()-t0:.0f}s", flush=True)
resp.close()
if TMP.exists(): TMP.unlink()
print(f"stream done: {nb} runs, {time.time()-t0:.0f}s", flush=True)

subs = ["sub-01", "sub-02", "sub-03"]; save = {}
for roi in ROIS:
    persub = {s: acc.get((s, roi), {}) for s in subs}
    common = sorted(set.intersection(*[set(persub[s].keys()) for s in subs]))
    save[f"{roi}_concepts"] = np.array(common)
    for s in subs:
        save[f"{roi}_{s}"] = np.stack([persub[s][c][0] / persub[s][c][1] for c in common]).astype(np.float32)
    print(f"[{roi}] {len(common)} concepts", flush=True)
np.savez(TF / "things_gradient_patterns.npz", **save)
print("wrote things_gradient_patterns.npz", flush=True)
