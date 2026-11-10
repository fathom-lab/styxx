"""build_anchors_v2.py: write the anchors of the geometry axis version 2 (styxx.mind_geometry), bare-word template dropped.

Written 2026-11-10 with the v2 geometry axis (NOTE_geometry_axis_v2_2026_11_10.md in this folder).

Input: a folder holding the per-template state arrays of the second sealed test of 2026-11-10
(papers/real-convergence/artifact_2026_11_10/extract_v2.py writes them), one file per model,
states2__original__<model>.npz, with keys states (96, 8, d) float32, seq_len (96, 8), words (96), cats (96).
Template 0 is the bare word. The arrays are not in the repository; their sha256 values are in
papers/real-convergence/artifact_2026_11_10/STATES2_SHA256SUMS.txt, and this script checks each input against
that list when the list is present. An input whose sha256 differs from the list stops the script, unless
--any-inputs is given (for states extracted afresh, which will not match bit for bit); an input the list does not
name is reported and used.

Output: papers/mind-instrument/anchors_v2_2026_11_10.npz (or --out), keys fixed__<model> for the five models
below, each float32 (96, d): the mean over templates 1 to 7 (TEMPLATES_V2 of styxx.mind_geometry), taken in float64 and
stored as float32. The zip is written with a fixed timestamp and in a fixed key order, so the same inputs give
the same output bytes. gemma-2-2b has no rebuilt anchor: it was not re-extracted.

Checks, each of which stops the script with exit code 2: the word order and category order of every input equal
styxx.mind.BATTERY and styxx.mind.BATTERY_CATEGORY; the states have shape (96, 8, d); every value is finite; the
template count matches styxx.mind.TEMPLATES; each input's sha256 is the published one (see above). The
script stops before reading any input if styxx.mind or styxx.mind_geometry is loaded from outside this
checkout.

usage: python papers/mind-instrument/build_anchors_v2.py <folder of states2 arrays> [--out PATH] [--any-inputs]
numpy only. No model, no tokenizer, no network.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import sys
import zipfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
MODELS = ["Qwen2.5-1.5B", "Qwen2.5-3B", "Llama-3.2-1B", "Llama-3.2-3B", "Phi-3.5-mini"]
SUMS = REPO / "papers" / "real-convergence" / "artifact_2026_11_10" / "STATES2_SHA256SUMS.txt"
DEFAULT_OUT = HERE / "anchors_v2_2026_11_10.npz"


def load_mind():
    """styxx.mind and styxx.mind_geometry of this checkout, not of an installed styxx.

    The checkout is put at the head of sys.path, and the script stops if either module still comes from elsewhere
    (an installed styxx, or an import hook that serves modules from another checkout).
    """
    sys.path.insert(0, str(REPO))
    import styxx.mind as mind
    import styxx.mind_geometry as mg
    for mod in (mind, mg):
        if not Path(mod.__file__).resolve().is_relative_to(REPO):
            raise SystemExit("REFUSED: %s was loaded from %s, not from this checkout" % (mod.__name__, mod.__file__))
    return mind, mg, Path(mg.__file__).resolve()


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def published_sums() -> dict:
    if not SUMS.exists():
        return {}
    out = {}
    for ln in SUMS.read_text(encoding="utf-8").splitlines():
        parts = ln.strip().split()
        if len(parts) == 2:
            out[parts[1].lstrip("*")] = parts[0]
    return out


def write_npz_fixed(path: Path, arrays: list[tuple[str, np.ndarray]]) -> None:
    """np.savez layout (one stored .npy per key) with a fixed timestamp, so the bytes depend on the arrays only."""
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
        for key, arr in arrays:
            buf = io.BytesIO()
            np.lib.format.write_array(buf, np.ascontiguousarray(arr), allow_pickle=False)
            info = zipfile.ZipInfo(key + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o600 << 16
            zf.writestr(info, buf.getvalue())


def fail(msg: str) -> int:
    print("REFUSED: " + msg)
    return 2


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("states_dir", help="folder holding states2__original__<model>.npz")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="output .npz (default: %(default)s)")
    ap.add_argument("--any-inputs", action="store_true",
                    help="use inputs whose sha256 differs from STATES2_SHA256SUMS.txt (fresh extractions)")
    a = ap.parse_args(argv)
    mind, mg, mg_path = load_mind()
    battery, cats_ref = list(mind.BATTERY), list(mind.BATTERY_CATEGORY)
    n_t = len(mind.TEMPLATES)
    if list(mg.TEMPLATES_V2) != list(mind.TEMPLATES[1:]):
        return fail("styxx.mind_geometry.TEMPLATES_V2 is not styxx.mind.TEMPLATES without the bare word")
    print("styxx/mind_geometry.py sha256 %s" % sha256(mg_path))
    sums = published_sums()
    arrays = []
    for m in MODELS:
        name = "states2__original__%s.npz" % m
        p = Path(a.states_dir) / name
        if not p.exists():
            return fail("%s not found in %s" % (name, a.states_dir))
        digest = sha256(p)
        pub = sums.get(name)
        tag = "no published sum found" if pub is None else ("matches STATES2_SHA256SUMS.txt" if pub == digest
                                                              else "DIFFERS from STATES2_SHA256SUMS.txt")
        print("input  %s  %s  (%s)" % (digest, name, tag))
        if pub is not None and pub != digest and not a.any_inputs:
            return fail("%s: sha256 differs from STATES2_SHA256SUMS.txt (pass --any-inputs to use it anyway)" % name)
        with np.load(p, allow_pickle=False) as z:
            words = [str(w) for w in z["words"]]
            cats = [str(c) for c in z["cats"]]
            st = z["states"]
        if words != battery:
            return fail("%s: word order differs from styxx.mind.BATTERY" % name)
        if cats != cats_ref:
            return fail("%s: category order differs from styxx.mind.BATTERY_CATEGORY" % name)
        if st.ndim != 3 or st.shape[:2] != (len(battery), n_t):
            return fail("%s: states shape %s, expected (%d, %d, d)" % (name, st.shape, len(battery), n_t))
        if not np.isfinite(st).all():
            return fail("%s: non-finite values in states" % name)
        rep7 = st[:, 1:, :].astype(np.float64).mean(axis=1).astype(np.float32)
        arrays.append(("fixed__" + m, rep7))
        print("        fixed__%-13s shape %s, top-direction share %.4f" % (m, rep7.shape, mg.dominant_direction_share(rep7)))
    out = Path(a.out)
    if out.exists():
        print("note: %s exists and is overwritten" % out.name)
    write_npz_fixed(out, arrays)
    with np.load(out, allow_pickle=False) as z:
        for key, arr in arrays:
            if not np.array_equal(z[key], arr):
                return fail("read-back of %s differs from what was written" % key)
    print("output %s  %s" % (sha256(out), out.name))
    return 0


if __name__ == "__main__":
    sys.exit(main())
