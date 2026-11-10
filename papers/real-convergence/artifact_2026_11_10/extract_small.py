"""extract_small.py: per-template last-token states at the lab's fixed layer, small cached models, CPU, offline.

Lead-written, 2026-11-10. Sealed by hash together with PREREG_start_token_artifact_2026_11_10.md and
score_artifact.py BEFORE its first real run. It loads no model that is not already in the local cache, makes no
network request (offline flags set before any import), uses the CPU only, and never touches another process.

What it stores, per model and per word set: the last-token hidden state after `fl` layers for each of the lab's
8 templates (array words x templates x d, float32), and the token length of each input. The lab's June runner
stored only the mean over the 8 templates.

Layer convention: the lab reads `hidden_states[fl]` with fl = max(1, int(0.66 * n_layers)), which is the
residual stream after `fl` decoder layers, before the final norm. Here that state is captured by a forward
hook on decoder layer index fl-1, and the layers after it are dropped to save memory (they cannot change it).
A self-check on the first model compares hook, truncated model and `hidden_states[fl]` and stops the run on
any difference.

Usage:
  py -3.12 -X utf8 -I -B extract_small.py <kit_dir> <out_dir> --only <name>            real run, one model
  py -3.12 -X utf8 -I -B extract_small.py <kit_dir> <out_dir> --only <name> --smoke    two words outside both sets, shapes only

One model per process: memory freed by one model was not returned to the system inside a single process
(smoke run, 2026-11-10: 6.5, 5.7, 4.9 GB available before the three loads). The smallest model runs first
and carries the self-check; the driver stops if it fails.
"""
import ctypes
import gc
import json
import os
import sys
import time

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np  # noqa: E402
import torch  # noqa: E402
import transformers  # noqa: E402
from transformers import AutoModel, AutoTokenizer  # noqa: E402

KIT, OUTDIR = sys.argv[1], sys.argv[2]
SMOKE = "--smoke" in sys.argv
ONLY = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else None
sys.path.insert(0, KIT)
import reproduce as kit  # noqa: E402  (word sets, templates; the kit copy of the lab's method)

# (display name, hub repo, pinned revision or None for the single cached snapshot, GB that must be available)
RUN = [
    ("Qwen2.5-0.5B", "Qwen/Qwen2.5-0.5B-Instruct", None, 3.0),
    ("Llama-3.2-1B", "meta-llama/Llama-3.2-1B-Instruct", "9213176726f574b556790deb65791e0c5aa438b6", 5.0),
    ("Qwen2.5-1.5B", "Qwen/Qwen2.5-1.5B-Instruct", "989aa7980e4cf806f80c7fef2b1adb7bc71aa306", 5.5),
]
SELF_CHECK_PROBES = ["dog", "a dog", "the strawberry", "look at that helicopter"]  # original set only
SMOKE_WORDS = ["zebra", "notebook"]  # in neither word set


class _MemStat(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


def avail_gb():
    s = _MemStat()
    s.dwLength = ctypes.sizeof(_MemStat)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s))
    return s.ullAvailPhys / 2 ** 30


def snapshot_dir(repo, rev):
    base = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub",
                        "models--" + repo.replace("/", "--"), "snapshots")
    if rev:
        p = os.path.join(base, rev)
        if not os.path.isdir(p):
            raise FileNotFoundError("pinned revision is not in the cache")
        return p, rev
    ds = sorted(d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d)))
    if len(ds) != 1:
        raise RuntimeError("expected exactly one cached snapshot, found %d" % len(ds))
    return os.path.join(base, ds[0]), ds[0]


def layer_out(o):
    return o[0] if isinstance(o, (tuple, list)) else o


@torch.no_grad()
def last_token_state(mdl, tok, text, cap):
    ids = tok(text, return_tensors="pt").input_ids
    cap.clear()
    mdl(input_ids=ids, use_cache=False)
    return cap[0][0, -1].float().numpy().copy(), int(ids.shape[1])


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    try:  # below-normal priority: the box runs a collector that must not be starved
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x00004000)
    except Exception:
        pass
    torch.set_num_threads(6)
    log = {"script": "extract_small.py", "smoke": SMOKE, "torch": torch.__version__,
           "transformers": transformers.__version__, "threads": torch.get_num_threads(),
           "device": "cpu", "templates": kit.TEMPLATES, "models": []}
    assert kit.TEMPLATES[0] == "{w}", "the bare-word template must be template index 0"
    sets = {"original": kit.flatten(kit.ORIGINAL), "fresh": kit.flatten(kit.FRESH)}
    for words, _ in sets.values():
        assert not (set(words) & set(SMOKE_WORDS)), "smoke words must be outside both sets"
    if ONLY is None or ONLY not in [r[0] for r in RUN]:
        print("REFUSED: pass --only with one of: " + ", ".join(r[0] for r in RUN), flush=True)
        return 2
    log["only"] = ONLY
    for name, repo, rev, need in [r for r in RUN if r[0] == ONLY]:
        first = (name == RUN[0][0])
        rec = {"name": name, "repo": repo, "avail_gb_before": round(avail_gb(), 2), "need_gb": need}
        if rec["avail_gb_before"] < need:
            rec["status"] = "REFUSED: not enough free memory"
            log["models"].append(rec)
            print("%-13s REFUSED: %.1f GB available, %.1f needed" % (name, rec["avail_gb_before"], need), flush=True)
            continue
        t0 = time.time()
        try:
            snap, used_rev = snapshot_dir(repo, rev)
            tok = AutoTokenizer.from_pretrained(snap)
            try:
                mdl = AutoModel.from_pretrained(snap, torch_dtype="auto", low_cpu_mem_usage=True)
            except ImportError:
                mdl = AutoModel.from_pretrained(snap, torch_dtype="auto")
        except Exception as e:
            rec["status"] = "REFUSED: load failed: %s" % type(e).__name__
            log["models"].append(rec)
            print("%-13s REFUSED: load failed (%s)" % (name, type(e).__name__), flush=True)
            continue
        mdl.eval()
        L = int(mdl.config.num_hidden_layers)
        fl = max(1, int(0.66 * L))
        stored_dtype = str(next(mdl.parameters()).dtype)
        cap = []
        if first:
            # self-check on the first (smallest) model, full float32: hook == hidden_states[fl] == truncated model
            # (runs in smoke mode too: it uses four original-set probes and prints one difference, no outcome)
            mdl = mdl.float()
            h = mdl.layers[fl - 1].register_forward_hook(lambda m, i, o: cap.append(layer_out(o).detach()))
            worst = 0.0
            ref = []
            with torch.no_grad():
                for p in SELF_CHECK_PROBES:
                    ids = tok(p, return_tensors="pt").input_ids
                    cap.clear()
                    out = mdl(input_ids=ids, use_cache=False, output_hidden_states=True)
                    a = out.hidden_states[fl][0, -1].float()
                    b = cap[0][0, -1].float()
                    worst = max(worst, float((a - b).abs().max() / (a.abs().max() + 1e-9)))
                    ref.append(a.clone())
            mdl.layers = torch.nn.ModuleList(list(mdl.layers)[:fl])
            gc.collect()
            with torch.no_grad():
                for p, a in zip(SELF_CHECK_PROBES, ref):
                    ids = tok(p, return_tensors="pt").input_ids
                    cap.clear()
                    mdl(input_ids=ids, use_cache=False)
                    b = cap[0][0, -1].float()
                    worst = max(worst, float((a - b).abs().max() / (a.abs().max() + 1e-9)))
            log["self_check_worst_relative_difference"] = worst
            print("self-check (hook, truncation, hidden_states[fl]) worst relative difference: %.3g" % worst, flush=True)
            if worst > 1e-5:
                log["status"] = "REFUSED: self-check failed"
                json.dump(log, open(os.path.join(OUTDIR, "extract_log__%s__SELFCHECK_FAILED.json" % name), "w", encoding="utf-8"), indent=1)
                print("REFUSED: self-check failed; nothing extracted", flush=True)
                return 3
        else:
            mdl.layers = torch.nn.ModuleList(list(mdl.layers)[:fl])
            gc.collect()
            mdl = mdl.float()
            h = mdl.layers[fl - 1].register_forward_hook(lambda m, i, o: cap.append(layer_out(o).detach()))
        d = int(mdl.config.hidden_size)
        rec.update({"revision": used_rev, "n_layers": L, "fixed_layer": fl, "layers_kept": fl,
                    "stored_dtype": stored_dtype, "run_dtype": str(next(mdl.parameters()).dtype), "hidden": d})
        if SMOKE:
            for w in SMOKE_WORDS:
                for t in kit.TEMPLATES[:2]:
                    v, n = last_token_state(mdl, tok, t.format(w=w), cap)
                    assert v.shape == (d,) and np.isfinite(v).all()
            rec["status"] = "SMOKE OK: shapes only"
            print("%-13s SMOKE OK  L=%d fl=%d d=%d stored=%s  %.0f s" % (name, L, fl, d, stored_dtype, time.time() - t0), flush=True)
        else:
            for sname, (words, cats) in sets.items():
                states = np.zeros((len(words), len(kit.TEMPLATES), d), dtype=np.float32)
                seqlen = np.zeros((len(words), len(kit.TEMPLATES)), dtype=np.int32)
                for i, w in enumerate(words):
                    for j, t in enumerate(kit.TEMPLATES):
                        states[i, j], seqlen[i, j] = last_token_state(mdl, tok, t.format(w=w), cap)
                if not np.isfinite(states).all():
                    raise RuntimeError("non-finite state for %s on %s" % (name, sname))
                path = os.path.join(OUTDIR, "states__%s__%s.npz" % (sname, name))
                if os.path.exists(path):
                    raise FileExistsError("refusing to overwrite " + path)
                np.savez(path, states=states, seq_len=seqlen, words=np.array(words), cats=np.array(cats))
                if not os.path.exists(path):
                    raise RuntimeError("write failed: " + path)
            rec["status"] = "OK"
            print("%-13s OK  L=%d fl=%d d=%d stored=%s  %.0f s" % (name, L, fl, d, stored_dtype, time.time() - t0), flush=True)
        rec["seconds"] = round(time.time() - t0, 1)
        log["models"].append(rec)
        h.remove()
        del mdl, tok
        gc.collect()
    log["status"] = "DONE"
    refused = any(str(m.get("status", "")).startswith("REFUSED") for m in log["models"])
    fname = "extract_log%s__%s.json" % ("_smoke" if SMOKE else "", ONLY)
    json.dump(log, open(os.path.join(OUTDIR, fname), "w", encoding="utf-8"), indent=1)
    print("wrote", fname, flush=True)
    return 4 if refused else 0


if __name__ == "__main__":
    sys.exit(main())
