"""extract_v2.py: per-template last-token states at the lab's fixed layer for models too large to hold in memory.

Lead-written, 2026-11-10. Sealed by hash with PREREG_corrected_geometry_five_models_2026_11_10.md and
score_v2.py before its first run on fresh words. CPU only, float32 arithmetic, offline, no downloads.

How it differs from extract_small.py (the sealed first test):
- The decoder weights are never all in memory. The model is built from its config with ONE decoder layer. The
  first `fl` layers of the real model are then run through that one layer object: before each call the layer's
  weights are read from the cached safetensors file (stored dtype, upcast exactly to float32) and copied in.
  The model's own forward pass still builds the masks and the rotary positions; only the weights move.
- Inputs are batched, one batch per template (96 words), padded on the right. With a causal mask a real token
  never attends to a pad on its right, so its state equals the unpadded state up to rounding.
- The state is captured by a hook on the last of the `fl` layer calls: the residual stream after `fl` layers,
  before the final norm. This is `hidden_states[fl]` of the full model, as the lab's June code read it.

Checks inside this file, each a refusal:
- the decoder layer holds nothing but parameters (no buffers), the config has one layer type, every tensor of
  every layer used is in the weights file under the expected name and shape;
- a pad check on ORIGINAL words: for two templates the shortest inputs are run again as their own batch, with
  no padding at all, and must give the same states as their rows in the padded batch (1e-4 relative);
- word order equals the kit's and the token-count file's.

Two more controls are scored by score_v2.py: the states of the three models of the first test must match the
first test's saved states, and the states of the three larger models must match the lab's stored June arrays.

Usage:
  py -3.12 -X utf8 -I -B extract_v2.py <kit_dir> <out_dir> --only <name>            the sealed run: both sets
  py -3.12 -X utf8 -I -B extract_v2.py <kit_dir> <out_dir> --only <name> --smoke    pad check, then two words that
                                                                                   are in neither set, shapes only
"""
import copy
import ctypes
import gc
import glob
import hashlib
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
from safetensors import safe_open  # noqa: E402
from transformers import AutoConfig, AutoModel, AutoTokenizer  # noqa: E402

KIT, OUTDIR = sys.argv[1], sys.argv[2]
ONLY = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else None
SMOKE = "--smoke" in sys.argv
sys.path.insert(0, KIT)
import reproduce as kit  # noqa: E402

# (display name, hub repo, pinned revision or None for the single cached snapshot, GB that must be available)
RUN = [
    ("Qwen2.5-0.5B", "Qwen/Qwen2.5-0.5B-Instruct", None, 2.0),
    ("Llama-3.2-1B", "meta-llama/Llama-3.2-1B-Instruct", "9213176726f574b556790deb65791e0c5aa438b6", 3.0),
    ("Qwen2.5-1.5B", "Qwen/Qwen2.5-1.5B-Instruct", "989aa7980e4cf806f80c7fef2b1adb7bc71aa306", 3.0),
    ("Qwen2.5-3B", "Qwen/Qwen2.5-3B-Instruct", "aa8e72537993ba99e69dfaafa59ed015b17504d1", 4.0),
    ("Llama-3.2-3B", "meta-llama/Llama-3.2-3B-Instruct", "0cb88a4f764b7a12671c53f0838cd831a0843b95", 4.5),
    ("Phi-3.5-mini", "microsoft/Phi-3.5-mini-instruct", "2fe192450127e6a83f7441aef6e3ca586c338b77", 3.5),
]
SMOKE_WORDS = ["zebra", "notebook"]      # in neither word set
PAD_CHECK_TEMPLATES = (1, 4)             # "a {w}" and "there is a {w}"
PAD_CHECK_TOL = 1e-4


class Refused(Exception):
    pass


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


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def snapshot_dir(repo, rev):
    base = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub",
                        "models--" + repo.replace("/", "--"), "snapshots")
    if rev:
        p = os.path.join(base, rev)
        if not os.path.isdir(p):
            raise Refused("pinned revision is not in the cache")
        return p, rev
    ds = sorted(d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d)))
    if len(ds) != 1:
        raise Refused("expected exactly one cached snapshot, found %d" % len(ds))
    return os.path.join(base, ds[0]), ds[0]


class Weights:
    """reads single tensors from the cached safetensors shard(s); nothing is held between calls"""

    def __init__(self, snap):
        self.where, self.handles = {}, {}
        files = sorted(glob.glob(os.path.join(snap, "*.safetensors")))
        if not files:
            raise Refused("no safetensors file in the snapshot")
        for f in files:
            h = safe_open(f, framework="pt")
            self.handles[f] = h
            for k in h.keys():
                self.where[k] = f
        self.n_files = len(files)
        self.prefix = "model." if "model.embed_tokens.weight" in self.where else ""
        if self.prefix + "embed_tokens.weight" not in self.where:
            raise Refused("embedding tensor not found under an expected name")
        self.stored_dtype = str(self.handles[self.where[self.prefix + "embed_tokens.weight"]]
                                .get_slice(self.prefix + "embed_tokens.weight").get_dtype())

    def get(self, name):
        key = self.prefix + name
        return self.handles[self.where[key]].get_tensor(key)

    def layer_keys(self, idx):
        head = "%slayers.%d." % (self.prefix, idx)
        return sorted(k[len(head):] for k in self.where if k.startswith(head))

    def n_layers_in_file(self):
        idx = set()
        for k in self.where:
            if k.startswith(self.prefix + "layers."):
                idx.add(int(k[len(self.prefix) + len("layers."):].split(".")[0]))
        return len(idx)


class JITLayer(torch.nn.Module):
    """one call of the shared decoder layer with the weights of layer `idx` copied in first"""

    def __init__(self, shared, idx, weights, names):
        super().__init__()
        self.shared, self.idx, self.weights, self.names = shared, idx, weights, names

    def __getattr__(self, name):
        # a model's forward may read plain attributes of its layers (Qwen2 reads `attention_type`):
        # anything this wrapper does not have is answered by the real layer
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(super().__getattr__("shared"), name)

    def forward(self, *a, **k):
        with torch.no_grad():
            params = dict(self.shared.named_parameters())
            for n in self.names:
                t = self.weights.get("layers.%d.%s" % (self.idx, n))
                if tuple(t.shape) != tuple(params[n].shape):
                    raise Refused("shape mismatch for layer %d tensor %s" % (self.idx, n))
                params[n].copy_(t.float())
        att = getattr(self.shared, "self_attn", None)
        if att is not None and hasattr(att, "layer_idx"):
            att.layer_idx = self.idx
        return self.shared(*a, **k)


def layer_out(o):
    return o[0] if isinstance(o, (tuple, list)) else o


def build(repo, rev):
    snap, used_rev = snapshot_dir(repo, rev)
    tok = AutoTokenizer.from_pretrained(snap)
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    cfg = AutoConfig.from_pretrained(snap)
    L = int(cfg.num_hidden_layers)
    fl = max(1, int(0.66 * L))
    layer_types = getattr(cfg, "layer_types", None)
    if layer_types and len(set(layer_types)) > 1:
        raise Refused("the config mixes layer types; this script runs every layer through one layer object")
    weights = Weights(snap)
    if weights.n_layers_in_file() != L:
        raise Refused("layer count in the weights file (%d) differs from the config (%d)" % (weights.n_layers_in_file(), L))
    cfg1 = copy.deepcopy(cfg)
    cfg1.num_hidden_layers = 1
    if layer_types:
        cfg1.layer_types = list(layer_types[:1])
    mdl = AutoModel.from_config(cfg1).float().eval()
    with torch.no_grad():
        emb = mdl.get_input_embeddings()
        w = weights.get("embed_tokens.weight")
        if tuple(w.shape) != tuple(emb.weight.shape):
            raise Refused("embedding shape mismatch")
        emb.weight.copy_(w.float())
        del w
    shared = mdl.layers[0]
    names = sorted(n for n, _ in shared.named_parameters())
    if list(shared.named_buffers()):
        raise Refused("the decoder layer holds buffers this script does not load")
    for i in range(fl):
        if weights.layer_keys(i) != names:
            raise Refused("layer %d: the tensors in the weights file are not exactly the layer's parameters" % i)
    mdl.layers = torch.nn.ModuleList([JITLayer(shared, i, weights, names) for i in range(fl)])
    mdl.config.num_hidden_layers = fl
    cap = []
    mdl.layers[fl - 1].register_forward_hook(lambda m, i, o: cap.append(layer_out(o).detach()))
    info = {"revision": used_rev, "n_layers": L, "fixed_layer": fl, "stored_dtype": weights.stored_dtype,
            "weight_files": weights.n_files, "hidden": int(cfg.hidden_size), "model_class": type(mdl).__name__,
            "padding_side": tok.padding_side, "run_dtype": "torch.float32"}
    return mdl, tok, cap, info


def run_batch(mdl, tok, cap, texts):
    """last real token state of each text, and each text's token length"""
    enc = tok(texts, return_tensors="pt", padding=True)
    if not bool((enc.attention_mask[:, 0] == 1).all()):
        raise Refused("padding is not on the right")
    cap.clear()
    with torch.no_grad():
        mdl(input_ids=enc.input_ids, attention_mask=enc.attention_mask, use_cache=False)
    h = cap[0]
    n = enc.attention_mask.sum(1)
    last = h[torch.arange(h.shape[0]), n - 1].float().numpy()
    if not np.isfinite(last).all():
        raise Refused("non-finite state")
    return last, n.numpy()


def pad_check(mdl, tok, cap, words):
    """the most padded rows of a batch must equal the same inputs run with no padding at all"""
    worst, rows = 0.0, 0
    for j in PAD_CHECK_TEMPLATES:
        texts = [kit.TEMPLATES[j].format(w=w) for w in words]
        full, n = run_batch(mdl, tok, cap, texts)
        idx = np.flatnonzero(n == n.min())
        if len(idx) < 2 or len(idx) == len(words):
            raise Refused("pad check needs a mix of lengths in template %d" % j)
        alone, n2 = run_batch(mdl, tok, cap, [texts[i] for i in idx])
        if not (n2 == n.min()).all():
            raise Refused("pad check: the short batch was padded")
        rel = np.linalg.norm(alone - full[idx], axis=1) / (np.linalg.norm(full[idx], axis=1) + 1e-12)
        worst, rows = max(worst, float(rel.max())), rows + len(idx)
    return worst, rows


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    try:
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x00004000)
    except Exception:
        pass
    torch.set_num_threads(6)
    if ONLY is None or ONLY not in [r[0] for r in RUN]:
        print("REFUSED: pass --only with one of: " + ", ".join(r[0] for r in RUN), flush=True)
        return 2
    name, repo, rev, need = [r for r in RUN if r[0] == ONLY][0]
    tag = "smoke" if SMOKE else "run"
    logname = os.path.join(OUTDIR, "extract2_log__%s__%s.json" % (ONLY, tag))
    k = 1
    while os.path.exists(logname):               # never overwrite a log: a refusal must stay on record
        k += 1
        logname = os.path.join(OUTDIR, "extract2_log__%s__%s__%d.json" % (ONLY, tag, k))
    log = {"script": "extract_v2.py", "script_sha256": sha256_file(os.path.abspath(__file__)),
           "kit_reproduce_sha256": sha256_file(os.path.join(KIT, "reproduce.py")),
           "only": ONLY, "name": name, "repo": repo, "mode": tag, "attempt": k,
           "torch": torch.__version__, "transformers": transformers.__version__, "threads": torch.get_num_threads(),
           "device": "cpu", "avail_gb_before": round(avail_gb(), 2), "need_gb": need, "status": "STARTED"}
    t0 = time.time()
    code = 1
    try:
        assert kit.TEMPLATES[0] == "{w}", "the bare-word template must be template index 0"
        toks = json.load(open(os.path.join(KIT, "lex_tokens.json"), encoding="utf-8"))
        sets = {"original": kit.flatten(kit.ORIGINAL), "fresh": kit.flatten(kit.FRESH)}
        for s, (words, cats) in sets.items():
            if list(words) != list(toks[s]["words"]):
                raise Refused("word order of the %s set differs from the token-count file" % s)
            if set(words) & set(SMOKE_WORDS):
                raise Refused("smoke words must be outside both sets")
        if log["avail_gb_before"] < need:
            log["status"] = "REFUSED: not enough free memory"
            print("%-13s REFUSED: %.1f GB available, %.1f needed" % (name, log["avail_gb_before"], need), flush=True)
            return 4
        mdl, tok, cap, info = build(repo, rev)
        log.update(info)
        worst, rows = pad_check(mdl, tok, cap, sets["original"][0])
        log["pad_check_worst_relative_difference"] = worst
        log["pad_check_rows"] = rows
        print("%-13s pad check on original words: worst relative difference %.3g over %d rows" % (name, worst, rows), flush=True)
        if worst > PAD_CHECK_TOL:
            log["status"] = "REFUSED: pad check failed"
            return 5
        d = info["hidden"]
        if SMOKE:
            last, n = run_batch(mdl, tok, cap, [t.format(w=w) for w in SMOKE_WORDS for t in kit.TEMPLATES[:2]])
            assert last.shape == (4, d)
            log["status"] = "SMOKE OK: pad check and shapes only"
            print("%-13s SMOKE OK  class=%s L=%d fl=%d d=%d stored=%s files=%d  %.0f s" % (
                name, info["model_class"], info["n_layers"], info["fixed_layer"], d, info["stored_dtype"], info["weight_files"], time.time() - t0), flush=True)
            return 0
        for sname, (words, cats) in sets.items():
            path = os.path.join(OUTDIR, "states2__%s__%s.npz" % (sname, name))
            if os.path.exists(path):
                raise Refused("refusing to overwrite " + os.path.basename(path))
            states = np.zeros((len(words), len(kit.TEMPLATES), d), dtype=np.float32)
            seqlen = np.zeros((len(words), len(kit.TEMPLATES)), dtype=np.int32)
            for j, t in enumerate(kit.TEMPLATES):
                states[:, j], seqlen[:, j] = run_batch(mdl, tok, cap, [t.format(w=w) for w in words])
            np.savez(path, states=states, seq_len=seqlen, words=np.array(words), cats=np.array(cats))
            if not os.path.exists(path):
                raise Refused("write failed: " + os.path.basename(path))
            log.setdefault("states_sha256", {})[sname] = sha256_file(path)
            print("%-13s %-8s written  %.0f s" % (name, sname, time.time() - t0), flush=True)
        log["status"] = "OK"
        print("%-13s OK  %.0f s" % (name, time.time() - t0), flush=True)
        code = 0
        return 0
    except Refused as e:
        log["status"] = "REFUSED: %s" % e
        print("%-13s REFUSED: %s" % (name, e), flush=True)
        code = 6
        return 6
    except Exception as e:
        log["status"] = "CRASHED: %s" % type(e).__name__
        print("%-13s CRASHED: %s" % (name, type(e).__name__), flush=True)
        raise
    finally:
        log["seconds"] = round(time.time() - t0, 1)
        log["avail_gb_after"] = round(avail_gb(), 2)
        json.dump(log, open(logname, "w", encoding="utf-8"), indent=1)
        gc.collect()


if __name__ == "__main__":
    sys.exit(main())
