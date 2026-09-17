#!/usr/bin/env python3
"""run_deploy_quant.py — the sealed deploy-scale checksum runs (PREREG_checksum_deploy_quant, 2026-09-13,
the hand-written 48; PREREG_checksum_beacon_draw, 2026-09-14, the 48 drawn from the pool by the block hash
of the slot that sealed it). --prereg selects which one governs and names the outputs.

    # the hand-set run, GPU, after the prereg hash is anchored:
    python papers/checksum/run_deploy_quant.py --model Qwen/Qwen2.5-1.5B
    # the beacon-draw run, after ITS prereg is sealed: the beacon is the one `styxx.clock verify` prints
    # ANCHORED for the earliest memo carrying that prereg's blob digest
    python papers/checksum/run_deploy_quant.py --prereg beacon_draw --beacon <64 hex>
    # pipeline smoke on cpu (no bitsandbytes; dynamic int8 stands in for the two bnb arms;
    # results are NOT the experiment and are written to *_smoke<tag>.* files, the tag appended when given):
    python papers/checksum/run_deploy_quant.py --smoke

Arms: A bf16, A' bf16 reloaded, A'' bf16 reloaded (floor = worst pairwise), Q4 bitsandbytes NF4,
Q8 bitsandbytes LLM.int8, R random init (seed 343). Every number the governing prereg names is written to
<stem>_certs.json — deploy_quant_certs.json under the 2026-09-13 PREREG, beacon_draw_certs.json under the
2026-09-14 one — and papers/checksum/score.py reads them; the RESULT swears to the scorecard and the certs,
not to this printout.

The sealed text, enforced (added 2026-09-14 after the red team found nothing tied the run to it): the
experiment — an untagged, non-smoke run of the PREREG's model — refuses to start unless the governing
PREREG's git blob at HEAD hashes to the digest that PREREG was sealed under (SEALED below, the same
values papers/checksum/score.py freezes and SEALS_2026_09_13.md rows 3 and 6 print), git answers every
question it is asked (a `git status` that fails is not a clean tree), no tracked file differs from HEAD,
no untracked, non-ignored file exists under styxx/ (`git ls-files --others --exclude-standard -- styxx`:
an untracked styxx/checksum/ package would be imported in place of the committed checksum.py), and the
checksum module that was imported is this checkout's styxx/checksum.py. Under --prereg beacon_draw the
experiment also refuses, before any model loads, unless papers/charon/anchors.jsonl carries a
sealed-prereg line for the sealed digest that `styxx.clock.check_line` reads ANCHORED on chain with a
beacon equal to --beacon (beacon_refusal below): a run under any other beacon is never the experiment.
An instrument check is --tag or --smoke; another model is an instrument check only under --tag (an
untagged, non-smoke run of another model is refused). An instrument check records all of it and is not
refused. `provenance.model_revisions` records, per arm, the resolved snapshot commit of the config that
arm loaded (for an arm that loads weights, normally the snapshot they came from; None for a local path;
the random arm R loads only a config, so its entry is that config's commit and no weights are involved).

What this runner binds, so a stranger can tell the sealed run from any other (added 2026-09-13 after
the red team found the run could import a styxx from outside this checkout and record nothing about
it): the styxx package it imported must live inside this checkout or it refuses to start; the certs
carry `provenance` — git HEAD, whether the tree was dirty, the untracked files under styxx/, the sha256
of the PREREG's git blob (the sealed bytes, LF as stored), the path of the checksum module that was
imported (`checksum_file`) and the sha256 of that file (`checksum_py_sha256`), the torch/transformers/
bitsandbytes versions, the CUDA device, CUBLAS_WORKSPACE_CONFIG — and `k1`, the PREREG's first kill
gate, evaluated in code: if the null floor exceeds 1e-2 nats/token no comparison is written, as the
PREREG demands. `distance_params` records n_boot, seed and the floor used. `top1_loss_vs_A` is
hits[A] − hits[arm], the definition the RESULT uses for H2/K3.
"""
import argparse, hashlib, json, os, re, subprocess, sys, time

PREREGS = {"deploy_quant": "PREREG_checksum_deploy_quant_2026_09_13.md",   # the hand-written 48, frozen 2026-09-13
           "beacon_draw": "PREREG_checksum_beacon_draw_2026_09_14.md"}     # the 48 drawn by the seal's block hash
SEALED = {"deploy_quant": "b3b9871090fdcd4491cc4a3171c27e1468414409da9d3b66b4d3ff2fd4b6dd9e",   # SEALS_2026_09_13.md row 3
          "beacon_draw": "d6a98261f44f31664cdedbc24769532d5a701bf46d2c8b9ce30cf6a227fb5519"}    # SEALS_2026_09_13.md row 6


ANCHORS = os.path.join("papers", "charon", "anchors.jsonl")    # the seals on record, relative to the checkout


def _same_file(a, b):
    return os.path.normcase(os.path.realpath(a)) == os.path.normcase(os.path.realpath(b))


def sealed_refusal(prov: dict, prereg_key: str, is_the_experiment: bool, root=None):
    """The reason the experiment may not start under this provenance, or None. Instrument checks are never refused.
    `root` is the checkout whose styxx/checksum.py must be the checksum module that was imported (default: ROOT)."""
    if not is_the_experiment:
        return None
    root = ROOT if root is None else root
    if prov.get("git_head") is None or prov.get("prereg_blob_sha256") is None:
        return "git did not answer: the experiment cannot show which commit and which PREREG text it ran under"
    if not isinstance(prov.get("git_dirty_tracked"), bool):
        return ("git did not answer `git status`: the experiment cannot show that no tracked file differs from HEAD, "
                "and an unanswered status is not a clean tree")
    if not isinstance(prov.get("styxx_untracked"), list):
        return ("git did not answer `git ls-files --others`: the experiment cannot show that no untracked file under styxx/ "
                "is imported in place of the committed code")
    if prov["prereg_blob_sha256"] != SEALED[prereg_key]:
        return (f"{PREREGS[prereg_key]} at HEAD hashes to {prov['prereg_blob_sha256']}, not the sealed digest {SEALED[prereg_key]}; "
                "the experiment runs only under the sealed text")
    if prov["git_dirty_tracked"]:
        return "tracked files differ from HEAD: the experiment runs only on a clean commit (commit or stash, then run)"
    if prov["styxx_untracked"]:
        shown = ", ".join(str(p) for p in prov["styxx_untracked"][:5])
        more = f" and {len(prov['styxx_untracked']) - 5} more" if len(prov["styxx_untracked"]) > 5 else ""
        return (f"untracked files exist under styxx/ ({shown}{more}): an untracked module or package there can be imported "
                "in place of the committed code; the experiment runs only on what the commit carries (remove them, then run)")
    ran, expected = prov.get("checksum_file"), os.path.join(root, "styxx", "checksum.py")
    if not isinstance(ran, str) or not _same_file(ran, expected):
        return (f"the checksum module that was imported is {ran}, not {expected}: the experiment runs only with this "
                "checkout's styxx/checksum.py")
    if not isinstance(prov.get("checksum_py_sha256"), str):
        return f"{ran} could not be read and hashed: the certs could not show which checksum ran"
    return None


def beacon_refusal(beacon: str, anchors_path: str, sealed_digest: str, fetch):
    """The reason the beacon_draw experiment may not run under `beacon`, or None. The experiment's beacon is its seal's:
    `anchors_path` must carry a sealed-prereg line for `sealed_digest` that styxx.clock.check_line, asking the chain
    through `fetch`, reads ANCHORED, and the beacon it reads must equal `beacon`. Every other case refuses: no file,
    no such line, no line ANCHORED (not found, not the creator, an earlier memo, an unanswered RPC), two ANCHORED lines
    that disagree, or another beacon. The PREREG: a run under any other beacon is an instrument check."""
    from styxx import clock
    if not os.path.isfile(anchors_path):
        return (f"{anchors_path} does not exist: no seal of {sealed_digest} is on record, so no beacon is the seal's "
                "(a run under another beacon is an instrument check: pass --tag)")
    lines = []
    try:
        with open(anchors_path, encoding="utf-8") as fh:
            for raw in fh:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    line = json.loads(raw)
                except ValueError:
                    continue
                if isinstance(line, dict) and line.get("kind") == "sealed-prereg" and line.get("digest") == sealed_digest:
                    lines.append(line)
    except (OSError, UnicodeDecodeError) as e:
        return f"{anchors_path} could not be read ({type(e).__name__}: {e}): the seal's beacon cannot be checked"
    if not lines:
        return (f"{anchors_path} has no sealed-prereg line for {sealed_digest}: the PREREG is not sealed on record, "
                "so no beacon is the seal's (a run under another beacon is an instrument check: pass --tag)")
    anchored, seen = {}, []
    for line in lines:
        try:
            r = clock.check_line(line, fetch=fetch)
        except Exception as e:  # a check that could not finish is not ANCHORED
            seen.append(f"tx {line.get('tx')}: check failed ({type(e).__name__}: {e})")
            continue
        seen.append(f"tx {line.get('tx')}: {r.get('status')}")
        if r.get("status") == "ANCHORED" and isinstance(r.get("beacon"), str):
            anchored.setdefault(r["beacon"], []).append(line.get("tx"))
    if not anchored:
        return (f"no sealed-prereg line for {sealed_digest} in {anchors_path} reads ANCHORED on chain ({'; '.join(seen)}): "
                "the seal's beacon is not established")
    if len(anchored) > 1:
        return (f"sealed-prereg lines for {sealed_digest} read ANCHORED with different beacons ({sorted(anchored)}): "
                "the seal's beacon is ambiguous")
    (seal_beacon,) = anchored
    if beacon != seal_beacon:
        return (f"--beacon {beacon} is not the seal's beacon {seal_beacon} (tx {anchored[seal_beacon][0]}, read ANCHORED): "
                "a run under any other beacon is an instrument check and never this experiment (pass --tag)")
    return None


def _parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--tag", default="", help="suffix for the output files of an INSTRUMENT CHECK that is not the "
                    "experiment (e.g. _dryrun_qwen0.5b); the experiment writes untagged files and only for the PREREG's model")
    ap.add_argument("--beacon", default="", help="64-hex beacon (a slot blockhash as styxx.clock reports it): draw the 48 "
                    "canaries from the committed pool instead of the hand set. The 2026-09-13 PREREG froze the hand set, so "
                    "under it a beacon-drawn run is never the experiment and must carry --tag; under --prereg beacon_draw "
                    "the beacon is required and is the block hash of the slot that sealed that PREREG.")
    ap.add_argument("--prereg", choices=sorted(PREREGS), default="deploy_quant",
                    help="which frozen PREREG governs the run and names its outputs: deploy_quant (the hand-written 48, "
                    "2026-09-13) or beacon_draw (the 48 drawn from the pool by the seal's block hash, 2026-09-14)")
    args = ap.parse_args(argv)
    # refusals that need no model stack: decided before torch is imported
    if args.prereg == "deploy_quant" and args.beacon and not args.tag:
        raise SystemExit("a beacon-drawn run is not the sealed experiment (the 2026-09-13 PREREG froze the hand-written 48); "
                         "pass --tag, or run under --prereg beacon_draw, whose experiment is the drawn set")
    if args.prereg == "beacon_draw" and not args.beacon:
        raise SystemExit("the beacon-draw PREREG's run draws its canaries from the block hash of the slot that sealed it; "
                         "pass --beacon <64 hex> (python -m styxx.clock verify prints it for the seal's line), "
                         "or --prereg deploy_quant for the hand-written set")
    if args.beacon and not re.fullmatch(r"[0-9a-f]{64}", args.beacon):
        raise SystemExit("--beacon must be 64 lowercase hex characters (styxx.clock.blockhash_to_beacon converts the chain's base58)")
    return args


ARGS = _parse_args() if __name__ == "__main__" else None

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)      # this checkout first; sealed_refusal checks the checksum imported is its styxx/checksum.py
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # before CUDA initialises: cuBLAS determinism

import numpy as np  # noqa: E402
import torch  # noqa: E402

import styxx  # noqa: E402
from styxx import checksum as ck  # noqa: E402
from styxx.geoplate import coefficients, render_grid  # noqa: E402

_STYXX_FILE = os.path.abspath(styxx.__file__)
if os.path.commonpath([_STYXX_FILE, ROOT]) != ROOT:
    raise SystemExit(f"styxx resolved to {_STYXX_FILE}, outside this checkout {ROOT}; "
                     "refusing to run the sealed experiment with another package")
# the file the checksum module was actually imported from: an untracked styxx/checksum/ package wins over checksum.py
_CHECKSUM_FILE = os.path.abspath(ck.__file__) if getattr(ck, "__file__", None) else None

HERE = os.path.dirname(os.path.abspath(__file__))
PREREG = PREREGS[ARGS.prereg] if ARGS is not None else PREREGS["deploy_quant"]
K1_FLOOR_NATS = 1e-2
N_BOOT, SEED = 2000, 20260913


def _git(*args):
    try:
        return subprocess.run(["git", "-C", ROOT, *args], capture_output=True, check=True)
    except Exception:
        return None


def provenance(device, prereg=None):
    prereg = prereg or PREREG

    def sha(path):
        try:
            with open(path, "rb") as fh:
                return hashlib.sha256(fh.read()).hexdigest()
        except (OSError, TypeError):
            return None                           # sealed_refusal refuses the experiment on a checksum it could not hash
    head = _git("rev-parse", "HEAD")
    dirty = _git("status", "--porcelain")
    dirty_tracked = _git("status", "--porcelain", "--untracked-files=no")
    untracked = _git("ls-files", "--others", "--exclude-standard", "--", "styxx")
    blob = _git("show", f"HEAD:papers/checksum/{prereg}")
    versions = {"torch": torch.__version__}
    try:
        import transformers
        versions["transformers"] = transformers.__version__
    except Exception:  # pragma: no cover
        versions["transformers"] = None
    try:
        import bitsandbytes
        versions["bitsandbytes"] = bitsandbytes.__version__
    except Exception:
        versions["bitsandbytes"] = None
    return {
        "git_head": head.stdout.decode().strip() if head else None,
        "git_dirty": bool(dirty.stdout.strip()) if dirty else None,
        "git_dirty_tracked": bool(dirty_tracked.stdout.strip()) if dirty_tracked else None,
        "prereg": prereg,
        "prereg_blob_sha256": hashlib.sha256(blob.stdout).hexdigest() if blob else None,
        "styxx_untracked": ([p for p in untracked.stdout.decode("utf-8", "replace").splitlines() if p.strip()]
                            if untracked else None),
        "styxx_file": _STYXX_FILE,
        "checksum_file": _CHECKSUM_FILE,               # the file `from styxx import checksum` resolved to
        "checksum_py_sha256": sha(_CHECKSUM_FILE),     # the sha256 of that file, the checksum that ran
        "versions": versions,
        "python": sys.version.split()[0],
        "device": device,
        "cuda_device": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "torch_quantized_engine": torch.backends.quantized.engine,
    }


def load(name, kind, device, dtype):
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
    tok = AutoTokenizer.from_pretrained(name)
    if kind == "bf16":
        m = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).to(device)
    elif kind == "fp32":
        m = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).to(device)
    elif kind == "nf4":
        from transformers import BitsAndBytesConfig
        q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
        m = AutoModelForCausalLM.from_pretrained(name, quantization_config=q, device_map={"": 0})
    elif kind == "int8":
        from transformers import BitsAndBytesConfig
        m = AutoModelForCausalLM.from_pretrained(name, quantization_config=BitsAndBytesConfig(load_in_8bit=True),
                                                 device_map={"": 0})
    elif kind == "int8-dynamic-cpu":
        base = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).eval()
        m = torch.ao.quantization.quantize_dynamic(base, {torch.nn.Linear}, dtype=torch.qint8)
    elif kind == "random":
        torch.manual_seed(343)
        m = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(name), dtype=dtype).to(device)
    else:
        raise ValueError(kind)
    return tok, m.eval()


def hf_probe_on(model, tokenizer, device):
    def probe(prompt, continuation):
        p_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        c_ids = tokenizer(continuation, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        ids = torch.cat([p_ids, c_ids], dim=1)
        with torch.no_grad():
            logits = model(input_ids=ids).logits[0].float()
        logp = torch.log_softmax(logits, dim=-1)
        n_p = p_ids.shape[1]
        cont = [logp[n_p - 1 + t, c_ids[0, t]].item() for t in range(c_ids.shape[1])]
        return ck.Probe(cont_logprobs=cont, next_logprobs=logp[n_p - 1].cpu().numpy())
    return probe


def top1(model, tok, device, canaries):
    hits = 0
    for _, p, c in canaries:
        ids = tok(p, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            pred = model(input_ids=ids).logits[0, -1].argmax().item()
        hits += int(pred == tok(c, add_special_tokens=False).input_ids[0])
    return hits


def _write_json(path, obj, indent=None):
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(obj, fh, indent=indent)
        fh.write("\n")


def main():
    args = ARGS if ARGS is not None else _parse_args()
    smoke = args.smoke
    name = "HuggingFaceTB/SmolLM2-135M" if smoke else args.model
    device = "cpu" if smoke or not torch.cuda.is_available() else "cuda"
    is_the_experiment = (not smoke) and (not args.tag) and name == "Qwen/Qwen2.5-1.5B"
    if not smoke and not args.tag and not is_the_experiment:
        raise SystemExit("an untagged, non-smoke run is the sealed experiment and its model is Qwen/Qwen2.5-1.5B; "
                         "pass --tag <suffix> for an instrument check on another model")
    suffix = ("_smoke" if smoke else "") + args.tag
    prereg, stem = PREREGS[args.prereg], args.prereg            # the outputs carry the PREREG's name
    base_kind = "fp32" if smoke else "bf16"
    dtype = torch.float32 if base_kind == "fp32" else torch.bfloat16
    arms = [("A", base_kind), ("A2", base_kind), ("A3", base_kind),
            ("Q4", "int8-dynamic-cpu" if smoke else "nf4"),
            ("Q8", "int8-dynamic-cpu" if smoke else "int8"),
            ("R", "random")]
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
    prov = provenance(device, prereg)
    print(f"  provenance: head {prov['git_head']} dirty={prov['git_dirty']} prereg blob {str(prov['prereg_blob_sha256'])[:12]} "
          f"styxx {prov['styxx_file']}")
    refusal = sealed_refusal(prov, args.prereg, is_the_experiment)
    prov["beacon_seal_checked"] = False
    if refusal is None and is_the_experiment and args.prereg == "beacon_draw":
        # the experiment's beacon is its seal's, read ANCHORED on chain; checked before any draw or model load
        from styxx import clock as _clock
        refusal = beacon_refusal(args.beacon, os.path.join(ROOT, ANCHORS), SEALED[args.prereg], _clock._rpc)
        prov["beacon_seal_checked"] = refusal is None
    if refusal:
        raise SystemExit("refusing to run the sealed experiment: " + refusal)
    prov["sealed_blob_sha256"] = SEALED[args.prereg]
    prov["prereg_blob_is_sealed"] = prov["prereg_blob_sha256"] == SEALED[args.prereg]
    prov["model_revisions"] = {}
    canaries, draw_record = ck.CANARIES, None
    if args.beacon:
        from styxx import beacon as _beacon
        canaries, draw_record = _beacon.draw(args.beacon, 48)
        print(f"  canaries drawn by beacon {draw_record['beacon'][:12]}… from pool {draw_record['pool_sha256'][:12]}… "
              f"(canary set {draw_record['canary_sha256'][:12]}…)")
    t0 = time.time()
    fps, hits = {}, {}
    for tag, kind in arms:
        tok, m = load(name, kind, device, dtype)
        prov["model_revisions"][tag] = getattr(getattr(m, "config", None), "_commit_hash", None)
        fps[tag] = ck.fingerprint(hf_probe_on(m, tok, device), f"{name} {kind} [{tag}]", canaries=canaries,
                                  tokenizer_id=name, draw=draw_record)
        hits[tag] = top1(m, tok, device, canaries)
        print(f"  {tag:3s} {kind:18s} top-1 {hits[tag]}/48   [{time.time() - t0:.0f}s]", flush=True)
        del m
        if device == "cuda":
            torch.cuda.empty_cache()
    floor = ck.null_floor([fps["A"], fps["A2"], fps["A3"]])
    k1 = {"threshold_nats": K1_FLOOR_NATS, "floor_nats": floor, "fired": bool(floor > K1_FLOOR_NATS)}
    out = {"prereg": prereg, "smoke": smoke, "tag": args.tag, "is_the_experiment": is_the_experiment,
           "model": name, "device": device,
           "provenance": prov,
           "canaries": {"n": len(canaries), "canary_sha256": ck.canary_sha256(canaries), "draw": draw_record,
                        "source": "styxx.beacon draw" if draw_record else "styxx.checksum.CANARIES (the hand-written set)"},
           "sanity": {"first_token_top1_hits": hits, "n_canaries": len(canaries), "null_floor_nats": floor,
                      "top1_loss_vs_A": {t: hits["A"] - hits[t] for t in ("A2", "A3", "Q4", "Q8", "R")}},
           "distance_params": {"n_boot": N_BOOT, "seed": SEED, "floor_nats": floor},
           "k1": k1}
    print(f"  null floor (worst of 3 {base_kind} loads): {floor:.6f} nats/token")
    if k1["fired"]:
        # PREREG K1: the serving is not deterministic enough for this probe; the run is INCONCLUSIVE and
        # says so; no other hypothesis is evaluated. The fingerprints are still written so a NEW prereg
        # can look at them; no comparison cert exists for this run.
        out["verdict"] = "INCONCLUSIVE"
        print(f"  K1 FIRED: null floor {floor:.6f} > {K1_FLOOR_NATS} nats/token — INCONCLUSIVE; no hypothesis evaluated")
    else:
        for tag in ("A2", "Q4", "Q8", "R"):
            d = ck.distance(fps["A"], fps[tag], n_boot=N_BOOT, seed=SEED, floor_nats=floor)
            out[tag] = ck.cert(fps["A"], fps[tag], d, note=f"{name} on {device}; teacher-forced; deterministic best effort")
            print(f"  A vs {tag:3s}: {d.verdict:12s} mean|Δlogp| = {d.mean_abs_nats:.5f} [{d.ci_mean_abs[0]:.5f}, {d.ci_mean_abs[1]:.5f}]"
                  f"   r = {d.rdm_r:.4f}   top-1 {hits['A']}->{hits[tag]}")
        # CORRECTION_prereg_deploy_quant_H1: the 2026-09-13 PREREG's H1 grades A vs A' against a floor that
        # includes that very pair; under it the held-out reading is unpreregistered and decides nothing.
        # The beacon-draw PREREG (2026-09-14) freezes the held-out form as H1's pair clause (the
        # CORRECTION's rule 5): the floor from the two pairs that do not contain A' (A-A'' and A'-A'').
        held_out_floor = max(float(np.abs(fps["A"].mean_lp - fps["A3"].mean_lp).mean()),
                             float(np.abs(fps["A2"].mean_lp - fps["A3"].mean_lp).mean()))
        d_ho = ck.distance(fps["A"], fps["A2"], n_boot=N_BOOT, seed=SEED, floor_nats=held_out_floor)
        preregistered = args.prereg == "beacon_draw"
        out["h1_held_out"] = {"unpreregistered": not preregistered, "preregistered_by": prereg if preregistered else None,
                              "floor_from": ["A-A3", "A2-A3"], "floor_nats": held_out_floor,
                              "cert": ck.cert(fps["A"], fps["A2"], d_ho, note="held-out floor reading of H1" +
                                              ("; the preregistered pair clause" if preregistered else "; decides nothing in this run"))}
        print(f"  H1 held-out reading ({'preregistered' if preregistered else 'unpreregistered'}): "
              f"A vs A2 {d_ho.verdict} against floor {held_out_floor:.6f}")
    _write_json(os.path.join(HERE, f"{stem}_certs{suffix}.json"), out, indent=1)
    _write_json(os.path.join(HERE, f"{stem}_fingerprints{suffix}.json"), {k: v.to_json() for k, v in fps.items()})
    if k1["fired"]:
        print("wrote certs (K1 fired: no comparisons) and fingerprints; no plates for an INCONCLUSIVE run")
        return
    lab = lambda t: f"vs A: {out[t]['distance']['mean_abs_nats']:.4f} nats/token, r = {out[t]['distance']['rdm_r']:.3f}  ({out[t]['distance']['verdict']})"
    items = [(f"{name.split('/')[-1]}  {base_kind}", "the model", coefficients(fps["A"].rdm)),
             ("same weights, reloaded", lab("A2"), coefficients(fps["A2"].rdm)),
             ("nf4 (4-bit)" if not smoke else "int8 dynamic (stand-in)", lab("Q4"), coefficients(fps["Q4"].rdm)),
             ("llm.int8" if not smoke else "int8 dynamic (stand-in)", lab("Q8"), coefficients(fps["Q8"].rdm)),
             ("control: random weights", lab("R"), coefficients(fps["R"].rdm))]
    png = render_grid(items, os.path.join(HERE, f"{stem}_plates{suffix}.png"),
                      title=f"checksum: {name}, 48 hashed canaries, {device} — what quantization moves, as sand", ncols=3)
    print("wrote", png)


if __name__ == "__main__":
    main()
