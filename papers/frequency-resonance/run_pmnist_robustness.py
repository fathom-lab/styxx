# -*- coding: utf-8 -*-
"""run_pmnist_robustness.py -- frozen by PREREG_pmnist_robustness_2026_10_11 (sealed blob 18aa265c..., SEALS row 8).

Builds on the two committed runners WITHOUT editing them:
  * run_pmnist_ablation.py  (R): data, FREE and CLAMPED arms, the training loop, the evaluation, the scan red team
  * run_pmnist_untied.py    (U): the REAL2 arm (RealClassifier, train_real2) and its parameter red team
Both are imported verbatim and their LF-normalised bytes must hash to the blobs the PREREG froze, or the
runner refuses before data. The only state this runner changes in R is the module attributes PERM_SEED, STEPS,
EVAL_EVERY (and H, D_SSM in the optional width stage), set before a run and restored after it; R.train,
U.train_real2 and R.load_data read them at call time. Curves are captured by wrapping R.test_acc for the
duration of a run (the committed function still computes every number).

  python run_pmnist_robustness.py --derive-perms [--beacon <64 hex>]      prints the five held-out permutations
  python run_pmnist_robustness.py --smoke                                 INVALID-only; measures the per-run time
  python run_pmnist_robustness.py --night 1 [--accept-schedule]           stage A + CLAMPED-LONG (no beacon needed)
  python run_pmnist_robustness.py --night 2 --beacon <hex> | --anchors papers/charon/anchors.jsonl | --fallback-perms
  python run_pmnist_robustness.py --night 3 ... (same)                    stage D
  python run_pmnist_robustness.py --aggregate                             scores every leaf through styxx.protocol

Nothing here trains until --night or --smoke is given, and neither starts without CUDA and 6 GiB free. No model
and no training loop is defined in this file. The pure functions (hashing, derivation, schedule, aggregate
arithmetic, the outcome walk) import without torch; the tests exercise them on CPU.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent

PREREG = HERE / "PREREG_pmnist_robustness_2026_10_11.md"
# the sha256 of the git blob of the PREREG as amended before any data (its section 15; SEALS_2026_09_13.md row 8,
# frozen at 98072bed; row 7, 2068e673..., is the same file before section 15 and is superseded); the PREREG on disk,
# LF-normalised, must hash to this or the runner refuses before data (section 9.4 item 3)
PREREG_BLOB_SHA256 = "18aa265c248fff55f49e71286f7e8baa2c54a9fc0e81dde3f397b83656776a72"
# LF-normalised git blobs of the imported runners, as frozen in the PREREG's section 1
FROZEN_RUNNERS = {
    "run_pmnist_ablation.py": "c8d7cdfbab3fabce94fc541e92c46947cf1909b13ad6d4994baa42d52a305423",
    "run_pmnist_untied.py": "caf276da27e0b6ad408adadc019d7f39d07747ee928fbdf9e4d636a645de94bf",
}
DATA_FILE = HERE / "_mnist.npz"
DATA_SHA256 = "98d685052d7a6ab8b6982491df8ec3eb366a252fd5415aa58f878bbe6510d4e8"

ANCHOR_PERM = 1234
ANCHORS = {"free": 0.9195, "clamped": 0.6073}       # pmnist_ablation_result.json /result/free_acc, /result/clamped_acc
FALLBACK_TAG = "PREREG_pmnist_robustness_2026_10_11"
FALLBACK_PERMS = [3584514041, 27380109, 2994719670, 3317084913, 2856893334]
N_HELD_OUT = 5
DERIVATION_BEACON = "beacon"
DERIVATION_FALLBACK = "sha256-of-tag fallback"

ARMS = ("free", "clamped", "real2")
STEPS_MAIN, STEPS_LONG, EVAL_EVERY = 4000, 12000, 500
H_DEFAULT = 64
ANCHOR_SEEDS = (0, 1, 2, 3, 4)
RERUN_SEEDS = (0, 1)                                # the receipt's seeds: determinism re-run, no gate
NEW_SEEDS = (2, 3, 4)                               # the gated anchor test
DERIVED_SEEDS = (0, 1, 2)
LONG_SEEDS = {"free": (0, 1), "clamped": (0, 1, 2)}
COMMITTED = {"BATCH": 64, "LR": 3e-3, "WD": 0.01, "T_LEN": 784, "N_BLK": 3, "H": 64, "D_SSM": 64}
READ_BACK_KEYS = ("PERM_SEED", "STEPS", "EVAL_EVERY", "H", "D_SSM", "BATCH", "LR", "WD", "T_LEN", "N_BLK")

PER_RUN_MIN_MEASURED = 13.6                         # sealed/SEALED_README.md: six 4000-step runs in 81.65 min
PER_RUN_MIN_REFUSE = 18.0                           # the smoke's projection above this refuses a night
NIGHT_HOURS_REFUSE = 8.0
FREE_GPU_MIN_BYTES = 6 * 2 ** 30                    # 6 GiB (6,144 MiB): the 2026-10-11 measured case is refused
DEGENERATE_ACC = 0.15                               # a constant responder scores <= 0.1135 (largest test class share)
T975 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}    # two-sided 95% t quantiles, n - 1 degrees of freedom
Z975 = 1.96                                         # the pooled interval's normal approximation (reported only)
BOOT_N, BOOT_SEED = 2000, 20261011
RECOVERY_MIN_GAP = 0.05

RUNS_DIR = HERE / "pmnist_robustness_runs"
CHAIN_FILE = RUNS_DIR / "leaf_chain.jsonl"
SMOKE_JSON = HERE / "pmnist_robustness_smoke.json"
RESULT_JSON = HERE / "pmnist_robustness_result.json"
LEAF_SCHEMA = "pmnist_robustness/leaf/v1"

GATED = ("plumbing_ok", "degenerate_fraction", "free_perm_mean", "anchor_gap_newseed", "gap_perm_mean",
         "gap_perm_ci_lo", "gap_long", "free_minus_real2_perm_mean", "real2_minus_clamped_perm_mean")

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_LOSS_RE = re.compile(r"\bloss\s+(\S+)")


class Refusal(RuntimeError):
    """A pre-run or pre-aggregate check failed: the runner stops before data, by name."""


# ----------------------------------------------------------------------------------------------- hashing

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _shown(p: Path) -> str:
    """A path as a receipt prints it: relative to the repository when inside it, so no receipt carries a home path."""
    try:
        return Path(p).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return Path(p).name


def lf_sha256(path: Path) -> str:
    """The sha256 of the file's bytes with CRLF folded to LF: equal to the git blob of an LF-stored file on
    any checkout (SEALS_2026_09_13.md names the CRLF trap; ATTACK.md F1 measured it on run_pmnist_untied.py)."""
    return sha256_bytes(Path(path).read_bytes().replace(b"\r\n", b"\n"))


def check_frozen_runners(freq_dir: Path = HERE) -> dict:
    out = {}
    for name, frozen in FROZEN_RUNNERS.items():
        p = Path(freq_dir) / name
        got = lf_sha256(p) if p.exists() else None
        out[name] = {"lf_sha256": got, "frozen": frozen, "ok": got == frozen}
    return out


def assert_frozen_runners(freq_dir: Path = HERE) -> dict:
    out = check_frozen_runners(freq_dir)
    bad = [f"{n}: {v['lf_sha256']} != frozen {v['frozen']}" for n, v in out.items() if not v["ok"]]
    if bad:
        raise Refusal("imported runner bytes are not the frozen blobs -- " + "; ".join(bad))
    return out


def check_data_pin(path: Path = DATA_FILE) -> dict:
    p = Path(path)
    got = sha256_bytes(p.read_bytes()) if p.exists() else None
    return {"path": _shown(p), "sha256": got, "pinned": DATA_SHA256, "bytes": p.stat().st_size if p.exists() else None,
            "ok": got == DATA_SHA256}


def assert_data_pin(path: Path = DATA_FILE) -> dict:
    d = check_data_pin(path)
    if not d["ok"]:
        raise Refusal(f"_mnist.npz at {d['path']} hashes to {d['sha256']}, not the pinned {DATA_SHA256}"
                      + ("" if d["sha256"] else " (file absent: copy the 2026-07-23 bytes beside the runner)"))
    return d


def check_prereg_bytes(path: Path = PREREG) -> dict:
    p = Path(path)
    got = lf_sha256(p) if p.exists() else None
    return {"path": _shown(p), "lf_sha256": got, "sealed": PREREG_BLOB_SHA256, "ok": got == PREREG_BLOB_SHA256}


def assert_prereg_bytes(path: Path = PREREG) -> dict:
    d = check_prereg_bytes(path)
    if not d["ok"]:
        raise Refusal(f"the PREREG on disk hashes to {d['lf_sha256']}, not the sealed blob {PREREG_BLOB_SHA256}; "
                      f"an edited PREREG scores nothing")
    return d


# ----------------------------------------------------------------------------------------------- permutations

def is_beacon_hex(s: str) -> bool:
    return isinstance(s, str) and bool(_HEX64.match(s))


def derive_perms(source: str, n: int = N_HELD_OUT, exclude=(ANCHOR_PERM,)) -> list[int]:
    """PREREG section 3: seed_k = top 32 bits of sha256(f"{source}:perm:{k}"), k = 1, 2, ..., skipping any value in
    `exclude` or already taken, until n are held. `source` is the beacon hex (64 lowercase hex) or the fallback tag."""
    out: list[int] = []
    k = 0
    while len(out) < n:
        k += 1
        v = int(hashlib.sha256(f"{source}:perm:{k}".encode("utf-8")).hexdigest()[:8], 16)
        if v in exclude or v in out:
            continue
        out.append(v)
    return out


def fallback_perms() -> list[int]:
    return derive_perms(FALLBACK_TAG)


def beacon_from_anchors(path: str | Path, digest: str = PREREG_BLOB_SHA256) -> dict:
    """Run styxx.clock.verify over the anchors file (network) and take the beacon of the ANCHORED sealed-prereg
    line whose digest is this PREREG's blob. Anything else is a refusal that names what the line read."""
    sys.path.insert(0, str(ROOT))
    from styxx.clock import verify
    rows = verify(str(path))
    mine = [r for r in rows if r.get("kind") == "sealed-prereg" and r.get("digest") == digest]
    if not mine:
        raise Refusal(f"{path} has no sealed-prereg line for digest {digest[:12]}...; the beacon derivation needs "
                      f"the anchor (or declare --fallback-perms, which is labelled as such)")
    r = mine[0]
    if r.get("status") != "ANCHORED" or not is_beacon_hex(r.get("beacon") or ""):
        raise Refusal(f"the sealed-prereg line for {digest[:12]}... reads {r.get('status')} "
                      f"({r.get('detail') or 'no detail'}); nights 2 and 3 do not start before it reads ANCHORED")
    return {"beacon": r["beacon"], "beacon_b58": r.get("beacon_b58"), "slot": r.get("slot"), "tx": r.get("tx"),
            "block_time": r.get("block_time"), "rpc": r.get("rpc")}


def resolve_perms(beacon: str | None, anchors: str | None, fallback: bool) -> dict:
    """What --beacon / --anchors / --fallback-perms mean, in one place; returns derivation, beacon, perms."""
    given = sum(bool(x) for x in (beacon, anchors, fallback))
    if given > 1:
        raise Refusal("give one of --beacon, --anchors, --fallback-perms, not several")
    if beacon:
        if not is_beacon_hex(beacon):
            raise Refusal("--beacon must be the 64 lowercase hex characters styxx.clock verify prints as beacon=")
        return {"derivation": DERIVATION_BEACON, "beacon": beacon, "perms": derive_perms(beacon), "anchor_line": None}
    if anchors:
        line = beacon_from_anchors(anchors)
        return {"derivation": DERIVATION_BEACON, "beacon": line["beacon"], "perms": derive_perms(line["beacon"]),
                "anchor_line": line}
    if fallback:
        return {"derivation": DERIVATION_FALLBACK, "beacon": None, "perms": fallback_perms(), "anchor_line": None}
    return {"derivation": None, "beacon": None, "perms": None, "anchor_line": None}


# ----------------------------------------------------------------------------------------------- schedule

def _run(perm, arm, seed, steps, h=H_DEFAULT) -> dict:
    return {"perm": int(perm), "arm": arm, "seed": int(seed), "steps": int(steps), "h": int(h)}


def schedule(night: int, perms: list[int] | None = None, width: bool = False) -> list[dict]:
    """The frozen run list of one night (PREREG section 8). Nights 2 and 3 need the five held-out permutations."""
    if width:
        return [_run(ANCHOR_PERM, arm, s, STEPS_MAIN, h) for h in (32, 128) for arm in ARMS for s in (0, 1)]
    if night == 1:
        return ([_run(ANCHOR_PERM, arm, s, STEPS_MAIN) for arm in ARMS for s in ANCHOR_SEEDS]
                + [_run(ANCHOR_PERM, "clamped", s, STEPS_LONG) for s in LONG_SEEDS["clamped"]])
    if night in (2, 3):
        if not perms or len(perms) != N_HELD_OUT:
            raise Refusal(f"night {night} needs the five held-out permutations (--beacon, --anchors or --fallback-perms)")
        held = perms[:2] if night == 2 else perms[2:]
        runs = [_run(p, arm, s, STEPS_MAIN) for p in held for arm in ARMS for s in DERIVED_SEEDS]
        if night == 2:
            runs += [_run(ANCHOR_PERM, "free", s, STEPS_LONG) for s in LONG_SEEDS["free"]]
        return runs
    raise Refusal(f"no night {night}")


def full_schedule(perms: list[int]) -> list[dict]:
    return schedule(1) + schedule(2, perms) + schedule(3, perms)


def leaf_name(r: dict) -> str:
    return f"p{r['perm']}_{r['arm']}_s{r['seed']}_{r['steps']}_h{r['h']}.json"


def leaf_path(r: dict) -> Path:
    return RUNS_DIR / leaf_name(r)


def expected_read_back(r: dict) -> dict:
    exp = dict(COMMITTED)
    exp.update({"PERM_SEED": r["perm"], "STEPS": r["steps"], "EVAL_EVERY": EVAL_EVERY, "H": r["h"], "D_SSM": r["h"]})
    return exp


def per_run_minutes(r: dict, per_4000: float) -> float:
    return per_4000 * r["steps"] / STEPS_MAIN * (r["h"] / H_DEFAULT)


# ----------------------------------------------------------------------------------------------- gates

def gates_spec(text: str | None = None) -> dict:
    sys.path.insert(0, str(ROOT))
    from styxx.protocol import _select_gates_block
    text = PREREG.read_text(encoding="utf-8") if text is None else text
    return json.loads(_select_gates_block(text))


def walk_outcomes(spec: dict, fired: dict) -> str | None:
    """The protocol's table walk, for enumeration: the earliest row whose `when` matches, else None."""
    for row in spec["outcomes"]:
        if all(fired.get(k) == v for k, v in row["when"].items()):
            return row["verdict"]
    return None


def synthetic_result() -> dict:
    """Every gated metric as 0.0, in the receipt's shape: what check_metrics() resolves before any night."""
    return {"result": {m: 0.0 for m in GATED}}


# ----------------------------------------------------------------------------------------------- arithmetic

def _mean(xs) -> float:
    xs = [float(x) for x in xs]
    return float(np.mean(xs)) if xs else float("nan")


def _var(xs) -> float:
    xs = [float(x) for x in xs]
    return float(np.var(xs, ddof=1)) if len(xs) >= 2 else float("nan")


def _r(x, nd=4):
    """Round for the receipt; a non-finite value becomes null so the receipt stays strict JSON (the gated
    metrics never reach here non-finite: section 9.7's rule replaces them with 0.0 and zeroes plumbing_ok)."""
    if x is None:
        return None
    x = float(x)
    return round(x, nd) if math.isfinite(x) else None


def t_ci(vals) -> dict:
    """Two-sided 95% t-interval with n - 1 degrees of freedom, n in 2..5 (the frozen quantiles)."""
    vals = [float(v) for v in vals]
    n = len(vals)
    m = _mean(vals)
    if n < 2 or n not in T975:
        return {"n": n, "mean": m, "sd": float("nan"), "t": None, "lo": float("nan"), "hi": float("nan")}
    sd = float(np.std(vals, ddof=1))
    hw = T975[n] * sd / math.sqrt(n)
    return {"n": n, "mean": m, "sd": sd, "t": T975[n], "lo": m - hw, "hi": m + hw}


def chain_entry(prev_id: str, leaf: str, leaf_sha256: str, n: int, wall_clock: str) -> dict:
    return {"n": n, "leaf": leaf, "sha256": leaf_sha256, "prev": prev_id,
            "entry_id": sha256_bytes((prev_id + leaf_sha256).encode("ascii")), "wall_clock": wall_clock}


def verify_chain(entries: list[dict], current: dict) -> dict:
    """Walk the append-only leaf log. `current` maps leaf name -> sha256 of the leaf's bytes on disk now.
    ok = the chain's structure holds and every leaf's current bytes are the latest entry for its name;
    rewrites = names with more than one entry (a leaf deleted and re-run; visible, not fatal)."""
    prev = "0" * 64
    latest: dict[str, str] = {}
    count: dict[str, int] = {}
    broken_at = None
    for i, e in enumerate(entries):
        want = sha256_bytes((prev + str(e.get("sha256"))).encode("ascii"))
        if e.get("prev") != prev or e.get("entry_id") != want or e.get("n") != i + 1:
            broken_at = i + 1
            break
        latest[e["leaf"]] = e["sha256"]
        count[e["leaf"]] = count.get(e["leaf"], 0) + 1
        prev = e["entry_id"]
    stale = sorted(n for n, s in current.items() if latest.get(n) != s)
    missing = sorted(n for n in current if n not in latest)
    rewrites = sorted(n for n, c in count.items() if c > 1)
    return {"ok": broken_at is None and not stale and not missing, "entries": len(entries), "broken_at": broken_at,
            "stale": stale, "missing_from_chain": missing, "rewrites": rewrites, "head": prev}


def _hier_bootstrap(by_perm: dict, rng: np.random.Generator, n: int) -> list[float]:
    """Permutations with replacement, then seeds within each resampled permutation per arm; gap_perm_mean each."""
    perms = list(by_perm)
    out = []
    for _ in range(n):
        picks = rng.choice(len(perms), size=len(perms), replace=True)
        gaps = []
        for i in picks:
            f, c = by_perm[perms[i]]["free"], by_perm[perms[i]]["clamped"]
            fb = rng.choice(f, size=len(f), replace=True)
            cb = rng.choice(c, size=len(c), replace=True)
            gaps.append(float(np.mean(fb) - np.mean(cb)))
        out.append(float(np.mean(gaps)))
    return out


def aggregate_leaves(leaves: list[dict], perms: list[int] | None, derivation: str | None, beacon: str | None,
                     chain: dict, disk: dict) -> dict:
    """PREREG section 4 over the leaves, and section 9.4's checks as `plumbing`. Pure: no file is read here.
    `chain` is verify_chain's output; `disk` holds the aggregate-time re-checks {"runners", "data", "prereg"}."""
    perms = list(perms) if perms else []
    held = [p for p in perms if p != ANCHOR_PERM]
    sched = full_schedule(held) if len(held) == N_HELD_OUT else schedule(1)
    by_name = {}
    for leaf in leaves:
        by_name.setdefault(leaf_name(leaf), leaf)       # the earliest wins; duplicates are a chain matter
    expected_names = {leaf_name(r): r for r in sched}

    main = [l for l in by_name.values() if l["steps"] == STEPS_MAIN and l["h"] == H_DEFAULT]
    long_ = [l for l in by_name.values() if l["steps"] == STEPS_LONG and l["h"] == H_DEFAULT]
    width = [l for l in by_name.values() if l["h"] != H_DEFAULT]        # optional stage E: reported, never gated

    def runs(perm, arm, seeds=None, pool=main, good_only=True):
        xs = [l for l in pool if l["perm"] == perm and l["arm"] == arm and (seeds is None or l["seed"] in seeds)]
        return [l for l in xs if not (good_only and l.get("degenerate"))]

    def acc(perm, arm, seeds=None, pool=main, good_only=True):
        return _mean([l["final_acc"] for l in runs(perm, arm, seeds, pool, good_only)])

    def floor_perm(perm, seeds, good_only=True):
        xs = runs(perm, "free", seeds, main, good_only=False)
        vals = [0.0 if (good_only and l.get("degenerate")) else l["final_acc"] for l in xs]
        return _mean(vals)

    def metrics(good_only: bool) -> dict:
        gap_p = {p: acc(p, "free", good_only=good_only) - acc(p, "clamped", good_only=good_only) for p in held}
        fr2_p = {p: acc(p, "free", good_only=good_only) - acc(p, "real2", good_only=good_only) for p in held}
        r2c_p = {p: acc(p, "real2", good_only=good_only) - acc(p, "clamped", good_only=good_only) for p in held}
        gaps = [gap_p[p] for p in held]
        ci = t_ci(gaps) if len(gaps) == N_HELD_OUT else t_ci([])
        floors = [floor_perm(ANCHOR_PERM, NEW_SEEDS, good_only)] + [floor_perm(p, DERIVED_SEEDS, good_only) for p in held]
        free_long = acc(ANCHOR_PERM, "free", LONG_SEEDS["free"], long_, good_only)
        clamped_long = acc(ANCHOR_PERM, "clamped", LONG_SEEDS["clamped"], long_, good_only)
        n_main_deg = sum(1 for l in main if l.get("degenerate"))
        res = {
            "degenerate_fraction": n_main_deg / 60.0,
            "free_perm_mean": _mean(floors),
            "anchor_gap_newseed": acc(ANCHOR_PERM, "free", NEW_SEEDS, good_only=good_only)
                                  - acc(ANCHOR_PERM, "clamped", NEW_SEEDS, good_only=good_only),
            "gap_perm_mean": _mean(gaps),
            "gap_perm_ci_lo": ci["lo"],
            "gap_long": free_long - clamped_long,
            "free_minus_real2_perm_mean": _mean(list(fr2_p.values())),
            "real2_minus_clamped_perm_mean": _mean(list(r2c_p.values())),
        }
        return res, gap_p, fr2_p, r2c_p, ci, floors, free_long, clamped_long

    res, gap_p, fr2_p, r2c_p, ci, floors, free_long, clamped_long = metrics(True)
    res_unexcluded = metrics(False)[0]

    # -- reported ---------------------------------------------------------------------------------------
    rec_p = {p: r2c_p[p] / gap_p[p] for p in held if math.isfinite(gap_p[p]) and gap_p[p] >= RECOVERY_MIN_GAP}
    rec_ci = t_ci(list(rec_p.values()))
    if len(rec_p) < 2:
        rec_class = "undefined"
    elif rec_ci["hi"] < 0.25:
        rec_class = "rotation"
    elif rec_ci["lo"] > 0.75:
        rec_class = "diversity"
    else:
        rec_class = "partial_or_unresolved_at_this_n"
    f_runs = [l["final_acc"] for p in held for l in runs(p, "free")]
    c_runs = [l["final_acc"] for p in held for l in runs(p, "clamped")]
    pooled_gap = _mean(f_runs) - _mean(c_runs)
    pooled_se = math.sqrt(_var(f_runs) / max(len(f_runs), 1) + _var(c_runs) / max(len(c_runs), 1)) \
        if len(f_runs) >= 2 and len(c_runs) >= 2 else float("nan")
    by_perm = {p: {"free": [l["final_acc"] for l in runs(p, "free")],
                   "clamped": [l["final_acc"] for l in runs(p, "clamped")]} for p in held}
    boot: dict = {"n": BOOT_N, "seed": BOOT_SEED, "ci_lo": None, "ci_hi": None, "between_perm_var_share": None}
    if held and all(len(v["free"]) >= 1 and len(v["clamped"]) >= 1 for v in by_perm.values()):
        bs = _hier_bootstrap(by_perm, np.random.default_rng(BOOT_SEED), BOOT_N)
        boot["ci_lo"], boot["ci_hi"] = _r(np.percentile(bs, 2.5)), _r(np.percentile(bs, 97.5))
        vg = _var([gap_p[p] for p in held])
        within = _mean([(_var(v["free"]) / len(v["free"]) if len(v["free"]) >= 2 else 0.0)
                        + (_var(v["clamped"]) / len(v["clamped"]) if len(v["clamped"]) >= 2 else 0.0)
                        for v in by_perm.values()])
        if math.isfinite(vg) and vg > 0:
            boot["between_perm_var_share"] = _r(max(0.0, vg - within) / vg)
        boot["gap_perm_var"], boot["seed_noise_var_of_gap_p"] = _r(vg, 6), _r(within, 6)

    free_01 = acc(ANCHOR_PERM, "free", RERUN_SEEDS, good_only=False)
    clamped_01 = acc(ANCHOR_PERM, "clamped", RERUN_SEEDS, good_only=False)
    a_fr2 = acc(ANCHOR_PERM, "free", NEW_SEEDS) - acc(ANCHOR_PERM, "real2", NEW_SEEDS)
    a_r2c = acc(ANCHOR_PERM, "real2", NEW_SEEDS) - acc(ANCHOR_PERM, "clamped", NEW_SEEDS)
    a_gap = res["anchor_gap_newseed"]
    anchor = {
        "per_seed": {arm: {str(l["seed"]): l["final_acc"] for l in runs(ANCHOR_PERM, arm, good_only=False)} for arm in ARMS},
        "free_01": _r(free_01), "clamped_01": _r(clamped_01),
        "anchor_max_abs_dev": _r(max(abs(free_01 - ANCHORS["free"]), abs(clamped_01 - ANCHORS["clamped"]))),
        "receipt": ANCHORS,
        "anchor_gap_5seed": _r(acc(ANCHOR_PERM, "free") - acc(ANCHOR_PERM, "clamped")),
        "anchor_free_minus_real2_newseed": _r(a_fr2), "anchor_real2_minus_clamped_newseed": _r(a_r2c),
        "anchor_recovery_newseed": _r(a_r2c / a_gap) if math.isfinite(a_gap) and a_gap >= RECOVERY_MIN_GAP else None,
    }

    def curve_at(l, step):
        i = step // EVAL_EVERY - 1
        return l["curve"][i] if l.get("curve") and 0 <= i < len(l["curve"]) else float("nan")

    long_c = runs(ANCHOR_PERM, "clamped", LONG_SEEDS["clamped"], long_)
    long_f = runs(ANCHOR_PERM, "free", LONG_SEEDS["free"], long_)
    main_c = [l for p in [ANCHOR_PERM] + held for l in runs(p, "clamped")]
    long_block = {
        "free_long": _r(free_long), "clamped_long": _r(clamped_long),
        "clamped_long_gain": _r(clamped_long - acc(ANCHOR_PERM, "clamped", LONG_SEEDS["clamped"])),
        "free_long_gain": _r(free_long - acc(ANCHOR_PERM, "free", LONG_SEEDS["free"])),
        "clamped_long_midgain": _r(_mean([curve_at(l, 8000) - curve_at(l, 4000) for l in long_c])),
        "free_long_midgain": _r(_mean([curve_at(l, 8000) - curve_at(l, 4000) for l in long_f])),
        "clamped_last_quarter_gain": _r(_mean([curve_at(l, 4000) - curve_at(l, 3000) for l in main_c])),
    }
    degenerate = [l for l in by_name.values() if l.get("degenerate")]
    degeneracy = {
        "by_arm": {arm: sorted(leaf_name(l) for l in degenerate if l["arm"] == arm) for arm in ARMS},
        "by_perm": {str(p): sorted(leaf_name(l) for l in degenerate if l["perm"] == p) for p in [ANCHOR_PERM] + held},
        "main_count": sum(1 for l in degenerate if l["steps"] == STEPS_MAIN),
        "long_count": sum(1 for l in degenerate if l["steps"] == STEPS_LONG),
        "free_counted_as_zero_in_floor": sorted(leaf_name(l) for l in degenerate
                                                if l["arm"] == "free" and l["steps"] == STEPS_MAIN),
    }
    reliance = [l["within_model_theta0_reliance"] for l in by_name.values()
                if l["arm"] == "free" and l.get("within_model_theta0_reliance") is not None]
    gap_by_width = {}
    for h in sorted({l["h"] for l in width}):
        f = [l["final_acc"] for l in width if l["h"] == h and l["arm"] == "free" and not l.get("degenerate")]
        c = [l["final_acc"] for l in width if l["h"] == h and l["arm"] == "clamped" and not l.get("degenerate")]
        gap_by_width[str(h)] = {"n": [len(f), len(c)], "gap": _r(_mean(f) - _mean(c)) if f and c else None}

    # -- plumbing (section 9.4) -----------------------------------------------------------------------------
    present = set(by_name)
    missing = sorted(set(expected_names) - present)
    extra = sorted(present - set(expected_names) - {leaf_name(l) for l in width})
    rb_bad = sorted(n for n, l in by_name.items() if n in expected_names
                    and {k: l.get("read_back", {}).get(k) for k in READ_BACK_KEYS} != expected_read_back(expected_names[n]))
    imports_bad = sorted(n for n, l in by_name.items()
                         if {k: (l.get("imports") or {}).get(k, {}).get("lf_sha256") for k in FROZEN_RUNNERS} != FROZEN_RUNNERS)
    data_bad = sorted(n for n, l in by_name.items() if l.get("data_sha256") != DATA_SHA256)
    prereg_bad = sorted(n for n, l in by_name.items() if l.get("prereg_blob_sha256") != PREREG_BLOB_SHA256)
    preflight_bad = sorted(n for n, l in by_name.items() if l.get("preflight_ok") is not True)
    curves_bad = sorted(n for n, l in by_name.items()
                        if len(l.get("curve") or []) != l["steps"] // EVAL_EVERY or l.get("curve_every") != EVAL_EVERY)
    mismatch = sorted(n for n, l in by_name.items() if l.get("curve_final_mismatch"))
    derived_leaves = [l for l in by_name.values() if l["perm"] != ANCHOR_PERM]
    derivations = {(l.get("perm_derivation"), l.get("beacon")) for l in derived_leaves}
    derivation_consistent = len(derivations) <= 1 and (not derivations or derivations == {(derivation, beacon)})
    perms_ok = len(held) == N_HELD_OUT and (
        (derivation == DERIVATION_BEACON and is_beacon_hex(beacon or "") and held == derive_perms(beacon))
        or (derivation == DERIVATION_FALLBACK and held == fallback_perms()))
    nonfinite = sorted(m for m in GATED if m != "plumbing_ok" and not math.isfinite(float(res[m])))
    for m in nonfinite:
        res[m] = 0.0
    plumbing = {
        "leaves_complete": not missing and not extra and len(expected_names) == 65,
        "read_back_ok": not rb_bad, "imports_ok": not imports_bad, "data_ok": not data_bad,
        "prereg_ok": not prereg_bad, "preflight_ok": not preflight_bad, "curves_ok": not curves_bad,
        "no_final_mismatch": not mismatch, "derivation_consistent": derivation_consistent,
        "perms_match_derivation": perms_ok, "chain_ok": bool(chain.get("ok")),
        "metrics_finite": not nonfinite,
        "disk_runners_ok": bool(disk.get("runners")), "disk_data_ok": bool(disk.get("data")),
        "disk_prereg_ok": bool(disk.get("prereg")),
    }
    plumbing_detail = {"missing": missing, "extra": extra, "read_back_bad": rb_bad, "imports_bad": imports_bad,
                       "data_bad": data_bad, "prereg_bad": prereg_bad, "preflight_bad": preflight_bad,
                       "curves_bad": curves_bad, "curve_final_mismatch": mismatch,
                       "derivations_seen": sorted(str(d) for d in derivations), "nonfinite_metrics": nonfinite}
    res["plumbing_ok"] = 1.0 if all(plumbing.values()) else 0.0
    result = {m: _r(res[m]) for m in GATED}
    result["nonfinite_metrics"] = nonfinite

    out = {
        "prereg": PREREG.name, "prereg_blob_sha256": PREREG_BLOB_SHA256,
        "config": {"steps_main": STEPS_MAIN, "steps_long": STEPS_LONG, "eval_every": EVAL_EVERY, "h": H_DEFAULT,
                   "anchor_perm": ANCHOR_PERM, "rerun_seeds": list(RERUN_SEEDS), "new_seeds": list(NEW_SEEDS),
                   "derived_seeds": list(DERIVED_SEEDS), "long_seeds": {k: list(v) for k, v in LONG_SEEDS.items()},
                   "held_out_perms": held, "perm_derivation": derivation, "beacon": beacon,
                   "fallback_tag": FALLBACK_TAG, "degenerate_acc": DEGENERATE_ACC, "t975": T975, "z975": Z975},
        "imports": FROZEN_RUNNERS, "data": {"file": DATA_FILE.name, "sha256": DATA_SHA256},
        "schedule": {"expected_leaves": len(expected_names), "present": len(present)},
        "per_permutation": {str(p): {"n": {arm: len(runs(p, arm)) for arm in ARMS},
                                     "mean": {arm: _r(acc(p, arm)) for arm in ARMS},
                                     "gap": _r(gap_p[p]), "free_minus_real2": _r(fr2_p[p]),
                                     "real2_minus_clamped": _r(r2c_p[p]), "recovery": _r(rec_p.get(p))} for p in held},
        "result": result,
        "reported": {
            "gap_perm_ci_hi": _r(ci["hi"]), "gap_perm_sd": _r(ci["sd"]),
            "gap_perm_min": _r(min([gap_p[p] for p in held], default=float("nan"))),
            "free_perm_min": _r(min(floors, default=float("nan"))),
            "recovery_perm_mean": _r(rec_ci["mean"]) if rec_p else None,
            "recovery_perm_ci": [_r(rec_ci["lo"]), _r(rec_ci["hi"])] if len(rec_p) >= 2 else None,
            "recovery_defined_on": [str(p) for p in rec_p], "recovery_class": rec_class,
            "pooled_gap": _r(pooled_gap), "pooled_n": [len(f_runs), len(c_runs)],
            "pooled_ci": [_r(pooled_gap - Z975 * pooled_se), _r(pooled_gap + Z975 * pooled_se)],
            "pooled_ci_note": "normal approximation, z = 1.96, run as the unit; reported beside the permutation interval",
            "reliance_mean": _r(_mean(reliance)) if reliance else None, "reliance_n": len(reliance),
            "gap_by_width": gap_by_width,
        },
        "unexcluded": {m: _r(res_unexcluded[m]) for m in GATED if m != "plumbing_ok"},
        "bootstrap": boot, "degeneracy": degeneracy, "anchor": anchor, "long": long_block,
        "leaf_chain": chain, "plumbing": plumbing, "plumbing_detail": plumbing_detail,
        "runs": [by_name[n] for n in sorted(by_name)],
    }
    return out


# ----------------------------------------------------------------------------------------------- torch side

def _load_runners():
    """Import the two committed runners (they import torch). Called only after every check that needs no torch has
    passed (PREREG section 15.4); a box without torch is refused by name, not by a traceback."""
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    try:
        import run_pmnist_ablation as R      # noqa: E402  (reads --smoke from sys.argv)
        import run_pmnist_untied as U        # noqa: E402
    except ImportError as e:
        raise Refusal(f"the committed runners do not import here ({type(e).__name__}: {e}); they need torch") from e
    return R, U


def gpu_info() -> dict:
    import torch
    info = {"cuda": bool(torch.cuda.is_available()), "torch": torch.__version__}
    if info["cuda"]:
        free, total = torch.cuda.mem_get_info()
        info.update({"free_mib": free // 2 ** 20, "total_mib": total // 2 ** 20, "held_mib": (total - free) // 2 ** 20,
                     "device_name": torch.cuda.get_device_name(0)})
    return info


def require_gpu() -> dict:
    info = gpu_info()
    if not info["cuda"]:
        raise Refusal("CUDA is not available; the PREREG's runs are GPU runs (section 8)")
    if info["free_mib"] * 2 ** 20 < FREE_GPU_MIN_BYTES:
        raise Refusal(f"{info['free_mib']} MiB free on the card ({info['held_mib']} MiB held by other processes); "
                      f"the PREREG requires {FREE_GPU_MIN_BYTES // 2**20} MiB free -- close the process holding the card")
    return info


def assert_rng_sharing(R, U, seeds=range(5)) -> dict:
    """PREREG section 9.4 item 5: after torch.manual_seed(s) and construction, FREE, CLAMPED and REAL2 leave the
    same RNG state and share every parameter outside the .ssm. modules bit for bit. CPU construction only."""
    import torch
    out = {}
    ctors = (lambda: R.SSMClassifier(True), lambda: R.SSMClassifier(False), lambda: U.RealClassifier())
    for s in seeds:
        states, params = [], []
        for ctor in ctors:
            torch.manual_seed(int(s))
            m = ctor()
            states.append(torch.get_rng_state().clone())
            params.append({n: p.detach().clone() for n, p in m.named_parameters() if ".ssm." not in n})
        if not (torch.equal(states[0], states[1]) and torch.equal(states[0], states[2])):
            raise Refusal(f"RNG state after construction differs between arms at seed {s}")
        names = set(params[0])
        if not (names == set(params[1]) == set(params[2])):
            raise Refusal(f"non-SSM parameter names differ between arms at seed {s}")
        for n in sorted(names):
            if not (torch.equal(params[0][n], params[1][n]) and torch.equal(params[0][n], params[2][n])):
                raise Refusal(f"non-SSM tensor {n} differs between arms at seed {s}")
        out[str(s)] = {"shared_tensors": len(names), "rng_state_equal": True}
    return out


def preflight_pure() -> dict:
    """PREREG section 9.4 items 1-3, with no torch imported: the two runner blobs, the data pin, the sealed PREREG,
    and styxx.protocol's own checks (Experiment constructs, no vacuous gate, check_metrics resolves every path)."""
    sys.path.insert(0, str(ROOT))
    from styxx.protocol import Experiment, PrologueError, GateSpecError
    pre = {"imports": assert_frozen_runners(HERE), "data": assert_data_pin(DATA_FILE), "prereg": assert_prereg_bytes(PREREG)}
    try:
        exp = Experiment(PREREG, repo_root=ROOT)
    except (PrologueError, GateSpecError) as e:
        raise Refusal(f"styxx.protocol refuses the PREREG: {e}") from e
    if exp.vacuous_gates or exp.undeclared_power_gates:
        raise Refusal(f"vacuous gates {exp.vacuous_gates}; gates without power basis {exp.undeclared_power_gates}")
    cm = exp.check_metrics(synthetic_result())
    unusable = sorted(k for k, v in cm.items() if not v["usable"])
    if unusable:
        raise Refusal(f"check_metrics: these gate paths do not resolve on the receipt shape: {unusable}")
    pre.update({"prereg_commit": exp.prereg_commit, "gates_sha256": exp.gates_sha256, "check_metrics_ok": True})
    return pre


def preflight_torch(R, U, pre: dict, *, want_gpu: bool = True) -> dict:
    """PREREG section 9.4 items 7, 4 and 5, in that order (section 15.4): the GPU guard runs before any tensor is
    placed on the device, and R.redteam()'s scan check is the earliest tensor the run puts on the card."""
    for k, v in COMMITTED.items():
        if getattr(R, k) != v:
            raise Refusal(f"R.{k} = {getattr(R, k)!r}, committed value {v!r}")
    pre["gpu"] = require_gpu() if want_gpu else gpu_info()
    try:
        R.redteam()
        U.redteam()
    except AssertionError as e:
        raise Refusal(f"the committed red team failed: {e}") from e
    pf, pc, pr = R.nparams(True), R.nparams(False), U.nparams(U.RealClassifier())
    if pr != pf or abs(pf - pc) / pf > 0.002:
        raise Refusal(f"parameter counts free={pf} clamped={pc} real2={pr}")
    pre["params"] = {"free": pf, "clamped": pc, "real2": pr}
    pre["rng_sharing"] = assert_rng_sharing(R, U)
    pre["smoke"] = bool(R.SMOKE)
    pre["preflight_ok"] = True
    return pre


def preflight(R, U, *, want_gpu: bool = True) -> dict:
    """Refuse before data if any check of PREREG section 9.4 fails; everything here rides in every leaf."""
    return preflight_torch(R, U, preflight_pure(), want_gpu=want_gpu)


class _Recorder:
    """Wraps R.test_acc for one run: the committed function computes, this records."""

    def __init__(self, original):
        self.original, self.values = original, []

    def __call__(self, m, xte, yte, n=None, bs=500):
        v = self.original(m, xte, yte, n, bs)
        self.values.append(float(v))
        return v


class _Tee(io.TextIOBase):
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
        return len(s)

    def flush(self):
        for st in self.streams:
            st.flush()


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def one_run(R, U, r: dict, data, pre: dict, derivation: str | None, beacon: str | None) -> dict:
    (xtr, ytr), (xte, yte) = data
    import torch
    saved = {k: getattr(R, k) for k in ("PERM_SEED", "STEPS", "EVAL_EVERY", "H", "D_SSM")}
    original_test_acc = R.test_acc
    rec = _Recorder(original_test_acc)
    buf = io.StringIO()
    try:
        R.PERM_SEED, R.STEPS, R.H, R.D_SSM = r["perm"], r["steps"], r["h"], r["h"]
        R.EVAL_EVERY = 100 if R.SMOKE else EVAL_EVERY
        R.test_acc = rec
        t0 = time.time()
        with contextlib.redirect_stdout(_Tee(buf, sys.__stdout__)):
            if r["arm"] == "real2":
                m = U.train_real2(r["seed"], xtr, ytr, xte, yte)
            else:
                m = R.train(r["arm"] == "free", r["seed"], xtr, ytr, xte, yte)
            final = float(original_test_acc(m, xte, yte))
            reliance = None
            if r["arm"] == "free":
                for b in m.blocks:
                    b.ssm.clamp_theta = True
                reliance = round(final - float(original_test_acc(m, xte, yte)), 4)
                for b in m.blocks:
                    b.ssm.clamp_theta = False
        seconds = time.time() - t0
        curve = rec.values[:]
        losses = []
        for tok in _LOSS_RE.findall(buf.getvalue()):
            try:
                losses.append(float(tok))
            except ValueError:
                losses.append(float("nan"))
        nonfinite = any(not math.isfinite(v) for v in losses)
        leaf = {"schema": LEAF_SCHEMA, "perm": r["perm"], "arm": r["arm"], "seed": r["seed"], "steps": r["steps"],
                "h": r["h"], "perm_derivation": None if r["perm"] == ANCHOR_PERM else derivation,
                "beacon": None if r["perm"] == ANCHOR_PERM else beacon,
                "read_back": {k: getattr(R, k) for k in READ_BACK_KEYS},
                "final_acc": round(final, 4), "curve_every": R.EVAL_EVERY, "curve": [round(v, 4) for v in curve],
                "curve_final_mismatch": bool((not R.SMOKE) and (not curve or abs(curve[-1] - final) > 1e-9)),
                "losses": losses, "nonfinite_loss": nonfinite,
                "degenerate": bool(nonfinite or final <= DEGENERATE_ACC),
                "within_model_theta0_reliance": reliance, "seconds": round(seconds, 1),
                "imports": pre["imports"], "data_sha256": pre["data"]["sha256"],
                "prereg_blob_sha256": pre["prereg"]["lf_sha256"], "prereg_commit": pre.get("prereg_commit"),
                "preflight_ok": pre.get("preflight_ok") is True, "gpu": pre.get("gpu"), "torch": torch.__version__,
                "smoke": bool(R.SMOKE), "wall_clock": _now()}
        del m
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return leaf
    finally:
        for k, v in saved.items():
            setattr(R, k, v)
        R.test_acc = original_test_acc


def _read_chain() -> list[dict]:
    """The leaf log, line by line. A torn last line (a crash inside an append) is refused by name, never repaired
    silently (PREREG section 15.3)."""
    if not CHAIN_FILE.exists():
        return []
    text = CHAIN_FILE.read_text(encoding="utf-8")
    if text and not text.endswith("\n"):
        raise Refusal(f"{CHAIN_FILE.name} does not end in a newline: its last append was torn; repair it by hand and "
                      f"report the repair in the RESULT")
    out = []
    for i, ln in enumerate(text.splitlines(), 1):
        if not ln.strip():
            continue
        try:
            out.append(json.loads(ln))
        except json.JSONDecodeError as e:
            raise Refusal(f"{CHAIN_FILE.name} line {i} does not parse ({e.msg}); repair it by hand and report the "
                          f"repair in the RESULT") from e
    return out


def _fsynced_write(path: Path, data: bytes, mode: str) -> None:
    with open(path, mode) as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())


def write_leaf(r: dict, leaf: dict) -> Path:
    """PREREG section 15.3: the log line is appended and fsynced, then the leaf is written to <leaf>.tmp, fsynced and
    renamed into place. A crash between the two leaves a log line with no leaf; the night re-runs that run and the
    receipt lists the name under leaf_chain.rewrites. No crash leaves a leaf that the log does not name."""
    RUNS_DIR.mkdir(exist_ok=True)
    raw = json.dumps(leaf, indent=1).encode("utf-8")
    path = leaf_path(r)
    entries = _read_chain()
    prev = entries[-1]["entry_id"] if entries else "0" * 64
    e = chain_entry(prev, leaf_name(r), sha256_bytes(raw), len(entries) + 1, _now())
    _fsynced_write(CHAIN_FILE, (json.dumps(e) + "\n").encode("utf-8"), "ab")
    tmp = path.with_name(path.name + ".tmp")
    _fsynced_write(tmp, raw, "wb")
    os.replace(tmp, path)
    if path.read_bytes() != raw:
        raise Refusal(f"{path.name} does not read back as written")
    return path


def _data_for(R, perm: int, h: int, cache: dict):
    key = (perm, h)
    if key not in cache:
        saved = (R.PERM_SEED, R.H, R.D_SSM)
        try:
            R.PERM_SEED, R.H, R.D_SSM = perm, h, h
            cache[key] = R.load_data()
        finally:
            R.PERM_SEED, R.H, R.D_SSM = saved
    return cache[key]


def run_night(args) -> int:
    """PREREG section 15.4's order: every check that needs no torch, then the runners, the GPU guard, the red team."""
    if args.night in (2, 3) and not (args.beacon or args.anchors or args.fallback_perms):
        raise Refusal(f"night {args.night} runs the held-out permutations: give --beacon <hex> from a styxx.clock "
                      f"verify line reading ANCHORED for this PREREG, --anchors <anchors.jsonl>, or --fallback-perms")
    pre = preflight_pure()
    if not SMOKE_JSON.exists():
        raise Refusal(f"{SMOKE_JSON.name} is absent: run --smoke before a night (it measures the per-run time)")
    smoke_rec = json.loads(SMOKE_JSON.read_text(encoding="utf-8"))
    per_4000 = float(smoke_rec["projected_min_per_4000"])
    sel = resolve_perms(args.beacon, args.anchors, args.fallback_perms)
    todo = [r for r in schedule(args.night, sel["perms"], args.width) if not leaf_path(r).exists()]
    hours = sum(per_run_minutes(r, per_4000) for r in todo) / 60.0
    print(f"night {args.night}{' (width stage)' if args.width else ''}: {len(todo)} runs to do, projected {hours:.1f} h "
          f"at {per_4000:.1f} min per 4000-step run (smoke); measured 2026-10-11: {PER_RUN_MIN_MEASURED} min; "
          f"derivation {sel['derivation']} beacon {sel['beacon']} perms {sel['perms']}", flush=True)
    if not args.accept_schedule:
        if per_4000 > PER_RUN_MIN_REFUSE:
            raise Refusal(f"the smoke projects {per_4000:.1f} min per 4000-step run, over the {PER_RUN_MIN_REFUSE} min "
                          f"threshold; re-run with --accept-schedule to proceed with the schedule printed above")
        if hours > NIGHT_HOURS_REFUSE:
            raise Refusal(f"this night projects {hours:.1f} h, over the {NIGHT_HOURS_REFUSE} h window; re-run with "
                          f"--accept-schedule to proceed (leaves persist; a spilled night is finished the next)")
    R, U = _load_runners()
    if R.SMOKE:
        raise Refusal("--smoke and --night together: a night is never a smoke")
    pre = preflight_torch(R, U, pre)
    cache: dict = {}
    for r in todo:
        print(f"--- perm {r['perm']} {r['arm']} seed {r['seed']} steps {r['steps']} h {r['h']} ---", flush=True)
        data = _data_for(R, r["perm"], r["h"], cache)
        leaf = one_run(R, U, r, data, pre, sel["derivation"], sel["beacon"])
        p = write_leaf(r, leaf)
        print(f"    final {leaf['final_acc']:.4f}  degenerate={leaf['degenerate']}  ({leaf['seconds']:.0f}s)  wrote {p.name}",
              flush=True)
    print(f"night {args.night} done: {len(todo)} runs written", flush=True)
    return 0


def smoke(args) -> int:
    pre = preflight_pure()
    R, U = _load_runners()
    if not R.SMOKE:
        raise Refusal("run_pmnist_ablation did not see --smoke on the command line")
    sys.path.insert(0, str(ROOT))
    from styxx.protocol import Experiment
    pre = preflight_torch(R, U, pre)
    exp = Experiment(PREREG, repo_root=ROOT)
    low = min(fallback_perms())
    plan = [_run(ANCHOR_PERM, "free", 0, R.STEPS), _run(ANCHOR_PERM, "clamped", 0, R.STEPS),
            _run(ANCHOR_PERM, "real2", 0, R.STEPS), _run(low, "free", 0, R.STEPS)]
    cache: dict = {}
    leaves = []
    for r in plan:
        data = _data_for(R, r["perm"], r["h"], cache)           # the reload sits outside the timed window
        leaves.append(one_run(R, U, r, data, pre, DERIVATION_FALLBACK, None))
    secs = [l["seconds"] for l in leaves]
    projected = float(np.median(secs)) * (STEPS_MAIN / R.STEPS) / 60.0
    v = exp.score({}, smoke=True)
    out = {"smoke": True, "prereg": PREREG.name, "prereg_blob_sha256": PREREG_BLOB_SHA256, "preflight": pre,
           "runs": leaves, "seconds_per_run": secs, "steps_per_run": R.STEPS,
           "projected_min_per_4000": round(projected, 2), "measured_min_per_4000_2026_10_11": PER_RUN_MIN_MEASURED,
           "refuse_above_min": PER_RUN_MIN_REFUSE, "verdict": v.verdict, "gates_sha256": v.gates_sha256,
           "prereg_commit": v.prereg_commit, "wall_clock": _now()}
    SMOKE_JSON.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"smoke: projected {out['projected_min_per_4000']} min per 4000-step run; verdict {v.verdict}; wrote {SMOKE_JSON.name}")
    return 0


def aggregate(args) -> int:
    sys.path.insert(0, str(ROOT))
    from styxx.protocol import Experiment, PrologueError, GateSpecError
    leaves = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(RUNS_DIR.glob("p*.json"))] if RUNS_DIR.exists() else []
    leaves = [l for l in leaves if not l.get("smoke")]
    # each leaf hashed with CRLF folded to LF (a leaf as written holds no CR byte), so a checkout made before the -text
    # pin of section 15.3 still matches the log: the rule section 1 applies to the imported runners
    current = ({p.name: lf_sha256(p) for p in sorted(RUNS_DIR.glob("p*.json"))} if RUNS_DIR.exists() else {})
    chain = verify_chain(_read_chain(), current)
    sel = resolve_perms(args.beacon, args.anchors, args.fallback_perms)
    if sel["derivation"] is None:
        seen = {(l.get("perm_derivation"), l.get("beacon")) for l in leaves if l["perm"] != ANCHOR_PERM}
        if len(seen) == 1:
            d, b = next(iter(seen))
            sel = {"derivation": d, "beacon": b,
                   "perms": derive_perms(b) if d == DERIVATION_BEACON and is_beacon_hex(b or "") else
                   fallback_perms() if d == DERIVATION_FALLBACK else None, "anchor_line": None}
    perms = [ANCHOR_PERM] + (sel["perms"] or [])
    disk = {"runners": all(v["ok"] for v in check_frozen_runners(HERE).values()),
            "data": check_data_pin(DATA_FILE)["ok"], "prereg": check_prereg_bytes(PREREG)["ok"]}
    out = aggregate_leaves(leaves, perms, sel["derivation"], sel["beacon"], chain, disk)
    out["anchor_line"] = sel.get("anchor_line")
    try:
        exp = Experiment(PREREG, repo_root=ROOT)
        v = exp.score(out)
        out.update({"verdict": v.verdict, "gates": v.gates, "gates_sha256": v.gates_sha256, "prereg_commit": v.prereg_commit,
                    "vacuous_gates": v.vacuous_gates, "undeclared_power_gates": v.undeclared_power_gates})
    except (PrologueError, GateSpecError) as e:              # an uncommitted or edited PREREG scores nothing; say so
        out.update({"verdict": None, "scoring_refused": f"{type(e).__name__}: {e}"})
    out["wall_clock"] = _now()
    RESULT_JSON.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({"result": out["result"], "plumbing": out["plumbing"], "reported": out["reported"]}, indent=1))
    print(f"\n===== VERDICT: {out.get('verdict')} =====  wrote {RESULT_JSON.name}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--derive-perms", action="store_true")
    ap.add_argument("--night", type=int)
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--beacon", help="64 lowercase hex: the beacon styxx.clock verify prints for this PREREG's seal")
    ap.add_argument("--anchors", help="papers/charon/anchors.jsonl: verified here, the beacon taken from this PREREG's line")
    ap.add_argument("--fallback-perms", action="store_true", help="the labelled sha256-of-tag derivation (no anchor)")
    ap.add_argument("--accept-schedule", action="store_true")
    ap.add_argument("--width", action="store_true", help="optional stage E (decision 4), reported only")
    args = ap.parse_args(argv)
    try:
        if args.derive_perms:
            sel = resolve_perms(args.beacon, args.anchors, args.fallback_perms or not (args.beacon or args.anchors))
            for k, s in enumerate(sel["perms"], 1):
                print(k, s)
            print(f"derivation: {sel['derivation']}  source: {sel['beacon'] or FALLBACK_TAG}")
            return 0
        if args.smoke:
            return smoke(args)
        if args.aggregate:
            return aggregate(args)
        if args.night is not None:
            return run_night(args)
    except Refusal as e:
        print(f"refused: {e}", file=sys.stderr, flush=True)
        return 2
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
