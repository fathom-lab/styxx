"""Loading the published verdict artifact, and the two detectors this calibration measures.

Everything here reads `papers/v8/first_verdict_2026_09_09/`, the published verdict on
`google/gemma-2-2b-it` (64-prompt battery, greedy, 16 new tokens, five bf16 runs, one fp16
comparison, a seven-entry log).  Nothing here re-runs a model.

The two detectors, stated precisely, because an uncalibrated detector is not a measurement:

DETECTOR A -- the k-item output-digest challenge.
    A second party draws k of the 64 item ids uniformly without replacement, runs those k prompts
    itself under the reference run's nuisance assignment, and compares its own output digest
    against the certificate's.  It is CAUGHT if any drawn item's digest differs from the honest
    value.  The honest values are the published certificate's, so detection is exactly a set
    intersection and its probability has a closed form; `montecarlo_catch` re-derives the same
    number by drawing, as a check on the formula rather than on the arithmetic.

    Cost: k forward passes.  This is the only detector in this file that requires a second party
    to spend anything.

DETECTOR B -- the internal-consistency battery, verbatim.
    `papers/v8/class_two_empty_2026_09_09/first_claim_battery.py`, imported and its `check_item`
    called, not reimplemented.  It reads one certificate and compares it only to itself.
    Cost: zero.  It needs no second party, no model and no prior log entry.

DETECTOR C -- the k-item FULL-RECORD challenge (an extension this calibration proposes, not part
    of the published system).  Same draw as A, but the challenger also compares `seq_logprob` and
    the recorded `topk` distributions exactly.  It is listed because A is blind to every mutation
    that leaves the emitted tokens alone, and C is what closes that -- at a false-alarm rate that
    this file measures rather than assumes.
"""
from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import math
import pathlib
import random
import sys

# .../<root>/papers/v8/sampled_challenge_2026_09_09/calibration/artifact.py
ROOT = pathlib.Path(__file__).resolve().parents[4]
VERDICT = ROOT / "papers" / "v8" / "first_verdict_2026_09_09"
BATTERY_PY = ROOT / "papers" / "v8" / "class_two_empty_2026_09_09" / "first_claim_battery.py"

BF16 = {
    0: VERDICT / "fp_bf16" / "fingerprint-canonical-73a09ffa3f1e.json",
    1: VERDICT / "fp_bf16" / "fingerprint-run1-8e13ac182b48.json",
    2: VERDICT / "fp_bf16" / "fingerprint-run2-5fffce6bd465.json",
    3: VERDICT / "fp_bf16" / "fingerprint-run3-243b385e9982.json",
    4: VERDICT / "fp_bf16" / "fingerprint-run4-d521ed3169d3.json",
}
FP16 = VERDICT / "fp_fp16" / "fingerprint-canonical-d92ebe2089f9.json"
BATTERY_CERT = VERDICT / "log" / "entries" / "000000" / "00000000.json"

N_ITEMS = 64
MAX_NEW = 16

# The digest fields a second party reproduces by running the prompt.  `token_ids_sha256` is the
# key the `exact` channel actually compares (styxx/v8/distances.py `_ids_key` prefers token_ids);
# `output_sha256` covers the rendered text, which no channel compares at all.
DIGEST_FIELDS = ("output_sha256", "token_ids_sha256")
FULL_FIELDS = ("output_sha256", "token_ids_sha256", "n_generated", "seq_logprob", "topk")


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def load_cert(path: pathlib.Path) -> dict:
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))


def items_of(cert: dict) -> dict:
    """`{item_id: item record}`.  The five runs store their items in different orders (the plan
    varied `item_order`), so every comparison in this calibration is keyed by item_id."""
    return {it["item_id"]: it for it in cert["body"]["items"]}


def roles_of() -> list:
    return load_cert(BATTERY_CERT)["body"]["items"]


def load_battery_module():
    """Import the published first-claim battery and hand back its `check_item`.

    The module prints a report at import time (it is a script).  That output is captured and
    discarded here; running it for its own sake is the job of the file itself, not of this one.
    """
    spec = importlib.util.spec_from_file_location("first_claim_battery", BATTERY_PY)
    mod = importlib.util.module_from_spec(spec)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- detector A / C

def differing(forged_items: dict, honest_items: dict, fields) -> list:
    """The item ids a challenger would find disagreeing, comparing `fields`.

    This is ground truth, not an estimate: the honest values are the published certificate's, and
    a challenger who reproduces the reference run reproduces them exactly -- which is a measured
    claim about this artifact, not an assumption.  See `same_batch_reproducibility` below.
    """
    out = []
    for iid, honest in honest_items.items():
        f = forged_items.get(iid)
        if f is None:
            out.append(iid)
            continue
        if any(f.get(k) != honest.get(k) for k in fields):
            out.append(iid)
    return sorted(out)


def catch_prob(n_bad: int, k: int, n: int = N_ITEMS) -> float:
    """P(a uniform k-subset of n hits at least one of n_bad marked items) = 1 - C(n-b,k)/C(n,k)."""
    if n_bad <= 0 or k <= 0:
        return 0.0
    if k > n - n_bad:
        return 1.0
    return 1.0 - math.comb(n - n_bad, k) / math.comb(n, k)


def montecarlo_catch(n_bad: int, k: int, trials: int = 20000, seed: int = 20260909,
                     n: int = N_ITEMS) -> float:
    """Draw the challenge `trials` times.  Only here to falsify `catch_prob`, not to replace it."""
    rng = random.Random(seed)
    pool = list(range(n))
    bad = set(range(n_bad))
    hit = 0
    for _ in range(trials):
        if bad & set(rng.sample(pool, k)):
            hit += 1
    return hit / trials


# --------------------------------------------------------------------------- detector B

def internal_verdict(cert: dict, check_item) -> tuple[bool, dict]:
    """Run the published battery over every item of `cert`.

    Returns `(caught, {predicate: count})`.  The battery is item-local by construction, so it is
    structurally unable to see any mutation that leaves every individual item internally
    consistent -- a copy, a swap, a relabelling.  That is a property of the detector and is
    reported as one.
    """
    fails: dict = {}
    for it in cert["body"]["items"]:
        for msg in check_item(it, MAX_NEW):
            key = msg.split(":")[0].split(" ")[0]
            fails[key] = fails.get(key, 0) + 1
    return bool(fails), fails


# --------------------------------------------------------------------------- false alarm

def same_batch_reproducibility() -> dict:
    """What the published log says about a challenger's chance of a spurious accusation.

    The five bf16 floor runs are five executions of the same battery on the same machine under
    assignments the plan fixed.  Every pair of them is a stand-in for an honest challenge: one
    party's bytes against another execution's.  The disagreement counts below are the false-alarm
    numerator for a challenge run under that pair's conditions.
    """
    runs = {i: (load_cert(p), items_of(load_cert(p))) for i, p in BF16.items()}
    out = []
    for i in range(5):
        for j in range(i + 1, 5):
            ci, ai = runs[i]
            cj, aj = runs[j]
            out.append({
                "pair": [i, j],
                "batch_size": [ci["recipe"]["decoding"]["batch_size"],
                               cj["recipe"]["decoding"]["batch_size"]],
                "item_order": [ci["body"]["nuisance"]["item_order"],
                               cj["body"]["nuisance"]["item_order"]],
                "digest_disagreements": len(differing(ai, aj, DIGEST_FIELDS)),
                "full_record_disagreements": len(differing(ai, aj, FULL_FIELDS)),
            })
    return {"pairs": out}


if __name__ == "__main__":
    m = load_battery_module()
    print("battery imported:", BATTERY_PY.name, "->", m.check_item.__name__)
    print(json.dumps(same_batch_reproducibility(), indent=1))
