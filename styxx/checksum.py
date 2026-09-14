# -*- coding: utf-8 -*-
"""styxx.checksum — a checksum for model behavior (v8 layer 1, prototype).

A fingerprint is what a model does on a fixed, hashed set of canary items, measured in the
continuous quantity the model itself exposes: log-probabilities. Two fingerprints of the same
weights are identical (deterministic decoding); two fingerprints of different weights differ by
an amount this module measures, with a bootstrap resolution, and never by an amount it invents.

The interface is a single callable so any model — a Hugging Face checkpoint, an API with
logprobs, a scripted mock — can be fingerprinted the same way:

    probe(prompt: str, continuation: str) -> Probe
        Probe.cont_logprobs : list[float]   teacher-forced log-prob of each continuation token
        Probe.next_logprobs : np.ndarray    log-probs over the vocabulary for the first
                                            continuation position (the model's next-token belief)

What this module claims, and pins in tests/test_checksum.py:
  - the same probe twice gives distance exactly 0 and verdict SAME;
  - a small perturbation gives a small distance, a different model a large one, monotonically;
  - a degenerate probe (constant outputs) is INCONCLUSIVE, never SAME — the degeneracy guard
    the frequency arc's real-model bridge taught us to run before any verdict;
  - the canary set is hashed into every cert, so two certs are comparable only if the hash agrees.

What it does not claim: that a distance is "meaningful". Thresholds for that are the business of a
preregistered use, not of this module. It reports magnitudes and resolutions.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field, asdict
from typing import Callable, Sequence

import numpy as np

# --------------------------------------------------------------------------------------- canaries
# 48 short items across kinds. Continuations are the behavior under test; keep them short.
CANARIES: list[tuple[str, str, str]] = [
    ("geo-01", "The capital of France is", " Paris"),
    ("geo-02", "The largest ocean on Earth is the", " Pacific"),
    ("geo-03", "Mount Everest is in the", " Himalayas"),
    ("geo-04", "The Nile flows into the", " Mediterranean"),
    ("geo-05", "Tokyo is the capital of", " Japan"),
    ("geo-06", "The Sahara is a desert in", " Africa"),
    ("sci-01", "Water boils at 100 degrees", " Celsius"),
    ("sci-02", "The chemical symbol for gold is", " Au"),
    ("sci-03", "Light travels faster than", " sound"),
    ("sci-04", "The powerhouse of the cell is the", " mitochondria"),
    ("sci-05", "Humans have 23 pairs of", " chromosomes"),
    ("sci-06", "The planet closest to the Sun is", " Mercury"),
    ("math-01", "Two plus two equals", " four"),
    ("math-02", "The square root of 81 is", " 9"),
    ("math-03", "Seven times eight is", " 56"),
    ("math-04", "The next prime after 7 is", " 11"),
    ("math-05", "Half of 100 is", " 50"),
    ("math-06", "Ten minus three is", " seven"),
    ("code-01", "In Python, a list is created with square", " brackets"),
    ("code-02", "def add(a, b):\n    return a", " +"),
    ("code-03", "To print in Python you call", " print"),
    ("code-04", "A for loop in Python ends its header with a", " colon"),
    ("code-05", "The file extension for a Python script is", " .py"),
    ("code-06", "In JSON, keys must be", " strings"),
    ("lang-01", "The opposite of hot is", " cold"),
    ("lang-02", "The plural of mouse is", " mice"),
    ("lang-03", "A synonym for big is", " large"),
    ("lang-04", "Monday, Tuesday, Wednesday,", " Thursday"),
    ("lang-05", "Red, orange, yellow, green,", " blue"),
    ("lang-06", "Once upon a", " time"),
    ("logic-01", "All cats are animals. Tom is a cat. Therefore Tom is an", " animal"),
    ("logic-02", "If it rains the ground gets wet. It rained. So the ground is", " wet"),
    ("logic-03", "The day after Monday is", " Tuesday"),
    ("logic-04", "The day before two days after Monday is", " Tuesday"),
    ("logic-05", "Bigger than a mouse, smaller than a horse: a", " dog"),
    ("logic-06", "A bachelor is an unmarried", " man"),
    ("inst-01", "Answer with one word. The color of the sky on a clear day:", " blue"),
    ("inst-02", "Translate to French: thank you ->", " merci"),
    ("inst-03", "Complete the list with a fruit: apple, banana,", " orange"),
    ("inst-04", "Say yes or no. Is fire hot?", " Yes"),
    ("inst-05", "Reverse the word 'cat':", " tac"),
    ("inst-06", "Repeat after me: hello. You say:", " hello"),
    ("hist-01", "The Second World War ended in", " 1945"),
    ("hist-02", "The first person to walk on the Moon was Neil", " Armstrong"),
    ("hist-03", "The Roman Empire's capital was", " Rome"),
    ("hist-04", "The Declaration of Independence was signed in", " 1776"),
    ("hist-05", "The Great Wall is in", " China"),
    ("hist-06", "Shakespeare wrote Romeo and", " Juliet"),
]


def canary_sha256(canaries: Sequence[tuple[str, str, str]] = CANARIES) -> str:
    blob = json.dumps([list(c) for c in canaries], ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


# ------------------------------------------------------------------------------------------ probe
@dataclass
class Probe:
    cont_logprobs: list[float]
    next_logprobs: np.ndarray  # (V,) log-probs at the first continuation position


ProbeFn = Callable[[str, str], Probe]


def hf_probe(model, tokenizer) -> ProbeFn:
    """A probe for a Hugging Face causal LM on CPU/GPU, teacher-forced, deterministic."""
    import torch

    def probe(prompt: str, continuation: str) -> Probe:
        p_ids = tokenizer(prompt, return_tensors="pt").input_ids
        c_ids = tokenizer(continuation, return_tensors="pt", add_special_tokens=False).input_ids
        ids = torch.cat([p_ids, c_ids], dim=1)
        with torch.no_grad():
            logits = model(input_ids=ids).logits[0].float()
        logp = torch.log_softmax(logits, dim=-1)
        n_p = p_ids.shape[1]
        cont = [logp[n_p - 1 + t, c_ids[0, t]].item() for t in range(c_ids.shape[1])]
        return Probe(cont_logprobs=cont, next_logprobs=logp[n_p - 1].cpu().numpy())

    return probe


# ------------------------------------------------------------------------------------ fingerprint
@dataclass
class Fingerprint:
    model_id: str
    canary_sha256: str
    tokenizer_id: str            # anything that identifies the tokenization; "" if unknown
    ids: list[str]
    mean_lp: np.ndarray          # (n,) mean log-prob per continuation token, per item
    rdm: np.ndarray              # (n, n) correlation distance between items' next-token beliefs
    n_tokens: list[int]
    created: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    kind: str = "full"           # "full": teacher-forced per-token mean log-prob; "topk": the API variant
    k: int = 0                   # for "topk": the k the belief vectors were built from; comparable only at equal k
    draw: dict | None = None     # styxx.beacon draw record (pool hash, beacon, n) when the canaries were drawn; None for a hand set

    def to_json(self) -> dict:
        """The written form: values rounded to 1e-6, -0.0 normalised to 0.0 (a sign on float noise
        would otherwise hash differently on another machine), the rdm diagonal written as exactly 0,
        and a hash over the written rdm and the written mean_lp — the two things a stranger compares."""
        d = asdict(self)
        d["mean_lp"] = [round(float(x), 6) + 0.0 for x in self.mean_lp]
        d["rdm"] = [[(0.0 if i == j else round(float(x), 6) + 0.0) for j, x in enumerate(row)] for i, row in enumerate(self.rdm)]
        d["rdm_sha256"] = hashlib.sha256(json.dumps(d["rdm"], separators=(",", ":")).encode()).hexdigest()
        d["mean_lp_sha256"] = hashlib.sha256(json.dumps(d["mean_lp"], separators=(",", ":")).encode()).hexdigest()
        return d

    def written_hashes(self) -> tuple[str, str]:
        j = self.to_json()
        return j["rdm_sha256"], j["mean_lp_sha256"]


def _check_draw(draw, canaries) -> dict | None:
    """A draw record must name the canaries it produced, and it must be TRUE: the beacon and pool it
    names are re-run and must reproduce exactly these items. Before 2026-09-13 (evening) only the
    canary hash was compared, so a record whose canary hash matched but whose beacon and pool were
    lies was accepted and the cert digested the lie — found by the lab's own probe before the red
    team reached it."""
    if draw is None:
        return None
    want = canary_sha256(canaries)
    if not isinstance(draw, dict) or draw.get("canary_sha256") != want:
        raise ValueError("the draw record's canary_sha256 is not the hash of these canaries; "
                         "use the items styxx.beacon.draw returned with its record")
    for key in ("pool_sha256", "beacon", "n"):
        if key not in draw:
            raise ValueError(f"the draw record lacks {key!r}")
    from .beacon import POOL, pool_sha256, select   # lazy: beacon imports this module
    if draw["pool_sha256"] != pool_sha256(POOL):
        raise ValueError("the draw record names a pool this package does not have; the draw cannot be re-derived")
    if int(draw["n"]) != len(list(canaries)):
        raise ValueError(f"the draw record says n={draw['n']} but {len(list(canaries))} canaries were given")
    rederived = select(str(draw["beacon"]), int(draw["n"]), POOL)
    if canary_sha256(rederived) != want:
        raise ValueError("the draw record's beacon does not produce these canaries from the named pool; "
                         "the record is not the draw that made this set")
    return dict(draw)


def fingerprint(probe: ProbeFn, model_id: str, canaries=CANARIES, tokenizer_id: str = "", draw: dict | None = None) -> Fingerprint:
    draw = _check_draw(draw, canaries)
    ids, mean_lp, n_tok, beliefs = [], [], [], []
    for cid, prompt, cont in canaries:
        pr = probe(prompt, cont)
        ids.append(cid)
        mean_lp.append(float(np.mean(pr.cont_logprobs)))
        n_tok.append(len(pr.cont_logprobs))
        beliefs.append(np.asarray(pr.next_logprobs, dtype=np.float64))
    B = np.stack(beliefs)                      # (n, V)
    B = B - B.mean(1, keepdims=True)
    B /= (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    rdm = 1.0 - B @ B.T
    return Fingerprint(model_id=model_id, canary_sha256=canary_sha256(canaries), tokenizer_id=tokenizer_id,
                       ids=ids, mean_lp=np.asarray(mean_lp), rdm=rdm, n_tokens=n_tok, draw=draw)


# --------------------------------------------------------------------------------------- distance
@dataclass
class Distance:
    mean_abs_nats: float         # mean |Δ mean log-prob| per item, in nats per token
    corr_dist: float             # 1 - pearson(mean_lp_a, mean_lp_b)
    rdm_r: float                 # pearson of the two RDM upper triangles (belief geometry agreement)
    ci_mean_abs: tuple[float, float]
    ci_rdm_r: tuple[float, float]   # a SUBSAMPLING interval (unique bootstrap indices), not a multiplicity bootstrap
    n_items: int
    n_boot: int
    verdict: str                 # SAME | DRIFT | INCONCLUSIVE
    reason: str
    floor_measured: float = float("nan")   # what the caller passed (the in-situ null floor)
    floor_effective: float = float("nan")  # what the verdict was graded against: max(measured, RESOLUTION_NATS)
    seed: int = 0                          # the bootstrap seed, so the interval re-derives


DEGENERATE_STD = 1e-6           # a fingerprint with no variation across items measures nothing
RESOLUTION_NATS = 1e-4          # below this, two deterministic fingerprints are the same weights


def _rdm_r(a: np.ndarray, b: np.ndarray, idx=None) -> float:
    if idx is not None:
        a = a[np.ix_(idx, idx)]; b = b[np.ix_(idx, idx)]
    iu = np.triu_indices(a.shape[0], 1)
    x, y = a[iu], b[iu]
    if x.std() < 1e-12 or y.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def distance(a: Fingerprint, b: Fingerprint, n_boot: int = 2000, seed: int = 20260913,
             floor_nats: float = RESOLUTION_NATS) -> Distance:
    """Compare two fingerprints.

    floor_nats is the null floor: the largest mean |Δ log-prob| seen between fingerprints of the
    SAME weights under the serving conditions in use (see null_floor). On deterministic cpu it is
    exactly 0 and the default resolution applies; on GPUs, batched or sampled serving it is not 0
    and must be measured in situ before any DRIFT verdict means anything.
    """
    if a.canary_sha256 != b.canary_sha256:
        raise ValueError("fingerprints were taken on different canary sets; not comparable")
    if a.tokenizer_id != b.tokenizer_id:
        raise ValueError("fingerprints were taken with different tokenizers; mean log-prob per token is "
                         "not comparable across tokenizations (the belief-geometry rdm still is)")
    if a.kind != b.kind:
        raise ValueError(f"a {a.kind} fingerprint and a {b.kind} fingerprint measure different quantities; not comparable")
    if (a.draw or b.draw) and a.draw != b.draw:
        raise ValueError("the fingerprints were drawn under different beacons or pools (or one was a hand set); "
                         "not comparable")
    if a.kind == "topk" and a.k != b.k:
        raise ValueError(f"top-k fingerprints at k={a.k} and k={b.k} are not comparable")
    if not a.tokenizer_id or not b.tokenizer_id:
        raise ValueError("a fingerprint with no tokenizer_id cannot be compared: name the tokenization")
    if list(a.ids) != list(b.ids):
        raise ValueError("the fingerprints list different items (or a different order); the canary hash was copied")
    floor_measured = float(floor_nats)
    if not np.isfinite(floor_measured) or floor_measured < 0:
        raise ValueError(f"floor_nats must be a finite, non-negative measured floor; got {floor_nats!r}")
    floor = max(floor_measured, RESOLUTION_NATS)
    n = len(a.ids)
    da = np.abs(a.mean_lp - b.mean_lp)
    mean_abs = float(da.mean())
    iu = np.triu_indices(n, 1)
    if (a.mean_lp.std() < DEGENERATE_STD or b.mean_lp.std() < DEGENERATE_STD
            or a.rdm[iu].std() < 1e-12 or b.rdm[iu].std() < 1e-12):
        return Distance(mean_abs, float("nan"), float("nan"), (float("nan"),) * 2, (float("nan"),) * 2,
                        n, 0, "INCONCLUSIVE", "degenerate: a fingerprint has no variation across items "
                        "(in its mean log-probs or in its belief geometry)", floor_measured, floor, seed)
    corr_dist = float(1.0 - np.corrcoef(a.mean_lp, b.mean_lp)[0, 1])
    rdm_r = _rdm_r(a.rdm, b.rdm)
    rng = np.random.default_rng(seed)
    boots_abs, boots_r = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        boots_abs.append(float(da[idx].mean()))
        boots_r.append(_rdm_r(a.rdm, b.rdm, np.unique(idx)))
    ci_abs = (float(np.percentile(boots_abs, 2.5)), float(np.percentile(boots_abs, 97.5)))
    br = np.array([x for x in boots_r if not np.isnan(x)])
    ci_r = (float(np.percentile(br, 2.5)), float(np.percentile(br, 97.5))) if len(br) else (float("nan"),) * 2
    if ci_abs[1] < floor:
        verdict, reason = "SAME", f"upper 95% bound {ci_abs[1]:.2e} nats/token is below the floor {floor:g}"
    elif ci_abs[0] > floor:
        verdict, reason = "DRIFT", f"lower 95% bound {ci_abs[0]:.4f} nats/token is above the floor {floor:g}"
    else:
        verdict, reason = "INCONCLUSIVE", f"the 95% interval straddles the floor {floor:g}"
    return Distance(mean_abs, corr_dist, rdm_r, ci_abs, ci_r, n, n_boot, verdict, reason, floor_measured, floor, seed)


def null_floor(fingerprints: Sequence[Fingerprint]) -> float:
    """The largest mean |Δ log-prob| between any two fingerprints of the SAME weights, taken under the
    serving conditions in use. Measure this first; pass it as floor_nats to distance()."""
    fs = list(fingerprints)
    if len(fs) < 2:
        raise ValueError("a null floor needs at least two fingerprints of the same weights")
    worst = 0.0
    for i in range(len(fs)):
        for j in range(i + 1, len(fs)):
            worst = max(worst, float(np.abs(fs[i].mean_lp - fs[j].mean_lp).mean()))
    return worst


# ------------------------------------------------------------------------------- top-k (api) variant
TopKFn = Callable[[str], list[tuple[str, float]]]   # prompt -> [(token_text, logprob)] for the next token


def fingerprint_topk(topk: TopKFn, model_id: str, canaries=CANARIES, tokenizer_id: str = "",
                     k_min: int = 5, draw: dict | None = None) -> Fingerprint:
    """The API variant. Chat APIs do not teacher-force, but most return top-k next-token log-probs.
    Per item: the log-prob of the gold first token if it is in the top-k, else the item's k-th
    log-prob (a floor, marked); the belief vector is the top-k over the union of tokens seen
    across items, missing entries filled with the item's floor. Coarser than the local variant
    and comparable only with other top-k fingerprints of the same k; the cert carries `topk`."""
    rows, floors, gold_lp = [], [], []
    for cid, prompt, cont in canaries:
        tk = topk(prompt)
        if len(tk) < k_min:
            raise ValueError(f"{cid}: fewer than {k_min} top-k entries returned")
        d = {t: float(lp) for t, lp in tk}
        floor = min(d.values())
        rows.append(d); floors.append(floor)
        first = cont[:1] + cont[1:].split(" ")[0] if cont.startswith(" ") else cont.split(" ")[0]
        gold_lp.append(d.get(first, d.get(first.strip(), floor)))
    vocab = sorted({t for d in rows for t in d})
    B = np.array([[d.get(t, fl) for t in vocab] for d, fl in zip(rows, floors)], dtype=np.float64)
    B = B - B.mean(1, keepdims=True)
    B /= (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    rdm = 1.0 - B @ B.T
    k = min(len(d) for d in rows)
    fp = Fingerprint(model_id=f"{model_id} [topk={k}]", canary_sha256=canary_sha256(canaries),
                     tokenizer_id=tokenizer_id or "topk", ids=[c[0] for c in canaries], mean_lp=np.asarray(gold_lp),
                     rdm=rdm, n_tokens=[1] * len(canaries), kind="topk", k=k, draw=_check_draw(draw, canaries))
    return fp


# ------------------------------------------------------------------------------------------- cert
def _finite(v):
    """JSON without NaN tokens: a strict parser must be able to load a cert and re-derive its digest."""
    if isinstance(v, float) and not np.isfinite(v):
        return None
    if isinstance(v, dict):
        return {k: _finite(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_finite(x) for x in v]
    return v


def cert(a: Fingerprint, b: Fingerprint, d: Distance, note: str = "") -> dict:
    """A comparison certificate: the comparison, and the written hashes of the two fingerprints it
    grades, so a stranger holding the cert and the two fingerprint files can re-derive the verdict.

    v1 (2026-09-13): v0's digest covered no hash of either fingerprint, no seed and no floor — two
    different fingerprint pairs with the same model_id strings produced the same digest, and an
    INCONCLUSIVE cert carried a bare NaN token no strict parser accepts. v1 adds a.rdm_sha256 and
    a.mean_lp_sha256 (and b's), the seed, and the measured and effective floors inside the digest,
    and writes non-finite values as null. Committed v0 certs stay as history under their own schema.
    """
    a_rdm, a_lp = a.written_hashes()
    b_rdm, b_lp = b.written_hashes()
    body = {
        "schema": "styxx.checksum/compare/v1",
        "canary_sha256": a.canary_sha256,
        "tokenizer_id": a.tokenizer_id,
        "kind": a.kind,
        "draw": a.draw,              # the beacon draw record when the canaries were drawn; None for the hand set
        "n_items": d.n_items,
        "a": {"model_id": a.model_id, "rdm_sha256": a_rdm, "mean_lp_sha256": a_lp},
        "b": {"model_id": b.model_id, "rdm_sha256": b_rdm, "mean_lp_sha256": b_lp},
        "distance": _finite({k: v for k, v in asdict(d).items()}),
        "floor_measured_nats": _finite(d.floor_measured),
        "floor_effective_nats": _finite(d.floor_effective),
        "seed": d.seed,
        "resolution_nats": RESOLUTION_NATS,
        "note": note,
    }
    blob = json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    body["digest"] = hashlib.sha256(blob).hexdigest()   # over the comparison and the written fingerprint hashes — re-runs on the same bytes agree
    body["created"] = {"a": a.created, "b": b.created}   # timestamps ride outside the digest
    return body
