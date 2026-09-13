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
    ids: list[str]
    mean_lp: np.ndarray          # (n,) mean log-prob per continuation token, per item
    rdm: np.ndarray              # (n, n) correlation distance between items' next-token beliefs
    n_tokens: list[int]
    created: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))

    def to_json(self) -> dict:
        d = asdict(self)
        d["mean_lp"] = [round(float(x), 6) for x in self.mean_lp]
        d["rdm_sha256"] = hashlib.sha256(np.ascontiguousarray(self.rdm, dtype=np.float64).tobytes()).hexdigest()
        d["rdm"] = [[round(float(x), 6) for x in row] for row in self.rdm]
        return d


def fingerprint(probe: ProbeFn, model_id: str, canaries=CANARIES) -> Fingerprint:
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
    return Fingerprint(model_id=model_id, canary_sha256=canary_sha256(canaries), ids=ids,
                       mean_lp=np.asarray(mean_lp), rdm=rdm, n_tokens=n_tok)


# --------------------------------------------------------------------------------------- distance
@dataclass
class Distance:
    mean_abs_nats: float         # mean |Δ mean log-prob| per item, in nats per token
    corr_dist: float             # 1 - pearson(mean_lp_a, mean_lp_b)
    rdm_r: float                 # pearson of the two RDM upper triangles (belief geometry agreement)
    ci_mean_abs: tuple[float, float]
    ci_rdm_r: tuple[float, float]
    n_items: int
    n_boot: int
    verdict: str                 # SAME | DRIFT | INCONCLUSIVE
    reason: str


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


def distance(a: Fingerprint, b: Fingerprint, n_boot: int = 2000, seed: int = 20260913) -> Distance:
    if a.canary_sha256 != b.canary_sha256:
        raise ValueError("fingerprints were taken on different canary sets; not comparable")
    n = len(a.ids)
    da = np.abs(a.mean_lp - b.mean_lp)
    mean_abs = float(da.mean())
    if a.mean_lp.std() < DEGENERATE_STD or b.mean_lp.std() < DEGENERATE_STD:
        return Distance(mean_abs, float("nan"), float("nan"), (float("nan"),) * 2, (float("nan"),) * 2,
                        n, 0, "INCONCLUSIVE", "degenerate: a fingerprint has no variation across items")
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
    if ci_abs[1] < RESOLUTION_NATS:
        verdict, reason = "SAME", f"upper 95% bound {ci_abs[1]:.2e} nats/token is below resolution {RESOLUTION_NATS:g}"
    elif ci_abs[0] > RESOLUTION_NATS:
        verdict, reason = "DRIFT", f"lower 95% bound {ci_abs[0]:.4f} nats/token is above resolution {RESOLUTION_NATS:g}"
    else:
        verdict, reason = "INCONCLUSIVE", "the 95% interval straddles the resolution"
    return Distance(mean_abs, corr_dist, rdm_r, ci_abs, ci_r, n, n_boot, verdict, reason)


# ------------------------------------------------------------------------------------------- cert
def cert(a: Fingerprint, b: Fingerprint, d: Distance, note: str = "") -> dict:
    """A comparison certificate: everything needed to re-derive the verdict from the two fingerprints."""
    body = {
        "schema": "styxx.checksum/compare/v0",
        "canary_sha256": a.canary_sha256,
        "n_items": d.n_items,
        "a": {"model_id": a.model_id, "created": a.created},
        "b": {"model_id": b.model_id, "created": b.created},
        "distance": {k: v for k, v in asdict(d).items()},
        "resolution_nats": RESOLUTION_NATS,
        "note": note,
    }
    blob = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    body["digest"] = hashlib.sha256(blob).hexdigest()
    return body
