"""Forged variants of the published fingerprint certificate.

Every mutation here is a move a dishonest issuer would actually make -- a claim they want to be
true, written into a record nobody else has run.  None of them is random noise, and the ones that
matter are built to survive the internal-consistency battery on purpose.  A calibration whose
forgeries are all caught has measured the generator, not the detector.

The target is `fp_bf16/fingerprint-canonical-73a09ffa3f1e.json`: 64 items, 37 of them with a
recorded distribution for every emitted token, 27 with a generation that runs past the last
recorded distribution and 193 emitted tokens carrying no distribution at all.  That asymmetry is
the substrate for classes 5, 6 and 9.

TWO FORMAT HOLES THE GENERATOR EXPLOITS, both found by writing these mutations rather than by
reading the schema:

  H1  Nothing ties `output_text` to `token_ids`.  The battery digests each separately (P3, P4) and
      no predicate detokenizes.  So the rendered answer a reader sees can be replaced wholesale
      while every number in the certificate stays honest -- and the `exact` channel will not
      notice either, because styxx/v8/distances.py compares `token_ids`, not text.
      Classes: text_only_rewrite, text_swap.

  H2  `seq_logprob` is tied to the recorded distributions by equality only where a distribution
      exists for every emitted token.  Past that the battery has one inequality (P8b: the score
      may not EXCEED the recorded prefix sum), so on the 27 partial items the score may be set to
      any value at or below the prefix.  The honest value already sits 0.03 to 3.4 nats below it.
      Classes: seqlp_partial_under, topk_partial_free.

Naming: `careful` variants recompute everything a competent forger would recompute; `careless`
variants are the same mutation with one field left stale.  Both are generated so the report can
say which column each is in, and the careless ones are the control that shows the battery works.
"""
from __future__ import annotations

import copy
import json
import math
import random

from artifact import DIGEST_FIELDS, MAX_NEW, sha

# A confident-looking top-5 that is a legal distribution: descending, mass exactly 1.0.
# NOT a sharpening.  Many published positions are already near-certain (top-1 log-probability as
# high as -8.6e-05), so rewriting them to log(0.90) = -0.105 LOWERS the recorded prefix sum there
# and raises it on the uncertain ones.  Which direction a given item moves is why the naive
# variant below is caught on some items and not on others, and the rates are measured, not assumed.
RESHAPE = [math.log(p) for p in (0.90, 0.05, 0.03, 0.015, 0.005)]

BODY_FIELDS = ("n_generated", "output_text", "output_sha256", "token_ids",
               "token_ids_sha256", "seq_logprob", "topk")

# Plausible replacement answers -- the point of H1 is that these need no relation to the tokens.
FAKE_TEXTS = [
    "Yes. \n", "No. \n", "42 \n", "blue \n", "True \n", "Paris \n",
    "```json\n{\"ok\": true}\n``` \n", "**Ascend** \n", "OK \n", "green \n",
]


def _set_text(item: dict, text: str) -> None:
    item["output_text"] = text
    item["output_sha256"] = sha(text.encode("utf-8"))


def _alt_text(item: dict, n: int) -> str:
    """A replacement answer that is not the one already there.

    Three of the ten replacement strings ("Yes. \\n", "blue \\n", "OK \\n") are genuine outputs of
    this battery, so a naive rewrite is sometimes a no-op.  A forgery that changes nothing is not
    an undetected forgery, and letting it into the population would understate the challenge.
    """
    cur = item.get("output_text")
    for j in range(len(FAKE_TEXTS)):
        cand = FAKE_TEXTS[(n + j) % len(FAKE_TEXTS)]
        if cand != cur:
            return cand
    return (cur or "") + " \n"


def coverage(item: dict) -> tuple[float, int]:
    """`(prefix_sum, covered)` -- the sum of the emitted tokens' own log-probabilities over the
    positions that HAVE a recorded distribution, and how many positions that is."""
    ids = item["token_ids"]
    total = 0.0
    covered = 0
    for t in item.get("topk") or []:
        pos = t["pos"]
        if pos < len(ids) and ids[pos] in t["ids"]:
            total += t["lps"][t["ids"].index(ids[pos])]
            covered += 1
    return total, covered


def is_partial(item: dict) -> bool:
    _, covered = coverage(item)
    return covered < item["n_generated"]


def _ids(cert: dict) -> list:
    return [it["item_id"] for it in cert["body"]["items"]]


def _index(cert: dict) -> dict:
    return {it["item_id"]: it for it in cert["body"]["items"]}


def _pick(cert: dict, f: int, rng: random.Random, pred=None) -> list:
    pool = [i for i in _ids(cert) if pred is None or pred(_index(cert)[i])]
    if f > len(pool):
        return list(pool)
    return rng.sample(pool, f)


# ---------------------------------------------------------------- 1/2  fabricate outright

def fabricate_careful(cert: dict, f: int, rng: random.Random) -> dict:
    """f items the issuer never ran, written so every internal predicate holds.

    The arithmetic (token ids, distributions, score) is borrowed from a genuine item elsewhere in
    the same battery, which is what makes it realizable for free; the answer text is invented.
    A forger with a calculator could synthesize the numbers instead -- borrowing is simply the
    cheapest way to be internally consistent, and the calibration is about detection, not effort.
    """
    out = copy.deepcopy(cert)
    idx = _index(out)
    src = _index(cert)                       # donors are read from the ORIGINAL, so f=64 works
    victims = _pick(out, f, rng)
    ids = _ids(out)
    for n, iid in enumerate(victims):
        donors = [i for i in ids if i != iid]
        d = copy.deepcopy(src[donors[(n * 7 + 3) % len(donors)]])
        for k in BODY_FIELDS:
            idx[iid][k] = copy.deepcopy(d[k])
        _set_text(idx[iid], _alt_text(idx[iid], n))
    return out


def fabricate_careless(cert: dict, f: int, rng: random.Random) -> dict:
    """The same fabrication with the digests left stale -- the control for P3/P4."""
    out = copy.deepcopy(cert)
    idx = _index(out)
    for n, iid in enumerate(_pick(out, f, rng)):
        idx[iid]["output_text"] = _alt_text(idx[iid], n)
        idx[iid]["seq_logprob"] = -0.05
    return out


# ---------------------------------------------------------------- 3  copy one output onto another

def copy_output(cert: dict, f: int, rng: random.Random) -> dict:
    """f items whose whole result is a verbatim copy of another item's genuine result.

    Every copied body is real, so no item-local predicate can object.  Whether a challenger sees
    anything depends on whether the source and destination honestly produced the same answer --
    ten groups of the 64 published items share an output digest, so some copies are invisible to
    the digest challenge too.  Those are counted, not hidden.
    """
    out = copy.deepcopy(cert)
    idx = _index(out)
    orig = _index(cert)              # sources are the ORIGINAL bodies, so copies never chain
    ids = _ids(out)
    for iid in _pick(out, f, rng):
        src = rng.choice([j for j in ids if j != iid])
        for k in BODY_FIELDS:
            idx[iid][k] = copy.deepcopy(orig[src][k])
    return out


# ---------------------------------------------------------------- 4/5/6  the score alone

def seqlp_tied(cert: dict, f: int, rng: random.Random, shift: float = -0.25) -> dict:
    """Alter `seq_logprob` on f items that HAVE a distribution for every emitted token."""
    out = copy.deepcopy(cert)
    idx = _index(out)
    for iid in _pick(out, f, rng, pred=lambda it: not is_partial(it)):
        idx[iid]["seq_logprob"] += shift
    return out


def seqlp_partial_under(cert: dict, f: int, rng: random.Random, shift: float = -1.0) -> dict:
    """H2, the understatement.  Push the score DOWN on f partial items.

    P8b bounds the score above by the recorded prefix sum and says nothing below it, so any
    deflation is permitted however large.  The emitted tokens are untouched, so a digest challenge
    has nothing to compare.
    """
    out = copy.deepcopy(cert)
    idx = _index(out)
    for iid in _pick(out, f, rng, pred=is_partial):
        idx[iid]["seq_logprob"] += shift
    return out


def seqlp_partial_over(cert: dict, f: int, rng: random.Random) -> dict:
    """The same region, the forbidden direction: set the score above the recorded prefix sum."""
    out = copy.deepcopy(cert)
    idx = _index(out)
    for iid in _pick(out, f, rng, pred=is_partial):
        prefix, _ = coverage(idx[iid])
        idx[iid]["seq_logprob"] = prefix + 0.5
    return out


def seqlp_partial_to_prefix(cert: dict, f: int, rng: random.Random) -> dict:
    """H2 pushed to its limit: the score set to the largest value P8b allows, the prefix sum.

    On the published certificate that is an improvement of 0.03 to 3.38 nats per item over the
    honest value, obtained by claiming the unrecorded tail was free.
    """
    out = copy.deepcopy(cert)
    idx = _index(out)
    for iid in _pick(out, f, rng, pred=is_partial):
        prefix, _ = coverage(idx[iid])
        idx[iid]["seq_logprob"] = prefix
    return out


# ---------------------------------------------------------------- 7/8/9  the distributions

def topk_reshape_naive(cert: dict, f: int, rng: random.Random) -> dict:
    """Rewrite the recorded distributions on f items to RESHAPE and leave the score alone."""
    out = copy.deepcopy(cert)
    idx = _index(out)
    for iid in _pick(out, f, rng):
        for t in idx[iid].get("topk") or []:
            t["lps"] = RESHAPE[:len(t["lps"])]
    return out


def topk_reshape_coordinated(cert: dict, f: int, rng: random.Random) -> dict:
    """Rewrite the distributions AND move the score to match, keeping the emitted tokens.

    For a fully-tied item the new score is the new sum, so P8 holds exactly.  For a partial item
    the unrecorded tail's honest contribution is carried across unchanged, so P8b holds too.  The
    certificate now claims the model was far more confident than it was, and no byte a challenger
    would reproduce with an output digest has changed.
    """
    out = copy.deepcopy(cert)
    idx = _index(out)
    for iid in _pick(out, f, rng):
        it = idx[iid]
        old_prefix, _ = coverage(it)
        tail = it["seq_logprob"] - old_prefix          # <= 0, the unrecorded remainder
        for t in it.get("topk") or []:
            t["lps"] = RESHAPE[:len(t["lps"])]
        new_prefix, _ = coverage(it)
        it["seq_logprob"] = new_prefix + tail
    return out


def topk_partial_free(cert: dict, f: int, rng: random.Random) -> dict:
    """Rewrite the distributions on partial items only and leave the score alone.

    Where the rewrite raises the prefix sum the one-sided bound P8b gets looser, not tighter, and
    nothing objects.  Where it lowers the prefix below the untouched score, P8b fires.  Which of
    the two happens is a property of the item, and the split is measured rather than assumed.
    """
    out = copy.deepcopy(cert)
    idx = _index(out)
    for iid in _pick(out, f, rng, pred=is_partial):
        for t in idx[iid].get("topk") or []:
            t["lps"] = RESHAPE[:len(t["lps"])]
    return out


# ---------------------------------------------------------------- 10/11  truncation

def truncate_careful(cert: dict, f: int, rng: random.Random) -> dict:
    """Cut a long generation short and fix up every dependent field.

    Truncating to a length within the recorded distributions turns a partial item into a fully
    tied one, so the score is set to the exact prefix sum of what remains and P8 holds.  Both
    digests are recomputed.  The text is cut proportionally -- nothing checks that it is the
    detokenization of anything (H1).
    """
    out = copy.deepcopy(cert)
    idx = _index(out)
    for iid in _pick(out, f, rng, pred=lambda it: it["n_generated"] >= 6):
        it = idx[iid]
        keep = max(3, min(len(it.get("topk") or []), it["n_generated"] - 2))
        frac = keep / it["n_generated"]
        it["token_ids"] = it["token_ids"][:keep]
        it["n_generated"] = keep
        it["topk"] = [t for t in (it.get("topk") or []) if t["pos"] < keep]
        it["token_ids_sha256"] = sha(
            json.dumps(it["token_ids"], separators=(",", ":")).encode("utf-8"))
        _set_text(it, it["output_text"][:max(1, int(len(it["output_text"]) * frac))])
        prefix, _ = coverage(it)
        it["seq_logprob"] = prefix
    return out


def truncate_careless(cert: dict, f: int, rng: random.Random) -> dict:
    """The same cut with `n_generated` left at its old value -- the control for P1."""
    out = copy.deepcopy(cert)
    idx = _index(out)
    for iid in _pick(out, f, rng, pred=lambda it: it["n_generated"] >= 6):
        it = idx[iid]
        it["token_ids"] = it["token_ids"][:it["n_generated"] - 2]
    return out


# ---------------------------------------------------------------- 12/13  swaps and relabelling

def swap_pairs(cert: dict, f: int, rng: random.Random) -> dict:
    """f disjoint transpositions: two items exchange results, both bodies genuine.

    The item id stays put, and the battery cert binds item_id to a prompt digest -- so after the
    swap the certificate asserts that prompt A produced prompt B's answer.  Nothing item-local can
    see it: both records are real records.
    """
    out = copy.deepcopy(cert)
    idx = _index(out)
    ids = _ids(out)
    chosen = rng.sample(ids, min(2 * f, len(ids) - len(ids) % 2))
    for a, b in zip(chosen[0::2], chosen[1::2]):
        for k in BODY_FIELDS:
            idx[a][k], idx[b][k] = copy.deepcopy(idx[b][k]), copy.deepcopy(idx[a][k])
    return out


def text_only_rewrite(cert: dict, f: int, rng: random.Random) -> dict:
    """H1.  Replace the rendered answer on f items; leave every number alone.

    Both digests stay internally consistent because `output_sha256` is recomputed.  The `exact`
    channel compares token ids, so the verdict arithmetic is untouched -- but a reader of the
    certificate now sees a different answer, and a challenger who compares the output digest sees
    it too.  This is the class the sampled challenge exists for.
    """
    out = copy.deepcopy(cert)
    idx = _index(out)
    for n, iid in enumerate(_pick(out, f, rng)):
        _set_text(idx[iid], _alt_text(idx[iid], n))
    return out


def text_swap(cert: dict, f: int, rng: random.Random) -> dict:
    """H1 again, without inventing anything: f pairs of items exchange their answer text only."""
    out = copy.deepcopy(cert)
    idx = _index(out)
    ids = _ids(out)
    chosen = rng.sample(ids, min(2 * f, len(ids) - len(ids) % 2))
    for a, b in zip(chosen[0::2], chosen[1::2]):
        ta, tb = idx[a]["output_text"], idx[b]["output_text"]
        _set_text(idx[a], tb)
        _set_text(idx[b], ta)
    return out


# ---------------------------------------------------------------- the registry

# (name, function, sweep values for f, one-line description for the report)
CLASSES = [
    ("fabricate_careful", fabricate_careful, [1, 2, 3, 4, 8, 16, 32, 64],
     "f items never run, written internally consistent"),
    ("fabricate_careless", fabricate_careless, [1, 2, 4, 8],
     "same, with the digests and score left stale"),
    ("copy_output", copy_output, [1, 2, 4, 8],
     "f items carrying a verbatim copy of another item's genuine result"),
    ("swap_pairs", swap_pairs, [1, 2, 4, 8],
     "f disjoint transpositions, both bodies genuine"),
    ("text_only_rewrite", text_only_rewrite, [1, 2, 4, 8, 16],
     "the rendered answer replaced, every number honest"),
    ("text_swap", text_swap, [1, 2, 4, 8],
     "f pairs exchange answer text only, token ids untouched"),
    ("seqlp_tied", seqlp_tied, [1, 2, 4, 8],
     "score altered where every emitted token has a distribution"),
    ("seqlp_partial_under", seqlp_partial_under, [1, 2, 4, 8, 16, 27],
     "score understated in the unrecorded-tail region"),
    ("seqlp_partial_to_prefix", seqlp_partial_to_prefix, [1, 2, 4, 8, 16, 27],
     "score set to the largest value P8b permits"),
    ("seqlp_partial_over", seqlp_partial_over, [1, 2, 4, 8],
     "score pushed above the recorded prefix sum"),
    ("topk_reshape_naive", topk_reshape_naive, [1, 2, 4, 8],
     "recorded distributions rewritten, score left behind"),
    ("topk_reshape_coordinated", topk_reshape_coordinated, [1, 2, 4, 8, 16, 64],
     "recorded distributions rewritten and the score moved to match"),
    ("topk_partial_free", topk_partial_free, [1, 2, 4, 8, 27],
     "recorded distributions rewritten on partial items only"),
    ("truncate_careful", truncate_careful, [1, 2, 4, 8],
     "a long generation cut short, every field fixed up"),
    ("truncate_careless", truncate_careless, [1, 2, 4, 8],
     "the same cut with n_generated left stale"),
]
