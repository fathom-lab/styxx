"""The predicates a FIRST claim must satisfy, run against this lab's own published log.

THE_BOUNDARY's fourth correction says the residue is "a first claim fabricated carefully enough to
be internally realizable", and lists the constraints that make that qualifier do work: outputs must
tokenize, must respect the declared token limit, and a topk channel must be a valid log-softmax
agreeing with the token the run says it emitted.

That list was asserted. This script tests it. Every check below reads ONE certificate and needs no
prior entry, no second party and no model -- which is the point, because these are the only
predicates available to a claim about a subject nobody has measured before.

Eight predicates, in rough order of how much work it costs a fabricator to satisfy them:

  1  n_generated equals the number of token ids recorded
  2  n_generated is within the recipe's declared max_new_tokens
  3  output_sha256 is the digest of output_text
  4  token_ids_sha256 is the digest of the token ids
  5  every topk position lists its log-probabilities in descending order
  6  every topk position's probabilities sum to at most 1, being the top k of a distribution
  7  under greedy decoding the emitted token is the argmax, so token_ids[pos] == topk[pos].ids[0]
  8  seq_logprob is the sum of the emitted tokens' own log-probabilities

  8b seq_logprob does not EXCEED the recorded prefix's sum, for items whose generation runs past
     the last recorded distribution

7 and 8 are the ones with teeth. They tie three independently written fields -- the emitted tokens,
the per-position distributions, and the sequence score -- to each other by arithmetic, so a
fabricator cannot write any one of them freely.

8b exists because 8 does not always apply, and finding that out was the point. The certificate
records at most 8 topk positions while items generate up to 16 tokens, so for the longer items the
sequence score has no recorded distribution to be the sum of. Only an inequality survives there:
log-probabilities are at most 0, so the score cannot exceed the prefix sum. That refuses an inflated
score and permits any deflated one.

MEASURED, on the published log: all predicates hold on all 320 items across 5 certificates, and
185 of 320 items are fully tied down by 8, while 135 are only one-sided under 8b, with 965 emitted
tokens carrying no recorded distribution at all. The first number is what a first claim cannot
escape. The second is what it can.

A failure here is a finding about the published artifact and is reported as one, not repaired. The
first version of this script reported 135 failures; every one was this script's own fault for
assuming a distribution existed for every emitted token, and the diagnosis came before the report.
"""
import glob
import json
import math
import pathlib
from collections import Counter

LOG = pathlib.Path(
    r"C:\Users\heyzo\clawd\wt\v8\papers\v8\first_verdict_2026_09_09\log\entries"
)
import hashlib

TOL = 1e-6


def sha(b):
    return hashlib.sha256(b).hexdigest()


def check_item(item, max_new, unconstrained=None):
    """Return a list of failure strings; empty means every predicate held.

    `unconstrained` collects, per item, how many emitted tokens have no recorded distribution.
    Those positions are not a failure and not a pass: they are the part of the certificate that
    nothing ties down, and counting them is the honest output.
    """
    bad = []
    if unconstrained is None:
        unconstrained = []
    ids = item.get("token_ids")
    n = item.get("n_generated")
    topk = item.get("topk") or []

    if ids is not None and n is not None and n != len(ids):
        bad.append(f"P1 n_generated={n} but {len(ids)} token ids")
    if n is not None and max_new is not None and n > max_new:
        bad.append(f"P2 n_generated={n} exceeds max_new_tokens={max_new}")

    if "output_text" in item and "output_sha256" in item:
        got = sha(item["output_text"].encode("utf-8"))
        if got != item["output_sha256"]:
            bad.append("P3 output_sha256 is not the digest of output_text")
    if ids is not None and "token_ids_sha256" in item:
        # the digest's preimage convention is not stated in the item, so try the obvious ones
        cands = {
            sha(json.dumps(ids, separators=(",", ":")).encode("utf-8")),
            sha(",".join(map(str, ids)).encode("utf-8")),
            sha(b"".join(i.to_bytes(4, "little") for i in ids)),
            sha(" ".join(map(str, ids)).encode("utf-8")),
        }
        if item["token_ids_sha256"] not in cands:
            bad.append("P4 token_ids_sha256 matches no tried preimage convention")

    for t in topk:
        pos, lps, tids = t.get("pos"), t.get("lps") or [], t.get("ids") or []
        if any(lps[i] < lps[i + 1] - 1e-12 for i in range(len(lps) - 1)):
            bad.append(f"P5 pos {pos}: log-probabilities not descending")
        mass = sum(math.exp(v) for v in lps)
        if mass > 1 + TOL:
            bad.append(f"P6 pos {pos}: top-{len(lps)} probability mass {mass:.9f} exceeds 1")
        if ids is not None and pos is not None and pos < len(ids) and tids:
            if ids[pos] != tids[0]:
                bad.append(f"P7 pos {pos}: emitted {ids[pos]} but argmax is {tids[0]}")

    if "seq_logprob" in item and topk and ids:
        chosen = []
        for t in topk:
            pos, lps, tids = t.get("pos"), t.get("lps") or [], t.get("ids") or []
            if pos is None or pos >= len(ids):
                continue
            if ids[pos] in tids:
                chosen.append(lps[tids.index(ids[pos])])
        if len(chosen) == len(ids):
            # P8, the full tie: every emitted position has a recorded distribution, so the
            # sequence score is determined and cannot be written freely.
            total = sum(chosen)
            if abs(total - item["seq_logprob"]) > 1e-6:
                bad.append(f"P8 seq_logprob {item['seq_logprob']:.9f} != sum of emitted "
                           f"log-probabilities {total:.9f}")
        elif chosen:
            # P8b, the one-sided bound. The certificate records at most 8 topk positions, so an
            # item generating more than that has no recorded distribution for its tail and its
            # sequence score is NOT determined. One inequality survives: every log-probability is
            # at most 0, so the total cannot exceed the recorded prefix's sum. That refuses an
            # inflated score and says nothing about a deflated one.
            prefix = sum(chosen)
            if item["seq_logprob"] > prefix + 1e-6:
                bad.append(f"P8b seq_logprob {item['seq_logprob']:.9f} exceeds the recorded "
                           f"prefix sum {prefix:.9f}, which is impossible for logprobs <= 0")
            unconstrained.append(len(ids) - len(chosen))
        else:
            bad.append("P8 no emitted token could be resolved in any recorded distribution")
    return bad


certs = []
for f in sorted(glob.glob(str(LOG / "*" / "*[0-9].json"))):
    c = json.loads(pathlib.Path(f).read_text(encoding="utf-8"))
    if c.get("type") == "fingerprint":
        c["_entry"] = pathlib.Path(f).stem
        certs.append(c)

print(f"{len(certs)} fingerprint certificates in the published log\n")
grand = Counter()
total_items = 0
loose = 0
partial = 0
for c in certs:
    max_new = c.get("recipe", {}).get("decoding", {}).get("max_new_tokens")
    items = c["body"].get("items") or []
    fails = Counter()
    free = []
    for it in items:
        for b in check_item(it, max_new, free):
            fails[b.split(":")[0].split(" ")[0]] += 1
            grand[b.split(":")[0].split(" ")[0]] += 1
    total_items += len(items)
    loose += sum(free)
    partial += len(free)
    status = "all predicates hold" if not fails else dict(fails)
    print(f"  entry {c['_entry']}  batch={c['recipe']['decoding']['batch_size']:>2}  "
          f"{len(items)} items  ->  {status};  {len(free)} item(s) only bounded, "
          f"{sum(free)} emitted token(s) with no recorded distribution")

print(f"\n{total_items} items checked across {len(certs)} certificates")
if grand:
    print("failures by predicate:", dict(grand))
else:
    print("every predicate held on every item")
print(f"fully tied down by P8      : {total_items - partial} of {total_items} items")
print(f"only one-sided under P8b   : {partial} items, {loose} emitted tokens carrying no "
      f"recorded distribution at all")

# --- does the battery have teeth? Perturb one real item and see which predicates fire.
print("\n--- controls: one real item, minimally falsified")
victim = next(json.loads(json.dumps(i)) for i in certs[-1]["body"]["items"]
          if len(i.get("topk") or []) == i["n_generated"])
max_new = certs[-1]["recipe"]["decoding"]["max_new_tokens"]
assert not check_item(victim, max_new), "the control victim must start clean"


def remint(v):
    """Recompute every derived field the way an honest mint would.

    This is the difference between the two columns below, and it is the whole calibration. A
    forgery that leaves derived fields stale is a careless forgery. An honest mint recomputes
    digests from the bytes it is writing and sets the sequence score to the sum of the emitted
    tokens' log-probabilities, so a forger who runs the same code gets all of that for free.
    """
    v["output_sha256"] = sha(v["output_text"].encode("utf-8"))
    v["token_ids_sha256"] = sha(json.dumps(v["token_ids"], separators=(",", ":")).encode("utf-8"))
    v["n_generated"] = len(v["token_ids"])
    chosen = []
    for t in v.get("topk") or []:
        pos, lps, tids = t.get("pos"), t.get("lps") or [], t.get("ids") or []
        if pos is not None and pos < len(v["token_ids"]) and v["token_ids"][pos] in tids:
            chosen.append(lps[tids.index(v["token_ids"][pos])])
    if len(chosen) == len(v["token_ids"]):
        v["seq_logprob"] = sum(chosen)
    return v


def show(name, mutate):
    careless = mutate(json.loads(json.dumps(victim)))
    careful = remint(mutate(json.loads(json.dumps(victim))))
    a, b = check_item(careless, max_new), check_item(careful, max_new)
    print(f"  careless {'CAUGHT ' if a else 'MISSED '} | careful "
          f"{'CAUGHT ' if b else 'MISSED '} | {name}")
    for line in b[:1]:
        print(f"      still caught: {line}")


def swap_emitted(v):
    """Claim a different token, and reorder its distribution so the claim is self-consistent."""
    t = v["topk"][0]
    i = 1
    t["ids"][0], t["ids"][i] = t["ids"][i], t["ids"][0]
    t["lps"][0], t["lps"][i] = t["lps"][i], t["lps"][0]
    t["lps"].sort(reverse=True)
    v["token_ids"][0] = t["ids"][0]
    return v


def inflate_score(v):
    """Make the run look more confident, keeping the distribution normalised.

    The first attempt at this control raised the top probability without renormalising, so the
    top-5 mass went over 1 and P6 caught it. That is not a careful forger, it is a careless one
    wearing the word careful, and reporting it as detection power would have repeated the exact
    error this section exists to correct. The mass is now held at what the honest item had.
    """
    import math as _m
    t = v["topk"][0]
    mass = sum(_m.exp(x) for x in t["lps"])
    p = [_m.exp(x) for x in t["lps"]]
    p[0] = p[0] * 1.5
    scale = mass / sum(p)
    t["lps"] = sorted((_m.log(x * scale) for x in p), reverse=True)
    return v


def flatten(v):
    """A confident-looking distribution that is still a valid normalised log-softmax."""
    import math as _m
    k = len(v["topk"][0]["lps"])
    p = [0.96] + [0.04 / (k - 1)] * (k - 1)
    v["topk"][0]["lps"] = [_m.log(x) for x in p]
    return v


def rewrite_text(v):
    """Change the answer. An honest mint recomputes the digest over whatever text it is given."""
    v["output_text"] = "Venus \n"
    return v


print("  careless = the mutation alone. careful = the same mutation with every derived field")
print("  recomputed as an honest mint would recompute it. The second column is the real number.\n")
show("a different token claimed as emitted", swap_emitted)
show("a flattering sequence score", inflate_score)
show("a confident-looking distribution", flatten)
show("the answer text changed", rewrite_text)

print("""
  THE CAREFUL COLUMN IS THE DETECTION POWER AND THE CARELESS COLUMN IS NOT. The first version of
  this script reported only the careless column, and reported four of four caught. An adversary
  re-ran the same four in careful form and the battery missed them. By this lab's own standing rule
  -- an agreement number without its detection power is not a number -- the earlier claim was the
  thing the rule forbids.

  A NINTH PREDICATE would catch the fourth one and is not implemented here: `output_text` must be
  the detokenization of `token_ids` under the tokenizer the subject names by its A.2 join-hash.
  That requires the tokenizer, so it is not a pure-bytes check and does not belong in this file --
  but it is available to a first claim, it holds on all 320 published items when run against the
  local snapshot, and it is owed.""")

print("""
READING
  These are the predicates a first claim cannot escape, because they compare a certificate only to
  itself. They do not establish that the computation happened. They establish that whoever wrote it
  had to write it consistently, and 7 and 8 mean the emitted tokens, the per-position distributions
  and the sequence scores cannot be chosen independently of one another.

  That is the whole content of "fabricated carefully enough to be internally realizable". The
  qualifier is not decoration: it is this list, and the cost of satisfying it is the only thing
  standing between a first claim and free invention.""")
