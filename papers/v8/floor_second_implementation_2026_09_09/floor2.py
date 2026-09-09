#!/usr/bin/env python3
"""floor2.py -- a second, standalone implementation of the v8 noise floor.

Written from the specification only:
  papers/v8/SPEC_v8_v0.2_draft.md  -- section 3.2 (channels), 3.3, 5.1 (procedure),
                                      5.2 (rounding), 5.7 (overall size),
                                      Appendix A.3 (item order), Appendix B (distances)
  papers/v8/first_verdict_2026_09_09/RESULT_first_verdict_2026_09_09.md

It imports nothing from styxx. It reads the published log entries as JSON and
recomputes, per channel, the ten pairwise distances between the five runs, the
floor as their maximum, alpha_single, and the section 5.7 standardized maxima.

Usage:
    python floor2.py <log_dir>   [default: ../first_verdict_2026_09_09/log]

Spec sentences this implements, quoted so a reader can check the reading:

  App. B, exact : "1 - (matching items / items), over items with role in
                   {item, canary}; anchors are counted separately as anchor_flips"
  Sec 3.2 exact : "greedy output token ids per item"
  App. B, seqlp : "mean over items of |delta seq_logprob|"
  App. B, topk  : "mean over items and over positions present on both sides of the
                   L1 distance between top-k vectors after union-of-vocab alignment,
                   a token absent from one side assigned lp = -20; a position present
                   on one side only contributes the maximum per-position distance
                   (5 . 20 nats) and is counted"
  App. B, head  : "every mean is over items in A.3 order with exact summation"
  App. A.3      : "Items are ordered by item_id ascending (byte order of the UTF-8 string)"
  Sec 5.1 (3)   : "compute all pairwise distances between runs -> the empirical null D_c.
                   floor_c = max(D_c) ... alpha_single = 1/(pairs+1)"
  Sec 5.2       : "Distances are compared after rounding both sides to 9 decimal places"

Where the spec does not determine a reading, this file computes EVERY reading and
reports them all rather than choosing one silently. Those points are marked AMBIGUITY.
"""

import json
import math
import os
import sys
from itertools import combinations

ABSENT_LP = -20.0          # App. B: "a token absent from one side assigned lp = -20"
TOPK_K = 5                 # Sec 3.2: "top-5 log-probs"
MISSING_POS_COST = 5 * 20.0  # App. B: "the maximum per-position distance (5 . 20 nats)"
ROUND_DP = 9               # Sec 5.2

EXACT_ROLES = {"item", "canary"}   # App. B, exact row


# ---------------------------------------------------------------- loading

def load_entries(log_dir):
    """Return the list of (index, meta, cert) for every entry file, index ascending."""
    ent_root = os.path.join(log_dir, "entries")
    out = []
    for shard in sorted(os.listdir(ent_root)):
        sdir = os.path.join(ent_root, shard)
        if not os.path.isdir(sdir):
            continue
        for name in sorted(os.listdir(sdir)):
            if not name.endswith(".json") or name.endswith(".meta.json"):
                continue
            with open(os.path.join(sdir, name), "r", encoding="utf-8") as fh:
                cert = json.load(fh)
            mpath = os.path.join(sdir, name[:-5] + ".meta.json")
            with open(mpath, "r", encoding="utf-8") as fh:
                meta = json.load(fh)
            out.append((meta["index"], meta, cert))
    out.sort(key=lambda t: t[0])
    return out


def sorted_items(cert):
    """App. A.3: items ordered by item_id ascending, byte order of the UTF-8 string.

    The published run certs store their items in RUN order, not item_id order, on
    the four non-canonical runs, so this sort is load-bearing rather than cosmetic.
    """
    return sorted(cert["body"]["items"], key=lambda it: it["item_id"].encode("utf-8"))


# ---------------------------------------------------------------- channels

def d_exact(a_items, b_items):
    """1 - (matching items / items), over items with role in {item, canary}."""
    a = {it["item_id"]: it for it in a_items if it["role"] in EXACT_ROLES}
    b = {it["item_id"]: it for it in b_items if it["role"] in EXACT_ROLES}
    ids = sorted(set(a) & set(b), key=lambda s: s.encode("utf-8"))
    if len(a) != len(b) or len(ids) != len(a):
        raise ValueError("item sets differ between the two runs")
    match = sum(1 for i in ids if a[i]["token_ids"] == b[i]["token_ids"])
    return 1.0 - (match / len(ids))


def d_seqlp(a_items, b_items, roles=None):
    """mean over items of |delta seq_logprob|, exact summation, item_id order."""
    a = {it["item_id"]: it for it in a_items if roles is None or it["role"] in roles}
    b = {it["item_id"]: it for it in b_items if roles is None or it["role"] in roles}
    ids = sorted(set(a) & set(b), key=lambda s: s.encode("utf-8"))
    diffs = [abs(a[i]["seq_logprob"] - b[i]["seq_logprob"]) for i in ids]
    return math.fsum(diffs) / len(ids)


def pos_l1(pa, pb):
    """L1 distance between two top-k vectors after union-of-vocab alignment."""
    la = dict(zip(pa["ids"], pa["lps"]))
    lb = dict(zip(pb["ids"], pb["lps"]))
    union = sorted(set(la) | set(lb))
    terms = [abs(la.get(v, ABSENT_LP) - lb.get(v, ABSENT_LP)) for v in union]
    return math.fsum(terms)


def d_topk(a_items, b_items, mode):
    """topk distance.

    AMBIGUITY-1 -- "mean over items and over positions".  Two honest readings:
      mode='flat'   : one mean over the flat set of (item, position) pairs
      mode='nested' : mean over items of the per-item mean over positions
    AMBIGUITY-2 -- a position present on one side only "contributes the maximum
      per-position distance (5 . 20 nats) and is counted".  Counted where:
      mode='flat_bothonly' : such positions are dropped entirely (the literal reading
                             of "over positions present on both sides", which the
                             following clause then contradicts)
    """
    a = {it["item_id"]: it for it in a_items}
    b = {it["item_id"]: it for it in b_items}
    ids = sorted(set(a) & set(b), key=lambda s: s.encode("utf-8"))

    per_item_sums, per_item_counts, flat_terms = [], [], []
    for i in ids:
        pa = {p["pos"]: p for p in a[i]["topk"]}
        pb = {p["pos"]: p for p in b[i]["topk"]}
        both = sorted(set(pa) & set(pb))
        only = sorted(set(pa) ^ set(pb))
        terms = [pos_l1(pa[p], pb[p]) for p in both]
        if mode != "flat_bothonly":
            terms += [MISSING_POS_COST] * len(only)
        per_item_sums.append(math.fsum(terms))
        per_item_counts.append(len(terms))
        flat_terms.extend(terms)

    if mode == "nested":
        means = [s / c for s, c in zip(per_item_sums, per_item_counts) if c]
        return math.fsum(means) / len(means)
    n = len(flat_terms)
    return math.fsum(flat_terms) / n if n else 0.0


# ---------------------------------------------------------------- floor

def build(log_dir):
    entries = load_entries(log_dir)
    fps = [(idx, c) for idx, m, c in entries if m["type"] == "fingerprint"]
    canon = [(idx, c) for idx, c in fps if "noise_floor" in c["body"]]
    if len(canon) != 1:
        raise ValueError("expected exactly one cert carrying noise_floor, got %d" % len(canon))
    canon_idx, canon_cert = canon[0]
    stored = canon_cert["body"]["noise_floor"]

    # Sec 5.1 step 4: "Every run is logged as its own fingerprint cert with run_index".
    runs = {}
    for idx, c in fps:
        runs[c["body"]["run_index"]] = (idx, c)
    order = sorted(runs)
    if order != list(range(len(order))):
        raise ValueError("run_index values are not 0..R-1: %r" % order)

    items = {r: sorted_items(runs[r][1]) for r in order}
    nuis = {r: runs[r][1]["body"]["nuisance"] for r in order}

    # Sec 5.1 step 3: "all pairwise distances between runs".
    # AMBIGUITY-3 -- the spec fixes neither the ORDER of the ten distances in the
    # array nor that they are unordered pairs.  This uses ascending (i, j), i < j,
    # over run_index, which is the only enumeration that reproduces the stored array.
    pairs = list(combinations(order, 2))

    channels = {
        "exact":  lambda i, j: d_exact(items[i], items[j]),
        "seqlp":  lambda i, j: d_seqlp(items[i], items[j]),
        "topk":   lambda i, j: d_topk(items[i], items[j], "flat"),
    }
    variants = {
        "seqlp/roles_item_canary": lambda i, j: d_seqlp(items[i], items[j], EXACT_ROLES),
        "topk/nested":             lambda i, j: d_topk(items[i], items[j], "nested"),
        "topk/both_positions_only": lambda i, j: d_topk(items[i], items[j], "flat_bothonly"),
    }

    computed = {}
    for name, fn in list(channels.items()) + list(variants.items()):
        vals = [round(fn(i, j), ROUND_DP) for (i, j) in pairs]
        computed[name] = {
            "distances": vals,
            "floor": max(vals),
            "runs": len(order),
            "pairs": len(pairs),
            "alpha_single": 1.0 / (len(pairs) + 1),
        }

    # Sec 5.7 / the stored standardization string: max over channels of d/floor
    # against the reference run (run_index 0).
    std = []
    for r in order:
        ratios, exceeds = [], False
        for name in ("exact", "seqlp", "topk"):
            f = computed[name]["floor"]
            d = round(channels[name](0, r), ROUND_DP) if r != 0 else 0.0
            if f == 0:
                if d > 0:
                    exceeds = True
            else:
                ratios.append(d / f)
        m = max(ratios) if ratios else 0.0
        std.append({"run": r, "standardized_max": round(m, ROUND_DP),
                    "exceeds": bool(exceeds or m > 1.0)})

    return {
        "standardized_max": std,
        "log_dir": os.path.abspath(log_dir),
        "canonical_entry_index": canon_idx,
        "canonical_cert_id": canon_cert["id"],
        "run_index_to_entry_index": {r: runs[r][0] for r in order},
        "nuisance": nuis,
        "pair_order": [list(p) for p in pairs],
        "computed": computed,
        "stored": stored,
    }


def compare(res):
    """Report agreement to the last digit, per channel, per pairwise distance."""
    lines = []
    allok = True
    for name in ("exact", "seqlp", "topk"):
        got = res["computed"][name]
        want = res["stored"]["per_channel"][name]
        lines.append("channel %s" % name)
        for k, (p, g, w) in enumerate(zip(res["pair_order"], got["distances"], want["distances"])):
            ok = (repr(float(g)) == repr(float(w))) or (float(g) == float(w))
            allok = allok and ok
            lines.append("  pair %d (run %d, run %d)  mine=%-14r  stored=%-14r  %s"
                         % (k, p[0], p[1], g, w, "AGREE" if ok else "DISAGREE"))
        okf = float(got["floor"]) == float(want["floor"])
        allok = allok and okf
        lines.append("  floor            mine=%-14r  stored=%-14r  %s"
                     % (got["floor"], want["floor"], "AGREE" if okf else "DISAGREE"))
        oka = abs(got["alpha_single"] - want["alpha_single"]) == 0
        lines.append("  alpha_single     mine=%-14r  stored=%-14r  %s"
                     % (got["alpha_single"], want["alpha_single"], "AGREE" if oka else "DISAGREE"))
        allok = allok and oka
        lines.append("")

    lines.append("standardized_max (section 5.7)")
    stored_std = {e["run"]: e for e in res["stored"]["standardized_max"]}
    for e in res["standardized_max"]:
        w = stored_std.get(e["run"])
        ok = w is not None and float(e["standardized_max"]) == float(w["standardized_max"]) \
            and e["exceeds"] == w["exceeds"]
        allok = allok and ok
        lines.append("  run %d  mine=%-14r exceeds=%-5s  stored=%-14r exceeds=%-5s  %s"
                     % (e["run"], e["standardized_max"], e["exceeds"],
                        w["standardized_max"] if w else None,
                        w["exceeds"] if w else None,
                        "AGREE" if ok else "DISAGREE"))
    lines.append("")
    okao = float(res["stored"]["alpha_overall"]) == float(
        sum(1 for e in res["standardized_max"] if e["exceeds"]) / len(res["standardized_max"]))
    lines.append("  alpha_overall    mine=%r  stored=%r  %s"
                 % (sum(1 for e in res["standardized_max"] if e["exceeds"]) / len(res["standardized_max"]),
                    res["stored"]["alpha_overall"], "AGREE" if okao else "DISAGREE"))
    allok = allok and okao
    lines.append("")
    return allok, lines


def main():
    log_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "first_verdict_2026_09_09", "log")
    res = build(log_dir)

    print("log            : %s" % res["log_dir"])
    print("canonical cert : entry %d  %s" % (res["canonical_entry_index"], res["canonical_cert_id"]))
    print("runs           :")
    for r, n in sorted(res["nuisance"].items()):
        print("  run %d  entry %d  batch_size=%-3s item_order=%s"
              % (r, res["run_index_to_entry_index"][r], n.get("batch_size"), n.get("item_order")))
    print()
    ok, lines = compare(res)
    print("\n".join(lines))

    print("readings the specification does not choose between")
    for name in ("seqlp/roles_item_canary", "topk/nested", "topk/both_positions_only"):
        c = res["computed"][name]
        print("  %-26s floor=%r" % (name, c["floor"]))
        print("  %-26s distances=%r" % ("", c["distances"]))
    print("OVERALL: %s" % ("all 30 pairwise distances + 3 floors agree to the last digit"
                           if ok else "DISAGREEMENT -- see above"))

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "floor2_result.json")
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(res, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print("wrote %s" % out)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
