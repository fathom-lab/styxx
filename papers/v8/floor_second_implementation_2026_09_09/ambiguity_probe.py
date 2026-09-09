#!/usr/bin/env python3
"""ambiguity_probe.py -- does the Appendix B topk ambiguity change the VERDICT?

floor2.py reproduces the published floor exactly under one reading of Appendix B's
topk row.  A second honest reading of the same sentence gives a different floor.
This script asks the only question that matters about that: does the published
bf16-vs-fp16 verdict survive the other reading?

It also probes whether the second ambiguity in the same row (positions present on
one side only) is reachable at all on these bytes.

Reads: log entry 6 (bf16 canonical, carries the floor) and
       fp_fp16/fingerprint-canonical-d92ebe2089f9.json
Imports nothing from styxx.
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from floor2 import (build, d_exact, d_seqlp, d_topk, sorted_items, ROUND_DP)  # noqa: E402


def load(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def main():
    root = os.path.join(HERE, "..", "first_verdict_2026_09_09")
    res = build(os.path.join(root, "log"))

    bf16 = load(os.path.join(root, "fp_bf16", "fingerprint-canonical-73a09ffa3f1e.json"))
    fp16 = load(os.path.join(root, "fp_fp16", "fingerprint-canonical-d92ebe2089f9.json"))
    A, B = sorted_items(bf16), sorted_items(fp16)

    print("== the bf16 canonical in fp_bf16/ against the log's entry 6 ==")
    print("   same cert id : %s" % (bf16["id"] == res["canonical_cert_id"]))
    print("   bf16 id      : %s" % bf16["id"])
    print("   fp16 id      : %s" % fp16["id"])
    print("   fp16 body keys: %s" % sorted(fp16["body"].keys()))
    print()

    # -- RESULT_first_verdict published these three diff distances.
    published = {"exact": 0.062500000, "seqlp": 0.058564664, "topk": 1.407087824}

    print("== the published diff, recomputed here ==")
    for name, fn in (("exact", lambda: d_exact(A, B)),
                     ("seqlp", lambda: d_seqlp(A, B)),
                     ("topk",  lambda: d_topk(A, B, "flat"))):
        d = round(fn(), ROUND_DP)
        print("   %-6s mine=%-14r published=%-14r %s"
              % (name, d, published[name],
                 "AGREE" if float(d) == float(published[name]) else "DISAGREE"))
    print()

    # -- AMBIGUITY-1: "mean over items and over positions"
    print("== AMBIGUITY-1: topk, flat mean over (item,position) vs nested mean ==")
    rows = []
    for mode, label in (("flat", "flat (item,pos)"), ("nested", "mean of per-item means")):
        floor = res["computed"]["topk" if mode == "flat" else "topk/nested"]["floor"]
        d = round(d_topk(A, B, mode), ROUND_DP)
        verdict = "same" if d <= floor else "exceeds_floor"
        rows.append((label, floor, d, d / floor, verdict))
        print("   %-24s floor=%-14r diff=%-14r ratio=%.9f -> %s"
              % (label, floor, d, d / floor, verdict))
    print("   verdict identical under both readings: %s" % (rows[0][4] == rows[1][4]))
    print()

    # -- AMBIGUITY-2: is the one-sided-position clause ever reached on these bytes?
    print("== AMBIGUITY-2: positions present on one side only ==")
    def pos_mismatch(x, y):
        a = {it["item_id"]: {p["pos"] for p in it["topk"]} for it in x}
        b = {it["item_id"]: {p["pos"] for p in it["topk"]} for it in y}
        return sum(len(a[i] ^ b[i]) for i in set(a) & set(b))
    order = sorted(res["nuisance"])
    entries_by_run = res["run_index_to_entry_index"]
    log_items = {}
    for r in order:
        p = os.path.join(root, "log", "entries", "000000", "%08d.json" % entries_by_run[r])
        log_items[r] = sorted_items(load(p))
    total = 0
    for i in range(len(order)):
        for j in range(i + 1, len(order)):
            total += pos_mismatch(log_items[order[i]], log_items[order[j]])
    print("   one-sided positions across the 10 floor pairs : %d" % total)
    print("   one-sided positions on the bf16/fp16 diff     : %d" % pos_mismatch(A, B))
    print("   -> the clause is unreached on these bytes; the ambiguity is untested here")
    print()

    # -- AMBIGUITY-3: does the exact-channel role filter reach seqlp/topk?
    print("== AMBIGUITY-3: role filter ==")
    import collections
    roles = collections.Counter(it["role"] for it in A)
    print("   roles present in this battery: %s" % dict(roles))
    print("   -> only 'item'; the filter is unreached on these bytes")
    print()

    # -- what the fp16 side is missing
    print("== is the diff even well-defined under section 3.2? ==")
    print("   bf16 topk_forced_on : %r" % bf16["body"].get("topk_forced_on"))
    print("   fp16 topk_forced_on : %r" % fp16["body"].get("topk_forced_on"))
    print("   bf16 item has prefix_token_ids : %s" % ("prefix_token_ids" in A[0]))
    print("   fp16 item has prefix_token_ids : %s" % ("prefix_token_ids" in B[0]))
    ident = ("weights_sha256", "tokenizer_sha256", "config_sha256", "generation_config_sha256")
    print("   identity fields differing: %s"
          % [k for k in ident if bf16["subject"].get(k) != fp16["subject"].get(k)])
    print("   precision bf16=%r fp16=%r"
          % (bf16["subject"].get("precision"), fp16["subject"].get("precision")))
    print()

    # -- section 3.2's comparability rule, applied to the bytes as they stand.
    print("== section 3.2 comparability, applied ==")
    print("   'two certs' topk are comparable iff their topk_forced_on values are equal")
    print("    and are not \"self\", or both are \"self\" and the two sides' greedy outputs")
    print("    are token-identical.'")
    print()

    def token_identical(x, y):
        a = {it["item_id"]: it["token_ids"] for it in x}
        b = {it["item_id"]: it["token_ids"] for it in y}
        return all(a[i] == b[i] for i in set(a) & set(b))

    print("   bf16 vs fp16 greedy outputs token-identical : %s" % token_identical(A, B))
    print("   -> under the 'self' branch this comparison is topk-INCONCLUSIVE and")
    print("      section 6 says no topk number is printed. RESULT printed 1.407087824.")
    print()
    print("   the same rule applied to the ten floor pairs:")
    surviving = []
    for k, (i, j) in enumerate(res["pair_order"]):
        ti = token_identical(log_items[i], log_items[j])
        d = res["computed"]["topk"]["distances"][k]
        if ti:
            surviving.append(d)
        print("     pair %d (run %d, run %d) token-identical=%-5s topk d=%r%s"
              % (k, i, j, ti, d, "" if ti else "   <- inconclusive under the rule"))
    print("   topk floor over only the comparable pairs = %r  (published: %r)"
          % (max(surviving) if surviving else None, res["stored"]["per_channel"]["topk"]["floor"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
