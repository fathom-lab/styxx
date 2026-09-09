#!/usr/bin/env python3
"""test_floor2.py -- unit tests for this second implementation.

Two of Appendix B's clauses are unreached by the published log (no item is ever
absent from one side's top-k union in a way that matters, and no position is
present on one side only), so the agreement with the published floor does not
exercise them.  These tests do, on hand-built inputs, so that the reading this
file implements is visible rather than merely asserted.

Run:  python test_floor2.py
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from floor2 import d_exact, d_seqlp, d_topk, pos_l1, ABSENT_LP, MISSING_POS_COST  # noqa: E402

FAILED = []


def check(name, got, want):
    ok = got == want
    if not ok:
        FAILED.append(name)
    print("  %-52s %s   got=%r want=%r" % (name, "ok" if ok else "FAIL", got, want))


def item(iid, ids, lp, topk, role="item"):
    return {"item_id": iid, "token_ids": ids, "seq_logprob": lp,
            "n_generated": len(ids), "role": role, "topk": topk}


def pos(p, ids, lps):
    return {"pos": p, "ids": ids, "lps": lps}


def main():
    print("exact")
    a = [item("aa", [1, 2], -1.0, []), item("bb", [3], -2.0, []),
         item("cc", [4], -3.0, []), item("dd", [5], -4.0, [])]
    b = [item("aa", [1, 2], -1.0, []), item("bb", [9], -2.0, []),
         item("cc", [4], -3.0, []), item("dd", [5], -4.0, [])]
    check("1 of 4 items differs -> 0.25", d_exact(a, b), 0.25)
    check("identical -> 0.0", d_exact(a, a), 0.0)

    # role filter: an 'anchor' item that differs must not move the exact distance
    a2 = a + [item("ee", [7], -1.0, [], role="anchor")]
    b2 = b + [item("ee", [8], -1.0, [], role="anchor")]
    check("anchor role excluded from exact", d_exact(a2, b2), 0.25)

    print("seqlp")
    c = [item("aa", [1], -1.0, []), item("bb", [1], -2.0, [])]
    d = [item("aa", [1], -1.5, []), item("bb", [1], -2.25, [])]
    check("mean(|0.5|,|0.25|) -> 0.375", d_seqlp(c, d), 0.375)

    print("topk / union-of-vocab alignment")
    # one shared token, one token present only on each side
    pa = pos(0, [10, 11], [-0.1, -1.0])
    pb = pos(0, [10, 12], [-0.2, -1.0])
    # union {10,11,12}: |(-0.1)-(-0.2)| + |(-1.0)-(-20)| + |(-20)-(-1.0)|
    want = abs(-0.1 - -0.2) + abs(-1.0 - ABSENT_LP) + abs(ABSENT_LP - -1.0)
    check("absent token takes lp=-20", pos_l1(pa, pb), want)

    x = [item("aa", [1], -1.0, [pa])]
    y = [item("aa", [1], -1.0, [pb])]
    check("single item, single position", d_topk(x, y, "flat"), want)

    print("topk / a position present on one side only")
    x2 = [item("aa", [1, 2], -1.0, [pa, pos(1, [1], [-0.5])])]
    y2 = [item("aa", [1], -1.0, [pb])]
    # flat: (want + 100) / 2 ; both-sides-only: want / 1
    check("one-sided position costs 5*20 and is counted",
          d_topk(x2, y2, "flat"), (want + MISSING_POS_COST) / 2)
    check("literal 'positions present on both sides' drops it",
          d_topk(x2, y2, "flat_bothonly"), want)

    print("topk / flat vs nested differ when items have unequal position counts")
    # item aa: 2 positions each worth 1.0 ; item bb: 1 position worth 4.0
    p1 = pos(0, [1], [0.0])
    p1b = pos(0, [1], [-1.0])
    p2 = pos(1, [1], [0.0])
    p2b = pos(1, [1], [-1.0])
    q = pos(0, [1], [0.0])
    qb = pos(0, [1], [-4.0])
    m = [item("aa", [1, 2], 0.0, [p1, p2]), item("bb", [1], 0.0, [q])]
    n = [item("aa", [1, 2], 0.0, [p1b, p2b]), item("bb", [1], 0.0, [qb])]
    check("flat  = (1+1+4)/3", d_topk(m, n, "flat"), 6.0 / 3)
    check("nested= (1 + 4)/2", d_topk(m, n, "nested"), 5.0 / 2)

    print()
    if FAILED:
        print("FAILED: %s" % ", ".join(FAILED))
        return 1
    print("all unit tests pass")
    return 0


if __name__ == "__main__":
    sys.exit(main())
