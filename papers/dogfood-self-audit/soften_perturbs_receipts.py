"""Does conscience softening change what the receipt gate sees?

This is a COMPARABILITY question, not a curiosity. The 0.397 fabrication prior that
drives the sixth sense was measured through a pipeline that softens first and scores
second (measure_confabulation_rate.py: conscience_gate -> receipt_gate). Restoring the
calibrated gate takes softening from 68% of turns to 2%, so if softening ever altered a
receipt verdict, the prior was measured on a pipeline that no longer exists and has to be
re-measured before it is quoted again.

The log carries both texts per turn -- `draft` (pre-soften) and `shipped` (post-soften)
-- so the question is answerable on real traffic instead of argued from the fact that
auto_soften "only touches register".

Both texts are scored with IDENTICAL evidence (none), because the comparison being made
is whether the TEXT CHANGE moves the verdict, not what either text's absolute standing
is. Holding evidence fixed is what makes the difference attributable to softening.

    python soften_perturbs_receipts.py
"""
from __future__ import annotations

import io
import json
import os
import sys

LOG = r"C:\Users\heyzo\.styxx\glimmer-day-zero\darkflobi_glimmer_log.jsonl"
sys.path.insert(0, r"C:\Users\heyzo\clawd\scripts")
from execution_receipt_gate import review                                # noqa: E402


def main():
    pairs = []
    for line in io.open(LOG, encoding="utf-8", errors="replace"):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        if not isinstance(r, dict):
            continue
        d, s = r.get("draft"), r.get("shipped")
        if isinstance(d, str) and isinstance(s, str) and d and s:
            pairs.append((d, s, bool(r.get("softened")), str(r.get("user") or "")))

    changed_text = [p for p in pairs if p[0] != p[1]]
    print(f"  {len(pairs)} turns carry both texts; {len(changed_text)} were altered by "
          f"softening ({len(changed_text) / max(1, len(pairs)):.1%})")

    # TWO DIFFERENT QUESTIONS, and the first version of this script ran them together.
    # It compared the tuple (fires, n_claims) and called any change a "verdict flip",
    # then concluded the prior was not comparable. But the prior is computed from `fires`
    # ALONE -- measure_confabulation_rate.py line: `fired = bool(receipts.get("fires"))`.
    # Both observed changes were 3 claims -> 2 claims with `fires` True on both sides, so
    # the binary the prior depends on never moved and the conclusion was wrong.
    # Conflating a count with a decision is the same error as reading a median as a
    # distribution, which this program already made once today.
    binary_flips, count_changes, examined = [], [], 0
    for draft, shipped, _soft, q in changed_text:
        examined += 1
        a = review(draft, toolcall_log=None, question=q)
        b = review(shipped, toolcall_log=None, question=q)
        rec = {"question": q[:70],
               "pre": {"fires": bool(a["fires"]), "n_claims": a["n_claims"]},
               "post": {"fires": bool(b["fires"]), "n_claims": b["n_claims"]},
               "draft": draft[:160], "shipped": shipped[:160]}
        if bool(a["fires"]) != bool(b["fires"]):
            binary_flips.append(rec)
        elif a["n_claims"] != b["n_claims"]:
            count_changes.append(rec)

    print(f"  scored {examined} altered pairs with evidence held fixed")
    print(f"  BINARY `fires` flips  : {len(binary_flips)}   <- what the prior is built on")
    print(f"  claim-COUNT changes   : {len(count_changes)}   <- same decision, fewer claims")
    for f in (binary_flips + count_changes)[:5]:
        print(f"    {f['pre']} -> {f['post']}   q: {f['question']}")

    if binary_flips:
        print("\n  THE PRIOR IS NOT COMPARABLE. Softening moved the binary receipt")
        print("  decision, so the 0.397 rate was measured on a pipeline this fix changes.")
        print("  Re-measure before quoting it again.")
    else:
        print("\n  The binary decision never moved, so the 0.397 fabrication prior")
        print("  survives the call-site fix and stays quotable -- a measured result, not")
        print("  the assumption it would otherwise have been.")
        if count_changes:
            print(f"  {len(count_changes)} turns did lose a claim to softening (3 -> 2) while")
            print("  still firing. Nothing downstream reads n_claims today, but any future")
            print("  severity or dose measure built on the COUNT would inherit this.")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "SOFTEN_PERTURBS_RECEIPTS.json")
    io.open(out, "w", encoding="utf-8", newline="\n").write(json.dumps(
        {"n_pairs": len(pairs), "n_text_altered": len(changed_text),
         "n_binary_flips": len(binary_flips), "n_count_changes": len(count_changes),
         "binary_flips": binary_flips, "count_changes": count_changes}, indent=1) + "\n")
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
